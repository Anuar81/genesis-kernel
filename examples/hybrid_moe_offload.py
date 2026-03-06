#!/usr/bin/env python3
"""
Hybrid MoE Inference: GPU + CPU offload with Genesis AVX-512 kernels.

Loads a Mixture-of-Experts model with some layers on GPU (bitsandbytes NF4)
and the rest on system RAM (Genesis AVX-512 fused dequant+matmul).

This enables running models that don't fit entirely in VRAM. For example:
  - 30B MoE in 13.4GB VRAM (instead of 24GB) at 86% of full-GPU speed
  - 80B MoE in 20.7GB VRAM (impossible without CPU offload)

The key trick: instead of dequantizing the full weight matrix and copying
it to GPU, Genesis performs the entire matmul on CPU and copies only the
result vector to GPU. This is ~1000x less data over PCIe:
  - Full matrix copy: M*K*2 bytes (e.g. 3072*2048*2 = 12MB per expert)
  - Result vector copy: M*4 bytes (e.g. 3072*4 = 12KB per expert)

Requirements:
  - CPU with AVX-512 (AMD Zen4+, Intel Skylake-X+)
  - NVIDIA GPU with CUDA
  - pip install -e . (from genesis-kernel repo root)
  - pip install torch bitsandbytes transformers safetensors

Usage:
  python hybrid_moe_offload.py                          # auto-detect split
  python hybrid_moe_offload.py --ram-layers 8           # force 8 layers on RAM
  python hybrid_moe_offload.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct

Tested models:
  - Qwen/Qwen3-Coder-30B-A3B-Instruct (128 experts, 8 active)
  - Qwen/Qwen3-Next-80B-A3B-Instruct (512 experts, 10 active)
"""
import argparse
import gc
import os
import re
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

import bitsandbytes as bnb
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from genesis_kernel import (
    quantize_nf4 as _quantize_nf4_slow,
    compile_best_nf4_matmul,
    reorder_activations,
    BLOCKSIZE as GEN_BLOCKSIZE,
    NF4_TABLE,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

QUANT_TYPE = "nf4"
BLOCKSIZE = 64
COMPRESS_STATISTICS = True
QUANT_STORAGE = torch.uint8

EXPERT_RE = re.compile(
    r'^(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$'
)


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid MoE inference with Genesis AVX-512")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                   help="HuggingFace model name")
    p.add_argument("--ram-layers", type=int, default=None,
                   help="Number of layers to offload to RAM (auto if not set)")
    p.add_argument("--vram-limit", type=float, default=23.0,
                   help="Max VRAM to use in GB (default: 23.0)")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max tokens to generate per test")
    p.add_argument("--prompt", type=str, default=None,
                   help="Custom prompt (default: factorial function request)")
    return p.parse_args()


def quantize_nf4_fast(weights_float: np.ndarray) -> tuple:
    """Vectorized NF4 quantization (~30x faster than loop-based)."""
    n = len(weights_float)
    bs = GEN_BLOCKSIZE
    n_padded = ((n + bs - 1) // bs) * bs
    padded = np.zeros(n_padded, dtype=np.float32)
    padded[:n] = weights_float
    blocks = padded.reshape(-1, bs)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0
    normalized = blocks / scales[:, None]
    flat = normalized.flatten()
    diffs = np.abs(flat[:, None] - NF4_TABLE[None, :])
    indices = np.argmin(diffs, axis=1).astype(np.uint8)
    lo = indices[0::2] & 0x0F
    hi = indices[1::2] & 0x0F
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed, scales.astype(np.float32)


_kernel_cache = {}

def get_kernel(M, K):
    """Get the best Genesis kernel for given dimensions (cached)."""
    key = (M, K)
    if key not in _kernel_cache:
        print(f"   Compiling Genesis AVX-512 kernel for {M}x{K}...")
        _kernel_cache[key] = compile_best_nf4_matmul(M=M, K=K)
        origin = getattr(_kernel_cache[key], '_origin', 'unknown')
        print(f"   Kernel {M}x{K} ready (source: {origin})")
    return _kernel_cache[key]


class GPUExperts:
    """MoE experts stored on GPU — dequantized with bitsandbytes."""
    def __init__(self, num_experts, shape):
        self.num_experts = num_experts
        self.shape_per_expert = shape
        self.qdata = [None] * num_experts
        self.qstate = [None] * num_experts
        self.on_cpu = False

    def set_expert(self, idx, qdata, qstate):
        self.qdata[idx] = qdata
        self.qstate[idx] = qstate

    def __getitem__(self, idx):
        return dequantize_4bit(
            self.qdata[idx], self.qstate[idx],
            quant_type=QUANT_TYPE, blocksize=BLOCKSIZE
        ).reshape(self.shape_per_expert)

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def dtype(self):
        return torch.float16


class CPUExperts:
    """MoE experts stored in RAM — fused dequant+matmul with Genesis AVX-512.

    Instead of dequantizing the full matrix and copying to GPU, Genesis
    performs the matmul on CPU and copies only the result vector (~1000x
    less data over PCIe).
    """
    def __init__(self, num_experts, shape):
        self.num_experts = num_experts
        self.shape_per_expert = shape
        self.nf4_list = [None] * num_experts
        self.scales_list = [None] * num_experts
        self.on_cpu = True

    def set_expert(self, idx, nf4_data, scales):
        self.nf4_list[idx] = nf4_data
        self.scales_list[idx] = scales

    def matmul_batch(self, expert_idx, activations_tensor):
        """Fused dequant+matmul on CPU, result transferred to GPU."""
        nf4 = self.nf4_list[expert_idx]
        scales = self.scales_list[expert_idx]
        M, K = self.shape_per_expert
        act_cpu = activations_tensor.float().cpu().numpy()
        kernel = get_kernel(M, K)

        if act_cpu.ndim == 1:
            act_reord = reorder_activations(act_cpu)
            result = kernel(nf4, scales, act_reord, M, K)
            return torch.from_numpy(result).to(dtype=torch.float16, device="cuda").unsqueeze(0)

        results = []
        for i in range(act_cpu.shape[0]):
            act_reord = reorder_activations(act_cpu[i])
            results.append(kernel(nf4, scales, act_reord, M, K))
        return torch.from_numpy(np.stack(results)).to(dtype=torch.float16, device="cuda")

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def dtype(self):
        return torch.float16


def make_hybrid_forward():
    """Create a hybrid forward function for MoE expert layers."""
    def hybrid_forward(self_exp, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = nn.functional.one_hot(top_k_index, num_classes=self_exp.num_experts)
            mask = mask.permute(2, 1, 0)
            active = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

        gate_up_cpu = getattr(self_exp.gate_up_proj, 'on_cpu', False)
        down_cpu = getattr(self_exp.down_proj, 'on_cpu', False)

        for eidx in active:
            eidx = eidx[0]
            if eidx == self_exp.num_experts:
                continue

            top_k_pos, token_idx = torch.where(mask[eidx])
            state = hidden_states[token_idx]

            if gate_up_cpu:
                gate_up_out = self_exp.gate_up_proj.matmul_batch(eidx, state)
            else:
                w = self_exp.gate_up_proj[eidx]
                gate_up_out = nn.functional.linear(state, w)
                del w

            gate, up = gate_up_out.chunk(2, dim=-1)
            del gate_up_out
            h = self_exp.act_fn(gate) * up
            del gate, up

            if down_cpu:
                h = self_exp.down_proj.matmul_batch(eidx, h)
            else:
                w = self_exp.down_proj[eidx]
                h = nn.functional.linear(h, w)
                del w

            h = h * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.to(final.dtype))

        return final
    return hybrid_forward


def should_quantize(name, tensor):
    """Check if a tensor should be NF4 quantized."""
    skip = ["layernorm", "layer_norm", "norm.weight", "norm.bias",
            "embed_tokens", "lm_head", "bias", "gate.weight", "rotary_emb"]
    nl = name.lower()
    return not any(p in nl for p in skip) and tensor.ndim >= 2 and tensor.numel() >= 1024


def navigate(model, dotted_path):
    """Navigate model hierarchy by dotted path."""
    parts = dotted_path.split(".")
    mod = model
    for p in parts[:-1]:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod, parts[-1]


def detect_moe_class(model_name):
    """Detect which MoE experts class to patch based on model name."""
    name_lower = model_name.lower()
    if "next" in name_lower or "80b" in name_lower:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextExperts
        return Qwen3NextExperts
    else:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
        return Qwen3MoeExperts


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Hybrid MoE Inference with Genesis AVX-512")
    print(f"Model: {args.model}")
    print("=" * 60)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_path = snapshot_download(args.model)

    num_experts = config.num_experts
    num_layers = config.num_hidden_layers
    moe_intermediate = getattr(config, 'moe_intermediate_size', config.intermediate_size)

    print(f"\n   {num_layers} layers, {num_experts} experts/layer, {moe_intermediate} intermediate")

    # Patch the MoE forward
    experts_cls = detect_moe_class(args.model)
    experts_cls.forward = make_hybrid_forward()
    print(f"   {experts_cls.__name__}.forward patched (hybrid GPU/CPU)")

    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    print(f"   {len(shard_files)} shards")

    # Instantiate model on meta device (no memory used)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()

    # Index tensors: separate experts from regular weights
    expert_keys_by_layer = defaultdict(list)
    regular_keys = []
    for sf in shard_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                m = EXPERT_RE.match(key)
                if m:
                    expert_keys_by_layer[int(m.group(2))].append(
                        (sf, key, int(m.group(3)), m.group(4)))
                else:
                    regular_keys.append((sf, key))

    print(f"   Regular tensors: {len(regular_keys)}, MoE layers: {len(expert_keys_by_layer)}")

    # Load regular weights (always on GPU)
    print(f"\nLoading {len(regular_keys)} regular tensors to GPU...")
    t0 = time.time()
    regular_by_shard = defaultdict(list)
    for sf, key in regular_keys:
        regular_by_shard[sf].append(key)

    skipped = []
    for sf, keys in regular_by_shard.items():
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in keys:
                try:
                    parent, attr = navigate(model, key)
                except (AttributeError, IndexError):
                    skipped.append(key)
                    continue
                tc = f.get_tensor(key)
                if should_quantize(key, tc) and tc.ndim == 2 and attr == "weight":
                    parts = key.split(".")
                    ln = parts[-2]
                    gp = model
                    for p in parts[:-2]:
                        gp = gp[int(p)] if p.isdigit() else getattr(gp, p)
                    out_f, in_f = tc.shape
                    old = getattr(gp, ln)
                    has_bias = old.bias is not None if hasattr(old, 'bias') else False
                    l4 = bnb.nn.Linear4bit(in_f, out_f, bias=has_bias,
                        compute_dtype=torch.float16, compress_statistics=COMPRESS_STATISTICS,
                        quant_type=QUANT_TYPE, quant_storage=QUANT_STORAGE, device="cuda")
                    tg = tc.to(dtype=torch.float16, device="cuda").contiguous()
                    qd, qs = quantize_4bit(tg, blocksize=BLOCKSIZE,
                        compress_statistics=COMPRESS_STATISTICS, quant_type=QUANT_TYPE,
                        quant_storage=QUANT_STORAGE)
                    del tg
                    l4.weight = bnb.nn.Params4bit(data=qd, requires_grad=False,
                        quant_state=qs, blocksize=BLOCKSIZE,
                        compress_statistics=COMPRESS_STATISTICS, quant_type=QUANT_TYPE,
                        quant_storage=QUANT_STORAGE, bnb_quantized=True)
                    setattr(gp, ln, l4)
                else:
                    tg = tc.to(dtype=torch.float16, device="cuda")
                    p = torch.nn.Parameter(tg, requires_grad=False)
                    if attr == "weight":
                        parent.weight = p
                    elif attr == "bias":
                        parent.bias = p
                    else:
                        setattr(parent, attr, p)
                del tc

    del regular_keys, regular_by_shard
    gc.collect()
    torch.cuda.empty_cache()
    vram_regulars = torch.cuda.memory_allocated() / 1024**3
    print(f"   Done: {time.time()-t0:.0f}s — VRAM: {vram_regulars:.1f}GB")
    if skipped:
        print(f"   {len(skipped)} tensors skipped (not in model)")

    # Calculate GPU/RAM split
    expert_vram_per_layer = (num_experts * (moe_intermediate * 2 * config.hidden_size +
                             config.hidden_size * moe_intermediate) * 2 / 4) / (1024**3) * 1.15
    vram_for_experts = args.vram_limit - vram_regulars

    if args.ram_layers is not None:
        n_ram = args.ram_layers
    else:
        n_gpu = min(int(vram_for_experts / expert_vram_per_layer), num_layers)
        n_ram = num_layers - n_gpu

    n_gpu_layers = num_layers - n_ram
    gpu_layer_set = set(range(n_gpu_layers))
    ram_layer_set = set(range(n_gpu_layers, num_layers))

    print(f"\n   GPU: layers 0-{n_gpu_layers-1} ({n_gpu_layers} layers)")
    print(f"   RAM: layers {n_gpu_layers}-{num_layers-1} ({n_ram} layers, Genesis AVX-512)")

    # Load experts layer by layer
    print(f"\nLoading experts...")
    t_exp = time.time()
    gpu_experts = 0
    ram_experts = 0

    for layer_idx in sorted(expert_keys_by_layer.keys()):
        entries = expert_keys_by_layer[layer_idx]
        is_ram = layer_idx in ram_layer_set

        by_proj = defaultdict(dict)
        entries_by_shard = defaultdict(list)
        for sf, key, eidx, proj in entries:
            entries_by_shard[sf].append((key, eidx, proj))
        for sf, items in entries_by_shard.items():
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key, eidx, proj in items:
                    by_proj[proj][eidx] = f.get_tensor(key)

        experts_module = model.model.layers[layer_idx].mlp.experts

        # Gate+Up projection (fused)
        if "gate_proj" in by_proj and "up_proj" in by_proj:
            gd, ud = by_proj["gate_proj"], by_proj["up_proj"]
            fused_shape = (gd[0].shape[0] + ud[0].shape[0], gd[0].shape[1])

            if is_ram:
                container = CPUExperts(num_experts, fused_shape)
                for eidx in range(num_experts):
                    fused = torch.cat([gd[eidx], ud[eidx]], dim=0)
                    nf4_data, scales = quantize_nf4_fast(fused.float().numpy().flatten())
                    container.set_expert(eidx, nf4_data, scales)
                    del fused
                    ram_experts += 1
            else:
                container = GPUExperts(num_experts, fused_shape)
                for eidx in range(num_experts):
                    fused = torch.cat([gd[eidx], ud[eidx]], dim=0)
                    tg = fused.to(dtype=torch.float16, device="cuda").contiguous()
                    qd, qs = quantize_4bit(tg, blocksize=BLOCKSIZE,
                        compress_statistics=COMPRESS_STATISTICS, quant_type=QUANT_TYPE,
                        quant_storage=QUANT_STORAGE)
                    del tg, fused
                    container.set_expert(eidx, qd, qs)
                    gpu_experts += 1

            if "gate_up_proj" in experts_module._parameters:
                del experts_module._parameters["gate_up_proj"]
            object.__setattr__(experts_module, "gate_up_proj", container)

        # Down projection
        if "down_proj" in by_proj:
            dd = by_proj["down_proj"]

            if is_ram:
                container = CPUExperts(num_experts, dd[0].shape)
                for eidx in range(num_experts):
                    nf4_data, scales = quantize_nf4_fast(dd[eidx].float().numpy().flatten())
                    container.set_expert(eidx, nf4_data, scales)
                    ram_experts += 1
            else:
                container = GPUExperts(num_experts, dd[0].shape)
                for eidx in range(num_experts):
                    tg = dd[eidx].to(dtype=torch.float16, device="cuda").contiguous()
                    qd, qs = quantize_4bit(tg, blocksize=BLOCKSIZE,
                        compress_statistics=COMPRESS_STATISTICS, quant_type=QUANT_TYPE,
                        quant_storage=QUANT_STORAGE)
                    del tg
                    container.set_expert(eidx, qd, qs)
                    gpu_experts += 1

            if "down_proj" in experts_module._parameters:
                del experts_module._parameters["down_proj"]
            object.__setattr__(experts_module, "down_proj", container)

        del by_proj, entries_by_shard
        if (layer_idx + 1) % 8 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            vram = torch.cuda.memory_allocated() / 1024**3
            tag = "RAM" if is_ram else "GPU"
            print(f"   Layer {layer_idx+1}/{num_layers} [{tag}] — VRAM: {vram:.1f}GB")

    del expert_keys_by_layer
    gc.collect()
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"   Experts loaded: {time.time()-t_exp:.0f}s")
    print(f"   GPU: {gpu_experts}, RAM (Genesis): {ram_experts}")
    print(f"   VRAM: {vram:.1f}GB")

    # Materialize meta buffers (rotary embeddings, etc.)
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            parts = name.split(".")
            mod = model
            for p in parts[:-1]:
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
            bn = parts[-1]
            if "inv_freq" in bn:
                rope_dim = getattr(config, 'qk_rope_head_dim',
                                   getattr(config, 'head_dim', 128))
                base = getattr(config, 'rope_theta', 10000.0)
                inv_freq = 1.0 / (base ** (
                    torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
                mod.register_buffer(bn, inv_freq.to("cuda"), persistent=False)
            else:
                mod.register_buffer(bn,
                    torch.zeros(buf.shape, dtype=torch.float16, device="cuda"),
                    persistent=False)

    # --- Generation tests ---
    print(f"\nGeneration test...")
    if args.prompt:
        user_msg = args.prompt
    else:
        user_msg = "Write a Python function that calculates the factorial of a number."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\n</think>\n\n"
    inp = tokenizer(prompt, return_tensors="pt")
    inp = {k: v.cuda() for k, v in inp.items()}

    print("   Generating (first run includes warmup)...")
    t_gen = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=args.max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    gen_time = time.time() - t_gen
    tokens_out = out.shape[1] - inp["input_ids"].shape[1]
    tok_s = tokens_out / gen_time if gen_time > 0 else 0
    response = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"   {tokens_out} tokens in {gen_time:.1f}s ({tok_s:.1f} tok/s)")
    print(f"   Response: {response[:300]}")
    del out, inp

    # Second generation (no warmup)
    print(f"\nSecond generation (no warmup)...")
    messages2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a linked list? Explain briefly."},
    ]
    prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    prompt2 += "<think>\n</think>\n\n"
    inp2 = tokenizer(prompt2, return_tensors="pt")
    inp2 = {k: v.cuda() for k, v in inp2.items()}

    t_gen2 = time.time()
    with torch.no_grad():
        out2 = model.generate(**inp2, max_new_tokens=args.max_tokens, temperature=0.7,
                              do_sample=True, pad_token_id=tokenizer.eos_token_id)
    gen_time2 = time.time() - t_gen2
    tokens_out2 = out2.shape[1] - inp2["input_ids"].shape[1]
    tok_s2 = tokens_out2 / gen_time2 if gen_time2 > 0 else 0
    response2 = tokenizer.decode(out2[0][inp2["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"   {tokens_out2} tokens in {gen_time2:.1f}s ({tok_s2:.1f} tok/s)")
    print(f"   Response: {response2[:300]}")

    # Summary
    vram_final = torch.cuda.memory_allocated() / 1024**3
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"GPU layers: {n_gpu_layers}, RAM layers (Genesis): {n_ram}")
    print(f"GPU experts: {gpu_experts}, RAM experts: {ram_experts}")
    print(f"VRAM: {vram_final:.1f}GB")
    print(f"Speed: {tok_s2:.1f} tok/s (second run, no warmup)")
    if tok_s2 > 0:
        print(f"Model generates text with Genesis hybrid offload")
    else:
        print(f"No tokens generated")


if __name__ == "__main__":
    main()
