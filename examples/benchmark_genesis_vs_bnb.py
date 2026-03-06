#!/usr/bin/env python3
"""
Benchmark: Genesis NF4 AVX-512 kernel (CPU) vs bitsandbytes (GPU & CPU)

Compares dequantization + matrix multiplication performance between:
  - bitsandbytes NF4 on GPU (the standard for local LLM inference)
  - Genesis AVX-512 on CPU (fused dequant+matmul kernel)
  - bitsandbytes NF4 on CPU (baseline — typically unusable)

Requirements:
  - CPU with AVX-512 support (AMD Zen4+, Intel Skylake-X+)
  - NVIDIA GPU with CUDA
  - pip install -e . (from genesis-kernel repo root)
  - pip install torch bitsandbytes

Usage: python benchmark_genesis_vs_bnb.py [--M 2048] [--K 1536] [--runs 50]
"""
import argparse
import time
import numpy as np
import torch
from bitsandbytes.functional import quantize_4bit, dequantize_4bit

from genesis_kernel import (
    quantize_nf4,
    compile_best_nf4_matmul,
    reorder_activations,
    dequant_nf4_reference,
    NF4_TABLE,
    BLOCKSIZE,
)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Genesis AVX-512 vs bitsandbytes")
    p.add_argument("--M", type=int, default=2048, help="Output dimension (rows)")
    p.add_argument("--K", type=int, default=1536, help="Input dimension (cols)")
    p.add_argument("--runs", type=int, default=50, help="Number of benchmark iterations")
    return p.parse_args()


def benchmark_bnb_gpu(bnb_qdata, bnb_qstate, activations, M, K, n_runs):
    """Benchmark bitsandbytes dequant+matmul on GPU."""
    for _ in range(5):
        w = dequantize_4bit(bnb_qdata, bnb_qstate, quant_type="nf4", blocksize=64)
        w = w.reshape(M, K)
        out = torch.nn.functional.linear(activations, w)
        del w, out
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        w = dequantize_4bit(bnb_qdata, bnb_qstate, quant_type="nf4", blocksize=64)
        w = w.reshape(M, K)
        out = torch.nn.functional.linear(activations, w)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        del w, out
    times.sort()
    return times


def benchmark_genesis_cpu(gen_nf4, gen_scales, act_reordered, M, K, n_runs):
    """Benchmark Genesis fused dequant+matmul on CPU (AVX-512)."""
    kernel = compile_best_nf4_matmul(M=M, K=K)

    for _ in range(5):
        kernel(gen_nf4, gen_scales, act_reordered, M, K)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        kernel(gen_nf4, gen_scales, act_reordered, M, K)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times, kernel


def benchmark_bnb_cpu(weights_fp16, K, n_runs):
    """Benchmark bitsandbytes dequant+matmul on CPU."""
    weights_cpu = weights_fp16.float().contiguous()
    qdata_cpu, qstate_cpu = quantize_4bit(
        weights_cpu, blocksize=64, compress_statistics=True,
        quant_type="nf4", quant_storage=torch.uint8
    )
    act_cpu = torch.randn(1, K, dtype=torch.float32)

    try:
        for _ in range(3):
            w = dequantize_4bit(qdata_cpu, qstate_cpu, quant_type="nf4", blocksize=64)
            M = weights_fp16.shape[0]
            w = w.reshape(M, K)
            torch.nn.functional.linear(act_cpu, w)
            del w
    except Exception as e:
        return None, str(e)

    M = weights_fp16.shape[0]
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        w = dequantize_4bit(qdata_cpu, qstate_cpu, quant_type="nf4", blocksize=64)
        w = w.reshape(M, K)
        torch.nn.functional.linear(act_cpu, w)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        del w
    times.sort()
    return times, None


def main():
    args = parse_args()
    M, K, n_runs = args.M, args.K, args.runs

    print("=" * 60)
    print("Benchmark: Genesis AVX-512 vs bitsandbytes")
    print(f"Dimensions: M={M}, K={K} (typical MoE expert projection)")
    print(f"Iterations: {n_runs}")
    print("=" * 60)

    print("\n1. Generating random FP16 weights...")
    torch.manual_seed(42)
    np.random.seed(42)
    weights_fp16 = torch.randn(M, K, dtype=torch.float16)
    print(f"   Shape: {weights_fp16.shape}")

    print("\n2. Quantizing with bitsandbytes NF4 (GPU)...")
    weights_gpu = weights_fp16.to("cuda").contiguous()
    bnb_qdata, bnb_qstate = quantize_4bit(
        weights_gpu, blocksize=64, compress_statistics=True,
        quant_type="nf4", quant_storage=torch.uint8
    )
    del weights_gpu

    print("   Quantizing with Genesis NF4 (CPU)...")
    weights_flat = weights_fp16.float().numpy().flatten()
    gen_nf4, gen_scales = quantize_nf4(weights_flat)

    print("\n3. Verifying correctness...")
    bnb_deq = dequantize_4bit(bnb_qdata, bnb_qstate, quant_type="nf4", blocksize=64)
    bnb_flat = bnb_deq.reshape(M, K).float().cpu().numpy().flatten()
    n_weights = M * K
    gen_deq = dequant_nf4_reference(gen_nf4, gen_scales, n_weights)
    diff = np.abs(bnb_flat[:n_weights] - gen_deq[:n_weights])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"   Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    if max_diff < 0.1:
        print("   ✅ Results match (tolerance < 0.1)")
    else:
        print("   ⚠️  Significant differences — investigate")

    print(f"\n4. Benchmarking bitsandbytes GPU ({n_runs} runs)...")
    activations = torch.randn(1, K, dtype=torch.float16, device="cuda")
    times_bnb = benchmark_bnb_gpu(bnb_qdata, bnb_qstate, activations, M, K, n_runs)
    median_bnb = times_bnb[len(times_bnb) // 2]
    print(f"   Median: {median_bnb:.3f} ms")

    print(f"\n5. Benchmarking Genesis AVX-512 CPU ({n_runs} runs)...")
    act_np = activations[0].float().cpu().numpy()
    act_reord = reorder_activations(act_np)
    times_gen, kernel = benchmark_genesis_cpu(gen_nf4, gen_scales, act_reord, M, K, n_runs)
    median_gen = times_gen[len(times_gen) // 2]
    print(f"   Median: {median_gen:.3f} ms")

    print("\n6. Verifying matmul correctness...")
    w_bnb = dequantize_4bit(bnb_qdata, bnb_qstate, quant_type="nf4", blocksize=64)
    w_bnb = w_bnb.reshape(M, K)
    out_bnb = torch.nn.functional.linear(activations, w_bnb)[0].float().cpu().numpy()
    out_gen = kernel(gen_nf4, gen_scales, act_reord, M, K)
    matmul_diff = np.abs(out_bnb - out_gen)
    rel_err = np.mean(matmul_diff) / (np.mean(np.abs(out_bnb)) + 1e-8)
    print(f"   Relative error: {rel_err:.4%}")

    print(f"\n7. Benchmarking bitsandbytes CPU ({n_runs} runs)...")
    times_bnb_cpu, err = benchmark_bnb_cpu(weights_fp16, K, n_runs)
    if times_bnb_cpu:
        median_bnb_cpu = times_bnb_cpu[len(times_bnb_cpu) // 2]
        print(f"   Median: {median_bnb_cpu:.3f} ms")
    else:
        median_bnb_cpu = None
        print(f"   ❌ bitsandbytes CPU FAILED: {err}")

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"{'Engine':<30} {'Device':<8} {'Median (ms)':<12}")
    print("-" * 50)
    print(f"{'bitsandbytes':<30} {'GPU':<8} {median_bnb:<12.3f}")
    print(f"{'Genesis AVX-512':<30} {'CPU':<8} {median_gen:<12.3f}")
    if median_bnb_cpu:
        print(f"{'bitsandbytes':<30} {'CPU':<8} {median_bnb_cpu:<12.3f}")
        speedup = median_bnb_cpu / median_gen
        print(f"\n🔥 Genesis CPU is {speedup:.0f}x faster than bitsandbytes CPU")
    else:
        print(f"{'bitsandbytes':<30} {'CPU':<8} {'FAILED':<12}")
        print(f"\n🔥 Genesis is the ONLY viable option for CPU NF4 inference")

    if median_gen < median_bnb:
        print(f"🚀 Genesis CPU is {median_bnb / median_gen:.1f}x faster than bnb GPU")
    else:
        print(f"📊 Genesis CPU is {median_gen / median_bnb:.1f}x slower than bnb GPU (expected)")

    print(f"\nMatmul relative error: {rel_err:.4%}")
    if rel_err < 0.05:
        print("✅ Results are correct — safe to use for inference")


if __name__ == "__main__":
    main()
