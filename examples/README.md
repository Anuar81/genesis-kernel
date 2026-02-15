# Genesis Kernel — Example Scripts

These scripts demonstrate how to use the Genesis AVX-512 kernel for
hybrid GPU/CPU inference with Mixture-of-Experts (MoE) language models.

## Scripts

### `benchmark_genesis_vs_bnb.py`

Isolated benchmark comparing Genesis AVX-512 CPU kernel against
bitsandbytes on both GPU and CPU. No model loading required — uses
random weights to measure raw kernel performance.

```bash
python benchmark_genesis_vs_bnb.py
python benchmark_genesis_vs_bnb.py --M 1024 --K 2048 --runs 100
```

Typical results on AMD Ryzen 9 7900 + RTX 4090:

| Engine           | Device | Time (ms) |
|------------------|--------|-----------|
| bitsandbytes     | GPU    | 0.063     |
| Genesis AVX-512  | CPU    | 0.150     |
| bitsandbytes     | CPU    | 24.842    |

Genesis CPU is **165x faster** than bitsandbytes CPU.

### `hybrid_moe_offload.py`

Full inference example: loads a MoE model with some layers on GPU
and the rest on system RAM using Genesis kernels.

```bash
# 30B model — 8 layers on RAM, ~13.4GB VRAM, ~5.7 tok/s
python hybrid_moe_offload.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --ram-layers 8

# 80B model — auto-detect split, ~20.7GB VRAM
python hybrid_moe_offload.py --model Qwen/Qwen3-Next-80B-A3B-Instruct

# Custom prompt
python hybrid_moe_offload.py --prompt "Explain quicksort in Python"
```

## Requirements

- CPU with AVX-512 support (AMD Zen4+, Intel Skylake-X+)
- NVIDIA GPU with CUDA
- Python 3.10+
- `pip install genesis-kernel torch bitsandbytes transformers safetensors huggingface_hub`

## How it works

The key insight: instead of dequantizing the full NF4 weight matrix
(~12MB per expert) and copying it to GPU, Genesis performs the entire
dequant+matmul fused on CPU using AVX-512 and copies only the result
vector (~12KB) to GPU. This is ~1000x less data over PCIe.

For a 30B MoE model with 8 layers offloaded to RAM:
- VRAM: 13.4GB (vs 24GB for full GPU)
- Speed: 5.7 tok/s (vs 6.6 tok/s full GPU = 86% efficiency)

For an 80B MoE model with 24 layers offloaded:
- VRAM: 20.7GB (impossible without CPU offload on a single 24GB GPU)
- Speed: 2.7-3.3 tok/s

## Troubleshooting

**bitsandbytes won't install on Python 3.14**: Use `gptqmodel` as an
alternative, or build bitsandbytes from source. Python 3.12 is the
safest choice for compatibility.

**OOM with 80B model**: The 80B model needs ~20GB VRAM + significant
system RAM. With 32GB total RAM, the OS may struggle. 64GB RAM
recommended for the 80B model.

**Slow first token**: The first generation includes kernel compilation
warmup. Second generation onwards shows true speed.
