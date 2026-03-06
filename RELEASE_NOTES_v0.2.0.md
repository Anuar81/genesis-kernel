# v0.2.0 — Q4_K Turbo7 Kernel

## What's new

**Q4_K dot product kernel** — drop-in replacement for `ggml_vec_dot_q4_K_q8_K_generic` in llama.cpp via `LD_PRELOAD`.

- AVX-512 VNNI in YMM registers (256-bit, avoids Zen 4 double-pumping)
- 34 inner loop ops (turbo6 baseline + 1 PREFETCHNTA hint found by brute-force injection)
- Bit-exact output vs ggml (`diff = 0.000000e+00` for N=1 to N=1024)

### Isolated benchmark (Ryzen 9 7900, llama.cpp b8184)

| N | turbo7 | ggml | Delta | Wins |
|---|--------|------|-------|------|
| 32 | 32.05 ns/blk | 36.59 ns/blk | +12.56% | 20/20 |
| 128 | 11.69 | 12.85 | +9.05% | 20/20 |
| 512 | 6.40 | 6.56 | +2.52% | 20/20 |
| 1024 | 5.51 | 5.59 | +1.46% | 20/20 |

### End-to-end (Qwen3-Coder-30B, llama.cpp b8184)

- Prompt: +3.0% to +8.4% tok/s
- Generation: +0.2% to +3.5% tok/s

## Also in this release

- x86 emitter: ~50 new instructions (VEX3, VNNI, prefetch, 32-bit GPR ops)
- pytest test suite: 16 tests (NF4 accuracy, Q4_K accuracy, reorder contract)
- Full README rewrite with reproduction steps, benchmark methodology, and FAQ

## Install

```bash
git clone https://github.com/Anuar81/genesis-kernel.git
cd genesis-kernel
pip install -e .
pytest tests/ -v
```

## Requirements

- Linux x86-64
- Python 3.10+, NumPy
- CPU with AVX-512 (AMD Zen 4+, Intel Skylake-X+ for NF4; Intel Ice Lake+ for Q4_K VNNI)
