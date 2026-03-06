# Changelog

## 0.2.0 — 2026-03-05

Q4_K turbo7 kernel: ggml-compatible `vec_dot_q4_K_q8_K` replacement.

- Q4_K dot product kernel using AVX-512 VNNI in YMM registers (256-bit)
- Turbo7: 34 inner loop ops (turbo6 baseline + 1 PREFETCHNTA hint)
- PREFETCHNTA [R9+358] at position 19 found by brute-force injection
  (5,000 random attempts, 3-stage verification: measure → re-measure × 5 → A/B × 20)
- Bit-exact output vs ggml (`diff = 0.000000e+00` for N=1 to N=1024)
- Isolated benchmark vs ggml (llama.cpp b8184, Ryzen 9 7900):
  - N=32: 32.05 vs 36.59 ns/blk (+12.56%)
  - N=128: 11.69 vs 12.85 ns/blk (+9.05%)
  - N=512: 6.40 vs 6.56 ns/blk (+2.52%)
  - N=1024: 5.51 vs 5.59 ns/blk (+1.46%)
  - 20/20 wins in all N
- llama.cpp integration via LD_PRELOAD (`generate_turbo7_so()`)
- End-to-end (Qwen3-Coder-30B, prompt): +3.0% to +8.4% tok/s
- x86 emitter: added VEX3 prefix infrastructure, ~50 new instructions
  (VNNI, prefetch variants, 32-bit GPR ops, VEX XMM/YMM ops)
- Tests: pytest-based, Q4_K accuracy + NF4 accuracy + reorder contract

## 0.1.0 — 2026-02-15

Initial public release.

- x86-64 emitter: generates AVX-512 machine code at runtime via `mmap` + `PROT_EXEC`
- NF4 fused dequantization + matrix multiplication kernel (2-pipeline baseline)
- 4 dimension-specific evolved kernels baked from genetic optimization:
  - 1024×2048 (+2.93% vs baseline)
  - 2048×512 (+19.25% vs baseline)
  - 3072×2048 (+3.90% vs baseline)
  - 2048×1536 (+2.89% vs baseline)
- Quantization utilities: `quantize_nf4`, `reorder_activations`, `pack_nf4`
- Python reference implementations for validation
- Automatic kernel selection via `compile_best_nf4_matmul(M, K)`
- Examples: standalone benchmark, hybrid GPU/CPU MoE inference
