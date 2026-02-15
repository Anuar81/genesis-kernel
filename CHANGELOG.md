# Changelog

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
