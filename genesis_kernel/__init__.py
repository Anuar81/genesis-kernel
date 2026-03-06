"""
Genesis Kernel — JIT x86-64 kernels for LLM inference.

Two kernel families:
  - NF4: Fused dequantization + matrix multiplication (AVX-512 ZMM)
  - Q4_K: ggml-compatible dot product (AVX-512 VNNI in YMM)

Quick start (NF4):
    from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations
    kernel = compile_best_nf4_matmul(M=2048, K=1536)

Quick start (Q4_K):
    from genesis_kernel import compile_turbo7, generate_turbo7_so
    kernel = compile_turbo7()
    # Or generate .so for llama.cpp:
    generate_turbo7_so(".")
"""

from .nf4_kernel import (
    compile_best_nf4_matmul,
    compile_nf4_matmul,
    generate_nf4_matmul_kernel,
    quantize_nf4,
    reorder_activations,
    dequant_nf4_reference,
    pack_nf4,
    matmul_nf4_reference,
    NF4_TABLE,
    BLOCKSIZE,
    # Backward-compatible aliases
    compilar_mejor_nf4_matmul,
    compilar_nf4_matmul,
    generar_kernel_nf4_matmul,
    cuantizar_nf4,
    reordenar_activaciones,
    dequant_nf4_python,
    empaquetar_nf4,
    NF4_TABLA,
)

from .q4k_kernel import (
    compile_q4k_kernel,
    compile_turbo7,
    generate_q4k_kernel,
    generate_turbo7_so,
    turbo7_ops,
    OpQ4K,
)

from .x86_emitter import X86Emitter

__version__ = "0.2.0"
__all__ = [
    # NF4
    "compile_best_nf4_matmul",
    "compile_nf4_matmul",
    "generate_nf4_matmul_kernel",
    "quantize_nf4",
    "reorder_activations",
    "dequant_nf4_reference",
    "pack_nf4",
    "matmul_nf4_reference",
    "NF4_TABLE",
    "BLOCKSIZE",
    # Q4_K
    "compile_q4k_kernel",
    "compile_turbo7",
    "generate_q4k_kernel",
    "generate_turbo7_so",
    "turbo7_ops",
    "OpQ4K",
    # Emitter
    "X86Emitter",
]
