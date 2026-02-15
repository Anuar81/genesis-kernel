"""
Genesis Kernel — AVX-512 fused NF4 dequant+matmul for LLM inference.

Provides JIT-compiled x86-64 kernels that fuse NF4 dequantization with
matrix multiplication, enabling fast CPU inference for 4-bit quantized
language models. Includes dimension-specific kernels discovered through
genetic evolution of instruction orderings.

Quick start:
    from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations

    nf4_weights, scales = quantize_nf4(weights_float32)
    act_reord = reorder_activations(activations_float32)
    kernel = compile_best_nf4_matmul(M=2048, K=1536)
    output = kernel(nf4_weights, scales, act_reord, M=2048, K=1536)
"""

from .nf4_kernel import (
    # Public API
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
    # Backward-compatible aliases (Spanish)
    compilar_mejor_nf4_matmul,
    compilar_nf4_matmul,
    generar_kernel_nf4_matmul,
    cuantizar_nf4,
    reordenar_activaciones,
    dequant_nf4_python,
    empaquetar_nf4,
    NF4_TABLA,
)

from .x86_emitter import X86Emitter

__version__ = "0.1.0"
__all__ = [
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
    "X86Emitter",
]
