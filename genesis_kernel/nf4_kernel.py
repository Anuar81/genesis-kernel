#!/usr/bin/env python3
"""
Genesis Kernel — NF4 fused dequantization + matrix multiplication (AVX-512).

Provides quantization utilities and JIT-compiled x86-64 kernels for NF4
(4-bit NormalFloat) inference. The matmul kernels fuse dequantization with
the dot product, avoiding materialization of the full weight matrix.

Key kernels were discovered through genetic evolution of x86 instruction
orderings (16,460 mutations across 25 evolutionary runs). The evolved
instruction schedules exploit Zen 4 microarchitectural properties:
  - Cache-line-aligned NOPs for instruction fetch optimization
  - Early scale broadcasts to hide memory latency
  - Reverse-distance activation loading for prefetcher efficiency
  - Interleaved pipeline execution to maximize port utilization

Four dimension-specific baked kernels are included for common MoE projections:
  - 1024x2048 (80B gate_up):  +2.93% vs baseline
  - 2048x512  (80B down_proj): +19.25% vs baseline
  - 3072x2048 (30B gate_up):  +3.90% vs baseline
  - 2048x1536 (30B down_proj): +2.89% vs baseline
"""

import ctypes
import mmap
import struct
import numpy as np

from .x86_emitter import (
    X86Emitter,
    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
    XMM0, XMM1, XMM2,
    ZMM0, ZMM1, ZMM2, ZMM3, ZMM4, ZMM5,
)

# NF4 lookup table: 16 quantile values from the normal distribution.
# Source: QLoRA (Dettmers et al.), Appendix E.
NF4_TABLE = np.array([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=np.float32)

# Backward-compatible alias
NF4_TABLA = NF4_TABLE

BLOCKSIZE = 64  # weights per scale block (bitsandbytes standard)


# ============================================================
# Quantization utilities
# ============================================================

def dequant_nf4_reference(weights_nf4: np.ndarray, scales: np.ndarray,
                           n_weights: int) -> np.ndarray:
    """
    Dequantize NF4 weights to float32 (Python reference implementation).

    Args:
        weights_nf4: uint8 array, each byte packs 2 weights (low nibble first)
        scales: float32 array, one scale per block of BLOCKSIZE weights
        n_weights: total number of weights

    Returns:
        float32 array of dequantized weights
    """
    result = np.zeros(n_weights, dtype=np.float32)
    for i in range(n_weights):
        byte_idx = i // 2
        if i % 2 == 0:
            nibble = weights_nf4[byte_idx] & 0x0F
        else:
            nibble = (weights_nf4[byte_idx] >> 4) & 0x0F
        block = i // BLOCKSIZE
        result[i] = NF4_TABLE[nibble] * scales[block]
    return result


# Backward-compatible alias
dequant_nf4_python = dequant_nf4_reference


def dot_nf4_reference(weights_nf4: np.ndarray, scales: np.ndarray,
                       activations: np.ndarray, n_weights: int) -> float:
    """NF4 dot product: sum(dequant(weight_i) * activation_i). Python reference."""
    weights_float = dequant_nf4_reference(weights_nf4, scales, n_weights)
    return float(np.dot(weights_float, activations[:n_weights]))


dot_nf4_python = dot_nf4_reference


def pack_nf4(indices: np.ndarray) -> np.ndarray:
    """Pack NF4 indices (0-15) into bytes (2 per byte, low nibble first)."""
    assert len(indices) % 2 == 0
    n_bytes = len(indices) // 2
    result = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        lo = indices[2 * i] & 0x0F
        hi = indices[2 * i + 1] & 0x0F
        result[i] = lo | (hi << 4)
    return result


# Backward-compatible alias
empaquetar_nf4 = pack_nf4


def quantize_nf4(weights_float: np.ndarray) -> tuple:
    """
    Quantize float32 weights to NF4 format (vectorized).

    For each block of BLOCKSIZE weights:
      1. Compute scale = max(abs(block))
      2. Normalize to [-1, 1]
      3. Find nearest NF4 table entry
      4. Pack indices into bytes

    Uses NumPy broadcasting for ~30x speedup over the loop-based version.

    Args:
        weights_float: float32 array of weights

    Returns:
        (nf4_packed, scales) — uint8 packed weights and float32 scales
    """
    n = len(weights_float)
    n_padded = ((n + BLOCKSIZE - 1) // BLOCKSIZE) * BLOCKSIZE
    padded = np.zeros(n_padded, dtype=np.float32)
    padded[:n] = weights_float

    # Reshape into blocks, compute scales vectorized
    blocks = padded.reshape(-1, BLOCKSIZE)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0

    # Normalize all blocks at once
    normalized = blocks / scales[:, None]

    # Find nearest NF4 table entry for all values at once
    flat = normalized.flatten()
    diffs = np.abs(flat[:, None] - NF4_TABLE[None, :])
    indices = np.argmin(diffs, axis=1).astype(np.uint8)

    # Pack two 4-bit indices per byte (low nibble first)
    lo = indices[0::2] & 0x0F
    hi = indices[1::2] & 0x0F
    packed = (lo | (hi << 4)).astype(np.uint8)

    return packed, scales.astype(np.float32)


# Backward-compatible alias
cuantizar_nf4 = quantize_nf4


def reorder_activations(activations: np.ndarray) -> np.ndarray:
    """
    Reorder activations for sequential AVX-512 loading.

    NF4 weights are packed 2 per byte. After nibble separation the kernel
    gets 16 even-indexed and 16 odd-indexed weights. This function reorders
    activations to match:
        reordered[0..15]  = activations[0, 2, 4, ..., 30]  (even)
        reordered[16..31] = activations[1, 3, 5, ..., 31]  (odd)
    Repeated for each group of 32 activations.

    Args:
        activations: float32 array, length must be multiple of 32

    Returns:
        float32 array with reordered activations
    """
    n = len(activations)
    assert n % 32 == 0, f"Length must be multiple of 32, got {n}"
    result = np.empty(n, dtype=np.float32)
    for i in range(0, n, 32):
        group = activations[i:i+32]
        result[i:i+16] = group[0::2]
        result[i+16:i+32] = group[1::2]
    return result


# Backward-compatible alias
reordenar_activaciones = reorder_activations


def matmul_nf4_reference(weights_nf4: np.ndarray, scales: np.ndarray,
                          activations: np.ndarray, M: int, K: int) -> np.ndarray:
    """
    NF4 matrix-vector multiply (Python reference).
    output[i] = dot(dequant(weights_row_i), activations) for i in 0..M-1.

    Args:
        weights_nf4: uint8[M * K/2], rows concatenated
        scales: float32[M * K/BLOCKSIZE]
        activations: float32[K]
        M: number of rows (output neurons)
        K: number of columns (input dimension)

    Returns:
        float32[M] output vector
    """
    output = np.zeros(M, dtype=np.float32)
    bytes_per_row = K // 2
    blocks_per_row = K // BLOCKSIZE

    for i in range(M):
        row_nf4 = weights_nf4[i * bytes_per_row:(i + 1) * bytes_per_row]
        row_scales = scales[i * blocks_per_row:(i + 1) * blocks_per_row]
        output[i] = dot_nf4_reference(row_nf4, row_scales, activations, K)

    return output


matmul_nf4_python = matmul_nf4_reference


# ============================================================
# ZMM register assignments for 2-pipeline kernel
# ============================================================

# Fixed registers
_ZMM_ACUM_EVEN_0 = 0    # accumulator: even weights, pipeline 0
_ZMM_ACUM_ODD_0 = 1     # accumulator: odd weights, pipeline 0
_ZMM_TABLE = 2           # NF4 lookup table (constant)
_ZMM_MASK = 3            # 0x0F mask (constant)

# Pipeline 0 temporaries
_ZMM_BYTES_0 = 4         # expanded bytes
_ZMM_NIB_LO_0 = 5        # low nibbles (even weights)
_ZMM_NIB_HI_0 = 6        # high nibbles (odd weights)
_ZMM_SCALE = 7           # broadcast scale
_ZMM_ACT_EVEN_0 = 8      # even activations
_ZMM_ACT_ODD_0 = 9       # odd activations

# Pipeline 1 accumulators and temporaries
_ZMM_ACUM_EVEN_1 = 10
_ZMM_ACUM_ODD_1 = 11
_ZMM_BYTES_1 = 14
_ZMM_NIB_LO_1 = 15
_ZMM_NIB_HI_1 = 16
_ZMM_ACT_EVEN_1 = 18
_ZMM_ACT_ODD_1 = 19


# ============================================================
# Matmul scaffolding: shared by base and baked kernels
# ============================================================

def _emit_matmul_prologue(emit):
    """Emit the fixed prologue for NF4 matmul kernels (save regs, embed data, load constants)."""
    # Save callee-saved registers
    emit.push(RBX)
    emit.push(R12)
    emit.push(R13)
    emit.push(R14)
    emit.push(R15)
    emit.push(RBP)

    # Save arguments (System V AMD64 ABI)
    # RDI=weights_nf4, RSI=scales, RDX=act_reord, RCX=output, R8=M, R9=K
    emit.mov_reg_reg(R12, RDI)   # R12 = weights_nf4
    emit.mov_reg_reg(R13, RSI)   # R13 = scales
    emit.mov_reg_reg(R14, RDX)   # R14 = act_reord
    emit.mov_reg_reg(R15, RCX)   # R15 = output
    emit.mov_reg_reg(RBP, R8)    # RBP = M (rows)
    # R9 = K (columns) — already there

    # Save K on stack
    emit.push(R9)                # [RSP] = K

    # Embed constant data
    emit.jmp("after_data")

    emit.label("nf4_tabla")
    for val in NF4_TABLE:
        emit._emit_bytes(struct.pack("<f", float(val)))

    emit.label("mask_0f")
    for _ in range(16):
        emit._emit_bytes(struct.pack("<I", 0x0000000F))

    emit.label("after_data")

    # CALL $+5 / POP RBX trick to get data address
    emit._emit(0xE8, 0x00, 0x00, 0x00, 0x00)
    emit.pop(RBX)

    # Load constants into ZMM registers
    # RBX = after_data + 5; table is 128+5=133 bytes before; mask is 64+5=69 before
    emit.vmovups_zmm_mem(ZMM2, RBX, -133)     # ZMM2 = NF4 table
    emit.vmovdqu32_zmm_mem(ZMM3, RBX, -69)    # ZMM3 = 0x0F mask

    # Precompute row constants
    # R8 = bytes_per_row = K / 2
    emit.mov_reg_mem(R8, RSP, 0)
    emit.shr_reg_imm8(R8, 1)

    # R11 = scale_bytes_per_row = (K / 64) * 4 = K >> 4
    emit.mov_reg_mem(R11, RSP, 0)
    emit.shr_reg_imm8(R11, 4)

    # Set up row loop
    emit.mov_reg_reg(RDI, R12)     # RDI = current row weights ptr
    emit.mov_reg_reg(RSI, R13)     # RSI = current row scales ptr
    emit.mov_reg_imm64(RCX, 0)    # RCX = row counter


def _emit_matmul_row_start(emit, n_accumulators):
    """Emit row loop header and accumulator initialization."""
    emit.label("loop_fila")
    emit.cmp_reg_reg(RCX, RBP)
    emit.jge("done")

    emit.push(RCX)                 # save row index

    # Zero accumulators
    emit.vxorps_zmm_zmm_zmm(ZMM0, ZMM0, ZMM0)
    emit.vxorps_zmm_zmm_zmm(ZMM1, ZMM1, ZMM1)
    if n_accumulators >= 2:
        emit.vxorps_zmm_zmm_zmm(_ZMM_ACUM_EVEN_1, _ZMM_ACUM_EVEN_1, _ZMM_ACUM_EVEN_1)
        emit.vxorps_zmm_zmm_zmm(_ZMM_ACUM_ODD_1, _ZMM_ACUM_ODD_1, _ZMM_ACUM_ODD_1)

    # Inner loop counters
    emit.mov_reg_imm64(R9, 0)     # weight_idx
    emit.mov_reg_imm64(R10, 0)    # byte_idx
    emit.mov_reg_imm64(RAX, 0)
    emit.push(RAX)                 # [RSP+0] = act_offset
    emit.mov_reg_imm64(RDX, 0)    # scale_offset

    # Load K for comparison (stack: +0=act_offset, +8=row_idx, +16=K)
    emit.mov_reg_mem(RCX, RSP, 16)

    # Inner loop header
    emit.label("loop_k")
    emit.cmp_reg_reg(R9, RCX)
    emit.jge("fila_reduce")


def _emit_matmul_inner_counters(emit, n_accumulators):
    """Emit inner loop counter advancement and scale update."""
    weights_per_iter = 32 * n_accumulators
    bytes_per_iter = 16 * n_accumulators
    act_bytes_per_iter = 128 * n_accumulators

    emit.add_reg_imm32(R9, weights_per_iter)
    emit.add_reg_imm32(R10, bytes_per_iter)

    # act_offset += act_bytes_per_iter
    emit.mov_reg_mem(RAX, RSP, 0)
    emit.add_reg_imm32(RAX, act_bytes_per_iter)
    emit.mov_mem_reg(RSP, RAX, 0)

    # Advance scale every 64 weights
    emit.mov_reg_reg(RAX, R9)
    emit.and_reg_imm32(RAX, 63)
    emit.cmp_reg_imm32(RAX, 0)
    emit.jne("skip_escala_m")
    emit.add_reg_imm32(RDX, 4)
    emit.label("skip_escala_m")

    emit.jmp("loop_k")


def _emit_matmul_row_reduce(emit, n_accumulators):
    """Emit horizontal reduction, result store, and row advancement."""
    emit.label("fila_reduce")

    emit.pop(RAX)                  # discard act_offset

    # Sum extra accumulators into primary
    if n_accumulators >= 2:
        emit.vaddps_zmm_zmm_zmm(ZMM0, ZMM0, _ZMM_ACUM_EVEN_1)
        emit.vaddps_zmm_zmm_zmm(ZMM1, ZMM1, _ZMM_ACUM_ODD_1)

    emit.vaddps_zmm_zmm_zmm(ZMM0, ZMM0, ZMM1)

    # Horizontal reduction: store ZMM0 to stack, sum 16 floats with SSE
    emit.sub_reg_imm32(RSP, 64)
    emit.vmovups_mem_zmm(RSP, ZMM0, 0)

    emit.xorps_xmm_xmm(XMM0, XMM0)
    emit.mov_reg_imm64(RAX, 0)
    emit.label("reduce_loop_m")
    emit.cmp_reg_imm32(RAX, 16)
    emit.jge("reduce_done_m")
    emit.mov_reg_reg(RCX, RAX)
    emit.shl_reg_imm8(RCX, 2)
    emit.add_reg_reg(RCX, RSP)
    emit.addss_xmm_mem(XMM0, RCX, 0)
    emit.inc_reg(RAX)
    emit.jmp("reduce_loop_m")

    emit.label("reduce_done_m")
    emit.add_reg_imm32(RSP, 64)

    # Store result: output[row_idx]
    emit.pop(RCX)                  # RCX = row_idx
    emit.mov_reg_reg(RAX, RCX)
    emit.shl_reg_imm8(RAX, 2)
    emit.add_reg_reg(RAX, R15)
    emit.movss_mem_xmm(RAX, XMM0, 0)

    # Advance row pointers
    emit.add_reg_reg(RDI, R8)      # weights_ptr += bytes_per_row
    emit.add_reg_reg(RSI, R11)     # scales_ptr += scale_bytes_per_row

    emit.inc_reg(RCX)
    emit.jmp("loop_fila")


def _emit_matmul_epilogue(emit):
    """Emit function epilogue (clean stack, restore regs, ret)."""
    emit.label("done")
    emit.pop(RAX)                  # clean K from stack
    emit.pop(RBP)
    emit.pop(R15)
    emit.pop(R14)
    emit.pop(R13)
    emit.pop(R12)
    emit.pop(RBX)
    emit.ret()


# ============================================================
# Instruction emitter for matmul register convention
# ============================================================
# Matmul registers:
#   RDI = current row weights ptr
#   RSI = current row scales ptr
#   R14 = act_reord ptr (constant)
#   R10 = byte_idx within row
#   [RSP+0] = act_offset (on stack, no free registers)
#   RDX = scale_offset

def _emit_vpmovzxbd(emit, dst, offset):
    """Load 16 NF4 bytes and zero-extend to 16 dwords."""
    emit.mov_reg_reg(RAX, RDI)
    emit.add_reg_reg(RAX, R10)
    if offset:
        emit.add_reg_imm32(RAX, offset)
    emit.vpmovzxbd_zmm_mem(dst, RAX, 0)

def _emit_vpandd(emit, dst, src1, src2):
    emit.vpandd_zmm_zmm_zmm(dst, src1, src2)

def _emit_vpsrld(emit, dst, src, count):
    emit.vpsrld_zmm_zmm_imm8(dst, src, count)

def _emit_vpermps(emit, dst, idx, tabla):
    emit.vpermps_zmm_zmm_zmm(dst, idx, tabla)

def _emit_vbroadcastss(emit):
    """Broadcast scale from current row's scale pointer."""
    emit.mov_reg_reg(RAX, RSI)
    emit.add_reg_reg(RAX, RDX)
    emit.vbroadcastss_zmm_mem(_ZMM_SCALE, RAX, 0)

def _emit_vmulps(emit, dst, src1, src2):
    emit.vmulps_zmm_zmm_zmm(dst, src1, src2)

def _emit_vmovups_act(emit, dst, offset):
    """Load 16 activations from act_reord + act_offset + offset."""
    emit.mov_reg_mem(RAX, RSP, 0)    # RAX = act_offset
    emit.add_reg_reg(RAX, R14)       # RAX = act_reord + act_offset
    if offset != 0:
        emit.add_reg_imm32(RAX, offset)
    emit.vmovups_zmm_mem(dst, RAX, 0)

def _emit_vfmadd231ps(emit, dst, src1, src2):
    emit.vfmadd231ps_zmm_zmm_zmm(dst, src1, src2)

def _emit_prefetcht0(emit, distance):
    """Prefetch weights at byte_idx + distance."""
    emit.mov_reg_reg(RAX, RDI)
    emit.add_reg_reg(RAX, R10)
    emit.prefetchT0_mem(RAX, distance)

def _emit_nop(emit, count):
    for _ in range(count):
        emit._emit(0x90)


# ============================================================
# Base 2-pipeline kernel generator
# ============================================================

def generate_nf4_matmul_kernel():
    """
    Generate the base 2-pipeline NF4 matmul kernel.

    This is the non-evolved baseline with 2 independent accumulator pipelines
    for hiding FMA latency on Zen 4 (4-cycle latency, 2-cycle throughput).

    Signature: void(uint8* weights, float* scales, float* act_reord,
                    float* output, int64 M, int64 K)
    """
    emit = X86Emitter()
    _emit_matmul_prologue(emit)
    _emit_matmul_row_start(emit, n_accumulators=2)

    # --- Inner loop body: 2-pipeline baseline ---
    # Pipeline 0: ZMM4-9 -> accum ZMM0/1
    # Pipeline 1: ZMM14-19 -> accum ZMM10/11

    # Load weights
    _emit_vpmovzxbd(emit, _ZMM_BYTES_0, 0)
    _emit_vpmovzxbd(emit, _ZMM_BYTES_1, 16)

    # Separate nibbles
    _emit_vpandd(emit, _ZMM_NIB_LO_0, _ZMM_BYTES_0, _ZMM_MASK)
    _emit_vpandd(emit, _ZMM_NIB_LO_1, _ZMM_BYTES_1, _ZMM_MASK)
    _emit_vpsrld(emit, _ZMM_NIB_HI_0, _ZMM_BYTES_0, 4)
    _emit_vpsrld(emit, _ZMM_NIB_HI_1, _ZMM_BYTES_1, 4)

    # Table lookup
    _emit_vpermps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_TABLE)
    _emit_vpermps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_TABLE)
    _emit_vpermps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_TABLE)
    _emit_vpermps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_TABLE)

    # Scale
    _emit_vbroadcastss(emit)
    _emit_vmulps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_SCALE)
    _emit_vmulps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_SCALE)
    _emit_vmulps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_SCALE)
    _emit_vmulps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_SCALE)

    # Load activations
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_0, 0)
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_0, 64)
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_1, 128)
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_1, 192)

    # FMA accumulate (interleaved for latency hiding)
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_0, _ZMM_NIB_LO_0, _ZMM_ACT_EVEN_0)
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_1, _ZMM_NIB_LO_1, _ZMM_ACT_EVEN_1)
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_0, _ZMM_NIB_HI_0, _ZMM_ACT_ODD_0)
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_1, _ZMM_NIB_HI_1, _ZMM_ACT_ODD_1)

    _emit_matmul_inner_counters(emit, n_accumulators=2)
    _emit_matmul_row_reduce(emit, n_accumulators=2)
    _emit_matmul_epilogue(emit)

    return emit


# Backward-compatible alias
generar_kernel_nf4_matmul = generate_nf4_matmul_kernel


# ============================================================
# Baked evolved kernels
# ============================================================
# Each function generates a full matmul kernel with the evolved
# instruction ordering hardcoded in the inner loop body.
# The scaffolding (prologue, row loop, counters, reduction, epilogue)
# is identical across all kernels.

def _baked_kernel_1024x2048():
    """
    Evolved kernel for 1024x2048 (80B gate_up projection).
    +2.93% vs baseline (94.2us -> 91.5us).

    Key optimizations discovered:
      - Early scale broadcast at position 2 (16 extra cycles for data arrival)
      - Prefetch at position 13 (between lookups and multiplies)
    """
    emit = X86Emitter()
    _emit_matmul_prologue(emit)
    _emit_matmul_row_start(emit, n_accumulators=2)

    # Evolved inner loop body (24 instructions)
    _emit_vpmovzxbd(emit, _ZMM_BYTES_0, 0)          # [0]
    _emit_vpmovzxbd(emit, _ZMM_BYTES_1, 16)         # [1]
    _emit_vbroadcastss(emit)                          # [2] early scale
    _emit_vpandd(emit, _ZMM_NIB_LO_0, _ZMM_BYTES_0, _ZMM_MASK)  # [3]
    _emit_vpandd(emit, _ZMM_NIB_LO_1, _ZMM_BYTES_1, _ZMM_MASK)  # [4]
    _emit_vpsrld(emit, _ZMM_NIB_HI_0, _ZMM_BYTES_0, 4)          # [5]
    _emit_vpsrld(emit, _ZMM_NIB_HI_1, _ZMM_BYTES_1, 4)          # [6]
    _emit_vpermps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_TABLE)  # [7]
    _emit_vpermps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_TABLE)  # [8]
    _emit_vpermps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_TABLE)  # [9]
    _emit_vpermps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_TABLE)  # [10]
    _emit_vmulps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_SCALE)  # [11]
    _emit_vmulps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_SCALE)  # [12]
    _emit_prefetcht0(emit, 256)                       # [13]
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_0, 0)     # [14]
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_0, 64)     # [15]
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_1, 128)   # [16]
    _emit_vmulps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_SCALE)  # [17]
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_1, 192)    # [18]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_0, _ZMM_NIB_LO_0, _ZMM_ACT_EVEN_0)  # [19]
    _emit_vmulps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_SCALE)  # [20]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_0, _ZMM_NIB_HI_0, _ZMM_ACT_ODD_0)    # [21]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_1, _ZMM_NIB_LO_1, _ZMM_ACT_EVEN_1)  # [22]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_1, _ZMM_NIB_HI_1, _ZMM_ACT_ODD_1)    # [23]

    _emit_matmul_inner_counters(emit, n_accumulators=2)
    _emit_matmul_row_reduce(emit, n_accumulators=2)
    _emit_matmul_epilogue(emit)
    return emit


def _baked_kernel_2048x512():
    """
    Evolved kernel for 2048x512 (80B down_proj).
    +19.25% vs baseline (74.9us -> 60.2us). The champion.

    Key optimizations discovered:
      - NOP alignment (2 bytes) before activation loads
      - Inverted activation loading order (P1 before P0)
      - Interleaved nibble separation between pipelines
      - FMA order 0,10,11,1 for maximum latency hiding
    """
    emit = X86Emitter()
    _emit_matmul_prologue(emit)
    _emit_matmul_row_start(emit, n_accumulators=2)

    # Evolved inner loop body (24 instructions + NOP)
    _emit_vpmovzxbd(emit, _ZMM_BYTES_0, 0)          # [0]
    _emit_vpmovzxbd(emit, _ZMM_BYTES_1, 16)         # [1]
    _emit_vpandd(emit, _ZMM_NIB_LO_0, _ZMM_BYTES_0, _ZMM_MASK)  # [2]
    _emit_vpsrld(emit, _ZMM_NIB_HI_1, _ZMM_BYTES_1, 4)          # [3] P1 high before P0
    _emit_vbroadcastss(emit)                          # [4]
    _emit_vpandd(emit, _ZMM_NIB_LO_1, _ZMM_BYTES_1, _ZMM_MASK)  # [5]
    _emit_vpsrld(emit, _ZMM_NIB_HI_0, _ZMM_BYTES_0, 4)          # [6]
    _emit_vpermps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_TABLE)  # [7]
    _emit_vpermps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_TABLE)  # [8]
    _emit_vpermps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_TABLE)  # [9]
    _emit_vpermps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_TABLE)  # [10]
    _emit_vmulps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_SCALE)  # [11]
    _emit_vmulps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_SCALE)  # [12]
    _emit_vmulps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_SCALE)  # [13]
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_1, 128)   # [14] P1 first (far cache line)
    _emit_vmulps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_SCALE)  # [15]
    _emit_nop(emit, 2)                                # [16] cache-line alignment
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_0, 0)     # [17]
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_1, 192)    # [18]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_0, _ZMM_NIB_LO_0, _ZMM_ACT_EVEN_0)  # [19]
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_0, 64)     # [20]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_1, _ZMM_NIB_LO_1, _ZMM_ACT_EVEN_1)  # [21]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_1, _ZMM_NIB_HI_1, _ZMM_ACT_ODD_1)    # [22]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_0, _ZMM_NIB_HI_0, _ZMM_ACT_ODD_0)    # [23]

    _emit_matmul_inner_counters(emit, n_accumulators=2)
    _emit_matmul_row_reduce(emit, n_accumulators=2)
    _emit_matmul_epilogue(emit)
    return emit


def _baked_kernel_3072x2048():
    """
    Evolved kernel for 3072x2048 (30B gate_up projection).
    +3.90% vs baseline (276.1us -> 265.3us).

    Key optimizations discovered:
      - Prefetch between lookups and scale broadcast
      - Reverse-distance activation loading (far offsets first)
    """
    emit = X86Emitter()
    _emit_matmul_prologue(emit)
    _emit_matmul_row_start(emit, n_accumulators=2)

    # Evolved inner loop body (24 instructions)
    _emit_vpmovzxbd(emit, _ZMM_BYTES_0, 0)          # [0]
    _emit_vpmovzxbd(emit, _ZMM_BYTES_1, 16)         # [1]
    _emit_vpandd(emit, _ZMM_NIB_LO_0, _ZMM_BYTES_0, _ZMM_MASK)  # [2]
    _emit_vpandd(emit, _ZMM_NIB_LO_1, _ZMM_BYTES_1, _ZMM_MASK)  # [3]
    _emit_vpsrld(emit, _ZMM_NIB_HI_0, _ZMM_BYTES_0, 4)          # [4]
    _emit_vpsrld(emit, _ZMM_NIB_HI_1, _ZMM_BYTES_1, 4)          # [5]
    _emit_vpermps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_TABLE)  # [6]
    _emit_vpermps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_TABLE)  # [7]
    _emit_vpermps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_TABLE)  # [8]
    _emit_prefetcht0(emit, 256)                       # [9] between lookups and scale
    _emit_vbroadcastss(emit)                          # [10]
    _emit_vpermps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_TABLE)  # [11]
    _emit_vmulps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_SCALE)  # [12]
    _emit_vmulps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_SCALE)  # [13]
    _emit_vmulps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_SCALE)  # [14]
    _emit_vmulps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_SCALE)  # [15]
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_1, 192)    # [16] farthest first
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_0, 64)     # [17]
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_1, 128)   # [18]
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_0, 0)     # [19] nearest last
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_0, _ZMM_NIB_LO_0, _ZMM_ACT_EVEN_0)  # [20]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_1, _ZMM_NIB_LO_1, _ZMM_ACT_EVEN_1)  # [21]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_0, _ZMM_NIB_HI_0, _ZMM_ACT_ODD_0)    # [22]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_1, _ZMM_NIB_HI_1, _ZMM_ACT_ODD_1)    # [23]

    _emit_matmul_inner_counters(emit, n_accumulators=2)
    _emit_matmul_row_reduce(emit, n_accumulators=2)
    _emit_matmul_epilogue(emit)
    return emit


def _baked_kernel_2048x1536():
    """
    Evolved kernel for 2048x1536 (30B down_proj).
    +2.89% vs baseline (143.3us -> 139.1us).

    Key optimizations discovered:
      - Prefetch as very first instruction (anticipatory)
      - Interleaved P1/P0 nibble separation
    """
    emit = X86Emitter()
    _emit_matmul_prologue(emit)
    _emit_matmul_row_start(emit, n_accumulators=2)

    # Evolved inner loop body (24 instructions)
    _emit_prefetcht0(emit, 256)                       # [0] first instruction!
    _emit_vpmovzxbd(emit, _ZMM_BYTES_0, 0)          # [1]
    _emit_vpmovzxbd(emit, _ZMM_BYTES_1, 16)         # [2]
    _emit_vpandd(emit, _ZMM_NIB_LO_1, _ZMM_BYTES_1, _ZMM_MASK)  # [3] P1 first
    _emit_vpandd(emit, _ZMM_NIB_LO_0, _ZMM_BYTES_0, _ZMM_MASK)  # [4]
    _emit_vpsrld(emit, _ZMM_NIB_HI_0, _ZMM_BYTES_0, 4)          # [5]
    _emit_vpsrld(emit, _ZMM_NIB_HI_1, _ZMM_BYTES_1, 4)          # [6]
    _emit_vbroadcastss(emit)                          # [7]
    _emit_vpermps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_TABLE)  # [8]
    _emit_vpermps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_TABLE)  # [9]
    _emit_vpermps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_TABLE)  # [10]
    _emit_vpermps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_TABLE)  # [11]
    _emit_vmulps(emit, _ZMM_NIB_LO_0, _ZMM_NIB_LO_0, _ZMM_SCALE)  # [12]
    _emit_vmulps(emit, _ZMM_NIB_LO_1, _ZMM_NIB_LO_1, _ZMM_SCALE)  # [13]
    _emit_vmulps(emit, _ZMM_NIB_HI_1, _ZMM_NIB_HI_1, _ZMM_SCALE)  # [14]
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_0, 0)     # [15]
    _emit_vmulps(emit, _ZMM_NIB_HI_0, _ZMM_NIB_HI_0, _ZMM_SCALE)  # [16]
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_0, 64)     # [17]
    _emit_vmovups_act(emit, _ZMM_ACT_EVEN_1, 128)   # [18]
    _emit_vmovups_act(emit, _ZMM_ACT_ODD_1, 192)    # [19]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_0, _ZMM_NIB_LO_0, _ZMM_ACT_EVEN_0)  # [20]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_EVEN_1, _ZMM_NIB_LO_1, _ZMM_ACT_EVEN_1)  # [21]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_0, _ZMM_NIB_HI_0, _ZMM_ACT_ODD_0)    # [22]
    _emit_vfmadd231ps(emit, _ZMM_ACUM_ODD_1, _ZMM_NIB_HI_1, _ZMM_ACT_ODD_1)    # [23]

    _emit_matmul_inner_counters(emit, n_accumulators=2)
    _emit_matmul_row_reduce(emit, n_accumulators=2)
    _emit_matmul_epilogue(emit)
    return emit


# Registry of baked kernels: (M, K) -> generator function
_BAKED_KERNELS = {
    (1024, 2048): _baked_kernel_1024x2048,
    (2048, 512):  _baked_kernel_2048x512,
    (3072, 2048): _baked_kernel_3072x2048,
    (2048, 1536): _baked_kernel_2048x1536,
}


# ============================================================
# Compilation: emitter -> executable function
# ============================================================

def compile_nf4_matmul(emit):
    """
    Compile an NF4 matmul emitter into an executable function.

    Args:
        emit: X86Emitter with a complete matmul kernel

    Returns:
        Callable: nf4_matmul(weights_nf4, scales, act_reord, M, K) -> np.ndarray[M]
    """
    emit._resolve_fixups()
    code_bytes = bytes(emit.code)
    page_size = mmap.PAGESIZE
    alloc_size = ((len(code_bytes) + page_size - 1) // page_size) * page_size

    buf = mmap.mmap(
        -1, alloc_size,
        prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
    )
    buf.write(code_bytes)
    buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))

    func_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,   # weights_nf4
        ctypes.c_void_p,   # scales
        ctypes.c_void_p,   # act_reord
        ctypes.c_void_p,   # output
        ctypes.c_int64,    # M
        ctypes.c_int64,    # K
    )
    raw_fn = func_type(buf_addr)
    raw_fn._buf = buf

    def nf4_matmul_wrapper(weights_nf4: np.ndarray, scales: np.ndarray,
                            act_reord: np.ndarray, M: int, K: int) -> np.ndarray:
        """
        NF4 fused dequant+matmul on AVX-512.

        Args:
            weights_nf4: uint8[M * K/2], rows concatenated
            scales: float32[M * K/BLOCKSIZE]
            act_reord: float32[K], reordered with reorder_activations()
            M: number of output rows
            K: input dimension (must be multiple of 64)

        Returns:
            float32[M] output vector
        """
        assert weights_nf4.dtype == np.uint8
        assert scales.dtype == np.float32
        assert act_reord.dtype == np.float32
        assert K % 64 == 0

        output = np.zeros(M, dtype=np.float32)
        raw_fn(
            weights_nf4.ctypes.data,
            scales.ctypes.data,
            act_reord.ctypes.data,
            output.ctypes.data,
            M,
            K,
        )
        return output

    nf4_matmul_wrapper._buf = buf
    nf4_matmul_wrapper._raw = raw_fn
    nf4_matmul_wrapper._emit = emit
    return nf4_matmul_wrapper


# Backward-compatible alias
compilar_nf4_matmul = compile_nf4_matmul


# ============================================================
# Public API: compile_best_nf4_matmul
# ============================================================

_kernel_cache = {}


def compile_best_nf4_matmul(M: int, K: int):
    """
    Compile the best available NF4 matmul kernel for dimensions M x K.

    If an evolved kernel exists for the given dimensions, it is used.
    Otherwise, the base 2-pipeline kernel is generated.

    Kernels are cached: subsequent calls with the same (M, K) return
    the previously compiled kernel instantly.

    Args:
        M: number of output rows
        K: input dimension

    Returns:
        Callable: kernel(weights_nf4, scales, act_reord, M, K) -> np.ndarray[M]

    Example:
        from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations
        kernel = compile_best_nf4_matmul(M=2048, K=1536)
        nf4, scales = quantize_nf4(weights)
        act_reord = reorder_activations(activations)
        output = kernel(nf4, scales, act_reord, M, K)
    """
    cache_key = (M, K)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    # Check for baked evolved kernel
    if cache_key in _BAKED_KERNELS:
        emit = _BAKED_KERNELS[cache_key]()
        origin = f"evolved ({cache_key[0]}x{cache_key[1]})"
    else:
        emit = generate_nf4_matmul_kernel()
        origin = "base (2-pipeline)"

    kernel = compile_nf4_matmul(emit)
    kernel._origin = origin
    kernel._origen = origin  # backward compat
    _kernel_cache[cache_key] = kernel
    return kernel


# Backward-compatible alias
compilar_mejor_nf4_matmul = compile_best_nf4_matmul
