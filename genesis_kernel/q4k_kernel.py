#!/usr/bin/env python3
"""
Genesis Kernel — Q4_K dot product kernel (turbo7).

Generates an x86-64 kernel that computes ggml's vec_dot_q4_K_q8_K
using AVX-512 VNNI instructions in YMM registers (256-bit).

The turbo7 kernel was built by:
1. Reverse-engineering ggml's compiled Q4_K dot product (objdump)
2. Reimplementing it in our x86 emitter (turbo6 baseline)
3. Brute-force injection of 5,000 random hint instructions
4. A PREFETCHNTA [R9+358] at position 19 survived 3-stage verification

Result: bit-exact output vs ggml, 1-12% faster depending on N.

ISA: AVX-512 F + VNNI (Zen 4, Alder Lake+, Sapphire Rapids+)
Registers: YMM 256-bit (1 cycle on Zen 4 vs 2 for ZMM 512-bit)
Size: 1,078 bytes, 34 inner loop ops
"""

import struct
import ctypes
from .x86_emitter import (
    X86Emitter,
    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
)


# ================================================================
# Genome representation
# ================================================================

class OpQ4K:
    """Single operation in the Q4_K inner loop genome."""
    __slots__ = ("nombre", "args", "grupo", "comentario")

    def __init__(self, nombre: str, args: dict, grupo: str = "", comentario: str = ""):
        self.nombre = nombre
        self.args = args
        self.grupo = grupo
        self.comentario = comentario

    def __repr__(self):
        parts = f"OpQ4K({self.nombre}, {self.args}"
        if self.grupo:
            parts += f", grupo={self.grupo}"
        return parts + ")"


# ================================================================
# Turbo7 genome (turbo6 + prefetchnta hint)
# ================================================================

def _turbo6_ops():
    """33 inner loop ops of turbo6: 4 groups × 8 ops + 1 scale prep."""
    return [
        # Group 0: weights [RCX-0x80], q8 [R9+4] and [R9+0x24]
        OpQ4K("vmovdqu_load", {"dst": "YMM1", "base": "RCX", "offset": -0x80}, grupo="g0"),
        OpQ4K("vpsrlw", {"dst": "YMM15", "src": "YMM1", "imm": 4}, grupo="g0"),
        OpQ4K("vpand", {"dst": "YMM1", "src1": "YMM1", "src2": "YMM3"}, grupo="g0"),
        OpQ4K("vpmaddubsw_mem", {"dst": "YMM1", "src": "YMM1", "q8_base": "R9", "q8_off": 4}, grupo="g0_low"),
        OpQ4K("vpdpwssd", {"dst": "YMM22", "scale": "YMM2", "src": "YMM1"}, grupo="g0_low"),
        OpQ4K("vpand", {"dst": "YMM1", "src1": "YMM15", "src2": "YMM3"}, grupo="g0_high"),
        OpQ4K("vpmaddubsw_mem", {"dst": "YMM1", "src": "YMM1", "q8_base": "R9", "q8_off": 0x24}, grupo="g0_high"),
        OpQ4K("vpdpwssd", {"dst": "YMM23", "scale": "YMM16", "src": "YMM1"}, grupo="g0_high"),
        # Group 1: weights [RCX-0x60], q8 [R9+0x44] and [R9+0x64]
        OpQ4K("vmovdqu_load", {"dst": "YMM1", "base": "RCX", "offset": -0x60}, grupo="g1"),
        OpQ4K("vpsrlw", {"dst": "YMM15", "src": "YMM1", "imm": 4}, grupo="g1"),
        OpQ4K("vpand", {"dst": "YMM1", "src1": "YMM1", "src2": "YMM3"}, grupo="g1"),
        OpQ4K("vpmaddubsw_mem", {"dst": "YMM1", "src": "YMM1", "q8_base": "R9", "q8_off": 0x44}, grupo="g1_low"),
        OpQ4K("vpdpwssd", {"dst": "YMM24", "scale": "YMM18", "src": "YMM1"}, grupo="g1_low"),
        OpQ4K("vpand", {"dst": "YMM1", "src1": "YMM15", "src2": "YMM3"}, grupo="g1_high"),
        OpQ4K("vpmaddubsw_mem", {"dst": "YMM1", "src": "YMM1", "q8_base": "R9", "q8_off": 0x64}, grupo="g1_high"),
        OpQ4K("vpdpwssd", {"dst": "YMM25", "scale": "YMM17", "src": "YMM1"}, grupo="g1_high"),
        # Group 2: weights [RCX-0x40], q8 [R9+0x84] and [R9+0xa4]
        OpQ4K("vmovdqu_load", {"dst": "YMM1", "base": "RCX", "offset": -0x40}, grupo="g2"),
        OpQ4K("vpsrlw", {"dst": "YMM15", "src": "YMM1", "imm": 4}, grupo="g2"),
        OpQ4K("vpand", {"dst": "YMM1", "src1": "YMM1", "src2": "YMM3"}, grupo="g2"),
        OpQ4K("vpmaddubsw_mem", {"dst": "YMM1", "src": "YMM1", "q8_base": "R9", "q8_off": 0x84}, grupo="g2_low"),
        OpQ4K("vpdpwssd", {"dst": "YMM22", "scale": "YMM19", "src": "YMM1"}, grupo="g2_low"),
        OpQ4K("vpand", {"dst": "YMM1", "src1": "YMM15", "src2": "YMM3"}, grupo="g2_high"),
        OpQ4K("vpmaddubsw_mem", {"dst": "YMM1", "src": "YMM1", "q8_base": "R9", "q8_off": 0xa4}, grupo="g2_high"),
        OpQ4K("vpdpwssd", {"dst": "YMM23", "scale": "YMM20", "src": "YMM1"}, grupo="g2_high"),
        # Group 3: weights [RCX-0x20], q8 [R9+0xc4] and [R9+0xe4]
        OpQ4K("vpshufb_scale", {"dst": "YMM1", "src1": "YMM0_scales", "src2": "YMM7"}, grupo="g3_prep"),
        OpQ4K("vmovdqu_load", {"dst": "YMM0", "base": "RCX", "offset": -0x20}, grupo="g3"),
        OpQ4K("vpsrlw", {"dst": "YMM15", "src": "YMM0", "imm": 4}, grupo="g3"),
        OpQ4K("vpand", {"dst": "YMM0", "src1": "YMM0", "src2": "YMM3"}, grupo="g3"),
        OpQ4K("vpmaddubsw_mem", {"dst": "YMM0", "src": "YMM0", "q8_base": "R9", "q8_off": 0xc4}, grupo="g3_low"),
        OpQ4K("vpdpwssd", {"dst": "YMM24", "scale": "YMM21", "src": "YMM0"}, grupo="g3_low"),
        OpQ4K("vpand", {"dst": "YMM0", "src1": "YMM15", "src2": "YMM3"}, grupo="g3_high"),
        OpQ4K("vpmaddubsw_mem", {"dst": "YMM0", "src": "YMM0", "q8_base": "R9", "q8_off": 0xe4}, grupo="g3_high"),
        OpQ4K("vpdpwssd", {"dst": "YMM25", "scale": "YMM1", "src": "YMM0"}, grupo="g3_high"),
    ]


def turbo7_ops():
    """34 inner loop ops: turbo6 + PREFETCHNTA [R9+358] at position 19.
    Found by brute-force injection (5,000 attempts, 3-stage verification).
    R9+358 = next Q8K block's 2nd cache line (292 + 66 bytes ahead)."""
    ops = _turbo6_ops()
    ops.insert(19, OpQ4K(
        "prefetchnta", {"base": "R9", "offset": 358},
        grupo="hint",
        comentario="turbo7: prefetch next Q8K block 2nd cache line (non-temporal)",
    ))
    return ops


# ================================================================
# Register name → number helpers
# ================================================================

_YMM_MAP = {f"YMM{i}": i for i in range(32)}
_REG_MAP = {
    "RAX": 0, "RCX": 1, "RDX": 2, "RBX": 3,
    "RSP": 4, "RBP": 5, "RSI": 6, "RDI": 7,
    "R8": 8, "R9": 9, "R10": 10, "R11": 11,
    "R12": 12, "R13": 13, "R14": 14, "R15": 15,
}


def _ymm(name: str) -> int:
    return _YMM_MAP[name]


def _reg(name: str) -> int:
    return _REG_MAP[name]


# ================================================================
# Op → machine code translation
# ================================================================

def _emit_op(emit, op: OpQ4K):
    """Translate one OpQ4K to x86 machine code."""
    n = op.nombre
    a = op.args

    if n == "vmovdqu_load":
        emit.vex_vmovdqu_ymm_mem(_ymm(a["dst"]), _reg(a["base"]), a["offset"])
    elif n == "vpsrlw":
        emit.vex_vpsrlw_ymm_imm8(_ymm(a["dst"]), _ymm(a["src"]), a["imm"])
    elif n == "vpand":
        emit.vex_vpand_ymm(_ymm(a["dst"]), _ymm(a["src1"]), _ymm(a["src2"]))
    elif n == "vpmaddubsw_mem":
        emit.vex_vpmaddubsw_ymm_mem(_ymm(a["dst"]), _ymm(a["src"]), _reg(a["q8_base"]), a["q8_off"])
    elif n == "vpdpwssd":
        emit.vpdpwssd_ymm_ymm_ymm(_ymm(a["dst"]), _ymm(a["scale"]), _ymm(a["src"]))
    elif n == "vpshufb_scale":
        emit.vex_vpshufb_ymm(_ymm(a["dst"]), 0, _ymm(a["src2"]))  # YMM0 = scales
    elif n == "prefetcht0":
        emit.prefetchT0_mem(_reg(a["base"]), a["offset"])
    elif n == "prefetcht1":
        emit.prefetchT1_mem(_reg(a["base"]), a["offset"])
    elif n == "prefetcht2":
        emit.prefetchT2_mem(_reg(a["base"]), a["offset"])
    elif n == "prefetchnta":
        emit.prefetchNTA_mem(_reg(a["base"]), a["offset"])
    elif n == "vmovdqa_load":
        emit.vmovdqa_ymm_mem(_ymm(a["dst"]), _reg(a["base"]), a["offset"])
    elif n == "multi_nop":
        emit.multi_nop(a.get("size", 1))
    elif n == "lfence":
        emit.lfence()
    else:
        raise ValueError(f"Unknown Q4K op: {n}")


# ================================================================
# Prologue: decode d, dmin, scales per block
# ================================================================

def _emit_prologue(emit):
    """Emit per-block prologue (fixed, not mutated). Decodes Q4K scales."""
    # Zero 4 VNNI accumulators
    emit.vxorps_ymm_ymm_ymm(22, 22, 22)
    emit.vxorps_ymm_ymm_ymm(23, 23, 23)
    emit.vxorps_ymm_ymm_ymm(24, 24, 24)
    emit.vxorps_ymm_ymm_ymm(25, 25, 25)

    # Load d_q8 (float32)
    emit.vex_vmovss_xmm_mem(0, R9, 0)
    # Load d_q4 (fp16)
    emit.movzx_r32_word_mem(RAX, RCX, 0)
    # Load 8 bytes of scales
    emit.mov_reg_mem(RDX, RCX, 4)
    # Increment block counter
    emit.inc_reg(RSI)
    # Load 4 more bytes of scales
    emit.mov_reg_mem32(R11, RCX, 12)
    # Negate d_q8 (for dmin term)
    emit.vex_vxorps_xmm_xmm_mem(1, 0, R14, 0)
    # Advance Q4K pointer
    emit.add_reg_imm32(RCX, 144)

    # d = d_q8 * fp32(d_q4)
    emit.vex_vmovd_xmm_reg(4, RAX)
    emit.vex_vcvtph2ps_xmm(4, 4)
    emit.mulss_xmm_xmm(4, 0)

    # dmin = -d_q8 * fp32(dmin_q4)
    emit.movzx_r32_word_mem(RAX, RCX, -0x8E)
    emit.vex_vmovd_xmm_reg(15, RAX)
    emit.vex_vcvtph2ps_xmm(15, 15)
    emit.mulss_xmm_xmm(1, 15)

    # Decode 6-bit scales from packed 12-byte field using bit manipulation
    emit.mov_reg_reg(RAX, R11)
    emit.shr_reg32_imm8(RAX, 4)
    emit.and_reg32_imm32(RAX, 0x0f0f0f0f)

    emit.mov_reg_reg(RBX, RDX)
    emit.shr_reg_imm8(RBX, 32)

    emit.mov_reg_reg(R8, RBX)
    emit.shr_reg32_imm8(R8, 6)
    emit.and_reg32_imm32(R8, 0x03030303)
    emit.shl_reg_imm8(R8, 4)
    emit.or_reg32_reg32(RAX, R8)

    emit.mov_reg_reg(R8, R11)
    emit.and_reg32_imm32(R8, 0x0f0f0f0f)

    emit.mov_reg_reg(R15, RDX)
    emit.shr_reg32_imm8(R15, 6)
    emit.and_reg32_imm32(R15, 0x03030303)
    emit.shl_reg_imm8(R15, 4)
    emit.or_reg32_reg32(R8, R15)

    emit.and_reg32_imm32(RBX, 0x3f3f3f3f)
    emit.and_reg32_imm32(RDX, 0x3f3f3f3f)

    # Build XMM0 = [utmp[0], utmp[1], utmp[2], utmp[3]] via VPINSRD
    emit.vex_vmovd_xmm_reg(0, RDX)
    emit.vex_vpinsrd_xmm_xmm_reg_imm8(0, 0, R8, 1)
    emit.vex_vpinsrd_xmm_xmm_reg_imm8(0, 0, RBX, 2)
    emit.vex_vpinsrd_xmm_xmm_reg_imm8(0, 0, RAX, 3)

    # Broadcast d → YMM4, dmin → XMM1
    emit.vex_vbroadcastss_ymm_xmm(4, 4)
    emit.vex_vbroadcastss_xmm_xmm(1, 1)

    # Expand 16 scale bytes → 16 × int16
    emit.vex_vpmovzxbw_ymm_xmm(0, 0)

    # Mins term: sum(mins[i] * scale[i]) accumulated in XMM6
    emit.vex_vmovdqu_ymm_mem(2, R9, 260)
    emit.vex_vextracti128_xmm_ymm_imm8(15, 2, 1)
    emit.vphaddw_xmm_xmm_xmm(2, 2, 15)

    emit.vex_vextracti128_xmm_ymm_imm8(15, 0, 1)
    emit.vex_vinserti128_ymm_ymm_xmm_imm8(0, 0, 0, 1)

    emit.vex_vpmaddwd_xmm(15, 15, 2)
    emit.vex_vcvtdq2ps_xmm(15, 15)
    emit.vex_vfmadd231ps_xmm(6, 1, 15)

    # Broadcast each scale to its own YMM via VPSHUFB with shuffle tables
    emit.vex_vpshufb_ymm(2, 0, 14)    # YMM2 = scale[0]
    emit.vpshufb_ymm(16, 0, 13)       # YMM16 = scale[1]
    emit.vpshufb_ymm(18, 0, 12)       # YMM18 = scale[2]
    emit.vpshufb_ymm(17, 0, 11)       # YMM17 = scale[3]
    emit.vpshufb_ymm(19, 0, 10)       # YMM19 = scale[4]
    emit.vpshufb_ymm(20, 0, 9)        # YMM20 = scale[5]
    emit.vpshufb_ymm(21, 0, 8)        # YMM21 = scale[6]


# ================================================================
# Epilogue: sum accumulators, convert, advance pointer
# ================================================================

def _emit_epilogue(emit):
    """Emit inner loop epilogue (sum 4 accumulators, convert to float, FMA)."""
    emit.vpaddd_ymm(22, 22, 23)
    emit.vpaddd_ymm(24, 24, 25)
    emit.vpaddd_ymm(22, 22, 24)
    emit.vcvtdq2ps_ymm(0, 22)
    emit.vex_vfmadd231ps_ymm(5, 4, 0)
    emit.add_reg_imm32(R9, 292)  # advance Q8K pointer


def _emit_final_reduction(emit):
    """Emit horizontal sum of acc (YMM5) and acc_m (XMM6), return as uint32."""
    # acc_m (XMM6) horizontal sum
    emit.vex_vmovhlps_xmm(0, 6, 6)
    emit.vex_vaddps_xmm(0, 0, 6)
    emit.vex_vmovshdup_xmm(1, 0)
    # ADDSS XMM1, XMM0
    emit._emit(0xF3)
    emit._rex_sse(r=1, b=0)
    emit._emit(0x0F); emit._emit(0x58)
    emit._modrm(0b11, 1, 0)

    # acc (YMM5) horizontal sum
    emit.vex_vextracti128_xmm_ymm_imm8(0, 5, 1)
    emit.vex_vaddps_xmm(0, 0, 5)
    emit.vex_vmovhlps_xmm(2, 0, 0)
    emit.vex_vaddps_xmm(0, 0, 2)
    emit.vex_vmovshdup_xmm(2, 0)
    # ADDSS XMM0, XMM2
    emit._emit(0xF3)
    emit._rex_sse(r=0, b=2)
    emit._emit(0x0F); emit._emit(0x58)
    emit._modrm(0b11, 0, 2)

    # result = acc + acc_m → ADDSS XMM1, XMM0
    emit._emit(0xF3)
    emit._rex_sse(r=1, b=0)
    emit._emit(0x0F); emit._emit(0x58)
    emit._modrm(0b11, 1, 0)

    # MOVD EAX, XMM1 (return float as uint32)
    emit._emit(0x66)
    emit._rex_sse(r=1, b=RAX)
    emit._emit(0x0F); emit._emit(0x7E)
    emit._modrm(0b11, 1, RAX)

    # Restore callee-saved registers
    emit.mov_reg_reg(RSP, RBP)
    emit.pop(RBP)
    emit.pop(R15)
    emit.pop(R14)
    emit.pop(R13)
    emit.pop(R12)
    emit.pop(RBX)
    emit.ret()


# ================================================================
# Full kernel generator
# ================================================================

def generate_q4k_kernel(ops=None):
    """
    Generate the complete Q4_K dot product kernel.

    Args:
        ops: list of OpQ4K for the inner loop. Defaults to turbo7.

    Returns:
        X86Emitter with the kernel ready to compile.
    """
    if ops is None:
        ops = turbo7_ops()

    emit = X86Emitter()

    # === Function prologue (save callee-saved registers) ===
    emit.push(RBX)
    emit.push(R12)
    emit.push(R13)
    emit.push(R14)
    emit.push(R15)
    emit.push(RBP)
    emit.mov_reg_reg(RBP, RSP)
    emit.sub_reg_imm32(RSP, 32)

    # Arguments: RDI=q4k_data, RSI=q8k_data, RDX=n_blocks
    emit.mov_reg_reg(RCX, RDI)
    emit.mov_reg_reg(R9, RSI)
    emit.mov_reg_reg(R10, RDX)

    # === CALL+POP trick: embed SIMD constants inline ===
    emit._emit(0xE8)  # CALL rel32
    call_patch = len(emit.code)
    for _ in range(4):
        emit._emit(0x00)

    data_start = len(emit.code)

    # Offset 0: signbit mask for negating d_q8 (16 bytes)
    for _ in range(4):
        emit._emit_bytes(struct.pack('<I', 0x80000000))

    # Offset 16: 8 shuffle tables for scale broadcast (256 bytes)
    SHUFFLE_OFF = 16
    for i in range(8):
        lo = (2 * i) & 0xFF
        hi = (2 * i + 1) & 0xFF
        for _ in range(16):
            emit._emit(lo)
            emit._emit(hi)

    after_data = len(emit.code)
    struct.pack_into("<i", emit.code, call_patch, after_data - (call_patch + 4))
    emit.pop(R14)  # R14 = address of embedded data

    # === Preload 8 shuffle tables into YMM14..YMM7 ===
    for i, ymm in enumerate([14, 13, 12, 11, 10, 9, 8, 7]):
        emit.vmovdqu_ymm_mem(ymm, R14, SHUFFLE_OFF + i * 32)

    # === Broadcast mask 0x0F → YMM3 ===
    emit.mov_reg_imm64(RAX, 0x0f0f0f0f)
    emit.vpbroadcastd_ymm(3, RAX)

    # === Initialize accumulators ===
    emit.vex_vxorps_xmm(6, 6, 6)   # XMM6 = 0 (acc_m for mins)
    emit.vex_vxorps_xmm(5, 5, 5)   # XMM5/YMM5 = 0 (acc for sums)

    # Block counter
    emit.mov_reg_imm64(RSI, 0)

    emit.nop_align(32)

    emit.label("loop")
    emit.cmp_reg_reg(RSI, R10)
    emit.jge("done")

    # === Per-block prologue (decode d, dmin, scales) ===
    _emit_prologue(emit)

    # === Inner loop SIMD (from genome) ===
    for op in ops:
        _emit_op(emit, op)

    # === Per-block epilogue (sum accumulators, convert) ===
    _emit_epilogue(emit)

    emit.jmp("loop")
    emit.label("done")

    # === Final horizontal reduction ===
    _emit_final_reduction(emit)

    return emit


# ================================================================
# Compile to callable function
# ================================================================

def compile_q4k_kernel(ops=None):
    """
    Compile the Q4_K kernel to an executable function.

    Args:
        ops: inner loop ops (defaults to turbo7).

    Returns:
        Callable(q4k_ptr: int, q8k_ptr: int, n_blocks: int) -> float
        or None on failure.
    """
    try:
        emit = generate_q4k_kernel(ops)
        func = emit.compile()

        def wrapper(q4k_data_ptr, q8k_data_ptr, n):
            raw = func(q4k_data_ptr, q8k_data_ptr, n)
            return struct.unpack('f', struct.pack('I', raw & 0xFFFFFFFF))[0]

        wrapper._func = func
        wrapper._size = len(emit.code)
        wrapper._n_ops = len(ops) if ops else 34
        return wrapper
    except Exception:
        return None


def compile_turbo7():
    """Compile the turbo7 Q4_K kernel (convenience function)."""
    return compile_q4k_kernel(turbo7_ops())


# ================================================================
# .so generator for LD_PRELOAD into llama.cpp
# ================================================================

def generate_turbo7_so(output_dir="."):
    """
    Generate genesis_turbo7.so for LD_PRELOAD into llama.cpp.

    This creates a shared library that intercepts
    ggml_vec_dot_q4_K_q8_K_generic with our turbo7 kernel.

    Args:
        output_dir: directory for output files.

    Returns:
        Path to the generated .so file, or None on failure.
    """
    import subprocess
    import os

    emit = generate_q4k_kernel(turbo7_ops())
    code_bytes = bytes(emit.code)

    hex_lines = []
    for i in range(0, len(code_bytes), 16):
        chunk = code_bytes[i:i+16]
        hex_lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    hex_array = "\n".join(hex_lines)

    c_code = f"""\
/*
 * genesis_turbo7.c — Q4_K kernel for LD_PRELOAD into llama.cpp.
 * Generated by genesis_kernel.q4k_kernel.generate_turbo7_so().
 * DO NOT EDIT.
 *
 * Turbo7: AVX-512 VNNI in YMM regs + PREFETCHNTA hint.
 * Intercepts ggml_vec_dot_q4_K_q8_K_generic.
 *
 * Usage:
 *   LD_PRELOAD=./genesis_turbo7.so llama-cli -m model.gguf ...
 */
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdlib.h>

static const unsigned char turbo7_code[] = {{
{hex_array}
}};

typedef uint32_t (*turbo7_fn_t)(const void*, const void*, int);
static turbo7_fn_t turbo7_fn = NULL;
static int genesis_initialized = 0;

__attribute__((constructor))
static void init_genesis_turbo7(void) {{
    size_t len = sizeof(turbo7_code);
    size_t page_size = 4096;
    size_t alloc_size = (len + page_size - 1) & ~(page_size - 1);
    void *mem = mmap(NULL, alloc_size,
                     PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {{
        fprintf(stderr, "[genesis] ERROR: mmap failed\\n");
        return;
    }}
    memcpy(mem, turbo7_code, len);
    turbo7_fn = (turbo7_fn_t)mem;
    genesis_initialized = 1;
    if (getenv("GENESIS_VERBOSE"))
        fprintf(stderr, "[genesis] turbo7 loaded: %zu bytes at %p\\n", len, mem);
}}

void ggml_vec_dot_q4_K_q8_K_generic(
    int n, float * __restrict__ s,
    __attribute__((unused)) size_t bs,
    const void * __restrict__ vx,
    __attribute__((unused)) size_t bx,
    const void * __restrict__ vy,
    __attribute__((unused)) size_t by,
    __attribute__((unused)) int nrc
) {{
    if (!genesis_initialized || !turbo7_fn) {{ *s = 0.0f; return; }}
    int n_blocks = n / 256;
    if (n_blocks <= 0) {{ *s = 0.0f; return; }}
    uint32_t raw = turbo7_fn(vx, vy, n_blocks);
    float result;
    memcpy(&result, &raw, sizeof(float));
    *s = result;
}}
"""

    c_path = os.path.join(output_dir, "genesis_turbo7.c")
    so_path = os.path.join(output_dir, "genesis_turbo7.so")

    with open(c_path, "w") as f:
        f.write(c_code)

    cmd = ["gcc", "-shared", "-fPIC", "-O2", "-o", so_path, c_path, "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"gcc failed: {{result.stderr}}")
        return None

    return so_path
