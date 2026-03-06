#!/usr/bin/env python3
"""
Genesis Kernel — x86-64 machine code emitter.

Generates raw x86-64 bytes directly without external assemblers (NASM, GAS).
Supports general-purpose instructions, SSE scalar, and AVX-512 (EVEX-encoded).

Each instruction is emitted as its raw byte encoding. The emitter handles
REX/EVEX prefixes, ModR/M, SIB, label resolution, and memory allocation
with executable permissions for JIT compilation.
"""

import ctypes
import mmap
import struct
from typing import List


# x86-64 general-purpose registers
RAX, RCX, RDX, RBX = 0, 1, 2, 3
RSP, RBP, RSI, RDI = 4, 5, 6, 7
R8, R9, R10, R11 = 8, 9, 10, 11
R12, R13, R14, R15 = 12, 13, 14, 15

# XMM registers (SSE, 128-bit)
XMM0, XMM1, XMM2, XMM3 = 0, 1, 2, 3
XMM4, XMM5, XMM6, XMM7 = 4, 5, 6, 7
XMM8, XMM9, XMM10, XMM11 = 8, 9, 10, 11
XMM12, XMM13, XMM14, XMM15 = 12, 13, 14, 15

# ZMM registers (AVX-512, 512-bit = 16 floats)
# Same numbering as XMM/YMM; the EVEX prefix (L'L=10) selects 512-bit width.
ZMM0, ZMM1, ZMM2, ZMM3 = 0, 1, 2, 3
ZMM4, ZMM5, ZMM6, ZMM7 = 4, 5, 6, 7
ZMM8, ZMM9, ZMM10, ZMM11 = 8, 9, 10, 11
ZMM12, ZMM13, ZMM14, ZMM15 = 12, 13, 14, 15

REG_NAMES = {
    0: "rax", 1: "rcx", 2: "rdx", 3: "rbx",
    4: "rsp", 5: "rbp", 6: "rsi", 7: "rdi",
    8: "r8", 9: "r9", 10: "r10", 11: "r11",
    12: "r12", 13: "r13", 14: "r14", 15: "r15",
}


class X86Emitter:
    """
    Generates x86-64 machine code byte by byte.

    Usage:
        emit = X86Emitter()
        emit.mov_reg_reg(RAX, RDI)    # rax = first argument
        emit.add_reg_reg(RAX, RSI)    # rax += second argument
        emit.ret()                     # return rax
        fn = emit.compile()
        result = fn(3, 4)              # -> 7
    """

    def __init__(self):
        self.code: bytearray = bytearray()
        self.labels: dict[str, int] = {}
        self.fixups: list[tuple[str, int, int]] = []

    def _emit(self, *bytes_: int):
        """Append raw bytes to the code buffer."""
        for b in bytes_:
            self.code.append(b & 0xFF)

    def _emit_bytes(self, data: bytes):
        """Append a byte sequence."""
        self.code.extend(data)

    def _rex(self, w: bool = True, r: int = 0, b: int = 0):
        """Emit REX prefix. W=1 for 64-bit operand size."""
        val = 0x40
        if w:
            val |= 0x08
        if r > 7:
            val |= 0x04
        if b > 7:
            val |= 0x01
        self._emit(val)

    def _modrm(self, mod: int, reg: int, rm: int):
        """Emit ModR/M byte."""
        self._emit((mod << 6) | ((reg & 7) << 3) | (rm & 7))

    # --- General-purpose instructions ---

    def mov_reg_reg(self, dst: int, src: int):
        """MOV dst, src"""
        self._rex(w=True, r=src, b=dst)
        self._emit(0x89)
        self._modrm(0b11, src, dst)

    def mov_reg_imm64(self, dst: int, value: int):
        """MOV dst, imm64"""
        self._rex(w=True, b=dst)
        self._emit(0xB8 + (dst & 7))
        self._emit_bytes(struct.pack("<q", value))

    def mov_reg_mem(self, dst: int, base: int, offset: int = 0):
        """MOV dst, [base + offset]"""
        self._rex(w=True, r=dst, b=base)
        self._emit(0x8B)
        if offset == 0 and (base & 7) != 5:
            self._modrm(0b00, dst, base)
            if (base & 7) == 4:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, dst, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, dst, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def mov_mem_reg(self, base: int, src: int, offset: int = 0):
        """MOV [base + offset], src"""
        self._rex(w=True, r=src, b=base)
        self._emit(0x89)
        if offset == 0 and (base & 7) != 5:
            self._modrm(0b00, src, base)
            if (base & 7) == 4:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, src, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, src, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def add_reg_reg(self, dst: int, src: int):
        """ADD dst, src"""
        self._rex(w=True, r=src, b=dst)
        self._emit(0x01)
        self._modrm(0b11, src, dst)

    def add_reg_imm32(self, dst: int, value: int):
        """ADD dst, imm32"""
        self._rex(w=True, b=dst)
        if dst == RAX:
            self._emit(0x05)
        else:
            self._emit(0x81)
            self._modrm(0b11, 0, dst)
        self._emit_bytes(struct.pack("<i", value))

    def sub_reg_reg(self, dst: int, src: int):
        """SUB dst, src"""
        self._rex(w=True, r=src, b=dst)
        self._emit(0x29)
        self._modrm(0b11, src, dst)

    def sub_reg_imm32(self, dst: int, value: int):
        """SUB dst, imm32"""
        self._rex(w=True, b=dst)
        if dst == RAX:
            self._emit(0x2D)
        else:
            self._emit(0x83)
            self._modrm(0b11, 5, dst)
        self._emit_bytes(struct.pack("<i", value) if value > 127 or value < -128 else struct.pack("<b", value))

    def imul_reg_reg(self, dst: int, src: int):
        """IMUL dst, src"""
        self._rex(w=True, r=dst, b=src)
        self._emit(0x0F, 0xAF)
        self._modrm(0b11, dst, src)

    def inc_reg(self, reg: int):
        """INC reg"""
        self._rex(w=True, b=reg)
        self._emit(0xFF)
        self._modrm(0b11, 0, reg)

    def dec_reg(self, reg: int):
        """DEC reg"""
        self._rex(w=True, b=reg)
        self._emit(0xFF)
        self._modrm(0b11, 1, reg)

    def shl_reg_imm8(self, reg: int, count: int):
        """SHL reg, imm8"""
        self._rex(w=True, b=reg)
        self._emit(0xC1)
        self._modrm(0b11, 4, reg)
        self._emit(count & 0xFF)

    def shr_reg_imm8(self, reg: int, count: int):
        """SHR reg, imm8"""
        self._rex(w=True, b=reg)
        self._emit(0xC1)
        self._modrm(0b11, 5, reg)
        self._emit(count & 0xFF)

    def and_reg_imm32(self, reg: int, value: int):
        """AND reg, imm32"""
        self._rex(w=True, b=reg)
        if reg == RAX:
            self._emit(0x25)
        else:
            self._emit(0x81)
            self._modrm(0b11, 4, reg)
        self._emit_bytes(struct.pack("<i", value))

    def cmp_reg_reg(self, a: int, b: int):
        """CMP a, b"""
        self._rex(w=True, r=b, b=a)
        self._emit(0x39)
        self._modrm(0b11, b, a)

    def cmp_reg_imm32(self, reg: int, value: int):
        """CMP reg, imm32"""
        self._rex(w=True, b=reg)
        if reg == RAX:
            self._emit(0x3D)
            self._emit_bytes(struct.pack("<i", value))
        elif -128 <= value <= 127:
            self._emit(0x83)
            self._modrm(0b11, 7, reg)
            self._emit(value & 0xFF)
        else:
            self._emit(0x81)
            self._modrm(0b11, 7, reg)
            self._emit_bytes(struct.pack("<i", value))

    # --- Control flow ---

    def ret(self):
        """RET"""
        self._emit(0xC3)

    def label(self, name: str):
        """Define a label at the current position."""
        self.labels[name] = len(self.code)

    def jmp(self, label_name: str):
        """JMP label"""
        self._emit(0xE9)
        self.fixups.append((label_name, len(self.code), 4))
        self._emit_bytes(b"\x00\x00\x00\x00")

    def jl(self, label_name: str):
        """JL label (signed less-than)"""
        self._emit(0x0F, 0x8C)
        self.fixups.append((label_name, len(self.code), 4))
        self._emit_bytes(b"\x00\x00\x00\x00")

    def jge(self, label_name: str):
        """JGE label (signed greater-or-equal)"""
        self._emit(0x0F, 0x8D)
        self.fixups.append((label_name, len(self.code), 4))
        self._emit_bytes(b"\x00\x00\x00\x00")

    def jle(self, label_name: str):
        """JLE label (signed less-or-equal)"""
        self._emit(0x0F, 0x8E)
        self.fixups.append((label_name, len(self.code), 4))
        self._emit_bytes(b"\x00\x00\x00\x00")

    def jne(self, label_name: str):
        """JNE label (not equal)"""
        self._emit(0x0F, 0x85)
        self.fixups.append((label_name, len(self.code), 4))
        self._emit_bytes(b"\x00\x00\x00\x00")

    def je(self, label_name: str):
        """JE label (equal)"""
        self._emit(0x0F, 0x84)
        self.fixups.append((label_name, len(self.code), 4))
        self._emit_bytes(b"\x00\x00\x00\x00")

    # --- Push/Pop ---

    def push(self, reg: int):
        """PUSH reg"""
        if reg > 7:
            self._emit(0x41)
        self._emit(0x50 + (reg & 7))

    def pop(self, reg: int):
        """POP reg"""
        if reg > 7:
            self._emit(0x41)
        self._emit(0x58 + (reg & 7))

    # --- SSE scalar instructions (operate on 1 float32) ---

    def _rex_sse(self, r: int = 0, b: int = 0, x: int = 0):
        """Emit REX prefix for SSE instructions if needed (W=0)."""
        val = 0x40
        if r > 7:
            val |= 0x04
        if x > 7:
            val |= 0x02
        if b > 7:
            val |= 0x01
        if val != 0x40:
            self._emit(val)

    def xorps_xmm_xmm(self, dst: int, src: int):
        """XORPS dst, src — typically used to zero a register."""
        self._rex_sse(r=dst, b=src)
        self._emit(0x0F, 0x57)
        self._modrm(0b11, dst, src)

    def movss_xmm_xmm(self, dst: int, src: int):
        """MOVSS dst, src — copy low float."""
        self._emit(0xF3)
        self._rex_sse(r=dst, b=src)
        self._emit(0x0F, 0x10)
        self._modrm(0b11, dst, src)

    def addss_xmm_xmm(self, dst: int, src: int):
        """ADDSS dst, src — scalar float add."""
        self._emit(0xF3)
        self._rex_sse(r=dst, b=src)
        self._emit(0x0F, 0x58)
        self._modrm(0b11, dst, src)

    def mulss_xmm_xmm(self, dst: int, src: int):
        """MULSS dst, src — scalar float multiply."""
        self._emit(0xF3)
        self._rex_sse(r=dst, b=src)
        self._emit(0x0F, 0x59)
        self._modrm(0b11, dst, src)

    def movss_xmm_mem(self, xmm: int, base: int, offset: int = 0):
        """MOVSS xmm, [base + offset] — load 1 float from memory."""
        self._emit(0xF3)
        self._rex_sse(r=xmm, b=base)
        self._emit(0x0F, 0x10)
        if offset == 0 and (base & 7) != 5:
            self._modrm(0b00, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def movss_mem_xmm(self, base: int, xmm: int, offset: int = 0):
        """MOVSS [base + offset], xmm — store 1 float to memory."""
        self._emit(0xF3)
        self._rex_sse(r=xmm, b=base)
        self._emit(0x0F, 0x11)
        if offset == 0 and (base & 7) != 5:
            self._modrm(0b00, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def movss_xmm_sib(self, xmm: int, base: int, index: int, scale: int, offset: int = 0):
        """MOVSS xmm, [base + index*scale + offset]"""
        scale_bits = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}[scale]
        self._emit(0xF3)
        self._rex_sse(r=xmm, b=base, x=index)
        self._emit(0x0F, 0x10)
        if offset == 0 and (base & 7) != 5:
            self._modrm(0b00, xmm, 0b100)
            self._emit((scale_bits << 6) | ((index & 7) << 3) | (base & 7))
        elif -128 <= offset <= 127:
            self._modrm(0b01, xmm, 0b100)
            self._emit((scale_bits << 6) | ((index & 7) << 3) | (base & 7))
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, xmm, 0b100)
            self._emit((scale_bits << 6) | ((index & 7) << 3) | (base & 7))
            self._emit_bytes(struct.pack("<i", offset))

    def movss_sib_xmm(self, base: int, index: int, scale: int, xmm: int, offset: int = 0):
        """MOVSS [base + index*scale + offset], xmm"""
        scale_bits = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}[scale]
        self._emit(0xF3)
        self._rex_sse(r=xmm, b=base, x=index)
        self._emit(0x0F, 0x11)
        if offset == 0 and (base & 7) != 5:
            self._modrm(0b00, xmm, 0b100)
            self._emit((scale_bits << 6) | ((index & 7) << 3) | (base & 7))
        elif -128 <= offset <= 127:
            self._modrm(0b01, xmm, 0b100)
            self._emit((scale_bits << 6) | ((index & 7) << 3) | (base & 7))
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, xmm, 0b100)
            self._emit((scale_bits << 6) | ((index & 7) << 3) | (base & 7))
            self._emit_bytes(struct.pack("<i", offset))

    def addss_xmm_mem(self, xmm: int, base: int, offset: int = 0):
        """ADDSS xmm, [base + offset]"""
        self._emit(0xF3)
        self._rex_sse(r=xmm, b=base)
        self._emit(0x0F, 0x58)
        if offset == 0 and (base & 7) != 5:
            self._modrm(0b00, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def mulss_xmm_mem(self, xmm: int, base: int, offset: int = 0):
        """MULSS xmm, [base + offset]"""
        self._emit(0xF3)
        self._rex_sse(r=xmm, b=base)
        self._emit(0x0F, 0x59)
        if offset == 0 and (base & 7) != 5:
            self._modrm(0b00, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, xmm, base)
            if (base & 7) == 4:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    # --- AVX-512 instructions (EVEX-encoded, operate on 16 float32s) ---

    def _evex(self, mm: int, pp: int, w: int, vvvv: int,
              r: int, x: int, b: int, rr: int = -1,
              z: int = 0, ll: int = 2, aaa: int = 0,
              reg_reg: bool = False):
        """
        Emit 4-byte EVEX prefix for AVX-512 instructions.

        mm: opcode map (1=0F, 2=0F38, 3=0F3A)
        pp: simulated prefix (0=none, 1=66, 2=F3, 3=F2)
        w: operand size (0=32-bit float, 1=64-bit)
        vvvv: extra source register (0-31, inverted internally)
        r: destination register (0-31)
        x: index register for SIB (memory mode) or bit 4 of rm (reg-reg mode)
        b: base/rm register (0-31)
        reg_reg: if True, X encodes bit 4 of rm (for ZMM16-31 as operand)
        """
        self._emit(0x62)

        R = 0 if (r & 8) else 1
        B = 0 if (b & 8) else 1
        if reg_reg:
            X = 0 if (b & 16) else 1
        else:
            X = 0 if (x & 8) else 1
        if rr == -1:
            Rp = 0 if (r & 16) else 1
        else:
            Rp = 0 if rr else 1
        byte1 = (R << 7) | (X << 6) | (B << 5) | (Rp << 4) | (mm & 0x03)
        self._emit(byte1)

        vvvv_inv = (~vvvv) & 0x0F
        byte2 = (w << 7) | (vvvv_inv << 3) | (1 << 2) | (pp & 0x03)
        self._emit(byte2)

        Vp = 0 if (vvvv & 16) else 1
        byte3 = (z << 7) | ((ll & 0x03) << 5) | (0 << 4) | (Vp << 3) | (aaa & 0x07)
        self._emit(byte3)

    def _evex_mem_modrm_sib(self, reg: int, base: int, offset: int = 0):
        """Emit ModRM (+SIB) for EVEX memory access. Disp compressed by 64 (ZMM width)."""
        needs_sib = (base & 7) == 4
        rbp_base = (base & 7) == 5

        if offset == 0 and not rbp_base:
            self._modrm(0b00, reg, base)
            if needs_sib:
                self._emit(0x24)
        elif offset != 0 and (offset % 64 == 0) and (-128 <= offset // 64 <= 127):
            self._modrm(0b01, reg, base)
            if needs_sib:
                self._emit(0x24)
            disp8 = offset // 64
            self._emit(disp8 & 0xFF)
        elif -128 <= offset <= 127 and offset % 64 == 0:
            self._modrm(0b01, reg, base)
            if needs_sib:
                self._emit(0x24)
            self._emit((offset // 64) & 0xFF)
        else:
            self._modrm(0b10, reg, base)
            if needs_sib:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def _evex_mem_modrm_sib_ss(self, reg: int, base: int, offset: int = 0):
        """Emit ModRM for EVEX memory access with scalar (4-byte) disp compression."""
        needs_sib = (base & 7) == 4
        rbp_base = (base & 7) == 5

        if offset == 0 and not rbp_base:
            self._modrm(0b00, reg, base)
            if needs_sib:
                self._emit(0x24)
        elif offset != 0 and (offset % 4 == 0) and (-128 <= offset // 4 <= 127):
            self._modrm(0b01, reg, base)
            if needs_sib:
                self._emit(0x24)
            self._emit((offset // 4) & 0xFF)
        else:
            self._modrm(0b10, reg, base)
            if needs_sib:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def vxorps_zmm_zmm_zmm(self, dst: int, src1: int, src2: int):
        """VXORPS zmm, zmm, zmm — 512-bit XOR (used to zero registers)."""
        self._evex(mm=1, pp=0, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=2, reg_reg=True)
        self._emit(0x57)
        self._modrm(0b11, dst, src2)

    def vmovups_zmm_mem(self, zmm: int, base: int, offset: int = 0):
        """VMOVUPS zmm, [base + offset] — load 16 floats (unaligned)."""
        self._evex(mm=1, pp=0, w=0, vvvv=0, r=zmm, x=0, b=base, ll=2)
        self._emit(0x10)
        self._evex_mem_modrm_sib(zmm, base, offset)

    def vmovups_mem_zmm(self, base: int, zmm: int, offset: int = 0):
        """VMOVUPS [base + offset], zmm — store 16 floats (unaligned)."""
        self._evex(mm=1, pp=0, w=0, vvvv=0, r=zmm, x=0, b=base, ll=2)
        self._emit(0x11)
        self._evex_mem_modrm_sib(zmm, base, offset)

    def vbroadcastss_zmm_mem(self, zmm: int, base: int, offset: int = 0):
        """VBROADCASTSS zmm, [base + offset] — broadcast 1 float to all 16 lanes."""
        self._evex(mm=2, pp=1, w=0, vvvv=0, r=zmm, x=0, b=base, ll=2)
        self._emit(0x18)
        self._evex_mem_modrm_sib_ss(zmm, base, offset)

    def vfmadd231ps_zmm_zmm_zmm(self, dst: int, src1: int, src2: int):
        """VFMADD231PS zmm, zmm, zmm — dst = src1 * src2 + dst (16 floats, fused)."""
        self._evex(mm=2, pp=1, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=2, reg_reg=True)
        self._emit(0xB8)
        self._modrm(0b11, dst, src2)

    def vfmadd231ps_zmm_zmm_mem(self, dst: int, src1: int, base: int, offset: int = 0):
        """VFMADD231PS zmm, zmm, [base + offset]"""
        self._evex(mm=2, pp=1, w=0, vvvv=src1, r=dst, x=0, b=base, ll=2)
        self._emit(0xB8)
        self._evex_mem_modrm_sib(dst, base, offset)

    def prefetchT0_mem(self, base: int, offset: int = 0):
        """PREFETCHT0 [base + offset] — prefetch data to L1 cache."""
        if base > 7:
            self._emit(0x41)
        self._emit(0x0F, 0x18)
        needs_sib = (base & 7) == 4
        rbp_base = (base & 7) == 5
        if offset == 0 and not rbp_base:
            self._modrm(0b00, 1, base)
            if needs_sib:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, 1, base)
            if needs_sib:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, 1, base)
            if needs_sib:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    # --- AVX-512 NF4 kernel instructions ---

    def vmulps_zmm_zmm_zmm(self, dst: int, src1: int, src2: int):
        """VMULPS zmm, zmm, zmm — multiply 16 floats."""
        self._evex(mm=1, pp=0, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=2, reg_reg=True)
        self._emit(0x59)
        self._modrm(0b11, dst, src2)

    def vaddps_zmm_zmm_zmm(self, dst: int, src1: int, src2: int):
        """VADDPS zmm, zmm, zmm — add 16 floats."""
        self._evex(mm=1, pp=0, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=2, reg_reg=True)
        self._emit(0x58)
        self._modrm(0b11, dst, src2)

    def vpermps_zmm_zmm_zmm(self, dst: int, idx: int, src: int):
        """VPERMPS zmm, zmm, zmm — permute: dst[i] = src[idx[i] & 0xF]. 16-way table lookup."""
        self._evex(mm=2, pp=1, w=0, vvvv=idx, r=dst, x=0, b=src, ll=2, reg_reg=True)
        self._emit(0x16)
        self._modrm(0b11, dst, src)

    def vpandd_zmm_zmm_zmm(self, dst: int, src1: int, src2: int):
        """VPANDD zmm, zmm, zmm — bitwise AND of 16 dwords."""
        self._evex(mm=1, pp=1, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=2, reg_reg=True)
        self._emit(0xDB)
        self._modrm(0b11, dst, src2)

    def vpsrld_zmm_zmm_imm8(self, dst: int, src: int, count: int):
        """VPSRLD zmm, zmm, imm8 — logical right shift each of 16 dwords."""
        self._evex(mm=1, pp=1, w=0, vvvv=dst, r=0, x=0, b=src, ll=2, reg_reg=True)
        self._emit(0x72)
        self._modrm(0b11, 2, src)
        self._emit(count & 0xFF)

    def vpmovzxbd_zmm_mem(self, zmm: int, base: int, offset: int = 0):
        """VPMOVZXBD zmm, [base + offset] — zero-extend 16 bytes to 16 dwords."""
        self._evex(mm=2, pp=1, w=0, vvvv=0, r=zmm, x=0, b=base, ll=2)
        self._emit(0x31)
        needs_sib = (base & 7) == 4
        rbp_base = (base & 7) == 5
        if offset == 0 and not rbp_base:
            self._modrm(0b00, zmm, base)
            if needs_sib:
                self._emit(0x24)
        elif offset != 0 and (offset % 16 == 0) and (-128 <= offset // 16 <= 127):
            self._modrm(0b01, zmm, base)
            if needs_sib:
                self._emit(0x24)
            self._emit((offset // 16) & 0xFF)
        elif -128 <= offset <= 127:
            self._modrm(0b01, zmm, base)
            if needs_sib:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, zmm, base)
            if needs_sib:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def vpbroadcastd_zmm_mem(self, zmm: int, base: int, offset: int = 0):
        """VPBROADCASTD zmm, [base + offset] — broadcast 1 dword to 16 lanes."""
        self._evex(mm=2, pp=1, w=0, vvvv=0, r=zmm, x=0, b=base, ll=2)
        self._emit(0x58)
        self._evex_mem_modrm_sib_ss(zmm, base, offset)

    def vmovdqu32_zmm_mem(self, zmm: int, base: int, offset: int = 0):
        """VMOVDQU32 zmm, [base + offset] — load 16 dwords (unaligned)."""
        self._evex(mm=1, pp=2, w=0, vvvv=0, r=zmm, x=0, b=base, ll=2)
        self._emit(0x6F)
        self._evex_mem_modrm_sib(zmm, base, offset)

    def vmovdqu32_mem_zmm(self, base: int, zmm: int, offset: int = 0):
        """VMOVDQU32 [base + offset], zmm — store 16 dwords (unaligned)."""
        self._evex(mm=1, pp=2, w=0, vvvv=0, r=zmm, x=0, b=base, ll=2)
        self._emit(0x7F)
        self._evex_mem_modrm_sib(zmm, base, offset)

    # --- Label resolution and compilation ---

    def _resolve_fixups(self):
        """Resolve pending jumps (calculate relative offsets to labels)."""
        for label_name, offset, size in self.fixups:
            if label_name not in self.labels:
                raise ValueError(f"Undefined label: {label_name}")
            target = self.labels[label_name]
            rel = target - (offset + size)
            if size == 4:
                struct.pack_into("<i", self.code, offset, rel)
            elif size == 1:
                if not (-128 <= rel <= 127):
                    raise ValueError(f"Jump too far for rel8: {rel}")
                struct.pack_into("<b", self.code, offset, rel)

    def compile(self) -> ctypes.CFUNCTYPE:
        """
        Compile the emitted code into an executable function.

        Allocates executable memory via mmap, resolves labels, and returns
        a callable following the System V AMD64 ABI:
          args in RDI, RSI, RDX, RCX, R8, R9; return in RAX.
        """
        self._resolve_fixups()

        code_bytes = bytes(self.code)
        size = len(code_bytes)
        page_size = mmap.PAGESIZE
        alloc_size = ((size + page_size - 1) // page_size) * page_size

        buf = mmap.mmap(
            -1, alloc_size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        )
        buf.write(code_bytes)

        buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))

        func_type = ctypes.CFUNCTYPE(
            ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
        )
        raw_fn = func_type(buf_addr)
        raw_fn._buf = buf

        def wrapper(*args):
            padded = list(args) + [0] * (4 - len(args))
            return raw_fn(*padded[:4])

        wrapper._buf = buf
        wrapper._raw = raw_fn
        return wrapper

    def hexdump(self) -> str:
        """Return hex representation of the generated code."""
        hex_str = " ".join(f"{b:02X}" for b in self.code)
        return f"[{len(self.code)} bytes] {hex_str}"

    def size(self) -> int:
        """Size of generated code in bytes."""
        return len(self.code)

    # ================================================================
    # EVEX DISP_N HELPER — Generic displacement compression
    # ================================================================

    def _evex_mem_modrm_sib_disp_n(self, reg: int, base: int, offset: int, disp_n: int):
        """
        Emit ModRM (+SIB if needed) for EVEX memory access with generic
        displacement compression. disp_n is the compression factor
        (1, 2, 4, 8, 16, 32, 64) depending on the instruction's operand size.
        """
        needs_sib = (base & 7) == 4
        rbp_base = (base & 7) == 5

        if offset == 0 and not rbp_base:
            self._modrm(0b00, reg, base)
            if needs_sib:
                self._emit(0x24)
        elif offset != 0 and (offset % disp_n == 0) and (-128 <= offset // disp_n <= 127):
            self._modrm(0b01, reg, base)
            if needs_sib:
                self._emit(0x24)
            self._emit((offset // disp_n) & 0xFF)
        else:
            self._modrm(0b10, reg, base)
            if needs_sib:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    # ================================================================
    # VEX 3-BYTE PREFIX INFRASTRUCTURE
    # ================================================================
    # VEX encoding is 2-3 bytes more compact than EVEX (4 bytes).
    # Format: C4 [RXBmmmmm] [WvvvvLpp] opcode modrm [disp]

    def _vex3(self, mmmmm: int, pp: int, w: int, vvvv: int,
              r: int, b: int, L: int):
        """Emit VEX 3-byte prefix (C4) for reg-reg operations."""
        R = 0 if (r >= 8) else 1
        X = 1  # No index register in reg-reg
        B = 0 if (b >= 8) else 1
        byte1 = (R << 7) | (X << 6) | (B << 5) | (mmmmm & 0x1F)
        vvvv_inv = (~vvvv) & 0x0F
        byte2 = (w << 7) | (vvvv_inv << 3) | (L << 2) | (pp & 0x03)
        self._emit(0xC4, byte1, byte2)

    def _vex3_mem(self, mmmmm: int, pp: int, w: int, vvvv: int,
                  r: int, base: int, L: int):
        """Emit VEX 3-byte prefix for memory operations."""
        R = 0 if (r >= 8) else 1
        X = 1  # No index
        B = 0 if (base >= 8) else 1
        byte1 = (R << 7) | (X << 6) | (B << 5) | (mmmmm & 0x1F)
        vvvv_inv = (~vvvv) & 0x0F
        byte2 = (w << 7) | (vvvv_inv << 3) | (L << 2) | (pp & 0x03)
        self._emit(0xC4, byte1, byte2)

    def _vex_mem_modrm(self, reg: int, base: int, offset: int):
        """Emit ModRM + SIB + displacement for VEX memory (no compression)."""
        needs_sib = (base & 7) == 4
        rbp_base = (base & 7) == 5
        if offset == 0 and not rbp_base:
            self._modrm(0b00, reg, base)
            if needs_sib:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, reg, base)
            if needs_sib:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, reg, base)
            if needs_sib:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    # ================================================================
    # VEX-ENCODED YMM INSTRUCTIONS (256-bit)
    # ================================================================

    def vex_vmovdqu_ymm_mem(self, ymm: int, base: int, offset: int = 0):
        """VMOVDQU ymm, [base+offset] — VEX.256.F3.0F 6F /r"""
        self._vex3_mem(0x01, 0x02, 0, 0, ymm, base, 1)
        self._emit(0x6F)
        self._vex_mem_modrm(ymm & 7, base, offset)

    def vex_vpshufb_ymm(self, dst: int, src1: int, src2: int):
        """VPSHUFB ymm,ymm,ymm — VEX.256.66.0F38 00 /r"""
        self._vex3(0x02, 0x01, 0, src1, dst, src2, 1)
        self._emit(0x00)
        self._modrm(0b11, dst & 7, src2 & 7)

    def vex_vpand_ymm(self, dst: int, src1: int, src2: int):
        """VPAND ymm,ymm,ymm — VEX.256.66.0F DB /r"""
        self._vex3(0x01, 0x01, 0, src1, dst, src2, 1)
        self._emit(0xDB)
        self._modrm(0b11, dst & 7, src2 & 7)

    def vex_vpsrlw_ymm_imm8(self, dst: int, src: int, count: int):
        """VPSRLW ymm,ymm,imm8 — VEX.256.66.0F 71 /2 ib"""
        self._vex3(0x01, 0x01, 0, dst, 2, src, 1)
        self._emit(0x71)
        self._modrm(0b11, 2, src & 7)
        self._emit(count & 0xFF)

    def vex_vpmaddubsw_ymm_mem(self, dst: int, src1: int, base: int, offset: int = 0):
        """VPMADDUBSW ymm,ymm,[base+offset] — VEX.256.66.0F38 04 /r (mem)."""
        self._vex3_mem(0x02, 0x01, 0, src1, dst, base, 1)
        self._emit(0x04)
        self._vex_mem_modrm(dst & 7, base, offset)

    def vex_vpmovzxbw_ymm_xmm(self, dst: int, src: int):
        """VPMOVZXBW ymm, xmm — VEX.256.66.0F38 30 /r"""
        self._vex3(0x02, 0x01, 0, 0, dst, src, 1)
        self._emit(0x30)
        self._modrm(0b11, dst & 7, src & 7)

    def vex_vextracti128_xmm_ymm_imm8(self, dst: int, src: int, imm8: int):
        """VEXTRACTI128 xmm, ymm, imm8 — VEX.256.66.0F3A 39 /r ib"""
        self._vex3(0x03, 0x01, 0, 0, src, dst, 1)
        self._emit(0x39)
        self._modrm(0b11, src & 7, dst & 7)
        self._emit(imm8 & 0xFF)

    def vex_vinserti128_ymm_ymm_xmm_imm8(self, dst: int, src1: int, src2: int, imm8: int):
        """VINSERTI128 ymm, ymm, xmm, imm8 — VEX.256.66.0F3A 38 /r ib"""
        self._vex3(0x03, 0x01, 0, src1, dst, src2, 1)
        self._emit(0x38)
        self._modrm(0b11, dst & 7, src2 & 7)
        self._emit(imm8 & 0xFF)

    def vex_vfmadd231ps_ymm(self, dst: int, src1: int, src2: int):
        """VFMADD231PS ymm,ymm,ymm — VEX.256.66.0F38.W0 B8 /r"""
        self._vex3(0x02, 0x01, 0, src1, dst, src2, 1)
        self._emit(0xB8)
        self._modrm(0b11, dst & 7, src2 & 7)

    def vex_vxorps_ymm(self, dst: int, src1: int, src2: int):
        """VXORPS ymm,ymm,ymm — VEX.256.0F 57 /r"""
        self._vex3(0x01, 0x00, 0, src1, dst, src2, 1)
        self._emit(0x57)
        self._modrm(0b11, dst & 7, src2 & 7)

    # ================================================================
    # VEX-ENCODED XMM INSTRUCTIONS (128-bit)
    # ================================================================

    def vex_vmovss_xmm_mem(self, xmm: int, base: int, offset: int = 0):
        """VMOVSS xmm, [mem] — VEX.LIG.F3.0F.WIG 10 /r"""
        self._vex3_mem(1, 2, 0, 0, xmm, base, 0)
        self._emit(0x10)
        self._vex_mem_modrm(xmm, base, offset)

    def vex_vmovd_xmm_reg(self, xmm: int, gpr: int):
        """VMOVD xmm, r32 — VEX.128.66.0F.W0 6E /r"""
        self._vex3(1, 1, 0, 0, xmm, gpr, 0)
        self._emit(0x6E)
        self._modrm(0b11, xmm & 7, gpr & 7)

    def vex_vcvtph2ps_xmm(self, dst: int, src: int):
        """VCVTPH2PS xmm, xmm — VEX.128.66.0F38.W0 13 /r (F16C)."""
        self._vex3(2, 1, 0, 0, dst, src, 0)
        self._emit(0x13)
        self._modrm(0b11, dst & 7, src & 7)

    def vex_vbroadcastss_ymm_xmm(self, dst: int, src: int):
        """VBROADCASTSS ymm,xmm — VEX.256.66.0F38 18 /r"""
        self._vex3(0x02, 0x01, 0, 0, dst, src, 1)
        self._emit(0x18)
        self._modrm(0b11, dst & 7, src & 7)

    def vex_vbroadcastss_xmm_xmm(self, dst: int, src: int):
        """VBROADCASTSS xmm,xmm — VEX.128.66.0F38 18 /r"""
        self._vex3(0x02, 0x01, 0, 0, dst, src, 0)
        self._emit(0x18)
        self._modrm(0b11, dst & 7, src & 7)

    def vex_vpinsrd_xmm_xmm_reg_imm8(self, dst: int, src1: int, gpr: int, imm8: int):
        """VPINSRD xmm, xmm, r32, imm8 — VEX.128.66.0F3A.W0 22 /r ib"""
        self._vex3(3, 1, 0, src1, dst, gpr, 0)
        self._emit(0x22)
        self._modrm(0b11, dst & 7, gpr & 7)
        self._emit(imm8 & 0xFF)

    def vex_vxorps_xmm_xmm_mem(self, dst: int, src1: int, base: int, offset: int = 0):
        """VXORPS xmm, xmm, [mem] — VEX.128.0F.WIG 57 /r"""
        self._vex3_mem(1, 0, 0, src1, dst, base, 0)
        self._emit(0x57)
        self._vex_mem_modrm(dst, base, offset)

    def vex_vpmaddwd_xmm(self, dst: int, src1: int, src2: int):
        """VPMADDWD xmm,xmm,xmm — VEX.128.66.0F F5 /r"""
        self._vex3(0x01, 0x01, 0, src1, dst, src2, 0)
        self._emit(0xF5)
        self._modrm(0b11, dst & 7, src2 & 7)

    def vex_vcvtdq2ps_xmm(self, dst: int, src: int):
        """VCVTDQ2PS xmm,xmm — VEX.128.0F 5B /r"""
        self._vex3(0x01, 0x00, 0, 0, dst, src, 0)
        self._emit(0x5B)
        self._modrm(0b11, dst & 7, src & 7)

    def vex_vfmadd231ps_xmm(self, dst: int, src1: int, src2: int):
        """VFMADD231PS xmm,xmm,xmm — VEX.128.66.0F38.W0 B8 /r"""
        self._vex3(0x02, 0x01, 0, src1, dst, src2, 0)
        self._emit(0xB8)
        self._modrm(0b11, dst & 7, src2 & 7)

    def vex_vxorps_xmm(self, dst: int, src1: int, src2: int):
        """VXORPS xmm,xmm,xmm — VEX.128.0F 57 /r"""
        self._vex3(0x01, 0x00, 0, src1, dst, src2, 0)
        self._emit(0x57)
        self._modrm(0b11, dst & 7, src2 & 7)

    def vex_vaddps_xmm(self, dst: int, src1: int, src2: int):
        """VADDPS xmm,xmm,xmm — VEX.128.0F 58 /r"""
        self._vex3(0x01, 0x00, 0, src1, dst, src2, 0)
        self._emit(0x58)
        self._modrm(0b11, dst & 7, src2 & 7)

    def vex_vmovhlps_xmm(self, dst: int, src1: int, src2: int):
        """VMOVHLPS xmm,xmm,xmm — VEX.128.0F 12 /r"""
        self._vex3(0x01, 0x00, 0, src1, dst, src2, 0)
        self._emit(0x12)
        self._modrm(0b11, dst & 7, src2 & 7)

    def vex_vmovshdup_xmm(self, dst: int, src: int):
        """VMOVSHDUP xmm,xmm — VEX.128.F3.0F 16 /r"""
        self._vex3(0x01, 0x02, 0, 0, dst, src, 0)
        self._emit(0x16)
        self._modrm(0b11, dst & 7, src & 7)

    def vphaddw_xmm_xmm_xmm(self, dst: int, src1: int, src2: int):
        """VPHADDW xmm, xmm, xmm — VEX.128.66.0F38.W0 01 /r (no EVEX)."""
        rxb = 0xE0
        if dst >= 8:
            rxb &= ~0x80
        if src2 >= 8:
            rxb &= ~0x20
        byte2 = rxb | 0x02  # mmmmm=00010 (0F38)
        vvvv = (~src1 & 0xF) << 3
        byte3 = vvvv | 0x01  # W=0, L=0, pp=01 (66)
        self._emit(0xC4, byte2, byte3)
        self._emit(0x01)
        self._modrm(0b11, dst & 7, src2 & 7)

    # ================================================================
    # EVEX YMM INSTRUCTIONS (256-bit, supports YMM0-31)
    # ================================================================

    def vxorps_ymm_ymm_ymm(self, dst: int, src1: int, src2: int):
        """VXORPS ymm, ymm, ymm — EVEX.256. Supports YMM0-31."""
        self._evex(mm=1, pp=0, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=1, reg_reg=True)
        self._emit(0x57)
        self._modrm(0b11, dst, src2)

    def vpshufb_ymm(self, dst: int, src1: int, src2: int):
        """VPSHUFB ymm, ymm, ymm — EVEX.256. Supports YMM0-31."""
        self._evex(mm=2, pp=1, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=1, reg_reg=True)
        self._emit(0x00)
        self._modrm(0b11, dst, src2)

    def vmovdqu_ymm_mem(self, ymm: int, base: int, offset: int = 0):
        """VMOVDQU32 ymm, [base+offset] — EVEX.256. Supports YMM0-31. disp_n=32."""
        self._evex(mm=1, pp=2, w=0, vvvv=0, r=ymm, x=0, b=base, ll=1)
        self._emit(0x6F)
        self._evex_mem_modrm_sib_disp_n(ymm, base, offset, 32)

    def vmovdqa_ymm_mem(self, ymm: int, base: int, offset: int = 0):
        """VMOVDQA32 ymm, [base+offset] — EVEX.256. Supports YMM0-31. disp_n=32."""
        self._evex(mm=1, pp=1, w=0, vvvv=0, r=ymm, x=0, b=base, ll=1)
        self._emit(0x6F)
        self._evex_mem_modrm_sib_disp_n(ymm, base, offset, 32)

    def vpbroadcastd_ymm(self, dst: int, src: int):
        """VPBROADCASTD ymm, reg32 — EVEX.256. Supports YMM0-31."""
        self._evex(mm=2, pp=1, w=0, vvvv=0, r=dst, x=0, b=src, ll=1, reg_reg=True)
        self._emit(0x7C)
        self._modrm(0b11, dst, src)

    def vpaddd_ymm(self, dst: int, src1: int, src2: int):
        """VPADDD ymm, ymm, ymm — EVEX.256. Supports YMM0-31."""
        self._evex(mm=1, pp=1, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=1, reg_reg=True)
        self._emit(0xFE)
        self._modrm(0b11, dst, src2)

    def vcvtdq2ps_ymm(self, dst: int, src: int):
        """VCVTDQ2PS ymm, ymm — EVEX.256. Supports YMM0-31."""
        self._evex(mm=1, pp=0, w=0, vvvv=0, r=dst, x=0, b=src, ll=1, reg_reg=True)
        self._emit(0x5B)
        self._modrm(0b11, dst, src)

    # ================================================================
    # AVX-512 VNNI (YMM, 256-bit)
    # ================================================================

    def vpdpwssd_ymm_ymm_ymm(self, dst: int, src1: int, src2: int):
        """
        VPDPWSSD ymm, ymm, ymm — AVX-512 VNNI multiply-accumulate.
        dst[i32] += src1[2i:s16]*src2[2i:s16] + src1[2i+1:s16]*src2[2i+1:s16]
        Replaces VPMADDWD + VPADDD in one instruction.
        EVEX.256.66.0F38.W0 52 /r. Supports YMM0-31.
        """
        self._evex(mm=2, pp=1, w=0, vvvv=src1, r=dst, x=0, b=src2, ll=1, reg_reg=True)
        self._emit(0x52)
        self._modrm(0b11, dst, src2)

    # ================================================================
    # PREFETCH VARIANTS + NOP + LFENCE
    # ================================================================

    def _prefetch_generic(self, base: int, offset: int, reg_opcode: int):
        """Helper for PREFETCH variants. reg_opcode: 0=NTA, 1=T0, 2=T1, 3=T2."""
        if base > 7:
            self._emit(0x41)
        self._emit(0x0F, 0x18)
        needs_sib = (base & 7) == 4
        rbp_base = (base & 7) == 5
        if offset == 0 and not rbp_base:
            self._modrm(0b00, reg_opcode, base)
            if needs_sib:
                self._emit(0x24)
        elif -128 <= offset <= 127:
            self._modrm(0b01, reg_opcode, base)
            if needs_sib:
                self._emit(0x24)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, reg_opcode, base)
            if needs_sib:
                self._emit(0x24)
            self._emit_bytes(struct.pack("<i", offset))

    def prefetchT1_mem(self, base: int, offset: int = 0):
        """PREFETCHT1 [base + offset] — bring data to L2 cache."""
        self._prefetch_generic(base, offset, reg_opcode=2)

    def prefetchT2_mem(self, base: int, offset: int = 0):
        """PREFETCHT2 [base + offset] — bring data to L3 cache."""
        self._prefetch_generic(base, offset, reg_opcode=3)

    def prefetchNTA_mem(self, base: int, offset: int = 0):
        """PREFETCHNTA [base + offset] — non-temporal prefetch (no cache pollution)."""
        self._prefetch_generic(base, offset, reg_opcode=0)

    def lfence(self):
        """LFENCE — load fence, serializes loads."""
        self._emit(0x0F, 0xAE, 0xE8)

    def multi_nop(self, size: int):
        """Variable-size NOP (1-15 bytes). AMD Zen recommended sequences."""
        nop_table = {
            1: [0x90],
            2: [0x66, 0x90],
            3: [0x0F, 0x1F, 0x00],
            4: [0x0F, 0x1F, 0x40, 0x00],
            5: [0x0F, 0x1F, 0x44, 0x00, 0x00],
            6: [0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
            7: [0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],
            8: [0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            9: [0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        }
        size = max(1, min(size, 15))
        if size <= 9:
            for b in nop_table[size]:
                self._emit(b)
        else:
            for b in nop_table[9]:
                self._emit(b)
            remainder = size - 9
            if remainder in nop_table:
                for b in nop_table[remainder]:
                    self._emit(b)
            else:
                for _ in range(remainder):
                    self._emit(0x90)

    def nop_align(self, alignment: int = 32):
        """Emit canonical multi-byte NOPs to align next instruction.
        Uses Intel SDM Vol 2, Table 4-12 NOP sequences."""
        pos = len(self.code)
        pad = (alignment - (pos % alignment)) % alignment
        while pad > 0:
            n = min(pad, 9)
            self.multi_nop(n)
            pad -= n

    # ================================================================
    # GPR 32-BIT OPERATIONS (no REX.W — zero-extends to 64-bit)
    # ================================================================

    def mov_reg_mem32(self, dst: int, base: int, offset: int = 0):
        """MOV r32, [base+offset] — 32-bit load (zero-extends to 64-bit)."""
        rex = 0x40
        if dst >= 8:
            rex |= 0x04
        if base >= 8:
            rex |= 0x01
        if rex != 0x40:
            self._emit(rex)
        self._emit(0x8B)
        if (base & 7) == 4:
            if offset == 0 and (base & 7) != 5:
                self._modrm(0b00, dst & 7, 0b100)
                self._emit(0x24)
            elif -128 <= offset <= 127:
                self._modrm(0b01, dst & 7, 0b100)
                self._emit(0x24)
                self._emit(offset & 0xFF)
            else:
                self._modrm(0b10, dst & 7, 0b100)
                self._emit(0x24)
                self._emit_bytes(struct.pack('<i', offset))
        elif offset == 0 and (base & 7) != 5:
            self._modrm(0b00, dst & 7, base & 7)
        elif -128 <= offset <= 127:
            self._modrm(0b01, dst & 7, base & 7)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, dst & 7, base & 7)
            self._emit_bytes(struct.pack('<i', offset))

    def movzx_r32_word_mem(self, dst: int, base: int, offset: int = 0):
        """MOVZX r32, word [base+offset] — 16-bit load, zero-extend to 32."""
        rex = 0x40
        if dst >= 8:
            rex |= 0x04
        if base >= 8:
            rex |= 0x01
        if rex != 0x40:
            self._emit(rex)
        self._emit(0x0F, 0xB7)
        if (base & 7) == 4:
            if offset == 0 and (base & 7) != 5:
                self._modrm(0b00, dst & 7, 0b100)
                self._emit(0x24)
            elif -128 <= offset <= 127:
                self._modrm(0b01, dst & 7, 0b100)
                self._emit(0x24)
                self._emit(offset & 0xFF)
            else:
                self._modrm(0b10, dst & 7, 0b100)
                self._emit(0x24)
                self._emit_bytes(struct.pack('<i', offset))
        elif offset == 0 and (base & 7) != 5:
            self._modrm(0b00, dst & 7, base & 7)
        elif -128 <= offset <= 127:
            self._modrm(0b01, dst & 7, base & 7)
            self._emit(offset & 0xFF)
        else:
            self._modrm(0b10, dst & 7, base & 7)
            self._emit_bytes(struct.pack('<i', offset))

    def and_reg32_imm32(self, reg: int, value: int):
        """AND r32, imm32 — 32-bit AND (no REX.W, zero-extends)."""
        rex = 0x40
        if reg >= 8:
            rex |= 0x01
        if rex != 0x40:
            self._emit(rex)
        self._emit(0x81)
        self._modrm(0b11, 4, reg & 7)
        self._emit_bytes(struct.pack('<I', value & 0xFFFFFFFF))

    def shr_reg32_imm8(self, reg: int, count: int):
        """SHR r32, imm8 — 32-bit shift right (no REX.W, zero-extends)."""
        rex = 0x40
        if reg >= 8:
            rex |= 0x01
        if rex != 0x40:
            self._emit(rex)
        self._emit(0xC1)
        self._modrm(0b11, 5, reg & 7)
        self._emit(count & 0xFF)

    def or_reg32_reg32(self, dst: int, src: int):
        """OR r32, r32 — 32-bit OR (no REX.W, zero-extends)."""
        rex = 0x40
        if src >= 8:
            rex |= 0x04
        if dst >= 8:
            rex |= 0x01
        if rex != 0x40:
            self._emit(rex)
        self._emit(0x09)
        self._modrm(0b11, src & 7, dst & 7)
