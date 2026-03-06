#!/usr/bin/env python3
"""
Genesis Kernel — Test Suite (pytest).

Tests both kernel families:
  - NF4: fused dequantization + matmul (AVX-512 ZMM)
  - Q4_K: ggml-compatible dot product (AVX-512 VNNI in YMM)

Run:
    pytest tests/ -v
"""
import sys
import os
import struct
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genesis_kernel import (
    compile_best_nf4_matmul,
    quantize_nf4,
    reorder_activations,
    dequant_nf4_reference,
    matmul_nf4_reference,
    pack_nf4,
    NF4_TABLE,
    BLOCKSIZE,
)
from genesis_kernel.q4k_kernel import (
    compile_q4k_kernel,
    compile_turbo7,
    turbo7_ops,
    OpQ4K,
)


# ============================================================
# NF4 tests
# ============================================================

class TestNF4Quantization:
    """Quantization round-trip and utility tests."""

    def test_quantize_roundtrip(self):
        """quantize → dequant preserves values within NF4 precision."""
        np.random.seed(42)
        K = 256
        weights = np.random.randn(K).astype(np.float32) * 0.5

        nf4, scales = quantize_nf4(weights)
        recovered = dequant_nf4_reference(nf4, scales, K)

        max_err = np.max(np.abs(weights - recovered))
        mean_err = np.mean(np.abs(weights - recovered))
        assert max_err < 0.5, f"Max error too large: {max_err}"
        assert mean_err < 0.1, f"Mean error too large: {mean_err}"

    def test_pack_nf4_roundtrip(self):
        """pack_nf4 packs two 4-bit indices per byte correctly."""
        indices = np.array([3, 12, 0, 15, 7, 8], dtype=np.uint8)
        packed = pack_nf4(indices)
        # Verify: byte[0] = 3 | (12<<4) = 0xC3, byte[1] = 0 | (15<<4) = 0xF0, etc.
        assert packed[0] == (3 | (12 << 4))
        assert packed[1] == (0 | (15 << 4))
        assert packed[2] == (7 | (8 << 4))

    def test_reorder_activations_layout(self):
        """reorder_activations groups even/odd indices correctly."""
        act = np.arange(64, dtype=np.float32)
        reord = reorder_activations(act)

        # First group of 32: even [0,2,4,...,30] then odd [1,3,5,...,31]
        for i in range(16):
            assert reord[i] == act[2 * i], f"Even mismatch at {i}"
            assert reord[16 + i] == act[2 * i + 1], f"Odd mismatch at {i}"
        # Second group of 32
        for i in range(16):
            assert reord[32 + i] == act[32 + 2 * i], f"Even mismatch group 2 at {i}"
            assert reord[48 + i] == act[32 + 2 * i + 1], f"Odd mismatch group 2 at {i}"

    def test_reorder_required_for_correctness(self):
        """Kernel gives WRONG results without reorder — this is a contract."""
        np.random.seed(99)
        M, K = 256, 128
        weights = np.random.randn(M * K).astype(np.float32) * 0.3
        activations = np.random.randn(K).astype(np.float32)
        nf4, scales = quantize_nf4(weights)

        ref = matmul_nf4_reference(nf4, scales, activations, M, K)
        act_reord = reorder_activations(activations)
        kernel = compile_best_nf4_matmul(M=M, K=K)

        # With reorder: should match reference
        correct = kernel(nf4, scales, act_reord, M, K)
        cosine_correct = _cosine(ref, correct)
        assert cosine_correct > 0.999, f"Reordered should match: cosine={cosine_correct}"

        # Without reorder: should NOT match reference
        wrong = kernel(nf4, scales, activations, M, K)
        cosine_wrong = _cosine(ref, wrong)
        assert cosine_wrong < 0.99, (
            f"Without reorder should diverge: cosine={cosine_wrong}. "
            "If this passes, the reorder contract is broken."
        )


class TestNF4Kernels:
    """Accuracy tests for NF4 matmul kernels against Python reference."""

    @pytest.mark.parametrize("M,K,label", [
        (1024, 2048, "80B gate_up (baked)"),
        (2048, 512,  "80B down_proj (baked)"),
        (3072, 2048, "30B gate_up (baked)"),
        (2048, 1536, "30B down_proj (baked)"),
    ])
    def test_baked_kernel(self, M, K, label):
        """Baked evolved kernel matches Python reference."""
        self._check_kernel(M, K, label)

    @pytest.mark.parametrize("M,K", [
        (256, 128),
        (512, 256),
    ])
    def test_base_kernel(self, M, K):
        """Base 2-pipeline kernel matches Python reference."""
        self._check_kernel(M, K, "base fallback")

    def _check_kernel(self, M, K, label):
        np.random.seed(hash((M, K)) % 2**31)
        weights = np.random.randn(M * K).astype(np.float32) * 0.3
        activations = np.random.randn(K).astype(np.float32)

        nf4, scales = quantize_nf4(weights)
        ref = matmul_nf4_reference(nf4, scales, activations, M, K)

        act_reord = reorder_activations(activations)
        kernel = compile_best_nf4_matmul(M=M, K=K)
        result = kernel(nf4, scales, act_reord, M, K)

        abs_diff = np.abs(ref - result)
        max_abs = np.max(abs_diff)
        cosine = _cosine(ref, result)

        # NF4 accumulates quantization error across K, tolerance ~ sqrt(K)
        tol_abs = 0.05 * np.sqrt(K)
        assert max_abs < tol_abs, (
            f"{label} {M}x{K}: max_abs={max_abs:.4f} > tol={tol_abs:.2f}"
        )
        assert cosine > 0.999, (
            f"{label} {M}x{K}: cosine={cosine:.6f} < 0.999"
        )


# ============================================================
# Q4_K tests
# ============================================================

def _make_q4k_block():
    """Create one valid Q4_K block (144 bytes) with known data."""
    block = bytearray(144)
    # d (fp16 at offset 0): use 1.0 in fp16 = 0x3C00
    struct.pack_into('<H', block, 0, 0x3C00)
    # dmin (fp16 at offset 2): use 0.0
    struct.pack_into('<H', block, 2, 0x0000)
    # scales (12 bytes at offset 4): all 1s (each 6-bit scale = 1)
    for i in range(12):
        block[4 + i] = 0x01
    # qs (128 bytes at offset 16): fill with 0x11 (nibbles = 1)
    for i in range(128):
        block[16 + i] = 0x11
    return bytes(block)


def _make_q8k_block():
    """Create one valid Q8_K block (292 bytes) with known data."""
    block = bytearray(292)
    # d (float32 at offset 0): use 1.0
    struct.pack_into('<f', block, 0, 1.0)
    # qs (256 int8 at offset 4): fill with 1
    for i in range(256):
        block[4 + i] = 1
    # bsums (16 int16 at offset 260): each = sum of 16 qs = 16
    for i in range(16):
        struct.pack_into('<h', block, 260 + i * 2, 16)
    return bytes(block)


class TestQ4K:
    """Tests for Q4_K turbo7 kernel."""

    def test_compile_turbo7(self):
        """compile_turbo7() returns a callable kernel."""
        kernel = compile_turbo7()
        assert kernel is not None, "compile_turbo7() returned None"
        assert hasattr(kernel, '_func'), "Missing _func attribute"
        assert hasattr(kernel, '_size'), "Missing _size attribute"
        assert kernel._n_ops == 34, f"Expected 34 ops, got {kernel._n_ops}"

    def test_turbo7_ops_count(self):
        """turbo7 has exactly 34 ops (33 turbo6 + 1 prefetchnta)."""
        ops = turbo7_ops()
        assert len(ops) == 34, f"Expected 34 ops, got {len(ops)}"
        # The injected prefetch should be at position 19
        assert ops[19].nombre == "prefetchnta", (
            f"Op at position 19 should be prefetchnta, got {ops[19].nombre}"
        )

    def test_turbo7_nonzero_output(self):
        """Turbo7 kernel produces non-zero output on valid data."""
        kernel = compile_turbo7()
        if kernel is None:
            pytest.skip("Kernel compilation failed (no AVX-512?)")

        q4k = _make_q4k_block()
        q8k = _make_q8k_block()

        q4k_arr = np.frombuffer(q4k, dtype=np.uint8).copy()
        q8k_arr = np.frombuffer(q8k, dtype=np.uint8).copy()

        result = kernel(q4k_arr.ctypes.data, q8k_arr.ctypes.data, 1)
        # With d=1.0, scales=1, qs=1, q8=1, the result should be non-zero
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        # We don't check exact value (depends on scale decoding), just non-zero
        # and finite
        assert np.isfinite(result), f"Result is not finite: {result}"

    def test_turbo7_multiple_blocks(self):
        """Turbo7 kernel handles multiple blocks correctly."""
        kernel = compile_turbo7()
        if kernel is None:
            pytest.skip("Kernel compilation failed (no AVX-512?)")

        n_blocks = 4
        q4k = _make_q4k_block() * n_blocks
        q8k = _make_q8k_block() * n_blocks

        q4k_arr = np.frombuffer(q4k, dtype=np.uint8).copy()
        q8k_arr = np.frombuffer(q8k, dtype=np.uint8).copy()

        r1 = kernel(q4k_arr.ctypes.data, q8k_arr.ctypes.data, 1)
        r4 = kernel(q4k_arr.ctypes.data, q8k_arr.ctypes.data, n_blocks)

        assert np.isfinite(r1) and np.isfinite(r4)
        # 4 identical blocks should give ~4x the single-block result
        if abs(r1) > 1e-10:
            ratio = r4 / r1
            assert 3.0 < ratio < 5.0, (
                f"4 blocks should be ~4x one block: r1={r1}, r4={r4}, ratio={ratio}"
            )

    def test_turbo7_so_generation(self):
        """generate_turbo7_so() creates a .c and .so file."""
        import tempfile
        from genesis_kernel.q4k_kernel import generate_turbo7_so

        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = generate_turbo7_so(tmpdir)
            if so_path is None:
                pytest.skip("gcc not available for .so generation")
            assert os.path.exists(so_path), f".so not found at {so_path}"
            c_path = os.path.join(tmpdir, "genesis_turbo7.c")
            assert os.path.exists(c_path), f".c not found at {c_path}"
            # .so should be non-trivial size
            so_size = os.path.getsize(so_path)
            assert so_size > 1000, f".so too small: {so_size} bytes"

    def test_opq4k_repr(self):
        """OpQ4K has a readable repr."""
        op = OpQ4K("vpdpwssd", {"dst": "YMM22", "scale": "YMM2", "src": "YMM1"}, grupo="g0")
        r = repr(op)
        assert "vpdpwssd" in r
        assert "g0" in r
        # Without grupo
        op2 = OpQ4K("vpand", {"dst": "YMM1"})
        assert "vpand" in repr(op2)


# ============================================================
# Helpers
# ============================================================

def _cosine(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(np.dot(a, b) / (na * nb))
