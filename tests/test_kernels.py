#!/usr/bin/env python3
"""
Test suite for Genesis Kernel — verifies all baked kernels and base kernel
produce correct results against the Python reference implementation.

Tests:
  1. All 4 baked evolved kernels (1024x2048, 2048x512, 3072x2048, 2048x1536)
  2. Base kernel fallback for non-baked dimensions (256x128)
  3. Quantization round-trip accuracy
  4. Activation reordering correctness

Run:
    python -m tests.test_kernels          (from genesis_public/)
    python tests/test_kernels.py          (from genesis_public/)
"""
import sys
import os
import time
import numpy as np

# Ensure genesis_kernel is importable when running from genesis_public/
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


def test_quantize_roundtrip():
    """Verify quantize -> dequant preserves values within NF4 precision."""
    np.random.seed(42)
    K = 256
    weights = np.random.randn(K).astype(np.float32) * 0.5

    nf4, scales = quantize_nf4(weights)
    recovered = dequant_nf4_reference(nf4, scales, K)

    # NF4 has 4-bit precision, so relative error can be significant
    # but the quantization should map to the nearest table entry
    max_err = np.max(np.abs(weights - recovered))
    mean_err = np.mean(np.abs(weights - recovered))

    print(f"  quantize roundtrip: max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 0.5, f"Max error too large: {max_err}"
    assert mean_err < 0.1, f"Mean error too large: {mean_err}"
    return True


def test_reorder_activations():
    """Verify activation reordering groups even/odd indices correctly."""
    act = np.arange(64, dtype=np.float32)
    reord = reorder_activations(act)

    # First group of 32: even indices [0,2,4,...,30] then odd [1,3,5,...,31]
    for i in range(16):
        assert reord[i] == act[2 * i], f"Even mismatch at {i}: {reord[i]} != {act[2*i]}"
        assert reord[16 + i] == act[2 * i + 1], f"Odd mismatch at {i}"

    # Second group of 32
    for i in range(16):
        assert reord[32 + i] == act[32 + 2 * i], f"Even mismatch at group 2, {i}"
        assert reord[48 + i] == act[32 + 2 * i + 1], f"Odd mismatch at group 2, {i}"

    print("  reorder_activations: OK")
    return True


def test_kernel(M, K, label=""):
    """
    Test a kernel for dimensions M x K against the Python reference.

    Returns (passed, max_relative_error, time_ms).
    """
    np.random.seed(hash((M, K)) % 2**31)

    # Generate random weights and activations
    weights = np.random.randn(M * K).astype(np.float32) * 0.3
    activations = np.random.randn(K).astype(np.float32)

    # Quantize
    nf4, scales = quantize_nf4(weights)

    # Python reference (uses raw activations, not reordered)
    ref_output = matmul_nf4_reference(nf4, scales, activations, M, K)

    # Genesis kernel (uses reordered activations)
    act_reord = reorder_activations(activations)
    kernel = compile_best_nf4_matmul(M=M, K=K)
    origin = getattr(kernel, '_origin', 'unknown')

    t0 = time.perf_counter()
    kernel_output = kernel(nf4, scales, act_reord, M, K)
    t1 = time.perf_counter()
    time_us = (t1 - t0) * 1e6

    # Compare
    abs_diff = np.abs(ref_output - kernel_output)
    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)

    # Relative error (avoid division by zero)
    ref_abs = np.abs(ref_output)
    mask = ref_abs > 1e-6
    if np.any(mask):
        max_rel = np.max(abs_diff[mask] / ref_abs[mask])
    else:
        max_rel = 0.0

    # Cosine similarity
    norm_ref = np.linalg.norm(ref_output)
    norm_kern = np.linalg.norm(kernel_output)
    if norm_ref > 0 and norm_kern > 0:
        cosine = np.dot(ref_output, kernel_output) / (norm_ref * norm_kern)
    else:
        cosine = 1.0

    # NF4 matmul accumulates quantization errors across K elements,
    # so we use a tolerance proportional to sqrt(K)
    tol_abs = 0.05 * np.sqrt(K)
    passed = max_abs < tol_abs and cosine > 0.999

    status = "PASS" if passed else "FAIL"
    tag = f" [{label}]" if label else ""
    print(f"  {status} {M}x{K}{tag} ({origin})")
    print(f"       max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}, "
          f"cosine={cosine:.6f}, time={time_us:.0f}us")

    if not passed:
        print(f"       TOLERANCE: max_abs < {tol_abs:.2f}, cosine > 0.999")
        print(f"       ref[:5]  = {ref_output[:5]}")
        print(f"       kern[:5] = {kernel_output[:5]}")

    return passed


def main():
    print("=" * 60)
    print("Genesis Kernel — Test Suite")
    print("=" * 60)

    results = []

    # Test utilities
    print("\n--- Utility tests ---")
    results.append(("quantize_roundtrip", test_quantize_roundtrip()))
    results.append(("reorder_activations", test_reorder_activations()))

    # Test all 4 baked evolved kernels
    baked_dims = [
        (1024, 2048, "80B gate_up"),
        (2048, 512,  "80B down_proj"),
        (3072, 2048, "30B gate_up"),
        (2048, 1536, "30B down_proj"),
    ]

    print("\n--- Baked evolved kernels ---")
    for M, K, label in baked_dims:
        results.append((f"baked_{M}x{K}", test_kernel(M, K, label)))

    # Test base kernel fallback (non-baked dimensions)
    print("\n--- Base kernel fallback ---")
    results.append(("base_256x128", test_kernel(256, 128, "base fallback")))
    results.append(("base_512x256", test_kernel(512, 256, "base fallback")))

    # Summary
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {n_pass}/{n_total} passed")
    print(f"{'=' * 60}")

    if n_pass == n_total:
        print("All tests passed.")
        return 0
    else:
        failed = [name for name, p in results if not p]
        print(f"FAILED: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
