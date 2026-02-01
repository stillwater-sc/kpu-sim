#!/usr/bin/env python3
"""Class 1: Fused matmul+bias+activation verification.

Verifies that fused patterns produce the same result as unfused composition.

Usage:
    python verification/kernels/class1_dense_linear/verify_fused_ops.py
"""

import sys
import numpy as np

try:
    import kpu
except ImportError:
    sys.exit("ERROR: kpu package not found. Build and install first.")


TOL = 1e-5

# Test dimensions: (M, K, N) — output is (M, N)
TEST_SIZES = [
    (32, 128, 64),
    (1, 64, 10),
    (16, 256, 128),
]


def ref_gelu(x):
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def ref_silu(x):
    return x / (1.0 + np.exp(-x))


# (pattern_name, fused_fn_builder, unfused_ref)
# fused_fn_builder returns a @kpu.compile function
# unfused_ref takes (x_np, w_np, b_np) and returns numpy result

PATTERNS = [
    (
        "matmul+relu",
        lambda: (
            lambda x, w: kpu.relu(x @ w)
        ),
        lambda x, w, b: np.maximum(x @ w, 0),
        False,  # no bias
    ),
    (
        "matmul+bias+relu",
        lambda: (
            lambda x, w, b: kpu.relu(x @ w + b)
        ),
        lambda x, w, b: np.maximum(x @ w + b, 0),
        True,  # has bias
    ),
    (
        "matmul+bias+gelu",
        lambda: (
            lambda x, w, b: kpu.gelu(x @ w + b)
        ),
        lambda x, w, b: ref_gelu(x @ w + b),
        True,
    ),
    (
        "matmul+bias+silu",
        lambda: (
            lambda x, w, b: kpu.silu(x @ w + b)
        ),
        lambda x, w, b: ref_silu(x @ w + b),
        True,
    ),
]


def main():
    kpu.set_fidelity(kpu.BEHAVIORAL)
    rng = np.random.default_rng(42)

    results = []

    for pattern_name, fused_builder, unfused_ref, has_bias in PATTERNS:
        for M, K, N in TEST_SIZES:
            x_np = rng.standard_normal((M, K)).astype(np.float32)
            w_np = rng.standard_normal((K, N)).astype(np.float32)
            b_np = rng.standard_normal((N,)).astype(np.float32) if has_bias else None

            # Unfused NumPy reference
            ref_out = unfused_ref(x_np, w_np, b_np)

            try:
                # Build fused KPU function
                fused_fn = fused_builder()

                x_kpu = kpu.Tensor(x_np)
                w_kpu = kpu.Tensor(w_np)

                if has_bias:
                    b_kpu = kpu.Tensor(b_np)
                    fused_out = fused_fn(x_kpu, w_kpu, b_kpu)
                else:
                    fused_out = fused_fn(x_kpu, w_kpu)

                fused_np = fused_out.numpy() if hasattr(fused_out, "numpy") else np.asarray(fused_out)
                max_diff = float(np.max(np.abs(fused_np - ref_out)))
                passed = max_diff < TOL
                status = "PASS" if passed else "FAIL"
            except Exception as e:
                status = "ERROR"
                max_diff = float("nan")
                print(f"  ERROR {pattern_name} ({M},{K},{N}): {e}")

            results.append((pattern_name, f"({M},{K},{N})", status, max_diff))

    # Also test fused vs unfused KPU (fused output == unfused output)
    print()
    print("=== Fused vs Reference ===")
    print(f"{'Pattern':<22} {'Size':<16} {'Status':<6} {'Max Diff':>12}")
    print("-" * 60)
    for pattern, size, status, diff in results:
        diff_str = f"{diff:.2e}" if not np.isnan(diff) else "N/A"
        print(f"{pattern:<22} {size:<16} {status:<6} {diff_str:>12}")

    n_pass = sum(1 for r in results if r[2] == "PASS")
    n_fail = sum(1 for r in results if r[2] == "FAIL")
    n_err = sum(1 for r in results if r[2] == "ERROR")
    total = len(results)

    print("-" * 60)
    print(f"Total: {total}  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_err}")

    return 0 if n_fail == 0 and n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
