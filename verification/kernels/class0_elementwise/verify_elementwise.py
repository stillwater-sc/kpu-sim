#!/usr/bin/env python3
"""Class 0: Elementwise kernel verification.

Tests each elementwise op against NumPy reference across shape sweeps.
Fidelity: BEHAVIORAL only (elementwise has no distinct transactional timing).

Usage:
    python verification/kernels/class0_elementwise/verify_elementwise.py
"""

import sys
import numpy as np

try:
    import kpu
except ImportError:
    sys.exit("ERROR: kpu package not found. Build and install first.")


# ---------------------------------------------------------------------------
# Reference implementations (NumPy)
# ---------------------------------------------------------------------------

def ref_relu(x):
    return np.maximum(x, 0)

def ref_add(a, b):
    return a + b

def ref_mul(a, b):
    return a * b

def ref_neg(x):
    return -x

def ref_gelu(x):
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def ref_silu(x):
    return x / (1.0 + np.exp(-x))

def ref_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def ref_tanh(x):
    return np.tanh(x)

def ref_exp(x):
    return np.exp(x)

def ref_log(x):
    return np.log(x)

def ref_sqrt(x):
    return np.sqrt(x)

def ref_softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

SHAPES = [
    (32,),
    (32, 128),
    (4, 32, 128),
    (1, 3, 64, 64),
]

# (name, kpu_fn, ref_fn, tolerance, input_domain)
# input_domain: "standard" uses randn, "positive" uses abs(randn)+eps
UNARY_OPS = [
    # Exact-match ops
    ("relu",    kpu.relu,    ref_relu,    0.0,   "standard"),
    ("neg",     lambda t: -t, ref_neg,    0.0,   "standard"),
    # ULP-bounded ops
    ("gelu",    kpu.gelu,    ref_gelu,    1e-6,  "standard"),
    ("silu",    kpu.silu,    ref_silu,    1e-6,  "standard"),
    ("sigmoid", kpu.sigmoid, ref_sigmoid, 1e-6,  "standard"),
    ("tanh",    kpu.tanh,    ref_tanh,    1e-6,  "standard"),
    ("exp",     kpu.exp,     ref_exp,     1e-6,  "small"),     # small input to avoid overflow
    ("log",     kpu.log,     ref_log,     1e-6,  "positive"),
    ("sqrt",    kpu.sqrt,    ref_sqrt,    1e-6,  "positive"),
    ("softmax", kpu.softmax, ref_softmax, 1e-6,  "standard"),
]

BINARY_OPS = [
    ("add", lambda a, b: a + b, ref_add, 0.0),
    ("mul", lambda a, b: a * b, ref_mul, 0.0),
]


def make_input(shape, domain, rng):
    """Generate input tensor for the given domain."""
    if domain == "positive":
        return np.abs(rng.standard_normal(shape).astype(np.float32)) + 1e-6
    if domain == "small":
        return rng.standard_normal(shape).astype(np.float32) * 0.5
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    kpu.set_fidelity(kpu.BEHAVIORAL)
    rng = np.random.default_rng(42)

    results = []  # (op, shape, status, max_diff)

    # --- Unary ops ---
    for op_name, kpu_fn, ref_fn, tol, domain in UNARY_OPS:
        for shape in SHAPES:
            x_np = make_input(shape, domain, rng)
            x_kpu = kpu.Tensor(x_np)

            try:
                out_kpu = kpu_fn(x_kpu)
                out_np = ref_fn(x_np)

                kpu_arr = out_kpu.numpy() if hasattr(out_kpu, "numpy") else np.asarray(out_kpu)
                max_diff = float(np.max(np.abs(kpu_arr - out_np)))

                if tol == 0.0:
                    passed = max_diff == 0.0
                else:
                    passed = max_diff < tol

                status = "PASS" if passed else "FAIL"
            except Exception as e:
                status = "ERROR"
                max_diff = float("nan")
                print(f"  ERROR {op_name} {shape}: {e}")

            results.append((op_name, str(shape), status, max_diff))

    # --- Binary ops ---
    for op_name, kpu_fn, ref_fn, tol in BINARY_OPS:
        for shape in SHAPES:
            a_np = rng.standard_normal(shape).astype(np.float32)
            b_np = rng.standard_normal(shape).astype(np.float32)
            a_kpu = kpu.Tensor(a_np)
            b_kpu = kpu.Tensor(b_np)

            try:
                out_kpu = kpu_fn(a_kpu, b_kpu)
                out_np = ref_fn(a_np, b_np)

                kpu_arr = out_kpu.numpy() if hasattr(out_kpu, "numpy") else np.asarray(out_kpu)
                max_diff = float(np.max(np.abs(kpu_arr - out_np)))

                passed = max_diff == 0.0 if tol == 0.0 else max_diff < tol
                status = "PASS" if passed else "FAIL"
            except Exception as e:
                status = "ERROR"
                max_diff = float("nan")
                print(f"  ERROR {op_name} {shape}: {e}")

            results.append((op_name, str(shape), status, max_diff))

    # --- Summary ---
    print()
    print(f"{'Op':<10} {'Shape':<22} {'Status':<6} {'Max Diff':>12}")
    print("-" * 54)
    for op, shape, status, diff in results:
        diff_str = f"{diff:.2e}" if not np.isnan(diff) else "N/A"
        print(f"{op:<10} {shape:<22} {status:<6} {diff_str:>12}")

    n_pass = sum(1 for r in results if r[2] == "PASS")
    n_fail = sum(1 for r in results if r[2] == "FAIL")
    n_err = sum(1 for r in results if r[2] == "ERROR")
    total = len(results)

    print("-" * 54)
    print(f"Total: {total}  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_err}")

    return 0 if n_fail == 0 and n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
