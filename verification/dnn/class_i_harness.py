#!/usr/bin/env python3
"""
Class I DNN Verification Harness

Verifies compute-bound (GEMM-dominant) models at BEHAVIORAL and TRANSACTIONAL
fidelity levels against the criteria in verification/dnn/README.md:

Functional (BEHAVIORAL):
  - MatMul: max_diff < 1e-5 vs NumPy
  - End-to-end model: max_diff < 1e-3 vs NumPy

Performance (TRANSACTIONAL):
  - XUE FLOP count: exact match or within tiling overhead (<20%)
  - Roofline: achieved GFLOPS within 10% of predicted

Usage:
    python verification/dnn/class_i_harness.py              # All tests
    python verification/dnn/class_i_harness.py --behavioral  # Behavioral only
    python verification/dnn/class_i_harness.py --transactional  # Transactional only
    python verification/dnn/class_i_harness.py -v            # Verbose output
"""

import sys
import os
import argparse
import traceback

# Add python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
import kpu


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name, fidelity, passed, details=""):
        self.name = name
        self.fidelity = fidelity
        self.passed = passed
        self.details = details

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        s = f"  [{status}] {self.name} ({self.fidelity})"
        if self.details:
            s += f" — {self.details}"
        return s


results: list[TestResult] = []


def record(name, fidelity, passed, details=""):
    results.append(TestResult(name, fidelity, passed, details))


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def run_minimal_mlp(fidelity, verbose=False):
    """Minimal 2-layer MLP: [32,784] -> 128 -> 10"""
    kpu.set_fidelity(fidelity)
    if fidelity == kpu.TRANSACTIONAL:
        kpu.set_clock_frequency(1.0)

    @kpu.compile
    def mlp(x, w1, w2):
        h = kpu.relu(kpu.matmul(x, w1))
        return kpu.matmul(h, w2)

    np.random.seed(42)
    B, D_in, D_h, D_out = 32, 784, 128, 10
    x_np = np.random.randn(B, D_in).astype(np.float32)
    w1_np = np.random.randn(D_in, D_h).astype(np.float32)
    w2_np = np.random.randn(D_h, D_out).astype(np.float32)

    x = kpu.Tensor(x_np)
    w1 = kpu.Tensor(w1_np)
    w2 = kpu.Tensor(w2_np)

    result = mlp(x, w1, w2)
    result_np = np.array(result.data)

    # NumPy reference
    ref = np.maximum(0, x_np @ w1_np) @ w2_np
    max_diff = float(np.max(np.abs(result_np - ref)))

    fid_name = "BEHAVIORAL" if fidelity == kpu.BEHAVIORAL else "TRANSACTIONAL"

    # Functional check
    record("minimal_mlp/functional", fid_name,
           max_diff < 1e-5, f"max_diff={max_diff:.2e}")

    if verbose:
        print(f"    minimal_mlp max_diff={max_diff:.2e}")

    # Performance checks (transactional only)
    if fidelity == kpu.TRANSACTIONAL:
        stats = mlp.stats
        expected_flops = 2 * B * D_in * D_h + 2 * B * D_h * D_out

        # FLOP count check
        if stats.matmul_flops > 0:
            flop_error = abs(stats.matmul_flops - expected_flops) / expected_flops * 100
            record("minimal_mlp/flop_count", fid_name,
                   flop_error < 20, f"expected={expected_flops:,} actual={stats.matmul_flops:,} err={flop_error:.1f}%")
        else:
            record("minimal_mlp/flop_count", fid_name, False, "no FLOP data")

        # Roofline check
        analysis = kpu.get_operational_analysis(
            peak_gflops=512.0, dram_bandwidth_gbs=64.0, clock_ghz=1.0)
        predicted = analysis.get('predicted_gflops', 0)
        actual = stats.gflops if hasattr(stats, 'gflops') else 0

        if predicted > 0 and actual > 0:
            roofline_err = abs(actual - predicted) / predicted * 100
            record("minimal_mlp/roofline", fid_name,
                   roofline_err <= 10, f"predicted={predicted:.1f} actual={actual:.1f} err={roofline_err:.1f}%")
        else:
            record("minimal_mlp/roofline", fid_name, False,
                   f"predicted={predicted:.1f} actual={actual:.1f}")

        if verbose:
            print(f"    compute_cycles={stats.compute_cycles:,} memory_cycles={stats.memory_cycles:,}")
            print(f"    matmul_flops={stats.matmul_flops:,} gflops={stats.gflops:.1f}")


def run_mnist_mlp(fidelity, verbose=False):
    """3-layer MNIST MLP: [32,784] -> 128 -> 64 -> 10"""
    kpu.set_fidelity(fidelity)
    if fidelity == kpu.TRANSACTIONAL:
        kpu.set_clock_frequency(1.0)

    @kpu.compile
    def mnist_mlp(x, w1, b1, w2, b2, w3, b3):
        h1 = kpu.relu(x @ w1 + b1)
        h2 = kpu.relu(h1 @ w2 + b2)
        return h2 @ w3 + b3

    np.random.seed(42)
    B = 32
    dims = [784, 128, 64, 10]

    x_np = np.random.randn(B, dims[0]).astype(np.float32) * 0.01
    w1_np = np.random.randn(dims[0], dims[1]).astype(np.float32) * 0.01
    b1_np = np.zeros(dims[1], dtype=np.float32)
    w2_np = np.random.randn(dims[1], dims[2]).astype(np.float32) * 0.01
    b2_np = np.zeros(dims[2], dtype=np.float32)
    w3_np = np.random.randn(dims[2], dims[3]).astype(np.float32) * 0.01
    b3_np = np.zeros(dims[3], dtype=np.float32)

    tensors = [kpu.Tensor(a) for a in [x_np, w1_np, b1_np, w2_np, b2_np, w3_np, b3_np]]
    result = mnist_mlp(*tensors)
    result_np = np.array(result.data)

    # NumPy reference
    h1 = np.maximum(0, x_np @ w1_np + b1_np)
    h2 = np.maximum(0, h1 @ w2_np + b2_np)
    ref = h2 @ w3_np + b3_np
    max_diff = float(np.max(np.abs(result_np - ref)))

    fid_name = "BEHAVIORAL" if fidelity == kpu.BEHAVIORAL else "TRANSACTIONAL"

    record("mnist_mlp/functional", fid_name,
           max_diff < 1e-3, f"max_diff={max_diff:.2e}")

    if verbose:
        print(f"    mnist_mlp max_diff={max_diff:.2e}")

    if fidelity == kpu.TRANSACTIONAL:
        stats = mnist_mlp.stats
        expected_flops = (2*B*dims[0]*dims[1] + 2*B*dims[1]*dims[2] + 2*B*dims[2]*dims[3])

        if stats.matmul_flops > 0:
            flop_error = abs(stats.matmul_flops - expected_flops) / expected_flops * 100
            record("mnist_mlp/flop_count", fid_name,
                   flop_error < 20, f"expected={expected_flops:,} actual={stats.matmul_flops:,} err={flop_error:.1f}%")
        else:
            record("mnist_mlp/flop_count", fid_name, False, "no FLOP data")

        analysis = kpu.get_operational_analysis(
            peak_gflops=512.0, dram_bandwidth_gbs=64.0, clock_ghz=1.0)
        predicted = analysis.get('predicted_gflops', 0)
        actual = stats.gflops if hasattr(stats, 'gflops') else 0

        if predicted > 0 and actual > 0:
            roofline_err = abs(actual - predicted) / predicted * 100
            record("mnist_mlp/roofline", fid_name,
                   roofline_err <= 10, f"predicted={predicted:.1f} actual={actual:.1f} err={roofline_err:.1f}%")
        else:
            record("mnist_mlp/roofline", fid_name, False,
                   f"predicted={predicted:.1f} actual={actual:.1f}")

        if verbose:
            print(f"    compute_cycles={stats.compute_cycles:,} gflops={stats.gflops:.1f}")


def run_xue_validation(fidelity, verbose=False):
    """XUE validation workload: 2-layer MLP with bias [32,784] -> 256 -> 10"""
    kpu.set_fidelity(fidelity)
    if fidelity == kpu.TRANSACTIONAL:
        kpu.set_clock_frequency(1.0)

    @kpu.compile
    def mlp_workload(x, w1, b1, w2, b2):
        h = kpu.relu(x @ w1 + b1)
        return h @ w2 + b2

    np.random.seed(42)
    B, D_in, D_h, D_out = 32, 784, 256, 10

    x_np = np.random.randn(B, D_in).astype(np.float32)
    w1_np = np.random.randn(D_in, D_h).astype(np.float32) * 0.01
    b1_np = np.zeros(D_h, dtype=np.float32)
    w2_np = np.random.randn(D_h, D_out).astype(np.float32) * 0.01
    b2_np = np.zeros(D_out, dtype=np.float32)

    tensors = [kpu.Tensor(a) for a in [x_np, w1_np, b1_np, w2_np, b2_np]]
    result = mlp_workload(*tensors)
    result_np = np.array(result.data)

    ref = np.maximum(0, x_np @ w1_np + b1_np) @ w2_np + b2_np
    max_diff = float(np.max(np.abs(result_np - ref)))

    fid_name = "BEHAVIORAL" if fidelity == kpu.BEHAVIORAL else "TRANSACTIONAL"

    record("xue_validation/functional", fid_name,
           max_diff < 1e-3, f"max_diff={max_diff:.2e}")

    if verbose:
        print(f"    xue_validation max_diff={max_diff:.2e}")

    if fidelity == kpu.TRANSACTIONAL:
        stats = mlp_workload.stats
        expected_flops = 2*B*D_in*D_h + 2*B*D_h*D_out

        if stats.matmul_flops > 0:
            flop_error = abs(stats.matmul_flops - expected_flops) / expected_flops * 100
            record("xue_validation/flop_count", fid_name,
                   flop_error < 20, f"expected={expected_flops:,} actual={stats.matmul_flops:,} err={flop_error:.1f}%")

        analysis = kpu.get_operational_analysis(
            peak_gflops=512.0, dram_bandwidth_gbs=64.0, clock_ghz=1.0)
        predicted = analysis.get('predicted_gflops', 0)
        actual = stats.gflops if hasattr(stats, 'gflops') else 0

        if predicted > 0 and actual > 0:
            roofline_err = abs(actual - predicted) / predicted * 100
            record("xue_validation/roofline", fid_name,
                   roofline_err <= 10, f"predicted={predicted:.1f} actual={actual:.1f} err={roofline_err:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_MODELS = [
    ("minimal_mlp", run_minimal_mlp),
    ("mnist_mlp", run_mnist_mlp),
    ("xue_validation", run_xue_validation),
]


def main():
    parser = argparse.ArgumentParser(description="Class I DNN Verification Harness")
    parser.add_argument("--behavioral", action="store_true", help="Run behavioral only")
    parser.add_argument("--transactional", action="store_true", help="Run transactional only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_behav = not args.transactional or args.behavioral
    run_trans = not args.behavioral or args.transactional

    # If neither flag, run both
    if not args.behavioral and not args.transactional:
        run_behav = True
        run_trans = True

    fidelities = []
    if run_behav:
        fidelities.append(("BEHAVIORAL", kpu.BEHAVIORAL))
    if run_trans:
        fidelities.append(("TRANSACTIONAL", kpu.TRANSACTIONAL))

    print("=" * 70)
    print("Class I DNN Verification Harness")
    print("=" * 70)

    for fid_name, fid_val in fidelities:
        print(f"\n--- {fid_name} ---")
        for model_name, model_fn in ALL_MODELS:
            print(f"  Running {model_name}...")
            try:
                model_fn(fid_val, verbose=args.verbose)
            except Exception as e:
                record(f"{model_name}/error", fid_name, False, str(e))
                if args.verbose:
                    traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("Verification Results")
    print("=" * 70)

    passed = 0
    failed = 0
    for r in results:
        print(r)
        if r.passed:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Total: {passed + failed}  Passed: {passed}  Failed: {failed}")

    # Print verification matrix
    print()
    print("Verification Matrix (DNN Class I):")
    print(f"  {'Model':<25} {'BEHAVIORAL':>12} {'TRANSACTIONAL':>15}")
    print(f"  {'-'*25} {'-'*12} {'-'*15}")

    models_seen = {}
    for r in results:
        key = r.name.split('/')[0]
        if key not in models_seen:
            models_seen[key] = {'BEHAVIORAL': None, 'TRANSACTIONAL': None}
        if models_seen[key][r.fidelity] is None:
            models_seen[key][r.fidelity] = r.passed
        else:
            models_seen[key][r.fidelity] = models_seen[key][r.fidelity] and r.passed

    for model, status in models_seen.items():
        behav = "PASS" if status['BEHAVIORAL'] else ("FAIL" if status['BEHAVIORAL'] is False else "—")
        trans = "PASS" if status['TRANSACTIONAL'] else ("FAIL" if status['TRANSACTIONAL'] is False else "—")
        print(f"  {model:<25} {behav:>12} {trans:>15}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
