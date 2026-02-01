#!/usr/bin/env python3
"""Class 1: Dense linear (matmul) kernel verification.

Tests matmul with systematic dimension sweeps at BEHAVIORAL and TRANSACTIONAL
fidelity levels.

Usage:
    python verification/kernels/class1_dense_linear/verify_matmul.py
"""

import sys
import numpy as np

try:
    import kpu
except ImportError:
    sys.exit("ERROR: kpu package not found. Build and install first.")


# ---------------------------------------------------------------------------
# Test configurations: (M, N, K)
# ---------------------------------------------------------------------------

SIZES = {
    "aligned": [
        (16, 16, 16),
        (32, 32, 32),
        (128, 128, 128),
        (256, 256, 256),
    ],
    "rectangular": [
        (1, 128, 10),
        (32, 784, 128),
        (32, 128, 10),
    ],
    "non_aligned": [
        (7, 13, 5),
        (33, 65, 17),
    ],
    "large": [
        (512, 512, 512),
    ],
}

BEHAVIORAL_TOL = 1e-5


def run_behavioral(rng):
    """BEHAVIORAL: verify max_diff < 1e-5 vs NumPy A @ B."""
    results = []

    kpu.set_fidelity(kpu.BEHAVIORAL)

    for category, sizes in SIZES.items():
        for M, N, K in sizes:
            A_np = rng.standard_normal((M, K)).astype(np.float32)
            B_np = rng.standard_normal((K, N)).astype(np.float32)
            ref = A_np @ B_np

            try:
                A_kpu = kpu.Tensor(A_np)
                B_kpu = kpu.Tensor(B_np)
                out = A_kpu @ B_kpu
                out_np = out.numpy() if hasattr(out, "numpy") else np.asarray(out)

                max_diff = float(np.max(np.abs(out_np - ref)))
                passed = max_diff < BEHAVIORAL_TOL
                status = "PASS" if passed else "FAIL"
            except Exception as e:
                status = "ERROR"
                max_diff = float("nan")
                print(f"  ERROR behavioral ({M},{N},{K}): {e}")

            results.append(("BEHAVIORAL", category, f"({M},{N},{K})", status, max_diff))

    return results


def run_transactional(rng):
    """TRANSACTIONAL: verify FLOP count and functional correctness via @kpu.compile."""
    results = []

    # Check if transactional is available
    # Order matters: set fidelity first so native runtime is created with
    # the correct fidelity, then set clock frequency on that runtime.
    try:
        kpu.set_fidelity(kpu.TRANSACTIONAL)
        kpu.set_clock_frequency(1.0)  # 1 GHz for transactional timing
    except Exception:
        print("  TRANSACTIONAL fidelity not available, skipping.")
        return results

    # Use a subset for transactional (compilation overhead)
    trans_sizes = [
        ("aligned", (32, 32, 32)),
        ("aligned", (128, 128, 128)),
        ("rectangular", (32, 784, 128)),
        ("rectangular", (32, 128, 10)),
    ]

    for category, (M, N, K) in trans_sizes:
        A_np = rng.standard_normal((M, K)).astype(np.float32)
        B_np = rng.standard_normal((K, N)).astype(np.float32)
        ref = A_np @ B_np
        expected_flops = 2 * M * N * K

        try:
            # Reset XUE counters so FLOP counts are per-test
            try:
                kpu.reset_xue_counters()
            except Exception:
                pass

            @kpu.compile
            def matmul_fn(a, b):
                return a @ b

            A_kpu = kpu.Tensor(A_np)
            B_kpu = kpu.Tensor(B_np)
            out = matmul_fn(A_kpu, B_kpu)
            out_np = out.numpy() if hasattr(out, "numpy") else np.asarray(out)

            max_diff = float(np.max(np.abs(out_np - ref)))
            func_ok = max_diff < BEHAVIORAL_TOL

            # Check FLOP count from stats or XUE
            stats = getattr(matmul_fn, "stats", None)
            xue = None
            try:
                xue = kpu.get_xue_summary()
            except Exception:
                pass

            flop_ok = True
            reported_flops = None
            if xue and "total_flops" in xue:
                reported_flops = xue["total_flops"]
                # Allow 20% overhead for tiling
                flop_ok = (expected_flops <= reported_flops <= expected_flops * 1.2)
            elif stats and hasattr(stats, "matmul_flops"):
                reported_flops = stats.matmul_flops
                flop_ok = (expected_flops <= reported_flops <= expected_flops * 1.2)

            # Roofline check (informational — does not gate pass/fail since
            # simulator throughput != hardware throughput)
            roofline_info = ""
            if xue:
                try:
                    analysis = kpu.get_operational_analysis(
                        peak_gflops=1024.0,
                        dram_bandwidth_gbs=64.0,
                        clock_ghz=1.0,
                    )
                    if "predicted_gflops" in analysis:
                        roofline_info = f" roofline={analysis['predicted_gflops']:.1f}GFLOPS"
                except Exception:
                    pass

            all_ok = func_ok and flop_ok
            status = "PASS" if all_ok else "FAIL"

            detail = f"diff={max_diff:.2e}"
            if reported_flops is not None:
                detail += f" flops={reported_flops}/{expected_flops}"
            detail += roofline_info
            if not func_ok:
                detail += " [func]"
            if not flop_ok:
                detail += " [flops]"

        except Exception as e:
            status = "ERROR"
            max_diff = float("nan")
            detail = str(e)
            print(f"  ERROR transactional ({M},{N},{K}): {e}")

        results.append(("TRANSACTIONAL", category, f"({M},{N},{K})", status, detail))

    return results


def main():
    rng = np.random.default_rng(42)

    # Reset XUE counters
    try:
        kpu.reset_xue_counters()
        kpu.set_xue_enabled(True)
    except Exception:
        pass

    beh_results = run_behavioral(rng)
    trans_results = run_transactional(rng)

    # --- Summary: BEHAVIORAL ---
    print()
    print("=== BEHAVIORAL ===")
    print(f"{'Category':<14} {'Size':<18} {'Status':<6} {'Max Diff':>12}")
    print("-" * 54)
    for _, cat, size, status, diff in beh_results:
        diff_str = f"{diff:.2e}" if not np.isnan(diff) else "N/A"
        print(f"{cat:<14} {size:<18} {status:<6} {diff_str:>12}")

    # --- Summary: TRANSACTIONAL ---
    if trans_results:
        print()
        print("=== TRANSACTIONAL ===")
        print(f"{'Category':<14} {'Size':<18} {'Status':<6} {'Detail'}")
        print("-" * 70)
        for _, cat, size, status, detail in trans_results:
            print(f"{cat:<14} {size:<18} {status:<6} {detail}")

    # --- Totals ---
    all_results = [(r[3]) for r in beh_results] + [(r[3]) for r in trans_results]
    n_pass = sum(1 for s in all_results if s == "PASS")
    n_fail = sum(1 for s in all_results if s == "FAIL")
    n_err = sum(1 for s in all_results if s == "ERROR")
    total = len(all_results)

    print()
    print(f"Total: {total}  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_err}")

    return 0 if n_fail == 0 and n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
