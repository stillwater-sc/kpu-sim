#!/usr/bin/env python3
"""ResNet-on-CSP benchmark regression check.

The m2_resnet demo emits a deterministic sweep (fixed-seed synthetic weights) as
JSON via `m2_resnet --json`. This script compares a fresh run against a committed
baseline so that any code change altering ResNet's timing / utilization / compute
metrics is caught in CI.

Because the sweep is deterministic, integer metrics (cycles, ops, tiles, bytes,
MACs, stalls) must match EXACTLY; floating metrics (utilization, GFLOP/s,
arithmetic intensity, efficiencies, max_err) are compared within a small relative
tolerance to absorb JSON float formatting.

Usage:
    resnet_regression_check.py generate --binary PATH --baseline FILE
    resnet_regression_check.py check    --binary PATH --baseline FILE [--rtol R]

`check` exits non-zero (and prints a diff) on any regression.
"""
import argparse
import json
import subprocess
import sys

REL_TOL = 1e-6      # relative tolerance for float fields
ABS_TOL = 1e-9      # absolute tolerance near zero

# Keys excluded from the regression comparison. max_err is the fp validation error
# (conv/matmul reduction order and FMA use differ across compilers/platforms), so
# it is NOT bit-deterministic; correctness is enforced separately by the m2_resnet
# PASS check (max_err < 5e-3). All timing/utilization/compute metrics ARE
# deterministic (integer schedule logic + IEEE double arithmetic on integer inputs).
IGNORE_KEYS = {"max_err"}


def run_binary(binary: str) -> dict:
    """Run `<binary> --json` and parse its stdout as JSON."""
    out = subprocess.run([binary, "--json"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _floats_close(a: float, b: float, rtol: float) -> bool:
    return abs(a - b) <= max(rtol * abs(b), ABS_TOL)


def _compare(path: str, base, cur, rtol: float, out: list):
    """Recursively compare cur against base, appending mismatches to `out`."""
    if isinstance(base, dict):
        if not isinstance(cur, dict):
            out.append(f"{path}: type changed dict -> {type(cur).__name__}")
            return
        for k in base:
            if k in IGNORE_KEYS:
                continue
            if k not in cur:
                out.append(f"{path}.{k}: missing in current run")
            else:
                _compare(f"{path}.{k}", base[k], cur[k], rtol, out)
        return
    if isinstance(base, list):
        if not isinstance(cur, list) or len(cur) != len(base):
            out.append(f"{path}: list changed {base} -> {cur}")
            return
        for i, (b, c) in enumerate(zip(base, cur)):
            _compare(f"{path}[{i}]", b, c, rtol, out)
        return
    # bool is a subclass of int; check it first so True != 1 confusion is avoided.
    if isinstance(base, bool) or isinstance(cur, bool):
        if base != cur:
            out.append(f"{path}: {base} -> {cur}")
        return
    if isinstance(base, int) and isinstance(cur, int):
        if base != cur:                                   # deterministic: exact
            out.append(f"{path}: {base} -> {cur}  (integer metric must be exact)")
        return
    if isinstance(base, (int, float)) and isinstance(cur, (int, float)):
        if not _floats_close(float(cur), float(base), rtol):
            out.append(f"{path}: {base} -> {cur}  (rtol {rtol})")
        return
    if base != cur:
        out.append(f"{path}: {base!r} -> {cur!r}")


def compare(baseline: dict, current: dict, rtol: float) -> list:
    """Return a list of human-readable regression messages (empty if clean)."""
    mismatches: list = []
    base_by_name = {r["name"]: r for r in baseline.get("results", [])}
    cur_by_name = {r["name"]: r for r in current.get("results", [])}
    for name, br in base_by_name.items():
        if name not in cur_by_name:
            mismatches.append(f"result '{name}': missing in current run")
        else:
            _compare(f"result '{name}'", br, cur_by_name[name], rtol, mismatches)
    for name in cur_by_name:
        if name not in base_by_name:
            mismatches.append(f"result '{name}': new, not in baseline")
    # top-level scalar knobs (clock/peak/bw) should be stable too
    for k in ("clock_ghz", "peak_gflops", "ext_bw_gbs"):
        if k in baseline:
            _compare(k, baseline[k], current.get(k), rtol, mismatches)
    return mismatches


def main() -> int:
    ap = argparse.ArgumentParser(description="ResNet-on-CSP regression check")
    ap.add_argument("command", choices=["generate", "check"])
    ap.add_argument("--binary", required=True, help="path to the m2_resnet executable")
    ap.add_argument("--baseline", required=True, help="path to the baseline JSON")
    ap.add_argument("--rtol", type=float, default=REL_TOL)
    args = ap.parse_args()

    current = run_binary(args.binary)

    if args.command == "generate":
        with open(args.baseline, "w") as f:
            json.dump(current, f, indent=2)
            f.write("\n")
        print(f"Wrote baseline: {args.baseline}")
        return 0

    with open(args.baseline) as f:
        baseline = json.load(f)
    mismatches = compare(baseline, current, args.rtol)
    if mismatches:
        print("ResNet benchmark REGRESSION (vs committed baseline):", file=sys.stderr)
        for m in mismatches:
            print("  " + m, file=sys.stderr)
        print(f"\n{len(mismatches)} metric(s) changed. If intentional, regenerate:\n"
              f"  python3 scripts/resnet_regression_check.py generate "
              f"--binary {args.binary} --baseline {args.baseline}", file=sys.stderr)
        return 1
    print("ResNet benchmark: no regression (all metrics match the baseline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
