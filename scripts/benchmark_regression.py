#!/usr/bin/env python3
"""
KPU Benchmark Regression Testing

This script manages benchmark baselines and detects performance regressions.
Part of v0.3.5 Performance Regression Testing.

Features:
- Baseline generation and management
- Performance regression detection (>5% threshold)
- JSON/markdown/text report generation
- CI integration support

Usage:
    # Generate a new baseline
    python benchmark_regression.py generate

    # Check for regressions against baseline
    python benchmark_regression.py check

    # Update baseline with current results
    python benchmark_regression.py update

    # Show comparison report
    python benchmark_regression.py report

    # Generate markdown report for PR comments
    python benchmark_regression.py markdown
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuration
BASELINE_DIR = Path(__file__).parent.parent / "tests" / "benchmarks" / "baselines"
BENCHMARK_CLI = Path(__file__).parent.parent / "build" / "tools" / "benchmark" / "kpu-benchmark"
REGRESSION_THRESHOLD = 0.05  # 5% regression threshold


def run_benchmark() -> dict:
    """Run the benchmark CLI and return JSON results."""
    if not BENCHMARK_CLI.exists():
        print(f"Error: Benchmark CLI not found at {BENCHMARK_CLI}")
        print("Build the project first: cmake --build build --target benchmark")
        sys.exit(1)

    result = subprocess.run(
        [str(BENCHMARK_CLI), "sweep", "--quick", "-f", "json"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        sys.exit(1)

    return json.loads(result.stdout)


def load_baseline() -> Optional[dict]:
    """Load the current baseline if it exists."""
    baseline_file = BASELINE_DIR / "baseline.json"
    if not baseline_file.exists():
        return None

    with open(baseline_file) as f:
        return json.load(f)


def save_baseline(data: dict):
    """Save benchmark results as the new baseline."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_file = BASELINE_DIR / "baseline.json"

    # Add metadata
    data["_metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "version": "0.3.5",
        "regression_threshold": REGRESSION_THRESHOLD
    }

    with open(baseline_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved baseline to {baseline_file}")


def compare_results(baseline: dict, current: dict) -> list:
    """Compare current results against baseline and return regressions."""
    regressions = []

    # Build lookup maps by config
    baseline_by_config = {r["config"]: r for r in baseline.get("results", [])}
    current_by_config = {r["config"]: r for r in current.get("results", [])}

    for config, current_result in current_by_config.items():
        if config not in baseline_by_config:
            continue

        baseline_result = baseline_by_config[config]

        # Compare GFLOPS
        baseline_gflops = baseline_result.get("compute", {}).get("gflops", 0)
        current_gflops = current_result.get("compute", {}).get("gflops", 0)

        if baseline_gflops > 0:
            change = (current_gflops - baseline_gflops) / baseline_gflops

            if change < -REGRESSION_THRESHOLD:
                regressions.append({
                    "config": config,
                    "metric": "gflops",
                    "baseline": baseline_gflops,
                    "current": current_gflops,
                    "change_pct": change * 100
                })

        # Compare efficiency
        baseline_eff = baseline_result.get("compute", {}).get("efficiency", 0)
        current_eff = current_result.get("compute", {}).get("efficiency", 0)

        if baseline_eff > 0:
            eff_change = (current_eff - baseline_eff) / baseline_eff

            if eff_change < -REGRESSION_THRESHOLD:
                regressions.append({
                    "config": config,
                    "metric": "efficiency",
                    "baseline": baseline_eff,
                    "current": current_eff,
                    "change_pct": eff_change * 100
                })

    return regressions


def print_report(baseline: dict, current: dict, regressions: list):
    """Print a comparison report."""
    print("\n" + "=" * 80)
    print("KPU BENCHMARK REGRESSION REPORT")
    print("=" * 80)

    baseline_meta = baseline.get("_metadata", {})
    print(f"\nBaseline generated: {baseline_meta.get('generated_at', 'unknown')}")
    print(f"Regression threshold: {REGRESSION_THRESHOLD * 100}%")

    # Summary table
    print("\n" + "-" * 80)
    print(f"{'Config':<20} {'Metric':<12} {'Baseline':<12} {'Current':<12} {'Change':<10} {'Status'}")
    print("-" * 80)

    baseline_by_config = {r["config"]: r for r in baseline.get("results", [])}
    current_by_config = {r["config"]: r for r in current.get("results", [])}

    for config in sorted(current_by_config.keys()):
        if config not in baseline_by_config:
            continue

        br = baseline_by_config[config]
        cr = current_by_config[config]

        # GFLOPS comparison
        bg = br.get("compute", {}).get("gflops", 0)
        cg = cr.get("compute", {}).get("gflops", 0)
        change_g = ((cg - bg) / bg * 100) if bg > 0 else 0
        status_g = "REGRESS" if change_g < -REGRESSION_THRESHOLD * 100 else "OK" if change_g >= 0 else "WARN"

        print(f"{config:<20} {'GFLOPS':<12} {bg:<12.2f} {cg:<12.2f} {change_g:>+8.1f}% {status_g}")

        # Efficiency comparison
        be = br.get("compute", {}).get("efficiency", 0)
        ce = cr.get("compute", {}).get("efficiency", 0)
        change_e = ((ce - be) / be * 100) if be > 0 else 0
        status_e = "REGRESS" if change_e < -REGRESSION_THRESHOLD * 100 else "OK" if change_e >= 0 else "WARN"

        print(f"{'':<20} {'Efficiency':<12} {be*100:<11.1f}% {ce*100:<11.1f}% {change_e:>+8.1f}% {status_e}")

    print("-" * 80)

    # Regression summary
    if regressions:
        print(f"\nREGRESSIONS DETECTED: {len(regressions)}")
        for r in regressions:
            print(f"  - {r['config']}: {r['metric']} dropped {abs(r['change_pct']):.1f}%")
        return 1
    else:
        print("\nNo regressions detected.")
        return 0


def cmd_generate(args):
    """Generate a new baseline."""
    print("Running benchmarks to generate baseline...")
    results = run_benchmark()
    save_baseline(results)
    print(f"Generated baseline with {len(results.get('results', []))} benchmark results.")


def cmd_check(args):
    """Check for regressions against baseline."""
    baseline = load_baseline()
    if baseline is None:
        print("No baseline found. Generate one first: python benchmark_regression.py generate")
        sys.exit(1)

    print("Running benchmarks to check for regressions...")
    current = run_benchmark()

    regressions = compare_results(baseline, current)

    if regressions:
        print(f"\nREGRESSIONS DETECTED: {len(regressions)}")
        for r in regressions:
            print(f"  - {r['config']}: {r['metric']} "
                  f"baseline={r['baseline']:.2f} current={r['current']:.2f} "
                  f"change={r['change_pct']:.1f}%")
        sys.exit(1)
    else:
        print("\nNo regressions detected.")
        sys.exit(0)


def cmd_update(args):
    """Update baseline with current results."""
    print("Running benchmarks to update baseline...")
    results = run_benchmark()

    # Archive old baseline if it exists
    baseline_file = BASELINE_DIR / "baseline.json"
    if baseline_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = BASELINE_DIR / f"baseline_{timestamp}.json"
        baseline_file.rename(archive_file)
        print(f"Archived previous baseline to {archive_file}")

    save_baseline(results)
    print(f"Updated baseline with {len(results.get('results', []))} benchmark results.")


def cmd_report(args):
    """Show comparison report."""
    baseline = load_baseline()
    if baseline is None:
        print("No baseline found. Generate one first: python benchmark_regression.py generate")
        sys.exit(1)

    print("Running benchmarks for comparison report...")
    current = run_benchmark()

    regressions = compare_results(baseline, current)
    exit_code = print_report(baseline, current, regressions)
    sys.exit(exit_code)


def generate_markdown_report(baseline: dict, current: dict, regressions: list) -> str:
    """Generate a markdown-formatted report for PR comments."""
    lines = []
    lines.append("## KPU Benchmark Regression Report")
    lines.append("")

    baseline_meta = baseline.get("_metadata", {})
    lines.append(f"**Baseline:** {baseline_meta.get('generated_at', 'unknown')}")
    lines.append(f"**Threshold:** {REGRESSION_THRESHOLD * 100}%")
    lines.append("")

    # Summary table
    lines.append("| Config | Metric | Baseline | Current | Change | Status |")
    lines.append("|--------|--------|----------|---------|--------|--------|")

    baseline_by_config = {r["config"]: r for r in baseline.get("results", [])}
    current_by_config = {r["config"]: r for r in current.get("results", [])}

    for config in sorted(current_by_config.keys()):
        if config not in baseline_by_config:
            continue

        br = baseline_by_config[config]
        cr = current_by_config[config]

        # GFLOPS comparison
        bg = br.get("compute", {}).get("gflops", 0)
        cg = cr.get("compute", {}).get("gflops", 0)
        change_g = ((cg - bg) / bg * 100) if bg > 0 else 0
        status_g = "REGRESS" if change_g < -REGRESSION_THRESHOLD * 100 else "OK"

        lines.append(f"| {config} | GFLOPS | {bg:.2f} | {cg:.2f} | {change_g:+.1f}% | {status_g} |")

        # Efficiency comparison
        be = br.get("compute", {}).get("efficiency", 0)
        ce = cr.get("compute", {}).get("efficiency", 0)
        change_e = ((ce - be) / be * 100) if be > 0 else 0
        status_e = "REGRESS" if change_e < -REGRESSION_THRESHOLD * 100 else "OK"

        lines.append(f"| | Efficiency | {be*100:.1f}% | {ce*100:.1f}% | {change_e:+.1f}% | {status_e} |")

    lines.append("")

    # Summary
    if regressions:
        lines.append(f"> **Warning:** {len(regressions)} regression(s) detected")
        for r in regressions:
            lines.append(f"> - `{r['config']}`: {r['metric']} dropped {abs(r['change_pct']):.1f}%")
    else:
        lines.append("> **Status:** No performance regressions detected")

    return "\n".join(lines)


def cmd_markdown(args):
    """Generate markdown report for PR comments."""
    baseline = load_baseline()
    if baseline is None:
        print("<!-- No baseline found -->")
        print("## KPU Benchmark Regression Report")
        print("")
        print("> **Note:** No baseline found. Run `python benchmark_regression.py generate` first.")
        sys.exit(0)

    current = run_benchmark()
    regressions = compare_results(baseline, current)
    print(generate_markdown_report(baseline, current, regressions))
    sys.exit(1 if regressions else 0)


def cmd_json_report(args):
    """Generate JSON report for CI integration."""
    baseline = load_baseline()
    if baseline is None:
        print(json.dumps({"error": "No baseline found", "regressions": []}))
        sys.exit(1)

    current = run_benchmark()
    regressions = compare_results(baseline, current)

    report = {
        "version": "0.3.5",
        "baseline_date": baseline.get("_metadata", {}).get("generated_at"),
        "threshold_percent": REGRESSION_THRESHOLD * 100,
        "regression_count": len(regressions),
        "regressions": regressions,
        "status": "FAIL" if regressions else "PASS"
    }
    print(json.dumps(report, indent=2))
    sys.exit(1 if regressions else 0)


def main():
    parser = argparse.ArgumentParser(
        description="KPU Benchmark Regression Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new baseline")
    gen_parser.set_defaults(func=cmd_generate)

    # check command
    check_parser = subparsers.add_parser("check", help="Check for regressions")
    check_parser.set_defaults(func=cmd_check)

    # update command
    update_parser = subparsers.add_parser("update", help="Update baseline")
    update_parser.set_defaults(func=cmd_update)

    # report command
    report_parser = subparsers.add_parser("report", help="Show comparison report")
    report_parser.set_defaults(func=cmd_report)

    # markdown command
    md_parser = subparsers.add_parser("markdown", help="Generate markdown report for PR comments")
    md_parser.set_defaults(func=cmd_markdown)

    # json-report command
    json_parser = subparsers.add_parser("json-report", help="Generate JSON report for CI")
    json_parser.set_defaults(func=cmd_json_report)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
