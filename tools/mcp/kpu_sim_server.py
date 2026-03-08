#!/usr/bin/env python3
"""
KPU Simulator MCP Server

Provides structured build, test, run, and validation tools for the
kpu-sim project. Communicates with Claude Code via stdio transport.

Tools:
  - kpu_build: Build the project, return structured results
  - kpu_test: Run tests, return per-test pass/fail
  - kpu_run_demo: Run a demo, return parsed transaction log
  - kpu_validate_trace: Validate trace files against invariants
  - kpu_test_status: Quick project health check

SPDX-License-Identifier: MIT
Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
"""

import os
import sys
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from parsers import (
    parse_build_output,
    parse_ctest_output,
    parse_demo_output,
    parse_trace_validator_output,
)

# Server root is the kpu-sim project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

mcp = FastMCP("kpu-sim")


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a subprocess and capture output."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or PROJECT_ROOT,
        timeout=timeout,
    )


@mcp.tool()
def kpu_build(preset: str = "release") -> dict:
    """Build the kpu-sim project.

    Args:
        preset: CMake build preset (default: "release")

    Returns:
        Dictionary with: success (bool), errors (list of {file, line, message}),
        warnings_count (int), output_tail (last lines of build output)
    """
    try:
        result = _run(["cmake", "--build", "--preset", preset], timeout=180)
        parsed = parse_build_output(result.stdout + result.stderr, result.returncode)
        return parsed
    except subprocess.TimeoutExpired:
        return {"success": False, "errors": [{"file": "", "line": 0, "message": "Build timed out after 180 seconds"}], "warnings_count": 0, "output_tail": ""}
    except Exception as e:
        return {"success": False, "errors": [{"file": "", "line": 0, "message": str(e)}], "warnings_count": 0, "output_tail": ""}


@mcp.tool()
def kpu_test(label: str = "timing", filter: str = "", verbose: bool = False) -> dict:
    """Run kpu-sim tests and return structured results.

    Args:
        label: CTest label to filter by (default: "timing")
        filter: Additional test name filter regex (optional)
        verbose: If true, include full output for failed tests

    Returns:
        Dictionary with: total (int), passed (int), failed (int),
        results (list of {name, status, duration}),
        failures (list of {name, file, line, assertion, actual, expected})
    """
    cmd = ["ctest", "-L", label, "--output-on-failure", "--test-output-size-passed", "0"]
    if filter:
        cmd.extend(["-R", filter])

    try:
        result = _run(cmd, cwd=PROJECT_ROOT / "build", timeout=120)
        parsed = parse_ctest_output(result.stdout + result.stderr)
        return parsed
    except subprocess.TimeoutExpired:
        return {"total": 0, "passed": 0, "failed": 0, "results": [], "failures": [], "error": "Tests timed out after 120 seconds"}
    except Exception as e:
        return {"total": 0, "passed": 0, "failed": 0, "results": [], "failures": [], "error": str(e)}


@mcp.tool()
def kpu_run_demo(demo: str = "csp_pipeline_demo", args: list[str] | None = None) -> dict:
    """Run a kpu-sim demo executable and return parsed output.

    Args:
        demo: Demo executable name (default: "csp_pipeline_demo")
        args: Additional command-line arguments (optional)

    Returns:
        Dictionary with: status (str), total_cycles (int),
        summary (dict of tile counts), transaction_log (list of events)
    """
    cmd = [str(PROJECT_ROOT / "build" / "examples" / "schedule" / demo)]
    if args:
        cmd.extend(args)

    try:
        result = _run(cmd, timeout=60)
        parsed = parse_demo_output(result.stdout)
        return parsed
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "total_cycles": 0, "summary": {}, "transaction_log": [], "error": "Demo timed out after 60 seconds"}
    except FileNotFoundError:
        return {"status": "NOT_FOUND", "total_cycles": 0, "summary": {}, "transaction_log": [], "error": f"Demo executable not found: {demo}. Run kpu_build first."}
    except Exception as e:
        return {"status": "ERROR", "total_cycles": 0, "summary": {}, "transaction_log": [], "error": str(e)}


@mcp.tool()
def kpu_validate_trace(trace_file: str = "all") -> dict:
    """Validate trace files against DRAM timing invariants.

    Args:
        trace_file: Path to trace file, or "all" to validate all traces

    Returns:
        Dictionary with: status (str), traces_checked (int),
        violations (list of {invariant_id, severity, message, fix_hint, trace_file})
    """
    validator = PROJECT_ROOT / "patterns" / "memory" / "lpddr5" / "common" / "trace_validator.py"

    if not validator.exists():
        return {"status": "ERROR", "traces_checked": 0, "violations": [], "error": "Trace validator not found"}

    if trace_file == "all":
        # Find all trace files
        traces_dir = PROJECT_ROOT / "traces"
        if not traces_dir.exists():
            return {"status": "NO_TRACES", "traces_checked": 0, "violations": [], "error": "No traces/ directory found. Run a pattern first."}

        trace_files = list(traces_dir.rglob("*.json"))
        if not trace_files:
            return {"status": "NO_TRACES", "traces_checked": 0, "violations": [], "error": "No trace files found in traces/"}
    else:
        trace_path = Path(trace_file)
        if not trace_path.is_absolute():
            trace_path = PROJECT_ROOT / trace_path
        if not trace_path.exists():
            return {"status": "NOT_FOUND", "traces_checked": 0, "violations": [], "error": f"Trace file not found: {trace_file}"}
        trace_files = [trace_path]

    all_violations = []
    for tf in trace_files:
        try:
            result = _run(["python3", str(validator), str(tf), "--json"], timeout=30)
            parsed = parse_trace_validator_output(result.stdout, str(tf))
            all_violations.extend(parsed.get("violations", []))
        except Exception as e:
            all_violations.append({
                "invariant_id": "PARSE_ERROR",
                "severity": "ERROR",
                "message": str(e),
                "fix_hint": "",
                "trace_file": str(tf),
            })

    status = "PASS" if not all_violations else "FAIL"
    return {
        "status": status,
        "traces_checked": len(trace_files),
        "violations": all_violations,
    }


@mcp.tool()
def kpu_test_status() -> dict:
    """Quick project health check: build state, test results, git status.

    Returns:
        Dictionary with: build_ok (bool), tests (dict with total/passed/failed),
        failing_tests (list of names), git_status (str), last_commit (str)
    """
    results = {}

    # Check build
    try:
        build_result = _run(["cmake", "--build", "--preset", "release"], timeout=180)
        results["build_ok"] = build_result.returncode == 0
    except Exception:
        results["build_ok"] = False

    # Run tests
    try:
        test_result = _run(["ctest", "-L", "timing", "--output-on-failure", "--test-output-size-passed", "0"],
                           cwd=PROJECT_ROOT / "build", timeout=120)
        parsed = parse_ctest_output(test_result.stdout + test_result.stderr)
        results["tests"] = {"total": parsed["total"], "passed": parsed["passed"], "failed": parsed["failed"]}
        results["failing_tests"] = [f["name"] for f in parsed.get("failures", [])]
    except Exception:
        results["tests"] = {"total": 0, "passed": 0, "failed": 0}
        results["failing_tests"] = []

    # Git status
    try:
        git_status = _run(["git", "status", "--short"], timeout=10)
        results["git_status"] = git_status.stdout.strip()
    except Exception:
        results["git_status"] = "unknown"

    # Last commit
    try:
        git_log = _run(["git", "log", "--oneline", "-1"], timeout=10)
        results["last_commit"] = git_log.stdout.strip()
    except Exception:
        results["last_commit"] = "unknown"

    return results


if __name__ == "__main__":
    mcp.run(transport="stdio")
