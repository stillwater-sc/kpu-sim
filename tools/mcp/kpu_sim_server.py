#!/usr/bin/env python3
"""
KPU Simulator MCP Server

Provides structured build, test, run, and validation tools for the
kpu-sim project. Communicates with Claude Code via stdio transport.

Tools:
  - kpu_build: Build the project, return structured results
  - kpu_test: Run tests, return per-test pass/fail
  - kpu_run_demo: Run a demo, return parsed transaction log
  - kpu_validate_trace: Check .mflow records against the memory-side invariants (mflow_check.py)
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
    parse_mflow_check_output,
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
def kpu_validate_trace(bundle: str = "all") -> dict:
    """Check memory-flow records (.mflow bundles) with tools/trace/mflow_check.py.

    Args:
        bundle: Path to a .mflow bundle (a directory), or "all" for the LPDDR5 patterns' records
                (build/patterns/) and every bundle under traces/ and runs/ (the self-tests'
                deliberately broken copies under build/tests/ are not runs)

    Returns:
        Dictionary with: status (PASS, FAIL, NO_RECORDS, NOT_FOUND or ERROR), bundles_checked
        (int), violations (list of {invariant_id, message, bundle}) and unreadable (bundles the
        checker could not read, with its message)
    """
    checker = PROJECT_ROOT / "tools" / "trace" / "mflow_check.py"
    if not checker.exists():
        return {"status": "ERROR", "bundles_checked": 0, "violations": [], "unreadable": [],
                "error": "tools/trace/mflow_check.py not found"}

    if bundle == "all":
        bundles = sorted(p for d in ("build/patterns", "traces", "runs") if (PROJECT_ROOT / d).is_dir()
                         for p in (PROJECT_ROOT / d).rglob("*.mflow") if p.is_dir())
        if not bundles:
            return {"status": "NO_RECORDS", "bundles_checked": 0, "violations": [], "unreadable": [],
                    "error": "No .mflow bundles found. Run the LPDDR5 patterns (ctest -R lpddr5_) first."}
    else:
        path = Path(bundle)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_dir():
            return {"status": "NOT_FOUND", "bundles_checked": 0, "violations": [], "unreadable": [],
                    "error": f"Bundle not found: {bundle}"}
        bundles = [path]

    violations, unreadable = [], []
    for b in bundles:
        try:
            result = _run(["python3", str(checker), str(b), "--json"], timeout=60)
            parsed = parse_mflow_check_output(result.stdout, result.returncode, str(b))
        except Exception as e:
            unreadable.append({"bundle": str(b), "error": str(e)})
            continue
        if parsed["status"] in ("UNREADABLE", "UNKNOWN"):
            unreadable.append({"bundle": str(b), "error": (result.stderr or "").strip() or parsed["status"]})
        violations.extend(parsed["violations"])

    return {
        "status": "PASS" if not violations and not unreadable else "FAIL",
        "bundles_checked": len(bundles),
        "violations": violations,
        "unreadable": unreadable,
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
