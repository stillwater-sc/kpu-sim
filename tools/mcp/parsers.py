"""
Output parsers for kpu-sim build tools, test runners, and demos.

Each parser takes raw text output and returns structured dictionaries.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
"""

import json
import re


def parse_build_output(output: str, returncode: int) -> dict:
    """Parse cmake build output into structured results.

    Returns:
        {
            "success": bool,
            "errors": [{"file": str, "line": int, "message": str}],
            "warnings_count": int,
            "output_tail": str  (last 20 lines)
        }
    """
    errors = []
    warnings_count = 0

    for line in output.splitlines():
        # Match GCC/Clang error format: file.cpp:123:45: error: message
        m = re.match(r"(.+?):(\d+):\d*:?\s*error:\s*(.+)", line)
        if m:
            errors.append({
                "file": m.group(1).strip(),
                "line": int(m.group(2)),
                "message": m.group(3).strip(),
            })

        # Count warnings
        if re.search(r":\d+:\d*:?\s*warning:", line):
            warnings_count += 1

        # Match linker errors
        if "undefined reference" in line.lower() or "ld returned" in line.lower():
            errors.append({
                "file": "",
                "line": 0,
                "message": line.strip(),
            })

    # Last 20 lines of output for context
    lines = output.strip().splitlines()
    output_tail = "\n".join(lines[-20:]) if lines else ""

    return {
        "success": returncode == 0,
        "errors": errors,
        "warnings_count": warnings_count,
        "output_tail": output_tail,
    }


def parse_ctest_output(output: str) -> dict:
    """Parse ctest output into structured test results.

    Returns:
        {
            "total": int,
            "passed": int,
            "failed": int,
            "results": [{"name": str, "status": str, "duration": str}],
            "failures": [{"name": str, "file": str, "line": int,
                          "assertion": str, "actual": str, "expected": str}]
        }
    """
    results = []
    failures = []

    # Parse individual test results
    # Passed: "  1/12 Test #83: test_credit_pool ..................   Passed    0.00 sec"
    # Failed: "  2/12 Test #84: test_tag_cam ......................***Failed    0.00 sec"
    for m in re.finditer(
        r"\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+[.*]+\s*(\*\*\*Failed|\*\*\*Not Run|Passed|Failed)\s+(\S+\s+sec)",
        output
    ):
        name = m.group(1)
        status = "PASSED" if "Passed" in m.group(2) else "FAILED"
        duration = m.group(3).strip()
        results.append({"name": name, "status": status, "duration": duration})

    # Parse summary line: "75% tests passed, 3 tests failed out of 12"
    total = 0
    passed = 0
    failed = 0
    summary_m = re.search(r"(\d+)%\s+tests\s+passed,\s+(\d+)\s+tests?\s+failed\s+out\s+of\s+(\d+)", output)
    if summary_m:
        failed = int(summary_m.group(2))
        total = int(summary_m.group(3))
        passed = total - failed
    elif re.search(r"All tests passed", output):
        # "All tests passed (51 assertions in 12 test cases)"
        all_m = re.search(r"(\d+)\s+assertions?\s+in\s+(\d+)\s+test\s+cases?", output)
        if all_m:
            total = int(all_m.group(2))
            passed = total
        elif results:
            total = len(results)
            passed = total
    elif not results:
        # Try "test cases: N | M passed | K failed" (Catch2 format)
        catch2_m = re.search(r"test cases:\s+(\d+)\s*\|\s*(\d+)\s+passed\s*\|\s*(\d+)\s+failed", output)
        if catch2_m:
            total = int(catch2_m.group(1))
            passed = int(catch2_m.group(2))
            failed = int(catch2_m.group(3))

    # Parse Catch2 failure details
    # Format: "file.cpp:52: FAILED:\n  REQUIRE_FALSE( expr )\nwith expansion:\n  !true"
    failure_blocks = re.finditer(
        r"([\w/._-]+\.cpp):(\d+):\s*FAILED:\s*\n\s*(REQUIRE\S*\(.*?\))\s*\nwith expansion:\s*\n\s*(.+?)(?:\n\n|\n-{3,}|\Z)",
        output,
        re.DOTALL,
    )
    seen_failures = set()
    for fb in failure_blocks:
        file_path = fb.group(1)
        line = int(fb.group(2))
        assertion = fb.group(3).strip()
        expansion = fb.group(4).strip()

        # Parse "actual == expected" or "!actual" from expansion
        actual = expansion
        expected = ""
        eq_m = re.match(r"(.+?)\s*==\s*(.+)", expansion)
        if eq_m:
            actual = eq_m.group(1).strip()
            expected = eq_m.group(2).strip()

        # Find which test this belongs to
        test_name = ""
        for r in results:
            if r["status"] == "FAILED" and r["name"] not in seen_failures:
                test_name = r["name"]
                seen_failures.add(test_name)
                break

        failures.append({
            "name": test_name,
            "file": file_path,
            "line": line,
            "assertion": assertion,
            "actual": actual,
            "expected": expected,
        })

    # If we have failed results but no parsed failure details, add basic entries
    if not failures:
        for r in results:
            if r["status"] == "FAILED":
                failures.append({
                    "name": r["name"],
                    "file": "",
                    "line": 0,
                    "assertion": "",
                    "actual": "",
                    "expected": "",
                })

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "results": results,
        "failures": failures,
    }


def parse_demo_output(output: str) -> dict:
    """Parse CSP pipeline demo output into structured results.

    Returns:
        {
            "status": str,
            "total_cycles": int,
            "summary": {"tiles_loaded": int, "tiles_moved": int, ...},
            "transaction_log": [{"cycle": int, "component": str, "event": str,
                                 "tile": str, "credit_flow": str, "tagcam_action": str}]
        }
    """
    status = "UNKNOWN"
    total_cycles = 0
    summary = {}
    transaction_log = []

    # Parse status
    if "Status: SUCCESS" in output:
        status = "SUCCESS"
    elif "Status: FAIL" in output or "Status: TIMEOUT" in output:
        status = "FAIL"

    # Parse total cycles
    m = re.search(r"Total cycles:\s*(\d+)", output)
    if m:
        total_cycles = int(m.group(1))

    # Parse summary counts
    for key, pattern in [
        ("tiles_loaded", r"Tiles loaded:\s*(\d+)"),
        ("tiles_moved", r"Tiles moved:\s*(\d+)"),
        ("tiles_fed", r"Tiles fed:\s*(\d+)"),
        ("tiles_drained", r"Tiles drained:\s*(\d+)"),
        ("tiles_writeback", r"Tiles writeback:\s*(\d+)"),
        ("tiles_stored", r"Tiles stored:\s*(\d+)"),
    ]:
        m = re.search(pattern, output)
        if m:
            summary[key] = int(m.group(1))

    # Parse transaction log lines
    # Format: "    1 |            MC0 |         LOAD_START |   A[0,0,0] |      L3: 4->3 | -"
    for m in re.finditer(
        r"\s*(\d+)\s*\|\s*(\S+(?:\s*\S+)*?)\s*\|\s*(\S+(?:\s*\S+)*?)\s*\|\s*(\S+(?:\[[\d,]+\])?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$",
        output,
        re.MULTILINE,
    ):
        transaction_log.append({
            "cycle": int(m.group(1)),
            "component": m.group(2).strip(),
            "event": m.group(3).strip(),
            "tile": m.group(4).strip(),
            "credit_flow": m.group(5).strip(),
            "tagcam_action": m.group(6).strip(),
        })

    return {
        "status": status,
        "total_cycles": total_cycles,
        "summary": summary,
        "transaction_log": transaction_log,
    }


def parse_trace_validator_output(output: str, trace_file: str) -> dict:
    """Parse trace validator JSON output.

    Returns:
        {
            "status": str,
            "violations": [{"invariant_id": str, "severity": str,
                           "message": str, "fix_hint": str, "trace_file": str}]
        }
    """
    # Try JSON output first
    try:
        data = json.loads(output)
        violations = []
        for v in data.get("violations", []):
            violations.append({
                "invariant_id": v.get("invariant", "UNKNOWN"),
                "severity": v.get("severity", "ERROR"),
                "message": v.get("message", ""),
                "fix_hint": v.get("fix_hint", ""),
                "trace_file": trace_file,
            })
        return {
            "status": data.get("status", "UNKNOWN"),
            "violations": violations,
        }
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to text parsing — scan for INV-NNN patterns regardless
    violations = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        inv_m = re.match(r"(INV-\d+):\s*(.+)", line)
        if inv_m:
            violations.append({
                "invariant_id": inv_m.group(1),
                "severity": "ERROR",
                "message": inv_m.group(2),
                "fix_hint": "",
                "trace_file": trace_file,
            })

    status = "FAIL" if violations else ("PASS" if "PASSED" in output or "pass" in output.lower() else "UNKNOWN")
    return {"status": status, "violations": violations}
