#!/usr/bin/env python3
"""
Integration tests for kpu-sim MCP server parsers.

Run with: python3 tools/mcp/test_parsers.py

SPDX-License-Identifier: MIT
Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
"""

import sys
import os
import json

# Add the mcp directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsers import (
    parse_build_output,
    parse_ctest_output,
    parse_demo_output,
    parse_trace_validator_output,
)

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")
        if detail:
            print(f"        {detail}")


# ============================================================================
# Build Parser Tests
# ============================================================================
print("\n=== Build Parser ===")

# Successful build
result = parse_build_output("[8/8] Linking CXX executable test_foo\n", 0)
test("successful build", result["success"] is True)
test("no errors on success", len(result["errors"]) == 0)

# Failed build with GCC error
gcc_output = """/home/user/src/foo.cpp:42:10: error: 'Bar' was not declared in this scope
   42 |     Bar baz;
      |          ^~~
/home/user/src/foo.cpp:50:5: warning: unused variable 'x' [-Wunused-variable]
"""
result = parse_build_output(gcc_output, 1)
test("failed build detected", result["success"] is False)
test("error parsed", len(result["errors"]) == 1, f"got {len(result['errors'])} errors")
test("error file correct", result["errors"][0]["file"] == "/home/user/src/foo.cpp")
test("error line correct", result["errors"][0]["line"] == 42)
test("error message correct", "'Bar' was not declared" in result["errors"][0]["message"])
test("warning counted", result["warnings_count"] == 1)

# Linker error
linker_output = "undefined reference to `foo::bar()'\ncollect2: error: ld returned 1 exit status\n"
result = parse_build_output(linker_output, 1)
test("linker errors detected", len(result["errors"]) >= 1)

# Multiple errors
multi_error = """src/a.cpp:10:5: error: first error
src/b.cpp:20:5: error: second error
src/c.cpp:30:5: warning: a warning
"""
result = parse_build_output(multi_error, 1)
test("multiple errors parsed", len(result["errors"]) == 2, f"got {len(result['errors'])}")
test("multiple warnings counted", result["warnings_count"] == 1)


# ============================================================================
# CTest Parser Tests
# ============================================================================
print("\n=== CTest Parser ===")

# All passing
all_pass = """Test project /home/user/build
      Start 83: test_credit_pool
 1/3 Test #83: test_credit_pool ..................   Passed    0.00 sec
      Start 84: test_tag_cam
 2/3 Test #84: test_tag_cam ......................   Passed    0.01 sec
      Start 85: test_work_queue
 3/3 Test #85: test_work_queue ...................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 3
"""
result = parse_ctest_output(all_pass)
test("all passing total", result["total"] == 3, f"got {result['total']}")
test("all passing passed", result["passed"] == 3)
test("all passing failed", result["failed"] == 0)
test("all passing results count", len(result["results"]) == 3)

# Mixed results with ***Failed format
mixed = """Test project /home/user/build
      Start 83: test_credit_pool
 1/3 Test #83: test_credit_pool ..................   Passed    0.00 sec
      Start 84: test_tag_cam
 2/3 Test #84: test_tag_cam ......................***Failed    0.00 sec
      Start 85: test_work_queue
 3/3 Test #85: test_work_queue ...................   Passed    0.00 sec

75% tests passed, 1 tests failed out of 3
"""
result = parse_ctest_output(mixed)
test("mixed total", result["total"] == 3)
test("mixed passed", result["passed"] == 2)
test("mixed failed", result["failed"] == 1)
test("mixed results count", len(result["results"]) == 3, f"got {len(result['results'])}")
test("failed test detected", any(r["status"] == "FAILED" for r in result["results"]))
test("failed test name", result["results"][1]["name"] == "test_tag_cam" if len(result["results"]) > 1 else False)

# With Catch2 failure details
catch2_failure = """
 1/1 Test #84: test_tag_cam ......................***Failed    0.00 sec

tests/timing/test_tag_cam.cpp:52: FAILED:
  REQUIRE_FALSE( cam.insert(tile, 3, 200) )
with expansion:
  !true

75% tests passed, 1 tests failed out of 1
"""
result = parse_ctest_output(catch2_failure)
test("catch2 failure parsed", len(result["failures"]) >= 1, f"got {len(result['failures'])} failures")
if result["failures"]:
    f = result["failures"][0]
    test("catch2 file parsed", f["file"] == "tests/timing/test_tag_cam.cpp")
    test("catch2 line parsed", f["line"] == 52)
    test("catch2 assertion parsed", "REQUIRE_FALSE" in f["assertion"])

# Empty / no tests
result = parse_ctest_output("No tests were found!!!\n")
test("no tests found", result["total"] == 0)

# Catch2 standalone format
catch2_standalone = """
Randomness seeded to: 12345
===============================================================================
All tests passed (51 assertions in 12 test cases)
"""
result = parse_ctest_output(catch2_standalone)
test("catch2 standalone total", result["total"] == 12, f"got {result['total']}")
test("catch2 standalone passed", result["passed"] == 12)


# ============================================================================
# Demo Parser Tests
# ============================================================================
print("\n=== Demo Parser ===")

demo_output = """
Running simulation...

====================================================================================================
Transaction Log: Credit/Debit Flow and Tag Matching
====================================================================================================
Cycle |      Component |              Event |       Tile |    Credit Flow | TagCAM Action
----------------------------------------------------------------------------------------------------
    1 |            MC0 |         LOAD_START |   A[0,0,0] |      L3: 4->3 | -
    2 |            MC0 |         LOAD_START |   B[0,0,0] |      L3: 3->2 | -
   35 |     L3(0,0):BM |         MOVE_START |   A[0,0,0] |      L2: 4->3 | L3.match(A[0,0,0]) HIT
   35 |            MC0 |      LOAD_COMPLETE |   A[0,0,0] |              - | L3.insert(A[0,0,0])

Summary:
  Status: SUCCESS
  Total cycles: 163

  Tiles loaded:    3
  Tiles moved:     2
  Tiles fed:       2
  Tiles drained:   1
  Tiles writeback: 1
  Tiles stored:    1
"""
result = parse_demo_output(demo_output)
test("demo status", result["status"] == "SUCCESS")
test("demo cycles", result["total_cycles"] == 163)
test("demo tiles_loaded", result["summary"].get("tiles_loaded") == 3)
test("demo tiles_drained", result["summary"].get("tiles_drained") == 1)
test("demo tiles_stored", result["summary"].get("tiles_stored") == 1)
test("demo log count", len(result["transaction_log"]) == 4, f"got {len(result['transaction_log'])}")
if result["transaction_log"]:
    first = result["transaction_log"][0]
    test("demo log cycle", first["cycle"] == 1)
    test("demo log component", first["component"] == "MC0")
    test("demo log event", first["event"] == "LOAD_START")
    test("demo log tile", first["tile"] == "A[0,0,0]")
    test("demo log credit", "4->3" in first["credit_flow"])

# Empty demo output
result = parse_demo_output("")
test("empty demo status", result["status"] == "UNKNOWN")
test("empty demo cycles", result["total_cycles"] == 0)


# ============================================================================
# Trace Validator Parser Tests
# ============================================================================
print("\n=== Trace Validator Parser ===")

# JSON output
json_output = json.dumps({
    "status": "FAILED",
    "violations": [
        {
            "invariant": "INV-001",
            "severity": "ERROR",
            "message": "txn_id=0 has no data operation",
            "fix_hint": "PRECHARGE should have same txn_id as READ/WRITE"
        }
    ]
})
result = parse_trace_validator_output(json_output, "trace.json")
test("json status", result["status"] == "FAILED")
test("json violations count", len(result["violations"]) == 1)
if result["violations"]:
    v = result["violations"][0]
    test("json invariant_id", v["invariant_id"] == "INV-001")
    test("json severity", v["severity"] == "ERROR")
    test("json trace_file", v["trace_file"] == "trace.json")

# Passing JSON
pass_json = json.dumps({"status": "PASSED", "violations": []})
result = parse_trace_validator_output(pass_json, "good_trace.json")
test("passing json", result["status"] == "PASSED")
test("passing no violations", len(result["violations"]) == 0)

# Text fallback
text_output = "INV-001: txn_id=0 has no data operation\nINV-002: Invalid bank state\n"
result = parse_trace_validator_output(text_output, "trace.json")
test("text fallback violations", len(result["violations"]) == 2)


# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)
