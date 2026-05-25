# KPU-SIM MCP Server Guide

## Why: The Problem with Raw CLI Output

When Claude Code runs build and test commands, it receives raw terminal output —
hundreds of lines of CMake progress, compiler warnings, CTest formatting, and
ANSI control sequences. Extracting the actual signal (which test failed, on what
line, with what assertion) requires parsing that output every time, wasting
context window and introducing errors.

The MCP (Model Context Protocol) server solves this by sitting between Claude Code
and the project's CLI tools. It runs the same commands but returns **structured
dictionaries** instead of raw text. Claude Code gets exactly the information it
needs — file paths, line numbers, assertion details, credit flows — without
spending tokens parsing terminal output.

This matters most for iterative development loops. When fixing a failing test,
Claude Code needs to:
1. Know which test failed
2. See the exact file and line
3. Read the assertion and actual-vs-expected values
4. Make a fix
5. Re-run and verify

Without the MCP server, steps 1-3 require reading and parsing potentially
thousands of lines. With it, the answer arrives as a compact JSON structure.

## What: Five Structured Tools

The server provides five tools, each wrapping a project CLI operation:

| Tool | Wraps | Returns |
|------|-------|---------|
| `kpu_build` | `cmake --build --preset <preset>` | `{success, errors[{file, line, message}], warnings_count}` |
| `kpu_test` | `ctest -L <label>` | `{total, passed, failed, results[{name, status}], failures[{file, line, assertion, actual, expected}]}` |
| `kpu_run_demo` | Demo executables | `{status, total_cycles, summary{tile_counts}, transaction_log[{cycle, component, event, tile, credit_flow}]}` |
| `kpu_validate_trace` | `trace_validator.py` | `{status, traces_checked, violations[{invariant_id, severity, message, fix_hint}]}` |
| `kpu_test_status` | Build + test + git | `{build_ok, tests{total, passed, failed}, failing_tests[], git_status, last_commit}` |

### Architecture

```
Claude Code  ←── stdio/JSON-RPC ──→  kpu_sim_server.py
                                          │
                                     parsers.py
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                           cmake       ctest    trace_validator.py
```

- **Transport**: stdio (JSON-RPC 2.0 over stdin/stdout)
- **Framework**: FastMCP (`mcp[cli]>=1.2.0`)
- **Registration**: `.mcp.json` at project root
- **Parsers**: `tools/mcp/parsers.py` — four regex-based parsers with 49 integration tests

## How: Configuration and Activation

The server is registered in `.mcp.json`:

```json
{
  "mcpServers": {
    "kpu-sim": {
      "type": "stdio",
      "command": "python3",
      "args": ["tools/mcp/kpu_sim_server.py"]
    }
  }
}
```

Claude Code reads this file on startup and launches the server process
automatically. No manual activation is needed after the initial setup.

**Dependency**: `pip install "mcp[cli]>=1.2.0"` (or `python3 -m pip install`).

---

## Example 1: Fix a Failing Test

**Scenario**: You've modified `tag_cam.hpp` and need to check if tests still pass.

**Without MCP server** — Claude Code runs `ctest`, receives ~80 lines of raw output,
scans for "Failed", extracts file/line from Catch2 output, reads the file.

**With MCP server** — Claude Code calls `kpu_test`:

```
Tool call: kpu_test(label="timing", filter="test_tag_cam")
```

Response:

```json
{
  "total": 1,
  "passed": 0,
  "failed": 1,
  "results": [
    {"name": "test_tag_cam", "status": "FAILED", "duration": "0.00 sec"}
  ],
  "failures": [
    {
      "name": "test_tag_cam",
      "file": "tests/timing/test_tag_cam.cpp",
      "line": 52,
      "assertion": "REQUIRE_FALSE( cam.insert(tile, 3, 200) )",
      "actual": "!true",
      "expected": ""
    }
  ]
}
```

Claude Code immediately knows:
- The test `test_tag_cam` failed at line 52
- The assertion `REQUIRE_FALSE(cam.insert(...))` got `true` instead of `false`
- It should read `test_tag_cam.cpp:52` and `tag_cam.hpp` to understand the fix

This turns a multi-step parse-and-scan into a single structured lookup.

---

## Example 2: Validate a Pipeline Change

**Scenario**: You've changed the credit flow logic in the concurrent timing executor
and want to verify the CSP pipeline demo still produces correct results.

```
Tool call: kpu_run_demo(demo="csp_pipeline_demo")
```

Response:

```json
{
  "status": "SUCCESS",
  "total_cycles": 163,
  "summary": {
    "tiles_loaded": 3,
    "tiles_moved": 2,
    "tiles_fed": 2,
    "tiles_drained": 1,
    "tiles_writeback": 1,
    "tiles_stored": 1
  },
  "transaction_log": [
    {"cycle": 1,  "component": "MC0",        "event": "LOAD_START",    "tile": "A[0,0,0]", "credit_flow": "L3: 4->3", "tagcam_action": "-"},
    {"cycle": 2,  "component": "MC0",        "event": "LOAD_START",    "tile": "B[0,0,0]", "credit_flow": "L3: 3->2", "tagcam_action": "-"},
    {"cycle": 35, "component": "L3(0,0):BM", "event": "MOVE_START",   "tile": "A[0,0,0]", "credit_flow": "L2: 4->3", "tagcam_action": "L3.match(A[0,0,0]) HIT"},
    {"cycle": 35, "component": "MC0",        "event": "LOAD_COMPLETE", "tile": "A[0,0,0]", "credit_flow": "-",        "tagcam_action": "L3.insert(A[0,0,0])"}
  ]
}
```

Claude Code can now programmatically verify:
- Status is SUCCESS (pipeline completed without deadlock)
- All tile counts are non-zero (data flowed through every stage)
- Credit decrements are monotonic (no double-release)
- TagCAM actions show correct insert/match sequencing

If the status were FAIL or tile counts were zero, Claude Code would know exactly
which stage stalled by examining the transaction log — without parsing any text.

---

## Example 3: Pre-Commit Health Check

**Scenario**: Before committing a batch of changes, you want a quick confirmation
that the project is healthy.

```
Tool call: kpu_test_status()
```

Response:

```json
{
  "build_ok": true,
  "tests": {"total": 12, "passed": 9, "failed": 3},
  "failing_tests": ["test_tag_cam", "test_block_mover_process", "test_streamer_process"],
  "git_status": "M include/sw/kpu/timing/dma_engine_process.hpp\nM tests/timing/test_component_integration.cpp",
  "last_commit": "57f7785 Add MCP server, project skills, and productivity tooling"
}
```

Claude Code sees at a glance:
- Build is green
- 3 tests fail (all pre-existing, known issues)
- Two files are modified and uncommitted
- Last commit context for the commit message

This single call replaces three separate commands (`cmake --build`, `ctest`, `git status`)
and their combined output parsing. It is particularly useful in the `/build-test` and
`/test-status` project skills, which can delegate to this tool instead of running
raw shell commands.

---

## File Locations

| File | Purpose |
|------|---------|
| `tools/mcp/kpu_sim_server.py` | MCP server (5 tools, FastMCP) |
| `tools/mcp/parsers.py` | Output parsers (build, ctest, demo, trace) |
| `tools/mcp/test_parsers.py` | Parser integration tests (49 tests) |
| `tools/mcp/requirements.txt` | Python dependencies |
| `.mcp.json` | Server registration for Claude Code |
