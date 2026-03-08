# Hooks, Agents, and MCP Server Design for kpu-sim

**Date:** 2026-03-07
**Status:** Design
**Purpose:** Plan for leveraging Claude Code hooks, subagents, and custom MCP
servers to create a productive multi-fidelity architecture, simulation, and
architecture verification environment.

---

## 1. Hooks: Deterministic Guards

Hooks are shell commands that run automatically on Claude Code events.
They can't reason about code, but they can grep, count, and block.

### Hook 1: Terminology Guard (Post-Edit)

**Trigger:** After every Edit to files in `timing/` directories.
**Action:** Scan for forbidden cache terminology.
**Why:** The #1 architectural violation in this codebase is accidentally
introducing cache semantics.

```json
{
  "hooks": {
    "postToolCall": [
      {
        "matcher": "Edit",
        "command": "bash -c 'if echo \"$CLAUDE_FILE_PATH\" | grep -q \"timing/\"; then VIOLATIONS=$(grep -n -i \"\\bcache\\b\\|\\bevict\\b\\|\\bLRU\\b\\|\\brefetch\\b\" \"$CLAUDE_FILE_PATH\" 2>/dev/null | grep -iv \"tag_cam\\|row_hit\\|row_miss\\|cache_model\\|deprecated\"); if [ -n \"$VIOLATIONS\" ]; then echo \"WARNING: Possible cache terminology in timing code:\"; echo \"$VIOLATIONS\"; echo \"Use buffer/credit semantics instead.\"; fi; fi'"
      }
    ]
  }
}
```

### Hook 2: Credit Balance Alert (Post-Bash)

**Trigger:** After any test execution.
**Action:** Scan output for credit lifecycle errors.
**Why:** CreditPool double-release and underflow are recurring bugs.

```json
{
  "hooks": {
    "postToolCall": [
      {
        "matcher": "Bash",
        "command": "bash -c 'if echo \"$CLAUDE_TOOL_OUTPUT\" | grep -q \"pool is already full\\|credit.*negative\\|underflow\"; then echo \"ALERT: Credit lifecycle error detected in output.\"; fi'"
      }
    ]
  }
}
```

### Hook 3: Session Log Reminder (User Prompt)

**Trigger:** When user types commit/push/done/finish.
**Action:** Remind about session log requirement.

```json
{
  "hooks": {
    "userPromptSubmit": [
      {
        "matcher": "commit|push|done|finish|wrapup",
        "command": "bash -c 'echo \"Reminder: Create session log in docs/sessions/ if not done yet.\"'"
      }
    ]
  }
}
```

### Limitations of Hooks

- Run shell commands only — no AI reasoning
- Best for pattern matching, not semantic analysis
- Can block tool calls (preToolCall) but should be used sparingly
- Cannot modify Claude Code's behavior, only provide warnings

---

## 2. Subagents: Parallel Intelligence

Subagents are spawned via the Agent tool. They run in their own context window,
can read files and execute commands, and return results to the main thread.

### When to Use Subagents

| Situation | Main Thread | Subagent |
|-----------|-------------|----------|
| Fix a single bug | Yes | No — overhead not worth it |
| Research architecture question | No | Yes — may read many files |
| Fix 3+ independent test failures | No | Yes — parallelize |
| Validate all traces | No | Yes — parallel validation |
| Compare two design approaches | No | Yes — explore independently |
| Explore codebase for integration points | No | Yes (Explore type) |

### Pattern 1: Parallel Test Fix

When multiple tests fail after an architectural change:

```
Main: "Fix all failing timing tests"
  -> Agent 1 (type: code): Fix test_tag_cam (TagCAM duplicate semantics)
  -> Agent 2 (type: code): Fix test_block_mover_process (name formatting)
  -> Agent 3 (type: code): Fix test_streamer_process (name formatting)
Main: Collect results, rebuild, verify no regressions
```

**Why this works:** Each test failure is independent. Agents read the test
file, the implementation, diagnose, and propose a fix. Main thread applies
all fixes at once and does a single verification build.

### Pattern 2: Architecture Audit

Spawn agents to check different aspects in parallel:

```
Main: "Audit the timing code for correctness"
  -> Agent 1: Check credit flow (acquire/release balance in all processes)
  -> Agent 2: Check tag operations (insert/match/invalidate consistency)
  -> Agent 3: Check terminology (no cache semantics in timing code)
  -> Agent 4: Check tick ordering (MC before DMA before BM before STR)
Main: Synthesize findings into single report
```

### Pattern 3: Explore-then-Implement

Use an Explore subagent before implementing broad tasks:

```
Main: "Add NoC routing to the timing model"
  -> Explore Agent: Read existing NoC code (include/sw/kpu/noc/),
     CSP process patterns, ConcurrentTimingExecutor integration points,
     existing NoC tests, and report:
     - What exists already
     - Integration points
     - Suggested approach
     - Potential conflicts
Main: (with explore results) Implement with full context
```

**Why this works:** Exploration can read dozens of files and burn through
context. By isolating it in a subagent, the main thread stays clean and
focused on implementation.

### Pattern 4: Design Review Agent

Before implementing a plan, validate it against the architecture:

```
Main: User provides plan document
  -> Review Agent: Read the plan, CLAUDE.md, relevant source files.
     Check for:
     - Conflicts with credit-based dataflow model
     - Missing invariants
     - Incomplete test coverage
     - Inconsistencies with existing patterns
     Report: List of issues to address before implementation
Main: Present review to user, iterate on plan
```

### Pattern 5: Fidelity Cross-Check

Verify that behavioral and transactional models agree:

```
Main: "Verify matmul produces same results at all fidelity levels"
  -> Agent 1: Run behavioral matmul, capture output values
  -> Agent 2: Run transactional matmul, capture timing + values
  -> Agent 3: Compare results, flag discrepancies
Main: Report fidelity consistency
```

---

## 3. Custom MCP Servers

MCP (Model Context Protocol) servers extend Claude Code with structured tools
that connect to external services or wrap local functionality.

### Architecture

```
Claude Code  <-->  MCP Protocol  <-->  MCP Server  <-->  Service
  (LLM)           (JSON-RPC)         (local process)    (simulator, GitHub, etc.)
```

### Why Custom MCP > Bash Commands

| Aspect | Bash Command | MCP Tool |
|--------|-------------|----------|
| Output | Raw text to parse | Structured JSON |
| Errors | Exit codes + stderr | Typed error objects |
| Discovery | Must know command | Tool list with descriptions |
| Composition | Pipe/parse | Direct field access |
| Documentation | Man pages | Schema + descriptions |

Example: Running a test via bash returns text I must parse:
```
3 tests passed, 2 failed
FAILED: test_tag_cam - assertion at line 52
```

Via MCP, the same operation returns:
```json
{
  "total": 5,
  "passed": 3,
  "failed": 2,
  "failures": [
    {
      "test": "test_tag_cam",
      "file": "tests/timing/test_tag_cam.cpp",
      "line": 52,
      "assertion": "REQUIRE_FALSE(cam.insert(tile, 3, 200))",
      "actual": "true",
      "expected": "false"
    }
  ]
}
```

I can immediately reason about this without text parsing.

---

### MCP Server 1: `kpu-sim-server` (Build/Test/Run)

**Priority:** HIGH — used every session, multiple times per session.
**Implementation:** Python or TypeScript wrapping cmake/ctest/demo executables.

#### Tools

| Tool | Input | Output |
|------|-------|--------|
| `kpu_build` | `{preset: "release"}` | `{success: bool, errors: [{file, line, message}]}` |
| `kpu_test` | `{label: "timing", filter: "tag_cam"}` | `{total, passed, failed, failures: [{test, file, line, assertion}]}` |
| `kpu_run_demo` | `{demo: "csp_pipeline", args: ["--verbose"]}` | `{status, cycles, transaction_log: [...], summary: {...}}` |
| `kpu_run_pattern` | `{pattern: "page_conflicts"}` | `{trace_file, events: int, duration_cycles}` |
| `kpu_validate_trace` | `{trace_file: "path"}` | `{status, violations: [{id, severity, message, fix_hint}]}` |

#### Implementation Sketch

```python
# kpu_sim_server.py
from mcp.server import Server
import subprocess
import json
import re

server = Server("kpu-sim")

@server.tool("kpu_build")
async def build(preset: str = "release") -> dict:
    result = subprocess.run(
        ["cmake", "--build", "--preset", preset],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return {"success": True, "errors": []}

    # Parse first error from compiler output
    errors = parse_compiler_errors(result.stderr)
    return {"success": False, "errors": errors}

@server.tool("kpu_test")
async def test(label: str = "timing", filter: str = None) -> dict:
    cmd = ["ctest", "-L", label, "--output-on-failure"]
    if filter:
        cmd.extend(["-R", filter])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="build")
    return parse_ctest_output(result.stdout + result.stderr)

@server.tool("kpu_run_demo")
async def run_demo(demo: str, args: list = None) -> dict:
    cmd = [f"./build/examples/schedule/{demo}"] + (args or [])
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_demo_output(result.stdout)

@server.tool("kpu_validate_trace")
async def validate_trace(trace_file: str) -> dict:
    result = subprocess.run(
        ["python3", "patterns/memory/lpddr5/common/trace_validator.py",
         trace_file, "--json"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)
```

#### Configuration

```json
// ~/.claude/settings.json
{
  "mcpServers": {
    "kpu-sim": {
      "command": "python3",
      "args": ["/home/stillwater/dev/stillwater/clones/kpu-sim/tools/mcp/kpu_sim_server.py"],
      "cwd": "/home/stillwater/dev/stillwater/clones/kpu-sim"
    }
  }
}
```

---

### MCP Server 2: `kpu-architecture-server` (Verification)

**Priority:** MEDIUM — used during architecture changes and reviews.
**Implementation:** Python, operates on source files + AST analysis.

#### Tools

| Tool | Input | Output |
|------|-------|--------|
| `kpu_check_credit_flow` | `{component: "DMAEngineProcess"}` | `{acquires: [...], releases: [...], balanced: bool, issues: [...]}` |
| `kpu_check_terminology` | `{path: "include/sw/kpu/timing/"}` | `{clean: bool, violations: [{file, line, term, context}]}` |
| `kpu_component_graph` | `{}` | `{nodes: [{name, type, credits, tags}], edges: [{from, to, via}]}` |
| `kpu_invariant_status` | `{}` | `{implemented: ["INV-001",...], pending: [...], coverage: 0.85}` |
| `kpu_process_audit` | `{component: "StreamerProcess"}` | `{has_tick: bool, has_credits: bool, has_tags: bool, has_is_complete: bool, issues: [...]}` |

#### Use Cases

**Before implementing a new component:**
```
Claude: Let me check the current architecture graph.
-> kpu_component_graph()
Returns: nodes=[DMA, MC, BM, STR, Compute], edges=[DMA->MC, DMA->L3, BM->L3->L2, ...]

Claude: I see where the new NoC fits: between BM and L2.
```

**After editing timing code:**
```
Claude: Let me verify I didn't break credit flow.
-> kpu_check_credit_flow({component: "DMAEngineProcess"})
Returns: {balanced: true, acquires: 3, releases: 3}

-> kpu_check_terminology({path: "include/sw/kpu/timing/"})
Returns: {clean: true, violations: []}
```

---

### MCP Server 3: `kpu-design-space-server` (Exploration)

**Priority:** MEDIUM-LOW — used during architecture design space exploration.
**Implementation:** Python, wraps simulator with parameter sweeps.

#### Tools

| Tool | Input | Output |
|------|-------|--------|
| `kpu_sweep_parameter` | `{param: "num_banks", values: [8,16,32], workload: "matmul_64x64"}` | `{results: [{value, latency, throughput, row_hit_rate}]}` |
| `kpu_sweep_credits` | `{l3_range: [4,8,16,32], l2_range: [8,16,32,64], workload: "matmul"}` | `{matrix: [[{l3,l2,cycles,stalls}]]}` |
| `kpu_bottleneck_analysis` | `{workload: "matmul_128x128"}` | `{bottleneck: "MC_command_bus", utilization: {mc: 0.95, dma: 0.45, bm: 0.30, str: 0.20}}` |
| `kpu_what_if` | `{changes: {"t_rcd": 10}, baseline: {"t_rcd": 15}, workload: "matmul"}` | `{baseline_cycles: 163, modified_cycles: 148, speedup: 1.10}` |
| `kpu_fidelity_compare` | `{workload: "matmul_16x16", tiers: ["BEHAVIORAL","TRANSACTIONAL"]}` | `{behavioral: {cycles: 1, values_correct: true}, transactional: {cycles: 163, values: "statistical"}}` |

#### Use Cases

**Architecture design space exploration:**
```
User: "What's the optimal L3 credit pool size for a 128x128 matmul?"

Claude:
-> kpu_sweep_credits({l3_range: [4,8,16,32,64], l2_range: [16], workload: "matmul_128x128"})
Returns: {results: [{l3:4, cycles:850}, {l3:8, cycles:520}, {l3:16, cycles:480}, {l3:32, cycles:478}, {l3:64, cycles:478}]}

Claude: "L3=16 captures 99% of the benefit. Beyond 16, additional credits
provide diminishing returns because the bottleneck shifts to MC command bus
throughput."
```

**What-if analysis:**
```
User: "What if we added a second memory controller?"

Claude:
-> kpu_what_if({changes: {"num_memory_controllers": 2}, baseline: {"num_memory_controllers": 1}})
Returns: {baseline_cycles: 480, modified_cycles: 310, speedup: 1.55,
          bottleneck_shift: "MC_command_bus -> BM_bandwidth"}

Claude: "A second MC gives 1.55x speedup but shifts the bottleneck to
BlockMover bandwidth. You'd need to also increase BM bus width to see
further gains."
```

---

### GitHub Integration: Use `gh` CLI (No MCP Needed)

The `gh` CLI already provides full GitHub access (issues, PRs, CI, reviews)
and is simpler than an MCP server. Skills like `/fix-issue` can invoke `gh`
commands directly via Bash. No additional setup required.

---

## 4. Implementation Roadmap

### Phase 1: Immediate (This Week)

| Item | Effort | Impact |
|------|--------|--------|
| Build `kpu-sim-server` MCP (build/test/run) | 2-4 hours | Structured build/test results |
| Add terminology guard hook | 10 min | Catches #1 bug class automatically |
| Add session log reminder hook | 5 min | Enforces governance |

### Phase 2: Short-term (Next 2 Weeks)

| Item | Effort | Impact |
|------|--------|--------|
| Create `/fix-issue` skill (uses `gh` CLI) | 15 min | Issue-driven development |
| Document subagent patterns in CLAUDE.md | 30 min | Consistent agent usage |

### Phase 3: Medium-term (Next Month)

| Item | Effort | Impact |
|------|--------|--------|
| Build `kpu-architecture-server` MCP | 4-8 hours | Automated architecture verification |
| Build `kpu-design-space-server` MCP | 8-16 hours | Architecture exploration automation |
| Create `/explore-architecture` skill | 15 min | Structured design space exploration |

### Phase 4: Aspirational

| Item | Effort | Impact |
|------|--------|--------|
| Nightly CI agent that reads failures and creates fix PRs | Complex | Autonomous maintenance |
| Multi-fidelity regression: auto-compare behavioral vs transactional | Complex | Fidelity consistency |
| Design space optimization: auto-sweep and recommend configurations | Complex | Architecture automation |

---

## 5. The Composition Effect

The real power comes from how these layers compose:

```
gh CLI: "Issue #15: DMA stalls when L3 credits exhausted"
    |
    v
CLAUDE.md: "DMA uses credit-based flow, must acquire before push"
    |
    v
/fix-issue skill: "Read issue, find code, diagnose, fix, test"
    |
    v
kpu-sim-server MCP: kpu_test({filter: "dma"}) -> structured results
    |
    v
Architecture server MCP: kpu_check_credit_flow({component: "DMA"}) -> balanced?
    |
    v
Memory: "Known bug: CreditPool double-release when ref_count not checked"
    |
    v
Hooks: Terminology guard confirms no cache semantics introduced
    |
    v
Subagent: Parallel test fix for any regressions
    |
    v
gh CLI: Create PR with fix, monitor CI
```

Each layer handles what it's best at:
- **MCP servers** provide structured data
- **CLAUDE.md** provides domain knowledge
- **Skills** provide workflow recipes
- **Memory** provides learned patterns
- **Hooks** provide automated guards
- **Subagents** provide parallelism

No single layer is "intelligent." The intelligence emerges from their composition.
