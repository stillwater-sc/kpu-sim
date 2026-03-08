# Claude Code Productivity Assessment for kpu-sim

**Date:** 2026-03-06
**Purpose:** Identify opportunities to leverage Claude Code's agentic capabilities
more effectively for this repository.

---

## 1. How It Works: The Mental Model

You asked "how can collections of markdown generate all this intelligence?" The
answer is: **they don't generate intelligence — they direct it.**

Claude Code is an LLM with broad capabilities. Without context, it produces
generic output. The markdown layers act as a **focusing lens**:

```
                Raw LLM Capability (broad, generic)
                         |
                    CLAUDE.md          "Always know this about KPU"
                         |
                      Skills           "When I say /X, follow this recipe"
                         |
                      Memory           "We've learned these patterns together"
                         |
                       Hooks           "Automatically do Z when event E happens"
                         |
                    MCP Servers        "Also talk to GitHub, Slack, databases..."
                         |
              Focused, Project-Specific Behavior
```

Each layer multiplies the others:
- CLAUDE.md tells me "never use cache terminology" → I avoid it in ALL interactions
- A `/new-component` skill + CLAUDE.md → I generate a CSP process that follows
  credit-based dataflow rules correctly, every time
- Memory says "TagCAM insert returns true for duplicates" → I don't write tests
  that assume false
- A hook on file save checks for "cache hit" in timing code → catches violations
  before they get committed

**The key insight:** you're not programming Claude Code — you're *constraining*
it. Each piece of markdown removes wrong paths, making the right path more likely.
The more specific and layered your constraints, the more "intelligent" the behavior
appears.

---

## 2. Current State Audit

### What You Have

| Layer | Current State | Effectiveness |
|-------|--------------|---------------|
| **CLAUDE.md** | 513 lines. Excellent coverage of execution model, validation, fidelity tiers, session governance | HIGH — but could add workflow shortcuts |
| **Skills** | 5 generic C++ skills + `/wrapup` | LOW — none are KPU-specific |
| **Memory** | Empty (no MEMORY.md) | ZERO — every session starts cold |
| **Hooks** | None configured | ZERO — no automation |
| **MCP Servers** | clangd LSP only | LOW — no GitHub integration |
| **Permissions** | ~130 individually allowed commands | NOISY — accumulated one-by-one |

### What's Missing

1. **No project-specific skills** — The C++ skills are generic. You need skills
   that encode KPU-specific workflows (build→test→validate, new component creation,
   architecture checking).

2. **No memory** — I forget everything between sessions. Common bug patterns,
   architectural decisions, your preferences — all lost. Every new session I
   re-discover that TagCAM uses ref_counting, that tests expect formatted names,
   that the drain path needs compute_result_tag_cam populated.

3. **No hooks** — Nothing automatic happens. No build check before commit, no
   terminology check on timing code edits, no test run after modifications.

4. **GitHub integration via `gh` CLI** — Already functional. The `gh` CLI
   provides full access to issues, PRs, CI, and reviews without needing
   an MCP server.

---

## 3. Recommended Skills

Skills are markdown files in `~/.claude/commands/` (global) or
`.claude/commands/` (project-local). They're invoked with `/skill-name`.

### 3.1 `/build-test` — The Core Development Loop

**Why:** This is your most common workflow. Every change triggers
build→test→analyze. Currently you type this manually or I construct it
from scratch.

```markdown
# File: .claude/commands/build-test.md

Build the project and run all timing tests. Report results clearly.

Steps:
1. Build: `cmake --build --preset release 2>&1`
   - If build fails, show ONLY the first error and fix it
   - Rebuild after fix

2. Run timing tests: `cd build && ctest -L timing --output-on-failure 2>&1`

3. Report results as a table:
   | Test | Status | Assertions |
   |------|--------|------------|

4. If any tests FAIL:
   - Show the specific assertion that failed
   - Read the test file and the source code it tests
   - Propose a fix (but don't apply without asking)

5. If all tests PASS: report "All N tests passing" with total assertion count.

Do NOT run tests individually — use ctest for consistent reporting.
```

### 3.2 `/new-csp-process` — Generate CSP Components

**Why:** Every new hardware component follows the same pattern: IProcess
interface, credit/tag management, tick method, config struct, test file.
This skill encodes the pattern.

```markdown
# File: .claude/commands/new-csp-process.md

Generate a new CSP process component: $ARGUMENTS

Follow the established KPU CSP process pattern:

1. Read these reference implementations first:
   - include/sw/kpu/timing/dma_engine_process.hpp (submit/poll with MC)
   - include/sw/kpu/timing/block_mover_process.hpp (tag match + credit)
   - include/sw/kpu/timing/streamer_process.hpp (feed/drain with compute)

2. Generate the component in include/sw/kpu/timing/ with:
   - Config struct with display_name()
   - Constructor taking credit pools and tag CAMs as references
   - IProcess interface: tick(), is_idle(), has_pending_work(), id(), name(), reset()
   - Schedule methods for the component's operations
   - Private state machine with clear state transitions
   - Statistics accessors

3. Generate a test file in tests/timing/ with:
   - Construction tests
   - Single-operation tests
   - Multi-operation tests
   - Stall/backpressure tests
   - Credit conservation checks

4. Update tests/timing/CMakeLists.txt to include the new test.

5. Build and verify: cmake --build --preset release

CRITICAL RULES (from CLAUDE.md):
- Use credit-based push semantics, NEVER fetch-on-demand
- Use buffer/credit terminology, NEVER cache terminology
- Component must wait for downstream credit before pushing
- Component must return credit upstream after consuming
```

### 3.3 `/validate-architecture` — Check for Dataflow Violations

**Why:** The most common class of bugs in this codebase is accidentally
introducing cache semantics or fetch-on-demand patterns. This skill does
a systematic audit.

```markdown
# File: .claude/commands/validate-architecture.md

Audit the codebase for credit-based dataflow violations.

Search for these anti-patterns in include/sw/kpu/timing/ and tests/timing/:

1. **Cache terminology** (FORBIDDEN):
   Search for: cache, hit, miss, evict, LRU, refetch, lookup.*miss
   Each occurrence is a violation.

2. **Fetch-on-demand patterns**:
   Search for: request.*response, poll.*data, fetch, demand
   Check if any component pulls data instead of waiting for push.

3. **Missing credit checks**:
   For each process's tick() method, verify:
   - Producer checks credit.acquire() before pushing downstream
   - Consumer calls credit.release() after consuming

4. **Missing tag operations**:
   - Insert after tile arrives at a level
   - Match before consuming from a level
   - Invalidate after tile is moved/consumed

Report findings as:
| File:Line | Violation | Severity | Suggested Fix |
|-----------|-----------|----------|---------------|
```

### 3.4 `/trace-check` — Validate Generated Traces

**Why:** After any change to timing code, traces need validation.

```markdown
# File: .claude/commands/trace-check.md

Run trace validators on all generated trace files.

1. Find all trace files:
   find traces/ -name "*.json" 2>/dev/null

2. For each trace file, run:
   python3 patterns/memory/lpddr5/common/trace_validator.py <file> --json

3. Report results:
   | Trace File | Status | Violations |
   |------------|--------|------------|

4. For any FAILED traces:
   - Show the invariant ID and violation message
   - Cross-reference with patterns/memory/lpddr5/INVARIANTS.md
   - Identify the C++ code likely causing the violation
```

### 3.5 `/test-status` — Quick Health Check

**Why:** Fast way to see what's broken without full investigation.

```markdown
# File: .claude/commands/test-status.md

Quick health check of the project.

Run these in parallel:
1. `cmake --build --preset release 2>&1 | tail -5` (build status)
2. `cd build && ctest -L timing --output-on-failure 2>&1` (test status)
3. `git status --short` (uncommitted changes)
4. `git log --oneline -5` (recent commits)

Report as:
## Project Health
- Build: PASS/FAIL
- Tests: X/Y passing
- Uncommitted: N files
- Last commit: <hash> <message>

If there are failing tests, list them with one-line failure reason.
```

### 3.6 `/fix-tests` — Identify and Fix Failing Tests

**Why:** After architectural changes, multiple tests often break. This
skill automates the diagnosis and fix cycle.

```markdown
# File: .claude/commands/fix-tests.md

Find and fix all failing tests. $ARGUMENTS

1. Build the project first. If build fails, fix build errors.

2. Run all timing tests with verbose output:
   cd build && ctest -L timing --output-on-failure 2>&1

3. For each failing test:
   a. Parse the failure output to get file:line and assertion
   b. Read the test source to understand intent
   c. Read the implementation being tested
   d. Determine if the bug is in the TEST or the IMPLEMENTATION:
      - If test expectations are wrong (API changed): fix the test
      - If implementation is wrong (regression): fix the implementation
   e. Apply the fix

4. After all fixes, rebuild and rerun ALL tests to verify no regressions.

5. Report what was fixed:
   | Test | Failure | Root Cause | Fix Applied |
   |------|---------|------------|-------------|

IMPORTANT: Never skip a failing test. Never mark it as "to fix later."
Debug root causes, don't mask symptoms.
```

### 3.7 `/plan` — Create a Design Plan

**Why:** You have a `docs/plans/` directory with good plan documents.
This skill ensures plans follow the established format.

```markdown
# File: .claude/commands/plan.md

Create a design plan for: $ARGUMENTS

1. Create a new file in docs/plans/ with a descriptive filename.

2. Follow this structure:
   # <Title>
   **Date:** YYYY-MM-DD
   **Status:** Design | In Progress | Implemented

   ## 1. Problem Statement
   What problem does this solve? What's wrong with the current approach?

   ## 2. Architecture
   ASCII diagram showing component relationships.
   Use the established style from docs/plans/dma_csp.md.

   ## 3. Design
   Key data structures and APIs with code snippets.
   Show both WRONG (what to avoid) and CORRECT (what to do) patterns.

   ## 4. Implementation Steps
   Numbered steps with specific files to modify.

   ## 5. Verification
   Specific test commands and expected outcomes.

   ## 6. Key Invariants
   What must always be true after this change?

3. Do NOT implement the plan. Just create the document for review.
```

---

## 4. Memory: What to Store

Create `~/.claude/projects/-home-stillwater-dev-stillwater-clones-kpu-sim/memory/MEMORY.md`:

```markdown
# KPU-SIM Project Memory

## Architecture Patterns
- CSP processes: DMA, BlockMover, Streamer (all implement IProcess)
- DMA submits to MemoryControllerProcess via submit_request(tile, is_load, engine_id)
- MC completions filtered by submitter_id (multiple DMAs share one MC)
- Tick order: MC first, then DMA, then BlockMover, then Streamer
- See: [architecture.md](architecture.md)

## Common Bug Classes
- Cache terminology creep (hit/miss/evict) — always use buffer/credit terms
- TagCAM.insert() returns true for duplicates (ref_count increment, not rejection)
- Drain operations need compute_result_tag_cam populated first
- CreditPool double-release when tile reuse ref_count not checked
- Name formatting: tests expect "BM_0", implementation returns "BM" (pre-existing)

## Build & Test
- Build: cmake --build --preset release
- Tests: cd build && ctest -L timing --output-on-failure
- Test presets: default, unit, integration, performance
- 3 pre-existing test failures (tag_cam duplicate, bm/str name format)

## User Preferences
- Session logs required in docs/sessions/
- Decision logs with wrong-decision documentation
- CHANGELOG.md updated with every significant change
- Commits with Co-Authored-By trailer
```

---

## 5. Hooks: Event-Driven Automation

Hooks are shell commands that run automatically when Claude Code does certain
things. They're configured in `.claude/settings.json` or `.claude/settings.local.json`.

### How Hooks Work

```json
{
  "hooks": {
    "preToolCall": [
      {
        "matcher": "Edit",
        "command": "echo 'Editing file...'"
      }
    ],
    "postToolCall": [
      {
        "matcher": "Bash",
        "command": "echo 'Command completed.'"
      }
    ]
  }
}
```

Hook types:
- **preToolCall**: Runs BEFORE a tool executes. Can block the tool.
- **postToolCall**: Runs AFTER a tool executes.
- **userPromptSubmit**: Runs when you submit a message.

### Recommended Hooks for kpu-sim

#### Hook 1: Terminology Guard (Post-Edit)

When I edit any timing header, check for forbidden cache terminology:

```json
{
  "hooks": {
    "postToolCall": [
      {
        "matcher": "Edit",
        "command": "bash -c 'if echo \"$CLAUDE_FILE_PATH\" | grep -q \"timing/\"; then if grep -n \"cache\\|\\bhit\\b\\|\\bmiss\\b\\|\\bevict\" \"$CLAUDE_FILE_PATH\" 2>/dev/null | grep -iv \"tag_cam\\|row_hit\\|row_miss\"; then echo \"WARNING: Possible cache terminology in timing code. Use buffer/credit semantics.\"; fi; fi'"
      }
    ]
  }
}
```

**What this does:** After every file edit, if the file is in a `timing/`
directory, it scans for words like "cache", "hit", "miss", "evict" (excluding
legitimate uses like "tag_cam" and "row_hit") and warns if found.

**Why it helps:** The #1 architectural violation in this codebase is accidentally
introducing cache semantics. This catches it at edit time, not after tests fail.

#### Hook 2: Build Check Reminder (Pre-Commit)

Note: Claude Code doesn't have a direct pre-commit hook, but you can configure
a git pre-commit hook separately:

```bash
# .git/hooks/pre-commit
#!/bin/bash
cmake --build --preset release 2>&1 | tail -1
if [ $? -ne 0 ]; then
    echo "Build failed. Fix errors before committing."
    exit 1
fi
cd build && ctest -L timing --output-on-failure -q 2>&1
if [ $? -ne 0 ]; then
    echo "Tests failing. Fix before committing."
    exit 1
fi
```

#### Hook 3: Session Log Reminder (User Prompt)

```json
{
  "hooks": {
    "userPromptSubmit": [
      {
        "matcher": "commit|push|done|finish",
        "command": "echo 'Reminder: Have you created a session log in docs/sessions/?'"
      }
    ]
  }
}
```

### Important Limitation

Hooks run shell commands — they don't have access to Claude Code's AI
capabilities. They're best for **simple, deterministic checks** (grep for
patterns, run a build, check git status). They're NOT suitable for complex
analysis that requires understanding code semantics.

---

## 6. MCP Servers: Extending Claude Code's Reach

MCP (Model Context Protocol) servers give Claude Code access to external services
as if they were built-in tools. Think of them as plugins.

### How MCP Works

```
Claude Code  ←→  MCP Protocol  ←→  MCP Server  ←→  External Service
                                    (local process)    (GitHub, Slack, DB...)
```

You configure MCP servers in `.claude/settings.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
```

### Currently Active

You have **clangd LSP** enabled as a plugin. This gives me structured code
intelligence (go-to-definition, find-references, diagnostics) beyond just
text search.

### Recommended MCP Servers for kpu-sim

#### MCP 1: GitHub Server (HIGH VALUE)

**What it provides:**
- List/read/create/close issues
- List/read/comment on PRs
- Read PR review comments and act on them
- Check CI/CD status
- Read workflow run logs

**Why it matters for you:**
- I could run `/fix-issue 42` and automatically read the issue, find the
  relevant code, propose a fix, and create a PR
- After you push, I could monitor CI and tell you if it passed
- When PR review comments come in, I could read them and propose fixes
- You mentioned wanting agents that "look at the issues list and try to fix
  issues" — this is exactly what GitHub MCP enables

**Setup:**
```bash
# Install the GitHub MCP server
npm install -g @modelcontextprotocol/server-github

# Add to ~/.claude/settings.json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}
```

**What changes:** Instead of me running `gh issue list` as a bash command
and parsing text output, I'd have structured tools like
`mcp__github__list_issues`, `mcp__github__create_pull_request` that return
structured data I can reason about directly.

#### MCP 2: Filesystem Watcher (MEDIUM VALUE)

For monitoring build output, test results, and trace files in real-time.
Less critical since I can already read files, but useful for background
monitoring.

#### MCP 3: Custom KPU Build Server (ASPIRATIONAL)

You could write a small MCP server that wraps your build/test/validate workflow:

```python
# Hypothetical kpu-build-server
tools:
  - kpu_build: Build the project, return structured results
  - kpu_test: Run tests, return pass/fail per test case
  - kpu_validate_traces: Run validators, return structured violations
  - kpu_architecture_check: Grep for anti-patterns, return violations
```

This would be more structured than bash commands, but is only worth building
if you find yourself doing these operations many times per day.

---

## 7. Subagent Patterns

Claude Code can spawn subagents using the Agent tool. These run in parallel
and protect the main conversation context from getting overloaded.

### When to Use Subagents

| Use Case | Main Thread | Subagent |
|----------|-------------|----------|
| Fix a single bug | Yes | No — overhead not worth it |
| Research architecture question | No | Yes — may read many files |
| Fix 3 independent test failures | No | Yes — parallelize |
| Validate all traces | No | Yes — parallel validation |
| Compare two approaches | No | Yes — explore independently |

### Useful Subagent Patterns for kpu-sim

#### Pattern 1: Parallel Test Fix

When multiple tests fail after an architectural change, spawn one subagent
per failing test:

```
Main: "Fix all failing timing tests"
  → Agent 1: Fix test_tag_cam (TagCAM duplicate semantics)
  → Agent 2: Fix test_block_mover_process (name formatting)
  → Agent 3: Fix test_streamer_process (name formatting)
Main: Collect results, rebuild, verify
```

#### Pattern 2: Architecture Audit

Spawn agents to check different aspects in parallel:

```
Main: "Audit the timing code for correctness"
  → Agent 1: Check credit flow (acquire/release balance)
  → Agent 2: Check tag operations (insert/match/invalidate)
  → Agent 3: Check terminology (no cache semantics)
  → Agent 4: Check tick ordering (MC before DMA before BM before STR)
Main: Synthesize findings
```

#### Pattern 3: Explore-then-Implement

When you give me a broad task, I should use an Explore subagent first:

```
Main: "Add NoC routing to the timing model"
  → Explore Agent: Read existing NoC code, understand patterns, find
    integration points, check for existing tests
Main: (with explore results) Implement the change with full context
```

---

## 8. CLAUDE.md Improvements

Your CLAUDE.md is already excellent. Here are targeted additions:

### Add: Common Workflow Shortcuts

```markdown
## Quick Commands

| Command | What It Does |
|---------|-------------|
| `/build-test` | Build + run all timing tests |
| `/test-status` | Quick health check |
| `/fix-tests` | Find and fix all failing tests |
| `/validate-architecture` | Check for dataflow violations |
| `/new-csp-process <name>` | Generate new CSP component |
| `/plan <feature>` | Create design plan document |
| `/wrapup` | Create changelog + session log |
```

### Add: Permission Simplification

Your `.claude/settings.local.json` has 130+ individually allowed bash commands.
Consider replacing with broader patterns:

```json
{
  "permissions": {
    "allow": [
      "Bash(cmake:*)",
      "Bash(ctest:*)",
      "Bash(./build/*:*)",
      "Bash(python3:*)",
      "Bash(git:*)",
      "Bash(gh:*)",
      "Bash(ls:*)",
      "Bash(find:*)",
      "Bash(tree:*)",
      "Bash(echo:*)",
      "Bash(wc:*)",
      "Bash(sort:*)",
      "Bash(head:*)",
      "Bash(tail:*)",
      "Bash(chmod:*)",
      "Bash(timeout:*)",
      "Skill(wrapup)"
    ]
  }
}
```

This is cleaner and covers the same ground. The key safety principle: allow
read-only and build/test commands broadly; require confirmation for destructive
operations (git push, rm, etc.).

---

## 9. Priority Ranking

What to do first, based on effort vs. impact:

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 1 | **Create MEMORY.md** | 10 min | HIGH — stops re-discovering known bugs |
| 2 | **Create `/build-test` skill** | 5 min | HIGH — used multiple times per session |
| 3 | **Create `/test-status` skill** | 5 min | HIGH — fast health check |
| 4 | **Create `/fix-tests` skill** | 5 min | MEDIUM — saves debugging time |
| 5 | **Simplify permissions** | 10 min | MEDIUM — reduces approval noise |
| 6 | **Create `/validate-architecture`** | 5 min | MEDIUM — catches dataflow violations |
| 7 | **Set up GitHub MCP** | 15 min | MEDIUM — enables CI/issue integration |
| 8 | **Create `/new-csp-process` skill** | 10 min | LOW-MEDIUM — less frequent |
| 9 | **Add terminology guard hook** | 10 min | LOW-MEDIUM — catches rare bugs |
| 10 | **Create `/plan` skill** | 5 min | LOW — plans are infrequent |

---

## 10. The Big Picture: Autonomous Development

The vision you're describing — agents that read issues, propose fixes, monitor
CI, act on PR reviews — is achievable by combining these layers:

```
GitHub MCP Server
    ↓ (structured issue/PR/CI data)
Claude Code + CLAUDE.md (knows KPU architecture)
    ↓ (uses skills for common workflows)
/fix-tests, /build-test, /validate-architecture
    ↓ (hooks catch mistakes)
Terminology guard, build check
    ↓ (memory prevents repeat mistakes)
MEMORY.md (known bugs, patterns, preferences)
    ↓ (subagents parallelize work)
Agent tool for multi-file fixes
    ↓
Commit + PR + CI monitoring
```

The "intelligence" isn't in any single markdown file. It's in how they
**compose**: CLAUDE.md ensures I understand the architecture, skills ensure
I follow the right workflow, memory ensures I don't repeat mistakes, hooks
catch violations automatically, and MCP gives me access to the external
systems where work actually happens.

**You're not writing AI programs. You're building a development environment
that constrains an AI to behave like an expert KPU hardware engineer.**
