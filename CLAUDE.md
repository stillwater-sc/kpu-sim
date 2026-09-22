# Claude Code Integration Guidelines

This document provides guidance for Claude Code when working on the KPU-SIM project.

## Quick Commands (Project Skills)

| Command | What It Does |
|---------|-------------|
| `/build-test` | Build + run all timing tests, report results |
| `/test-status` | Quick project health check (build, tests, git) |
| `/fix-tests` | Find and fix all failing tests systematically |
| `/validate-architecture` | Audit for credit/dataflow violations |
| `/new-csp-process <name>` | Generate new CSP component with tests |
| `/plan <feature>` | Create design plan in docs/plans/ |
| `/trace-check` | Validate all generated trace files |
| `/wrapup` | Create changelog + session log |

## Repository Purpose: Multi-Fidelity Simulation

**READ THIS FIRST** - The KPU simulator is a **multi-fidelity simulation environment**
that interprets one program at **four levels** of transaction granularity:

### Simulation Fidelity Tiers

**CSP is the program layer, not a fidelity** (ADR 0002). One CSP program — derived from a
Domain Flow Program — is executed by every level; what differs is **how finely the level
decomposes a CSP transaction**.

| Level | Decomposes a transaction into | Speed | Computes Values? |
|-------|-------------------------------|-------|------------------|
| **L-B** behavioral | a whole block move, atomic | ~100-1000x | **YES** |
| **L-T1** block-sequential | one tile move (finite-buffer credits and capacity arrive in #264 increment 4) | ~10-100x | **YES — exact**, bit-identical to L-B |
| **L-T2** resource transactional | `read`/`write` per resource; `push` into the compute tile | — | **YES — exact**, bit-identical to L-B |
| **L-CA** cycle-accurate | protocol events, per cycle | 1x (baseline) | **YES** — within tolerance of L-B |

**Every level computes values.** Decomposition changes *when* things happen, never *what*
is computed. A level that returns timing without values is not a cheaper simulation — it is
an unvalidated one. Full model: `docs/architecture/adr/0002-csp-interpretations-and-virtual-platform.md`.

### The Multi-Fidelity Philosophy

The progression works as follows:

1. **Cycle-Accurate First**: Model subsystems from first principles to capture emergent
   behavior with high confidence (e.g., DRAM timing, bank conflicts, page dynamics)

2. **Characterize Statistics**: Extract latency, concurrency, and resource occupation
   statistics from cycle-accurate simulation

3. **Build Transactional Models**: Use collected statistics to create faster models
   with queue-based contention and aggregate latencies

4. **Create Behavioral Models**: Highest abstraction with functional correctness,
   enabling software development and validation

### Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/01-architecture/kpu-execution-model.md` | **Credit-based dataflow model (MUST READ)** |
| `docs/02-simulation/fidelity-framework.md` | Full multi-fidelity design (READ THIS) |
| `include/sw/kpu/fidelity/simulation_fidelity.hpp` | Fidelity enums and types |
| `include/sw/kpu/fidelity/component_config.hpp` | Per-component configuration |

**Non-negotiable:** the BEHAVIORAL tier computes actual values and propagates results.
A behavioral component that only models timing is wrong.

**Authorities:** values answer to the L0 `TileProgramReference`, timing to **L-CA**, the
cycle-accurate level. Before comparing either, read D5 and §7.5 of
`docs/architecture/adr/0001-program-contract-and-transactional-engine.md` — they give
the required bar per tier (bit-exact vs tolerance) and the comparator to use.

---

## KPU Execution Model: Credit-Based Dataflow

**CRITICAL: READ THIS BEFORE ANY KPU-RELATED CODE GENERATION**

The KPU implements a **credit-based dataflow execution model**. This is fundamentally
different from stored-program (von Neumann) architectures. Failure to understand this
distinction leads to incorrect implementations.

**Authoritative Reference:** `docs/01-architecture/kpu-execution-model.md`

### Core Principle: Credits UP, Data DOWN

```
                    CREDITS (upstream)
                         ↑
    Host Memory ───→ L3 Buffers ───→ L2 Buffers ───→ L1 Streams ───→ Compute
                         ↓
                    DATA/TILES (downstream)
```

### MANDATORY Rules for KPU Code

1. **NO CACHE SEMANTICS**
   - L3, L2, L1 are **buffers**, NOT caches
   - NEVER use terms: cache hit, cache miss, cache evict, LRU, refetch
   - NEVER implement demand-driven fetching
   - CORRECT: "tile arrived at buffer", "buffer available (credit)"
   - WRONG: "cache hit", "cache miss", "tile evicted"

2. **CREDIT-BASED FLOW**
   - A producer can ONLY push data when it has a credit from downstream
   - When a consumer finishes with data, it returns a credit upstream
   - No polling, no request-response - only push with credit

3. **COMPONENT BEHAVIORS**
   - **DMA**: WAITS for L3 buffer credit, then PUSHES tile to L3
   - **BlockMover**: WAITS for tile arrival (tag CAM) + L2 credit, then PUSHES to L2
   - **Streamer**: WAITS for tile arrival (tag CAM) + L1 credit, then PUSHES to L1
   - All components use **tag CAM** for out-of-order tile matching

4. **CORRECT TRACE EVENTS**
   ```
   TILE_READY(T @ L3[i])      - Tile T arrived at L3 buffer i
   BUFFER_AVAILABLE(L3[i])    - L3 buffer i has credit (space available)
   DMA_PUSH                   - DMA pushing tile downstream
   BM_PUSH                    - BlockMover pushing tile L3→L2
   STR_FEED                   - Streamer feeding tile L2→L1
   ```

5. **FORBIDDEN TRACE EVENTS**
   ```
   L3_ACCESS with HIT/MISS    - WRONG: Implies cache lookup
   CACHE_HIT / CACHE_MISS     - WRONG: No cache exists
   L3_EVICT                   - WRONG: No eviction, only credit return
   REFETCH                    - WRONG: Tiles flow once, not re-fetched
   ```

### Quick Reference Table

| WRONG (Cache/Stored-Program) | CORRECT (Dataflow) |
|------------------------------|-------------------|
| Cache hit | Tile already in buffer (from previous push) |
| Cache miss | Waiting for tile to arrive |
| Cache eviction | Buffer available (credit returned) |
| LRU replacement | N/A - explicit buffer management |
| Fetch on demand | Push when credit available |
| Request-response | Credit-push flow |
| Content-addressed lookup | Tag CAM match for tile arrival |

### Implementation Reference

**USE THESE (correct dataflow semantics):**
```
include/sw/kpu/timing/                    # cycle-accurate: the live credit model
├── concurrent_timing_executor.hpp        # credits, tag CAM, queues, tick order
├── credit_pool.hpp                       # credit accounting (+ partitioned mode)
├── tag_cam.hpp                           # out-of-order tile-arrival matching
└── {dma,block_mover,streamer,memory_controller}_process.hpp   # push-with-credit

include/sw/kpu/program/                   # tile granularity
├── tile_transaction_executor.hpp         # transactional tier: fires on credit + residency
└── tile_dependencies.hpp                 # what must be ordered, and why
```

**AVOID THESE:**
```
include/sw/kpu/behavioral/
├── l3_cache_model.hpp            # WRONG: cache semantics

include/sw/kpu/models/dataflow/   # FROZEN, retiring (#266)
├── flow_graph_executor.hpp       # base execute_operation() is an empty default
├── dma_flow_executor.hpp         # tracks tile coords + accumulator STATE only
├── block_mover_flow_executor.hpp #   — no numeric payload anywhere in these files
└── streamer_flow_executor.hpp
```

The `models/dataflow/` path used to be listed as the reference to follow. It is not. Its
executors move **tile identity and credit state, never values**: the base
`execute_operation()` is an empty default, and where the three subclasses do override it
they update coordinates and accumulator bookkeeping. So the files model the credit *shapes*
faithfully while computing nothing — which makes them a misleading template. Read
`include/sw/kpu/timing/` for the credit model that carries real payloads.

### Before Writing KPU Code, Ask:

1. "Am I implementing push-with-credit or fetch-on-demand?"
   - If fetch-on-demand: STOP and redesign

2. "Am I using cache terminology (hit/miss/evict)?"
   - If yes: STOP and use buffer/credit terminology

3. "Does my component wait for downstream credit before pushing?"
   - If no: STOP and add credit checking

4. "Does my component return credit upstream after consuming data?"
   - If no: STOP and add credit return

---

## Validation Requirements

This section covers validation for **cycle-accurate** simulation, particularly
memory controller traces and timing invariants.

### Core Principle: Validate Before Declaring Complete

**Never declare code generation complete without validation.**

Claude Code must follow this workflow for any code that produces artifacts:

```
1. Generate code
2. Run validation tools
3. If validation fails:
   a. Parse error output
   b. Identify root cause
   c. Fix the issue
   d. Return to step 2
4. Only declare complete when validation passes
```

## Validation Tools

### 1. Trace Validator (Python)

**Location:** `patterns/memory/lpddr5/common/trace_validator.py`

**Usage:**
```bash
python3 patterns/memory/lpddr5/common/trace_validator.py <trace_file.json>
```

Add `--json` for machine-readable output. Exit codes are distinct and must be handled
differently: **1** = invariants violated, so parse the violations and fix the root cause,
never the symptom; **2** = the trace file could not be read or parsed, so there are no
violations to parse — fix the trace generation or the path instead.

**When to Run:**
- After generating any trace file
- After modifying trace generation code
- After modifying memory controller behavior

### 2. C++ Test Harness

**Location:** `patterns/memory/lpddr5/common/lpddr5_harness.hpp`

The harness provides:
- `verify_no_violations()` - Check for invariant violations during simulation
- `verify_stats()` - Verify expected statistics
- Runtime invariant checking in the memory controller

### 3. Build Verification

Always run builds and tests after code changes:
```bash
cmake --preset release && cmake --build --preset release
cd build && ctest -L timing --output-on-failure
```

There is **no `release` test preset** — `release` is a *configure* preset only. The test
presets are `default`, `unit`, `integration`, `performance` (and `windows-msvc`).

## Invariant Documentation

**Primary Location:** `patterns/memory/lpddr5/INVARIANTS.md`

Read `INVARIANTS.md` before generating trace-related code. To add an invariant, use the
`add-invariant` skill; for the trace generate→validate→fix loop and the known LPDDR5 trace
bugs, use `/trace-check` and the `lpddr5-trace-debug` skill.

## Session Logging

After significant work, create a session log:

**Location:** `docs/sessions/YYYY-MM-DD_description.md`

Include:
- What was done
- What bugs were found
- What invariants were added/modified
- Validation results

## Session Governance and Accountability

**MANDATORY: Every development session must produce a decision log.**

### Required Actions

1. **Create a session log** at the end of each session:
   - Location: `docs/sessions/YYYY-MM-DD_vX.Y_feature_name.md`
   - Use template: `docs/sessions/SESSION_LOG_TEMPLATE.md`

2. **Document all decisions** made during the session:
   - Technical choices and rationale
   - Alternatives considered
   - Files modified

3. **Document all wrong decisions** (CRITICAL):
   - If you skip failing tests: WRONG - document it
   - If you work around a crash instead of fixing it: WRONG - document it
   - If you declare completion with broken functionality: WRONG - document it

4. **Never skip failing tests**:
   - A crash is a bug to fix, not a reason to skip
   - Debug root causes, don't mask symptoms
   - All tests must pass before declaring completion

### Accountability Rules

| Situation | Wrong Response | Correct Response |
|-----------|----------------|------------------|
| Test crashes | Skip the test | Debug and fix the crash |
| Test fails | Mark as "to be fixed later" | Fix it now |
| Unclear requirement | Assume and proceed | Ask for clarification |
| Multiple approaches | Pick randomly | Document trade-offs, recommend one |

### Session Log Checklist

Before ending a session, verify:
- [ ] Decision log created in `docs/sessions/`
- [ ] All wrong decisions documented with lessons learned
- [ ] All tests passing
- [ ] Commit message references the work done

## Remember

**Correct code is more valuable than fast code.**
