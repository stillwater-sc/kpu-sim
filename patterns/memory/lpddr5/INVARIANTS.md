# LPDDR5 Memory Controller Trace Invariants

This document defines the invariants every run of the LPDDR5 memory controller must satisfy.
They are checked on the run's `.mflow` record by `tools/trace/mflow_check.py`, under the M-check
IDs below; check a record before viewing it or relying on it.

## Where each invariant is enforced on the `.mflow` record

The INV IDs were first checked on the controller's Chrome trace by `trace_validator.py`, which is
retired with that trace (`docs/plans/cycle-mflow-and-debug-server.md` §3.7, step 11). Each is
checked on the `.mflow` record by `tools/trace/mflow_check.py`, where the record names what the
Chrome trace left to be inferred (each command's burst, its exact controller ticks, the declared
device's timing table):

| Invariant | On the record | Note |
|---|---|---|
| INV-001, INV-002, INV-004 (a transaction has one data operation; page commands belong to it; ids are unique) | M4 (exactly one RD/WR serves each burst), M9 (every ACT, RD, WR and PRE serves a burst on its bank and row) | a burst is the transaction; its index is unique by construction |
| INV-003 (temporal order within a transaction) | M10 | also: a page closed before its CAS only for a refresh; a page hit exactly when no ACT serves it |
| INV-005 (bank state machine) | TF10 | one open row per bank, in issue order |
| INV-006 (a refresh belongs to no transaction) | M11 | |
| INV-100 to INV-103 (tRCD, tRP, tRRD, tFAW) | M5 | against the declared device's table, in controller ticks; M5 adds tRAS, tRC, tRFCpb and tCCD |
| INV-104, INV-105 (tRTW, tWTR) | M5 | the bus turnarounds: tWTR_S per channel and tWTR_L per bank group, tRTW per channel |
| INV-200, INV-201 (visualization) | none needed | the views read the record's links and outcomes; they infer no grouping |

## Purpose

Invariants serve three critical purposes:

1. **Correctness Verification** - Ensure trace generator produces semantically correct output
2. **Bug Detection** - Automatically detect common errors in trace generation
3. **Claude Code Guidance** - Provide Claude Code with constraints to validate generated code

## Trace Structure Invariants

### INV-001: Valid Transaction ID Semantics

**Description:** Every `txn_id` must represent exactly ONE user request (READ or WRITE).
DRAM commands (ACTIVATE, PRECHARGE) generated to satisfy that request must share the same `txn_id`.

**Rationale:** A user request may require multiple DRAM commands. For example, a READ to a closed
page requires: ACTIVATE (open row) → READ (column access). These commands must be traceable
back to the originating user request.

**Validation:**
```
For each txn_id:
  - MUST have exactly ONE data operation (BURST_READ or BURST_WRITE)
  - MAY have zero or one ACTIVATE
  - MAY have zero or one PRECHARGE
  - All events with same txn_id belong to the same logical request
```

**Failure Example:**
```json
// BAD: txn_id=0 has only PRECHARGE, no data operation
{"name": "PRECHARGE", "args": {"txn_id": 0, ...}}

// BAD: txn_id=1 has two BURST_READs
{"name": "BURST_READ", "args": {"txn_id": 1, ...}}
{"name": "BURST_READ", "args": {"txn_id": 1, ...}}
```

### INV-002: Command Ownership

**Description:** ACTIVATE and PRECHARGE commands must be associated with the transaction
that triggered them, not assigned arbitrary or sentinel transaction IDs.

**Rationale:** When visualizing a transaction, we need to see ALL commands required to
satisfy it. Orphaned ACTIVATE/PRECHARGE commands break the causal chain.

**Validation:**
```
For each ACTIVATE/PRECHARGE event:
  - txn_id MUST reference a valid user transaction
  - That transaction MUST have a data operation (READ/WRITE)
  - txn_id MUST NOT be a sentinel value (0, -1, etc.) unless explicitly documented
```

**Failure Example:**
```json
// BAD: PRECHARGE uses txn_id=0 as sentinel, disconnected from user request
{"name": "ACTIVATE", "args": {"txn_id": 1, ...}}
{"name": "BURST_READ", "args": {"txn_id": 1, ...}}
{"name": "PRECHARGE", "args": {"txn_id": 0, ...}}  // Should be txn_id=1!
```

### INV-003: Temporal Ordering

**Description:** Events within a transaction must be temporally ordered correctly
according to DRAM protocol.

**Validation:**
```
For each transaction with txn_id=T:
  - If ACTIVATE exists: cycle_issue(ACT) < cycle_issue(READ/WRITE)
  - If PRECHARGE exists: cycle_issue(PRE) > cycle_complete(READ/WRITE)
  - For page hits (no ACTIVATE): data operation can proceed immediately
```

**Failure Example:**
```json
// BAD: READ starts before ACTIVATE completes
{"name": "ACTIVATE", "args": {"txn_id": 1, "cycle_issue": 50, "cycle_complete": 64}}
{"name": "BURST_READ", "args": {"txn_id": 1, "cycle_issue": 55, ...}}  // Should be >= 64!
```

### INV-004: Unique Transaction IDs

**Description:** Each transaction ID should be used for exactly one logical request.
Reusing txn_ids for unrelated requests creates ambiguity.

**Validation:**
```
For all events in trace:
  - Count of unique (txn_id, request_type) pairs should equal number of user requests
  - No txn_id should map to both READ and WRITE operations
```

### INV-005: Bank State Consistency

**Description:** Commands targeting a specific bank must respect the bank's state machine.

**Validation:**
```
For each bank B:
  - Cannot issue ACTIVATE if bank is already active (no double-activate)
  - Cannot issue READ/WRITE if bank is idle (row not open)
  - Cannot issue PRECHARGE if bank is already idle
```

### INV-006: Refresh Belongs to No Transaction

**Description:** A REFRESH is a maintenance command the controller issues on its own schedule. It
serves no request, so it carries `txn_id` 0 (request ids start at 1), and the validator keeps it
out of transaction grouping: INV-001 to INV-004 apply to requests only.

**Rationale:** Without this, the first trace long enough to contain a refresh fails INV-001 with
"txn_id=0 has no data operation (only REFRESH events)" -- a trace that is correct, rejected by a
model that had no refresh in it. Found by #537 (`tests/cycle/cycle_dram_demo`); two committed traces
had the same failure unnoticed (`traces/memory/lpddr5/bandwidth/page_burst_trace.json` with 8
refreshes, `complex/random_trace.json` with 16), and pass with this rule.

**Validation:**
```
For each REFRESH event E:
  - E.txn_id == 0
  - E is not grouped into any transaction
```

**Failure example:** a REFRESH traced with the id of the request whose bank it refreshed, which
would make that request look like it had two owners for its page commands.

**Fix hint:** trace a REFRESH with txn_id 0, never a request's id.

## Timing Constraint Invariants

### INV-100: tRCD Constraint

**Description:** READ/WRITE command must wait for row activation to complete.

**Validation:**
```
For each (ACTIVATE, READ/WRITE) pair with same txn_id and bank:
  cycle_issue(READ/WRITE) >= cycle_complete(ACTIVATE)
```

### INV-101: tRP Constraint

**Description:** ACTIVATE command must wait for precharge to complete.

**Validation:**
```
For consecutive PRECHARGE then ACTIVATE on same bank:
  cycle_issue(ACTIVATE) >= cycle_complete(PRECHARGE)
```

### INV-102: tRRD Constraint

**Description:** Minimum time between consecutive ACTIVATE commands.

**Validation:**
```
For consecutive ACTIVATE commands:
  - Same bank group: gap >= tRRD_L cycles
  - Different bank group: gap >= tRRD_S cycles
```

### INV-103: tFAW Constraint

**Description:** Four Activate Window - maximum 4 ACTIVATE commands in any tFAW window.

**Validation:**
```
In any window of tFAW cycles:
  count(ACTIVATE commands) <= 4
```

### INV-104: Read-to-Write Turnaround (tRTW)

**Description:** Minimum delay when switching from read to write.

**Validation:**
```
For READ followed by WRITE:
  cycle_issue(WRITE) >= cycle_complete(READ) + tRTW
```

### INV-105: Write-to-Read Turnaround (tWTR)

**Description:** Minimum delay when switching from write to read.

**Validation:**
```
For WRITE followed by READ:
  cycle_issue(READ) >= cycle_complete(WRITE) + tWTR
```

## Visualization Invariants

### INV-200: Request Type Detection

**Description:** Request type must be determined by actual data operation, not by absence of READ.

**Validation:**
```
// CORRECT: Explicit check
hasRead = events.some(e => e.name.includes('READ'))
hasWrite = events.some(e => e.name.includes('WRITE'))
if (!hasRead && !hasWrite) SKIP  // Not a user request
type = hasRead ? 'READ' : 'WRITE'

// WRONG: Implicit default
type = hasRead ? 'READ' : 'WRITE'  // Defaults PRECHARGE-only to WRITE!
```

### INV-201: Complete Command Visualization

**Description:** All commands associated with a transaction must be visualized together.

**Rationale:** Timing constraints are relative to ACTIVATE and PRECHARGE. Filtering them
out breaks timing analysis.

## Checker Output

`mflow_check.py` exits 0 when every invariant holds, 1 when one is violated (parse the
violations and fix the root cause), and 2 when the record cannot be read (there are no
violations to parse: fix the recording or the path). With `--json`:

```json
{
  "bundle": "traces/memory/lpddr5/single-bank/page_conflicts.mflow",
  "passed": false,
  "checked": ["TF10", "TF11", "M1", "M2", "M3", "M4", "M6", "M5", "M7", "M9", "M10", "M11"],
  "violations": [
    {"invariant": "M9", "message": "mc[0]/ch[0]/bg[0]/ba[0]: PRE #2 (tick 36, cycle 36) serves no burst"}
  ]
}
```

`--summary` adds what the run did: requests and bursts, commands by kind, page outcomes, bytes
and throughput against the record's ceiling, each bank's commands, and the bursts' latency.

## Integration with Claude Code

Claude Code should:

1. **Pre-generation:** Read INVARIANTS.md to understand constraints
2. **Post-generation:** Run `python3 tools/trace/mflow_check.py <bundle.mflow>` on the record
3. **On failure:** Parse the violations and fix the root cause, never the symptom
4. **Iteration:** Re-run the checker until every invariant holds

A new invariant is an M-check in `mflow_check.py`, with a self-test in `test_mflow_check.py`
that breaks a clean record and sees the check fail (the `add-invariant` skill).

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-05 | 1.0 | Claude Code | Initial invariant documentation |
| 2026-11-24 | 1.1 | Claude Code | INV-006: a refresh belongs to no transaction (#537) |
| 2026-11-26 | 2.0 | Claude Code | The Chrome trace and `trace_validator.py` retired; every invariant mapped to its `mflow_check.py` check (#572) |
