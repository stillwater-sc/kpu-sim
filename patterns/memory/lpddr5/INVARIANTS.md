# LPDDR5 Memory Controller Trace Invariants

This document defines the invariants that all LPDDR5 memory controller traces must satisfy.
These invariants are enforced by the trace validator (`trace_validator.py`) and should be
checked before visualizing traces or using them for validation.

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

## Validator Output Format

When validation fails, the validator produces structured output:

```json
{
  "status": "FAILED",
  "trace_file": "page_conflicts_trace.json",
  "violations": [
    {
      "invariant": "INV-001",
      "severity": "ERROR",
      "txn_id": 0,
      "message": "txn_id=0 has no data operation (only PRECHARGE events)",
      "events": [
        {"name": "PRECHARGE", "cycle_issue": 78, "cycle_complete": 92},
        {"name": "PRECHARGE", "cycle_issue": 128, "cycle_complete": 142}
      ],
      "fix_hint": "PRECHARGE should have same txn_id as the READ/WRITE that opened the page"
    }
  ],
  "summary": {
    "total_transactions": 5,
    "valid_transactions": 4,
    "invalid_transactions": 1,
    "errors": 1,
    "warnings": 0
  }
}
```

## Integration with Claude Code

Claude Code should:

1. **Pre-generation:** Read INVARIANTS.md to understand constraints
2. **Post-generation:** Run trace_validator.py on generated traces
3. **On failure:** Parse validator output and fix identified issues
4. **Iteration:** Re-run validator until all invariants pass

See `CLAUDE.md` in project root for detailed Claude Code integration guidelines.

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-05 | 1.0 | Claude Code | Initial invariant documentation |
