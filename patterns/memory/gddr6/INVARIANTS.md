# GDDR6 Memory Controller Trace Invariants

This document defines the invariants that all GDDR6 memory controller traces must satisfy.
These invariants are enforced by the trace validator (`trace_validator.py`) and should be
checked before visualizing traces or using them for validation.

## GDDR6 Architecture Overview

### Key Characteristics
- **16 Banks** organized into **4 Bank Groups** (4 banks per group)
- **Dual 16-bit Channels** (x16 mode, two independent channels per chip)
- **16n Prefetch** architecture with Burst Length 16
- **Speed Grades**: 12-24 Gbps (typical: 14, 16, 18, 20, 24 Gbps)
- **WCK Clock**: Data transfer clock (data_rate / 2 for DDR)
- **CK Clock**: Command/Address clock (data_rate / 8)

### Clock Domains
GDDR6 uses multiple clock domains:
- **CK (Command Clock)**: 1/8 of data rate (e.g., 1.75 GHz for 14 Gbps)
- **WCK (Write Clock)**: 1/2 of data rate (e.g., 7.0 GHz for 14 Gbps DDR)
- **Timing parameters** in this document are expressed in CK cycles unless noted

### Memory Organization
```
GDDR6 Device
├── Channel A (16-bit)
│   ├── Bank Group 0
│   │   ├── Bank 0
│   │   ├── Bank 1
│   │   ├── Bank 2
│   │   └── Bank 3
│   ├── Bank Group 1 (Banks 4-7)
│   ├── Bank Group 2 (Banks 8-11)
│   └── Bank Group 3 (Banks 12-15)
└── Channel B (16-bit) [Same structure]
```

## Timing Parameters (GDDR6-16000 @ 2.0 GHz CK)

### Core Timing Parameters

| Parameter | Symbol | Value (CK) | Value (ns) | Description |
|-----------|--------|------------|------------|-------------|
| RAS to CAS (Read) | tRCDRD | 18 | 9.0 | Row address to column address delay (read) |
| RAS to CAS (Write) | tRCDWR | 18 | 9.0 | Row address to column address delay (write) |
| Row Precharge | tRP | 18 | 9.0 | Row precharge time |
| Row Active Time | tRAS | 28 | 14.0 | Minimum row active time |
| Row Cycle Time | tRC | 46 | 23.0 | Row cycle time (tRAS + tRP) |
| Read Latency | tRL | 20 | 10.0 | CAS read latency |
| Write Latency | tWL | 8 | 4.0 | CAS write latency |
| Write Recovery | tWR | 16 | 8.0 | Write recovery time |
| Read to Precharge | tRTP | 8 | 4.0 | Read to precharge delay |

### Bank Group Timing Parameters

| Parameter | Symbol | Same BG | Diff BG | Description |
|-----------|--------|---------|---------|-------------|
| ACT to ACT | tRRD_L/S | 4 | 4 | Minimum activate to activate delay |
| CAS to CAS | tCCD_L/S | 3 | 2 | Column command to column command |
| Write to Read | tWTR_L/S | 6 | 4 | Write to read turnaround |
| Read to Write | tRTW | 14 | 14 | Read to write turnaround |

### Refresh Timing Parameters

| Parameter | Symbol | Value (CK) | Description |
|-----------|--------|------------|-------------|
| Refresh Cycle (per-bank) | tRFCpb | 130 | Per-bank refresh cycle time |
| Refresh Cycle (all-bank) | tRFCab | 260 | All-bank refresh cycle time |
| Refresh Interval | tREFI | 3900 | Average refresh interval |
| Four Activate Window | tFAW | 16 | Max 4 activates in this window |

### Burst Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Burst Length | 16 | Fixed burst length |
| Burst Cycles | 4 | Data bus cycles for one burst (BL16 / 4 = 4 WCK cycles) |
| Data Width | 32-bit | Per channel (16-bit x 2 channels) |

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
according to GDDR6 protocol.

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
{"name": "ACTIVATE", "args": {"txn_id": 1, "cycle_issue": 50, "cycle_complete": 68}}
{"name": "BURST_READ", "args": {"txn_id": 1, "cycle_issue": 55, ...}}  // Should be >= 68!
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

### INV-006: Channel Independence

**Description:** GDDR6 channels operate independently. Commands to different channels
should not have timing dependencies.

**Validation:**
```
For each channel:
  - Timing constraints only apply within the same channel
  - Bank state is tracked per-channel
  - Data bus conflicts only within same channel
```

## Timing Constraint Invariants

### INV-100: tRCDRD/tRCDWR Constraint

**Description:** READ/WRITE command must wait for row activation to complete.

**Validation:**
```
For each (ACTIVATE, READ/WRITE) pair with same txn_id and bank:
  - For READ: cycle_issue(READ) >= cycle_issue(ACTIVATE) + tRCDRD
  - For WRITE: cycle_issue(WRITE) >= cycle_issue(ACTIVATE) + tRCDWR
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
For consecutive ACTIVATE commands on same channel:
  - Same bank group: gap >= tRRD_L cycles (4)
  - Different bank group: gap >= tRRD_S cycles (4)
```

### INV-103: tFAW Constraint

**Description:** Four Activate Window - maximum 4 ACTIVATE commands in any tFAW window.

**Validation:**
```
In any window of tFAW cycles (16):
  count(ACTIVATE commands on same channel) <= 4
```

### INV-104: Read-to-Write Turnaround (tRTW)

**Description:** Minimum delay when switching from read to write on same channel.

**Validation:**
```
For READ followed by WRITE on same channel:
  cycle_issue(WRITE) >= cycle_complete(READ) + tRTW
```

### INV-105: Write-to-Read Turnaround (tWTR)

**Description:** Minimum delay when switching from write to read.

**Validation:**
```
For WRITE followed by READ:
  - Same bank group: cycle_issue(READ) >= cycle_issue(WRITE) + tWTR_L
  - Different bank group: cycle_issue(READ) >= cycle_issue(WRITE) + tWTR_S
```

### INV-106: tCCD Constraint

**Description:** Minimum time between consecutive CAS commands (READ/WRITE).

**Validation:**
```
For consecutive READ/WRITE commands on same channel:
  - Same bank group: gap >= tCCD_L (3)
  - Different bank group: gap >= tCCD_S (2)
```

### INV-107: tRAS Constraint

**Description:** Minimum time a row must remain active before precharge.

**Validation:**
```
For each (ACTIVATE, PRECHARGE) pair on same bank:
  cycle_issue(PRECHARGE) >= cycle_issue(ACTIVATE) + tRAS
```

### INV-108: tRC Constraint

**Description:** Minimum row cycle time (activate to next activate on same bank).

**Validation:**
```
For consecutive ACTIVATE commands on same bank:
  gap >= tRC (46)
```

## GDDR6-Specific Invariants

### INV-200: Dual-Channel Operation

**Description:** Each GDDR6 device has two independent 16-bit channels that can operate
concurrently with separate command/address paths.

**Validation:**
```
- Channel A and Channel B can issue commands independently
- Each channel has its own bank state tracking
- Timing constraints apply per-channel, not across channels
```

### INV-201: Write Clock Domain

**Description:** Data transfers occur in the WCK clock domain which runs at higher
frequency than the command clock.

**Validation:**
```
- Burst completion times account for WCK frequency
- tRL and tWL are specified in CK cycles but data transfer uses WCK
- BL16 burst takes 4 WCK cycles (16 bits / 4 edges per WCK)
```

### INV-202: Bank Group Parallelism

**Description:** Different bank groups can be accessed with shorter timing constraints
than accesses within the same bank group.

**Validation:**
```
- tRRD_S < tRRD_L (or equal for GDDR6)
- tCCD_S < tCCD_L
- tWTR_S < tWTR_L
```

## Visualization Invariants

### INV-300: Request Type Detection

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

### INV-301: Complete Command Visualization

**Description:** All commands associated with a transaction must be visualized together.

**Rationale:** Timing constraints are relative to ACTIVATE and PRECHARGE. Filtering them
out breaks timing analysis.

### INV-302: Channel Separation in Visualization

**Description:** Visualization should clearly distinguish between channels.

**Validation:**
```
- Channel A and Channel B events should be on separate tracks
- Color coding or labeling should identify channel
```

## Validator Output Format

When validation fails, the validator produces structured output:

```json
{
  "status": "FAILED",
  "trace_file": "gddr6_trace.json",
  "violations": [
    {
      "invariant": "INV-001",
      "severity": "ERROR",
      "txn_id": 0,
      "message": "txn_id=0 has no data operation (only PRECHARGE events)",
      "events": [
        {"name": "PRECHARGE", "cycle_issue": 78, "cycle_complete": 96}
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

## Common GDDR6 Bugs and Fixes

### Bug: Wrong timing for bank group operations

**Symptom:** Validator reports INV-106 (tCCD) violation

**Root Cause:** Using tCCD_L timing for cross-bank-group accesses

**Fix:** Check bank group: `bg = bank / 4; same_bg = (bg1 == bg2)`

### Bug: Channel timing conflicts

**Symptom:** Spurious timing violations for independent channel operations

**Root Cause:** Applying single-channel timing rules across both channels

**Fix:** Track timing constraints per-channel, not globally

### Bug: WCK vs CK timing confusion

**Symptom:** Burst completion times off by factor of 4

**Root Cause:** Mixing WCK and CK cycle counts

**Fix:** Express all controller timing in CK cycles; convert burst to CK

## Integration with Claude Code

Claude Code should:

1. **Pre-generation:** Read INVARIANTS.md to understand constraints
2. **Post-generation:** Run trace_validator.py on generated traces
3. **On failure:** Parse validator output and fix identified issues
4. **Iteration:** Re-run validator until all invariants pass

See `CLAUDE.md` in project root for detailed Claude Code integration guidelines.

## Timing Parameter Reference by Speed Grade

### GDDR6-14000 (14 Gbps)
- CK: 1.75 GHz, WCK: 7.0 GHz
- tRCDRD/WR: 16 CK, tRP: 16 CK, tRAS: 25 CK, tRC: 41 CK

### GDDR6-16000 (16 Gbps)
- CK: 2.0 GHz, WCK: 8.0 GHz
- tRCDRD/WR: 18 CK, tRP: 18 CK, tRAS: 28 CK, tRC: 46 CK

### GDDR6-18000 (18 Gbps)
- CK: 2.25 GHz, WCK: 9.0 GHz
- tRCDRD/WR: 20 CK, tRP: 20 CK, tRAS: 32 CK, tRC: 52 CK

### GDDR6-20000 (20 Gbps)
- CK: 2.5 GHz, WCK: 10.0 GHz
- tRCDRD/WR: 22 CK, tRP: 22 CK, tRAS: 36 CK, tRC: 58 CK

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-07 | 1.0 | Claude Code | Initial GDDR6 invariant documentation |
