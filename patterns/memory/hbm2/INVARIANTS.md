# HBM2 Memory Controller Trace Invariants

This document defines the invariants that all HBM2 memory controller traces must satisfy.
These invariants are enforced by the trace validator (`trace_validator.py`) and should be
checked before visualizing traces or using them for validation.

## HBM2 Architecture Overview

### Key Characteristics
- **8 Channels** per stack (128-bit each)
- **2 Pseudo-Channels** per channel (64-bit each, 16 total)
- **16 Banks** per pseudo-channel (256 total per stack)
- **4 Bank Groups** per pseudo-channel (4 banks per group)
- **1024-bit Total I/O** (8 channels × 128-bit)
- **Speed Grades**: HBM2 (2.0 Gbps), HBM2E (3.6 Gbps)
- **CK Clock**: 1.0 GHz for HBM2-2000, 1.8 GHz for HBM2E-3600
- **Burst Length**: BL4 (pseudo-channel mode)

### Memory Organization
```
HBM2 Stack
├── Channel 0 (128-bit, DQ[127:0])
│   ├── Pseudo-Channel 0 (64-bit, DQ[63:0])
│   │   ├── Bank Group 0 (Banks 0-3)
│   │   ├── Bank Group 1 (Banks 4-7)
│   │   ├── Bank Group 2 (Banks 8-11)
│   │   └── Bank Group 3 (Banks 12-15)
│   └── Pseudo-Channel 1 (64-bit, DQ[127:64])
│       └── [Same 16-bank structure]
├── Channel 1 (DQ[255:128])
│   └── [Same PC0/PC1 structure]
├── ...
└── Channel 7 (DQ[1023:896])
    └── [Same PC0/PC1 structure]
```

### Bank ID Encoding
The bank_id in traces encodes channel, pseudo-channel, and bank:
```
bank_id = channel * 32 + pc * 16 + bank
         (0-7)    (0-1)    (0-15)

Example: bank_id = 50
  channel = 50 / 32 = 1
  remainder = 50 % 32 = 18
  pc = 18 / 16 = 1
  bank = 18 % 16 = 2
  -> Channel 1, PC 1, Bank 2
```

## Timing Parameters (HBM2-2000 @ 1.0 GHz CK)

### Core Timing Parameters

| Parameter | Symbol | Value (CK) | Value (ns) | Description |
|-----------|--------|------------|------------|-------------|
| RAS to CAS (Read) | tRCDRD | 12 | 12.0 | Row to column delay (read) |
| RAS to CAS (Write) | tRCDWR | 6 | 6.0 | Row to column delay (write) |
| Row Precharge | tRP | 14 | 14.0 | Row precharge time |
| Row Active Time | tRAS | 28 | 28.0 | Minimum row active time |
| Row Cycle Time | tRC | 42 | 42.0 | Row cycle time |
| Read Latency | tRL | 18 | 18.0 | CAS read latency |
| Write Latency | tWL | 7 | 7.0 | CAS write latency |
| Write Recovery | tWR | 16 | 16.0 | Write recovery time |
| Read to Precharge | tRTP | 6 | 6.0 | Read to precharge delay |

### Bank Group Timing Parameters

| Parameter | Symbol | Same BG | Diff BG | Description |
|-----------|--------|---------|---------|-------------|
| ACT to ACT | tRRD_L/S | 4 | 3 | Minimum activate to activate delay |
| CAS to CAS | tCCD_L/S | 4 | 2 | Column command to column command |
| Write to Read | tWTR_L/S | 8 | 4 | Write to read turnaround |
| Read to Write | tRTW | 10 | 10 | Read to write turnaround |

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
| Burst Length | 4 | BL4 in pseudo-channel mode |
| Burst Cycles | 2 | Data bus cycles for one burst |
| Data Width | 64-bit | Per pseudo-channel |

## Trace Structure Invariants

### INV-001: Valid Transaction ID Semantics

**Description:** Every `txn_id` must represent exactly ONE user request (READ or WRITE).
DRAM commands (ACTIVATE, PRECHARGE) generated to satisfy that request must share the same `txn_id`.

**Rationale:** A user request may require multiple DRAM commands. For example, a READ to a closed
page requires: ACTIVATE (open row) -> READ (column access). These commands must be traceable
back to the originating user request.

**Validation:**
```
For each txn_id:
  - MUST have exactly ONE data operation (BURST_READ or BURST_WRITE)
  - MAY have zero or one ACTIVATE
  - MAY have zero or one PRECHARGE
  - All events with same txn_id belong to the same logical request
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
```

### INV-003: Temporal Ordering

**Description:** Events within a transaction must be temporally ordered correctly
according to HBM2 protocol.

**Validation:**
```
For each transaction with txn_id=T:
  - If ACTIVATE exists: cycle_issue(ACT) < cycle_issue(READ/WRITE)
  - If PRECHARGE exists: cycle_issue(PRE) > cycle_complete(READ/WRITE)
  - For page hits (no ACTIVATE): data operation can proceed immediately
```

### INV-004: Unique Transaction IDs

**Description:** Each transaction ID should be used for exactly one logical request.
Reusing txn_ids for unrelated requests creates ambiguity.

### INV-005: Bank State Consistency

**Description:** Commands targeting a specific bank must respect the bank's state machine.

**Validation:**
```
For each bank B in each pseudo-channel:
  - Cannot issue ACTIVATE if bank is already active
  - Cannot issue READ/WRITE if bank is idle (row not open)
  - Cannot issue PRECHARGE if bank is already idle
```

### INV-006: Pseudo-Channel Independence

**Description:** HBM2 pseudo-channels operate independently within a channel. Each PC has
its own command/address path and 64-bit data bus.

**Validation:**
```
For each pseudo-channel:
  - Timing constraints only apply within the same PC
  - Bank state is tracked per-PC
  - Data bus conflicts only within same PC
```

## Timing Constraint Invariants

### INV-100: tRCDRD/tRCDWR Constraint

**Description:** READ/WRITE command must wait for row activation to complete.

**Validation:**
```
For each (ACTIVATE, READ/WRITE) pair with same txn_id and bank:
  - For READ: cycle_issue(READ) >= cycle_issue(ACTIVATE) + tRCDRD (12)
  - For WRITE: cycle_issue(WRITE) >= cycle_issue(ACTIVATE) + tRCDWR (6)
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
For consecutive ACTIVATE commands on same pseudo-channel:
  - Same bank group: gap >= tRRD_L (4)
  - Different bank group: gap >= tRRD_S (3)
```

### INV-103: tFAW Constraint

**Description:** Four Activate Window - maximum 4 ACTIVATE commands in any tFAW window.

**Validation:**
```
In any window of tFAW cycles (16):
  count(ACTIVATE commands on same pseudo-channel) <= 4
```

### INV-106: tCCD Constraint

**Description:** Minimum time between consecutive CAS commands (READ/WRITE).

**Validation:**
```
For consecutive READ/WRITE commands on same pseudo-channel:
  - Same bank group: gap >= tCCD_L (4)
  - Different bank group: gap >= tCCD_S (2)
```

### INV-107: tRAS Constraint

**Description:** Minimum time a row must remain active before precharge.

**Validation:**
```
For each (ACTIVATE, PRECHARGE) pair on same bank:
  cycle_issue(PRECHARGE) >= cycle_issue(ACTIVATE) + tRAS (28)
```

### INV-108: tRC Constraint

**Description:** Minimum row cycle time (activate to next activate on same bank).

**Validation:**
```
For consecutive ACTIVATE commands on same bank:
  gap >= tRC (42)
```

## HBM2-Specific Invariants

### INV-200: Pseudo-Channel Operation

**Description:** Each channel has two pseudo-channels that share the physical pins but
operate as independent 64-bit interfaces.

**Validation:**
```
- PC0 and PC1 within same channel can be accessed concurrently
- Each PC has its own bank state tracking
- Timing constraints apply per-PC, not per-channel
```

### INV-201: Bank Group Layout

**Description:** Each pseudo-channel has 16 banks organized into 4 bank groups.

**Validation:**
```
Bank group = bank / 4
  Group 0: Banks 0-3
  Group 1: Banks 4-7
  Group 2: Banks 8-11
  Group 3: Banks 12-15
```

### INV-202: 1KB Page Size

**Description:** HBM2 pseudo-channel mode uses 1KB pages (row size).

**Validation:**
```
- Column address bits: 6 (64 columns × 16 bytes = 1KB)
- Page hit detection based on row match within same bank
```

## Visualization Invariants

### INV-300: Request Type Detection

**Description:** Request type must be determined by actual data operation, not by absence of READ.

**Validation:**
```
hasRead = events.some(e => e.name.includes('READ'))
hasWrite = events.some(e => e.name.includes('WRITE'))
if (!hasRead && !hasWrite) SKIP  // Not a user request
type = hasRead ? 'READ' : 'WRITE'
```

### INV-301: Hierarchical Lane Organization

**Description:** Visualization should show the channel/PC/bank hierarchy.

**Validation:**
```
- Group by channel (0-7)
- Within channel, group by pseudo-channel (0-1)
- Within PC, show banks and data bus lanes
```

### INV-302: DQ Pin Mapping

**Description:** Display should show which DQ pins each PC uses.

**Validation:**
```
For Channel C, PC P:
  DQ_start = (C * 2 + P) * 64
  DQ_end = DQ_start + 63
  Display: "DQ[{DQ_end}:{DQ_start}]"
```

## Validator Output Format

When validation fails, the validator produces structured output:

```json
{
  "status": "FAILED",
  "trace_file": "hbm2_trace.json",
  "violations": [
    {
      "invariant": "INV-001",
      "severity": "ERROR",
      "txn_id": 0,
      "message": "txn_id=0 has no data operation",
      "events": [...],
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

## Common HBM2 Bugs and Fixes

### Bug: Wrong bank group calculation

**Symptom:** Validator reports tRRD or tCCD violations

**Root Cause:** Not accounting for pseudo-channel in bank group calculation

**Fix:** Extract bank within PC first: `bank = bank_id % 16; bg = bank / 4`

### Bug: Cross-PC timing applied

**Symptom:** False timing violations for independent PC operations

**Root Cause:** Applying timing constraints across pseudo-channels

**Fix:** Track timing per (channel, pc) pair, not just per channel

### Bug: Channel/PC/Bank confusion

**Symptom:** Wrong bank addressed in traces

**Root Cause:** Incorrect bank_id encoding/decoding

**Fix:** Use consistent formula: `bank_id = channel * 32 + pc * 16 + bank`

## Timing Parameter Reference by Speed Grade

### HBM2-2000 (2.0 Gbps, 1.0 GHz CK)
- tRCDRD: 12, tRCDWR: 6, tRP: 14, tRAS: 28, tRC: 42
- tRL: 18, tWL: 7, tBurst: 2
- Peak Bandwidth: 256 GB/s (8 channels × 128-bit × 2 Gbps / 8)

### HBM2E-3600 (3.6 Gbps, 1.8 GHz CK)
- tRCDRD: 7, tRCDWR: 4, tRP: 8, tRAS: 16, tRC: 24
- tRL: 10, tWL: 4, tBurst: 2
- Peak Bandwidth: 461 GB/s

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-08 | 1.0 | Claude Code | Initial HBM2 invariant documentation |
