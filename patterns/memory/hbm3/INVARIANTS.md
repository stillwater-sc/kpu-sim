# HBM3 Memory Controller Trace Invariants

This document defines the invariants that all HBM3 memory controller traces must satisfy.
These invariants are enforced by the trace validator (`trace_validator.py`) and should be
checked before visualizing traces or using them for validation.

## HBM3 Architecture Overview

### Key Characteristics
- **16 Channels** per stack (64-bit each)
- **2 Pseudo-Channels** per channel (32-bit each, 32 total)
- **16 Banks** per pseudo-channel (512 total per stack)
- **4 Bank Groups** per pseudo-channel (4 banks per group)
- **1024-bit Total I/O** (16 channels × 64-bit)
- **Speed Grades**: HBM3 (5.6 Gbps), HBM3E (9.6 Gbps)
- **CK Clock**: 2.8 GHz for HBM3-5600, 4.8 GHz for HBM3E-9600
- **Burst Length**: BL8 (pseudo-channel mode)

### Memory Organization
```
HBM3 Stack
├── Channel 0 (64-bit, DQ[63:0])
│   ├── Pseudo-Channel 0 (32-bit, DQ[31:0])
│   │   ├── Bank Group 0 (Banks 0-3)
│   │   ├── Bank Group 1 (Banks 4-7)
│   │   ├── Bank Group 2 (Banks 8-11)
│   │   └── Bank Group 3 (Banks 12-15)
│   └── Pseudo-Channel 1 (32-bit, DQ[63:32])
│       └── [Same 16-bank structure]
├── Channel 1 (DQ[127:64])
│   └── [Same PC0/PC1 structure]
├── ...
└── Channel 15 (DQ[1023:960])
    └── [Same PC0/PC1 structure]
```

### Bank ID Encoding
The bank_id in traces encodes channel, pseudo-channel, and bank:
```
bank_id = channel * 32 + pc * 16 + bank
         (0-15)   (0-1)    (0-15)

Example: bank_id = 82
  channel = 82 / 32 = 2
  remainder = 82 % 32 = 18
  pc = 18 / 16 = 1
  bank = 18 % 16 = 2
  -> Channel 2, PC 1, Bank 2
```

## Timing Parameters (HBM3-5600 @ 2.8 GHz CK)

### Core Timing Parameters

| Parameter | Symbol | Value (CK) | Value (ns) | Description |
|-----------|--------|------------|------------|-------------|
| RAS to CAS | tRCD | 8 | 2.86 | Row to column delay |
| Row Precharge | tRP | 8 | 2.86 | Row precharge time |
| Row Active Time | tRAS | 16 | 5.71 | Minimum row active time |
| Row Cycle Time | tRC | 24 | 8.57 | Row cycle time |
| Read Latency | tRL | 8 | 2.86 | CAS read latency |
| Write Latency | tWL | 4 | 1.43 | CAS write latency |
| Write Recovery | tWR | 12 | 4.29 | Write recovery time |
| Read to Precharge | tRTP | 4 | 1.43 | Read to precharge delay |

### Bank Group Timing Parameters

| Parameter | Symbol | Same BG | Diff BG | Description |
|-----------|--------|---------|---------|-------------|
| ACT to ACT | tRRD_L/S | 4 | 2 | Minimum activate to activate delay |
| CAS to CAS | tCCD_L/S | 4 | 2 | Column command to column command |
| Write to Read | tWTR_L/S | 6 | 3 | Write to read turnaround |
| Read to Write | tRTW | 8 | 8 | Read to write turnaround |

### Refresh Timing Parameters

| Parameter | Symbol | Value (CK) | Description |
|-----------|--------|------------|-------------|
| Refresh Cycle (per-bank) | tRFCpb | 130 | Per-bank refresh cycle time |
| Refresh Cycle (all-bank) | tRFCab | 260 | All-bank refresh cycle time |
| Refresh Interval | tREFI | 1950 | Average refresh interval |
| Four Activate Window | tFAW | 16 | Max 4 activates in this window |

### Burst Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Burst Length | 8 | BL8 in pseudo-channel mode |
| Burst Cycles | 4 | Data bus cycles for one burst |
| Data Width | 32-bit | Per pseudo-channel |

## Trace Structure Invariants

### INV-001: Valid Transaction ID Semantics

**Description:** Every `txn_id` must represent exactly ONE user request (READ or WRITE).
DRAM commands (ACTIVATE, PRECHARGE) generated to satisfy that request must share the same `txn_id`.

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

**Validation:**
```
For each ACTIVATE/PRECHARGE event:
  - txn_id MUST reference a valid user transaction
  - That transaction MUST have a data operation (READ/WRITE)
```

### INV-003: Temporal Ordering

**Description:** Events within a transaction must be temporally ordered correctly
according to HBM3 protocol.

**Validation:**
```
For each transaction with txn_id=T:
  - If ACTIVATE exists: cycle_issue(ACT) < cycle_issue(READ/WRITE)
  - If PRECHARGE exists: cycle_issue(PRE) > cycle_complete(READ/WRITE)
  - For page hits (no ACTIVATE): data operation can proceed immediately
```

### INV-004: Unique Transaction IDs

**Description:** Each transaction ID should be used for exactly one logical request.

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

**Description:** HBM3 pseudo-channels operate independently within a channel.

**Validation:**
```
For each pseudo-channel:
  - Timing constraints only apply within the same PC
  - Bank state is tracked per-PC
  - Data bus conflicts only within same PC
```

## Timing Constraint Invariants

### INV-100: tRCD Constraint

**Description:** READ/WRITE command must wait for row activation to complete.

**Validation:**
```
For each (ACTIVATE, READ/WRITE) pair with same txn_id and bank:
  cycle_issue(READ/WRITE) >= cycle_issue(ACTIVATE) + tRCD (8)
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
  - Different bank group: gap >= tRRD_S (2)
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
  cycle_issue(PRECHARGE) >= cycle_issue(ACTIVATE) + tRAS (16)
```

### INV-108: tRC Constraint

**Description:** Minimum row cycle time (activate to next activate on same bank).

**Validation:**
```
For consecutive ACTIVATE commands on same bank:
  gap >= tRC (24)
```

## HBM3-Specific Invariants

### INV-200: 16-Channel Operation

**Description:** HBM3 doubles the channel count from HBM2, with each channel having
narrower but faster pseudo-channels.

**Validation:**
```
- 16 independent channels (vs 8 for HBM2)
- Each channel has 64-bit I/O (vs 128-bit for HBM2)
- Each PC has 32-bit I/O (vs 64-bit for HBM2)
```

### INV-201: BL8 Burst Length

**Description:** HBM3 uses BL8 bursts for higher efficiency.

**Validation:**
```
- Burst Length: 8 transfers
- Burst Cycles: 4 CK cycles (DDR)
- Per-PC transfer: 32-bit × 8 = 256 bits = 32 bytes
```

### INV-202: Shorter Timing Cycles

**Description:** HBM3's higher clock frequency results in lower cycle counts for
the same absolute timing requirements.

**Validation:**
```
- tRCD: 8 cycles (vs 12 for HBM2)
- tRP: 8 cycles (vs 14 for HBM2)
- tRAS: 16 cycles (vs 28 for HBM2)
```

## Visualization Invariants

### INV-300: Request Type Detection

**Description:** Request type must be determined by actual data operation.

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
- Group by channel (0-15)
- Within channel, group by pseudo-channel (0-1)
- Within PC, show banks and data bus lanes
```

### INV-302: DQ Pin Mapping

**Description:** Display should show which DQ pins each PC uses.

**Validation:**
```
For Channel C, PC P:
  DQ_start = (C * 2 + P) * 32
  DQ_end = DQ_start + 31
  Display: "DQ[{DQ_end}:{DQ_start}]"
```

## Validator Output Format

```json
{
  "status": "FAILED",
  "trace_file": "hbm3_trace.json",
  "violations": [
    {
      "invariant": "INV-001",
      "severity": "ERROR",
      "txn_id": 0,
      "message": "txn_id=0 has no data operation",
      "events": [...],
      "fix_hint": "..."
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

## Timing Parameter Reference by Speed Grade

### HBM3-5600 (5.6 Gbps, 2.8 GHz CK)
- tRCD: 8, tRP: 8, tRAS: 16, tRC: 24
- tRL: 8, tWL: 4, tBurst: 4
- Peak Bandwidth: 716.8 GB/s (16 channels × 64-bit × 5.6 Gbps / 8)

### HBM3E-9600 (9.6 Gbps, 4.8 GHz CK)
- tRCD: 5, tRP: 5, tRAS: 10, tRC: 14
- tRL: 5, tWL: 3, tBurst: 4
- Peak Bandwidth: 1228.8 GB/s

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-08 | 1.0 | Claude Code | Initial HBM3 invariant documentation |
