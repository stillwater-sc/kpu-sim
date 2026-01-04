# LPDDR5 Memory Controller Pattern Test Suite

## Overview

This test suite validates the cycle-accurate LPDDR5 memory controller through progressively more complex memory access patterns. Each pattern exercises specific aspects of DRAM timing, bank management, and data bus behavior.

## Goals

1. **Validate LPDDR5 timing invariants** - Ensure all timing constraints (tRCD, tRP, tRAS, tRRD, tFAW, tWTR, tRTW) are respected
2. **Test bank access patterns** - Page hits, page empty, page conflicts across 1-4 banks
3. **Test turnaround behavior** - Read-to-write and write-to-read bus turnarounds
4. **Drive visualization** - Generate Chrome Trace output for Perfetto animation of memory paths
5. **Build confidence** - Progressive complexity ensures each level builds on validated foundations

---

## Standard LPDDR5 Configurations

### Single Channel (LPDDR5-6400)
```
Channels:     1
Banks:        16 (4 bank groups × 4 banks)
Speed:        6400 MT/s (3200 MHz clock)
Bus width:    16 bits (64 bytes/burst for BL32)
Peak BW:      12.8 GB/s
```

### Dual Channel (LPDDR5-6400)
```
Channels:     2
Banks:        16 per channel (32 total)
Speed:        6400 MT/s (3200 MHz clock)
Bus width:    32 bits total (16 per channel)
Peak BW:      25.6 GB/s
```

---

## Pattern Progression

### Level 1: Single Bank Fundamentals

**Pattern 01: Single Bank - Sequential Access**
- Configuration: Single channel, target bank 0 only
- Access pattern: Sequential reads to same row (page hits)
- What we verify:
  - Page empty (first access) → ACTIVATE + READ timing
  - Page hit (subsequent) → READ timing only
  - tRCD and tCL latency
  - Burst transfer cycles

**Pattern 02: Single Bank - Page Conflict**
- Configuration: Single channel, bank 0 only
- Access pattern: Accesses to different rows
- What we verify:
  - Page conflict detection
  - PRECHARGE + ACTIVATE + READ sequence
  - tRP, tRCD, tRAS timing

**Pattern 03: Single Bank - Mixed Read/Write**
- Configuration: Single channel, bank 0 only
- Access pattern: Alternating reads and writes
- What we verify:
  - Read-to-write turnaround (tRTW)
  - Write-to-read turnaround (tWTR_L)
  - Write recovery time (tWR)

---

### Level 2: Two Bank Operations

**Pattern 04: Two Banks - Same Bank Group**
- Configuration: Single channel, banks 0 and 1 (same group)
- Access pattern: Interleaved accesses
- What we verify:
  - tRRD_L (same bank group activate-to-activate)
  - tCCD_L (same bank group CAS-to-CAS)
  - Bank-level parallelism within group

**Pattern 05: Two Banks - Different Bank Groups**
- Configuration: Single channel, banks 0 and 4 (different groups)
- Access pattern: Interleaved accesses
- What we verify:
  - tRRD_S (different bank group activate-to-activate)
  - tCCD_S (different bank group CAS-to-CAS)
  - Improved parallelism across groups

---

### Level 3: Three Bank Operations

**Pattern 06: Three Banks - Mixed Groups**
- Configuration: Single channel, banks 0, 4, 8 (different groups)
- Access pattern: Round-robin across banks
- What we verify:
  - Multi-bank parallelism
  - Scheduler behavior with 3 candidates
  - Data bus utilization

**Pattern 07: Three Banks - Same Group**
- Configuration: Single channel, banks 0, 1, 2 (same group)
- Access pattern: Sequential activation
- What we verify:
  - Bank group timing limitations
  - Stall cycles when hitting tRRD_L/tCCD_L

---

### Level 4: Four Bank Operations (Full Bank Group)

**Pattern 08: Full Bank Group - Sequential**
- Configuration: Single channel, banks 0-3 (full bank group 0)
- Access pattern: Sequential activations
- What we verify:
  - tFAW (four-activate window) constraint
  - Maximum bank group throughput
  - Stall when hitting tFAW

**Pattern 09: Four Banks - Across Groups**
- Configuration: Single channel, banks 0, 4, 8, 12 (one per group)
- Access pattern: Round-robin
- What we verify:
  - Maximum parallelism (no tFAW limitation)
  - Data bus becomes the bottleneck
  - Peak sustainable bandwidth

**Pattern 10: Four Banks - Page Hit Burst**
- Configuration: Single channel, 4 banks with same row open
- Access pattern: Burst to each bank (page hits only)
- What we verify:
  - Sustained page hit performance
  - Command bus pipelining
  - Data bus saturation

---

### Level 5: Dual Channel Operations

**Pattern 11: Dual Channel - Independent**
- Configuration: Dual channel
- Access pattern: Parallel access to same bank in each channel
- What we verify:
  - Channel independence
  - Doubled peak bandwidth
  - No cross-channel interference

**Pattern 12: Dual Channel - Interleaved**
- Configuration: Dual channel
- Access pattern: Address-interleaved accesses
- What we verify:
  - Channel interleaving efficiency
  - Address decoding correctness
  - Load balancing

---

### Level 6: Complex Access Patterns

**Pattern 13: Strided Access**
- Configuration: Single channel
- Access pattern: Strided reads (modeling matrix column access)
- What we verify:
  - Page conflict rate vs stride
  - Impact on effective bandwidth
  - Bank group spread with different strides

**Pattern 14: Random Access**
- Configuration: Single channel
- Access pattern: Pseudo-random bank/row selection
- What we verify:
  - Worst-case latency
  - Page conflict rate
  - Scheduler fairness

**Pattern 15: Tile Load Pattern**
- Configuration: Dual channel
- Access pattern: 4KB tile load (matches NoC tile size)
- What we verify:
  - Complete tile transfer timing
  - Bank layout impact on tile load time
  - Optimal vs worst-case layouts

---

## Visualization Strategy

Each pattern produces a Chrome Trace JSON file for Perfetto visualization:

```
Bank State Track:     [IDLE] [ACTIVATING...] [ACTIVE] [READING] [ACTIVE] ...
Data Bus Track:       [IDLE] [READ_BURST...] [IDLE] [READ_BURST...] ...
Command Bus Track:    [IDLE] [ACT] [IDLE] [RD] [IDLE] ...
Request Timeline:     [REQ 0 |------latency----->| COMPLETE]
```

### Trace Export

```bash
# Run pattern with trace output
./patterns/pattern01_single_bank_sequential -o trace.json

# Open in Perfetto
# https://ui.perfetto.dev
```

---

## Directory Structure

```
patterns/
├── PLAN.md                              # This file
├── ARCHITECTURE.md                      # LPDDR5 timing reference
├── VOCABULARY.md                        # Terminology definitions
├── common/
│   ├── lpddr5_configs.hpp              # Standard SC/DC configurations
│   ├── pattern_harness.hpp             # Test harness base class
│   ├── trace_validator.hpp             # Trace verification helpers
│   └── tile_geometry.hpp               # Tile dimensions
├── level1_single_bank/
│   ├── pattern01_sequential/
│   │   ├── main.cpp
│   │   └── README.md
│   ├── pattern02_page_conflict/
│   │   ├── main.cpp
│   │   └── README.md
│   └── pattern03_mixed_rw/
│       ├── main.cpp
│       └── README.md
├── level2_two_banks/
│   ├── pattern04_same_group/
│   ├── pattern05_diff_groups/
├── level3_three_banks/
│   ├── pattern06_mixed_groups/
│   └── pattern07_same_group/
├── level4_four_banks/
│   ├── pattern08_full_group/
│   ├── pattern09_across_groups/
│   └── pattern10_page_hit_burst/
├── level5_dual_channel/
│   ├── pattern11_independent/
│   └── pattern12_interleaved/
├── level6_complex/
│   ├── pattern13_strided/
│   ├── pattern14_random/
│   └── pattern15_tile_load/
├── expected_traces/                     # Golden reference traces
│   ├── pattern01_expected.json
│   └── ...
└── CMakeLists.txt
```

---

## Success Criteria

Each pattern must:

1. **Complete without invariant violations** - No LPDDR5 timing violations
2. **Match expected statistics** - Page hits/misses/conflicts as predicted
3. **Produce valid trace** - Chrome Trace format for visualization
4. **Be deterministic** - Same input → same cycle counts

---

## Implementation Order

1. **Common infrastructure**
   - Pattern harness with LPDDR5 controller integration
   - Standard configurations (single/dual channel)
   - Trace export utilities

2. **Level 1** - Single bank fundamentals (patterns 01-03)
3. **Level 2** - Two bank operations (patterns 04-05)
4. **Level 3** - Three bank operations (patterns 06-07)
5. **Level 4** - Four bank operations (patterns 08-10)
6. **Level 5** - Dual channel (patterns 11-12)
7. **Level 6** - Complex patterns (patterns 13-15)

---

## Expected Timing Reference

### LPDDR5-6400 Key Timings (cycles @ 3200 MHz)

| Parameter | Cycles | Description |
|-----------|--------|-------------|
| tRCD | 14 | Row address to column address delay |
| tRP | 14 | Row precharge time |
| tRAS | 28 | Row active time (minimum) |
| tRC | 42 | Row cycle time (tRAS + tRP) |
| tCL | 14 | CAS read latency |
| tWL | 8 | CAS write latency |
| tWR | 24 | Write recovery time |
| tRTP | 6 | Read to precharge |
| tRRD_L | 6 | ACT to ACT (same bank group) |
| tRRD_S | 4 | ACT to ACT (different bank group) |
| tCCD_L | 6 | CAS to CAS (same bank group) |
| tCCD_S | 4 | CAS to CAS (different bank group) |
| tWTR_L | 10 | Write to read (same bank group) |
| tWTR_S | 4 | Write to read (different bank group) |
| tRTW | 14 | Read to write (bus turnaround) |
| tFAW | 24 | Four activate window |
| tBurst (BL16) | 8 | Burst length 16 |
| tBurst (BL32) | 16 | Burst length 32 |

### Access Latency Expectations

| Scenario | Minimum Latency (cycles) |
|----------|-------------------------|
| Page hit read | tCL + tBurst = 22 |
| Page empty read | tRCD + tCL + tBurst = 36 |
| Page conflict read | tRP + tRCD + tCL + tBurst = 50 |
| Page hit write | tWL + tBurst = 16 |
| Page empty write | tRCD + tWL + tBurst = 30 |
| Page conflict write | tRP + tRCD + tWL + tBurst = 44 |
