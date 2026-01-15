# DRAM Refresh Requirements

This document provides a comprehensive reference for DRAM refresh timing across all memory technologies supported by KPU-SIM. Understanding refresh behavior is critical for accurate performance modeling and for interpreting pattern test results.

## Overview: Why DRAM Needs Refresh

DRAM stores data as charge in capacitors. This charge leaks over time, and without periodic refresh, data would be lost. The JEDEC specifications define a **retention time** (typically 32-64ms at normal temperature) during which all rows must be refreshed at least once.

The refresh mechanism creates an inherent tension:
- **Data integrity** requires frequent refresh
- **Performance** suffers because refresh blocks memory access

Modern memory controllers employ sophisticated scheduling to minimize this impact.

## Key Timing Parameters

### tREFI (Refresh Interval)

**Definition:** The time interval between refresh commands. This determines *how often* the controller must issue a refresh.

**Key insight:** tREFI does not define how long a refresh takes; it defines the maximum time allowed between refresh commands to ensure all rows are refreshed within the retention window.

**Calculation:**
```
tREFI = Retention_Time / Number_of_Rows_per_Bank / Number_of_Refresh_Commands
```

For DDR5/LPDDR5 with 8192 refresh commands per 32ms retention:
```
tREFI = 32ms / 8192 = 3.9µs
```

### tRFC (Refresh Cycle Time)

**Definition:** The duration of the refresh operation itself. This determines *how long* the DRAM is blocked during refresh.

**Key insight:** During tRFC, the bank (or all banks) cannot accept read/write commands. This is the actual performance penalty.

**Variation:** tRFC scales with memory density because larger arrays take longer to refresh:
- Higher density = more rows = longer tRFC
- This is why tRFC varies from 110ns (DDR3) to 350ns (high-density DDR4)

### Per-Bank vs All-Bank Refresh

**All-Bank Refresh (REFab):**
- Single command refreshes all banks simultaneously
- All banks blocked for tRFCab duration
- Simpler but creates large latency bubbles

**Per-Bank Refresh (REFpb):**
- Separate commands refresh individual banks
- Only one bank blocked at a time (tRFCpb, typically shorter)
- Commands issued 8-16x more frequently (tREFIpb = tREFI/8 or tREFI/16)
- Distributes refresh overhead, reducing worst-case latency

LPDDR4/5 introduced per-bank refresh to improve responsiveness in mobile applications.

## Refresh Timing by Memory Technology

### Reference Table

| Technology | tRFC (duration) | tREFI (interval) | Refresh Commands | Retention | Overhead |
|------------|-----------------|------------------|------------------|-----------|----------|
| DDR3       | 110-160ns       | 7.8µs            | 8192/64ms        | 64ms      | ~1.4-2.0% |
| DDR4       | 260-350ns       | 7.8µs            | 8192/64ms        | 64ms      | ~3.3-4.5% |
| DDR5       | 195-295ns       | 3.9µs            | 8192/32ms        | 32ms      | ~5.0-7.6% |
| LPDDR4     | 130-280ns       | 3.9µs (ab) / 488ns (pb) | 8192/32ms | 32ms | ~3.3-7.2% |
| LPDDR5     | 130-280ns       | 3.9µs (ab) / 488ns (pb) | 8192/32ms | 32ms | ~3.3-7.2% |
| GDDR6      | 180-260ns       | 3.9µs            | 8192/32ms        | 32ms      | ~4.6-6.7% |
| GDDR7      | 150-220ns       | 3.9µs            | 8192/32ms        | 32ms      | ~3.8-5.6% |
| HBM2       | 220-260ns       | 3.9µs            | 8192/32ms        | 32ms      | ~5.6-6.7% |
| HBM2E      | 220-260ns       | 3.9µs            | 8192/32ms        | 32ms      | ~5.6-6.7% |
| HBM3       | 220-260ns       | 3.9µs            | 8192/32ms        | 32ms      | ~5.6-6.7% |
| HBM3E      | 220-260ns       | 3.9µs            | 8192/32ms        | 32ms      | ~5.6-6.7% |

**Notes:**
- tRFC ranges reflect density variation (lower for 4Gb, higher for 16Gb+)
- Overhead = tRFC / tREFI (percentage of time blocked by refresh)
- LPDDR pb = per-bank refresh; ab = all-bank refresh
- Actual overhead may be higher due to scheduling inefficiencies

### Cycle Counts at Common Clock Frequencies

For implementation reference, here are tREFI and tRFC in clock cycles:

| Technology | Clock | tREFI (cycles) | tRFC (cycles) | tRFCpb (cycles) |
|------------|-------|----------------|---------------|-----------------|
| DDR5       | 3.2GHz | 12,480        | 624-944       | N/A             |
| LPDDR5     | 3.2GHz | 12,480 (ab) / 1,560 (pb) | 832 | 416-576 |
| GDDR6      | 2.0GHz | 7,800         | 360-520       | N/A             |
| GDDR7      | 2.5GHz | 9,750         | 375-550       | N/A             |
| HBM2       | 1.0GHz | 3,900         | 220-260       | 110-130         |
| HBM3       | 2.0GHz | 7,800         | 440-520       | 220-260         |

## Refresh Scheduling in Memory Controllers

### The Refresh Queue

Real memory controllers don't simply inject refresh at fixed intervals. They maintain a **refresh queue** with sophisticated scheduling:

```
                    ┌─────────────────────────────────┐
                    │       Refresh Scheduler         │
                    │                                 │
     tREFI timer ──►│  ┌─────────────────────────┐   │
                    │  │     Refresh Queue       │   │
                    │  │  [Bank 0: +1 pending]   │   │
                    │  │  [Bank 1: 0 pending]    │   │
                    │  │  [Bank 2: +2 pending]   │   │◄── Deadline
                    │  │  ...                    │   │    Monitor
                    │  └─────────────────────────┘   │
                    │              │                 │
                    │              ▼                 │
                    │  ┌─────────────────────────┐   │
                    │  │  Opportunistic Inject   │   │
                    │  │  (when bus idle)        │   │
                    │  └─────────────────────────┘   │
                    │              │                 │
                    └──────────────┼─────────────────┘
                                   ▼
                           Issue REF Command
```

### Scheduling Policies

**1. Opportunistic Refresh:**
The controller monitors bus utilization and injects refresh during idle periods. This minimizes impact on active workloads but may delay refresh.

**2. Deadline Enforcement:**
Each bank has a refresh deadline (typically 8-9 × tREFI for DDR). If a bank approaches this deadline, refresh is forced regardless of bus activity. This prevents data loss but creates unpredictable latency spikes.

**3. Refresh Postponement:**
JEDEC allows limited postponement of refresh commands (up to 8 tREFI periods for DDR4/5). The controller can "bank" refresh credits during high activity and catch up during idle periods.

**4. Refresh Pulling:**
Conversely, refresh can be issued early ("pulled in") during low-activity periods to build a buffer against future busy periods.

### Impact on Workloads

**Streaming workloads (e.g., STREAM benchmark):**
- High bus utilization leaves few idle gaps
- Refresh must interrupt active transfers
- Higher effective overhead (can exceed theoretical tRFC/tREFI)

**Bursty workloads (e.g., inference with batch boundaries):**
- Natural idle periods between batches
- Refresh can be opportunistically scheduled
- Lower effective overhead (approaches theoretical minimum)

**Sustained compute (e.g., large GEMM):**
- Long periods of continuous memory access
- Deadline-forced refresh creates latency spikes
- May benefit from explicit refresh scheduling at tile boundaries

## Temperature Effects

Refresh requirements increase at elevated temperatures because charge leakage accelerates:

| Temperature Range | tREFI Adjustment |
|-------------------|------------------|
| 0-85°C (normal)   | 1× (baseline)    |
| 85-95°C           | 0.5× (2× refresh rate) |
| >95°C             | 0.25× (4× refresh rate) |

For data center and HPC applications, thermal management is critical to maintain memory bandwidth.

## Implications for KPU-SIM

### Pattern Tests
When studying specific memory access sequences (page hit/miss patterns, bank interleaving, etc.), automatic refresh can obscure the behavior being analyzed. The simulator should support:
- Disabling refresh for short deterministic tests
- Controlled refresh injection at specific points
- Clear indication when refresh events occur in traces

### Application Simulation
For realistic DNN workload simulation, refresh must be modeled accurately:
- The ~5% bandwidth overhead is significant for memory-bound operations
- Latency spikes from deadline-forced refresh affect tail latency
- Scheduling interactions between data transfers and refresh are important

### Trace Analysis
When analyzing traces, distinguish between:
- **Scheduled refresh:** Opportunistic refresh during idle periods
- **Forced refresh:** Deadline-triggered refresh interrupting transfers
- **Accumulated debt:** Multiple pending refreshes issued back-to-back

## References

1. JEDEC JESD209-5: LPDDR5 Standard
2. JEDEC JESD79-5: DDR5 SDRAM Standard
3. JEDEC JESD235: HBM DRAM Standard
4. "DRAM Refresh Mechanisms, Penalties, and Trade-Offs" - University of Maryland
5. "New Features in (LP)DDR5" - EyeKnowHow Design Resources
6. "DRAM Refresh and Its Impact on Memory Performance" - Chiplog.io

## Version History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-10 | Claude | Initial documentation |
