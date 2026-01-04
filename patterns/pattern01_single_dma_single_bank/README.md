# Pattern 01: LPDDR5 Bank Access Patterns

## Overview

This pattern validates the cycle-accurate LPDDR5 memory controller through progressively complex bank access patterns. It tests:

1. **Single Bank Page Hits** - Sequential reads to same row
2. **Single Bank Page Conflicts** - Reads to different rows
3. **Two Banks Same Group** - tRRD_L timing between activates
4. **Two Banks Different Groups** - tRRD_S timing (faster)
5. **Three Banks Mixed Groups** - Multi-bank parallelism
6. **Four Banks Full Group** - tFAW constraint testing
7. **Four Banks Across Groups** - Maximum parallelism
8. **Mixed Read/Write** - Bus turnaround (tRTW, tWTR)

## Configuration

Uses single-channel LPDDR5-6400:
- 1 channel
- 16 banks (4 bank groups × 4 banks per group)
- BL16 burst length (8 cycles)

## Key Timing Parameters

| Parameter | Cycles | Description |
|-----------|--------|-------------|
| tRCD | 14 | Row address to column address delay |
| tRP | 14 | Row precharge time |
| tCL | 14 | CAS read latency |
| tRRD_L | 6 | ACT to ACT (same bank group) |
| tRRD_S | 4 | ACT to ACT (different bank group) |
| tRTW | 14 | Read to write turnaround |
| tWTR_L | 10 | Write to read (same bank group) |

## Expected Latencies

| Scenario | Latency (cycles) |
|----------|------------------|
| Page hit read | 22 (tCL + tBurst) |
| Page empty read | 36 (tRCD + tCL + tBurst) |
| Page conflict read | 50 (tRP + tRCD + tCL + tBurst) |

## Running

```bash
# Build
cd build
cmake .. && make pattern01_bank_access

# Run tests
./patterns/pattern01_bank_access

# With trace output
./patterns/pattern01_bank_access my_trace.json
```

## Output

The pattern produces:
- Console output with test results and statistics
- Chrome Trace JSON file for Perfetto visualization

## Visualization

Open the trace file in Perfetto:
1. Go to https://ui.perfetto.dev
2. Drag and drop the JSON file
3. Explore bank states, data bus activity, and command timing

## Success Criteria

- All tests pass without invariant violations
- Statistics match expected page hits/conflicts
- Trace is valid and shows expected timing behavior
