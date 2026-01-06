# Level 2: Two Bank Patterns

Tests for memory controller behavior when accessing two banks, focusing on
activate-to-activate timing constraints.

## Bank Group Architecture

LPDDR5 organizes banks into **bank groups** to enable faster inter-bank access:

```
Bank Groups (4 groups × 4 banks = 16 banks total):

  Group 0: Banks 0, 1, 2, 3
  Group 1: Banks 4, 5, 6, 7
  Group 2: Banks 8, 9, 10, 11
  Group 3: Banks 12, 13, 14, 15
```

## Key Timing Parameters

| Parameter | Cycles | Description |
|-----------|--------|-------------|
| **tRRD_L** | 6 | ACT-to-ACT within same bank group |
| **tRRD_S** | 4 | ACT-to-ACT across different bank groups |

The 33% reduction from tRRD_L to tRRD_S is a significant optimization opportunity!

## Patterns

### same-group/

Tests two banks within the same bank group (e.g., Banks 0 and 1).

**Constraints Tested:**
- tRRD_L (6 cycles) between activate commands
- Page hit/miss behavior within each bank

**Test Cases:**
1. **Interleaved Access** - Alternating between banks 0 and 1
2. **Sequential Bursts** - Burst to bank 0, then burst to bank 1
3. **Page Conflicts** - Different rows in each bank
4. **Full Bank Group** - All 4 banks in group 0

### diff-groups/

Tests two banks in different bank groups (e.g., Banks 0 and 4).

**Constraints Tested:**
- tRRD_S (4 cycles) between activate commands
- Cross-group parallelism benefits

**Test Cases:**
1. **Interleaved Access** - Alternating between banks 0 and 4
2. **Four Groups Round-Robin** - Banks 0, 4, 8, 12 (maximum parallelism)
3. **Timing Comparison** - Same-group vs different-group performance
4. **Mixed Read/Write** - Reads to bank 0, writes to bank 4

## Expected Results

### Same Group (Banks 0, 1)

```
Pattern: Interleaved reads, 8 total
Expected: 8 reads, 6 page hits, 2 page empty, 0 conflicts
Timing: tRRD_L (6 cycles) between bank activates
```

### Different Groups (Banks 0, 4)

```
Pattern: Interleaved reads, 8 total
Expected: 8 reads, 6 page hits, 2 page empty, 0 conflicts
Timing: tRRD_S (4 cycles) between bank activates (faster!)
```

## Running

```bash
# Build
cmake --build build

# Run same-group pattern
./build/patterns/memory/lpddr5/lpddr5_same_group

# Run different-groups pattern
./build/patterns/memory/lpddr5/lpddr5_diff_groups

# With multi-fidelity comparison
./build/patterns/memory/lpddr5/lpddr5_same_group --fidelity

# Export trace for visualization
./build/patterns/memory/lpddr5/lpddr5_diff_groups --trace output.json
```

## Visualization

The trace shows bank group timing clearly in Perfetto:

1. Open https://ui.perfetto.dev
2. Load the trace JSON
3. Observe:
   - ACT command spacing (tRRD_L vs tRRD_S)
   - Bank state transitions
   - Data bus utilization

## Performance Implications

Spreading memory accesses across **different bank groups** is a key optimization:

| Scenario | ACT Spacing | Benefit |
|----------|-------------|---------|
| Same group | 6 cycles (tRRD_L) | - |
| Different groups | 4 cycles (tRRD_S) | 33% faster activate pipelining |

This is particularly important for:
- Tile data prefetching (spread tiles across groups)
- Double-buffering (use different groups for A/B buffers)
- Streaming workloads (round-robin across groups)
