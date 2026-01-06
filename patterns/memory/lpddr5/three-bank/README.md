# Level 3: Three Bank Patterns

Tests for memory controller behavior with three concurrent banks, comparing
same-group vs mixed-group performance.

## Overview

Three-bank patterns demonstrate the scaling of bank-level parallelism and
the impact of bank group selection on activate timing.

## Bank Group Selection

| Configuration | Banks | Groups | ACT-to-ACT |
|--------------|-------|--------|------------|
| Same Group | 0, 1, 2 | All BG0 | tRRD_L (6 cycles) |
| Mixed Groups | 0, 4, 8 | BG0, BG1, BG2 | tRRD_S (4 cycles) |

## Patterns

### mixed-groups/

Three banks from different bank groups (0, 4, 8).

**Benefits:**
- tRRD_S (4 cycles) between all activates
- Maximum bank-level parallelism
- Better command scheduling flexibility

**Test Cases:**
1. **Round-Robin** - Cyclic access: 0→4→8→0→4→8→...
2. **Sequential Bursts** - Burst to bank 0, then 4, then 8
3. **Page Conflicts** - Different rows per bank each round
4. **Mixed R/W** - Reads to bank 0,8; writes to bank 4
5. **Asymmetric Load** - Heavy/medium/light load distribution

### same-group/

Three banks from the same bank group (0, 1, 2).

**Limitations:**
- tRRD_L (6 cycles) between all activates
- Bank group becomes a bottleneck
- Reduced scheduling flexibility

**Test Cases:**
1. **Round-Robin** - Cyclic access: 0→1→2→0→1→2→...
2. **Sequential Bursts** - Burst to bank 0, then 1, then 2
3. **Page Conflicts** - Different rows per bank each round
4. **Timing Comparison** - Direct comparison with mixed groups
5. **Mixed R/W** - Reads to bank 0,2; writes to bank 1

## Expected Results

### Mixed Groups (Banks 0, 4, 8)

```
9 reads (3 per bank), round-robin
Expected: 9 reads, 6 page hits, 3 page empty, 0 conflicts
Timing: tRRD_S (4 cycles) between bank activates
```

### Same Group (Banks 0, 1, 2)

```
9 reads (3 per bank), round-robin
Expected: 9 reads, 6 page hits, 3 page empty, 0 conflicts
Timing: tRRD_L (6 cycles) between bank activates
```

## Running

```bash
# Build
cmake --build build

# Run mixed-groups pattern
./build/patterns/memory/lpddr5/lpddr5_three_mixed_groups

# Run same-group pattern
./build/patterns/memory/lpddr5/lpddr5_three_same_group

# With trace export
./build/patterns/memory/lpddr5/lpddr5_three_mixed_groups --trace output.json
```

## Performance Analysis

The timing comparison test directly measures the impact of bank group selection:

```
Same group (tRRD_L):   X cycles
Mixed groups (tRRD_S): Y cycles
Difference: (X - Y) cycles savings
```

### When Does Bank Group Matter?

Bank group selection has the **most impact** when:
1. Multiple banks are being activated in quick succession
2. The workload is activate-bound (not data-bus bound)
3. Queue depths are low (commands execute immediately)

Bank group selection has **less impact** when:
1. Page hits dominate (no new activates needed)
2. Data bus is the bottleneck (long bursts)
3. Queue is deep (scheduling hides latency)

## Architectural Implications

For KPU tile data movement:

| Strategy | Bank Selection | Advantage |
|----------|---------------|-----------|
| Tile per bank group | Banks 0,4,8,12 | Maximum activate parallelism |
| Double buffering | Different groups | Overlap load/compute |
| Streaming | Round-robin across groups | Sustained bandwidth |
