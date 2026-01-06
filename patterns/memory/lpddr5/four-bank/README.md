# Level 4: Four Bank Patterns

Tests for memory controller behavior with four concurrent banks, including
the critical tFAW (Four Activate Window) constraint.

## Key Constraint: tFAW

**tFAW (Four Activate Window) = 24 cycles**

This constraint limits how quickly four ACT commands can be issued:
- Maximum 4 activates within any rolling 24-cycle window
- Prevents excessive current draw from simultaneous row activations
- Becomes the bottleneck when activating many banks quickly

### tFAW Interaction with tRRD

| Configuration | ACT Spacing | 4 ACTs in | tFAW Limited? |
|--------------|-------------|-----------|---------------|
| Same group (tRRD_L=6) | 6 cycles | 18 cycles | No (18 < 24) |
| Diff groups (tRRD_S=4) | 4 cycles | 12 cycles | No (12 < 24) |
| 5th ACT same group | - | - | Yes, wait until 24 |

## Patterns

### full-group/

All four banks in one bank group (0, 1, 2, 3).

**Constraints:**
- tRRD_L (6 cycles) between activates
- tFAW (24 cycles) for 4-activate window

**Test Cases:**
1. **Round-Robin** - Cyclic access through all 4 banks
2. **Sequential Bursts** - Burst to each bank in sequence
3. **Page Conflicts** - Different rows causing conflicts
4. **Sustained Stress** - 8 rounds of 4-bank access
5. **Mixed R/W** - Reads to banks 0,2; writes to banks 1,3

### across-groups/

Four banks from different groups (0, 4, 8, 12).

**Benefits:**
- tRRD_S (4 cycles) between all activates
- No tFAW concern (one bank per group)
- Maximum activate parallelism

**Test Cases:**
1. **Round-Robin** - Optimal 4-bank access pattern
2. **Timing Comparison** - Full-group vs across-groups
3. **Sustained** - Long-running access pattern
4. **Mixed R/W** - Reads to banks 0,8; writes to banks 4,12
5. **Page Conflicts** - Conflict handling across groups

### page-hit-burst/

Sustained page hits across four banks for peak throughput.

**Strategy:**
- Open pages in 4 banks (preferably across groups)
- Burst page hits to maximize data bus utilization
- Minimal activate overhead

**Test Cases:**
1. **Basic Burst** - 4 opens + 16 page hits
2. **Sustained** - 4 opens + 64 page hits (>90% hit ratio)
3. **Double Buffer** - Read banks 0,4; write banks 8,12
4. **Hit vs Conflict** - Performance comparison
5. **Tile Access** - Simulated tile loading pattern

## Expected Results

### Full Group (Banks 0,1,2,3)

```
8 reads (2 per bank), round-robin
Expected: 4 page empty, 4 page hits
Constraints: tRRD_L + potential tFAW
```

### Across Groups (Banks 0,4,8,12)

```
8 reads (2 per bank), round-robin
Expected: 4 page empty, 4 page hits
Constraints: tRRD_S only (faster)
```

### Page Hit Burst

```
64 reads (16 rounds × 4 banks)
Expected: 4 page empty, 60 page hits
Page hit ratio: >90%
```

## Running

```bash
# Build
cmake --build build

# Run full-group pattern
./build/patterns/memory/lpddr5/lpddr5_full_group

# Run across-groups pattern
./build/patterns/memory/lpddr5/lpddr5_across_groups

# Run page-hit-burst pattern
./build/patterns/memory/lpddr5/lpddr5_page_hit_burst

# With trace export
./build/patterns/memory/lpddr5/lpddr5_page_hit_burst --trace output.json
```

## Architectural Recommendations

### For KPU Tile Data Movement

| Scenario | Recommended Banks | Rationale |
|----------|------------------|-----------|
| Single tile load | Any bank | No parallelism needed |
| 4-tile parallel load | 0, 4, 8, 12 | Maximum activate speed |
| Double buffering | 0,4 (input) + 8,12 (output) | Separate groups avoid conflicts |
| Sustained streaming | Across groups, round-robin | Avoids tFAW limits |

### Page Hit Optimization

For best throughput:
1. **Pre-open pages** before compute starts
2. **Align tiles to rows** to maximize page hits
3. **Use different banks per tile** to enable parallelism
4. **Spread banks across groups** to minimize activate delays

### Bandwidth Calculation

With optimal page hits across 4 banks:
- tCL = 14 cycles, tBurst = 8 cycles
- Burst size = 64 bytes
- Theoretical: 64 bytes / 22 cycles = 2.9 bytes/cycle per bank
- 4 banks: ~11.6 bytes/cycle peak (with perfect scheduling)
