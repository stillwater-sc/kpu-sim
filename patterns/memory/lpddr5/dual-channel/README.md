# Level 5: Dual Channel Patterns

Tests for memory controller behavior with dual-channel configurations,
demonstrating 2x theoretical bandwidth potential.

## Dual Channel Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Controller                         │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │     Channel 0        │  │     Channel 1        │        │
│  │  ┌────┬────┬────┐   │  │  ┌────┬────┬────┐   │        │
│  │  │BG0 │BG1 │BG2 │BG3│  │  │BG0 │BG1 │BG2 │BG3│        │
│  │  └────┴────┴────┘   │  │  └────┴────┴────┘   │        │
│  │  16 banks total     │  │  16 banks total     │        │
│  └──────────────────────┘  └──────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Bandwidth Potential

| Configuration | Peak Bandwidth | Use Case |
|--------------|----------------|----------|
| Single Channel | 12.8 GB/s | Simple access |
| Dual Channel | 25.6 GB/s | Parallel access |

## Patterns

### independent/

Tests channels working independently without interference.

**Test Cases:**
1. **Channel 0 Only** - Baseline single-channel performance
2. **Channel 1 Only** - Verify second channel works
3. **Alternating** - Switch between channels per request
4. **Parallel Load** - Simultaneous access to both channels
5. **Sustained** - Long-running dual-channel workload
6. **Bandwidth Comparison** - Single vs dual channel timing
7. **R/W Split** - Read from CH0, write to CH1

**Address Generation:**
```cpp
// Independent channel addressing
make_address_dual(channel, bank, row, col);

// Channel 0: banks 0-15
make_address_dual(0, bank, row, col);

// Channel 1: banks 0-15 (separate space)
make_address_dual(1, bank, row, col);
```

### interleaved/

Tests address interleaving across channels for maximum bandwidth.

**Interleaving Strategy:**
- Cache line granularity: addr[6] selects channel
- Sequential addresses alternate between channels
- Maximizes channel utilization for streaming

**Test Cases:**
1. **Cache Line Interleaving** - Line 0→CH0, Line 1→CH1, ...
2. **Row Interleaving** - Even rows→CH0, Odd rows→CH1
3. **Streaming Read** - 16KB read with interleaving
4. **Streaming Write** - 16KB write with interleaving
5. **Interleaved Copy** - Read/write both interleaved
6. **Interleaving Benefit** - Compare vs non-interleaved
7. **Tile Load** - 4KB tile load optimized

**Optimal Access Pattern:**
```
Address 0x0000 → Channel 0, Bank 0
Address 0x0040 → Channel 1, Bank 0
Address 0x0080 → Channel 0, Bank 0
Address 0x00C0 → Channel 1, Bank 0
...
```

## Expected Results

### Independent Access

```
Single channel baseline: 8 reads, ~230 cycles
Dual channel parallel:   8 reads, ~230 cycles (same total, but 2x throughput possible)
```

### Interleaved Access

```
Non-interleaved (single channel): 16 reads, ~450 cycles
Interleaved (dual channel):       16 reads, ~300 cycles (30%+ improvement)
```

### Streaming Throughput

```
64 cache lines (4KB):
- Effective throughput: ~2.5-3.0 bytes/cycle
- Peak theoretical: 4.0 bytes/cycle (dual channel)
```

## Running

```bash
# Build
cmake --build build

# Run independent pattern
./build/patterns/memory/lpddr5/lpddr5_independent

# Run interleaved pattern
./build/patterns/memory/lpddr5/lpddr5_interleaved

# With trace export
./build/patterns/memory/lpddr5/lpddr5_interleaved --trace output.json
```

## Architectural Recommendations

### For KPU Data Movement

| Scenario | Recommended Strategy |
|----------|---------------------|
| Single tile load | Either channel |
| Dual tile load | One tile per channel |
| Streaming input | Interleaved addressing |
| Double buffering | Input CH0, Output CH1 |
| Matrix multiply | Interleave A, B, C matrices |

### Address Mapping for Interleaving

For optimal interleaving at cache line granularity:

```
Address bits:  [row | bank | col | channel | byte_offset]
                                     ^
                                     |
                              Bit 6 selects channel
```

This ensures:
- Sequential cache line accesses alternate channels
- Maximum parallelism for streaming patterns
- No channel conflicts for contiguous accesses

### Bandwidth Utilization

To maximize dual-channel bandwidth:

1. **Balance traffic** - Distribute requests evenly
2. **Avoid channel conflicts** - Don't hammer one channel
3. **Use interleaving** - For streaming workloads
4. **Separate read/write** - One channel per direction for copy

### Copy Operation Optimization

For `memcpy`-like operations:
```
Option 1: Same-channel copy
  Read CH0 → Write CH0
  Limited by single channel bandwidth

Option 2: Cross-channel copy
  Read CH0 → Write CH1
  Better: full read + write bandwidth

Option 3: Interleaved copy
  Read interleaved → Write interleaved
  Best: maximum parallelism on both ops
```
