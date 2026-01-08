# Memory Controller Latency and Bandwidth Characterization

This document characterizes the latency and bandwidth behavior of the LPDDR5 and GDDR6 memory controllers in the KPU simulator.

## Summary

| Metric | LPDDR5-6400 | GDDR6-16000 |
|--------|-------------|-------------|
| Clock Frequency | 3.2 GHz | 2.0 GHz |
| Banks per Channel | 8 (2 groups × 4) | 16 (4 groups × 4) |
| Channels | 1-2 | 2 |
| Page Size | 8 KB (128 cache lines) | 8 KB (128 cache lines) |
| Burst Length | BL16/BL32 | BL16 |
| Peak Bandwidth | 25.6 GB/s (dual) | 64 GB/s |

---

## LPDDR5-6400 Characterization

### Timing Parameters

| Parameter | Value (cycles) | Description |
|-----------|---------------|-------------|
| tRCD | 14 | Row to Column Delay |
| tCL | 16 | CAS Latency |
| tRP | 14 | Row Precharge |
| tRAS | 28 | Row Active Time |
| tRC | 42 | Row Cycle Time |
| tBurst | 8 | Burst Duration (BL16) |
| tRRD_L | 4 | ACT-to-ACT (same group) |
| tRRD_S | 4 | ACT-to-ACT (different group) |
| tCCD_L | 4 | CAS-to-CAS (same group) |
| tCCD_S | 2 | CAS-to-CAS (different group) |

### Latency Characterization

| Access Pattern | Avg Latency (cycles) | Description |
|---------------|---------------------|-------------|
| Page Hit | ~243 | Row already open, CAS only |
| Page Empty | ~305 | Bank idle, ACT + CAS |
| Page Conflict | ~305 | Wrong row open, PRE + ACT + CAS |

**Latency Breakdown:**
- **Page Hit**: tCL + tBurst = 16 + 8 = 24 cycles (minimum)
- **Page Empty**: tRCD + tCL + tBurst = 14 + 16 + 8 = 38 cycles (minimum)
- **Page Conflict**: tRP + tRCD + tCL + tBurst = 14 + 14 + 16 + 8 = 52 cycles (minimum)

*Note: Measured latencies include queue delays and scheduling overhead.*

### Bandwidth Characterization

#### Single Bank Performance

| Pattern | Throughput (bytes/cycle) | Page Hit Rate |
|---------|-------------------------|---------------|
| Sequential Page Hits | 5.59 | 93.8% |
| Page Conflicts | 0.96 | 0% |

#### Multi-Bank Scaling

| Banks | Throughput (bytes/cycle) | Speedup |
|-------|-------------------------|---------|
| 1 | 2.20 | 1.0x |
| 2 | 2.50 | 1.1x |
| 4 | 2.69 | 1.2x |
| 8 | 2.80 | 1.3x |

#### STREAM Benchmark Performance

| Kernel | Operations | Total Bytes | Throughput (bytes/cycle) |
|--------|-----------|-------------|-------------------------|
| Copy | 1R + 1W | 32 KB | 19.04 |
| Scale | 1R + 1W | 32 KB | 19.04 |
| Add | 2R + 1W | 48 KB | 26.67 |
| Triad | 2R + 1W | 48 KB | 26.67 |

#### Multi-DMA Tile Loading

| DMA Engines | Total Bytes | Throughput (bytes/cycle) | Speedup |
|-------------|-------------|-------------------------|---------|
| 4 | 64 KB | 44.73 | 1.0x |
| 8 | 128 KB | 89.47 | 2.0x |
| 16 | 128 KB | 89.47 | 2.0x |

#### Page Utilization Impact

| Accesses/Page | Page Hit Rate | Throughput (bytes/cycle) |
|---------------|---------------|-------------------------|
| 8 | 87.5% | 2.80 |
| 32 | ~97% | ~4.0 |
| 128 (full page) | 99.2% | 5.59 |

**Key Insight**: Maximizing page hits (128 cache lines per 8KB page) yields 2.5x bandwidth improvement over minimal page utilization.

---

## GDDR6-16000 Characterization

### Timing Parameters

| Parameter | Value (cycles) | Description |
|-----------|---------------|-------------|
| tRCDRD | 18 | Row to Column Delay (Read) |
| tRCDWR | 18 | Row to Column Delay (Write) |
| tRL | 20 | CAS Read Latency |
| tWL | 8 | CAS Write Latency |
| tRP | 18 | Row Precharge |
| tRAS | 28 | Row Active Time |
| tRC | 46 | Row Cycle Time |
| tBurst | 4 | Burst Duration (BL16) |
| tRRD_L | 4 | ACT-to-ACT (same group) |
| tRRD_S | 4 | ACT-to-ACT (different group) |
| tCCD_L | 3 | CAS-to-CAS (same group) |
| tCCD_S | 2 | CAS-to-CAS (different group) |
| tFAW | 16 | Four Activate Window |

### Latency Characterization

| Access Pattern | Avg Latency (cycles) | Description |
|---------------|---------------------|-------------|
| Page Hit | ~172-184 | Row already open, CAS only |
| Page Empty | ~88 | Bank idle, ACT + CAS |
| Page Conflict | ~298-328 | Wrong row open, PRE + ACT + CAS |

**Latency Breakdown:**
- **Page Hit Read**: tRL + tBurst = 20 + 4 = 24 cycles (minimum)
- **Page Empty Read**: tRCDRD + tRL + tBurst = 18 + 20 + 4 = 42 cycles (minimum)
- **Page Conflict Read**: tRP + tRCDRD + tRL + tBurst = 18 + 18 + 20 + 4 = 60 cycles (minimum)

### Bandwidth Characterization

#### Single Bank Performance

| Pattern | Throughput (bytes/cycle) | Page Hit Rate |
|---------|-------------------------|---------------|
| Sequential Page Hits | 5.12 | 93.8% |
| Page Conflicts | ~1.0 | 0% |

#### Multi-Bank Scaling (16 Banks)

| Banks | Throughput (bytes/cycle) | Speedup |
|-------|-------------------------|---------|
| 1 | 1.59 | 1.0x |
| 2 | 1.99 | 1.3x |
| 4 | 2.28 | 1.4x |
| 8 | 2.46 | 1.5x |
| 16 | 2.56 | 1.6x |

#### STREAM Benchmark Performance

| Kernel | Operations | Total Bytes | Throughput (bytes/cycle) |
|--------|-----------|-------------|-------------------------|
| Copy | 1R + 1W | 64 KB | 38.24 |
| Add | 2R + 1W | 96 KB | 59.08 |
| Triad | 2R + 1W | 96 KB | 59.08 |

#### Multi-DMA Tile Loading (16 Banks)

| DMA Engines | Total Bytes | Throughput (bytes/cycle) | Speedup |
|-------------|-------------|-------------------------|---------|
| 4 | 64 KB | 40.93 | 1.0x |
| 8 | 128 KB | 81.87 | 2.0x |
| 16 | 128 KB | 81.87 | 2.0x |
| 32 | 128 KB | 81.87 | 2.0x |

#### Access Pattern Comparison

| Pattern | Throughput (bytes/cycle) | Notes |
|---------|-------------------------|-------|
| GPU-style (64 threads) | 2.56 | Round-robin across 16 banks |
| Accelerator-style (tiles) | 1.72 | Sequential tile loads |
| Sustained Max | 10.23 | 256 accesses, optimal scheduling |

**Key Insight**: GDDR6's 16 banks provide 1.5-2x advantage over 8-bank configurations when fully utilized.

---

## Comparative Analysis

### LPDDR5 vs GDDR6

| Metric | LPDDR5-6400 | GDDR6-16000 | Winner |
|--------|-------------|-------------|--------|
| Banks | 8 | 16 | GDDR6 |
| Bank Groups | 2 | 4 | GDDR6 |
| Page Hit Latency | 24 cycles | 24 cycles | Tie |
| Page Empty Latency | 38 cycles | 42 cycles | LPDDR5 |
| Page Conflict Latency | 52 cycles | 60 cycles | LPDDR5 |
| STREAM Copy | 19.04 B/cyc | 38.24 B/cyc | GDDR6 |
| STREAM Triad | 26.67 B/cyc | 59.08 B/cyc | GDDR6 |
| Multi-DMA (8 engines) | 89.47 B/cyc | 81.87 B/cyc | LPDDR5 |

### Recommendations

1. **For Bandwidth-Bound Workloads**: Use GDDR6 with all 16 banks active
2. **For Latency-Sensitive Workloads**: Use LPDDR5 with page-hit optimized access
3. **For Tile-Based Accelerators**: Maximize page utilization (128 lines/page)
4. **For Multi-DMA Systems**: Scale DMA engines to match bank count

---

## Pattern Categories

### Level 1: Single Bank
- `page_hits` - Sequential access to same row
- `page_conflicts` - Alternating rows (worst case)
- `mixed_rw` - Interleaved reads and writes

### Level 2: Two Banks
- `same_group` - Banks in same bank group
- `diff_groups` - Banks in different bank groups

### Level 3: Three Banks
- `same_group` - All banks in one group
- `mixed_groups` - Banks across multiple groups

### Level 4: Four Banks
- `full_group` - Complete bank group
- `across_groups` - One bank per group
- `page_hit_burst` - Sustained page hits

### Level 5: Dual Channel (GDDR6)
- `independent` - Separate channel access
- `interleaved` - Alternating channel access

### Level 6: Complex Patterns
- `stream` - STREAM benchmark (Copy, Scale, Add, Triad)
- `tile_load` - ML accelerator tile loading
- `multi_dma` - Concurrent DMA engine simulation
- `random` - Random access pattern
- `strided` - Regular strided access

### Level 7: Bandwidth Patterns
- `page_burst` - Maximum page hits (128/page)
- `max_bandwidth` - Peak achievable bandwidth

---

## Trace Files

All patterns generate Chrome Trace Format JSON files viewable in:
- Perfetto UI: https://ui.perfetto.dev
- Chrome: chrome://tracing
- Custom viewers in `traces/memory/{lpddr5,gddr6}/tools/`

See `traces/README.md` for detailed visualization instructions.
