# Session Log: DMA Pattern Test Suite

**Date:** 2026-01-09
**Duration:** ~3 hours
**Focus:** Implement DMA pattern test suite with memory controller trace integration and swimlane visualization

## Summary

Created a comprehensive DMA pattern test suite following the established memory controller pattern infrastructure. Implemented 13 DMA patterns covering STREAM benchmarks, GEMM tile operations, and Conv2D data movement. Added memory controller trace integration with explicit DMA-to-MC command linkage. Created a swimlane visualization tool for analyzing DMA/MC interactions. Fixed timing issues where DMA transfers appeared to start at cycle 0.

## Context

The KPU simulator needed DMA pattern tests to validate data movement between device memory and on-chip storage. The patterns explore how matrix pitch, tile size, and page boundaries affect memory access efficiency - critical factors for ML workload performance.

## Architecture

```
                              DRAM
                               |
                         Memory Controller
                               |
                +--------------+--------------+
              DMA[W0]        DMA[W1]        DMA[W2]
                |              |              |
                v              v              v
    +---------------------------------------------+
    |               NoC Mesh (4x4)                |
    |   [R0,0]--[R0,1]--[R0,2]--[R0,3]           |
    |     |       |       |       |               |
    |   [R1,0]--[R1,1]--[R1,2]--[R1,3]  <- L3    |
    |     |       |       |       |               |
    |   [R2,0]--[R2,1]--[R2,2]--[R2,3]           |
    |     |       |       |       |               |
    |   [R3,0]--[R3,1]--[R3,2]--[R3,3]           |
    +---------------------------------------------+

Patterns explore: Matrix pitch vs tile size vs page size interplay
```

## Files Created

### DMA Harness Infrastructure

| File | Description |
|------|-------------|
| `patterns/dma/common/dma_harness.hpp` | Test harness integrating DMA + MC + NoC |
| `patterns/dma/common/dma_configs.hpp` | Standard DMA configuration presets |
| `patterns/dma/common/matrix_layouts.hpp` | Matrix addressing with pitch support |

### STREAM Patterns (2 patterns)

| File | Description |
|------|-------------|
| `patterns/dma/stream/stream_copy.cpp` | A[i] = B[i] - sequential copy benchmark |
| `patterns/dma/stream/stream_triad.cpp` | A[i] = B[i] + k*C[i] - classic triad |

### GEMM Tile Patterns (6 patterns)

| File | Description |
|------|-------------|
| `patterns/dma/gemm/tile_aligned.cpp` | Best case: tile size matches pitch |
| `patterns/dma/gemm/tile_pitched_narrow.cpp` | Small tile from wide matrix (worst case) |
| `patterns/dma/gemm/tile_pitched_wide.cpp` | Wide tile from narrow matrix |
| `patterns/dma/gemm/tile_page_boundary.cpp` | Tile crosses page boundaries |
| `patterns/dma/gemm/a_tile_row_major.cpp` | A matrix tile (row-major) |
| `patterns/dma/gemm/b_tile_col_major.cpp` | B matrix tile (column-major strided) |

### Conv2D Pattern (1 pattern)

| File | Description |
|------|-------------|
| `patterns/dma/conv2d/input_tile_nhwc.cpp` | Input tile in NHWC format |

### Visualization

| File | Description |
|------|-------------|
| `traces/dma/tools/swimlane.html` | Interactive swimlane visualization |

### Documentation

| File | Description |
|------|-------------|
| `patterns/dma/README.md` | Pattern documentation |
| `patterns/dma/INVARIANTS.md` | DMA timing constraints |

## Key Features Implemented

### 1. DMA-to-MC Trace Linkage

Added explicit `dma_transfer_id` field to memory controller trace entries, enabling:
- Click-to-highlight in visualization (select DMA transfer to see its MC commands)
- Accurate timing correlation between DMA and MC components
- Bank utilization tracking per DMA transfer

### 2. Accurate DMA Start Cycles

Fixed issue where all DMA transfers showed `submit_cycle=0`:

**Problem:** All transfers submitted before simulation starts, so `cycle_` is 0.

**Solution:** Compute actual start cycle from associated MC commands:
1. Build MC-to-transfer mapping based on completion timing
2. For each transfer, find earliest MC command associated with it
3. Use that command's `issue_cycle` as the actual start time

```cpp
// For each MC command, find transfer with smallest complete_cycle >= MC complete_cycle
for (size_t i = 0; i < mc_trace_entries.size(); ++i) {
    // Find transfer this command serves
    int64_t best_id = -1;
    uint64_t best_complete = UINT64_MAX;
    for (const auto& evt : transfer_events_) {
        if (evt.complete_cycle >= entry.cycle_complete) {
            if (evt.complete_cycle < best_complete) {
                best_complete = evt.complete_cycle;
                best_id = evt.transfer_id;
            }
        }
    }
    mc_to_transfer[i] = best_id;
}
```

### 3. Swimlane Visualization

Created `traces/dma/tools/swimlane.html` with:
- Left sidebar with statistics (DMA transfers, bandwidth, page hits)
- DMA channels showing transfer timing
- MC banks showing command timing
- Click-to-highlight DMA-MC associations
- Bank utilization display
- File loading capability
- Zoom and pan controls

## Trace Format

DMA traces use Chrome Trace Event format with explicit linkage:

```json
{
  "name": "DMA_READ",
  "cat": "dma",
  "ph": "X",
  "ts": 0.013,
  "dur": 0.012,
  "pid": 1,
  "tid": 1,
  "args": {
    "id": 0,
    "src": "0x200400",
    "dst": "0x0",
    "size": 256,
    "channel": 1,
    "submit_cycle": 42,
    "complete_cycle": 81
  }
},
{
  "name": "READ",
  "cat": "mc",
  "ph": "X",
  "ts": 0.018,
  "dur": 0.007,
  "pid": 2,
  "tid": 0,
  "args": {
    "dma_transfer_id": 0,
    "txn_id": 1,
    "issue_cycle": 56,
    "complete_cycle": 78,
    "desc": "READ Ch0 Bank0"
  }
}
```

## Pattern Results

### GEMM Tile Patterns

| Pattern | Page Hit Rate | Bandwidth | Key Insight |
|---------|---------------|-----------|-------------|
| tile_aligned | 75% | 30.34 GB/s | Best case - pitch matches tile |
| tile_pitched_wide | 91.7% | 30.34 GB/s | Narrow matrix = more page hits |
| tile_pitched_narrow | 75% | 15.17 GB/s | Wide matrix = lower efficiency |
| tile_page_boundary | 75% | 30.34 GB/s | Boundary crossing handled |

### STREAM Patterns

| Pattern | Bandwidth | Description |
|---------|-----------|-------------|
| stream_copy | 30.34 GB/s | Sequential R/W pairs |
| stream_triad | 8.89 GB/s | 2R + 1W interleaved (page conflicts) |

## Interface Changes

### Memory Controller Interface

Added trace methods to `IMemoryController`:
```cpp
virtual const std::vector<sw::trace::TraceEntry>& trace_entries() const;
virtual void clear_trace_entries();
```

## Commits

1. `02bccd9` - Add memory controller trace entries to DMA traces
2. `c6d1b0f` - Fix DMA transfer start cycle computation in traces

## Validation

All traces verified for swimlane visualization compatibility:
- GEMM traces: 12 transfers each, all linked to MC commands
- STREAM copy: 12 transfers (6 READ + 6 WRITE), READs linked
- STREAM triad: 24 transfers, all linked to 183 MC commands

## Known Limitations

1. **WRITE traces not generated:** Mock memory controller only traces READs
   - Affects stream_copy WRITE transfers (show at cycle 0)
   - Could be fixed by implementing WRITE trace generation

2. **Single memory controller:** Current harness uses one MC
   - Multi-channel patterns would need MC array

## Next Steps

1. Add WRITE trace generation to memory controller
2. Implement multi-channel DMA patterns
3. Add NoC congestion modeling
4. Create automated trace validation script
