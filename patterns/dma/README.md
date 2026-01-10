# DMA Pattern Test Suite

This directory contains characteristic DMA patterns for tracing and visualization, reflecting real-world memory access patterns in AI/ML workloads.

## Overview

The patterns demonstrate the interplay between:
- Matrix allocation and pitch (row stride)
- DRAM page size and page policy
- Tile size and access patterns
- DMA engine behavior and memory controller response

## Directory Structure

```
patterns/dma/
├── common/
│   ├── dma_harness.hpp          # Test harness integrating DMA + MC + NoC
│   ├── dma_configs.hpp          # Standard DMA and memory configurations
│   └── matrix_layouts.hpp       # Matrix addressing with pitch support
├── stream/                      # STREAM benchmark patterns
│   ├── stream_copy.cpp          # A[i] = B[i] - sequential copy
│   └── stream_triad.cpp         # A[i] = B[i] + k*C[i] - 2R+1W pattern
├── gemm/                        # Block linear algebra tile patterns
│   ├── tile_aligned.cpp         # Best case: aligned tiles
│   ├── tile_pitched_narrow.cpp  # Worst case: narrow tile, wide matrix
│   ├── tile_pitched_wide.cpp    # Good case: wide tile, narrow matrix
│   ├── tile_page_boundary.cpp   # Tile crossing page boundaries
│   ├── a_tile_row_major.cpp     # A matrix tile (row-sequential)
│   └── b_tile_col_major.cpp     # B matrix tile (strided access)
├── conv2d/                      # Convolution patterns
│   └── input_tile_nhwc.cpp      # Input tile in NHWC format
└── README.md                    # This file
```

## Running Patterns

Build and run individual patterns:

```bash
# Build all DMA patterns
cmake --build --preset release

# Run a specific pattern
./build/patterns/dma/dma_stream_copy
./build/patterns/dma/dma_tile_pitched_narrow

# Disable trace export for faster execution
./build/patterns/dma/dma_tile_aligned --no-trace
```

## Pattern Categories

### STREAM Benchmark Patterns

Classic memory bandwidth benchmarks:

| Pattern | Operation | Access Pattern |
|---------|-----------|----------------|
| `stream_copy` | A[i] = B[i] | Sequential R→W |
| `stream_triad` | A[i] = B[i] + k*C[i] | Interleaved 2R+1W |

### GEMM Tile Patterns

Block linear algebra patterns demonstrating pitch vs tile size interplay:

| Pattern | Matrix | Tile | Page Behavior |
|---------|--------|------|---------------|
| `tile_aligned` | 4096×4096 | 64×64 | Best case |
| `tile_pitched_narrow` | 4096×4096 (16KB pitch) | 32×32 | Worst case |
| `tile_pitched_wide` | 256×256 (1KB pitch) | 64×64 | Good locality |
| `tile_page_boundary` | 4096×4096 | 64×64 @ offset | Crosses boundaries |
| `a_tile_row_major` | 1024×512 | 64×64 | Row-sequential |
| `b_tile_col_major` | 512×1024 | 64×64 | Strided access |

### Conv2D Patterns

Convolution data movement patterns:

| Pattern | Tensor | Format | Access Pattern |
|---------|--------|--------|----------------|
| `input_tile_nhwc` | 56×56×256 | NHWC | Spatial + channel sequential |

## Key Insights

### Pitch vs Page Size

When `pitch_bytes >= page_size`, every consecutive row access causes a page conflict:

```
Row 0: Page 0
Row 1: Page 4 (different page → conflict)
Row 2: Page 8 (different page → conflict)
...
```

**Mitigation**: Use wider tiles, allocate with smaller pitch, or use bank-parallel access.

### A vs B Matrix Access in GEMM

- **A matrix (row-major)**: Good locality, tile rows are contiguous
- **B matrix (row-major, accessed by columns)**: Poor locality, strided access

**Mitigation**: Store B transposed, use tiled layouts, or prefetch.

### NHWC Conv2D Access

- Channels are contiguous (good for vectorization)
- Large pitch between spatial rows (56KB for 56×256)
- Each spatial row may span multiple DRAM pages

## Trace Visualization

Traces are exported to `traces/dma/` in Chrome Trace format:

```bash
# Run with trace export (default)
./build/patterns/dma/dma_tile_pitched_narrow

# View in browser
open https://ui.perfetto.dev
# Drag and drop: traces/dma/gemm/tile_pitched_narrow_trace.json
```

## Metrics Collected

- **DMA Engine**: Transfers completed, bytes transferred, stall cycles
- **Memory Controller**: Reads, writes, page hits, page empty, page conflicts
- **Derived**: Page hit rate, effective bandwidth (GB/s)
