# DMA Pattern Test Suite Plan

## Overview

Create a DMA pattern test suite similar to the memory controller patterns, with traceable and visualizable patterns that reflect real-world workloads:
- STREAM benchmark vector accesses
- Block linear algebra tile sequences (GEMM)
- Tiled conv2d data movement

Key insight: Block linear algebra pulls sub-rows from larger matrices where the **pitch** (row stride in bytes) determines memory access patterns and page behavior.

## Architecture

```
                              DRAM
                               │
                         Memory Controller
                               │
                ┌──────────────┼──────────────┐
              DMA[W0]        DMA[W1]        DMA[W2]
                │              │              │
                ▼              ▼              ▼
    ┌─────────────────────────────────────────────────┐
    │               NoC Mesh (4x4)                    │
    │   [R0,0]──[R0,1]──[R0,2]──[R0,3]               │
    │     │       │       │       │                   │
    │   [R1,0]──[R1,1]──[R1,2]──[R1,3]  ← L3 Tiles   │
    │     │       │       │       │                   │
    │   [R2,0]──[R2,1]──[R2,2]──[R2,3]               │
    │     │       │       │       │                   │
    │   [R3,0]──[R3,1]──[R3,2]──[R3,3]               │
    └─────────────────────────────────────────────────┘

Patterns explore: Matrix pitch vs tile size vs page size interplay
```

## Directory Structure

```
patterns/dma/
├── common/
│   ├── dma_harness.hpp          # DMA test harness
│   ├── dma_configs.hpp          # Standard DMA configurations
│   ├── matrix_layouts.hpp       # Matrix addressing with pitch
│   └── trace_validator.py       # DMA-specific trace validation
├── stream/                      # STREAM benchmark patterns
│   ├── stream_copy.cpp          # Copy: A[i] = B[i]
│   ├── stream_scale.cpp         # Scale: A[i] = k*B[i]
│   ├── stream_add.cpp           # Add: A[i] = B[i] + C[i]
│   └── stream_triad.cpp         # Triad: A[i] = B[i] + k*C[i]
├── gemm/                        # Block linear algebra patterns
│   ├── tile_aligned.cpp         # Tile size = pitch (best case)
│   ├── tile_pitched_narrow.cpp  # Small tile from wide matrix
│   ├── tile_pitched_wide.cpp    # Wide tile from narrow matrix
│   ├── tile_page_boundary.cpp   # Tile crosses page boundaries
│   ├── a_tile_row_major.cpp     # A matrix tile (M×K)
│   └── b_tile_col_major.cpp     # B matrix tile (K×N, strided)
├── conv2d/                      # Convolution patterns
│   ├── input_tile_nhwc.cpp      # Input tile in NHWC format
│   ├── weight_tile.cpp          # Weight filter tile
│   └── output_tile.cpp          # Output tile writeback
├── INVARIANTS.md                # DMA timing constraints
└── README.md                    # Pattern documentation
```

## DMA Patterns to Implement

### 1. STREAM Benchmark Patterns (4 patterns)

| Pattern | Description | Memory Pattern |
|---------|-------------|----------------|
| `stream_copy` | Single source, single dest | Sequential reads, sequential writes |
| `stream_scale` | Load, multiply, store | Sequential R→W |
| `stream_add` | Two sources, one dest | Interleaved 2×R→W |
| `stream_triad` | Two loads, multiply-add, store | Interleaved R→R→W |

**Parameters:**
- Vector length: 1MB, 4MB, 16MB
- Element size: 4B (float), 8B (double)
- Burst size: 64B, 256B, 1KB

### 2. GEMM Tile Patterns (6 patterns)

Focus on **pitch vs tile size vs page size interplay**:

| Pattern | Matrix | Tile | Pitch | Page Behavior |
|---------|--------|------|-------|---------------|
| `tile_aligned` | 4096×4096 | 64×64 | 4096×4=16KB | All page hits |
| `tile_pitched_narrow` | 4096×4096 | 32×32 | 16KB | Page conflicts every row |
| `tile_pitched_wide` | 256×256 | 64×64 | 1KB | Page hits, limited |
| `tile_page_boundary` | 4096×4096 | 64×64 @ offset | Tile crosses pages |
| `a_tile_row_major` | Row-major A | M×K tile | M pitch | Row-sequential |
| `b_tile_col_major` | Col-major B | K×N tile | K pitch | Strided access |

**Key metrics:**
- Page hit rate vs tile dimensions
- Memory bandwidth utilization
- DMA channel stall cycles

### 3. Conv2D Patterns (3 patterns)

| Pattern | Tensor | Format | Access Pattern |
|---------|--------|--------|----------------|
| `input_tile_nhwc` | Input | NHWC | Spatial + channel sequential |
| `weight_tile` | Weights | OIHW | Filter × channel blocks |
| `output_tile` | Output | NHWC | Coalesced writes |

## DMA Harness Design

```cpp
// patterns/dma/common/dma_harness.hpp

class DMAHarness {
public:
    DMAHarness(const DMAConfig& dma_config,
               const MemoryControllerConfig& mc_config);

    // === Matrix Layout ===
    void set_matrix_layout(size_t rows, size_t cols,
                           size_t pitch_bytes, size_t element_size);

    // === Tile Operations ===
    uint64_t submit_tile_read(size_t tile_row, size_t tile_col,
                              size_t tile_height, size_t tile_width);
    uint64_t submit_tile_write(size_t tile_row, size_t tile_col,
                               size_t tile_height, size_t tile_width);

    // === STREAM Operations ===
    void submit_stream_copy(uint64_t src_base, uint64_t dst_base,
                            size_t count, size_t element_size);
    void submit_stream_triad(uint64_t a_base, uint64_t b_base,
                             uint64_t c_base, size_t count);

    // === Simulation ===
    bool run_until_complete(uint64_t max_cycles = 100000);
    void tick();

    // === Statistics ===
    struct Stats {
        uint64_t dma_transfers;
        uint64_t memory_reads, memory_writes;
        uint64_t page_hits, page_conflicts;
        uint64_t dma_stall_cycles;      // Waiting for memory
        uint64_t noc_stall_cycles;      // Waiting for NoC
        uint64_t total_bytes;
        double effective_bandwidth_gbps;
        double page_hit_rate;
    };
    const Stats& stats() const;
    void print_stats();

    // === Tracing ===
    void export_trace(const std::string& filename);
    void export_memory_trace(const std::string& filename);

private:
    std::unique_ptr<CycleAccurateDMAEngine> dma_;
    std::unique_ptr<IMemoryController> mc_;
    std::unique_ptr<INoC> noc_;

    // Matrix layout for tile addressing
    size_t matrix_rows_, matrix_cols_;
    size_t pitch_bytes_, element_size_;

    // Address computation
    uint64_t compute_element_address(size_t row, size_t col);
    uint64_t compute_row_address(size_t row, size_t col_start);
};
```

## Matrix Layout Helper

```cpp
// patterns/dma/common/matrix_layouts.hpp

struct MatrixLayout {
    size_t rows;
    size_t cols;
    size_t pitch_bytes;      // Stride between rows (may be > cols * elem_size)
    size_t element_size;
    uint64_t base_addr;

    // Derived
    size_t row_bytes() const { return cols * element_size; }
    bool is_contiguous() const { return pitch_bytes == row_bytes(); }
    size_t padding_bytes() const { return pitch_bytes - row_bytes(); }

    // Address computation
    uint64_t address(size_t row, size_t col) const {
        return base_addr + row * pitch_bytes + col * element_size;
    }

    // Tile extraction
    std::vector<DMATransfer> tile_transfers(
        size_t tile_row, size_t tile_col,
        size_t tile_height, size_t tile_width,
        uint64_t dst_base) const;

    // Factory methods
    static MatrixLayout row_major(size_t rows, size_t cols, size_t elem_size);
    static MatrixLayout pitched(size_t rows, size_t cols, size_t pitch, size_t elem_size);
    static MatrixLayout aligned_to_page(size_t rows, size_t cols, size_t elem_size,
                                        size_t page_size = 4096);
};
```

## Example Pattern: tile_pitched_narrow.cpp

```cpp
// patterns/dma/gemm/tile_pitched_narrow.cpp
//
// Small tile from wide matrix - demonstrates pitch/tile interaction
// Matrix: 4096x4096 (16KB pitch), Tile: 32x32

int main() {
    std::cout << "=== DMA Tile Pitched Narrow Pattern ===" << std::endl;
    std::cout << "32x32 tile from 4096x4096 matrix (16KB pitch)" << std::endl;
    std::cout << "Expected: Page conflict on each tile row" << std::endl;

    DMAConfig dma_cfg;
    dma_cfg.num_channels = 4;

    MemoryControllerConfig mc_cfg;
    mc_cfg.technology = MemoryTechnology::LPDDR5;

    DMAHarness harness(dma_cfg, mc_cfg);

    // 4096x4096 float matrix with 16KB pitch (row-major)
    MatrixLayout matrix = MatrixLayout::pitched(
        4096, 4096, 16384, sizeof(float));
    harness.set_matrix_layout(matrix);

    // Extract 32x32 tile starting at (128, 256)
    const size_t TILE_ROW = 128;
    const size_t TILE_COL = 256;
    const size_t TILE_SIZE = 32;

    std::cout << "\nTile position: (" << TILE_ROW << ", " << TILE_COL << ")" << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE << std::endl;
    std::cout << "Matrix pitch: " << matrix.pitch_bytes << " bytes" << std::endl;

    harness.submit_tile_read(TILE_ROW, TILE_COL, TILE_SIZE, TILE_SIZE);

    harness.run_until_complete();
    harness.print_stats();

    const auto& stats = harness.stats();

    // With 16KB pitch and 32 elements (128B) per row:
    // Each tile row accesses a new memory row → page conflict
    size_t expected_page_conflicts = TILE_SIZE - 1;  // First is page_empty

    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Expected page conflicts: " << expected_page_conflicts << std::endl;
    std::cout << "Actual page conflicts: " << stats.page_conflicts << std::endl;

    harness.export_trace("gemm/tile_pitched_narrow_trace.json");

    return (stats.page_conflicts >= expected_page_conflicts - 2) ? 0 : 1;
}
```

## Trace Format

DMA traces extend the memory trace format:

```json
{
  "name": "DMA_TRANSFER",
  "cat": "DMA_ENGINE",
  "ph": "X",
  "ts": 0.0,
  "dur": 1500.0,
  "pid": 4,
  "tid": 0,
  "args": {
    "txn_id": 1,
    "channel": 0,
    "state_sequence": ["IDLE", "WAITING_MEMORY_READ", "MEMORY_READ_COMPLETE",
                       "NOC_INJECTING", "COMPLETE"],
    "src_addr": "0x10000",
    "dst_addr": "0x0",
    "size": 128,
    "memory_latency_cycles": 45,
    "noc_latency_cycles": 12,
    "stall_cycles": 0
  }
}
```

## Files to Create

| File | Purpose |
|------|---------|
| `patterns/dma/common/dma_harness.hpp` | Test harness with DMA+MC+NoC |
| `patterns/dma/common/dma_configs.hpp` | Standard DMA configurations |
| `patterns/dma/common/matrix_layouts.hpp` | Matrix addressing helpers |
| `patterns/dma/stream/stream_copy.cpp` | STREAM Copy benchmark |
| `patterns/dma/stream/stream_triad.cpp` | STREAM Triad benchmark |
| `patterns/dma/gemm/tile_aligned.cpp` | Best-case aligned tiles |
| `patterns/dma/gemm/tile_pitched_narrow.cpp` | Narrow tile, wide matrix |
| `patterns/dma/gemm/tile_pitched_wide.cpp` | Wide tile, narrow matrix |
| `patterns/dma/gemm/tile_page_boundary.cpp` | Page boundary crossing |
| `patterns/dma/gemm/a_tile_row_major.cpp` | A matrix row-major tile |
| `patterns/dma/gemm/b_tile_col_major.cpp` | B matrix col-major strided |
| `patterns/dma/conv2d/input_tile_nhwc.cpp` | Conv2D input tile |
| `patterns/dma/INVARIANTS.md` | DMA timing constraints |
| `patterns/dma/README.md` | Pattern documentation |
| `patterns/CMakeLists.txt` | Update with DMA patterns |

## CMakeLists.txt Addition

```cmake
# ============================================================================
# DMA Patterns
# ============================================================================

# DMA-specific common headers
add_library(dma_pattern_common INTERFACE)
target_include_directories(dma_pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/dma/common
    ${PROJECT_SOURCE_DIR}/include
)
set_target_properties(dma_pattern_common PROPERTIES FOLDER "Patterns/DMA/Common")

# Helper function to add DMA pattern
function(add_dma_pattern NAME SOURCE_FILE FOLDER_LEVEL)
    add_executable(dma_${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/dma/${SOURCE_FILE}
    )
    target_link_libraries(dma_${NAME} PRIVATE
        pattern_common
        dma_pattern_common
        kpu_datamovement_components
        kpu_memory_components
        kpu_noc
        kpu_trace
    )
    set_target_properties(dma_${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/patterns/dma
        FOLDER "Patterns/DMA/${FOLDER_LEVEL}"
    )
endfunction()

# STREAM Patterns
add_dma_pattern(stream_copy   stream/stream_copy.cpp   "STREAM")
add_dma_pattern(stream_triad  stream/stream_triad.cpp  "STREAM")

# GEMM Patterns
add_dma_pattern(tile_aligned        gemm/tile_aligned.cpp        "GEMM")
add_dma_pattern(tile_pitched_narrow gemm/tile_pitched_narrow.cpp "GEMM")
add_dma_pattern(tile_pitched_wide   gemm/tile_pitched_wide.cpp   "GEMM")
add_dma_pattern(tile_page_boundary  gemm/tile_page_boundary.cpp  "GEMM")
add_dma_pattern(a_tile_row_major    gemm/a_tile_row_major.cpp    "GEMM")
add_dma_pattern(b_tile_col_major    gemm/b_tile_col_major.cpp    "GEMM")

# Conv2D Patterns
add_dma_pattern(input_tile_nhwc     conv2d/input_tile_nhwc.cpp   "Conv2D")
```

## Verification

1. **Build:** `cmake --build --preset release`
2. **Run patterns:**
   ```bash
   ./build/patterns/dma/dma_stream_copy
   ./build/patterns/dma/dma_tile_pitched_narrow
   ```
3. **Verify traces:** Check `traces/dma/` directory
4. **Visualize:** Load JSON in https://ui.perfetto.dev
5. **Analyze:**
   - Page hit rates match expectations
   - DMA stall cycles correlate with memory contention
   - NoC injection patterns visible

## Key Metrics to Capture

| Pattern Category | Primary Metrics |
|-----------------|-----------------|
| STREAM | Bandwidth (GB/s), memory utilization |
| GEMM tiles | Page hit rate, stall cycles, effective BW |
| Conv2D | Spatial locality benefit, channel grouping |

## Expected Insights

1. **Tile-aligned (best):** ~75% page hits, high bandwidth
2. **Pitched narrow (worst):** ~0% page hits, page conflict every row
3. **Page boundary:** Partial hits, depends on alignment
4. **B tile col-major:** Strided access, needs bank interleaving
