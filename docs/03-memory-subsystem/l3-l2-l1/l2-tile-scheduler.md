# L2 Tile Scheduler for KPU Matrix Multiplication

## Overview

The L2 Tile Scheduler is a critical component of the KPU compiler infrastructure that manages tile allocations and scheduling in the L2 cache banks to minimize DRAM accesses during matrix multiplication. This scheduler operates on the tile configuration produced by the TileOptimizer and generates an optimal sequence of tile loads/reloads to complete a block matrix multiplication.

## Memory Hierarchy

The KPU architecture consists of a multi-level memory hierarchy:

```
DRAM (External) → L3 Tiles (128KB × 4) → L2 Banks (64KB × 8) → L1 ASM → Systolic Array
```

### Key Components

1. **DRAM/External Memory**: Off-chip memory with high latency
2. **L3 Tiles**: Distributed on-chip cache (128KB per tile, 4 tiles total)
3. **L2 Banks**: Intermediate cache banks (64KB per bank, 8 banks total = 512KB)
4. **L1 ASM (Auto-Sequencing Memory)**: Register sets with head/tail pointers
5. **Systolic Array**: Compute fabric (default 16×16 PEs)

## Design Goals

The L2 Tile Scheduler aims to:

1. **Minimize L2 reloads from L3**: Keep frequently used tiles resident in L2
2. **Maximize data reuse**: Exploit the reuse patterns in output-stationary execution
3. **Track capacity constraints**: Manage limited L2 space (default 128 tile slots)
4. **Generate optimal load sequences**: Order tile loads to avoid unnecessary evictions
5. **Provide visibility**: Show which tiles are allocated where and when

## Key Data Structures

### TileID

Identifies a specific matrix tile:

```cpp
struct TileID {
    char matrix;        // 'A', 'B', or 'C'
    Size row_idx;       // Row index in tile grid
    Size col_idx;       // Column index in tile grid
};
```

### L2Slot

Represents an allocation slot in L2:

```cpp
struct L2Slot {
    size_t bank_id;         // Which L2 bank (0-7)
    Address offset;          // Offset within bank
    Size size_bytes;         // Size of allocation
    std::optional<TileID> tile;  // Which tile is allocated
};
```

### TileLoad

Records a tile load/reload event:

```cpp
struct TileLoad {
    enum class Type { INITIAL_LOAD, RELOAD, PREFETCH };

    Type type;
    TileID tile_id;
    size_t slot_index;      // Which L2 slot
    Size time_step;         // When this load occurs
    bool from_dram;         // Load from DRAM vs L3

    // Context: which compute operation needs this tile
    Size compute_ti, compute_tj, compute_tk;
};
```

### L2Schedule

Complete schedule with all tile operations:

```cpp
struct L2Schedule {
    // Matrix and tile configuration
    Size M, N, K;
    TileConfig config;
    Size num_tile_rows_A, num_tile_cols_A;  // Tile grid dimensions
    Size num_tile_rows_B, num_tile_cols_B;
    Size num_tile_rows_C, num_tile_cols_C;

    // L2 configuration
    size_t num_l2_banks;
    Size l2_bank_size;
    Size l2_total_capacity;
    size_t max_l2_slots;

    // Allocation state
    std::vector<L2Slot> slots;

    // Load sequence
    std::vector<TileLoad> load_sequence;

    // Statistics
    Size total_loads;
    Size initial_loads;
    Size reloads;
    Size l3_hits, l3_misses;
    double l2_hit_rate, l3_hit_rate;
    Size total_bytes_loaded;

    // Reuse tracking
    std::map<TileID, Size> tile_access_count;
    std::map<TileID, Size> tile_load_count;
};
```

## Algorithm

### Output-Stationary Execution

The scheduler implements output-stationary dataflow where:

1. **C tiles** accumulate in the systolic array PEs
2. **A tiles** are streamed from L2 → L1 → Systolic Array (column-wise)
3. **B tiles** are streamed from L2 → L1 → Systolic Array (row-wise)

### Compute Order

For a matrix multiplication C[M,N] = A[M,K] × B[K,N] with tiles (Ti, Tj, Tk):

```python
for ti in range(M / Ti):      # Iterate over C rows
    for tj in range(N / Tj):  # Iterate over C columns
        for tk in range(K / Tk):  # Accumulate across K dimension
            C[ti,tj] += A[ti,tk] × B[tk,tj]
```

This order maximizes reuse:
- **A tiles** are reused across the N dimension (reuse factor = N/Tj)
- **B tiles** are reused across the M dimension (reuse factor = M/Ti)
- **C tiles** are reused across the K dimension (accumulation factor = K/Tk)

### Tile Allocation and Eviction

1. **Initial Allocation**: Load as many A and B tiles as fit in L2
2. **On-Demand Loading**: Load tiles as needed for each compute operation
3. **LRU Eviction**: When L2 is full, evict the least recently used tile
4. **L3 Tracking**: Track which tiles are likely in L3 to estimate L3 hits

## Cache Replacement Policies

The scheduler supports multiple replacement policies:

- **LRU (Least Recently Used)**: Evict the tile that was accessed longest ago
- **FIFO (First In First Out)**: Evict the oldest loaded tile
- **OPTIMAL (Belady)**: Evict the tile that will be used furthest in the future (requires future knowledge)
- **MANUAL**: User-specified eviction order

## Performance Metrics

The scheduler tracks and reports:

### Load Statistics
- **Total Loads**: Number of tile loads from L3→L2
- **Initial Loads**: First-time loads (cold misses)
- **Reloads**: Tiles loaded again after eviction (capacity misses)
- **L3 Hits**: Loads satisfied from L3 cache
- **L3 Misses**: Loads requiring DRAM access

### Hit Rates
- **L2 Hit Rate**: Percentage of tile accesses satisfied by L2
  - Formula: `(total_accesses - total_loads) / total_accesses`
- **L3 Hit Rate**: Percentage of L2 loads satisfied by L3
  - Formula: `l3_hits / total_loads`

### Data Movement
- **Total Bytes Loaded**: Data transferred from L3→L2
- **Bandwidth**: Effective bandwidth utilization

### Reuse Statistics
- **Tile Access Count**: How many times each tile is accessed
- **Tile Load Count**: How many times each tile is loaded
- **Reuse Factor**: Accesses per load (higher is better)

## Example Results

### Small Matrix (256×256×256)
```
Tile Grid: A: 4×3 = 12 tiles, B: 3×4 = 12 tiles, C: 4×4 = 16 tiles
L2 Capacity: 128 tiles
Results:
  - L2 Hit Rate: 100.00% (all tiles fit in L2)
  - Total Loads: 0 (after initial allocation)
  - Average Reuse: 3.60
```

### Medium Matrix (512×512×512)
```
Tile Grid: A: 8×6 = 48 tiles, B: 6×8 = 48 tiles, C: 8×8 = 64 tiles
L2 Capacity: 128 tiles
Results:
  - L2 Hit Rate: 100.00% (all tiles fit in L2)
  - Total Loads: 0 (after initial allocation)
  - Average Reuse: 7.20
```

### Large Matrix (1024×1024×1024)
```
Tile Grid: A: 16×11 = 176 tiles, B: 11×16 = 176 tiles, C: 16×16 = 256 tiles
L2 Capacity: 128 tiles
Results:
  - L2 Hit Rate: 64.61% (capacity constraints)
  - Total Loads: 2,990
  - Initial Loads: 350
  - Reloads: 2,640
  - L3 Hit Rate: 88.29%
  - Data Movement: 67.95 MB
  - Average Reuse: 8.98
```

### Rectangular Matrix (2048×128×512)
```
Tile Grid: A: 26×7 = 182 tiles, B: 7×8 = 56 tiles, C: 26×8 = 208 tiles
L2 Capacity: 128 tiles
Results:
  - L2 Hit Rate: 94.57% (better fit due to shape)
  - Total Loads: 237
  - Reloads: 1
  - Data Movement: 4.21 MB
```

## API Usage

### Basic Usage

```cpp
#include <sw/compiler/l2_tile_scheduler.hpp>
#include <sw/compiler/tile_optimizer.hpp>

using namespace sw::kpu::compiler;

// 1. Create tile optimizer
TileOptimizer optimizer;
auto tile_config = optimizer.optimize(M, N, K);

// 2. Create L2 tile scheduler
L2TileScheduler scheduler;

// 3. Generate L2 schedule
auto schedule = scheduler.generate_schedule(
    M, N, K,
    tile_config,
    L2TileScheduler::ReplacementPolicy::LRU,
    L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY
);

// 4. Analyze results
scheduler.print_schedule(schedule, true);
```

### Advanced Configuration

```cpp
// Custom memory hierarchy
TileOptimizer::MemoryHierarchy mem;
mem.L2_size = 128 * 1024;  // 128 KB per bank
mem.L2_bank_count = 8;      // 8 banks
mem.systolic_rows = 16;
mem.systolic_cols = 16;

L2TileScheduler scheduler(mem, 16);

// Try different replacement policies
auto lru_schedule = scheduler.generate_schedule(
    M, N, K, tile_config,
    L2TileScheduler::ReplacementPolicy::LRU
);

auto optimal_schedule = scheduler.generate_schedule(
    M, N, K, tile_config,
    L2TileScheduler::ReplacementPolicy::OPTIMAL
);
```

### Visualization

```cpp
// Print detailed schedule
scheduler.print_schedule(schedule, true);

// Print just L2 state
scheduler.print_l2_state(schedule);

// Print load sequence
scheduler.print_load_sequence(schedule, 50);

// Print reuse statistics
scheduler.print_reuse_stats(schedule);

// Export to JSON
std::string json = scheduler.export_json(schedule);
```

## Integration with KPU Architecture

### L1 Auto-Sequencing Memory (ASM)

The L1 ASM units sit between L2 and the systolic array:

- **North/South ASMs**: Feed columns from B tiles (vertical injection)
- **West/East ASMs**: Feed rows from A tiles (horizontal injection)
- **Auto-sequencing**: Head/tail pointers automatically advance each cycle
- **Size**: Matches systolic array dimension (e.g., 16 elements for 16×16 array)

### BlockMover

Transfers 2D blocks between L3 and L2:

- **Bandwidth**: ~32 GB/s
- **Operations**: Identity, transpose, reshape, shuffle
- **Dimensions**: North-East-South-West connectivity

### Streamer

Streams data from L2 to L1:

- **Bandwidth**: ~64 GB/s
- **Patterns**: Row-wise and column-wise streaming
- **Cache-aware**: Optimized for cache line transfers

## File Organization

```
include/sw/compiler/
  └── l2_tile_scheduler.hpp          # Header with all declarations

src/compiler/
  └── l2_tile_scheduler.cpp          # Implementation
  └── CMakeLists.txt                 # Build configuration (updated)

examples/compiler/
  └── l2_tile_scheduler_demo.cpp     # Demonstration program
  └── CMakeLists.txt                 # Build configuration (updated)

docs/
  └── l2-tile-scheduler.md           # This documentation
```

## Building and Running

### Build

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
ninja example_l2_tile_scheduler_demo
```

### Run

```bash
./examples/compiler/l2_tile_scheduler_demo
```

The demo runs 6 different scenarios showing:
1. Small matrix (256×256×256)
2. Medium matrix (512×512×512)
3. Large matrix (1024×1024×1024) with detailed output
4. Rectangular matrix (2048×128×512)
5. L2 capacity comparison
6. Impact of systolic array size

## Future Enhancements

### Short-term
1. **Optimal (Belady) replacement**: Implement true lookahead
2. **Prefetching**: Speculative tile loads
3. **L3 capacity modeling**: More accurate L3 hit/miss prediction
4. **Timeline visualization**: ASCII art showing L2 state over time
5. **Double-buffering support**: Overlap compute and data movement

### Medium-term
1. **Multi-level scheduling**: Coordinate L3 and L2 together
2. **Alternative dataflow**: Input-stationary and weight-stationary
3. **Tile fusion**: Combine small tiles to reduce overhead
4. **NUMA awareness**: Bank-level placement optimization
5. **Performance prediction**: Cycle-accurate cost model

### Long-term
1. **ML-based scheduling**: Learn optimal policies from traces
2. **Dynamic adaptation**: Runtime policy switching
3. **Multi-tenancy**: Share L2 across multiple operations
4. **Hardware co-design**: Inform L2 cache sizing decisions

## References

### Related KPU Components
- `TileOptimizer`: Determines optimal tile sizes (Ti, Tj, Tk)
- `ScheduleGenerator`: Converts tiles into hardware commands
- `L1Buffer`, `L2Bank`, `L3Tile`: Memory hierarchy components
- `BlockMover`, `Streamer`: Data movement engines
- `SystolicArray`: Compute fabric

### Academic Papers
- Goto & van de Geijn (2008): "Anatomy of High-Performance Matrix Multiplication"
- Pouchet et al. (2012): "Polyhedral-Based Data Reuse Optimization for Configurable Computing"
- Chen et al. (2016): "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs"
- Jouppi et al. (2017): "In-Datacenter Performance Analysis of a Tensor Processing Unit"

## Conclusion

The L2 Tile Scheduler is a critical component that bridges the gap between high-level tile optimization and low-level hardware execution. By carefully managing L2 cache allocations and generating optimal load sequences, it minimizes DRAM traffic and maximizes the reuse of data in the KPU's memory hierarchy.

The scheduler provides detailed visibility into:
- Which tiles are allocated in which L2 banks
- When tiles need to be loaded or reloaded
- What the hit rates and reuse factors are
- How much data movement is required

This visibility is essential for understanding performance, debugging scheduling issues, and optimizing the overall system.
