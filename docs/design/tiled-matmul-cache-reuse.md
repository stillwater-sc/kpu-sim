# Tiled Matrix Multiplication with L3 Cache Reuse

## Overview

This document describes the design for a parameterized tiled matrix multiplication system that properly models L3 cache behavior, tile reuse optimization, and provides comprehensive visualization of the block matrix decomposition and cache dynamics.

## Problem Statement

For the matrix multiplication: **D[M×N] = C[M×N] + A[M×K] × B[K×N]**

Example configuration:

- D[1000×1000] = C[1000×1000] + A[1000×100] × B[100×1000]
- Tile size: 16×16 (matched to systolic array)
- Tile counts: M_tiles=63, K_tiles=7, N_tiles=63

### Reuse Opportunity Analysis

Each tile can be reused multiple times:

- **A[i,k] tile**: Used for all j ∈ [0, N_tiles) → 63× reuse potential
- **B[k,j] tile**: Used for all i ∈ [0, M_tiles) → 63× reuse potential
- **C/D[i,j] tile**: Accumulated over k ∈ [0, K_tiles) → 7× access (accumulation)

**Optimal case**: Load each A tile once, reuse 63×. Load each B tile once, reuse 63×.

**Naive case**: Refetch every tile for every use → no reuse, maximum bandwidth waste.

## Configuration Parameters

### Programmable L3 Capacity

```cpp
struct L3Config {
    uint32_t capacity_tiles = 16;     // Total tiles that fit in L3
    uint32_t a_buffer_tiles = 4;      // Tiles reserved for A
    uint32_t b_buffer_tiles = 8;      // Tiles reserved for B
    uint32_t c_buffer_tiles = 4;      // Tiles reserved for C/D
    ReplacementPolicy policy = LRU;   // LRU, FIFO, or OPTIMAL
};
```

The capacity can be swept from 4 tiles (minimum) to 128+ tiles to study the impact on reuse efficiency.

### Programmable Loop Order

```cpp
enum class LoopOrder {
    IJK,      // for i: for j: for k: → A-row stays in L3
    JIK,      // for j: for i: for k: → B-col stays in L3
    IKJ,      // for i: for k: for j: → A-row reuse, partial sum accumulation
    KIJ,      // for k: for i: for j: → B-row reuse
    BLOCKED   // 2-level blocking for L3 + L2 reuse
};
```

#### Loop Order Impact

| Order | A Reuse | B Reuse | C Access Pattern | Best For |
|-------|---------|---------|------------------|----------|
| IJK | Row reuse (K×) | No reuse | Sequential write | Tall-skinny A |
| JIK | No reuse | Col reuse (K×) | Column access | Wide B |
| IKJ | Row reuse (N×) | Full column | Partial sums | Square matrices |
| KIJ | No reuse | Row reuse (M×) | Accumulate | Large K |
| BLOCKED | Block reuse | Block reuse | Block access | Large matrices |

## L3 Cache Model

```cpp
class L3CacheModel {
public:
    struct TileKey {
        OperandType operand;  // A, B, C, or D
        uint32_t tile_i;
        uint32_t tile_j;

        bool operator==(const TileKey& other) const;
        size_t hash() const;
    };

    struct TileEntry {
        TileKey key;
        uint64_t last_access_cycle;
        uint32_t access_count;
        bool dirty;
    };

    enum class AccessResult {
        HIT,        // Tile already in L3
        MISS,       // First load of this tile
        REFETCH     // Tile was evicted and reloaded
    };

    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;       // First-time loads
        uint64_t refetches = 0;    // Re-loads after eviction
        uint64_t evictions = 0;

        // Per-operand breakdown
        std::array<uint64_t, 4> hits_by_operand;
        std::array<uint64_t, 4> misses_by_operand;
        std::array<uint64_t, 4> refetches_by_operand;

        double hit_rate() const {
            return double(hits) / (hits + misses + refetches);
        }
        double reuse_efficiency() const {
            uint64_t total_accesses = hits + misses + refetches;
            return double(hits) / total_accesses;
        }
    };

    AccessResult access(const TileKey& key, uint64_t cycle);
    bool contains(const TileKey& key) const;
    void evict_if_needed();
    const Stats& stats() const;

    // For visualization
    std::vector<TileEntry> resident_tiles() const;

private:
    uint32_t capacity_;
    std::unordered_map<TileKey, TileEntry> cache_;
    std::set<TileKey> ever_loaded_;  // Track refetches vs first loads
};
```

## Enhanced Trace Format

### New Event Types

```cpp
enum class EventType {
    // Existing
    DMA_LOAD, DMA_STORE,
    BM_PUSH, BM_PULL,
    STR_FEED, STR_DRAIN,
    COMPUTE,

    // New for cache visualization
    L3_ACCESS,      // Cache hit/miss/refetch
    L3_EVICT,       // Tile evicted from cache
    LOOP_STATE,     // Current loop iteration state
    TILE_COMPLETE   // Output tile finished
};
```

### Enhanced Event Structure

```cpp
struct TraceEvent {
    uint64_t cycle;
    EventType type;
    OperandType operand;
    uint32_t tile_i, tile_j, tile_k;

    // L3 cache info
    L3CacheModel::AccessResult cache_result;
    uint32_t l3_occupancy;      // Tiles currently in L3
    uint32_t access_count;      // How many times this tile accessed

    // Loop state
    struct {
        uint32_t outer_idx;     // Outer loop position
        uint32_t middle_idx;    // Middle loop position
        uint32_t inner_idx;     // Inner loop position
    } loop_position;
};
```

### JSON Trace Additions

```json
{
  "metadata": {
    "loop_order": "IKJ",
    "l3_capacity_tiles": 24,
    "l3_config": {
      "a_buffer_tiles": 8,
      "b_buffer_tiles": 12,
      "c_buffer_tiles": 4
    }
  },
  "cache_stats": {
    "total_accesses": 27783,
    "hits": 25000,
    "misses": 441,
    "refetches": 2342,
    "hit_rate": 0.90,
    "reuse_efficiency": 0.90,
    "bandwidth_saved_percent": 89.5
  },
  "events": [
    {
      "cycle": 1000,
      "type": "L3_ACCESS",
      "operand": "A",
      "tile_i": 0, "tile_j": 0,
      "cache_result": "MISS",
      "l3_occupancy": 5,
      "loop_position": {"outer": 0, "middle": 0, "inner": 0}
    }
  ]
}
```

## Block Matrix Visualization

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Block Matrix Decomposition                    L3 Cache State       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   A [1000×100]        B [100×1000]         ┌─────────────────────┐  │
│   63×7 tiles          7×63 tiles           │ L3: 18/24 tiles     │  │
│                                            │                     │  │
│   ┌─┬─┬─┬─┬─┬─┬─┐    ┌─┬─┬─┬─...─┬─┐       │ A: ████░░░░ 4       │  │
│   │●│●│●│○│○│○│○│    │●│●│○│○    │○│       │ B: ████████ 10      │  │
│   ├─┼─┼─┼─┼─┼─┼─┤    ├─┼─┼─┼─...─┼─┤       │ C: ████░░░░ 4       │  │
│   │●│●│●│○│○│○│○│    │●│●│○│○    │○│       │                     │  │
│   ├─┼─┼─┼─┼─┼─┼─┤    │... 7rows ...│       │ Hit Rate: 89.5%     │  │
│   │★│★│★│○│○│○│○│    │●│●│○│○    │○│       │ Reuse: 12.3×        │  │
│   │ ... 63 rows ...  ├─┼─┼─┼─...─┼─┤       └─────────────────────┘  │
│   └─┴─┴─┴─┴─┴─┴─┘    └─┴─┴─┴─...─┴─┘                                │
│                                                                     │
│   C/D [1000×1000]                          Loop Progress            │
│   63×63 tiles                              ┌─────────────────────┐  │
│                                            │ i: ████░░░░ 15/63   │  │
│   ┌─┬─┬─┬─┬─...─┬─┐                        │ k: ███████░ 6/7     │  │
│   │✓│✓│✓│✓│     │○│                        │ j: ████████ 63/63   │  │
│   ├─┼─┼─┼─┼─...─┼─┤                        │                     │  │
│   │✓│✓│✓│★│     │○│                        │ Order: IKJ          │  │
│   │   63×63 grid  │                        └─────────────────────┘  │
│   └─┴─┴─┴─┴─...─┴─┘                                                 │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Legend: ○ Not loaded  ● In L3  ★ Computing  ✓ Complete  ⊗ Refetch  │
└─────────────────────────────────────────────────────────────────────┘
```

### Tile State Colors

| State | Symbol | Color | Meaning |
|-------|--------|-------|---------|
| NOT_LOADED | ○ | Gray (#808080) | Not yet accessed |
| IN_L3 | ● | Blue (#4A90D9) | Currently resident in L3 |
| IN_L2 | ◆ | Cyan (#00CED1) | Pushed to L2 for compute |
| COMPUTING | ★ | Yellow (#FFD700) | Active in systolic array |
| COMPLETE | ✓ | Green (#32CD32) | Finished (for C/D) |
| EVICTED | ◇ | Light blue (#87CEEB) | Was in L3, now evicted |
| REFETCH | ⊗ | Red (#FF6347) | Being refetched (was evicted) |

### Real-Time Updates

The visualization updates on each event:

1. L3_ACCESS with MISS: Tile transitions NOT_LOADED → IN_L3
2. L3_ACCESS with HIT: Flash effect on tile, increment counter
3. L3_ACCESS with REFETCH: Tile shows REFETCH state briefly, then IN_L3
4. L3_EVICT: Tile transitions IN_L3 → EVICTED
5. BM_PUSH: Tile shows IN_L2 state
6. COMPUTE: Tile shows COMPUTING state
7. TILE_COMPLETE: C/D tile transitions to COMPLETE

## Efficiency Metrics

### Core Metrics

```text
Reuse Efficiency = Hits / Total Accesses

Bandwidth Saved = 1 - (Actual Loads / Naive Loads)
  where Naive Loads = Total tile accesses (no reuse)
        Actual Loads = Misses + Refetches

Average Reuse = Hits / (Misses + Refetches)
  → Measures how many times each loaded tile is reused

Refetch Ratio = Refetches / (Misses + Refetches)
  → Lower is better; high ratio means L3 too small
```

### Per-Operand Analysis

```text
A Reuse Factor = A_hits / A_loads
  Theoretical max with IJK: N_tiles = 63×

B Reuse Factor = B_hits / B_loads
  Theoretical max with JIK: M_tiles = 63×

C/D Access Efficiency = 1.0 (always sequential for output)
```

## Implementation Phases

### Phase 1: L3CacheModel

- Implement LRU cache with configurable capacity
- Track hits, misses, refetches per operand
- Generate L3_ACCESS and L3_EVICT events

### Phase 2: Enhanced TiledMatmulProgram

- Add L3Config to TiledMatmulConfig
- Implement all loop orders (IJK, JIK, IKJ, KIJ, BLOCKED)
- Integrate L3CacheModel into execution
- Generate enhanced trace events

### Phase 3: Visualization Updates

- Add left pane with block matrix grids
- Implement tile state coloring
- Add L3 occupancy display
- Add loop progress indicators
- Add real-time metrics display

### Phase 4: Configuration UI

- L3 capacity slider (4-128 tiles)
- Loop order selector
- Speed controls for animation
- Statistics overlay toggle

## File Structure

```text
include/sw/kpu/behavioral/
├── l3_cache_model.hpp           # NEW: L3 cache simulation
├── tiled_matmul_program.hpp     # MODIFY: Add L3 integration
└── ...

src/models/behavioral/
├── l3_cache_model.cpp           # NEW: Implementation
├── tiled_matmul_program.cpp     # MODIFY: Enhanced trace generation
└── ...

tools/visualization/
└── ofg_execution_animation.html # MODIFY: Block matrix visualization

docs/design/
└── tiled-matmul-cache-reuse.md  # This document
```

## Success Criteria

1. **Correctness**: Matrix multiplication produces correct results
2. **Cache Model Accuracy**: Hit/miss/refetch counts match expected patterns
3. **Loop Order Impact**: Different orders show different reuse characteristics
4. **Visualization Clarity**: Block matrices update in real-time with each event
5. **Metrics Accuracy**: Reported efficiency matches actual behavior

## Future Extensions

- Support for multi-level blocking (L3 + L2 blocking)
- Prefetching strategies visualization
- Memory bandwidth utilization display
- Comparison mode: side-by-side different loop orders
- Export statistics to CSV for analysis
