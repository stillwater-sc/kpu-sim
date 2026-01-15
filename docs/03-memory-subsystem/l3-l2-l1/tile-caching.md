# Tile Caching Architecture: Design and Implementation Plan

This document describes the tile caching architecture for the KPU memory hierarchy, addressing tile reuse optimization, protection guarantees, and the implementation roadmap.

## Problem Statement

The current implementation reloads tiles from external memory on every access, even when the same tile could be reused from L3 cache. For a 64×64×64 matmul with 32×32×32 tiling:

**Current behavior (no reuse):**
```
Loop: for ti=0..1, for tj=0..1, for tk=0..1

ti=0, tj=0: Load A[0,0], A[0,1], B[0,0], B[1,0] → 4 tiles
ti=0, tj=1: Load A[0,0], A[0,1], B[0,1], B[1,1] → 4 tiles (A reloaded!)
ti=1, tj=0: Load A[1,0], A[1,1], B[0,0], B[1,0] → 4 tiles (B reloaded!)
ti=1, tj=1: Load A[1,0], A[1,1], B[0,1], B[1,1] → 4 tiles

Total: 16 A/B tile loads (8 redundant) + 4 C stores = 20 DMA ops
```

**Optimal behavior (with reuse):**
```
- Each A tile loaded once, reused across all tj iterations
- Each B tile loaded once, reused across all ti iterations

Total: 8 A/B tile loads (0 redundant) + 4 C stores = 12 DMA ops
```

**Impact:**
| Metric | No Reuse | With Reuse | Improvement |
|--------|----------|------------|-------------|
| External memory traffic | 0.08 MB | 0.05 MB | 1.6× less |
| Reuse factor | 1.67× | 1.0× (optimal) | - |
| DMA operations | 20 | 12 | 40% fewer |

## Architectural Considerations

### Third-Party Replacement Threats

In a real system, cached tiles could be invalidated by:

| Threat | Description | Risk Level |
|--------|-------------|------------|
| **Multi-tenant execution** | Another process/kernel uses the KPU | High |
| **OS preemption** | Kernel preempts execution mid-schedule | Medium |
| **Host DMA** | Host CPU writes to external memory region | Medium |
| **Cache coherency** | Another agent modifies source data | Low (KPU is typically non-coherent) |

### Protection Requirements

1. **Context isolation**: Kernel A cannot evict tiles owned by Kernel B
2. **Tile locking**: Tiles marked for reuse cannot be evicted until released
3. **Memory ordering**: Fences ensure loads complete before compute, stores complete before host notification
4. **Capacity management**: Graceful handling when L3 is full

## Software vs Hardware Approaches

### Option 1: Pure Software Tracking (Phase 1)

The compiler/simulator tracks which tiles are in L3:

```cpp
struct TileKey {
    MatrixID matrix;
    uint16_t ti, tj, tk;
};

class TileCache {
    std::set<TileKey> resident_tiles;
    size_t capacity_bytes;
    size_t used_bytes;

    bool is_resident(TileKey key);
    bool can_allocate(size_t bytes);
    void allocate(TileKey key, size_t bytes);
    void evict_lru();
};
```

**Pros:**
- No ISA changes required
- Simple to implement
- Good for initial performance modeling

**Cons:**
- No hardware protection guarantees
- Assumes single-tenant, non-preemptive execution
- Software must perfectly track hardware state

### Option 2: Hardware-Assisted Tile Management (Phase 2-3)

Hardware provides tile caching with protection:

#### Tile Descriptor Table (TDT)

A hardware structure mapping logical tiles to physical L3 locations:

```
┌─────────────────────────────────────────────────────────────┐
│                   Tile Descriptor Table                      │
├──────────┬──────────┬───────────┬─────────┬─────────────────┤
│ Matrix   │ TileCoord│ L3 Tile   │ Status  │ Reference Count │
├──────────┼──────────┼───────────┼─────────┼─────────────────┤
│ A        │ (0,0)    │ L3[0]     │ VALID   │ 2               │
│ A        │ (0,1)    │ L3[1]     │ VALID   │ 1               │
│ B        │ (0,0)    │ L3[2]     │ VALID   │ 2               │
│ B        │ (0,1)    │ L3[3]     │ LOADING │ 0               │
└──────────┴──────────┴───────────┴─────────┴─────────────────┘
```

#### Tile Descriptor Structure

```cpp
struct TileDescriptor {
    // Identification
    uint8_t  context_id;     // Owning kernel/process (for isolation)
    uint8_t  matrix_id;      // A=0, B=1, C=2
    uint16_t tile_ti;        // M-dimension tile index
    uint16_t tile_tj;        // N-dimension tile index
    uint16_t tile_tk;        // K-dimension tile index

    // Physical location
    uint8_t  l3_tile_id;     // Which physical L3 tile (0-3 typically)
    uint32_t l3_offset;      // Offset within L3 tile
    uint32_t size_bytes;     // Tile size

    // State
    uint8_t  refcount;       // Active references (0 = evictable)
    uint8_t  flags;          // Status flags (below)
};

// Flag bits
constexpr uint8_t TILE_VALID    = 0x01;  // Contains valid data
constexpr uint8_t TILE_DIRTY    = 0x02;  // Modified, needs writeback
constexpr uint8_t TILE_LOCKED   = 0x04;  // Cannot be evicted
constexpr uint8_t TILE_LOADING  = 0x08;  // DMA in progress
constexpr uint8_t TILE_PREFETCH = 0x10;  // Prefetched, low priority
```

## Proposed ISA Extensions (Phase 2)

### New Instructions

```cpp
enum class DMOpcode : uint8_t {
    // Existing tile operations
    DMA_LOAD_TILE,           // Unconditional load (current)
    DMA_STORE_TILE,          // Unconditional store (current)

    // New: Cached tile operations
    DMA_LOAD_TILE_CACHED,    // Load if miss, refcount++ on hit or after load
    DMA_PREFETCH_TILE_CACHED,// Non-blocking cached load (hint)

    // New: Tile lifecycle management
    TILE_ACQUIRE,            // Increment refcount (assert tile valid)
    TILE_RELEASE,            // Decrement refcount, allow eviction when 0
    TILE_INVALIDATE,         // Force eviction, writeback if dirty

    // New: Synchronization
    TILE_FENCE,              // Wait for tile operations to complete
    TILE_QUERY,              // Query tile status (for debugging/profiling)
};
```

### Instruction Semantics

#### DMA_LOAD_TILE_CACHED

```cpp
struct DMALoadTileCachedOperands {
    MatrixID matrix;
    TileCoord tile;
    Address ext_mem_addr;
    Size size_bytes;

    // Flags
    bool lock;      // Set TILE_LOCKED flag (prevent eviction)
    bool prefetch;  // Non-blocking, set TILE_PREFETCH flag
};

// Hardware behavior:
// 1. Lookup (matrix, tile) in TDT
// 2. If HIT and VALID:
//    - refcount++
//    - Return immediately (0 cycles for data, 1 cycle for lookup)
// 3. If MISS:
//    - Allocate L3 slot (evict LRU with refcount=0 if needed)
//    - Create TDT entry
//    - Issue DMA transfer
//    - Set LOADING flag
//    - When complete: clear LOADING, set VALID, refcount=1
// 4. If lock=true: set LOCKED flag
```

#### TILE_RELEASE

```cpp
struct TileReleaseOperands {
    MatrixID matrix;
    TileCoord tile;
};

// Hardware behavior:
// 1. Lookup (matrix, tile) in TDT
// 2. Assert entry exists and VALID
// 3. refcount--
// 4. If refcount == 0: clear LOCKED flag, tile becomes evictable
```

#### TILE_FENCE

```cpp
struct TileFenceOperands {
    enum Scope { CHANNEL, ALL } scope;
};

// Hardware behavior:
// - Wait for all pending tile operations to complete
// - CHANNEL: only this memory channel
// - ALL: all channels (global fence)
```

## Hardware Architecture

### Tile Cache Controller Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Tile Cache Controller                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐                      ┌──────────────────┐         │
│  │  Command Queue   │                      │  Status Register │         │
│  │  (from ISA)      │                      │  (to ISA)        │         │
│  └────────┬─────────┘                      └────────▲─────────┘         │
│           │                                         │                   │
│           ▼                                         │                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         Lookup Engine                           │    │
│  │  ┌─────────────┐    ┌─────────────────────┐    ┌─────────────┐  │    │
│  │  │ Tag CAM     │    │ Tile Descriptor     │    │ Hit/Miss    │  │    │
│  │  │ (matrix,    │◄──▶│ Table (TDT)         │◄──▶│ Logic       │  │    │
│  │  │  ti,tj,tk)  │    │ (N entries)         │    │             │  │    │
│  │  └─────────────┘    └─────────────────────┘    └─────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                      │                       │              │
│           ▼                      ▼                       ▼              │
│  ┌─────────────┐       ┌─────────────────┐      ┌─────────────────┐     │
│  │ Refcount    │       │ Eviction        │      │ Allocation      │     │
│  │ Manager     │       │ Controller      │      │ Manager         │     │
│  │             │       │ (LRU + refcnt)  │      │                 │     │
│  └─────────────┘       └─────────────────┘      └─────────────────┘     │
│           │                      │                       │              │
│           └──────────────────────┼───────────────────────┘              │
│                                  ▼                                      │
│                    ┌─────────────────────────┐                          │
│                    │      DMA Scheduler      │                          │
│                    │  - Load requests        │                          │
│                    │  - Writeback requests   │                          │
│                    └───────────┬─────────────┘                          │
│                                │                                        │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      DMA Engine         │
                    │  (to/from ext memory)   │
                    └─────────────────────────┘
```

### Eviction Policy

The eviction controller selects victims using:

1. **Refcount = 0**: Only tiles with no active references are candidates
2. **Not LOCKED**: LOCKED tiles are never evicted
3. **LRU**: Among candidates, evict least recently used
4. **Prefer PREFETCH**: Prefetched tiles evicted before demand-loaded tiles
5. **Writeback**: If DIRTY, schedule writeback before reusing slot

```cpp
TileDescriptor* select_eviction_victim() {
    TileDescriptor* victim = nullptr;
    uint64_t oldest_access = UINT64_MAX;

    for (auto& entry : tdt) {
        if (entry.refcount > 0) continue;      // In use
        if (entry.flags & TILE_LOCKED) continue; // Locked

        // Prefer prefetch tiles
        uint64_t priority = entry.last_access_cycle;
        if (entry.flags & TILE_PREFETCH) {
            priority -= 1000000;  // Much more likely to evict
        }

        if (priority < oldest_access) {
            oldest_access = priority;
            victim = &entry;
        }
    }

    return victim;  // nullptr if no evictable tiles (stall required)
}
```

## Implementation Roadmap

### Phase 1: Software Tile Cache Simulation (Current)

**Goal**: Get accurate reuse numbers without ISA changes

**Changes**:
1. Add `TileCache` class to track simulated L3 residency
2. Modify `OutputStationaryProgramBuilder` to query cache before emitting loads
3. Add cache hit/miss statistics to execution reports

**Files to modify**:
- `include/sw/kpu/isa/tile_cache.hpp` (new)
- `src/isa/tile_cache.cpp` (new)
- `src/isa/data_movement_isa.cpp` (program builder)
- `src/isa/concurrent_executor.cpp` (statistics)

**Deliverables**:
- Tile reuse factor improves from 1.67× to 1.0× for simple cases
- Cache hit/miss rates reported in execution output
- L3 capacity constraints respected

### Phase 2: ISA Extensions

**Goal**: Add cached load instructions and refcount management

**Changes**:
1. Add new opcodes to `DMOpcode` enum
2. Define operand structures for new instructions
3. Implement hardware behavior in executor
4. Update program builder to emit cached loads

**Files to modify**:
- `include/sw/kpu/isa/data_movement_isa.hpp` (new opcodes)
- `include/sw/kpu/isa/concurrent_executor.hpp` (TDT structure)
- `src/isa/concurrent_executor.cpp` (instruction execution)
- `src/isa/data_movement_isa.cpp` (instruction factories)

**Deliverables**:
- `DMA_LOAD_TILE_CACHED`, `TILE_RELEASE`, `TILE_FENCE` instructions
- Refcount-based eviction policy
- Context isolation (multi-tenant support)

### Phase 3: Hardware Modeling

**Goal**: Accurate cycle-level modeling of tile cache hardware

**Changes**:
1. Model TDT lookup latency (CAM search)
2. Model eviction decision latency
3. Model writeback scheduling
4. Add contention modeling for concurrent accesses

**Files to modify**:
- `include/sw/kpu/isa/tile_cache_controller.hpp` (new)
- `src/isa/tile_cache_controller.cpp` (new)
- `src/isa/concurrent_executor.cpp` (integration)

**Deliverables**:
- TDT lookup: 1 cycle (hit), 2 cycles (miss + allocate)
- Eviction overhead modeled
- Writeback traffic included in bandwidth calculations

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Reuse factor (64³ matmul) | 1.0× (optimal) |
| 1 | DMA operations reduction | 40% fewer |
| 2 | Multi-kernel isolation | No cross-contamination |
| 2 | Locked tile guarantee | 0 unexpected evictions |
| 3 | TDT lookup accuracy | ±1 cycle |
| 3 | Total cycle count accuracy | ±5% vs RTL |

## Appendix: Example Schedule with Tile Caching

### Before (No Caching)

```
// ti=0, tj=0
DMA_LOAD A[0,0]    // 64 cycles
DMA_LOAD B[0,0]    // 64 cycles
BARRIER
... compute ...

// ti=0, tj=1
DMA_LOAD A[0,0]    // 64 cycles (REDUNDANT!)
DMA_LOAD B[0,1]    // 64 cycles
BARRIER
... compute ...
```

### After (With Caching)

```
// ti=0, tj=0
DMA_LOAD_CACHED A[0,0], LOCK=true   // 64 cycles (miss)
DMA_LOAD_CACHED B[0,0], LOCK=true   // 64 cycles (miss)
BARRIER
... compute ...
TILE_RELEASE B[0,0]                  // B no longer needed for this tj

// ti=0, tj=1
DMA_LOAD_CACHED A[0,0], LOCK=true   // 1 cycle (HIT!)
DMA_LOAD_CACHED B[0,1], LOCK=true   // 64 cycles (miss)
BARRIER
... compute ...
TILE_RELEASE A[0,0]                  // Last use of A[0,0] in this ti
TILE_RELEASE B[0,1]
```

**Cycle savings**: 63 cycles per reused tile × number of reuses
