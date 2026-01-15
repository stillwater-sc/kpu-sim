# KPU Execution Model: Credit-Based Dataflow

## Overview

The KPU (Knowledge Processing Unit) implements a **credit-based dataflow execution model**, fundamentally different from stored-program (von Neumann) architectures. This document is the authoritative reference for KPU execution semantics.

## Core Principles

### 1. Credits Flow Upstream, Data Flows Downstream

```
                        CREDITS (upstream)
                             ↑
    ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
    │   Host   │ ───→  │    L3    │ ───→  │    L2    │ ───→  │    L1    │ ───→ Compute
    │  Memory  │       │  Buffers │       │  Buffers │       │  Streams │
    └──────────┘       └──────────┘       └──────────┘       └──────────┘
                             ↓
                        DATA/TILES (downstream)
```

**Rule:** A producer can only push data when it has received a credit from a downstream buffer.

### 2. No Cache, No Misses

- L3, L2, and L1 are **buffers**, NOT caches
- There is no demand-driven fetching
- There is no concept of "cache miss" or "cache hit"
- Data movement is **producer-push** with **consumer-credit**
- Progress is made by **writing inputs to downstream operators**

### 3. Firing Rules

A dataflow node **fires** (executes) when:
1. **All input tokens are ready** (data available)
2. **Downstream has credit** (space available for output)

This is the fundamental synchronization mechanism - no locks, no polling, just token availability.

---

## Memory Hierarchy as Buffers

### L3 Tile Buffers

L3 is a collection of **tile buffers**, not a cache.

| Property | Value |
|----------|-------|
| Purpose | Stage tiles from host memory for BlockMover consumption |
| Capacity | Fixed number of tile slots (e.g., 4 buffers) |
| Addressing | By buffer ID, not by content address |
| Replacement | None - buffers are explicitly managed via credits |

**Correct Mental Model:**
```
L3[0]: [ tile slot ] ← DMA writes here when it has credit for buffer 0
L3[1]: [ tile slot ] ← DMA writes here when it has credit for buffer 1
L3[2]: [ tile slot ] ← DMA writes here when it has credit for buffer 2
L3[3]: [ tile slot ] ← DMA writes here when it has credit for buffer 3
```

**WRONG Mental Model (Cache):**
```
L3 Cache: { A[0,0], B[1,2], A[0,1], ... } ← LRU eviction, content-addressed
```

### L2 Bank Buffers

L2 is a collection of **bank buffers** for operand staging.

| Property | Value |
|----------|-------|
| Purpose | Stage tiles from L3 for Streamer consumption |
| Capacity | Fixed number of banks (e.g., 8 banks) |
| Organization | Banks partitioned by operand type (A banks, B banks, C banks) |
| Addressing | By bank ID |

### L1 Stream Buffers

L1 is a collection of **stream buffers** (edge registers) feeding the systolic array.

| Property | Value |
|----------|-------|
| Purpose | Buffer rows/columns for systolic array feeding |
| Capacity | Matches systolic array dimensions |
| Organization | West buffers (A rows), North buffers (B columns) |

---

## Component Execution Models

### DMA Engine

**Role:** Orchestrated data mover between host memory and L3 buffers.

**Execution Model:**
```
INPUTS:
  - Programmed descriptors (what tiles, what addresses)
  - BUFFER_AVAILABLE(L3[i]) credit from downstream

BEHAVIOR:
  1. WAIT for BUFFER_AVAILABLE(L3[i]) credit
  2. Fetch tile from host memory
  3. PUSH tile to L3[i]
  4. Emit TILE_READY(T @ L3[i]) downstream

OUTPUTS:
  - TILE_READY(T @ L3[i]) token to BlockMover
```

**State Diagram:**
```
    ┌─────────────────────────────────────────────┐
    │                                             │
    ▼                                             │
┌────────┐  credit arrives  ┌──────────┐         │
│ WAIT   │ ───────────────→ │ FETCHING │         │
│ CREDIT │                  │          │         │
└────────┘                  └────┬─────┘         │
                                 │ fetch complete │
                                 ▼               │
                           ┌──────────┐          │
                           │ PUSH TO  │──────────┘
                           │   L3     │ emit TILE_READY
                           └──────────┘
```

**WRONG:** DMA checking if tile is "in cache" and skipping fetch on "hit".

### BlockMover

**Role:** Move tiles between L3 and L2, with optional transpose.

**Execution Model:**
```
INPUTS:
  - TILE_READY(T @ L3[i]) from DMA (tile has arrived)
  - BUFFER_AVAILABLE(L2[j]) credit from Streamer (implicit via bank availability)

BEHAVIOR:
  1. WAIT for TILE_READY(T @ L3[i]) via tag CAM match
  2. CHECK L2[j] has space (credit available)
  3. PUSH tile from L3[i] to L2[j]
  4. Emit TILE_READY(T @ L2[j]) downstream
  5. Emit BUFFER_AVAILABLE(L3[i]) upstream (credit return)

OUTPUTS:
  - TILE_READY(T @ L2[j]) token to Streamer
  - BUFFER_AVAILABLE(L3[i]) credit to DMA
```

**Key Insight:** When BlockMover consumes a tile from L3, it **returns a credit** to DMA, enabling DMA to fetch the next tile into that buffer.

**Tag CAM:** BlockMover uses a tag Content-Addressable Memory to handle out-of-order tile arrivals. It waits for a specific tile tag, not just any tile.

### Streamer

**Role:** Feed tiles from L2 to L1 stream buffers for systolic array.

**Execution Model:**
```
INPUTS:
  - TILE_READY(T @ L2[j]) from BlockMover
  - L1 stream buffer credit (implicit via compute ready)

BEHAVIOR:
  1. WAIT for TILE_READY(T @ L2[j]) via tag CAM match
  2. CHECK L1 stream buffer has space
  3. PUSH row/column data to L1 stream buffers
  4. Emit BUFFER_AVAILABLE(L2[j]) upstream (credit return)

OUTPUTS:
  - Data in L1 stream buffers (feeds systolic array)
  - BUFFER_AVAILABLE(L2[j]) credit to BlockMover
```

### Compute Fabric (Systolic Array)

**Role:** Execute matrix multiplication on streaming data.

**Execution Model:**
```
INPUTS:
  - A rows from West L1 stream buffers
  - B columns from North L1 stream buffers

BEHAVIOR:
  - Systolic data flow through PE array
  - Accumulation in output registers

OUTPUTS:
  - Result tile in accumulator registers
  - Drain triggers writeback to L2
```

---

## Token Types

### TILE_READY Token

Signals that a specific tile has arrived at a location.

```cpp
struct TileReadyToken {
    OperandType operand;    // A, B, C, D
    uint16_t tile_i;        // Tile row index
    uint16_t tile_j;        // Tile column index
    uint16_t tile_k;        // K dimension index (for A, B)
    Location location;      // L3, L2, L1
    uint8_t buffer_id;      // Which buffer/bank
};
```

**Semantics:** "Tile T is now available at location L in buffer B"

### BUFFER_AVAILABLE Token (Credit)

Signals that a buffer slot is available for writing.

```cpp
struct BufferAvailableToken {
    Location location;      // L3, L2, L1
    uint8_t buffer_id;      // Which buffer/bank is available
};
```

**Semantics:** "Buffer B at location L can accept a new tile"

**Flow Direction:** Always upstream (consumer → producer)

---

## Trace Event Semantics

### Correct Event Types

| Event | Meaning | Direction |
|-------|---------|-----------|
| `TILE_READY` | Tile T arrived at location L | Downstream |
| `BUFFER_AVAILABLE` | Buffer B is free to accept data | Upstream (credit) |
| `DMA_FETCH` | DMA fetching tile from host | Internal |
| `DMA_PUSH` | DMA pushing tile to L3 | Downstream |
| `BM_PUSH` | BlockMover pushing tile L3→L2 | Downstream |
| `STR_FEED` | Streamer feeding tile L2→L1 | Downstream |
| `COMPUTE` | Systolic array active | Internal |
| `DRAIN` | Result draining from accumulator | Downstream |

### WRONG Event Types (Do Not Use)

| Wrong Event | Why It's Wrong |
|-------------|----------------|
| `L3_ACCESS` | Implies cache lookup - L3 is a buffer, not cache |
| `CACHE_HIT` | No cache in dataflow - only buffers |
| `CACHE_MISS` | No cache in dataflow - only "waiting for tile" |
| `L3_EVICT` | No eviction - explicit credit-based management |
| `REFETCH` | No refetch - tiles flow once through pipeline |

---

## Execution Flow Example

### Single Tile: A[0,0] from Host to Compute

```
Time  Event                              Credits                  Data
────  ─────                              ───────                  ────
T0    Initial state                      L3[0]=1, L2[0]=1         -
T1    DMA consumes L3[0] credit          L3[0]=0                  -
T2    DMA fetches A[0,0] from host       -                        fetching
T3    DMA pushes to L3[0]                -                        A[0,0]@L3[0]
T4    TILE_READY(A[0,0]@L3[0])           -                        ↓
T5    BM consumes L2[0] credit           L2[0]=0                  -
T6    BM pushes A[0,0] to L2[0]          -                        A[0,0]@L2[0]
T7    BM returns L3[0] credit            L3[0]=1                  -
T8    TILE_READY(A[0,0]@L2[0])           -                        ↓
T9    STR feeds A[0,0] to L1             -                        A[0,0]@L1
T10   STR returns L2[0] credit           L2[0]=1                  -
T11   Compute consumes A[0,0]            -                        processed
```

**Key Observations:**
1. Each downstream push is preceded by upstream credit consumption
2. Each consumption returns a credit to the producer
3. No polling, no cache lookups - pure token flow
4. Backpressure: If L2 is full, BlockMover waits (doesn't drop or evict)

---

## Double Buffering

Double buffering enables pipelining by having two buffer slots per operand stream.

```
L3[0] ←→ DMA fills while L3[1] drains to L2
L3[1] ←→ DMA fills while L3[0] drains to L2

Credits:
- Initially: L3[0]=1, L3[1]=1 (both available)
- DMA consumes L3[0] credit, starts filling L3[0]
- BM drains L3[1] (previously filled), returns L3[1] credit
- DMA consumes L3[1] credit, starts filling L3[1]
- BM drains L3[0], returns L3[0] credit
- ... ping-pong continues
```

---

## Loop Order Effects in Dataflow

In a credit-based dataflow system, loop order affects:

1. **When credits return:** Different orders cause credits to return at different times
2. **Buffer utilization:** Some orders keep buffers fuller than others
3. **Pipelining efficiency:** Affects how well stages overlap

Loop order does NOT affect:
- Whether tiles are "cached" (no cache!)
- Cache "hit rates" (no cache!)
- "Reuse" in the cache sense

**Correct Thinking:** "This loop order keeps B tiles in L3 longer before their buffers are reused"

**WRONG Thinking:** "This loop order gives better cache hit rates for B tiles"

---

## Invariants

### Must Always Hold

1. **Credit Conservation:** Total credits + occupied buffers = buffer capacity
2. **No Orphan Data:** Every tile in a buffer has a consumer waiting
3. **No Deadlock:** Cyclic credit dependencies are impossible by construction
4. **Progress Guarantee:** If inputs ready and credit available, node fires

### Never Allowed

1. **Demand-Driven Fetch:** No component "requests" data from upstream
2. **Cache Semantics:** No HIT/MISS/EVICT terminology
3. **Content-Addressed Buffers:** Buffers addressed by ID, not by tile content
4. **Speculative Execution:** Data only moves when credit is available

---

## Implementation Reference

### Correct Implementation (use these)

```
include/sw/kpu/models/dataflow/
├── flow_graph_executor.hpp      # Base dataflow executor
├── dma_flow_executor.hpp        # DMA with credit semantics
├── block_mover_flow_executor.hpp # BlockMover with credit/push
└── streamer_flow_executor.hpp   # Streamer with credit/push
```

### Incorrect Implementation (deprecated for correctness-critical code)

```
include/sw/kpu/behavioral/
├── l3_cache_model.hpp           # WRONG: Cache semantics
└── tiled_matmul_program.hpp     # WRONG: Uses cache model
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Credit** | Permission to write to a downstream buffer |
| **Token** | Data item (tile) plus its location metadata |
| **Fire** | Execute a dataflow node when inputs ready and credit available |
| **Push** | Producer writes data to consumer buffer (with credit) |
| **Backpressure** | Stalling when downstream has no credit |
| **Tag CAM** | Content-Addressable Memory for matching tile tags |

---

## Document History

| Date | Change |
|------|--------|
| 2025-01-14 | Initial version documenting correct dataflow semantics |
