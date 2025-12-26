# Systolic Array Tile Scheduling Analysis

## Problem Statement

For a 16×16 systolic array executing a 64×64 matmul, we should achieve at least **16/17 ≈ 94% utilization** with proper pipelining. This document analyzes the tile scheduling requirements and bandwidth implications.

## Systolic Array Timing Model

### Output-Stationary Dataflow

In output-stationary mode:
- Partial sums C[i,j] remain stationary in PE(i,j)
- A rows stream from the west (left edge)
- B columns stream from the north (top edge)
- Each PE accumulates: `c[i,j] += a[i,k] * b[k,j]` for all k

### Timing for 16×16 Array

At cycle t:
- Element `a[row][t mod K]` enters row `row` from the west
- Element `b[t mod K][col]` enters column `col` from the north
- Due to wavefront propagation, PE(i,j) receives data at cycle `t + i + j`

For K accumulations:
```
First element enters:     cycle 0
Last element enters:      cycle K-1
Last PE computation:      cycle K-1 + (15 + 15) = K + 29

Total cycles = K + 30    (for 16×16 array, includes fill + drain)
```

### Efficiency Calculation

For K=64 (full 64×64 matmul reduced to one 16×16 output tile):
```
Compute cycles = 64 + 30 = 94
Ideal cycles = 64 (one per accumulation step)
Efficiency = 64/94 ≈ 68%
```

But with **pipelining across output tiles**, the fill/drain overhead amortizes:
```
For N output tiles computed back-to-back:
  Total cycles = K + 30 + (N-1) × K
  Efficiency → K/(K + 30/N) → ~100% as N increases
```

The 16/17 = 94% comes from:
- Per output tile: K cycles of useful compute + ~K/16 cycles of overhead
- With tight pipelining: 16 cycles compute + 1 cycle overhead per tile

---

## Tile Decomposition: 64×64 Matmul

### Tile Structure

```
A: 64×64 → 4×4 tiles of 16×16 each: A[i,k] for i,k ∈ {0,1,2,3}
B: 64×64 → 4×4 tiles of 16×16 each: B[k,j] for k,j ∈ {0,1,2,3}
C: 64×64 → 4×4 tiles of 16×16 each: C[i,j] for i,j ∈ {0,1,2,3}
```

### Computing One Output Tile

To compute C[i,j] completely (without partial writeback):
```
C[i,j] = A[i,0]×B[0,j] + A[i,1]×B[1,j] + A[i,2]×B[2,j] + A[i,3]×B[3,j]
```

This requires:
- 4 A tiles: A[i,0], A[i,1], A[i,2], A[i,3]
- 4 B tiles: B[0,j], B[1,j], B[2,j], B[3,j]
- Stream 64 rows of A (16 from each tile)
- Stream 64 columns of B (16 from each tile)

---

## Schedule Analysis

### Schedule A: Fully Pipelined (No Partial C Writeback)

**Requirements:**
- L2 must hold all k-dimension tiles for one output tile
- For C[i,j]: 4 A tiles + 4 B tiles = 8 tiles = 8 KB
- Plus C output buffer: 1 KB
- **Total L2 for one active output: 9 KB**

**Data Flow:**
```
Time →
──────────────────────────────────────────────────────────────────
DMA:     [Load A[0,*]] [Load B[*,0..3]]
L3→L2:   [A tiles to L2] [B tiles to L2]
L2→L1:   [Stream A[0,0]→A[0,3]] [Stream B[0,0]→B[3,0]]  ← interleaved
Compute: [────────────── C[0,0] accumulates ──────────────]
L1→L2:   [Drain C[0,0]]
```

**Bandwidth per Output Tile:**
```
L2 Read:  4 A tiles + 4 B tiles = 8 KB
L2 Write: 1 C tile = 1 KB
Total:    9 KB per output tile
```

**For all 16 C tiles (naive, no reuse):**
```
16 × 9 KB = 144 KB L2 traffic
```

**With A-row reuse (process C[i,*] together):**
```
For row i of C (4 output tiles):
  - Load A[i,*] once: 4 tiles = 4 KB (reused 4 times)
  - Load B[*,j] for j=0,1,2,3: 4×4 = 16 tiles = 16 KB
  - Write C[i,*]: 4 tiles = 4 KB
  Total per C row: 24 KB

Total for all 4 C rows: 4 × 24 = 96 KB
Savings vs naive: 144 - 96 = 48 KB (33% reduction)
```

**With optimal A and B reuse (need sufficient L2):**
```
- Load all A once: 16 tiles = 16 KB
- Load all B once: 16 tiles = 16 KB
- Write all C: 16 tiles = 16 KB
Total: 48 KB (matches raw data size)
```

---

### Schedule B: With Partial C Writeback (Limited L2 Space)

If L2 can only hold 2 A tiles and 2 B tiles at a time:

**Process each C[i,j] in two passes:**
```
Pass 1: k=0,1
  - Load A[i,0], A[i,1], B[0,j], B[1,j]
  - Compute partial C[i,j]
  - Write partial C[i,j] to L2

Pass 2: k=2,3
  - Load A[i,2], A[i,3], B[2,j], B[3,j]
  - Read partial C[i,j] from L2
  - Accumulate, write final C[i,j]
```

**Bandwidth per Output Tile:**
```
L2 Read:  4 A tiles + 4 B tiles + 1 partial C = 9 KB
L2 Write: 1 partial C + 1 final C = 2 KB
Total:    11 KB per output tile
```

**For all 16 C tiles:**
```
16 × 11 KB = 176 KB L2 traffic
Extra vs Schedule A: 176 - 144 = 32 KB (22% overhead)
```

---

## L2 Storage Requirements Analysis

### Minimum for Schedule A (no partial writeback)

For one active output tile:
```
4 A tiles (k-dimension) × 1 KB = 4 KB
4 B tiles (k-dimension) × 1 KB = 4 KB
1 C tile output buffer    = 1 KB
────────────────────────────────
Total: 9 KB minimum
```

### With Double Buffering (for pipelining)

To overlap data movement with compute:
```
Buffer set 0 (computing): 8 KB input + 1 KB output
Buffer set 1 (loading):   8 KB input
────────────────────────────────────────────────────
Total: 17 KB for double-buffered operation
```

### With A-Row Reuse + Double Buffering

Process C[i,0], C[i,1], C[i,2], C[i,3] before moving to next i:
```
A[i,*] tiles (resident): 4 KB (reused for all j)
B[*,j] double buffer:    2 × 4 KB = 8 KB
C[i,j] output buffer:    1 KB
────────────────────────────────────────────────────
Total: 13 KB
```

---

## Pipelined Execution Timeline

### Single Output Tile (No Pipelining)

```
Cycle:  0        64       94
        |--------|--------|
        [Stream A+B] [Drain]
        [Accumulate ]

Fill: 30 cycles, Compute: 64 cycles, Drain: 30 cycles (overlapped)
Total: 94 cycles
Efficiency: 64/94 = 68%
```

### Pipelined Output Tiles

```
Tile 0:  [Fill|──────── Compute ────────|Drain]
Tile 1:       [Fill|──────── Compute ────────|Drain]
Tile 2:            [Fill|──────── Compute ────────|Drain]
...

Cycle:   0    30   64   94  128  158  192
         |──────|──────|──────|──────|──────|
         [  T0  ][  T1  ][  T2  ][  T3  ]...

With perfect pipelining:
  - First tile: 94 cycles (includes fill + drain)
  - Each additional tile: 64 cycles (just compute)

For 16 output tiles:
  Total = 94 + 15×64 = 94 + 960 = 1054 cycles
  Total MACs = 16 × 64 × 256 = 262,144
  Ideal cycles = 262,144 / 256 = 1024
  Efficiency = 1024/1054 = 97%
```

### Tighter Pipelining with Streaming

With register-based FIFOs at array edges:
```
The systolic array can accept new data every cycle.
If we interleave A and B streams properly:

Cycle 0:    A[i,0] row 0 → west edge, B[0,j] col 0 → north edge
Cycle 1:    A[i,0] row 1 → west edge, B[0,j] col 1 → north edge
...
Cycle 15:   A[i,0] row 15, B[0,j] col 15
Cycle 16:   A[i,1] row 0, B[1,j] col 0   ← seamless transition!
...
Cycle 63:   A[i,3] row 15, B[3,j] col 15

After fill (30 cycles), steady-state produces 1 complete output per cycle.
```

---

## Current Implementation Issues

The current `OutputStationaryProgramBuilder::build()` has the following structure:

```cpp
for ti in 0..M/Ti:
  for tj in 0..N/Tj:
    for tk in 0..K/Tk:
      Load A[ti,tk], B[tk,tj]
      BARRIER                  // <-- Unnecessary!
      Move A, B to L2
      BARRIER                  // <-- Unnecessary!
      Stream A rows, B cols
      BARRIER                  // <-- Prevents pipelining!
    Drain C
    Store C
```

**Problems:**
1. Barriers after every operation prevent overlap
2. Each k-iteration treated separately instead of continuously
3. No prefetching of next data while current is processing

**Correct Pipelined Structure:**

```cpp
for ti in 0..M/Ti:
  for tj in 0..N/Tj:
    // Prefetch first k-tiles
    Load A[ti,0], B[0,tj]
    Move A[ti,0], B[0,tj] to L2

    for tk in 0..K/Tk:
      // OVERLAPPED: prefetch next while computing current
      if tk+1 < K/Tk:
        Load A[ti,tk+1], B[tk+1,tj]        // DMA (overlapped)
        Move A[ti,tk+1], B[tk+1,tj] to L2  // BM (overlapped)

      // Stream current - feeds systolic array
      Stream A[ti,tk] rows, B[tk,tj] cols  // Continuous, no barrier!
      // Accumulation happens in PEs (output-stationary)

    BARRIER  // Only after ALL k tiles streamed
    Drain C[ti,tj]
    Store C[ti,tj]
```

---

## Streaming Data Organization

### L1 Buffer Requirements

For continuous streaming without stalls:
```
A buffer: Must hold one complete tile row set (16 elements × 4 bytes = 64 bytes/row)
          For 4 tiles in k: 4 × 16 rows × 64 bytes = 4 KB

B buffer: Must hold one complete tile column set
          For 4 tiles in k: 4 × 16 cols × 64 bytes = 4 KB

Total L1: 8 KB for input streaming
```

### Register FIFO at Array Edges

```
West edge FIFOs (16 total, one per row):
  - Depth: ~32 elements for latency hiding
  - Width: 4 bytes (float32)
  - Total: 16 × 32 × 4 = 2 KB

North edge FIFOs (16 total, one per column):
  - Same sizing
  - Total: 2 KB

Total edge buffers: 4 KB
```

---

## Optimal Schedule Implementation

### Phase 1: L2 Tile Layout

```cpp
// L2 bank assignment for 64×64 matmul
struct TileLayout {
    // A tiles: A[i,k] stored in bank (i*4 + k) % 8
    // B tiles: B[k,j] stored in bank (k*4 + j + 4) % 8
    // This spreads tiles across banks for parallel access

    static uint8_t a_bank(size_t i, size_t k) { return (i*4 + k) % 8; }
    static uint8_t b_bank(size_t k, size_t j) { return (k*4 + j + 4) % 8; }
};
```

### Phase 2: Pipelined Streamer Schedule

```cpp
// For computing all C[i,*] (one row of output tiles)
void schedule_c_row(size_t i) {
    // Preload A[i,*] tiles - these stay resident
    for (k = 0; k < 4; k++) {
        prefetch_to_l2(A[i,k]);
    }

    // Process C[i,j] for j = 0,1,2,3
    for (j = 0; j < 4; j++) {
        // Double-buffer B tiles
        if (j == 0) {
            load_b_tiles(j);  // B[*,0]
        }

        // Overlap: load next B tiles while computing current
        if (j < 3) {
            prefetch_b_tiles(j+1);  // B[*,j+1]
        }

        // Stream A[i,*] and B[*,j] to systolic array
        for (k = 0; k < 4; k++) {
            stream_tile_pair(A[i,k], B[k,j]);  // Pipelined, 16 cycles each
        }

        // Drain C[i,j] - overlapped with next tile's streaming
        drain_output(C[i,j]);
    }
}
```

### Phase 3: Instruction Sequence

```
; Compute C[0,0] through C[0,3]
; A[0,*] resident in L2 banks 0-3
; B tiles double-buffered in L2 banks 4-7

DMA_LOAD A[0,0], A[0,1], A[0,2], A[0,3]    ; Load A row to L3
DMA_LOAD B[0,0], B[1,0], B[2,0], B[3,0]    ; Load first B column set
BM_MOVE A[0,*] L3→L2                        ; Move A to L2 (stays resident)
BM_MOVE B[*,0] L3→L2                        ; Move B column 0 to L2

; Pipeline loop
.loop_j:
    ; Prefetch next B column (overlapped)
    DMA_PREFETCH B[*,j+1]                   ; If j < 3

    ; Stream k=0..3 to systolic array
    STR_FEED_ROWS A[0,0]                    ; 16 rows
    STR_FEED_COLS B[0,j]                    ; 16 cols (parallel)
    ; ... cycles pass, accumulation happens ...
    STR_FEED_ROWS A[0,1]                    ; Next k tile
    STR_FEED_COLS B[1,j]
    STR_FEED_ROWS A[0,2]
    STR_FEED_COLS B[2,j]
    STR_FEED_ROWS A[0,3]
    STR_FEED_COLS B[3,j]

    ; Drain result (overlapped with next iteration's prefetch)
    STR_DRAIN C[0,j]
    BM_MOVE B[*,j+1] L3→L2                  ; Swap B buffer

    j++
    if j < 4: goto .loop_j
```

---

## Bandwidth Summary

| Schedule | L2 Read/Tile | L2 Write/Tile | Total/Tile | Total 64×64 | Notes |
|----------|-------------|---------------|------------|-------------|-------|
| A (no partial C) | 8 KB | 1 KB | 9 KB | 144 KB | Requires 9KB L2 |
| A + row reuse | 5 KB avg | 1 KB | 6 KB | 96 KB | A resident, B streams |
| A + full reuse | 2 KB avg | 1 KB | 3 KB | 48 KB | A+B resident |
| B (partial C) | 9 KB | 2 KB | 11 KB | 176 KB | 22% overhead |

---

## Efficiency Targets

| Configuration | Cycles | Efficiency | Notes |
|--------------|--------|------------|-------|
| Single tile, no pipeline | 94 | 68% | Fill + drain overhead |
| 4 tiles, pipelined | 94 + 3×64 = 286 | 256/286 = 89% | Amortized overhead |
| 16 tiles, pipelined | 94 + 15×64 = 1054 | 1024/1054 = 97% | Near-optimal |
| 16 tiles, tight pipeline | ~1024 + 30 = 1054 | 97% | Fill once, drain once |

---

## Implementation Recommendations

1. **L2 capacity**: Ensure at least 17 KB available for double-buffered operation
2. **Bank assignment**: Spread A and B tiles across banks to avoid conflicts
3. **Streaming order**: Process output tiles in row-major order to maximize A reuse
4. **Double buffering**: Always prefetch next B column while computing current
5. **Edge FIFOs**: Implement 32-deep FIFOs at array edges for latency hiding
6. **Instruction emission**: Emit interleaved A/B stream instructions for pipelining
