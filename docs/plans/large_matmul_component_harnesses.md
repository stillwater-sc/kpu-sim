# Large MatMul Component Test Harnesses

## Goal

Create test harnesses for DMA, BlockMover, and Streamer components that can execute
and observe tile movement programs for a large matmul (4096×1024 @ 1024×8192) on
T256 hardware configuration.

**Key Insight**: The KPU implements a **two-level block matrix multiplier**:
- Level 1: Compute tile (16×16 systolic array) — element-level matmul
- Level 2: Checkerboard (16×16 compute tiles) — block-level matmul

Programs must be **compact and parametric** using loops with address generation,
not explicit enumeration of millions of operations.

## Problem Dimensions

```
A[4096, 1024] × B[1024, 8192] = C[4096, 8192]

M = 4096    (A rows, C rows)
K = 1024    (A cols, B rows)
N = 8192    (B cols, C cols)

FLOPs = 2 × M × N × K = 68,719,476,736 (68.7 GFLOP)
```

## Hardware Configuration: T256

| Component | Size | Capacity (floats) | Notes |
|-----------|------|-------------------|-------|
| Systolic Array | 16×16 | 256 PEs | T256 configuration |
| L3 Tile | 256 KB | 65,536 | Tile cache, 4 tiles |
| L2 Bank | 16 KB | 4,096 | Per-bank, 8 banks |
| L1 Buffer | 1 KB | 256 | Per side, 4 sides |

## Two-Level Block Matrix Structure

### Level 1: Compute Tile (Element MatMul)

```
16×16 systolic array
Each PE: c += a × b (scalar MAC)
Tile operation: C[16,16] += A[16,16] × B[16,16]
```

### Level 2: Checkerboard (Block MatMul)

```
16×16 grid of compute tiles
Each "element": 16×16 tile
Super-tile: 256×256 elements

Block operation: C_super += A_super × B_super
Where each block multiply triggers 16×16 = 256 tile matmuls
```

### Tiling the Large MatMul

```
Matrix A[4096, 1024]:
  - 256 row tiles (4096/16)
  - 64 col tiles (1024/16)
  - Total: 16,384 tiles

Matrix B[1024, 8192]:
  - 64 row tiles (1024/16)
  - 512 col tiles (8192/16)
  - Total: 32,768 tiles

Matrix C[4096, 8192]:
  - 256 row tiles
  - 512 col tiles
  - Total: 131,072 tiles

At checkerboard level (256×256 super-tiles):
  - A: 16×4 super-tiles
  - B: 4×32 super-tiles
  - C: 16×32 super-tiles

Block matmul: 16×32 output blocks, 4 reduction blocks each
```

---

# ISA Extensions: Loop Machinery and Address Generation

## Motivation

The current ISA has LOOP_BEGIN/LOOP_END opcodes but lacks:
1. Loop counter registers
2. Base address registers
3. Stride registers
4. Automatic address computation from loop indices

Without these, every tile operation requires an explicit instruction with
hardcoded addresses — impossible for large matmuls (millions of operations).

## Design Goals

1. **Compact programs**: Express millions of operations in ~50 instructions
2. **Parametric**: Same program works for different matrix sizes (with config)
3. **Hardware-friendly**: Maps to simple address generation logic
4. **Hierarchical**: Supports nested loops for block matrix algorithms

## Register File Design

### Loop Counter Registers (8 registers)

```
LC[0..7]: 16-bit loop counters

Each loop counter has:
  - count:    Current iteration (counts down from limit-1 to 0)
  - limit:    Total iterations
  - stride:   Tile index increment per iteration (usually 1)
```

**Usage:**
```asm
LOOP_BEGIN 0, 256, 1    ; LC[0].limit=256, LC[0].stride=1, LC[0].count=255
  ; ... loop body ...
LOOP_END 0              ; LC[0].count--, jump if count >= 0
```

### Base Address Registers (6 registers)

```
BA[A], BA[B], BA[C]:     External memory base addresses (64-bit)
L3A[A], L3A[B], L3A[C]:  L3 tile base addresses (32-bit)
L2A[A], L2A[B], L2A[C]:  L2 bank base addresses (32-bit)
```

### Stride Registers (per matrix)

```
For matrix X ∈ {A, B, C}:
  STRIDE_X_ROW:    Bytes between consecutive rows in external memory
  STRIDE_X_COL:    Bytes between consecutive columns (element size)
  STRIDE_X_TILE_I: Bytes between consecutive row tiles
  STRIDE_X_TILE_J: Bytes between consecutive column tiles
```

For row-major layout:
```
A[M, K]:  STRIDE_A_ROW = K × elem_size
          STRIDE_A_TILE_I = Ti × K × elem_size
          STRIDE_A_TILE_J = Tj × elem_size

B[K, N]:  STRIDE_B_ROW = N × elem_size
          STRIDE_B_TILE_I = Ti × N × elem_size
          STRIDE_B_TILE_J = Tj × elem_size

C[M, N]:  STRIDE_C_ROW = N × elem_size
          STRIDE_C_TILE_I = Ti × N × elem_size
          STRIDE_C_TILE_J = Tj × elem_size
```

### Tile Dimension Registers

```
TILE_I:   Tile height (typically 16)
TILE_J:   Tile width (typically 16)
TILE_K:   Reduction tile depth (typically 16)
ELEM_SIZE: Element size in bytes (4 for float32)
```

## New/Modified Opcodes

### Configuration Opcodes

```cpp
// Set external memory base address for matrix
SET_BASE,           // SET_BASE matrix, addr64

// Set L3 base offset for matrix
SET_L3_BASE,        // SET_L3_BASE matrix, offset32

// Set L2 base offset for matrix
SET_L2_BASE,        // SET_L2_BASE matrix, offset32

// Set stride configuration
SET_STRIDE,         // SET_STRIDE matrix, row_stride, tile_i_stride, tile_j_stride

// Set tile dimensions
SET_TILE_DIM,       // SET_TILE_DIM Ti, Tj, Tk

// Set matrix dimensions (for address computation)
SET_MATRIX_DIM,     // SET_MATRIX_DIM matrix, rows, cols
```

### Loop Opcodes (Enhanced)

```cpp
// Begin hardware loop
// loop_id: 0-7 (which loop counter)
// limit: iteration count
// index_role: which tile index this loop drives
LOOP_BEGIN,         // LOOP_BEGIN loop_id, limit, index_role

// End hardware loop (decrement counter, branch if not zero)
LOOP_END,           // LOOP_END loop_id

// Nested loop example:
//   LOOP_BEGIN 0, 256, TI    ; Loop 0 drives ti (output rows)
//   LOOP_BEGIN 1, 512, TJ    ; Loop 1 drives tj (output cols)
//   LOOP_BEGIN 2, 64, TK     ; Loop 2 drives tk (reduction)
```

### Index Role Enum

```cpp
enum class IndexRole : uint8_t {
    TI = 0,     // Output row tile index (M dimension)
    TJ = 1,     // Output col tile index (N dimension)
    TK = 2,     // Reduction tile index (K dimension)
    NONE = 3,   // Generic loop, not tied to tile index
};
```

### AUTO Addressing Mode

```cpp
// DMA with automatic address computation
DMA_LOAD_TILE_AUTO,   // DMA_LOAD_TILE_AUTO matrix, l3_slot, buffer
DMA_STORE_TILE_AUTO,  // DMA_STORE_TILE_AUTO matrix, l3_slot, buffer

// BlockMover with automatic addressing
BM_MOVE_TILE_AUTO,    // BM_MOVE_TILE_AUTO matrix, l2_bank, buffer
BM_WRITEBACK_AUTO,    // BM_WRITEBACK_AUTO matrix, l3_slot, buffer

// Streamer with automatic addressing
STR_FEED_ROWS_AUTO,   // STR_FEED_ROWS_AUTO l1_buffer, buffer
STR_FEED_COLS_AUTO,   // STR_FEED_COLS_AUTO l1_buffer, buffer
STR_DRAIN_AUTO,       // STR_DRAIN_AUTO l2_bank, buffer
```

## Address Computation

### External Memory Address (for DMA)

```
For matrix A (indexed by ti, tk):
  addr = BA[A] + ti × STRIDE_A_TILE_I + tk × STRIDE_A_TILE_J

For matrix B (indexed by tk, tj):
  addr = BA[B] + tk × STRIDE_B_TILE_I + tj × STRIDE_B_TILE_J

For matrix C (indexed by ti, tj):
  addr = BA[C] + ti × STRIDE_C_TILE_I + tj × STRIDE_C_TILE_J
```

### Loop Index Binding

```
When LOOP_BEGIN specifies index_role:
  - TI: Current loop counter value is used as 'ti'
  - TJ: Current loop counter value is used as 'tj'
  - TK: Current loop counter value is used as 'tk'

The innermost loop for each index role determines its value.
```

### Example: Address Computation for A[ti, tk]

```
Given:
  BA[A] = 0x80000000           ; A base in external memory
  M = 4096, K = 1024           ; Matrix dimensions
  Ti = 16, Tk = 16             ; Tile dimensions
  elem_size = 4                ; float32

Computed strides:
  STRIDE_A_TILE_I = Ti × K × 4 = 16 × 1024 × 4 = 65536
  STRIDE_A_TILE_J = Tk × 4 = 16 × 4 = 64

For tile A[ti=3, tk=7]:
  addr = 0x80000000 + 3 × 65536 + 7 × 64
       = 0x80000000 + 196608 + 448
       = 0x80030000 + 0x1C0
       = 0x800301C0
```

## Compact Program Example

### Output-Stationary MatMul (4096×1024 @ 1024×8192)

```asm
; ============================================================
; Configuration
; ============================================================
.name "matmul_4096x1024x8192"
.version 2                      ; Version 2 = loop-capable

; Matrix dimensions
SET_MATRIX_DIM A, 4096, 1024
SET_MATRIX_DIM B, 1024, 8192
SET_MATRIX_DIM C, 4096, 8192

; Tile dimensions
SET_TILE_DIM 16, 16, 16

; External memory bases (set by loader)
SET_BASE A, 0x00000000
SET_BASE B, 0x01000000          ; 4096×1024×4 = 16MB offset
SET_BASE C, 0x03000000          ; 16MB + 1024×8192×4 = 16MB + 32MB

; Strides (computed from dimensions)
SET_STRIDE A, 4096, 65536, 64   ; row=K×4, tile_i=Ti×K×4, tile_j=Tk×4
SET_STRIDE B, 32768, 32768, 64  ; row=N×4, tile_i=Tk×N×4, tile_j=Tj×4
SET_STRIDE C, 32768, 524288, 64 ; row=N×4, tile_i=Ti×N×4, tile_j=Tj×4

; ============================================================
; Main Loop Nest (Output-Stationary)
; ============================================================

LOOP_BEGIN 0, 256, TI           ; ti = 0..255 (M/Ti = 4096/16)
  LOOP_BEGIN 1, 512, TJ         ; tj = 0..511 (N/Tj = 8192/16)

    ; C[ti,tj] accumulates in PE registers across K
    LOOP_BEGIN 2, 64, TK        ; tk = 0..63 (K/Tk = 1024/16)

      ; DMA: Load A[ti,tk] and B[tk,tj] to L3
      DMA_LOAD_TILE_AUTO A, 0, BUF_0
      DMA_LOAD_TILE_AUTO B, 1, BUF_0
      BARRIER

      ; BM: Move tiles L3 → L2
      BM_MOVE_TILE_AUTO A, 0, BUF_0
      BM_MOVE_TILE_AUTO B, 1, BUF_0
      BARRIER

      ; STR: Feed tiles to systolic array
      STR_FEED_ROWS_AUTO 0, BUF_0
      STR_FEED_COLS_AUTO 0, BUF_0
      ; Compute fires reactively
      BARRIER

    LOOP_END 2                  ; End tk loop

    ; Drain accumulated C[ti,tj]
    STR_DRAIN_AUTO 2, BUF_0
    BARRIER

    ; Writeback and store
    BM_WRITEBACK_AUTO C, 2, BUF_0
    BARRIER
    DMA_STORE_TILE_AUTO C, 2, BUF_0
    BARRIER

  LOOP_END 1                    ; End tj loop
LOOP_END 0                      ; End ti loop

HALT
```

**Program size: ~35 instructions** (not 8.4 million!)

## Instruction Encoding

### LOOP_BEGIN Encoding

```
| Opcode (8) | Loop ID (4) | Index Role (4) | Limit (16) |
```

### SET_STRIDE Encoding

```
| Opcode (8) | Matrix (4) | Reserved (4) |
| Row Stride (32) |
| Tile I Stride (32) |
| Tile J Stride (32) |
```

### DMA_LOAD_TILE_AUTO Encoding

```
| Opcode (8) | Matrix (4) | L3 Slot (4) | Buffer (4) | Reserved (4) |
```

Address computed at execution time from:
- BA[matrix]
- Current ti, tj, tk from loop counters
- STRIDE registers

## Hardware Implementation

### Loop Controller

```
For each loop level (0-7):
  - 16-bit counter register
  - 16-bit limit register
  - 2-bit index role
  - Compare logic (counter vs 0)
  - Branch target (instruction after LOOP_BEGIN)
```

### Address Generator

```
Inputs:
  - Base address (64-bit)
  - Loop indices ti, tj, tk (16-bit each)
  - Strides (32-bit each)

Output:
  - Computed address (64-bit)

Logic:
  addr = base + ti × stride_i + tj × stride_j + tk × stride_k

Hardware: 3 multipliers + 3 adders (can be pipelined)
```

### Index Extraction

```
For each index role (TI, TJ, TK):
  - Scan active loops from innermost to outermost
  - First loop with matching role provides the index value
```

---

# Updated Implementation Roadmap

## Phase 1: ISA Extensions (Week 1)

| Task | Files | Description |
|------|-------|-------------|
| 1.1 | `include/sw/kpu/isa/data_movement_isa.hpp` | Add new opcodes, IndexRole enum |
| 1.2 | `include/sw/kpu/isa/loop_state.hpp` | LoopState class with counters |
| 1.3 | `include/sw/kpu/isa/address_generator.hpp` | AddressGenerator class |
| 1.4 | `include/sw/kpu/isa/register_file.hpp` | Base/stride registers |
| 1.5 | `docs/isa/loop_address_generation.md` | Specification document |

## Phase 2: Assembler Updates (Week 2)

| Task | Files | Description |
|------|-------|-------------|
| 2.1 | `src/software/isa/assembler.cpp` | Parse new opcodes |
| 2.2 | `src/software/isa/assembler.cpp` | Parse SET_* directives |
| 2.3 | `src/software/isa/assembler.cpp` | Parse *_AUTO instructions |
| 2.4 | `tests/isa/test_assembler_loops.cpp` | Test loop assembly |

## Phase 3: Executor Updates (Week 3)

| Task | Files | Description |
|------|-------|-------------|
| 3.1 | `src/software/isa/behavioral_program_executor.cpp` | Loop execution |
| 3.2 | `src/software/isa/behavioral_program_executor.cpp` | Address generation |
| 3.3 | `src/software/isa/transactional_program_executor.cpp` | Loop timing |
| 3.4 | `tests/isa/test_loop_execution.cpp` | Loop execution tests |

## Phase 4: Component Harnesses (Week 4-5)

| Task | Files | Description |
|------|-------|-------------|
| 4.1 | `include/sw/kpu/harness/dma_harness.hpp` | DMA harness with loops |
| 4.2 | `include/sw/kpu/harness/block_mover_harness.hpp` | BM harness with loops |
| 4.3 | `include/sw/kpu/harness/streamer_harness.hpp` | STR harness with loops |
| 4.4 | `tests/harness/test_*_harness.cpp` | Component tests |

## Phase 5: Large MatMul Validation (Week 6)

| Task | Files | Description |
|------|-------|-------------|
| 5.1 | `kernels/asm/matmul_4096x1024x8192.kpuasm` | Compact loop program |
| 5.2 | `tests/harness/test_large_matmul.cpp` | End-to-end validation |
| 5.3 | Trace generation and visualization | Observe tile movements |

---

# Data Movement Analysis (Revised)

With proper loop machinery, the program is ~35 instructions but generates:

| Component | Operations | Unique Tiles | Traffic (with caching) |
|-----------|------------|--------------|------------------------|
| DMA Load A | 8.4M | 16,384 | 16 MB |
| DMA Load B | 8.4M | 32,768 | 32 MB |
| DMA Store C | 131K | 131,072 | 128 MB |
| BM Move A | 8.4M | — | 8 GB |
| BM Move B | 8.4M | — | 8 GB |
| BM Writeback C | 131K | — | 128 MB |
| STR Feed A | 8.4M | — | 8 GB |
| STR Feed B | 8.4M | — | 8 GB |
| STR Drain C | 131K | — | 128 MB |

**Key**: L3 caching converts 8.4M DMA loads into 49K unique tile loads.

---

# Multi-Level Block MatMul Hierarchy

```
Level 3 (Future): Multi-SoC
  - Block matmul of KPU chips
  - Each "block" = full KPU computation
  - Inter-chip NoC or PCIe

Level 2: KPU SoC (Checkerboard)
  - 16×16 grid of compute tiles
  - Block matmul where each element = 256×256 super-tile
  - L3 ↔ L2 data movement

Level 1: Compute Tile
  - 16×16 systolic array
  - Element matmul: C += A × B
  - L1 streaming

Level 0: PE
  - Single MAC unit
  - c += a × b
```

The loop machinery enables expressing computation at any level of this hierarchy.
