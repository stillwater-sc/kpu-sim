# System-Level Schedule DSL

## The Problem: Bridging Algorithms to Hardware

A matrix multiplication `C = A × B` is a single line of mathematics. Executing it
on a KPU requires orchestrating dozens of concurrent operations across a deep memory
hierarchy:

```
DRAM  ──DMA──▶  L3 Buffers  ──BlockMover──▶  L2 Banks  ──Streamer──▶  L1 Streams  ──▶  Compute
                                                                                          │
DRAM  ◀──DMA──  L3 Buffers  ◀──BlockMover──  L2 Banks  ◀──Streamer──  L1 Drains  ◀──────┘
```

Each arrow is a separate hardware engine. Each engine waits for a credit from
downstream before it can push data. Each buffer has finite capacity. The tiles are
too large to fit in a single level. The tensors are too large to fit in all levels
combined. The data must be partitioned into tiles, and those tiles must flow through
the hierarchy in a precise order—one that maximizes reuse, overlaps data movement
with compute, and never deadlocks the credit network.

This orchestration is the **system-level schedule**. It is the program that the KPU
actually executes. The compute fabric has no program counter. It fires reactively
when data arrives. The schedule's sole job is to deliver the right data to the right
place at the right time.

Writing these schedules by hand as instruction sequences is error-prone. The
`OutputStationaryProgramBuilder` in the existing ISA layer generates correct matmul
schedules, but its logic is specific to one kernel shape and one dataflow strategy.
Adding convolution or softmax meant duplicating hundreds of lines of address
calculation, buffer assignment, and loop nest generation.

The Schedule DSL addresses this by providing a **declarative vocabulary** for
expressing system-level schedules. A schedule declares *what* data moves *where*
and *in what order*. The compiler handles the mechanical translation to
`DMInstruction` sequences.

---

## What a Schedule Describes

A schedule is not an algorithm. It does not specify how to multiply matrices or
compute exponentials. The compute fabric handles that reactively. A schedule
describes five things:

### 1. Tensors

What data exists, its shape, its location in external memory, and its element type.

```cpp
sched.tensor(Tensor{"A", MatrixID::A, {M, K}, 0x0000, DataType::FP32});
sched.tensor(Tensor{"B", MatrixID::B, {K, N}, 0x4000, DataType::FP32});
sched.tensor(Tensor{"C", MatrixID::C, {M, N}, 0x8000, DataType::FP32});
```

Tensors are the raw material. They reside in DRAM. The schedule's job is to carve
them into tiles and move those tiles through the hierarchy.

### 2. Tiling

How tensors are partitioned. A tile is the unit of data movement. A `256×256` matrix
with `Ti=64, Tj=64` becomes a 4×4 grid of tiles.

```cpp
sched.tile("ti", 64);   // M dimension: 64 rows per tile
sched.tile("tj", 64);   // N dimension: 64 columns per tile
sched.tile("tk", 32);   // K dimension: 32 columns of A / rows of B per tile
```

Tile sizes are constrained by buffer capacities. An L3 buffer must hold at least one
A tile and one B tile. An L2 bank must hold at least one tile for streaming. The DSL
does not enforce these constraints at declaration time—the compiler checks them
during compilation.

### 3. Loop Nesting

The order in which tiles are visited. This is the schedule's most consequential
decision. It determines:

- **Which tiles are reused** across iterations (and thus don't need re-fetching)
- **How long tiles occupy buffers** (and thus when credits return upstream)
- **How many tiles are in flight simultaneously** (and thus the pipeline depth)

```cpp
sched.for_tiles("ti")           // outer: M tile rows
    .for_tiles("tj")            // middle: N tile columns
        .for_tiles("tk")        // inner: K reduction tiles
            // ... data movement per tile ...
        .end()
        // ... drain and store per output tile ...
    .end()
.end();
```

The loop nest is expressed through a fluent API that returns `LoopScope` objects.
Each `for_tiles()` opens a new scope whose operations become the loop body. `end()`
closes the scope and returns to the parent. Operations placed between `end()` and
the next `end()` belong to the enclosing loop.

### 4. Data Movement

The operations that move tiles between memory levels. Each operation maps to a
specific hardware engine:

| Operation | Engine | Direction | Credit Required From |
|-----------|--------|-----------|---------------------|
| `load(A)` | DMA | DRAM → L3 | L3 buffer |
| `move(A)` | BlockMover | L3 → L2 | L2 bank |
| `stream_rows(A)` | Streamer | L2 → L1 west edge | L1 stream buffer |
| `stream_cols(B)` | Streamer | L2 → L1 north edge | L1 stream buffer |
| `drain()` | Streamer | Accumulator → L2 | L2 bank |
| `writeback(C)` | BlockMover | L2 → L3 | L3 buffer |
| `store(C)` | DMA | L3 → DRAM | DRAM (always available) |

Each operation is a **push with credit**. The DMA does not check whether the tile
is "already in L3." The BlockMover does not evict an old tile to make room. These
concepts do not exist. When the schedule says `load(A)`, it means: wait for an L3
credit, fetch the tile from DRAM, push it to the L3 buffer.

### 5. Synchronization

Points where the schedule must wait for preceding operations to complete before
proceeding.

```cpp
.load(MatrixID::A)
.load(MatrixID::B)
.barrier()           // wait for both DMA loads to complete
.move(MatrixID::A)   // now safe to move from L3 to L2
```

`barrier()` waits for all pending operations. In the credit-based model, barriers
ensure that tiles have arrived at a level before the next engine attempts to consume
them. Without barriers, a BlockMover might try to read an L3 buffer that the DMA
hasn't finished filling.

---

## The Three Prototypical Schedules

The DSL is validated against three kernel patterns that exercise different hardware
capabilities.

### MatMul: Output-Stationary

```
C[M,N] += A[M,K] × B[K,N]
```

This is the canonical schedule. C tiles accumulate in the PE registers across all K
iterations. Neither A nor B tiles are resident in the PEs—they stream through.

```
for ti in 0..M/Ti:
  for tj in 0..N/Tj:
    for tk in 0..K/Tk:
      load(A[ti,tk])          ← DMA: DRAM → L3
      load(B[tk,tj])          ← DMA: DRAM → L3
      barrier()               ← wait for DMA
      move(A[ti,tk])          ← BM: L3 → L2
      move(B[tk,tj])          ← BM: L3 → L2
      stream_rows(A)          ← STR: L2 → L1 west edge
      stream_cols(B)          ← STR: L2 → L1 north edge
      // PE: C += A × B       ← fires reactively on data arrival
    end
    drain(C[ti,tj])           ← STR: accumulator → L2
    writeback(C[ti,tj])       ← BM: L2 → L3
    store(C[ti,tj])           ← DMA: L3 → DRAM
  end
end
```

**Why output-stationary?** Because C never leaves the PE registers until all K tiles
have been accumulated. This avoids the cost of draining and reloading partial sums.
The price is that A and B tiles must stream through for every output tile, but their
total volume is `O(M·K + K·N)` versus `O(M·N·K)` for partial sum traffic in a
non-stationary schedule.

**Credit flow per inner iteration:**

```
DMA waits  BUFFER_AVAILABLE(L3)  → pushes tile → emits  TILE_READY(L3)
BM  waits  TILE_READY(L3)       → pushes tile → emits  TILE_READY(L2)
                                                        + BUFFER_AVAILABLE(L3)
STR waits  TILE_READY(L2)       → feeds L1    → emits  BUFFER_AVAILABLE(L2)
PE  fires when both A row and B column arrive at L1
```

### Conv2D: Im2Col + MatMul

```
out[N,Co,Oh,Ow] = in[N,Ci,H,W] ⊛ W[Co,Ci,Kh,Kw] + bias[Co]
```

Convolution is lowered to matrix multiplication via im2col. The input tensor is
gathered into a patch matrix where each row is one receptive field. The weight tensor
is reshaped into a filter matrix. The output is their product plus bias.

```
Lowering:
  A_col[N·Oh·Ow, Ci·Kh·Kw] = im2col(input)     ← patch matrix
  B_w  [Ci·Kh·Kw, Co]       = reshape(weights)   ← filter matrix
  C_out[N·Oh·Ow, Co]        = A_col × B_w + bias  ← output
```

The schedule follows the same output-stationary structure as matmul, with two key
differences:

1. **DMA gather for A tiles.** The input tensor is stored in NCHW format, but im2col
   needs receptive field patches. The DMA uses strided gather to extract non-contiguous
   elements from DRAM and pack them into a contiguous L3 tile. This is the
   `load_gather()` operation.

2. **Fused drain with Vector Engine.** After accumulation, the output tile passes
   through the Vector Engine during the drain phase. The VE adds the bias vector and
   applies ReLU activation in a single pass, without writing the unbiased result to
   L2 and re-reading it.

```
for ti in 0..M/Ti:
  for tj in 0..Co/Tj:
    for tk in 0..K/Tk:
      load_gather(A[ti,tk], im2col_params)  ← DMA: strided gather
      load(B[tk,tj])                         ← DMA: contiguous weight tile
      barrier()
      move(A), move(B)
      stream_rows(A), stream_cols(B)
    end
    drain_fused(C[ti,tj], bias, RELU)        ← STR+VE: drain with fusion
    writeback(C[ti,tj])
    store(C[ti,tj])
  end
end
```

### Softmax: Multi-Pass Vector Engine

```
y[b,i] = exp(x[b,i] - max_b) / Σ exp(x[b,i] - max_b)
```

Softmax is fundamentally different from matmul. It uses no systolic array. All
compute is done by the Vector Engine (VE), which performs elementwise and reduction
operations on data streaming through L1. The schedule makes three passes over the
data:

**Pass 1 — Find max per row:**
```
load(X[tb])                    ← DMA: batch chunk to L3
move(X[tb])                    ← BM: L3 → L2
stream_rows(X)                 ← STR: L2 → L1
compute_reduce(MAX)            ← VE: running max across D
drain_to_scratch(max_buf)      ← STR: max values → L2 scratch
```

**Pass 2 — Compute exp(x − max):**
```
broadcast(max_buf)             ← STR: broadcast max to all positions
stream_rows(X)                 ← STR: re-stream input from L2 (no DRAM re-fetch)
compute_elementwise(SUB)       ← VE: x - max
compute_elementwise(EXP)       ← VE: exp(x - max)
drain()                        ← STR: exp values → L2
```

**Pass 3 — Sum and divide:**
```
stream_rows(exp_buf)           ← STR: stream exp values
compute_reduce(SUM)            ← VE: running sum
drain_to_scratch(sum_buf)      ← STR: sum → L2 scratch
broadcast(sum_buf)             ← STR: broadcast sum
stream_rows(exp_buf)           ← STR: re-stream exp values
compute_elementwise(DIV)       ← VE: exp / sum
drain()                        ← STR: final output → L2
writeback(Y), store(Y)
```

**Key observations:**

- **No DRAM re-fetch.** The input and intermediate results stay in L2 across all
  three passes. L2 serves as scratch storage. Only one DMA load and one DMA store
  per batch chunk.

- **Reduction and broadcast.** `compute_reduce(MAX)` produces a scalar per row.
  `broadcast()` replicates that scalar across all D positions. These are VE
  operations, not systolic array operations.

- **Multiple passes over L2 data.** The Streamer re-reads L2 data for each pass.
  This is legal because the data is still in L2 banks—L2 is a buffer, and nothing
  evicts it between passes. The credit for the L2 bank is not returned until the
  final writeback.

---

## ISA Gaps Revealed by the Three Schedules

Comparing the three schedules reveals operations that the original ISA lacked.
The DSL implementation added these to `DMOpcode`:

| New Opcode | Used By | Purpose |
|------------|---------|---------|
| `DMA_LOAD_GATHER` | Conv2D | Strided gather for im2col |
| `DMA_STORE_SCATTER` | Conv2D | Strided scatter for NCHW output |
| `VE_ELEMENTWISE` | Softmax | SUB, EXP, DIV via Vector Engine |
| `VE_REDUCE` | Softmax | MAX, SUM reduction via Vector Engine |
| `L2_SCRATCH_WRITE` | Softmax | Write to L2 scratch region |
| `L2_SCRATCH_READ` | Softmax | Read from L2 scratch region |

These opcodes extend the ISA without modifying existing instruction semantics. The
original matmul schedule continues to use `DMA_LOAD_TILE`, `BM_MOVE_TILE`,
`STR_FEED_ROWS`, etc., unchanged.

---

## From Schedule to DMProgram

The `compile_schedule()` function walks the schedule's operation tree and emits
`DMInstruction` objects with concrete tile coordinates, buffer assignments, and
memory addresses.

### Compilation Process

1. **Loop unrolling.** Each `for_tiles("tk")` is expanded by iterating `tk` from 0
   to `dim/tile_size`. For every iteration, the compiler emits the loop body with
   the current tile indices.

2. **Address calculation.** For each `load(A)` at tile coordinate `(ti, tk)`, the
   compiler computes the external memory address: `A_base + (ti * Ti * K + tk * Tk) * element_size`.

3. **Buffer assignment.** The compiler alternates buffer slots for double-buffering.
   Odd K iterations use `BUF_0`, even iterations use `BUF_1`.

4. **Traffic estimation.** Each `load` increments the external memory byte counter.
   Each `move` increments the L2 byte counter. These accumulators produce the
   program's `estimates` structure.

5. **HALT insertion.** The compiler appends a `HALT` instruction after the last
   operation.

### What the Compiler Does Not Do

- **Scheduling.** The compiler emits instructions in program order. It does not
  reorder instructions for latency hiding. That is the executor's job.

- **Credit management.** The compiler does not track credit availability. The
  credit-based dataflow model is an execution concern, not a compilation concern.

- **Buffer capacity checking.** The compiler does not verify that tiles fit in
  buffers. That is a configuration concern handled during system setup.

- **Compute instruction emission.** In a dataflow architecture, compute fires
  reactively. The compiler emits annotated NOPs for compute operations (e.g.,
  `COMPUTE_MATMUL (reactive)`) as documentation markers, not executable instructions.

---

## The Dataflow Contract

Every schedule, regardless of kernel, obeys the same dataflow contract:

1. **Data flows downstream.** DRAM → L3 → L2 → L1 → Compute → L1 → L2 → L3 → DRAM.
   No level is skipped. No data flows upstream except credits.

2. **Each push requires a credit.** Before DMA can write to L3, it must have a
   `BUFFER_AVAILABLE(L3)` credit. Before BlockMover can write to L2, it must have
   a `BUFFER_AVAILABLE(L2)` credit. The schedule's `barrier()` operations ensure
   that credits have been consumed and tiles have arrived before the next stage
   attempts to push.

3. **Credits return upstream on consumption.** When the Streamer consumes a tile
   from L2, it returns a `BUFFER_AVAILABLE(L2)` credit to the BlockMover. This
   credit enables the BlockMover to push the next tile. The credit chain propagates
   all the way back to the DMA.

4. **Compute is reactive.** The schedule never says "compute now." It says "stream
   A rows and B columns." When both arrive at L1, the systolic array fires
   automatically. When the VE receives data and an opcode, it executes automatically.
   The schedule's job ends at L1.

5. **No buffer is a cache.** L3 is not a cache. L2 is not a cache. There is no
   hit/miss/eviction. The schedule explicitly controls what is in each buffer at
   each point in time. If a tile is in L2 and the schedule streams it twice (as
   softmax does), the tile is still there because nothing evicted it—it was
   explicitly placed and has not been explicitly replaced.

---

## Source Files

| File | Purpose |
|------|---------|
| `include/sw/kpu/dsl/schedule.hpp` | Schedule, LoopScope, Tensor, operation types |
| `include/sw/kpu/dsl/schedule_compiler.hpp` | `compile_schedule()` declaration |
| `src/dsl/schedule.cpp` | Schedule and LoopScope fluent builder |
| `src/dsl/schedule_compiler.cpp` | IR → DMProgram compilation |
| `include/sw/kpu/schedules/matmul_schedule.hpp` | `matmul_output_stationary()` |
| `include/sw/kpu/schedules/conv2d_schedule.hpp` | `conv2d_im2col()` |
| `include/sw/kpu/schedules/softmax_schedule.hpp` | `softmax()` |
| `src/schedules/{matmul,conv2d,softmax}_schedule.cpp` | Schedule construction |
| `tests/dsl/test_schedule_dsl.cpp` | 51 verification tests |
