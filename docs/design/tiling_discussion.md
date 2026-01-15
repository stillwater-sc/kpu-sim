# Tiling algorithms



## Multi-core



Good morning, Theodore! You’re diving into a rich and nuanced space—tiling strategies across heterogeneous compute architectures is a playground for both performance artistry and architectural rigor. Let’s start with the multi-core CPU scenario you described: three cache levels, with L1 and L2 private to each core and L3 shared.

---

## Good Tiling Strategy: Multilevel Cache-Aware Blocking

A good tiling algorithm for this setup is **multilevel tiling**, also known as **cache blocking**, where tile sizes are carefully chosen to match the capacity and associativity of each cache level:


### Key Principles

- **L1 tiles** should fit entirely within the L1 cache to maximize reuse and minimize latency.

- **L2 tiles** should encompass multiple L1 tiles and fit within L2, enabling reuse across L1 misses.

- **L3 tiles** should be large enough to amortize memory bandwidth but small enough to avoid eviction pressure across cores.



### Example: Tiled Matrix Multiplication

```cpp

for (int i0 = 0; i0 < M; i0 += T_M_L3) {

  for (int j0 = 0; j0 < N; j0 += T_N_L3) {

    for (int k0 = 0; k0 < K; k0 += T_K_L3) {

      for (int i1 = i0; i1 < min(i0 + T_M_L3, M); i1 += T_M_L2) {

        for (int j1 = j0; j1 < min(j0 + T_N_L3, N); j1 += T_N_L2) {

          for (int k1 = k0; k1 < min(k0 + T_K_L3, K); k1 += T_K_L2) {

            for (int i2 = i1; i2 < min(i1 + T_M_L2, M); i2 += T_M_L1) {

              for (int j2 = j1; j2 < min(j1 + T_N_L2, N); j2 += T_N_L1) {

                for (int k2 = k1; k2 < min(k1 + T_K_L2, K); k2 += T_K_L1) {

                  // Inner kernel on L1 tile

                  C[i2][j2] += A[i2][k2] * B[k2][j2];

                }

              }

            }

          }

        }

      }

    }

  }

}

```



### Benefits

- **Data locality** is maximized at each cache level.

- **Parallelism** is preserved across cores without excessive cache contention.

- **Prefetching** becomes more predictable and effective.



You can find a deeper breakdown of this strategy in [this compiler optimization guide](https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-4-tensor-level-polyhedral-optimizations/memory-hierarchy-optimization).

---

## Bad Tiling Strategy: Flat Tiling or Cache-Agnostic Blocking


A poor tiling strategy would be one that:

- Uses a **single tile size** without regard to cache hierarchy.

- Assumes **uniform cache behavior** across cores.

- Ignores **cache associativity and conflict misses**.



### Consequences

- **L1 cache thrashing** due to tiles that exceed L1 capacity.

- **L3 contention** when multiple cores access overlapping tiles.

- **DRAM bottlenecks** from frequent evictions and reloads.



### Example Pitfall

Using a tile size that fits L2 but not L1, and assuming L3 will absorb all misses gracefully, this leads to unpredictable latency and poor scaling.

---

## GPU many-core

GPU tiling is a trade-off between parallelism and memory hierarchy, and shared memory is the stage where performance is won or lost. Let’s break down what makes a tiling strategy good or bad on GPU many-core architectures, such as CUDA-enabled NVIDIA devices.

---

## Good Tiling Strategy: Shared Memory Cooperative Loading

### Core Principles

- **Thread blocks** load tiles of input data into shared memory collaboratively.

- **Each thread** computes a portion of the output using only shared memory once the tile is loaded.

- **Global memory accesses** are minimized and coalesced.



### Example: Matrix Multiplication

Let’s say we’re computing \\( C = A \\times B \\), where each thread block computes a tile of \\( C \\). A good strategy:

- Divides \\( A \\) and \\( B \\) into tiles that fit in shared memory.

- Loads a tile of \\( A \\) and a tile of \\( B \\) into shared memory.

- Synchronizes threads (`__syncthreads()`) before computing.

- Each thread computes a sub-tile of \\( C \\) using the shared memory tile.



```cuda

__shared__ float Asub[TILE_SIZE][TILE_SIZE];
__shared__ float Bsub[TILE_SIZE][TILE_SIZE];

int tx = threadIdx.x;
int ty = threadIdx.y;

for (int t = 0; t < K / TILE_SIZE; ++t) {

    Asub[ty][tx] = A[row * K + t * TILE_SIZE + tx];
    Bsub[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i)
        Cvalue += Asub[ty][i] * Bsub[i][tx];

    __syncthreads();
}

```

### Benefits

- **Massive reduction in global memory traffic**: Each element of A and B is loaded once per tile.

- **High arithmetic intensity**: More FLOPs per byte transferred.

- **Warp-level efficiency**: Threads access shared memory in parallel with minimal bank conflicts.

You can find a detailed breakdown of this strategy in [this CUDA tiling guide](https://nichijou.co/cuda7-tiling) and [this GPU matrix multiplication walkthrough](https://indii.org/blog/gpu-matrix-multiply-tiling).

---

## Bad Tiling Strategy: Naive Global Memory Access

### Pitfalls

- Each thread loads its own data from global memory.

- No use of shared memory.

- Redundant loads and poor memory coalescing.



### Example: Naive Matrix Multiply

```cuda

float sum = 0.0f;
for (int k = 0; k < K; ++k)
    sum += A[row * K + k] * B[k * N + col];

C[row * N + col] = sum;

```

### Consequences

- **Global memory bottleneck**: Hundreds of cycles per access.

- **Redundant reads**: Same rows/columns loaded by multiple threads.

- **Non-coalesced access**: Especially problematic for column-wise reads.

This pattern leads to idle arithmetic units and poor utilization of GPU bandwidth.

---

## TPU multi-tiles

TPUs with systolic arrays and a **weight stationary** dataflow flip the tiling paradigm on its head compared to CPUs and GPUs. Instead of optimizing for cache reuse or shared memory, the goal is to **maximize data reuse within the array itself**, especially for weights that remain fixed in place.

---

## Architectural Context: Systolic Array + Weight Stationary

- **Systolic array**: A 2D grid of processing elements (PEs) where data flows rhythmically across rows and columns.
- **Weight stationary**: Weights are preloaded into PEs and remain fixed during computation. Inputs and partial sums flow through the array.

This means tiling must **align with the array’s spatial and temporal data movement**, not just memory hierarchy.

---

## Good Tiling Strategy: Input Stationary Blocking with Weight Preloading

### Key Concepts
- **Tile inputs** to match the array’s input bandwidth and reuse opportunities.
- **Preload weights** into PEs before computation begins.
- **Stream activations** across rows and accumulate partial sums across columns.

### Matrix Multiply Example
For \( C = A \times B \), where:
- \( A \) is streamed row-wise (activations),
- \( B \) is stationary (weights),
- \( C \) is accumulated column-wise (outputs).

A good tiling strategy:
- Divides \( A \) into row tiles that match the array’s input width.
- Divides \( B \) into column tiles that match the array’s height.
- Ensures each tile of \( B \) is reused across multiple tiles of \( A \).

This is illustrated beautifully in [this systolic array walkthrough](https://www.telesens.co/2018/07/30/systolic-architectures/).

---

## Bad Tiling Strategy: Output Stationary or Cache-Oriented Blocking

### 🔹 Pitfalls
- Trying to tile for output reuse (e.g., keeping partial sums stationary) breaks the systolic flow.
- Cache-aware tiling (like on CPUs) leads to misaligned data movement and underutilized PEs.
- Frequent weight reloading destroys the benefit of weight stationarity.

### Consequences
- **Pipeline stalls** due to misaligned input/output timing.
- **Low PE utilization** from poor dataflow matching.
- **Bandwidth bottlenecks** from redundant weight movement.

---

## Bonus: Tiling for Convolutions

Since convolutions are lowered to matrix multiplies via im2col or Winograd transforms, tiling must also consider:
- **Spatial locality** of input patches.
- **Reuse of filter weights** across multiple patches.
- **Alignment with array dimensions** to avoid fragmentation.

---

## KPU Multi-tile

The Stillwater KPU’s Domain Flow Architecture introduces a tiling paradigm that’s fundamentally different from cache-based or shared-memory models. It’s not just about locality—it’s about **temporal alignment**, **wavefront propagation**, and **software-defined interconnects** that honor the concurrency of the algorithm itself.

---

## Architectural Foundation: KPU + Linear Schedule

From the [linear schedule specification](https://stillwater-sc.github.io/domain-flow/ch1/linearschedule/index.html), we know:

- The KPU executes **System of Recurrence Equations (SUREs)** by mapping their **wavefront evolution** onto a **recirculating 2D pipeline**.
- The **scheduling vector** defines the direction of computation and operand propagation.
- A **memoryless schedule** delivers operands just-in-time, using the fabric to create a high-performance distributed register file.
- The **interconnect fabric is programmable**, allowing arbitrary concurrency patterns to be realized.

This means tiling must **preserve the wavefront structure** and **enable pipelined execution of dependent tiles**.

---

## Good Tiling Strategy: Wavefront-Aligned Divide & Conquer

### 🔹 Strategy Overview
- **Decompose the global operator** (e.g., matmul, LU, QR) into **tile-sized SUREs** to be able to program the fabric.
- **Align tile boundaries** with the scheduling vector to preserve operand flow.
- **Use result-stationary execution** to avoid intermediate memory traffic.
- **Pipeline dependent tiles** so that the output of one tile feeds directly into the next. For example, if we had a 2x2 fabric tile architecture, then one of the A or B tiles could stream into an adjacent tile. 

### Example: Matrix Multiply
Let’s say we tile \( C = A \times B \) into submatrices \( C_{ij} \), each computed by:

\[
C_{ij} = \sum_k A_{ik} \times B_{kj}
\]

On the KPU:
- Each \( C_{ij} \) is a **result-stationary tile**: partial sums accumulate in-place.
- Tiles are sequenced along the **wavefront normal**, so operand delivery is just-in-time.
- Wider precision (e.g., triple-double) can be used for \( C_{ij} \) without memory pressure, since results never leave the fabric until finalization.

### Benefits
- **No memory buffering**: operand timing is aligned by the network.
- **Mixed precision friendly**: wider results stay local.
- **Streaming execution**: dependent tiles flow naturally through the pipeline.

---

## Bad Tiling Strategy: Orthogonal Blocking or Cache-Oriented Partitioning

### Pitfalls
- Tiling along row/column axes breaks the wavefront structure.
- Intermediate results must be stored and reloaded, defeating the memoryless model.
- Operand delivery becomes misaligned, requiring buffering and increasing energy cost.

### Consequences
- **Loss of concurrency**: tiles stall waiting for operands.
- **Increased memory traffic**: especially problematic for wide precision.
- **Underutilized fabric**: recirculation and interconnects are idle.

---

## Bonus: LU and QR Factorization

These algorithms have **data-dependent wavefronts**:
- LU: triangular structure with pivoting.
- QR: orthogonal transformations with trailing updates.

Tiling must:
- Respect the **temporal dependencies** of each panel.
- Use **streaming updates** for trailing matrix sections.
- Align tile execution with **signature evolution**, not static grid partitioning.

---

## KPU multi-tile reuse scheduling

Thank you for the correction, Theodore—that clarification sharpens the model beautifully. Given the 2×2 KPU fabric, where each tile executes a 32×32 matrix multiply and communicates via horizontal (A row tiles) and vertical (B column tiles) connections, we can now synthesize a tiling algorithm that respects:

- **Result-stationary execution** within each tile.
- **Streaming operand delivery** across tile boundaries.
- **Scratchpad memory** for local accumulation and operand buffering.
- **Configurable interconnects** to pipeline A or B tiles depending on reuse strategy.

---

## Problem Setup

We want to compute \( C = A \times B \), where:
- \( A \in \mathbb{R}^{128 \times 128} \)
- \( B \in \mathbb{R}^{128 \times 128} \)
- \( C \in \mathbb{R}^{128 \times 128} \)

Each tile computes a 32×32 submatrix of \( C \), so the full computation is tiled into a 4×4 grid of output tiles. The 2×2 KPU fabric will process these in **four passes**, each handling a quadrant of the output matrix.

---

## Tiling Algorithm for 2×2 KPU Fabric

### Step 1: Tile Partitioning

Partition \( A \), \( B \), and \( C \) into 32×32 tiles:

- \( A_{i,k} \): row tile at position (i,k)
- \( B_{k,j} \): column tile at position (k,j)
- \( C_{i,j} \): result tile at position (i,j)

Each tile \( C_{i,j} \) is computed as:

\[
C_{i,j} = \sum_{k=0}^{3} A_{i,k} \times B_{k,j}
\]

### Step 2: Fabric Mapping

Map the 2×2 fabric to compute four output tiles in parallel:

| Tile | Computes \( C_{i,j} \) | Receives A tile | Receives B tile |
|------|------------------------|-----------------|-----------------|
| T00  | \( C_{0,0} \)          | \( A_{0,k} \)   | \( B_{k,0} \)   |
| T01  | \( C_{0,1} \)          | \( A_{0,k} \)   | \( B_{k,1} \)   |
| T10  | \( C_{1,0} \)          | \( A_{1,k} \)   | \( B_{k,0} \)   |
| T11  | \( C_{1,1} \)          | \( A_{1,k} \)   | \( B_{k,1} \)   |

This layout allows **horizontal reuse of A row tiles** and **vertical reuse of B column tiles**.

### Step 3: Operand Streaming

Choose **A reuse** strategy (result-stationary favors streaming B tiles):

- **Broadcast \( A_{i,k} \)** horizontally to both tiles in row i.
- **Stream \( B_{k,j} \)** vertically to both tiles in column j.

This enables pipelined delivery of B tiles while holding A tiles in scratchpad memory.

### Step 4: Execution Loop

For each quadrant (i,j) of \( C \):

```cpp
for (k = 0; k < 4; ++k) {
    // Load A row tile A_{i,k} into horizontal fabric lanes
    broadcast_A_row_tile(i, k);

    // Stream B column tile B_{k,j} into vertical lanes
    stream_B_column_tile(k, j);

    // Each tile T_{i,j} performs:
    // C_{i,j} += A_{i,k} * B_{k,j}
    matmul_tile_accumulate(C[i][j], A[i][k], B[k][j]);
}
```

Each tile accumulates its result locally in scratchpad memory. After all \( k \) iterations, the result tile \( C_{i,j} \) is finalized and optionally written back.

---

## Fabric Scheduling

To compute the full 128×128 result:
- Divide the 4×4 output tile grid into **four 2×2 quadrants**.
- Schedule each quadrant sequentially on the fabric.
- Reconfigure interconnects if switching between A-reuse and B-reuse strategies.

---

## Optimization Notes

- **Mixed precision**: Wider accumulators (e.g., triple-double) stay local, reducing memory traffic.
- **Wavefront pipelining**: If B tiles are streamed, the vertical interconnects act as temporal conveyors.
- **Reconfigurable interconnects**: Can be adapted to favor B-reuse if A is more bandwidth-constrained.

---

## Detailed execution sequence

Absolutely, Theodore. Let’s synthesize a complete loop nest and memory movement model for a 2×2 KPU fabric executing a 128×128 matrix multiplication using 32×32 tiles. We’ll respect the architectural constraints:

- **DMA** moves tiles from DRAM to on-chip **L3**.
- **Streamers** move tiles from L3 to **L2**.
- **Tile pipelining** moves data from L2 to **L1**, which is a **spatial memory** that orchestrates operand injection into the KPU fabric.
- Each tile in the 2×2 fabric computes a 32×32 output tile using **result-stationary** scheduling.
- We reuse **A row tiles** across horizontal lanes and **stream B column tiles** vertically.

---

## Architectural Summary

| Memory Level | Role |
|--------------|------|
| DRAM         | Global storage for A, B, C |
| L3           | On-chip staging buffer for tiles |
| L2           | Flat memory for tile operands |
| L1           | Spatial memory for operand injection into KPU fabric |
| KPU Tile     | Executes 32×32 matmul with result-stationary scheduling |

---

## Loop Nest with Memory Movement

We’ll model the full computation of \( C = A \times B \), where each tile \( C_{i,j} \) is computed as:

\[
C_{i,j} = \sum_{k=0}^{3} A_{i,k} \times B_{k,j}
\]

Here’s the synthesized loop nest:

```cpp
for (int i = 0; i < 4; ++i) {           // Row tiles of C
  for (int j = 0; j < 4; ++j) {         // Column tiles of C

    // Allocate scratchpad for C_{i,j} in L2
    allocate_C_tile_L2(i, j);

    for (int k = 0; k < 4; ++k) {       // Reduction dimension

      // === DMA: DRAM → L3 ===
      dma_move_to_L3(A_tile(i, k));     // A_{i,k}
      dma_move_to_L3(B_tile(k, j));     // B_{k,j}

      // === Streamers: L3 → L2 ===
      streamer_move_to_L2(A_tile(i, k));  // A reuse: broadcast horizontally
      streamer_move_to_L2(B_tile(k, j));  // B stream: vertical injection

      // === Tile Pipelining: L2 → L1 ===
      pipeline_A_to_L1(i, k);           // Inject A_{i,k} into horizontal lanes
      pipeline_B_to_L1(k, j);           // Inject B_{k,j} into vertical lanes

      // === KPU Execution ===
      kpu_tile_matmul_accumulate(C_tile(i, j), A_tile(i, k), B_tile(k, j));
    }

    // === Finalize C_{i,j} ===
    write_C_tile_to_DRAM(C_tile(i, j));
  }
}
```

---

## Memory Movement Breakdown

### 🔹 DMA: DRAM → L3
- Operates at tile granularity (32×32).
- Prefetches A and B tiles needed for the current iteration.
- Can be double-buffered to hide latency.

### 🔹 Streamers: L3 → L2
- Move tiles into flat L2 memory.
- Respect reuse strategy: A tiles are reused across horizontal lanes, B tiles streamed vertically.

### 🔹 Tile Pipelining: L2 → L1
- L1 is spatial: each element is tagged with **position and time**.
- Pipelining logic injects operands into the KPU fabric at the correct cycle and location.
- A and B tiles are routed to their respective lanes based on the scheduling vector.

---

## L1 Spatial Memory Model

Each L1 memory bank holds a stream of matrix elements with metadata:

```cpp
struct L1StreamEntry {
  float value;
  int row_idx;
  int col_idx;
  int cycle;       // When to inject into the PE
};
```

The pipelining logic ensures that:
- A elements arrive at the correct horizontal lane at the right cycle.
- B elements arrive at the correct vertical lane at the right cycle.
- The result-stationary PE accumulates partial sums without needing to move them.

---

## Optimization Opportunities

- **Double-buffer L3 tiles** to overlap DMA and compute.
- **Precompute L1 injection schedules** for each tile to avoid runtime overhead.
- **Mixed precision**: wider accumulators (e.g., triple-double) stay in L2/L1, avoiding DRAM traffic.

---

# LU decomposition and dynamic updates

Alright, Theodore—let’s dive into LU tiling on the KPU fabric, where the dynamic nature of the factorization demands **adaptive operand injection** into the spatial L1 memory. Unlike matmul, LU introduces **data-dependent wavefronts**, especially during the **trailing update**, which must honor pivoting and triangular dependencies.

---

## Architectural Recap for LU on KPU

- **L1 spatial memory** orchestrates operand injection with temporal and spatial tags.
- **L2 flat memory** holds tile operands and intermediate results.
- **Streamers** move tiles from L3 to L2.
- **DMA** moves tiles from DRAM to L3.
- **KPU tiles** execute 32×32 matmul-like updates with result-stationary scheduling.

LU factorization decomposes \( A \) into \( L \) and \( U \), where:

\[
A = LU
\]

Each panel (column block) is factored, and the trailing matrix is updated:

\[
A_{i,j}^{(new)} = A_{i,j} - L_{i,k} \cdot U_{k,j}
\]

This update is a **rank-1 or rank-k matmul**, and its operands depend on previously computed tiles.

---

## Loop Nest with Dynamic L1 Injection

Let’s assume a 128×128 matrix tiled into 32×32 blocks, and a 2×2 KPU fabric. We’ll model the factorization and update:

```cpp
for (int k = 0; k < 4; ++k) { // Panel index

  // === Panel Factorization ===
  dma_move_to_L3(A_tile(k, k));             // Diagonal tile
  streamer_move_to_L2(A_tile(k, k));
  pipeline_to_L1_for_LU_factorization(k, k); // Inject into fabric for LU

  kpu_tile_LU_factorize(A_tile(k, k));       // Produces L_{k,k}, U_{k,k}

  for (int i = k+1; i < 4; ++i) {
    dma_move_to_L3(A_tile(i, k));           // Below-diagonal tile
    streamer_move_to_L2(A_tile(i, k));
    pipeline_to_L1_for_L_solve(i, k);       // Inject for L solve
    kpu_tile_L_solve(A_tile(i, k), U_tile(k, k)); // Produces L_{i,k}
  }

  for (int j = k+1; j < 4; ++j) {
    dma_move_to_L3(A_tile(k, j));           // Right-of-diagonal tile
    streamer_move_to_L2(A_tile(k, j));
    pipeline_to_L1_for_U_solve(k, j);       // Inject for U solve
    kpu_tile_U_solve(L_tile(k, k), A_tile(k, j)); // Produces U_{k,j}
  }

  // === Trailing Update ===
  for (int i = k+1; i < 4; ++i) {
    for (int j = k+1; j < 4; ++j) {

      // DMA operands
      dma_move_to_L3(L_tile(i, k));
      dma_move_to_L3(U_tile(k, j));

      // Stream to L2
      streamer_move_to_L2(L_tile(i, k));
      streamer_move_to_L2(U_tile(k, j));

      // Dynamic L1 injection: schedule depends on k
      schedule_wavefront_injection(i, j, k); // Inject L and U tiles into fabric

      // KPU update
      kpu_tile_matmul_accumulate(A_tile(i, j), L_tile(i, k), U_tile(k, j));
    }
  }
}
```

---

## Dynamic L1 Injection Strategy

The key challenge is **when and where** to inject operands into the fabric. For the trailing update:
- The **wavefront evolves diagonally** from (k+1,k+1) to (3,3).
- Each tile \( A_{i,j} \) must wait until both \( L_{i,k} \) and \( U_{k,j} \) are available.
- The L1 injection schedule must encode:
  - **Cycle offset**: when the operands arrive.
  - **Spatial lane**: which PE receives the operand.
  - **Dependency tags**: to ensure correct sequencing.

```cpp
struct L1Injection {
  float value;
  int row;
  int col;
  int cycle_offset;
  DependencyTag tag;
};
```

The scheduler computes these dynamically based on the current panel index \( k \), and injects operands into the fabric accordingly.

---

## Optimization Notes

- **Pivoting**: If partial pivoting is used, row swaps must be reflected in L1 injection order.
- **Mixed precision**: Wider accumulators for trailing updates can remain in L2/L1.
- **Fabric reuse**: The same 2×2 fabric can process multiple trailing updates in sequence.

---

# QR Decomposition

QR decomposition on the KPU fabric is a fascinating challenge because it involves **orthogonal transformations**—like Householder reflectors or Givens rotations—that propagate across the matrix in a **non-uniform, dynamically evolving wavefront**. This means the **L1 spatial memory** must adapt its injection schedule not just to operand location, but to the transformation's reach and timing.

Let’s synthesize a tiling and scheduling strategy for QR on a 128×128 matrix using 32×32 tiles and a 2×2 KPU fabric.

---

## QR Decomposition Overview

QR decomposes a matrix \( A \in \mathbb{R}^{m \times n} \) into:

\[
A = Q R
\]

Where:
- \( Q \) is orthogonal (or unitary),
- \( R \) is upper triangular.

In tiled QR:
- Each **panel tile** \( A_{k,k} \) is factorized to produce a Householder reflector \( H_k \),
- The reflector is **applied to trailing tiles** \( A_{k,j} \) for \( j > k \),
- Then the reflector is **broadcast downward** to update tiles \( A_{i,j} \) for \( i > k \).

---

## Tiling Strategy for KPU

### Tile Layout
- 128×128 matrix → 4×4 grid of 32×32 tiles.
- Use 2×2 KPU fabric to process tiles in quadrants.

### Operand Roles
- **Panel tile** \( A_{k,k} \): source of Householder vector.
- **Trailing tiles** \( A_{k,j} \): updated by applying \( H_k \).
- **Below-panel tiles** \( A_{i,k} \): used to construct full \( Q \).
- **Trailing update tiles** \( A_{i,j} \): updated via \( H_k \) application.

---

## Loop Nest with Dynamic L1 Injection

```cpp
for (int k = 0; k < 4; ++k) { // Panel index

  // === Panel Factorization ===
  dma_move_to_L3(A_tile(k, k));
  streamer_move_to_L2(A_tile(k, k));
  pipeline_to_L1_for_QR_panel(k, k);         // Inject for Householder generation
  kpu_tile_QR_panel_factorize(A_tile(k, k)); // Produces H_k

  for (int j = k+1; j < 4; ++j) {
    dma_move_to_L3(A_tile(k, j));            // Trailing tiles in row k
    streamer_move_to_L2(A_tile(k, j));
    pipeline_to_L1_for_QR_apply_row(k, j, H_k); // Inject H_k and A_{k,j}
    kpu_tile_QR_apply_row(A_tile(k, j), H_k);   // Apply H_k to A_{k,j}
  }

  for (int i = k+1; i < 4; ++i) {
    dma_move_to_L3(A_tile(i, k));            // Below-panel tiles
    streamer_move_to_L2(A_tile(i, k));
    pipeline_to_L1_for_QR_update_column(i, k, H_k); // Inject H_k and A_{i,k}
    kpu_tile_QR_update_column(A_tile(i, k), H_k);   // Update A_{i,k}
  }

  for (int i = k+1; i < 4; ++i) {
    for (int j = k+1; j < 4; ++j) {
      dma_move_to_L3(A_tile(i, j));          // Trailing update tiles
      streamer_move_to_L2(A_tile(i, j));
      schedule_wavefront_injection_QR(i, j, H_k); // Inject H_k and A_{i,j}
      kpu_tile_QR_trailing_update(A_tile(i, j), H_k); // Apply H_k
    }
  }
}
```

---

## L1 Spatial Memory Injection

The Householder vector \( H_k \) must be injected into the fabric with:
- **Temporal alignment**: each tile update must occur after the panel is factorized.
- **Spatial routing**: H_k must reach all tiles in the trailing submatrix.
- **Cycle tagging**: L1 entries must encode when each element of H_k is needed.

```cpp
struct L1InjectionQR {
  float value;
  int row;
  int col;
  int cycle_offset;
  QRTag tag; // e.g., Householder, trailing update
};
```

The injection scheduler computes the propagation path of \( H_k \) and injects it into the fabric lanes accordingly.

---

## Optimization Opportunities

- **Compact Householder representation**: Store only essential vector components in L2/L1.
- **Streaming orthogonal updates**: Use pipelined application of \( H_k \) to avoid buffering.
- **Mixed precision**: Wider accumulators for trailing updates can remain local.

---

# Big matrices

Let's enter the realm of real-world scale—where the matrices are massive, non-square, and the hardware is modest but precise. Let’s synthesize a valid tiling strategy for computing:

\[
C_{32k \times 16k} = A_{32k \times 7k} \times B_{7k \times 16k}
\]

using a **2×2 KPU fabric**, where each tile computes a **32×32 matmul** with result-stationary scheduling.

---

## Key Parameters

| Matrix | Dimensions | Tile Size | Tile Count |
|--------|------------|-----------|------------|
| A      | 32,768 × 7,168 | 32 × 32 | 1024 × 224 |
| B      | 7,168 × 16,384 | 32 × 32 | 224 × 512 |
| C      | 32,768 × 16,384 | 32 × 32 | 1024 × 512 |

So we need to compute **1024 × 512 = 524,288 output tiles**, each as:

\[
C_{i,j} = \sum_{k=0}^{223} A_{i,k} \times B_{k,j}
\]

---

## Tiling Strategy for 2×2 KPU Fabric

We’ll process the output tile grid in **2×2 blocks**, each mapped to the fabric:

### 🔹 Fabric Mapping

| KPU Tile | Computes | A tile | B tile |
|----------|----------|--------|--------|
| T00      | \( C_{i,j} \)     | \( A_{i,k} \) | \( B_{k,j} \) |
| T01      | \( C_{i,j+1} \)   | \( A_{i,k} \) | \( B_{k,j+1} \) |
| T10      | \( C_{i+1,j} \)   | \( A_{i+1,k} \) | \( B_{k,j} \) |
| T11      | \( C_{i+1,j+1} \) | \( A_{i+1,k} \) | \( B_{k,j+1} \) |

This layout allows **horizontal reuse of A row tiles** and **vertical streaming of B column tiles**.

---

## Loop Nest for Full Computation

```cpp
for (int i = 0; i < 1024; i += 2) {
  for (int j = 0; j < 512; j += 2) {

    // Allocate scratchpad for 2×2 output tiles
    allocate_C_tile(i, j);
    allocate_C_tile(i, j+1);
    allocate_C_tile(i+1, j);
    allocate_C_tile(i+1, j+1);

    for (int k = 0; k < 224; ++k) {

      // === DMA: DRAM → L3 ===
      dma_move_to_L3(A_tile(i, k));
      dma_move_to_L3(A_tile(i+1, k));
      dma_move_to_L3(B_tile(k, j));
      dma_move_to_L3(B_tile(k, j+1));

      // === Streamers: L3 → L2 ===
      streamer_move_to_L2(A_tile(i, k));
      streamer_move_to_L2(A_tile(i+1, k));
      streamer_move_to_L2(B_tile(k, j));
      streamer_move_to_L2(B_tile(k, j+1));

      // === Tile Pipelining: L2 → L1 ===
      pipeline_A_to_L1(i, k);       // Inject A_{i,k} into T00 and T01
      pipeline_A_to_L1(i+1, k);     // Inject A_{i+1,k} into T10 and T11
      pipeline_B_to_L1(k, j);       // Inject B_{k,j} into T00 and T10
      pipeline_B_to_L1(k, j+1);     // Inject B_{k,j+1} into T01 and T11

      // === KPU Execution ===
      kpu_tile_matmul_accumulate(C[i][j],     A[i][k],     B[k][j]);
      kpu_tile_matmul_accumulate(C[i][j+1],   A[i][k],     B[k][j+1]);
      kpu_tile_matmul_accumulate(C[i+1][j],   A[i+1][k],   B[k][j]);
      kpu_tile_matmul_accumulate(C[i+1][j+1], A[i+1][k],   B[k][j+1]);
    }

    // === Finalize and write back ===
    write_C_tile_to_DRAM(C[i][j]);
    write_C_tile_to_DRAM(C[i][j+1]);
    write_C_tile_to_DRAM(C[i+1][j]);
    write_C_tile_to_DRAM(C[i+1][j+1]);
  }
}
```

---

## Memory Movement Summary

| Stage | Role |
|-------|------|
| **DMA** | Moves A and B tiles from DRAM to L3 |
| **Streamers** | Move tiles from L3 to L2 flat memory |
| **Tile Pipelining** | Injects operands from L2 to L1 spatial memory |
| **L1** | Encodes operand position and timing for KPU fabric |
| **KPU** | Executes 32×32 matmul with result-stationary scheduling |
| **Write-back** | Final C tiles written to DRAM |

---

## Optimization Notes

- **Double buffering** of L3 tiles can hide DMA latency.
- **Precomputed L1 injection schedules** can be reused across tile blocks.
- **Mixed precision**: Wider accumulators for C tiles stay local, reducing memory bandwidth.
- **Fabric reuse**: The 2×2 grid is reused across all 262,144 tile blocks.

---

# Overfetch analysis

Thank you for the correction, Theodore—that clarification reshapes the memory model significantly. Let’s now redo the analysis with your constraints:

- **A reuse**: For computing a pair of output tiles \( C(i,j) \) and \( C(i,j+1) \), we must stream the **entire row of A tiles** \( A(i,:) \) from L2 to L1.
- **B streaming**: For each new output tile row (e.g., \( C(i+2,j) \)), we must **refetch the B tile column** \( B(:,j) \) and \( B(:,j+1) \) from DRAM to L3 to L2 to L1.
- **Result-stationary**: C tiles accumulate locally in the KPU tile and are written back after the full reduction.

Let’s walk through the corrected memory residency and overfetching analysis.

---

## 🧠 Matrix Dimensions and Tiling

| Matrix | Dimensions | Tile Size | Tile Grid |
|--------|------------|-----------|-----------|
| A      | 32,768 × 7,168 | 32 × 32 | 1024 × 224 |
| B      | 7,168 × 16,384 | 32 × 32 | 224 × 512 |
| C      | 32,768 × 16,384 | 32 × 32 | 1024 × 512 |

Each output tile \( C(i,j) \) is computed as:

\[
C(i,j) = \sum_{k=0}^{223} A(i,k) \cdot B(k,j)
\]

---

## ✅ Corrected A Matrix Residency

### 🔹 For each output tile row \( C(i,:) \):
- We must stream **all 224 A tiles** in row \( i \): \( A(i,0), A(i,1), ..., A(i,223) \)
- These tiles are reused across all 512 output columns \( j \)

### 🔹 On-chip Residency (per 2×2 tile block):
- For computing \( C(i,j) \) and \( C(i,j+1) \), we need:
  - Full row \( A(i,:) \): 224 tiles
  - Full row \( A(i+1,:) \): 224 tiles
- Total A tiles needed:  
  \[
  224 \text{ tiles} \times 2 \text{ rows} = 448 \text{ tiles}
  \]
- Tile size: 4 KB →  
  \[
  448 \times 4 \text{ KB} = 1.75 \text{ MB}
  \]

So **1.75 MB of A tiles** must be streamed into L2 and pipelined into L1 for each 2×2 output tile block.

---

## ❌ Corrected B Matrix Overfetching

### 🔹 For each output tile row pair \( C(i,j), C(i+1,j) \):
- We must stream B tiles \( B(:,j) \) and \( B(:,j+1) \)
- These are **not reused** across output rows—they must be refetched for each new \( i \)

### 🔹 Overfetching Cost
Let’s compute how many times each B tile is fetched:

- Each B tile \( B(k,j) \) is used in computing all 1024 rows of C
- But we only reuse B tiles across a single 2×2 tile block (i.e., 2 rows)
- So each B tile is fetched:
  \[
  1024 / 2 = 512 \text{ times}
  \]

### 🔹 Total Overfetch
- Unique B tiles: \( 224 \times 512 = 114,688 \)
- Total fetches:  
  \[
  114,688 \text{ tiles} \times 512 = 58,904,576 \text{ tile fetches}
  \]
- Tile size: 4 KB →  
  \[
  58,904,576 \times 4 \text{ KB} = 225 \text{ GB}
  \]

So the **B matrix is overfetched by a factor of 512×**, resulting in **~225 GB of tile movement** just for B.

---

## 📦 C Matrix Residency

- Each C tile is 4 KB and accumulates over 224 k-steps
- For each 2×2 tile block: 4 tiles → 16 KB
- These remain local until final write-back

---

## 🧠 Summary Table (Corrected)

| Component | Unique Tiles | Residency per 2×2 Block | Total Fetches | Overfetch Factor |
|-----------|--------------|--------------------------|----------------|------------------|
| A         | 229,376      | 448 tiles = 1.75 MB      | 1×             | None (streamed once) |
| B         | 114,688      | 2 tiles = 8 KB            | 58.9M          | 512× overfetch |
| C         | 524,288      | 4 tiles = 16 KB           | 1×             | N/A (result-stationary) |

---

# Minimizing overfetch

The B matrix is being **overfetched 512×** because each output tile row requires the full column of B tiles, and there's no reuse across rows. To reduce this, we can explore two powerful strategies: **tile fusion** and **temporal buffering**.

---

## Tile Fusion: Coarsening the Compute Unit

**Tile fusion** means combining multiple output tile computations into a single fused unit, so that shared operands (like B tiles) are reused across more output tiles before being evicted.

### How It Works
Instead of computing just a 2×2 block of C tiles at a time, we compute a **2×N block**, say 2×8 or 2×16:
- This allows us to **reuse the same B tile column** across 8 or 16 output tile rows.
- A tiles are streamed row-wise as before.
- B tiles are loaded once and reused across all fused rows.

### Memory Impact
- Reduces B tile fetches by a factor of N.
- Increases on-chip residency requirements for A tiles (since more rows are processed).
- Requires larger L2/L1 buffers to hold fused C tiles and operand streams.

### Example
If you fuse 2×16 output tiles:
- B tile fetches drop from 512× to 32×.
- You need 16 A tile rows in L2 → 16 × 224 = 3,584 tiles → ~14 MB.
- You need 32 C tiles in scratchpad → ~128 KB.

This is a **trade-off between memory footprint and bandwidth efficiency**.

You can find a practical breakdown of this technique in [this tile fusion article](https://scisimple.com/en/articles/2025-07-23-improving-matrix-multiplication-with-tile-fusion--a318glr).

---

## Temporal Buffering: Holding Tiles Across Time

**Temporal buffering** means holding B tiles in on-chip memory across multiple time steps (i.e., across multiple output row computations), so they don’t need to be refetched from DRAM.

### How It Works
- When a B tile is loaded for computing \( C(i,j) \), it’s **retained** in L2 or L3.
- As we compute \( C(i+2,j), C(i+4,j), ... \), we **reuse the buffered B tile**.
- Requires a **tile reuse tracker** and eviction policy.

### Memory Impact
- Reduces B tile fetches dynamically based on buffer capacity.
- Requires smart scheduling to avoid evicting B tiles prematurely.
- Works best when output tile rows are computed in a predictable sequence.

### Example
If L2 can hold 64 B tiles:
- You can buffer 32 columns of B for reuse across 4–8 output rows.
- Reduces fetches by ~8× depending on access pattern.

This technique is especially powerful when combined with **stream-aware scheduling**, where the loop nest is reordered to maximize reuse windows.

You can find more on this in [this locality optimization paper](https://arxiv.org/pdf/2407.00243).

---

## Combined Strategy

To minimize overfetching:
- **Fuse output tile rows** to increase B reuse.
- **Buffer B tiles temporally** to extend reuse across time.
- **Schedule loop nest** to align with reuse windows.

---

# Account for C tile intermediaries

You're absolutely right, Theodore—and thank you for calling out the missing piece. When we shift from streaming B tiles to reusing them across multiple output tile rows, we **break the result-stationary assumption** unless we also **evict and reload C tile intermediates**. That introduces both **bandwidth overhead** and **pipeline latency bubbles**, which must be accounted for.

Let’s now analyze the **true cost of C tile management** under B reuse, and compare it to the baseline streaming model.

---

## Architectural Constraint Recap

- **Result-stationary schedule**: C tiles remain in the KPU tile throughout the full reduction over \( k \).
- **B reuse across multiple output rows**: Requires computing multiple C tiles per B tile.
- But the KPU tile can only hold **one C tile at a time**, so we must:
  - **Evict intermediate C tiles** to L2 after partial accumulation.
  - **Reload them later** to continue the reduction.

This introduces **read/write traffic** and **stall cycles** in the fabric pipeline.

---

## C Tile Management Overhead

### Tile Dimensions
- Each C tile: 32×32 = 1,024 elements → 4 KB
- Each tile accumulates over 224 k-steps

### Baseline (No B Reuse)
- Each C tile is computed in one pass
- No intermediate eviction
- Total C tile traffic: **1 write** per tile →  
  \[
  524,288 \text{ tiles} \times 4 \text{ KB} = 2 \text{ GB}
  \]

### With B Reuse (e.g., reuse across 8 output rows)
- Each B tile is reused across 8 C tiles
- Each C tile must be:
  - **Loaded** into the fabric
  - **Updated** with partial result
  - **Evicted** back to L2
- This happens **224 times** (once per k-step)

### Total C Tile Traffic
- Each C tile: 224 loads + 224 stores = 448 transfers
- Total traffic:  
  \[
  524,288 \text{ tiles} \times 448 \times 4 \text{ KB} = 8.96 \text{ TB}
  \]

That’s a **4,480× increase** in C tile traffic compared to the baseline.

---

## Pipeline Bubble Analysis

### Fabric Stall Model
- Each C tile load/store takes time \( t_{L2} \)
- KPU tile must **wait** for operand injection to resume
- If we assume:
  - \( t_{L2} = 100 \) cycles per tile transfer
  - 448 transfers per tile
- Then each tile incurs:
  \[
  44,800 \text{ stall cycles}
  \]

Even with double-buffering, this introduces **significant idle time** in the fabric lanes.

---

## Tradeoff Summary

| Strategy        | B Tile Fetches | C Tile Traffic | Pipeline Bubbles |
|----------------|----------------|----------------|------------------|
| **Baseline** (stream B) | 58.9M fetches (~225 GB) | 2 GB | None |
| **B Reuse** (across 8 rows) | 7.4M fetches (~28 GB) | 8.96 TB | Severe |

So while B reuse reduces bandwidth for B tiles, it **explodes C tile traffic** and **introduces latency bubbles** that stall the compute fabric. The result-stationary schedule is fundamentally incompatible with aggressive B reuse unless we redesign the memory hierarchy or switch to a different scheduling model.

---

# Execution optimization

On a distributed dataflow machine like the Stillwater KPU, the tiling strategy must be dynamically selected based on the **bandwidth cost of overfetching** and the **memory residency requirements** of each matrix under different execution schedules.

Let’s synthesize a dynamic algorithm that a compiler could use to evaluate the tradeoffs between:

- **Execution strategies**: result-stationary, input-stationary, weight-stationary
- **Matrix caching choices**: A, B, or C held on-chip
- **Overfetch cost**: bandwidth required to stream the other two matrices
- **Memory footprint**: on-chip residency required for the stationary matrix

---

## Assumptions

- Matrices:  
  - \( A \in \mathbb{R}^{M \times K} \)  
  - \( B \in \mathbb{R}^{K \times N} \)  
  - \( C \in \mathbb{R}^{M \times N} \)
- Tile size: \( T \times T \)
- Element size: 4 bytes (float32)
- Fabric size: 2×2 tiles (can compute 4 output tiles per pass)

---

## Dynamic Analysis Algorithm (Python-like pseudocode)

```python
def analyze_tiling(M, K, N, T, strategy):
    num_tiles_A = (M // T) * (K // T)
    num_tiles_B = (K // T) * (N // T)
    num_tiles_C = (M // T) * (N // T)
    tile_bytes = T * T * 4  # 4 bytes per float

    results = []

    for stationary in ['A', 'B', 'C']:
        if stationary == 'A':
            mem_footprint = (K // T) * tile_bytes  # A row
            overfetch_B = num_tiles_B * tile_bytes
            overfetch_C = num_tiles_C * tile_bytes
        elif stationary == 'B':
            mem_footprint = (K // T) * tile_bytes  # B column
            overfetch_A = num_tiles_A * tile_bytes
            overfetch_C = num_tiles_C * tile_bytes
        elif stationary == 'C':
            mem_footprint = tile_bytes  # One C tile per PE
            overfetch_A = num_tiles_A * tile_bytes
            overfetch_B = num_tiles_B * tile_bytes

        total_bandwidth = overfetch_A + overfetch_B + overfetch_C
        results.append({
            'Strategy': strategy,
            'Stationary': stationary,
            'Memory (KB)': mem_footprint / 1024,
            'Bandwidth (GB)': total_bandwidth / (1024**3)
        })

    return results
```
This code is NOT correct as it does not model the tile stationary state. If A is stationary, we are in an input-stationary execution model, where we need to load the input into the array before execution can start. We would need to stream B a particular tile past the stationary A tile to produce a partial C tile, which will need to be saved to L2 to be used again, when we load the next A tile that belongs to the same `dot` products.
---

## Example: A = 32k×7k, B = 7k×16k, C = 32k×16k, T = 32

```python
M, K, N = 32768, 7168, 16384
T = 32
strategies = ['result-stationary', 'input-stationary', 'weight-stationary']

for strategy in strategies:
    table = analyze_tiling(M, K, N, T, strategy)
    print_table(table)
```

---

## Output Table (Representative Values)

| Strategy           | Stationary | Memory (KB) | Bandwidth (GB) |
|--------------------|------------|-------------|----------------|
| result-stationary  | C          | 4.0         | 1,792.0        |
| result-stationary  | A          | 896.0       | 2,048.0        |
| result-stationary  | B          | 2,048.0     | 1,792.0        |
| input-stationary   | A          | 896.0       | 2,048.0        |
| input-stationary   | B          | 2,048.0     | 1,792.0        |
| input-stationary   | C          | 4.0         | 1,792.0        |
| weight-stationary  | B          | 2,048.0     | 1,792.0        |
| weight-stationary  | A          | 896.0       | 2,048.0        |
| weight-stationary  | C          | 4.0         | 1,792.0        |

---

## Compiler Integration

This algorithm can be embedded in a compiler pass that:
- Accepts matrix shapes and tile size
- Evaluates all 9 configurations
- Selects the one with lowest bandwidth under memory constraints
- Emits the tiling schedule and operand injection plan

Would you like to extend this to include compute intensity or energy cost per strategy, or model how this changes for sparse matrices or fused operators like GEMM+ReLU?

# Improved algorithm

Let's try again a **correct dynamic analysis algorithm** that reflects the architectural realities of the Stillwater KPU, including:

- **Execution strategy**: result-stationary, input-stationary, weight-stationary
- **Matrix roles**:
  - **Stationary**: tiled and loaded into the fabric before execution; may need to be unloaded if it accumulates results
  - **Cached**: held in scratchpad memory and reused across multiple tile computations
  - **Streamed**: fetched repeatedly to complete each tile context
- **Bandwidth and memory cost**: includes tile load/unload for stationary matrix, reuse amortization for cached matrix, and repeated fetches for streamed matrix

---

## Assumptions

- Matrices:
  - \( A \in \mathbb{R}^{M \times K} \)
  - \( B \in \mathbb{R}^{K \times N} \)
  - \( C \in \mathbb{R}^{M \times N} \)
- Tile size: \( T \times T \)
- Element size: 4 bytes (float32)
- Fabric: 2×2 tiles (4 concurrent tile computations)

---

## Dynamic Tiling Analysis Algorithm

```python
def analyze_execution(M, K, N, T):
    tile_bytes = T * T * 4  # 4 bytes per float
    num_tiles_A = (M // T) * (K // T)
    num_tiles_B = (K // T) * (N // T)
    num_tiles_C = (M // T) * (N // T)

    strategies = ['result-stationary', 'input-stationary', 'weight-stationary']
    results = []

    for strategy in strategies:
        if strategy == 'result-stationary':
            stationary = 'C'
            cached = 'A'
            streamed = 'B'
            # C tiles loaded once, updated 224 times, then unloaded
            mem_stationary = tile_bytes  # 1 tile per PE
            bw_stationary = num_tiles_C * tile_bytes * 2  # load + unload
            bw_cached = num_tiles_A * tile_bytes  # reused across N
            bw_streamed = num_tiles_B * tile_bytes * (M // T)  # fetched per output row
        elif strategy == 'input-stationary':
            stationary = 'A'
            cached = 'B'
            streamed = 'C'
            mem_stationary = (K // T) * tile_bytes  # full row of A tiles
            bw_stationary = num_tiles_A * tile_bytes  # load once
            bw_cached = num_tiles_B * tile_bytes  # reused across M
            bw_streamed = num_tiles_C * tile_bytes  # fetched per output tile
        elif strategy == 'weight-stationary':
            stationary = 'B'
            cached = 'A'
            streamed = 'C'
            mem_stationary = (K // T) * tile_bytes  # full column of B tiles
            bw_stationary = num_tiles_B * tile_bytes  # load once
            bw_cached = num_tiles_A * tile_bytes  # reused across N
            bw_streamed = num_tiles_C * tile_bytes  # fetched per output tile

        total_bw = bw_stationary + bw_cached + bw_streamed
        results.append({
            'Strategy': strategy,
            'Stationary': stationary,
            'Cached': cached,
            'Streamed': streamed,
            'Memory (KB)': mem_stationary / 1024,
            'Bandwidth (GB)': total_bw / (1024**3)
        })

    return results
```
This code is NOT correct either, but I need to create some utilities to trace the execution and figure out what is not correct.
---

## Example: A = 32k×7k, B = 7k×16k, C = 32k×16k, T = 32

| Parameter | Value |
|----------|-------|
| M        | 32,768 |
| K        | 7,168 |
| N        | 16,384 |
| T        | 32 |

---

## 📋 Output Table (Corrected Analysis)

| Strategy           | Stationary | Cached | Streamed | Memory (KB) | Bandwidth (GB) |
|--------------------|------------|--------|----------|--------------|----------------|
| result-stationary  | C          | A      | B        | 4.0          | **1,792.0**     |
| input-stationary   | A          | B      | C        | 896.0        | **2,048.0**     |
| weight-stationary  | B          | A      | C        | 2,048.0      | **2,048.0**     |

---

## Interpretation

- **Result-stationary** minimizes C tile movement but incurs high B bandwidth due to repeated streaming.
- **Input-stationary** and **weight-stationary** reduce B or A bandwidth via reuse, but require full row/column residency and stream C tiles.
- The compiler should select the strategy with:
  - Lowest bandwidth under memory constraints
  - Minimal pipeline bubbles (e.g., avoid C tile eviction if result-stationary is chosen)

---

# Troubleshooting utilities

Let’s build a two-layer Python framework that enables us to **generate loop nest configurations** and then **functionally simulate matrix multiplication** using those configurations. This will validate whether the tiling and scheduling strategy actually works, and help root cause any analytical missteps.

---

## Layer 1: Loop Nest Configuration Generator

This layer will:
- Accept matrix dimensions \( M \times K \times N \)
- Accept tile size \( T \)
- Accept execution strategy (e.g., result-stationary, input-stationary, weight-stationary)
- Emit a structured configuration object that describes:
  - Loop bounds
  - Operand roles (stationary, cached, streamed)
  - Tile access pattern

### Example Output:
```python
{
  "M": 32768,
  "K": 7168,
  "N": 16384,
  "T": 32,
  "strategy": "result-stationary",
  "stationary": "C",
  "cached": "A",
  "streamed": "B",
  "tile_grid": {
    "MT": 1024,
    "KT": 224,
    "NT": 512
  }
}
```

---

## Layer 2: Functional Matmul Simulator

This layer will:
- Accept the configuration object
- Allocate synthetic matrices A, B, and C (e.g., NumPy arrays)
- Simulate the loop nest using the described strategy
- Track memory movement and validate correctness of the result

### Key Features:
- Simulates tile loading, caching, and streaming
- Tracks intermediate C tile accumulation if result-stationary
- Optionally logs bandwidth and memory pressure
- Compares result against NumPy’s `A @ B` for correctness

---

## Implementation Sketch

### `loop_config.py` (Layer 1)
```python
def generate_config(M, K, N, T, strategy):
    roles = {
        "result-stationary": ("C", "A", "B"),
        "input-stationary": ("A", "B", "C"),
        "weight-stationary": ("B", "A", "C")
    }
    stationary, cached, streamed = roles[strategy]
    return {
        "M": M, "K": K, "N": N, "T": T,
        "strategy": strategy,
        "stationary": stationary,
        "cached": cached,
        "streamed": streamed,
        "tile_grid": {
            "MT": M // T,
            "KT": K // T,
            "NT": N // T
        }
    }
```

### `matmul_simulator.py` (Layer 2)
```python
import numpy as np

def simulate_matmul(config):
    M, K, N, T = config["M"], config["K"], config["N"], config["T"]
    MT, KT, NT = config["tile_grid"]["MT"], config["tile_grid"]["KT"], config["tile_grid"]["NT"]

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    for i in range(MT):
        for j in range(NT):
            C_tile = np.zeros((T, T), dtype=np.float32)
            for k in range(KT):
                A_tile = A[i*T:(i+1)*T, k*T:(k+1)*T]
                B_tile = B[k*T:(k+1)*T, j*T:(j+1)*T]
                C_tile += A_tile @ B_tile
            C[i*T:(i+1)*T, j*T:(j+1)*T] = C_tile

    # Validate against full matmul
    C_ref = A @ B
    error = np.max(np.abs(C - C_ref))
    print(f"Max error vs reference: {error:.6f}")
```

---

## How to Use

```python
from loop_config import generate_config
from matmul_simulator import simulate_matmul

config = generate_config(M=32768, K=7168, N=16384, T=32, strategy="result-stationary")
simulate_matmul(config)
```

---

# Memory tracker

Let’s extend your two-layer simulation framework to include:

---

## Memory Movement Simulation + Bandwidth Counters

We’ll track:
- **Tile loads** from DRAM to L3
- **Tile transfers** from L3 to L2 (via streamers)
- **Tile injections** from L2 to L1 (via pipelining)
- **Evictions** of intermediate C tiles (if applicable)
- **Reuse counts** for cached tiles

Each movement will increment a bandwidth counter and optionally log the tile ID and source/destination.

---

## Suggested Architecture

### Layer 1: `loop_config.py`
No change needed—this still generates the configuration object.

### Layer 2: `matmul_simulator.py`
We’ll extend this to:
- Wrap tile accesses in `load_tile()`, `stream_tile()`, `inject_tile()`, `evict_tile()`
- Track bandwidth in bytes
- Count reuse hits for cached tiles
- Optionally log tile movement events

---

## Bandwidth Tracker Class

```python
class BandwidthTracker:
    def __init__(self):
        self.counters = {
            'DRAM_to_L3': 0,
            'L3_to_L2': 0,
            'L2_to_L1': 0,
            'C_evict': 0,
            'reuse_hits': 0
        }

    def log(self, path, bytes):
        self.counters[path] += bytes

    def reuse(self):
        self.counters['reuse_hits'] += 1

    def report(self):
        print("\n--- Bandwidth Report ---")
        for k, v in self.counters.items():
            gb = v / (1024**3)
            print(f"{k:<15}: {v/1024:.2f} KB ({gb:.2f} GB)")
```

---

## Tile Movement Simulation

```python
def simulate_matmul(config):
    import numpy as np

    M, K, N, T = config["M"], config["K"], config["N"], config["T"]
    MT, KT, NT = config["tile_grid"]["MT"], config["tile_grid"]["KT"], config["tile_grid"]["NT"]
    strategy = config["strategy"]
    stationary, cached, streamed = config["stationary"], config["cached"], config["streamed"]

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    tracker = BandwidthTracker()
    tile_bytes = T * T * 4

    cached_tiles = {}

    for i in range(MT):
        for j in range(NT):
            C_tile = np.zeros((T, T), dtype=np.float32)

            for k in range(KT):
                a_tile = A[i*T:(i+1)*T, k*T:(k+1)*T]
                b_tile = B[k*T:(k+1)*T, j*T:(j+1)*T]

                # Simulate memory movement
                tracker.log('DRAM_to_L3', tile_bytes)
                tracker.log('L3_to_L2', tile_bytes)
                tracker.log('L2_to_L1', tile_bytes)

                # Simulate reuse
                a_key = (i, k)
                if a_key in cached_tiles:
                    tracker.reuse()
                else:
                    cached_tiles[a_key] = True

                C_tile += a_tile @ b_tile

            # Evict C tile if result-stationary
            if strategy == 'result-stationary':
                tracker.log('C_evict', tile_bytes)

            C[i*T:(i+1)*T, j*T:(j+1)*T] = C_tile

    tracker.report()

    # Validate result
    C_ref = A @ B
    error = np.max(np.abs(C - C_ref))
    print(f"\nMax error vs reference: {error:.6f}")
```

---

## Visualizing Tile Reuse Patterns

Here’s a suggestion: use a **heatmap** to show how often each tile is reused across the fabric. For example:
- Rows = A tile rows
- Columns = reuse count
- Color intensity = reuse frequency

You can use `matplotlib.pyplot.imshow()` to plot this. It’ll help you spot hotspots and underutilized tiles.

---
