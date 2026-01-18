# Memory Hierarchy Performance Modeling for Software-Managed Architectures

**Document Version:** 1.0
**KPU Simulator Version:** v0.3.2
**Authors:** Stillwater Supercomputing
**Date:** 2026-01-18

---

## Abstract

This document provides a comprehensive analysis of memory hierarchy performance modeling, with particular emphasis on **software-managed memory architectures** such as the Knowledge Processing Unit (KPU). Unlike cache-based systems that rely on hardware-driven replacement policies and demand-fetching, software-managed architectures require explicit, a-priori scheduling of data movement. We survey the fundamental theoretical results from I/O complexity theory, communication-avoiding algorithms, and distributed memory machine research, then articulate the specific challenges and tool requirements for the KPU architecture.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Fundamental Theory: I/O Complexity](#2-fundamental-theory-io-complexity)
3. [Communication-Avoiding Algorithms](#3-communication-avoiding-algorithms)
4. [Working Set Theory and Reuse Analysis](#4-working-set-theory-and-reuse-analysis)
5. [The Roofline Model and Its Limitations](#5-the-roofline-model-and-its-limitations)
6. [Software-Managed Memory Hierarchies](#6-software-managed-memory-hierarchies)
7. [Distributed Memory Machine Insights](#7-distributed-memory-machine-insights)
8. [KPU Architecture: Software-Managed Layers](#8-kpu-architecture-software-managed-layers)
9. [Tool Requirements for Schedule Analysis](#9-tool-requirements-for-schedule-analysis)
10. [Research Directions](#10-research-directions)
11. [References](#references)

---

## 1. Introduction

The performance of computational kernels on modern architectures is increasingly dominated by **data movement costs** rather than arithmetic operations. This "memory wall" phenomenon, first articulated by Wulf and McKee [1], has driven decades of research into understanding and minimizing communication between levels of the memory hierarchy.

The KPU architecture represents a departure from conventional cache-based designs by employing a **software-managed memory hierarchy**. Rather than relying on hardware cache controllers to make real-time decisions about data placement, the KPU requires that all data movement be explicitly scheduled by software—either by a programmer, a compiler, or an automated scheduling tool.

This design choice has profound implications:

1. **No demand-driven fetching**: Data must arrive at the correct memory layer *before* it is needed
2. **No automatic replacement**: Software must explicitly manage buffer allocation and deallocation
3. **Deterministic behavior**: Execution timing is predictable given the schedule
4. **Optimization opportunity**: Global knowledge enables optimal scheduling (in principle)
5. **Complexity burden**: The scheduling problem is NP-hard in the general case

To build effective tools for KPU schedule analysis, we must understand the theoretical foundations that govern data movement costs and the practical techniques developed for similar software-managed systems.

---

## 2. Fundamental Theory: I/O Complexity

### 2.1 The Red-Blue Pebble Game

The foundational result in I/O complexity theory comes from Hong and Kung's seminal 1981 paper [2]. They introduced the **Red-Blue Pebble Game** as a formal model for analyzing data movement in computations with a two-level memory hierarchy.

**Model Definition:**
- A computation is represented as a directed acyclic graph (DAG)
- **Red pebbles** (limited quantity M) represent fast memory (cache/scratchpad)
- **Blue pebbles** (unlimited) represent slow memory (DRAM)
- Rules govern when computation can proceed and when data must move

**Key Theorem (Hong-Kung, 1981):**

For matrix multiplication of n×n matrices with fast memory of size M words:

```
I/O Lower Bound: Ω(n³/√M)
```

This result establishes that **no algorithm**—regardless of cleverness—can perform n×n matrix multiplication with fewer than Θ(n³/√M) memory operations when fast memory is limited to M words.

**Implications:**
- Optimal blocking uses tiles of size B ≈ √M
- I/O cost scales inversely with √M, not M
- Doubling cache size only reduces traffic by factor of √2

### 2.2 Extension to Multi-Level Hierarchies

Savage and Vitter [3] extended the I/O model to multiple memory levels. For L levels with capacities M₁ < M₂ < ... < Mₗ:

```
Traffic at level i: Θ(n³/√Mᵢ)
```

This result has critical implications for the KPU's three-level hierarchy (L1/L2/L3):
- Each level has its own optimal tile size
- Hierarchical blocking (tiles of tiles) is necessary for optimality
- The working set at level i must fit in Mᵢ for full reuse

### 2.3 General DAG Computations

Bilardi and Preparata [4] generalized the Hong-Kung analysis to arbitrary computation DAGs, introducing the concept of the **S-span**—the minimum number of I/O operations required for any valid schedule.

For a computation with work W and parallelism P:
```
I/O ≥ W / (M × P)  (trivial bound)
I/O ≥ S-span      (structural bound)
```

The gap between these bounds determines how much optimization is possible through clever scheduling.

---

## 3. Communication-Avoiding Algorithms

### 3.1 The Communication-Avoiding Framework

Demmel, Ballard, Holtz, and Schwartz developed a comprehensive framework for **communication-avoiding algorithms** [5, 6, 7]. Their key insight: classical algorithms often perform asymptotically more communication than the lower bounds require.

**Definition (Communication Optimality):**
An algorithm is **communication-optimal** if it achieves the I/O lower bound to within a constant factor.

### 3.2 Results for Dense Linear Algebra

| Operation | Classical I/O | Optimal I/O | Speedup |
|-----------|---------------|-------------|---------|
| Matrix Multiply | O(n³) | O(n³/√M) | √M |
| LU Factorization | O(n³) | O(n³/√M) | √M |
| QR Factorization | O(n³) | O(n³/√M) | √M |
| Eigenvalue | O(n³) | O(n³/√M) | √M |
| Cholesky | O(n³) | O(n³/√M) | √M |

**Key Insight:** All these operations achieve the same I/O complexity as matrix multiplication because they are dominated by matrix multiplication at their core.

### 3.3 Communication-Avoiding Matrix Multiplication (CARMA)

Demmel et al. [7] developed **CARMA** (Communication-Avoiding Recursive Matrix multiplication Algorithm):

```
CARMA achieves:
- Sequential: O(n³/√M) I/O operations
- Parallel (P processors): O(n³/P) computation, O(n²/√P) communication
- Optimal for all problem sizes and parallelism levels
```

The algorithm uses **recursive decomposition** rather than fixed-size blocking:
1. Recursively divide matrices until subproblems fit in cache
2. Schedule subproblems to maximize reuse
3. Automatically adapts to cache size (cache-oblivious property)

### 3.4 The 2.5D Algorithm

Solomonik and Demmel [8] introduced the **2.5D matrix multiplication algorithm** for distributed memory:

```
For P processors with memory M per processor:
Communication Volume: O(n²/√(P×c))
Where c = M/(n²/P) is the "replication factor"
```

This shows that **trading memory for communication** can reduce data movement, a principle directly applicable to KPU buffer sizing.

---

## 4. Working Set Theory and Reuse Analysis

### 4.1 Denning's Working Set Model

Denning's seminal work [9] introduced the **working set concept**:

**Definition:** The working set W(t, τ) is the set of distinct memory locations referenced during the time interval [t-τ, t].

The **working set size** |W(t, τ)| characterizes the memory footprint of a computation:
- If |W(t, τ)| ≤ M: Full temporal reuse possible
- If |W(t, τ)| > M: Capacity misses inevitable

### 4.2 Reuse Distance Analysis

Mattson, Gecsei, Slutz, and Traiger [10] introduced **stack algorithms** and the concept of **reuse distance**:

**Definition:** The reuse distance of a memory access is the number of distinct memory locations accessed since the previous access to the same location.

**Key Property:** For a fully-associative LRU cache of size M:
```
If reuse_distance < M: Cache hit
If reuse_distance ≥ M: Cache miss
```

The **reuse distance distribution** completely characterizes cache behavior:
```
Miss Rate(M) = P(reuse_distance ≥ M)
```

### 4.3 Miss Ratio Curves

The **Miss Ratio Curve (MRC)** plots miss rate versus cache size:

```
MRC(M) = fraction of accesses with reuse_distance ≥ M
```

MRCs are fundamental for:
- Cache sizing decisions
- Working set characterization
- Predicting performance across memory configurations

**Efficient MRC Construction:** Ding and Zhong [11] developed O(N log N) algorithms for computing reuse distances from memory traces.

### 4.4 Locality Metrics for Tiled Algorithms

For blocked/tiled algorithms, locality manifests at multiple scales:

| Scale | Locality Type | Characteristic |
|-------|--------------|----------------|
| Intra-tile | Temporal + Spatial | Reuse within a single tile computation |
| Inter-tile | Temporal | Reuse of tiles across outer loop iterations |
| Cross-operand | Spatial | Streaming access patterns for different matrices |

Lam, Rothberg, and Wolf [12] analyzed blocked matrix algorithms and established:

```
For optimal blocking with tile size B:
- Intra-tile reuse: O(B) accesses per load
- Total loads: O(n³/B)
- Optimal B = √(M/3) for C = A×B (three matrices)
```

---

## 5. The Roofline Model and Its Limitations

### 5.1 The Standard Roofline Model

Williams, Waterman, and Patterson [13] introduced the **Roofline Model** as a visual performance bound:

```
Performance ≤ min(Peak_FLOPS, Peak_Bandwidth × Arithmetic_Intensity)
```

Where **Arithmetic Intensity (AI)** = FLOPs / Bytes transferred

The **ridge point** occurs at:
```
Ridge = Peak_FLOPS / Peak_Bandwidth (FLOP/byte)
```

### 5.2 Limitations of the Roofline Model

The roofline model makes several **simplifying assumptions** that limit its applicability:

1. **Steady-state assumption**: Ignores warm-up and cool-down phases
2. **Perfect overlap**: Assumes computation fully overlaps with communication
3. **Single bandwidth**: Ignores multi-level hierarchy effects
4. **No contention**: Assumes full bandwidth always available
5. **Infinite reuse**: Assumes all reuse opportunities are captured

### 5.3 Effective vs. Peak Metrics

For realistic modeling, we need **effective metrics**:

```
Effective_Bandwidth = Actual_Bytes_Transferred / Time
Effective_AI = FLOPs / Actual_Bytes_Transferred (including overfetch)

Where: Actual_Bytes ≥ Theoretical_Bytes due to:
- Capacity misses (working set > cache)
- Conflict misses (mapping collisions)
- Compulsory misses (first access)
- Coherence traffic (multi-core)
```

### 5.4 The Cache-Aware Roofline

Ilic, Pratas, and Sousa [14] extended the roofline to multiple memory levels:

```
Performance ≤ min(Peak_FLOPS,
                  BW_L1 × AI_L1,
                  BW_L2 × AI_L2,
                  BW_L3 × AI_L3,
                  BW_DRAM × AI_DRAM)
```

Each level has its own ridge point, creating a **staircase roofline**.

---

## 6. Software-Managed Memory Hierarchies

### 6.1 Historical Context

Software-managed memory hierarchies predate caches and have seen periodic resurgence:

- **Early vector machines** (Cray-1): Explicit register allocation
- **Scratchpad memories**: Common in embedded systems
- **Cell Broadband Engine** (2006): 256KB software-managed local store
- **GPU shared memory**: Explicitly managed per-thread-block
- **Modern accelerators**: TPU, KPU, various AI chips

### 6.2 The Cell Processor Experience

The Cell Broadband Engine [15, 16] provides the most extensively studied software-managed architecture:

**Architecture:**
- 1 PPE (PowerPC) + 8 SPEs (Synergistic Processing Elements)
- Each SPE: 256KB Local Store (LS), no cache
- All data movement via explicit DMA

**Key Research Findings:**

1. **Double buffering essential**: Overlapping DMA with computation requires 2× buffer space
2. **DMA latency hiding**: Need sufficient computation per transfer to amortize latency
3. **Optimal block sizes**: Determined by LS capacity and DMA characteristics
4. **Software pipelining**: Multi-stage pipelines for complex algorithms

Buttari et al. [17] demonstrated that matrix operations on Cell achieved near-peak performance through careful software management, validating the approach.

### 6.3 Scratchpad Memory Research

Banakar et al. [18] and Udayakumaran et al. [19] extensively studied scratchpad memory allocation:

**Key Results:**

1. **Static allocation** (compile-time) is optimal for regular access patterns
2. **Dynamic allocation** (runtime) needed for data-dependent patterns
3. **Optimal allocation is NP-hard** in the general case
4. **Integer Linear Programming (ILP)** formulations can find optimal solutions for small problems

**Allocation Strategies:**
```
Static: Assign data to scratchpad at compile time
Dynamic: Copy data in/out at runtime
Hybrid: Static for persistent data, dynamic for temporary
```

### 6.4 The SPM Allocation Problem

Formally, scratchpad allocation can be modeled as:

```
Minimize: Total_Data_Movement
Subject to:
  - At any time t: Σ allocated_data(t) ≤ SPM_capacity
  - Data must be present before use
  - Precedence constraints from computation DAG
```

This is equivalent to a **scheduling problem** with resource constraints—specifically, a variant of job-shop scheduling.

---

## 7. Distributed Memory Machine Insights

### 7.1 The BSP Model

Valiant's **Bulk Synchronous Parallel (BSP)** model [20] provides a clean abstraction for distributed memory:

**Model Parameters:**
- p: Number of processors
- L: Synchronization latency (barrier cost)
- g: Bandwidth gap (time per word communicated)

**Cost Model:**
```
T = W/p + h×g + L
Where: W = work, h = communication volume
```

**Superstep Structure:**
1. Local computation
2. Global communication
3. Barrier synchronization

The BSP model directly maps to KPU's pipelined execution with explicit data movement phases.

### 7.2 LogP and LogGP Models

Culler et al. [21] introduced the **LogP model** for more precise communication modeling:

**Parameters:**
- L: Latency (time for small message)
- o: Overhead (processor busy time per message)
- g: Gap (minimum time between messages)
- P: Number of processors

**Extension - LogGP** [22] adds:
- G: Gap per byte for large messages

These models capture the **pipelining** of communication that the KPU exploits.

### 7.3 Data Distribution Research

The **High Performance Fortran (HPF)** and **ScaLAPACK** communities developed extensive theory for data distribution [23, 24]:

**Distribution Types:**
- **Block**: Contiguous chunks to processors
- **Cyclic**: Round-robin distribution
- **Block-cyclic**: Combines both (used in ScaLAPACK)

**Key Insight:** The distribution determines communication patterns:
```
Block distribution:
  - Good for algorithms with local dependencies
  - May cause load imbalance

Cyclic distribution:
  - Good load balance
  - May cause excessive communication

Block-cyclic (block size b):
  - Tunable trade-off
  - b → ∞: Block distribution
  - b → 1: Cyclic distribution
```

### 7.4 Automatic Data Layout

The **PARADIGM** compiler [25] and later work automated data layout decisions:

**Approach:**
1. Analyze data access patterns
2. Build affinity graph (which data accessed together)
3. Partition affinity graph (minimize edge cuts = communication)
4. Map partitions to memory locations

**Result:** Automated methods achieve 80-95% of hand-tuned performance for regular codes.

---

## 8. KPU Architecture: Software-Managed Layers

### 8.1 KPU Memory Hierarchy

The KPU implements a three-level software-managed memory hierarchy:

```
┌─────────────────────────────────────────────────────────┐
│                    External DRAM                         │
│                    (Large, High Latency)                 │
└────────────────────────┬────────────────────────────────┘
                         │ DMA Engine
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    L3 Buffers                            │
│              (Software-Managed Tile Store)               │
│                   Capacity: ~MB                          │
└────────────────────────┬────────────────────────────────┘
                         │ Block Mover
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    L2 Buffers                            │
│              (Per-Tile Working Store)                    │
│                   Capacity: ~KB                          │
└────────────────────────┬────────────────────────────────┘
                         │ Streamer
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    L1 Streams                            │
│              (Operand Delivery to Compute)               │
│                   Capacity: ~Elements                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Systolic Array                          │
│                 (16×16 Compute Fabric)                   │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Data Movement Primitives

Each level transition has a dedicated data movement engine:

| Engine | Source | Destination | Characteristics |
|--------|--------|-------------|-----------------|
| DMA | DRAM | L3 | High latency, high bandwidth, asynchronous |
| Block Mover | L3 | L2 | Medium latency, software-triggered |
| Streamer | L2 | L1 | Low latency, feeds systolic array |

### 8.3 The Credit-Based Flow Model

Unlike cache-based systems, the KPU uses **credit-based dataflow**:

```
Credits flow UP (consumer → producer):
  "I have space, send me data"

Data flows DOWN (producer → consumer):
  "Here is the tile you requested"

Invariant: Producer only sends when credit available
Result: No overflow, deterministic flow
```

This model requires **a-priori knowledge** of data dependencies to issue credits at the right time.

### 8.4 The Scheduling Challenge

For compute-bound operation (e.g., large GEMM), the goal is:

```
Compute Utilization = 100%
⟺ Tiles arrive at L1 exactly when needed
⟺ L2→L1 transfer completes just-in-time
⟺ L3→L2 transfer completes before L2→L1 needed
⟺ DRAM→L3 transfer completes before L3→L2 needed
```

This creates a **cascade of timing constraints**:
```
t_compute(tile_i) = t_arrive_L1(tile_i) + L1_access_time
t_arrive_L1(tile_i) ≤ t_needed(tile_i)
t_start_L2_to_L1(tile_i) = t_arrive_L1(tile_i) - L2_to_L1_latency
t_arrive_L2(tile_i) ≤ t_start_L2_to_L1(tile_i)
... (cascade continues to DRAM)
```

### 8.5 Reuse in Software-Managed Systems

Without automatic caching, reuse must be **explicitly managed**:

**Temporal Reuse (same tile used multiple times):**
```
Strategy: Keep tile resident at appropriate level
Cost: Buffer space occupied longer
Benefit: Avoid re-fetch from slower level
```

**Spatial Reuse (adjacent tiles share data):**
```
Strategy: Overlap tile boundaries or fetch larger regions
Cost: May fetch unused data (overfetch)
Benefit: Reduce total transfers
```

### 8.6 The Overfetch Problem

When working sets exceed buffer capacity, **overfetch** is unavoidable:

```
Overfetch_Ratio = Actual_Bytes_Moved / Minimum_Bytes_Needed

Causes:
1. Capacity: Working set > buffer size
2. Alignment: Hardware requires aligned transfers
3. Granularity: Minimum transfer size constraints
4. Redundancy: Same data fetched multiple times (no tracking)
```

For a tile schedule, the overfetch ratio determines **effective bandwidth**:
```
Effective_BW = Peak_BW / Overfetch_Ratio
```

---

## 9. Tool Requirements for Schedule Analysis

### 9.1 Current Gap Analysis

To enable confident schedule development for the KPU, we need tools that answer:

1. **Feasibility**: Can this schedule achieve 100% compute utilization?
2. **Optimality**: Is this the minimum data movement schedule?
3. **Sensitivity**: How does performance change with buffer sizes?
4. **Bottleneck**: Which resource limits throughput?
5. **Comparison**: How does this schedule compare to alternatives?

### 9.2 Proposed Tool Suite

#### 9.2.1 Operator Analyzer

**Purpose:** Analyze a single operator (e.g., GEMM) and compute theoretical bounds.

**Inputs:**
- Operator specification (dimensions, data types)
- Hardware configuration (buffer sizes, bandwidths)

**Outputs:**
```
Theoretical Analysis:
  Minimum data movement: X bytes (Hong-Kung bound)
  Optimal tile size: Ti×Tj×Tk
  Arithmetic intensity: Y FLOP/byte
  Bottleneck prediction: compute-bound / memory-bound

For each memory level:
  Working set requirement: W bytes
  Fits in level: yes/no
  Expected reuse factor: R
```

#### 9.2.2 Schedule Simulator

**Purpose:** Simulate a specific tile schedule and measure actual behavior.

**Inputs:**
- Operator specification
- Tile schedule (order, timing, buffer assignments)
- Hardware configuration

**Outputs:**
```
Execution Trace:
  Total cycles: N
  Compute cycles: C (utilization = C/N)
  Stall cycles: S (breakdown by cause)

Data Movement:
  DRAM → L3: X bytes (overfetch ratio: r1)
  L3 → L2: Y bytes (overfetch ratio: r2)
  L2 → L1: Z bytes (overfetch ratio: r3)

Reuse Statistics:
  L3 tile reuse: R3 (times each tile used before eviction)
  L2 tile reuse: R2
  L1 element reuse: R1

Buffer Utilization:
  L3: peak/average/timeline
  L2: peak/average/timeline
  L1: peak/average/timeline
```

#### 9.2.3 Schedule Comparator

**Purpose:** Compare multiple schedules for the same operator.

**Outputs:**
```
┌─────────────────────────────────────────────────────────────┐
│ Schedule Comparison: GEMM 4096×4096×4096                    │
├──────────────┬──────────┬──────────┬──────────┬─────────────┤
│ Metric       │ Sched A  │ Sched B  │ Sched C  │ Optimal     │
├──────────────┼──────────┼──────────┼──────────┼─────────────┤
│ Cycles       │ 1.2M     │ 1.1M     │ 1.05M    │ 1.0M        │
│ Utilization  │ 83%      │ 91%      │ 95%      │ 100%        │
│ DRAM traffic │ 150MB    │ 140MB    │ 135MB    │ 128MB       │
│ Overfetch    │ 1.17×    │ 1.09×    │ 1.05×    │ 1.00×       │
│ L3 reuse     │ 2.1×     │ 2.8×     │ 3.2×     │ 4.0×        │
└──────────────┴──────────┴──────────┴──────────┴─────────────┘
```

#### 9.2.4 Reuse Distance Profiler

**Purpose:** Compute reuse distance distribution for a schedule.

**Outputs:**
```
Reuse Distance Distribution:
  Distance 0-100: 45% of accesses (L1 hits)
  Distance 100-1K: 30% of accesses (L2 hits)
  Distance 1K-10K: 20% of accesses (L3 hits)
  Distance >10K: 5% of accesses (DRAM)

Miss Ratio Curve (MRC):
  [Graph: miss rate vs. buffer size]

Working Set Curve:
  [Graph: working set size vs. time]
```

#### 9.2.5 Schedule Synthesizer

**Purpose:** Automatically generate schedules meeting constraints.

**Inputs:**
- Operator specification
- Performance target (utilization, latency)
- Resource constraints (buffer sizes)

**Algorithm:**
1. Compute Hong-Kung lower bound
2. Enumerate feasible tile sizes
3. For each tile size, find optimal loop order (polyhedral analysis)
4. Simulate candidate schedules
5. Return Pareto-optimal set (utilization vs. buffer usage)

### 9.3 Integration with Compiler

The tools should integrate with the KPU compiler:

```
Source Code (DNN operator)
        │
        ▼
┌─────────────────────┐
│   Operator Analyzer │ ← Theoretical bounds
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Schedule Synthesizer│ ← Candidate schedules
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Schedule Simulator │ ← Detailed simulation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Schedule Comparator│ ← Best schedule selection
└─────────┬───────────┘
          │
          ▼
    Code Generation
```

---

## 10. Research Directions

### 10.1 Open Problems

1. **Optimal schedule synthesis**: Polynomial-time algorithms for restricted cases
2. **Multi-operator scheduling**: Fusing operators with different tile sizes
3. **Dynamic adaptation**: Runtime schedule modification based on actual behavior
4. **Heterogeneous memory**: Different buffer types (SRAM, HBM, etc.)

### 10.2 Relevant Ongoing Research

**Polyhedral compilation** [26, 27] provides mathematical frameworks for:
- Automatic tiling
- Loop fusion
- Data layout optimization

**Machine learning for compilers** [28, 29] applies ML to:
- Tile size selection
- Schedule search
- Performance prediction

### 10.3 KPU-Specific Research Needs

1. **Credit-based flow scheduling**: Optimal credit issuance timing
2. **Multi-level double buffering**: Coordinated buffering across L3/L2/L1
3. **Foundation model patterns**: KV-cache, attention, long sequences
4. **Mixed precision**: Different data types at different levels

---

## References

[1] W. A. Wulf and S. A. McKee, "Hitting the Memory Wall: Implications of the Obvious," *Computer Architecture News*, vol. 23, no. 1, pp. 20-24, 1995.

[2] J. W. Hong and H. T. Kung, "I/O Complexity: The Red-Blue Pebble Game," *Proceedings of the 13th Annual ACM Symposium on Theory of Computing (STOC)*, pp. 326-333, 1981.

[3] J. E. Savage and J. S. Vitter, "Parallelism in Space-Time Tradeoffs," *Advances in Computing Research*, vol. 4, pp. 117-146, 1987.

[4] G. Bilardi and F. P. Preparata, "Processor-Time Tradeoffs under Bounded-Speed Message Propagation," *Theory of Computing Systems*, vol. 32, pp. 531-559, 1999.

[5] G. Ballard, J. Demmel, O. Holtz, and O. Schwartz, "Minimizing Communication in Numerical Linear Algebra," *SIAM Journal on Matrix Analysis and Applications*, vol. 32, no. 3, pp. 866-901, 2011.

[6] J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, "Communication-Optimal Parallel and Sequential QR and LU Factorizations," *SIAM Journal on Scientific Computing*, vol. 34, no. 1, pp. A206-A239, 2012.

[7] J. Demmel, D. Eliahu, A. Fox, S. Kamil, B. Lipshitz, O. Schwartz, and O. Spillinger, "Communication-Optimal Parallel Recursive Rectangular Matrix Multiplication," *IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, 2013.

[8] E. Solomonik and J. Demmel, "Communication-Optimal Parallel 2.5D Matrix Multiplication and LU Factorization Algorithms," *Proceedings of the 17th International Conference on Parallel Processing (Euro-Par)*, pp. 90-109, 2011.

[9] P. J. Denning, "The Working Set Model for Program Behavior," *Communications of the ACM*, vol. 11, no. 5, pp. 323-333, 1968.

[10] R. L. Mattson, J. Gecsei, D. R. Slutz, and I. L. Traiger, "Evaluation Techniques for Storage Hierarchies," *IBM Systems Journal*, vol. 9, no. 2, pp. 78-117, 1970.

[11] C. Ding and Y. Zhong, "Predicting Whole-Program Locality through Reuse Distance Analysis," *ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, pp. 245-257, 2003.

[12] M. S. Lam, E. E. Rothberg, and M. E. Wolf, "The Cache Performance and Optimizations of Blocked Algorithms," *Proceedings of the 4th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)*, pp. 63-74, 1991.

[13] S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65-76, 2009.

[14] A. Ilic, F. Pratas, and L. Sousa, "Cache-Aware Roofline Model: Upgrading the Loft," *IEEE Computer Architecture Letters*, vol. 13, no. 1, pp. 21-24, 2014.

[15] J. A. Kahle, M. N. Day, H. P. Hofstee, C. R. Johns, T. R. Maeurer, and D. Shippy, "Introduction to the Cell Multiprocessor," *IBM Journal of Research and Development*, vol. 49, no. 4/5, pp. 589-604, 2005.

[16] M. Kistler, M. Perrone, and F. Petrini, "Cell Multiprocessor Communication Network: Built for Speed," *IEEE Micro*, vol. 26, no. 3, pp. 10-23, 2006.

[17] A. Buttari, J. Dongarra, J. Kurzak, J. Langou, P. Luszczek, and S. Tomov, "The Impact of Multicore on Math Software," *Proceedings of the 8th International Conference on Applied Parallel Computing (PARA)*, pp. 1-10, 2006.

[18] R. Banakar, S. Steinke, B.-S. Lee, M. Balakrishnan, and P. Marwedel, "Scratchpad Memory: A Design Alternative for Cache On-chip Memory in Embedded Systems," *Proceedings of the 10th International Symposium on Hardware/Software Codesign (CODES)*, pp. 73-78, 2002.

[19] S. Udayakumaran and R. Barua, "Compiler-Decided Dynamic Memory Allocation for Scratch-Pad Based Embedded Systems," *International Conference on Compilers, Architecture, and Synthesis for Embedded Systems (CASES)*, pp. 276-286, 2003.

[20] L. G. Valiant, "A Bridging Model for Parallel Computation," *Communications of the ACM*, vol. 33, no. 8, pp. 103-111, 1990.

[21] D. E. Culler, R. M. Karp, D. Patterson, A. Sahay, K. E. Schauser, E. Santos, R. Subramonian, and T. von Eicken, "LogP: Towards a Realistic Model of Parallel Computation," *ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP)*, pp. 1-12, 1993.

[22] A. Alexandrov, M. F. Ionescu, K. E. Schauser, and C. Scheiman, "LogGP: Incorporating Long Messages into the LogP Model," *Journal of Parallel and Distributed Computing*, vol. 44, no. 1, pp. 71-79, 1997.

[23] High Performance Fortran Forum, "High Performance Fortran Language Specification Version 2.0," *Rice University Technical Report*, 1997.

[24] L. S. Blackford, J. Choi, A. Cleary, E. D'Azevedo, J. Demmel, I. Dhillon, J. Dongarra, S. Hammarling, G. Henry, A. Petitet, K. Stanley, D. Walker, and R. C. Whaley, "ScaLAPACK Users' Guide," *SIAM*, 1997.

[25] S. Chatterjee, J. R. Gilbert, R. Schreiber, and S.-H. Teng, "Automatic Array Alignment in Data-Parallel Programs," *ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL)*, pp. 16-28, 1993.

[26] U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan, "A Practical Automatic Polyhedral Parallelizer and Locality Optimizer," *ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, pp. 101-113, 2008.

[27] S. Verdoolaege, J. Carlos Juega, A. Cohen, J. Ignacio Gómez, C. Tenllado, and F. Catthoor, "Polyhedral Parallel Code Generation for CUDA," *ACM Transactions on Architecture and Code Optimization*, vol. 9, no. 4, Article 54, 2013.

[28] C. Cummins, P. Petoumenos, Z. Wang, and H. Leather, "End-to-End Deep Learning of Optimization Heuristics," *International Conference on Parallel Architectures and Compilation Techniques (PACT)*, pp. 219-232, 2017.

[29] A. H. Ashouri, W. Killian, J. Cavazos, G. Palermo, and C. Silvano, "A Survey on Compiler Autotuning using Machine Learning," *ACM Computing Surveys*, vol. 51, no. 5, Article 96, 2018.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Arithmetic Intensity (AI)** | FLOPs per byte of memory traffic |
| **BSP** | Bulk Synchronous Parallel model |
| **Communication-avoiding** | Algorithms that minimize data movement |
| **I/O Complexity** | Theoretical study of memory access requirements |
| **MRC** | Miss Ratio Curve |
| **Overfetch** | Data moved but not used in computation |
| **Reuse distance** | Unique accesses between repeated access to same location |
| **Ridge point** | AI where memory-bound transitions to compute-bound |
| **S-span** | Minimum I/O operations for a computation DAG |
| **Working set** | Set of memory locations accessed in a time window |

---

## Appendix B: Hong-Kung Bound Derivation

For completeness, we sketch the Hong-Kung lower bound proof for matrix multiplication.

**Setup:** Multiply C = A × B where A, B, C are n × n matrices. Fast memory has M words.

**Key Lemma:** In any valid computation, each multiplication aᵢₖ × bₖⱼ must be performed, and both operands must be simultaneously present in fast memory.

**Observation:** Consider a "segment" of computation where fast memory contents change by at most S loads/stores. During this segment, at most:
- √(2MS) distinct elements of A can be present
- √(2MS) distinct elements of B can be present

**Counting:** The number of multiplications performable in this segment is at most 2MS (pigeonhole argument on which aᵢₖ and bₖⱼ pairs can co-exist).

**Result:** Total multiplications = n³, each segment does ≤ 2MS, so number of segments ≥ n³/(2MS). Each segment has S I/O operations, giving total I/O ≥ n³/(2M) = Ω(n³/M).

**Refinement:** Tighter analysis gives Ω(n³/√M), achieved by optimal blocking.

---

*Document generated for KPU Simulator v0.3.2*
*Part of the Benchmarking & Observability milestone (v0.3.x)*
