# Weight-Stationary Dataflow Implementation Plan

## Executive Summary

Implementing weight-stationary (WS) dataflow to enable true comparison with output-stationary (OS) and input-stationary (IS) strategies in the Pareto frontier characterization framework.

**Goal**: Enable the characterizer to show REAL differences between dataflow strategies based on their fundamental data movement and reuse patterns.

## Background: Dataflow Comparison

### Output-Stationary (OS) - Currently Implemented

**What stays in PEs**: Partial sums (output accumulators)
**What streams**: Input activations (A) and weights (B)

```
For C[ti,tj] += A[ti,tk] × B[tk,tj]:
  - C tile stays in PE registers (accumulates across K)
  - A tiles streamed from L2 → L1 → PEs (row-wise)
  - B tiles streamed from L2 → L1 → PEs (column-wise)
```

**Best for**:
- Large K dimension (many accumulations)
- Output reuse across K tiles

**Reuse**:
- A: `⌈N/Tj⌉` (reused across output columns)
- B: `⌈M/Ti⌉` (reused across output rows)
- C: `⌈K/Tk⌉` (accumulated across K dimension)

### Weight-Stationary (WS) - To Implement

**What stays in PEs**: Weights (typically matrix B)
**What streams**: Input activations (A) and partial sums (C)

```
For C[ti,tj] += A[ti,tk] × B[tk,tj]:
  - B tile loaded ONCE into PE registers (stays stationary)
  - A tiles streamed repeatedly through PEs
  - C tiles accumulated in L2/L1 (not in PEs)
```

**Best for**:
- Weight reuse (CNNs, attention layers)
- Small weight matrices that fit in PE registers
- Batch processing (reuse weights across batches)

**Reuse**:
- A: `⌈Tj/systolic_cols⌉` (minimal - flows through)
- B: `⌈M/Ti⌉ × ⌈K/Tk⌉` (maximal - stays in PEs)
- C: `⌈K/Tk⌉` (accumulated in memory, not PEs)

### Key Differences: WS vs OS

| Aspect | Output-Stationary | Weight-Stationary |
|--------|-------------------|-------------------|
| **PE register usage** | Accumulators (C) | Weights (B) |
| **Weight loading** | Every K iteration | Once per tile |
| **Output handling** | Accumulate in PEs | Accumulate in memory |
| **Memory pressure** | L2 stores A/B tiles | L2 stores A/C tiles |
| **Best workload** | Large K, small M/N | Large batch, small weights |
| **Energy trade-off** | More weight reads | More accumulator writes |

## Implementation Strategy

### Phase 1: Analysis & Design

#### 1.1 Understand Current OS Implementation

Current flow (OS):
```
TileOptimizer::optimize()
  → picks Ti, Tj, Tk for OS dataflow
  → optimizes for C staying in PEs

L2TileScheduler::generate_schedule()
  → allocates L2 for A and B tiles
  → computes order: for ti, for tj, for tk
  → loads A[ti,tk], B[tk,tj] as needed
  → C[ti,tj] assumed in PE registers
```

#### 1.2 Design WS Modifications

Key insight: **Invert the loop order and tile allocation**

WS needs:
```
TileOptimizer::optimize_weight_stationary()
  → picks Ti, Tj, Tk optimized for B staying in PEs
  → Constraint: Tk × Tj must fit in PE register file
  → Maximize B reuse: prefer larger Tk, Tj

L2TileScheduler::generate_schedule_ws()
  → allocates L2 for A and C tiles (NOT B!)
  → computes order: for tk, for tj, for ti
    - Load B[tk,tj] ONCE into PEs
    - Stream A[ti,tk] for all ti
    - Accumulate C[ti,tj] in L2
  → B tiles loaded once, stay in PEs
  → C tiles accumulate in L2 (not PE registers)
```

### Phase 2: Tile Optimizer for WS

#### 2.1 Constraint Changes

**OS Constraint** (current):
```cpp
Ti × Tk + Tk × Tj + Ti × Tj ≤ L2_capacity
// C[Ti,Tj] fits in PE registers (free)
```

**WS Constraint** (new):
```cpp
Ti × Tk + Ti × Tj ≤ L2_capacity  // A and C tiles in L2
Tk × Tj ≤ PE_register_capacity   // B tile in PE registers!
```

#### 2.2 Optimization Objective

**OS objective**: Minimize DRAM accesses by maximizing C reuse in PEs

**WS objective**: Minimize DRAM accesses by maximizing B reuse in PEs

```cpp
// WS reuse factors
reuse_B_ws = (M/Ti) × (K/Tk)  // B reused across all A tiles
reuse_A_ws = Tj / systolic_cols  // A flows through
reuse_C_ws = K / Tk  // C accumulated in L2
```

#### 2.3 New Tile Selection Algorithm

```cpp
TileOptimizer::TileConfig TileOptimizer::optimize_weight_stationary(
    Size M, Size N, Size K)
{
    // Start with PE constraint: B must fit in registers
    Size PE_capacity = systolic_rows * systolic_cols * bytes_per_PE_register;

    // Step 1: Find maximum Tk × Tj that fits in PEs
    Size Tj_max = std::min(N, systolic_cols * max_vector_length);
    Size Tk_max = std::min(K, PE_capacity / (Tj_max * element_size));

    // Step 2: Find Ti given L2 capacity for A and C
    // A[Ti, Tk] + C[Ti, Tj] ≤ L2_capacity
    Size Ti_max = L2_capacity / ((Tk_max + Tj_max) * element_size);

    // Step 3: Round to systolic array multiples
    Ti = round_to_multiple(Ti_max, systolic_rows);
    Tj = round_to_multiple(Tj_max, systolic_cols);
    Tk = round_to_multiple(Tk_max, systolic_rows);

    // Step 4: Calculate reuse
    config.reuse_A = Tj / systolic_cols;
    config.reuse_B = (M / Ti) * (K / Tk);  // Key difference!
    config.reuse_C = K / Tk;

    return config;
}
```

### Phase 3: L2 Scheduler for WS

#### 3.1 Loop Order Inversion

**OS loop order** (current):
```cpp
for (ti = 0; ti < M/Ti; ti++)
  for (tj = 0; tj < N/Tj; tj++)
    for (tk = 0; tk < K/Tk; tk++)
      C[ti,tj] += A[ti,tk] × B[tk,tj]
      // C accumulates in PEs
```

**WS loop order** (new):
```cpp
for (tk = 0; tk < K/Tk; tk++)       // Outer: weight tiles
  for (tj = 0; tj < N/Tj; tj++)     // Middle: output columns
    LOAD B[tk,tj] INTO PE_REGISTERS  // ← KEY: Load once!
    for (ti = 0; ti < M/Ti; ti++)    // Inner: stream inputs
      STREAM A[ti,tk] THROUGH PEs
      ACCUMULATE C[ti,tj] IN L2      // ← KEY: Not in PEs!
```

#### 3.2 L2 Allocation Changes

**OS allocation** (current):
```cpp
L2_slots = {
  A_tiles[num_A_tiles],  // All A tiles
  B_tiles[num_B_tiles],  // All B tiles
  // C tiles in PE registers (not in L2)
}
```

**WS allocation** (new):
```cpp
L2_slots = {
  A_tiles[num_A_tiles],  // All A tiles
  C_tiles[num_C_tiles],  // All C tiles (accumulate in L2!)
  // B tiles in PE registers (not in L2)
}
```

#### 3.3 Key Differences in Scheduling

| Aspect | Output-Stationary | Weight-Stationary |
|--------|-------------------|-------------------|
| **Outer loop** | ti (output rows) | tk (weight tiles) |
| **B tile lifetime** | One K iteration | All M iterations |
| **C tile location** | PE registers | L2 memory |
| **L2 pressure** | A + B tiles | A + C tiles |
| **Critical path** | A/B → L2 → L1 → PEs | B → PEs (once), A → L1 → PEs |

### Phase 4: Energy & Latency Models for WS

#### 4.1 Energy Model Differences

**OS energy**:
```cpp
// Read A and B from L2 every K iteration
energy_A = num_A_tiles × load_count_A × L2_read_energy;
energy_B = num_B_tiles × load_count_B × L2_read_energy;
energy_C = 0;  // C in PE registers

// MAC energy same
energy_compute = 2 × M × N × K × MAC_energy;
```

**WS energy**:
```cpp
// Read A from L2 (similar to OS)
energy_A = num_A_tiles × load_count_A × L2_read_energy;

// Read B ONCE per tile (major savings!)
energy_B = num_B_tiles × 1 × L2_read_energy;

// Write C to L2 (new cost!)
energy_C = num_C_tiles × (K/Tk) × L2_write_energy;

// MAC energy same
energy_compute = 2 × M × N × K × MAC_energy;
```

**Trade-off**:
- **WS saves**: B reads reduced by `(M/Ti)` factor
- **WS costs**: C writes to L2 instead of accumulating in PEs

#### 4.2 Latency Model Differences

**OS latency**:
```cpp
// Systolic array latency per tile
latency_per_tile = Tk + max(Ti, Tj);

// Total for all tiles
latency_total = (M/Ti) × (N/Tj) × (K/Tk) × latency_per_tile;
```

**WS latency**:
```cpp
// Load B tile into PEs (amortized)
latency_load_B = Tk × Tj / PE_bandwidth;

// Stream A through PEs
latency_per_A = Tk + Ti;

// Total
latency_total = (K/Tk) × (N/Tj) × [
  latency_load_B +
  (M/Ti) × latency_per_A
];
```

**Trade-off**:
- **WS faster if**: M >> N (many batches/rows)
- **OS faster if**: K >> M, N (deep accumulation)

### Phase 5: Integration with Characterizer

#### 5.1 Modify ScheduleCharacterizer::evaluate_schedule()

```cpp
ScheduleEvaluation ScheduleCharacterizer::evaluate_schedule(
    const TensorShape& shape,
    DataflowStrategy strategy)
{
    ScheduleEvaluation eval;
    eval.shape = shape;
    eval.strategy = strategy;

    // *** KEY CHANGE: Select optimizer based on strategy ***
    switch (strategy) {
        case DataflowStrategy::OUTPUT_STATIONARY:
            eval.tile_config = optimizer_.optimize(shape.M, shape.N, shape.K);
            schedule = scheduler_.generate_schedule_os(...);
            break;

        case DataflowStrategy::WEIGHT_STATIONARY:
            eval.tile_config = optimizer_.optimize_weight_stationary(
                shape.M, shape.N, shape.K);
            schedule = scheduler_.generate_schedule_ws(...);
            break;

        case DataflowStrategy::INPUT_STATIONARY:
            // Phase 6 (future)
            break;
    }

    // Calculate energy/latency with strategy-specific models
    eval.metrics.total_energy = calculate_energy_for_strategy(
        shape, schedule, strategy);
    eval.metrics.total_cycles = calculate_latency_for_strategy(
        shape, schedule, strategy);

    return eval;
}
```

### Phase 6: Expected Results & Validation

#### 6.1 Performance Characteristics

**When WS Should Win** (lower energy):
- Small weight matrices: B[K,N] where K×N << M
- Large batch dimension: M >> K, N
- Weight reuse workloads: CNNs, attention

**Example**: BERT Q/K/V projections
- Shape: [128, 768, 768] → batch 128, hidden 768
- **WS**: Load 768×768 weights once, stream 128 inputs
- **OS**: Load weights 128 times (once per batch)

**When OS Should Win** (lower energy):
- Large K dimension: K >> M, N
- Accumulation-heavy: Many K tiles
- Small batches: M ≈ 1

**Example**: Deep MLP layers
- Shape: [1, 4096, 4096] → single batch, large hidden
- **OS**: Accumulate in PEs across 4096 dimension
- **WS**: Must write partial sums to L2 repeatedly

#### 6.2 Validation Tests

```cpp
TEST_CASE("WS vs OS - Batch Processing") {
    // Large batch, small weights
    TensorShape shape(256, 64, 64);  // M >> N, K

    auto ws_eval = characterizer.evaluate_schedule(
        shape, DataflowStrategy::WEIGHT_STATIONARY);
    auto os_eval = characterizer.evaluate_schedule(
        shape, DataflowStrategy::OUTPUT_STATIONARY);

    // WS should have lower energy (weight reuse)
    REQUIRE(ws_eval.metrics.total_energy < os_eval.metrics.total_energy);

    // WS should have higher B reuse
    REQUIRE(ws_eval.metrics.reuse_B > os_eval.metrics.reuse_B * 2);
}

TEST_CASE("WS vs OS - Deep Accumulation") {
    // Large K, small batch
    TensorShape shape(1, 512, 4096);  // K >> M, N

    auto ws_eval = characterizer.evaluate_schedule(
        shape, DataflowStrategy::WEIGHT_STATIONARY);
    auto os_eval = characterizer.evaluate_schedule(
        shape, DataflowStrategy::OUTPUT_STATIONARY);

    // OS should have lower energy (C reuse in PEs)
    REQUIRE(os_eval.metrics.total_energy < ws_eval.metrics.total_energy);
}
```

## Implementation Checklist

### Week 1: Foundation
- [ ] Add `optimize_weight_stationary()` to TileOptimizer
- [ ] Implement WS tile size selection with PE capacity constraint
- [ ] Calculate WS-specific reuse factors
- [ ] Unit tests for WS tile optimizer

### Week 2: L2 Scheduling
- [ ] Add `generate_schedule_ws()` to L2TileScheduler
- [ ] Implement WS loop order (tk → tj → ti)
- [ ] Update L2 allocation for A+C tiles (not A+B)
- [ ] Track B tile lifetime in PE registers
- [ ] Unit tests for WS L2 scheduler

### Week 3: Performance Models
- [ ] Implement `calculate_energy_ws()` with C write costs
- [ ] Implement `calculate_latency_ws()` with different timing
- [ ] Add WS-specific metrics to PerformanceMetrics
- [ ] Validate energy/latency calculations

### Week 4: Integration & Testing
- [ ] Update ScheduleCharacterizer to dispatch on strategy
- [ ] Run characterization with WS enabled
- [ ] Verify Pareto frontier shows WS vs OS differences
- [ ] Validate expected performance characteristics
- [ ] Write comprehensive test suite

## Success Criteria

1. **Differentiation**: WS produces measurably different results from OS
2. **Physical validity**: Energy/latency differences match dataflow theory
3. **Performance patterns**: WS wins on batch workloads, OS wins on accumulation
4. **Pareto diversity**: Frontier contains both WS and OS points
5. **Coverage**: ~10-20% of workloads favor WS over OS

## Future: Input-Stationary (Phase 7)

After WS is complete, IS follows similar pattern:
- **What stays**: Input activations (A)
- **What streams**: Weights (B), outputs (C)
- **Best for**: Small input, large weight fanout
- **Implementation**: Similar to WS but with A stationary instead of B

## References

- Eyeriss (Chen et al., 2016): Row-stationary (hybrid of WS/OS)
- TPU (Jouppi et al., 2017): Weight-stationary for inference
- Simba (Shao et al., 2019): Flexible dataflow comparison
- Timeloop (Parashar et al., 2019): Analytical dataflow modeling
