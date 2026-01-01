# NoC Benchmarking and Characterization Framework

## Executive Summary

This document proposes a comprehensive framework for benchmarking and characterizing the KPU Network-on-Chip (NoC) implementations. The goal is to evaluate how well the **WormholeNoC** and **DataflowNoC** implementations support the data movement patterns required by DNN operators.

## DNN Operator Analysis

### Core Operators and Data Movement Patterns

| Operator | Primary Data Pattern | Secondary Pattern | Communication Intensity |
|----------|---------------------|-------------------|------------------------|
| **MatMul/Linear** | Systolic flow (A→East, B→South) | Broadcast weights | High |
| **MLP** | Same as MatMul + bias broadcast | Activation local | High |
| **Conv2D** | im2col + MatMul OR sliding window | Filter broadcast | High |
| **Conv3D** | 3D sliding window | Filter broadcast | Very High |
| **Pooling** | Local reduction | Minimal cross-tile | Low |
| **BatchNorm** | Reduce across batch | Broadcast mean/var | Medium |
| **LayerNorm** | Reduce across features | Local normalize | Low-Medium |
| **SoftMax** | Reduce max, reduce sum | Local exp/div | Medium |
| **Attention** | Multiple MatMuls + SoftMax | See composite analysis | Very High |

### Attention Block Decomposition

The Attention mechanism is the most communication-intensive composite operator:

```
Input: X [batch, seq_len, d_model]

Phase 1: QKV Projections (3 parallel MatMuls)
    Q = X @ W_Q   [batch, seq_len, d_k]      ← MatMul: d_model × d_k
    K = X @ W_K   [batch, seq_len, d_k]      ← MatMul: d_model × d_k
    V = X @ W_V   [batch, seq_len, d_v]      ← MatMul: d_model × d_v

Phase 2: Attention Scores
    Scores = Q @ K^T   [batch, seq_len, seq_len]   ← MatMul: seq_len × seq_len
    Scores = Scores / sqrt(d_k)                     ← Elementwise scale

Phase 3: SoftMax (per-row reduction)
    max_scores = reduce_max(Scores, dim=-1)         ← Reduce max
    exp_scores = exp(Scores - max_scores)           ← Elementwise exp
    sum_scores = reduce_sum(exp_scores, dim=-1)     ← Reduce sum
    Attention_weights = exp_scores / sum_scores     ← Elementwise div

Phase 4: Weighted Sum
    Output = Attention_weights @ V  [batch, seq_len, d_v]  ← MatMul

Phase 5: Output Projection (optional)
    Output = Output @ W_O           [batch, seq_len, d_model]  ← MatMul
```

**Multi-Head Attention adds:**
- Head splitting (reshape/partition)
- Parallel execution of h heads
- Head concatenation (gather)
- Final projection

**Data Movement Requirements:**
1. **Broadcast**: Input X to all compute tiles
2. **Reduce**: Max and sum for SoftMax (all-reduce pattern)
3. **All-to-All**: Q @ K^T requires all Q tiles to interact with all K tiles
4. **Scatter/Gather**: Head splitting and concatenation

### FlashAttention Insights

From [FlashAttention research](https://arxiv.org/abs/2205.14135), the key insight is **IO-awareness**:

- Standard attention: O(N²) memory accesses to main memory
- FlashAttention: O(N·d) memory accesses via tiling
- **Implication for NoC**: Tile-based communication patterns reduce total NoC traffic

The FlashAttention tiling strategy:
1. Load Q, K, V blocks that fit in on-chip memory (L3 in KPU)
2. Compute partial attention within blocks
3. Accumulate results using online softmax
4. Never materialize full N×N attention matrix

---

## Proposed Benchmarking Framework

### 1. Microbenchmarks (Low-Level NoC Characterization)

#### 1.1 Latency Benchmarks
```cpp
// Single-flit latency (minimum transfer unit)
struct LatencyBenchmark {
    // Measure cycle count for single 64-byte flit
    // Vary: hop_count (1, 2, 4, 8 hops)
    uint64_t measure_single_flit_latency(uint8_t src, uint8_t dst);

    // Measure tile transfer latency
    // Vary: tile_size (64B, 256B, 1KB, 4KB, 16KB, 64KB)
    uint64_t measure_tile_latency(uint8_t src, uint8_t dst, uint32_t size);
};
```

**Metrics:**
- `min_latency`: Best-case single-flit latency
- `latency_per_hop`: Latency increase per router hop
- `tile_latency(size)`: Latency as function of tile size

#### 1.2 Throughput Benchmarks
```cpp
struct ThroughputBenchmark {
    // Sustained bandwidth between two routers
    // Inject continuously, measure delivered flits/cycle
    double measure_point_to_point_bandwidth(uint8_t src, uint8_t dst);

    // Aggregate bandwidth with multiple concurrent transfers
    double measure_aggregate_bandwidth(
        const std::vector<std::pair<uint8_t, uint8_t>>& transfers);

    // Bisection bandwidth (half of mesh talks to other half)
    double measure_bisection_bandwidth();
};
```

**Metrics:**
- `peak_bandwidth`: Maximum bytes/cycle achieved
- `sustained_bandwidth`: Average over long transfer
- `bisection_bandwidth`: Cross-mesh aggregate bandwidth

#### 1.3 Contention Benchmarks
```cpp
struct ContentionBenchmark {
    // Multiple sources to single destination
    double measure_hot_spot_contention(
        const std::vector<uint8_t>& sources, uint8_t dst);

    // Crossing traffic at center routers
    double measure_center_congestion();

    // Back-pressure propagation
    uint64_t measure_backpressure_depth(uint8_t src, uint8_t dst);
};
```

**Metrics:**
- `contention_slowdown`: Bandwidth reduction under contention
- `fairness`: Relative bandwidth per source under contention
- `backpressure_cycles`: Cycles stalled due to downstream congestion

### 2. Pattern Benchmarks (Communication Patterns)

#### 2.1 Broadcast Pattern
```
One source sends same data to multiple destinations
Use case: Weight distribution, bias broadcast
```

```cpp
struct BroadcastBenchmark {
    // Single source to all routers
    Result measure_broadcast_all(uint8_t src, uint32_t tile_size);

    // Single source to row/column
    Result measure_broadcast_row(uint8_t src, uint8_t row, uint32_t tile_size);
    Result measure_broadcast_col(uint8_t src, uint8_t col, uint32_t tile_size);
};
```

#### 2.2 Reduce Pattern
```
Multiple sources combine data at single destination
Use case: BatchNorm mean/var, SoftMax sum, gradient aggregation
```

```cpp
struct ReduceBenchmark {
    // All routers reduce to one
    Result measure_reduce_all(uint8_t dst, uint32_t tile_size);

    // Tree reduction
    Result measure_reduce_tree(uint8_t dst, uint32_t tile_size);

    // All-reduce (reduce then broadcast)
    Result measure_allreduce(uint32_t tile_size);
};
```

#### 2.3 Systolic Flow Pattern
```
Data flows through mesh in structured pattern
Use case: Matrix multiplication, convolution
```

```cpp
struct SystolicBenchmark {
    // East flow (A tiles for matmul)
    Result measure_east_flow(uint8_t start_col, uint32_t tile_size);

    // South flow (B tiles for matmul)
    Result measure_south_flow(uint8_t start_row, uint32_t tile_size);

    // Combined systolic (both flows simultaneously)
    Result measure_systolic_flow(
        uint32_t a_tile_size, uint32_t b_tile_size);
};
```

#### 2.4 All-to-All Pattern
```
Every router sends to every other router
Use case: Attention Q@K^T, large gather operations
```

```cpp
struct AllToAllBenchmark {
    // Full mesh all-to-all
    Result measure_all_to_all(uint32_t tile_size);

    // Permutation (each router sends to one unique destination)
    Result measure_permutation(const std::vector<uint8_t>& mapping);
};
```

### 3. Operator Benchmarks (DNN-Specific)

#### 3.1 MatMul/Linear Benchmark
```cpp
struct MatMulBenchmark {
    struct Config {
        uint32_t M, N, K;           // Problem size
        uint32_t Ti, Tj, Tk;        // Tile sizes
        DataflowPattern dataflow;   // output-stationary, etc.
    };

    struct Result {
        uint64_t total_cycles;
        uint64_t compute_cycles;
        uint64_t noc_cycles;        // Time spent waiting for data
        double noc_utilization;     // NoC bandwidth utilization
        double compute_utilization; // Compute unit utilization
    };

    Result run(const Config& config);
};
```

**Sweep parameters:**
- Problem sizes: 512×512, 1024×1024, 2048×2048, 4096×4096
- Tile sizes: 64, 128, 256 (constrained by L3 capacity)
- Dataflow: output-stationary, weight-stationary, input-stationary

#### 3.2 Attention Benchmark
```cpp
struct AttentionBenchmark {
    struct Config {
        uint32_t batch_size;
        uint32_t seq_len;
        uint32_t d_model;
        uint32_t num_heads;
        bool use_flash_attention;   // Tiled algorithm
    };

    struct PhaseResult {
        std::string name;
        uint64_t cycles;
        uint64_t noc_transfers;
        double noc_utilization;
    };

    struct Result {
        std::vector<PhaseResult> phases;
        uint64_t total_cycles;
        double overall_noc_utilization;
        double compute_utilization;
    };

    Result run(const Config& config);
};
```

**Phases measured separately:**
1. QKV projection
2. Q @ K^T computation
3. SoftMax reduction
4. Attention @ V computation
5. Output projection

#### 3.3 Convolution Benchmark
```cpp
struct ConvolutionBenchmark {
    struct Config {
        uint32_t batch, in_channels, out_channels;
        uint32_t H, W;              // Input spatial dimensions
        uint32_t kH, kW;            // Kernel size
        uint32_t stride, padding;
        bool use_im2col;            // Transform to matmul
    };

    Result run(const Config& config);
};
```

#### 3.4 Reduction Operator Benchmarks
```cpp
struct ReductionBenchmark {
    // SoftMax: max reduction + exp + sum reduction + div
    Result run_softmax(uint32_t rows, uint32_t cols);

    // BatchNorm: mean + var reduction, normalize
    Result run_batchnorm(uint32_t batch, uint32_t channels, uint32_t HW);

    // LayerNorm: per-sample normalization
    Result run_layernorm(uint32_t batch, uint32_t features);

    // Pooling: local reduction
    Result run_pooling(uint32_t H, uint32_t W, uint32_t pool_size);
};
```

### 4. Composite Benchmarks (Full Model Traces)

```cpp
struct ModelBenchmark {
    // Transformer layer (Attention + FFN)
    Result run_transformer_layer(
        uint32_t batch, uint32_t seq_len, uint32_t d_model,
        uint32_t d_ff, uint32_t num_heads);

    // MLP block (multiple Linear + Activation layers)
    Result run_mlp_block(
        const std::vector<uint32_t>& layer_sizes);

    // ConvNet block (Conv + Pool + Norm)
    Result run_convnet_block(
        uint32_t in_channels, uint32_t out_channels,
        uint32_t H, uint32_t W);
};
```

---

## Benchmark Infrastructure

### 1. NoCBenchmarkHarness

```cpp
class NoCBenchmarkHarness {
public:
    struct Config {
        NoCType noc_type;           // WORMHOLE or DATAFLOW
        NoCConfig noc_config;       // Mesh dimensions, buffer depth
        uint32_t warmup_cycles;     // Cycles before measurement
        uint32_t measurement_cycles;// Cycles to measure
        uint32_t repetitions;       // For statistical significance
    };

    NoCBenchmarkHarness(const Config& config);

    // Run all microbenchmarks
    MicroBenchmarkResults run_microbenchmarks();

    // Run all pattern benchmarks
    PatternBenchmarkResults run_pattern_benchmarks();

    // Run operator benchmarks
    OperatorBenchmarkResults run_operator_benchmarks();

    // Compare two NoC types
    ComparisonReport compare_noc_types(
        NoCType baseline, NoCType candidate);

private:
    std::unique_ptr<INoC> noc_;
    Config config_;
};
```

### 2. Result Structures

```cpp
struct BenchmarkResult {
    std::string name;
    std::string config;

    // Timing
    uint64_t cycles;
    uint64_t min_cycles;
    uint64_t max_cycles;
    double stddev_cycles;

    // Throughput
    double bytes_per_cycle;
    double peak_bandwidth_utilization;

    // Efficiency
    double noc_efficiency;          // Actual / theoretical bandwidth
    double compute_to_noc_ratio;    // Time computing / time moving data

    // Detailed stats
    NoCStats noc_stats;
};

struct ComparisonReport {
    std::string baseline_name;
    std::string candidate_name;

    struct Comparison {
        std::string benchmark_name;
        double baseline_value;
        double candidate_value;
        double speedup;             // candidate / baseline
        double improvement_pct;
    };

    std::vector<Comparison> comparisons;

    std::string to_markdown() const;
    std::string to_csv() const;
};
```

### 3. Visualization Outputs

```cpp
class BenchmarkVisualizer {
public:
    // Heatmap of router-to-router latencies
    void generate_latency_heatmap(
        const LatencyBenchmarkResults& results,
        const std::string& output_path);

    // Bandwidth utilization over time
    void generate_bandwidth_timeline(
        const ThroughputBenchmarkResults& results,
        const std::string& output_path);

    // Roofline-style plot for operators
    void generate_noc_roofline(
        const OperatorBenchmarkResults& results,
        const std::string& output_path);

    // Comparison bar charts
    void generate_comparison_chart(
        const ComparisonReport& report,
        const std::string& output_path);
};
```

---

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `include/sw/benchmark/noc_benchmark.hpp` - Benchmark harness interface
2. Create `src/benchmark/noc_benchmark.cpp` - Harness implementation
3. Create `tools/benchmark/kpu-noc-bench/main.cpp` - CLI tool

### Phase 2: Microbenchmarks
1. Implement latency benchmarks
2. Implement throughput benchmarks
3. Implement contention benchmarks
4. Add statistical analysis (min/max/mean/stddev)

### Phase 3: Pattern Benchmarks
1. Implement broadcast pattern
2. Implement reduce pattern
3. Implement systolic flow pattern
4. Implement all-to-all pattern

### Phase 4: Operator Benchmarks
1. MatMul/Linear benchmark
2. Attention benchmark (with phase breakdown)
3. Convolution benchmark
4. Reduction operator benchmarks

### Phase 5: Analysis and Visualization
1. Comparison report generation
2. Heatmap generation (using Python/matplotlib or gnuplot)
3. Timeline visualization
4. Integration with existing roofline tools

---

## Expected Metrics

### Microbenchmark Targets

| Metric | WormholeNoC Expected | DataflowNoC Expected |
|--------|---------------------|---------------------|
| Single-hop latency | 1-2 cycles | 1-2 cycles |
| Multi-hop latency | Not supported | ~1 cycle/hop |
| Peak bandwidth | 64 bytes/cycle | 64 bytes/cycle |
| Bisection bandwidth | N/A (single-hop) | 32 bytes/cycle (4×4) |

### Pattern Benchmark Targets

| Pattern | Key Metric | Target |
|---------|-----------|--------|
| Broadcast | All receive time | O(N) hops + tile_size/BW |
| Reduce | Aggregation time | O(log N) for tree |
| Systolic | Steady-state throughput | Near-peak compute |
| All-to-All | Completion time | O(N²) tiles / BW |

### Operator Benchmark Targets

| Operator | NoC Bound? | Target Efficiency |
|----------|-----------|------------------|
| MatMul (large) | Compute | >90% compute utilization |
| MatMul (small) | NoC | >50% NoC utilization |
| Attention | Mixed | Phase-dependent |
| Conv2D | Compute | >80% compute utilization |
| BatchNorm | NoC | >50% NoC utilization |

---

## File Structure

```
include/
└── sw/
    └── benchmark/
        ├── noc_benchmark.hpp           # Main harness
        ├── noc_microbenchmarks.hpp     # Latency/throughput/contention
        ├── noc_pattern_benchmarks.hpp  # Communication patterns
        └── noc_operator_benchmarks.hpp # DNN operators

src/
└── benchmark/
    ├── noc_benchmark.cpp
    ├── noc_microbenchmarks.cpp
    ├── noc_pattern_benchmarks.cpp
    └── noc_operator_benchmarks.cpp

tools/
└── benchmark/
    └── kpu-noc-bench/
        └── main.cpp

tests/
└── benchmark/
    └── noc_benchmark_test.cpp
```

---

## Command Line Interface

```bash
# Run all benchmarks
kpu-noc-bench --all --noc dataflow --output results.json

# Run specific benchmark category
kpu-noc-bench --microbenchmarks --noc wormhole
kpu-noc-bench --patterns --noc dataflow

# Compare NoC implementations
kpu-noc-bench --compare wormhole dataflow --output comparison.md

# Run operator benchmarks with specific sizes
kpu-noc-bench --operator matmul --M 1024 --N 1024 --K 1024

# Run attention benchmark
kpu-noc-bench --operator attention --seq 512 --d_model 768 --heads 12

# Generate visualizations
kpu-noc-bench --visualize results.json --output plots/
```

---

## References

### Attention Mechanism
- [Transformer Attention Guide](https://www.billparker.ai/2024/10/transformer-attention-simple-guide-to-q.html)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-3 (2024)](https://tridao.me/blog/2024/flash3/)

### NoC for DNN Accelerators
- [NoCDAS Simulator](https://dl.acm.org/doi/10.1145/3729169)
- [Tree-based Multicast for DNN (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0026269224000739)
- [HT-NoC Reconfigurable Architecture](https://link.springer.com/chapter/10.1007/978-3-031-87995-1_2)
