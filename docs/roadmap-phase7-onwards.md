# KPU Simulator Roadmap: Phase 7 Onwards

## Current State Summary (December 2025)

### Completed Phases (All 6 from the original implementation plan)

| Phase | Status | Key Deliverables |
|-------|--------|------------------|
| 1. Resource API | ✅ Complete | ResourceManager, ResourceHandle, memory allocation |
| 2. Kernel Abstraction | ✅ Complete | Kernel class, KernelCompiler, auto-tiling |
| 3. Runtime Library | ✅ Complete | KPURuntime, GraphExecutor, CUDA-like API |
| 4. Serialization | ✅ Complete | ProgramSerializer, KernelSerializer (binary + JSON) |
| 5. Multi-Kernel Graphs | ✅ Complete | KernelGraph, topological sort, fusion detection |
| 6. Python Integration | ✅ Complete | Full pybind11 bindings, test suites |

### Codebase Metrics
- **47 test files** with comprehensive coverage
- **52 source files** implementing the simulator
- **60 header files** defining the APIs
- **50+ documentation files** covering architecture and usage
- **CI/CD** with GitHub Actions (Linux/Windows, GCC/Clang/MSVC)

---

## Phase 7: Benchmarking and Observability

**Goal**: Create a comprehensive benchmarking suite to validate simulator correctness, measure performance characteristics, and establish baselines for optimization.

### 7.1 Microbenchmarks

Individual operation performance measurement:

| Benchmark | Description | Metrics |
|-----------|-------------|---------|
| `matmul_sweep` | Matmul across sizes (64 to 16K) | Cycles, GFLOPS, efficiency |
| `tile_size_sensitivity` | Vary Ti, Tj, Tk for fixed problem | Cycles, memory traffic |
| `memory_bandwidth` | DMA, BlockMover, Streamer throughput | GB/s achieved vs peak |
| `activation_overhead` | Compare matmul vs MLP | VE overhead cycles |
| `graph_overhead` | Single kernel vs graph compilation | Compilation time, instruction count |

### 7.2 Roofline Analysis

Validate achieved performance against theoretical peaks:

```
Peak Compute: 1024 GFLOPS (16x16 systolic @ 2 GHz, 2 ops/cycle)
Peak Memory BW:
  - External: 64 GB/s (4 channels × 16 GB/s)
  - L3↔L2: 128 GB/s
  - L2↔L1: 256 GB/s

Roofline intersection (arithmetic intensity threshold):
  - External-bound: AI < 16 FLOP/byte
  - L3-bound: AI < 8 FLOP/byte
  - Compute-bound: AI > 16 FLOP/byte
```

### 7.3 Statistics and Observability

Enhance simulator with detailed metrics:

| Metric Category | Measurements |
|-----------------|--------------|
| **Cycle Breakdown** | Compute, DMA wait, BM wait, Streamer wait, barriers |
| **Memory Traffic** | Bytes per level, cache hit rates, reuse factors |
| **Resource Utilization** | PE utilization, memory channel utilization |
| **Pipeline Efficiency** | Overlap ratio, bubble cycles, stall causes |
| **Energy Estimates** | pJ per operation, memory access energy |

### 7.4 Regression Testing

Establish performance baselines:

- Record expected cycles for canonical workloads
- Fail CI if performance regresses by >5%
- Track metrics over time

### Deliverables

1. `tests/benchmarks/` - Benchmark test files
2. `include/sw/kpu/benchmark.hpp` - Benchmark harness API
3. `tools/benchmark/` - Standalone benchmark runner
4. `docs/benchmark-results.md` - Baseline results documentation

---

## Phase 8: Additional Kernel Types

**Goal**: Implement kernel types required for real neural network workloads.

### 8.1 Convolution Kernels

```cpp
Kernel::create_conv2d(
    Size batch, Size H, Size W,
    Size C_in, Size C_out,
    Size kH, Size kW,
    Size stride_h, Size stride_w,
    Size pad_h, Size pad_w,
    DataType dtype
);
```

Implementation approach:
- im2col transformation to reduce to matmul
- Or direct convolution with sliding window tiling

### 8.2 Attention Kernels

```cpp
Kernel::create_attention(
    Size batch, Size seq_len,
    Size num_heads, Size head_dim,
    bool causal_mask,
    DataType dtype
);
```

Components:
- Q, K, V projections (3 matmuls)
- QK^T computation
- Softmax
- Attention × V
- Output projection

### 8.3 Normalization Kernels

```cpp
Kernel::create_layernorm(Size batch, Size seq_len, Size hidden);
Kernel::create_rmsnorm(Size batch, Size seq_len, Size hidden);
Kernel::create_batchnorm(Size batch, Size C, Size H, Size W);
```

### 8.4 Elementwise Kernels

```cpp
Kernel::create_elementwise(
    std::vector<Size> shape,
    ElementwiseOp op  // ADD, MUL, GELU, SILU, etc.
);
```

---

## Phase 9: True Kernel Fusion

**Goal**: Implement actual kernel fusion to reduce memory traffic.

### 9.1 Fusion Patterns

| Pattern | Before | After | Memory Savings |
|---------|--------|-------|----------------|
| MatMul+Bias+Act | 3 kernels, 3 passes | 1 kernel, 1 pass | 2× reduction |
| Attention Block | 7 kernels | 2-3 fused kernels | 3-4× reduction |
| FFN Block | 4 kernels | 1-2 fused kernels | 2-3× reduction |

### 9.2 Implementation

Complete the TODO items:
- `src/simulator/kernel_graph.cpp:501` - Fusion strategies
- `src/simulator/kernel_graph.cpp:614` - True fusion

### 9.3 Fusion Compiler

```cpp
class FusionCompiler {
    // Analyze graph for fusion opportunities
    std::vector<FusionGroup> find_fusion_groups(const KernelGraph& graph);

    // Generate fused kernel
    Kernel compile_fused(const FusionGroup& group);

    // Replace original kernels with fused version
    KernelGraph apply_fusion(const KernelGraph& graph);
};
```

---

## Phase 10: Quantization Support

**Goal**: Enable INT8/INT4 inference for memory bandwidth reduction.

### 10.1 Quantized Data Types

Already defined, need compute support:
- `DataType::INT8` - 8-bit signed integer
- `DataType::INT4` - 4-bit packed (2 per byte)

### 10.2 Quantized Compute

```cpp
// INT8 matmul with INT32 accumulator
Kernel::create_matmul_int8(M, N, K);

// Mixed precision: INT8 weights, FP16 activations
Kernel::create_matmul_mixed(M, N, K,
    DataType::INT8,   // weights
    DataType::FLOAT16 // activations
);
```

### 10.3 Quantization Kernels

```cpp
Kernel::create_quantize(shape, DataType::FLOAT32, DataType::INT8, scale, zero_point);
Kernel::create_dequantize(shape, DataType::INT8, DataType::FLOAT32, scale, zero_point);
```

---

## Phase 11: Model-Level Execution

**Goal**: Run complete neural network models on the simulator.

### 11.1 Model Loading

```cpp
class ModelLoader {
    // Load from ONNX
    KernelGraph load_onnx(const std::string& path);

    // Load from custom format
    KernelGraph load_kpumodel(const std::string& path);
};
```

### 11.2 Inference Pipeline

```cpp
class InferencePipeline {
    void load_model(const std::string& path);
    void set_input(const std::string& name, const Tensor& data);
    Tensor run();

    // Performance
    InferenceStats get_stats();
};
```

### 11.3 Reference Models

| Model | Layers | Parameters | Target Use Case |
|-------|--------|------------|-----------------|
| GPT-2 FFN | 2 layers | ~3M | Transformer validation |
| BERT-base | 12 layers | 110M | Full transformer |
| ResNet-18 | 18 layers | 11M | CNN validation |
| MobileNetV2 | 53 layers | 3.4M | Efficient CNN |

---

## Implementation Priority

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 7: Benchmarking                     │
│  Priority: HIGH                                              │
│  Effort: Medium                                              │
│  Value: Foundation for all optimization work                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 8: Additional Kernels                  │
│  Priority: HIGH                                              │
│  Effort: High                                                │
│  Value: Required for real model execution                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 9: Kernel Fusion                      │
│  Priority: MEDIUM                                            │
│  Effort: High                                                │
│  Value: 2-4× memory traffic reduction                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 10: Quantization                       │
│  Priority: MEDIUM                                            │
│  Effort: Medium                                              │
│  Value: 4-8× memory savings for inference                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 11: Model-Level Execution                 │
│  Priority: LOW (depends on 8-10)                             │
│  Effort: High                                                │
│  Value: Complete inference pipeline                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Open TODO Items (from codebase)

| File | Line | TODO |
|------|------|------|
| `program_executor.cpp` | 156 | Configuration implementation |
| `schedule_characterizer.cpp` | 278 | Weight/input stationary schedulers |
| `schedule_characterizer.cpp` | 288 | Strategy-specific energy/latency |
| `graph_loader.cpp` | 49 | Proper topological sort |
| `graph_loader.cpp` | 359 | Tensor metadata from domain_flow |
| `tile_optimizer.cpp` | 21 | ML-based prediction |
| `schedule_generator.cpp` | 307 | Smarter dependency analysis |
| `schedule_generator.cpp` | 648 | JSON export |
| `l2_tile_scheduler.cpp` | 453 | Optimal lookahead |
| `kernel_graph.cpp` | 501 | Fusion strategies |
| `kernel_graph.cpp` | 614 | True fusion |

---

## Success Metrics

### Phase 7 (Benchmarking)
- [ ] Matmul benchmark covering 64×64 to 8192×8192
- [ ] Achieved GFLOPS within 80% of theoretical peak for large problems
- [ ] Memory bandwidth utilization >70% for bandwidth-bound cases
- [ ] Cycle-accurate regression tests for 10+ canonical workloads
- [ ] Roofline plot generation

### Phase 8 (Kernels)
- [ ] Conv2D kernel passing correctness tests
- [ ] Attention kernel for transformer inference
- [ ] LayerNorm/Softmax kernels

### Phase 9 (Fusion)
- [ ] 2× memory traffic reduction on FFN pattern
- [ ] Automatic fusion detection and application

### Phase 10 (Quantization)
- [ ] INT8 matmul with <1% accuracy loss
- [ ] 4× memory bandwidth reduction vs FP32

### Phase 11 (Models)
- [ ] GPT-2 FFN layer inference
- [ ] Complete BERT-base inference
- [ ] Performance within 2× of theoretical predictions
