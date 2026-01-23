# KPU Simulator Roadmap

**Master Development Roadmap with Semantic Versioning**

This document consolidates all development tracks into a unified roadmap with semantic versions for release tagging and tracking.

---

## Version Numbering Convention

```
MAJOR.MINOR.PATCH

0.x.y  - Pre-production development
1.0.0  - Production-ready: Full DNN execution with timing simulation
1.x.y  - Post-1.0 feature additions and optimizations
```

---

## Current Status: v0.8.0 ✅

Released: 2026-01-22

### What's Complete

| Component | Version | Status |
|-----------|---------|--------|
| **C++ Simulator Core** | v0.1.0 | ✅ Phases 1-6 complete |
| **Python kpu Package** | v0.8.0 | ✅ Model-Level Execution |
| **torch.compile Backend** | v0.2.0 | ✅ `backend="kpu"` |
| **BEHAVIORAL Runtime** | v0.8.0 | ✅ Full functional simulation |
| **CNN Operators** | v0.5.7 | ✅ conv2d, pooling, normalization |
| **3D/Video Operators** | v0.6.4 | ✅ conv3d, pool3d, batchnorm3d |
| **Transformer Operators** | v0.5.7 | ✅ attention, layernorm, softmax |
| **Kernel Fusion** | v0.6.3 | ✅ MatMul+Bias+Act, Conv2D+BN+ReLU |
| **Quantization** | v0.7.11 | ✅ INT8/INT4/FP16/BF16/FP8/FP4, calibration |
| **Model Classes** | v0.8.0 | ✅ Layer, Sequential, Model, SqueezeNet |
| **Inference Pipeline** | v0.8.0 | ✅ End-to-end execution with stats |
| **Memory Planner** | v0.8.0 | ✅ Optimal buffer allocation |
| **MNIST Examples** | v0.2.0 | ✅ MLP and CNN verified |
| **C++ Kernel Types** | v0.5.7 | ✅ All v0.5.x kernels complete |
| **Model Compatibility** | v0.6.4 | ✅ 40/45 models (89%) |

---

## Roadmap Overview

```
v0.2.0 ──→ v0.3.0 ──→ v0.4.0 ──→ v0.5.0 ──→ v0.6.0 ──→ v0.7.0 ──→ v0.8.0 ──→ v1.0.0
  │          │          │          │          │          │          │          │
  │          │          │          │          │          │          │          │
Current   Bench-    TRANS-     Add'l     Kernel    Quant-    Model     Prod
          marking   ACTIONAL   Kernels   Fusion    ization   Level     Ready
                    Runtime    (C++)
```

---

## v0.3.0 - Benchmarking & Observability

**Priority:** HIGH
**Effort:** 2-3 weeks
**Theme:** Measurement infrastructure for all optimization work

### Features

| SEMVER | Feature | Description | Files |
|--------|---------|-------------|-------|
| v0.3.1 | Microbenchmarks | matmul sweep (64→8K), conv2d sweep, memory BW | `tests/benchmarks/` |
| v0.3.2 | Roofline Tooling | Plot generation, benchmark runner | `tools/benchmark/` |
| v0.3.3 | XUE Event Hierarchy | 45+ event types, C++ EventCollector | `include/sw/xue/` |
| v0.3.4 | XUE pybind11 Bindings | Python API for XUE summary/analysis | `python/kpu/_native/` |
| v0.3.5 | Analytical Validation | Tests with known solutions (16×16→128×128) | `tests/validation/` |
| v0.3.6 | XUE Prediction Accuracy | Validate predictions within 10% of actual | `tests/validation/` |
| v0.3.7 | Regression Baselines | 10+ baselines, CI fails on >5% regression | `.github/workflows/` |

### Completed Versions

| Version | Feature | Status |
|---------|---------|--------|
| v0.3.3 | XUE Event Hierarchy | ✅ Released (45+ event types) |
| v0.3.4 | XUE pybind11 Bindings | ✅ Released (get_xue_summary, get_operational_analysis) |

### Success Criteria

- [ ] Matmul benchmark 64×64 to 8192×8192 (v0.3.1)
- [ ] Achieved GFLOPS within 80% of peak for large problems (v0.3.1)
- [ ] Memory bandwidth utilization >70% for BW-bound cases (v0.3.1)
- [ ] Roofline plot generation (v0.3.2)
- [x] XUE Event Hierarchy (v0.3.3) ✅
- [x] XUE Python API (v0.3.4) ✅
- [ ] Analytical validation tests pass (v0.3.5)
- [ ] XUE predictions within 10% accuracy (v0.3.6)
- [ ] 10+ regression baselines (v0.3.7)

### Gap Analysis

See `docs/plans/v0.3-benchmarking-gap-analysis.md` for detailed implementation plan.

### Tag: `v0.3.0-benchmarks`

---

## v0.4.0 - TRANSACTIONAL Runtime

**Priority:** HIGH
**Effort:** 3-4 weeks
**Theme:** Connect Python to C++ for timing simulation

### Features

| SEMVER | Feature | Description | Files |
|--------|---------|-------------|-------|
| v0.4.0 | pybind11 Integration | Full native bindings | `python/kpu/_native/` |
| v0.4.1 | DFX→C++ Parser | Parse DFX JSON in C++ | `src/bindings/dfx_parser.cpp` |
| v0.4.2 | Timing Stats | Cycles, memory access counts | `include/sw/kpu/stats/` |
| v0.4.3 | Python API | `kpu.set_fidelity(TRANSACTIONAL)` | `python/kpu/runtime.py` |
| v0.4.4 | Compute Validation | XUE metrics for systolic array | `patterns/compute-tile/systolic/` |
| v0.4.5 | Tiled MatMul E2E | Full memory hierarchy (DRAM→L3→L2→L1→Compute) | `patterns/compute-tile/systolic/` |
| v0.4.6 | Python↔C++ Execution | DFX execution on native simulator | `python/kpu/_native/` |
| v0.4.7 | torch.compile TRANSACTIONAL | Timing simulation via torch backend | `python/kpu/torch_backend.py` |

### Architecture

```
Python kpu package
       │
       │ DFX JSON
       ▼
┌─────────────────┐
│  _native module │  ← pybind11
│  (kpu_native.so)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   C++ kpu-sim   │
│  KernelCompiler │
│  GraphExecutor  │
└─────────────────┘
```

### Performance Tests (v0.4.5)

For KPU configuration of a single 16x16 PE array, 1M L3, 128K L2, 1k L1, create a performance test that has analytical solutions to event occurrences, and measure service times and latencies with XUE framework:

- [ ] 16x16 matmul, single tile, from DRAM to DRAM. All occurrences correct, service time for matmul, latency for tile to L3, L2, first L1 stream, last L1 stream buffer, latency for result tile to DRAM
- [ ] 32x32 matmul, four tiles, from DRAM to DRAM, measure compute tile efficiency to see if we are pipelining properly
- [ ] 64x64 matmul, 16 tiles, from DRAM to DRAM, measure compute tile efficiency
- [ ] 128x128 matmul, 64 tiles, from DRAM to DRAM, measure compute tile efficiency

### Python↔C++ Integration (v0.4.6)

- [x] DFX JSON parsed and executed by C++ simulator
- [x] Timing stats (cycles, memory traffic) returned to Python
- [x] ExecutionStats populated from native execution

### torch.compile TRANSACTIONAL (v0.4.7)

- [x] torch.compile backend works with TRANSACTIONAL fidelity
- [x] Timing stats accessible after model execution
- [x] MNIST MLP timing matches C++ direct execution

### Success Criteria

- [x] v0.4.4: XUE metrics validated for compute patterns
- [x] v0.4.5: End-to-end tiled matmul with memory hierarchy
- [x] v0.4.6: Python can execute DFX on C++ simulator
- [x] v0.4.7: torch.compile works with TRANSACTIONAL mode

### Tag: `v0.4-transactional`

---

## v0.5.0 - Additional Kernel Types (C++)

**Priority:** HIGH
**Effort:** 5-7 weeks
**Theme:** C++ kernels for real neural networks

### Features

| SEMVER | Kernel | API | Implementation |
|--------|--------|-----|----------------|
| v0.5.0 | Conv2D | `Kernel::create_conv2d(...)` | im2col + GEMM |
| v0.5.1 | Attention | `Kernel::create_attention(...)` | Q,K,V projections + softmax |
| v0.5.2 | LayerNorm | `Kernel::create_layernorm(...)` | Mean/var + affine |
| v0.5.3 | RMSNorm | `Kernel::create_rmsnorm(...)` | RMS + scale |
| v0.5.4 | BatchNorm | `Kernel::create_batchnorm(...)` | Foldable to preceding conv |
| v0.5.5 | Elementwise | `Kernel::create_elementwise(...)` | add, mul, residual |
| v0.5.6 | Pool2D | `Kernel::create_pool2d(...)` | max, avg, global_avg |
| v0.5.7 | Softmax | `Kernel::create_softmax(...)` | Reduction-based |

### Files to Create/Modify

```
include/sw/kpu/kernel.hpp         - Factory methods
src/kpu/kernel.cpp                - Implementations
include/sw/compiler/
├── conv2d_compiler.hpp           - Conv2D compilation
├── attention_compiler.hpp        - Attention compilation
└── pool_compiler.hpp             - Pooling compilation
src/compiler/
├── conv2d_compiler.cpp
├── attention_compiler.cpp
└── pool_compiler.cpp
```

### Success Criteria

- [x] Conv2D kernel passes correctness tests (validated in test_v05_kernel_validation.py)
- [x] Attention kernel for transformer inference (SDPA + MHA working)
- [x] LayerNorm/Softmax kernels working (validated)
- [x] All kernels accessible from Python via TRANSACTIONAL (validated)

### Kernels Completed (v0.5.0 - v0.5.7)

| Version | Kernel | Status |
|---------|--------|--------|
| v0.5.0 | Conv2D | ✅ Released |
| v0.5.1 | Attention | ✅ Released |
| v0.5.2 | LayerNorm | ✅ Released |
| v0.5.3 | RMSNorm | ✅ Released |
| v0.5.4 | BatchNorm | ✅ Released |
| v0.5.5 | Elementwise | ✅ Released |
| v0.5.6 | Pool2D | ✅ Released |
| v0.5.7 | Softmax | ✅ Released |

### Tag: `v0.5-kernels`

---

## v0.6.0 - Kernel Fusion

**Priority:** MEDIUM
**Effort:** 4-6 weeks
**Theme:** Reduce memory traffic through fusion

### Fusion Patterns

| Pattern | Before | After | Memory Savings |
|---------|--------|-------|----------------|
| MatMul+Bias+Act | 3 passes | 1 pass | 2× |
| Attention Block | 7 kernels | 2-3 fused | 3-4× |
| FFN Block | 4 kernels | 1-2 fused | 2-3× |

### Features

| SEMVER | Feature | Description | Files |
|--------|---------|-------------|-------|
| v0.6.0 | FusionCompiler | Find and apply fusion groups | `include/sw/compiler/fusion_compiler.hpp` |
| v0.6.1 | Fusion Detection | Analyze graph for opportunities | `src/compiler/fusion_detection.cpp` |
| v0.6.2 | Fused Kernels | Generate merged programs | `src/compiler/fused_kernel.cpp` |
| v0.6.3 | Compute Efficiency | Demonstrate Compute Efficiency improvement | `examples/fusion/fusing_kernels.cpp` |

### Success Criteria

- [x] 2× memory traffic reduction on FFN pattern (2.77× achieved)
- [x] Automatic fusion detection (FusionAnalyzer detects opportunities)
- [x] MatMul+Bias+ReLU fuses correctly (behavioral output matches)
- [x] Unfused baseline and Fused baseline (`optimize=False` vs `optimize=True`)
- [x] Validate unfused is memory bound, fused is compute bound (64% → 100% efficiency)

### Completed Versions

| Version | Feature | Status |
|---------|---------|--------|
| v0.6.0 | FusionCompiler | ✅ Released |
| v0.6.1 | Fusion Detection & Roofline | ✅ Released |
| v0.6.2 | Conv2D Fused Kernels | ✅ Released |
| v0.6.3 | Compute Efficiency Validation | ✅ Released |
| v0.6.4 | Conv3d & Video Model Support | ✅ Released |

### Tag: `v0.6.4`

---

## v0.7.0 - Quantization Support

**Priority:** MEDIUM
**Effort:** 3-4 weeks
**Theme:** INT8/INT4 inference for bandwidth reduction

### Features

| SEMVER | Feature | Description |
|--------|---------|-------------|
| v0.7.0 | INT8 MatMul | `Kernel::create_matmul_int8(M, N, K)` |
| v0.7.1 | FP16 MatMul | `Kernel::create_matmul_fp16(M, N, K)` |
| v0.7.2 | BF16 MatMul | `Kernel::create_matmul_bf16(M, N, K)` |
| v0.7.3 | FP8e2 MatMul | `Kernel::create_matmul_fp8e2(M, N, K)` |
| v0.7.4 | FP8e3 MatMul | `Kernel::create_matmul_fp8e3(M, N, K)` |
| v0.7.5 | FP8e4 MatMul | `Kernel::create_matmul_fp8e4(M, N, K)` |
| v0.7.6 | FP8e5 MatMul | `Kernel::create_matmul_fp8e5(M, N, K)` |
| v0.7.7 | INT4 MatMul | `Kernel::create_matmul_int4(M, N, K)` |
| v0.7.8 | FP4 MatMul | `Kernel::create_matmul_fp4(M, N, K)` |
| v0.7.9 | Mixed Precision | INT8 weights, FP16 activations |
| v0.7.10 | Quantize/Dequantize | Scale and zero-point handling |
| v0.7.11 | Calibration Utils | Post-training quantization helpers |

### Data Types

| Type | Size | Status |
|------|------|--------|
| INT8 | 1Byte | Implemented in v0.7.0 |
| UINT8 | 1Byte | Implemented in v0.7.0 |
| INT4 | 0.5Byte | Packed format in v0.7.7 |
| FP16 | 2Bytes | v0.7.1 |
| BF16 | 2Bytes | v0.7.2 |
| FP8e2 | 1Byte | v0.7.3 |
| FP8e3 | 1Byte | v0.7.4 |
| FP8e4 | 1Byte | v0.7.5 |
| FP8e5 | 1Byte | v0.7.6 |
| FP4 | 0.5Byte | Packed format in v0.7.8 |

### Success Criteria

- [x] INT8 matmul with <1% accuracy loss
- [x] 4× memory bandwidth reduction vs FP32
- [x] Quantization-aware simulation timing
- [x] Full calibration support (MinMax, Percentile, MSE, Entropy)

### Completed Versions

| Version | Feature | Status |
|---------|---------|--------|
| v0.7.0 | INT8 quantization | ✅ Released |
| v0.7.1 | FP16 operations | ✅ Released |
| v0.7.2 | BF16 operations | ✅ Released |
| v0.7.3-6 | FP8 variants | ✅ Released |
| v0.7.7 | INT4 operations | ✅ Released |
| v0.7.8 | FP4 operations | ✅ Released |
| v0.7.9 | Mixed precision | ✅ Released |
| v0.7.10 | Q/DQ operations | ✅ Released |
| v0.7.11 | Calibration | ✅ Released |

### Tag: `v0.7.11`

---

## v0.8.0 - Model-Level Execution ✅

**Priority:** MEDIUM
**Effort:** 4-6 weeks
**Theme:** Complete neural network inference

### Features

| SEMVER |Feature | Description | Files |
|--------|--------|-------------|-------|
| v0.8.0 | Model Classes | Layer, Sequential, Model | `python/kpu/model.py` |
| v0.8.1 | Model Loader | JSON and ONNX loading | `python/kpu/model_loader.py` |
| v0.8.2 | Inference Pipeline | End-to-end execution with stats | `python/kpu/inference.py` |
| v0.8.3 | Memory Planning | Optimal buffer allocation | `python/kpu/memory_planner.py` |
| v0.8.4 | Reference Models | SqueezeNet, MobileNetV2, MNIST | `python/kpu/models/` |

### Reference Models

| Model | Params | Status |
|-------|--------|--------|
| SqueezeNet 1.0 | 740K | ✅ Implemented |
| SqueezeNet 1.1 | 720K | ✅ Implemented |
| MobileNetV2 | 3.4M | ✅ Implemented |
| MNIST MLP | 109K | ✅ Implemented |
| MNIST CNN | 422K | ✅ Implemented |

### Success Criteria

- [x] SqueezeNet executes on kpu-sim
- [x] Output computed correctly (BEHAVIORAL mode)
- [x] Performance metrics collected per layer
- [x] JSON model loading works
- [x] Memory planning and optimization

### Tag: `v0.8.0`

---

## v1.0.0 - Production Ready

**Priority:** HIGH (milestone)
**Theme:** Full DNN execution with complete timing simulation

### Requirements for 1.0

- [ ] All v0.x features stable
- [ ] CYCLE_ACCURATE mode fully functional
- [ ] Documentation complete
- [ ] API stability guaranteed
- [ ] Performance within 2× of theoretical predictions
- [ ] Validated against hardware (FPGA or silicon)

### Validation Models

| Model | Status |
|-------|--------|
| MNIST MLP | Must pass |
| MNIST CNN | Must pass |
| SqueezeNet 1.0 | Must pass |
| MobileNetV2 | Must pass |
| BERT-base attention | Must pass |

### Tag: `v1.0.0`

---

## Post-1.0 Roadmap

### v1.1.0 - Advanced Optimizations
- Auto-tuning for tile sizes
- ML-based tile prediction
- Advanced scheduling heuristics

### v1.2.0 - Multi-Device
- Multi-chip simulation
- Distributed inference
- Model parallelism

### v1.3.0 - Training Support
- Backward pass kernels
- Gradient accumulation
- Optimizer kernels

---

## Hardware Development Milestones

These simulator versions align with hardware development:

| Hardware Milestone | Timeline | Simulator Version |
|-------------------|----------|-------------------|
| **M1: fsim** | Months 1-3 | v0.3.0+ (benchmarks) |
| **M2: FPGA** | Months 3-6 | v0.4.0+ (transactional) |
| **M3: MPW** | Months 6-9 | v0.5.0+ (all kernels) |
| **M4: Test Chip** | Months 9-12 | v0.8.0+ (model execution) |
| **M5: Production** | Months 12-24 | v1.0.0 (production ready) |

---

## Priority Matrix

### P0 - Critical Path (blocks everything)

| Version | Feature | Effort |
|---------|---------|--------|
| v0.3.0 | Benchmarking | 2-3 weeks |
| v0.4.0 | TRANSACTIONAL runtime | 3-4 weeks |
| v0.5.0 | Conv2D kernel | 2 weeks |
| v0.5.0 | Attention kernel | 2 weeks |

### P1 - Important (needed for real models)

| Version | Feature | Effort |
|---------|---------|--------|
| v0.5.0 | Pooling ops | 1 week |
| v0.5.0 | Softmax | 1 week |
| v0.5.0 | Elementwise | 3 days |
| v0.6.0 | MatMul+Bias+Act fusion | 2 weeks |
| v0.8.0 | Model loader (JSON) | 1 week |

### P2 - Nice to Have (optimization)

| Version | Feature | Effort |
|---------|---------|--------|
| v0.6.0 | Attention fusion | 2 weeks |
| v0.7.0 | INT8 inference | 2 weeks |
| v0.8.0 | ONNX loader | 2 weeks |

---

## Open TODOs from Codebase

These items should be addressed in the appropriate version:

| File | TODO | Version |
|------|------|---------|
| `program_executor.cpp:156` | Configuration implementation | v0.3.0 |
| `schedule_characterizer.cpp:278` | Weight/input stationary | v0.5.0 |
| `tile_optimizer.cpp:21` | ML-based prediction | v1.1.0 |
| `kernel_graph.cpp:501` | Fusion strategies | v0.6.0 |
| `kernel_graph.cpp:614` | True fusion | v0.6.0 |

---

## Consolidated Documents

This roadmap supersedes:

| Document | Status |
|----------|--------|
| `docs/plans/roadmap-phase7-onwards.md` | Incorporated |
| `docs/09-virtual-platform/unified-dnn-roadmap.md` | Incorporated |
| `docs/09-virtual-platform/api-gaps-roadmap.md` | Incorporated |
| `docs/project/project_plan.md` | Hardware timeline (reference only) |

---

## Release Process

For each version:

1. Complete all features for that version
2. Run full test suite
3. Update CHANGELOG.md
4. Tag with `git tag vX.Y.Z`
5. Create GitHub release with notes

```bash
# Example release
git tag -a v0.3.0 -m "Benchmarking & Observability"
git push origin v0.3.0
```

---

*Document created: 2026-01-18*
*Last updated: 2026-01-20*
