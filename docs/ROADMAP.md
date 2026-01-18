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

## Current Status: v0.2.0 ✅

Released: 2026-01-17

### What's Complete

| Component | Version | Status |
|-----------|---------|--------|
| **C++ Simulator Core** | v0.1.0 | ✅ Phases 1-6 complete |
| **Python kpu Package** | v0.2.0 | ✅ `@kpu.compile` decorator |
| **torch.compile Backend** | v0.2.0 | ✅ `backend="kpu"` |
| **BEHAVIORAL Runtime** | v0.2.0 | ✅ Full functional simulation |
| **CNN Operators** | v0.2.0 | ✅ conv2d, pooling, normalization |
| **MNIST Examples** | v0.2.0 | ✅ MLP and CNN verified |

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

| Feature | Description | Files |
|---------|-------------|-------|
| Microbenchmarks | matmul sweep (64→16K), tile sensitivity | `tests/benchmarks/` |
| Roofline Analysis | Peak vs achieved FLOPS/bandwidth | `tools/benchmark/` |
| Statistics Collection | Cycle breakdown, memory traffic, utilization | `include/sw/kpu/stats/` |
| Performance Regression | CI fails if >5% regression | `.github/workflows/` |

### Success Criteria

- [ ] Matmul benchmark 64×64 to 8192×8192
- [ ] Achieved GFLOPS within 80% of peak for large problems
- [ ] Memory bandwidth utilization >70% for BW-bound cases
- [ ] Roofline plot generation
- [ ] 10+ regression baselines

### Tag: `v0.3.0-benchmarks`

---

## v0.4.0 - TRANSACTIONAL Runtime

**Priority:** HIGH
**Effort:** 3-4 weeks
**Theme:** Connect Python to C++ for timing simulation

### Features

| Feature | Description | Files |
|---------|-------------|-------|
| pybind11 Integration | Full native bindings | `python/kpu/_native/` |
| DFX→C++ Parser | Parse DFX JSON in C++ | `src/bindings/dfx_parser.cpp` |
| Timing Stats | Cycles, memory access counts | `include/sw/kpu/stats/` |
| Python API | `kpu.set_fidelity(TRANSACTIONAL)` | `python/kpu/runtime.py` |

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

### Success Criteria

- [ ] Python can execute DFX on C++ simulator
- [ ] Timing stats returned to Python
- [ ] torch.compile works with TRANSACTIONAL mode
- [ ] MNIST MLP timing matches C++ direct execution

### Tag: `v0.4.0-transactional`

---

## v0.5.0 - Additional Kernel Types (C++)

**Priority:** HIGH
**Effort:** 5-7 weeks
**Theme:** C++ kernels for real neural networks

### Features

| Kernel | API | Implementation |
|--------|-----|----------------|
| Conv2D | `Kernel::create_conv2d(...)` | im2col + GEMM |
| Attention | `Kernel::create_attention(...)` | Q,K,V projections + softmax |
| LayerNorm | `Kernel::create_layernorm(...)` | Mean/var + affine |
| RMSNorm | `Kernel::create_rmsnorm(...)` | RMS + scale |
| BatchNorm | `Kernel::create_batchnorm(...)` | Foldable to preceding conv |
| Elementwise | `Kernel::create_elementwise(...)` | add, mul, residual |
| Pool2D | `Kernel::create_pool2d(...)` | max, avg, global_avg |
| Softmax | `Kernel::create_softmax(...)` | Reduction-based |

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

- [ ] Conv2D kernel passes correctness tests
- [ ] Attention kernel for transformer inference
- [ ] LayerNorm/Softmax kernels working
- [ ] All kernels accessible from Python via TRANSACTIONAL

### Tag: `v0.5.0-kernels`

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

| Feature | Description | Files |
|---------|-------------|-------|
| FusionCompiler | Find and apply fusion groups | `include/sw/compiler/fusion_compiler.hpp` |
| Fusion Detection | Analyze graph for opportunities | `src/compiler/fusion_detection.cpp` |
| Fused Kernels | Generate merged programs | `src/compiler/fused_kernel.cpp` |

### Success Criteria

- [ ] 2× memory traffic reduction on FFN pattern
- [ ] Automatic fusion detection
- [ ] MatMul+Bias+ReLU fuses correctly

### Tag: `v0.6.0-fusion`

---

## v0.7.0 - Quantization Support

**Priority:** MEDIUM
**Effort:** 3-4 weeks
**Theme:** INT8/INT4 inference for bandwidth reduction

### Features

| Feature | Description |
|---------|-------------|
| INT8 MatMul | `Kernel::create_matmul_int8(M, N, K)` |
| Mixed Precision | INT8 weights, FP16 activations |
| Quantize/Dequantize | Scale and zero-point handling |
| Calibration Utils | Post-training quantization helpers |

### Data Types

| Type | Size | Status |
|------|------|--------|
| INT8 | 1B | Implemented in v0.7.0 |
| UINT8 | 1B | Implemented in v0.7.0 |
| INT4 | 0.5B | Packed format in v0.7.0 |

### Success Criteria

- [ ] INT8 matmul with <1% accuracy loss
- [ ] 4× memory bandwidth reduction vs FP32
- [ ] Quantization-aware simulation timing

### Tag: `v0.7.0-quantization`

---

## v0.8.0 - Model-Level Execution

**Priority:** MEDIUM
**Effort:** 4-6 weeks
**Theme:** Complete neural network inference

### Features

| Feature | Description | Files |
|---------|-------------|-------|
| Model Loader | ONNX and custom JSON | `include/sw/compiler/model_loader.hpp` |
| Inference Pipeline | End-to-end execution | `include/sw/runtime/inference.hpp` |
| Memory Planning | Optimal buffer allocation | `src/runtime/memory_planner.cpp` |

### Reference Models

| Model | Params | Purpose |
|-------|--------|---------|
| SqueezeNet 1.0 | 1.2M | First real torchvision model |
| MobileNetV2 | 3.4M | Efficient CNN validation |
| GPT-2 FFN | ~3M | Transformer validation |
| BERT-base | 110M | Full transformer |

### Success Criteria

- [ ] SqueezeNet executes on kpu-sim
- [ ] Output matches PyTorch reference within tolerance
- [ ] Performance metrics collected per layer
- [ ] ONNX model loading works

### Tag: `v0.8.0-models`

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
*Last updated: 2026-01-18*
