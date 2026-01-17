# Unified DNN Execution Roadmap

## Overview

This document consolidates two related roadmaps into a unified plan:

1. **API Gaps Roadmap** (`api-gaps-roadmap.md`) - C++ simulator gaps for full DNN execution
2. **Exaloop Integration Design** (`exaloop-integration-design.md`) - Python interface for compiler team

The goal is a single coherent roadmap that enables executing full DNNs (starting with MNIST MLP, progressing to SqueezeNet) via a Python interface backed by the C++ kpu-sim.

---

## Current Status (2026-01-17)

### Completed

| Component | Status | Notes |
|-----------|--------|-------|
| **Python kpu package** | ✅ Done | `python/kpu/` with `@kpu.compile` decorator |
| **Tensor class** | ✅ Done | Wraps NumPy with tracing support |
| **OpGraph/DFX emission** | ✅ Done | Builds graph, emits DFX IR JSON |
| **BEHAVIORAL runtime** | ✅ Done | Pure Python execution (computes values) |
| **MNIST MLP example** | ✅ Done | 784→128→64→10, verified against NumPy |
| **CNN operators** | ✅ Done | conv2d, pooling, layer_norm, batch_norm |
| **MNIST CNN example** | ✅ Done | Conv→Pool→FC, verified against NumPy |
| **MNIST data loader** | ✅ Done | Downloads from S3 mirror |
| **torch.compile backend** | ✅ Done | `torch.compile(model, backend="kpu")` |
| **FX graph converter** | ✅ Done | Walks FX IR, maps to kpu operations |
| **Unit tests** | ✅ Done | All passing |
| **pybind11 infrastructure** | ✅ Done | `_native/` module skeleton |

### Not Yet Done

| Component | Priority | Notes |
|-----------|----------|-------|
| TRANSACTIONAL runtime | P1 | Connect to C++ kpu-sim |
| CYCLE_ACCURATE runtime | P2 | Full timing simulation |
| Codon DSL plugin | P2 | Native compilation via Exaloop |
| Offline model export | P3 | ONNX/flatbuffers for deployment |

---

## torch.compile Integration (NEW)

The primary development workflow is now through `torch.compile`:

```python
import torch
import torchvision.models as models

# Load any PyTorch model
model = models.squeezenet1_0(pretrained=True)
model.eval()

# Compile with KPU backend
compiled_model = torch.compile(model, backend="kpu")

# Execute - Dynamo captures graph, KPU backend runs on simulator
output = compiled_model(input_tensor)
```

### How It Works

1. **Dynamo captures** the model's execution as an FX GraphModule
2. **KPU backend receives** the FX graph with all operations
3. **FX converter walks** the graph, mapping PyTorch ops to kpu ops
4. **Behavioral runtime** executes operations using NumPy
5. **Results are validated** against PyTorch eager mode

### Supported Operations

The FX converter supports all common DNN operations:

| Category | Operations |
|----------|------------|
| **Activations** | relu, gelu, silu, sigmoid, tanh, softmax |
| **Convolutions** | conv2d (nn.Conv2d or F.conv2d) |
| **Pooling** | max_pool2d, avg_pool2d, adaptive_avg_pool2d |
| **Normalization** | batch_norm, layer_norm |
| **Linear** | linear, matmul, mm, bmm |
| **Elementwise** | add, sub, mul, div |
| **Shape** | reshape, view, flatten, transpose, permute |
| **Reductions** | mean, sum |

### Files

| File | Purpose |
|------|---------|
| `kpu/torch_backend.py` | Backend registration, KPUBackend class |
| `kpu/fx_converter.py` | FXToKPUConverter, op mapping |
| `examples/torch_compile_demo.py` | Demo with MLP, CNN, pretrained models |

---

## Unified Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           User Python Code                                 │
│                                                                            │
│  @kpu.compile                                                              │
│  def network(x, w1, w2, ...):                                              │
│      h1 = kpu.relu(x @ w1)                                                 │
│      ...                                                                   │
│      return kpu.softmax(output)                                            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         Python kpu Package                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Tensor     │  │   OpGraph    │  │  DFXEmitter  │  │   Runtime    │    │
│  │  (tracing)   │  │  (DAG)       │  │  (JSON IR)   │  │  (execute)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │  BEHAVIORAL  │ │TRANSACTIONAL │ │CYCLE_ACCURATE│
           │  (NumPy)     │ │ (C++ stats)  │ │ (C++ full)   │
           │     Done     │ │  TODO        │ │   TODO       │
           └──────────────┘ └──────────────┘ └──────────────┘
                                    │               │
                                    └───────┬───────┘
                                            ▼
                                ┌───────────────────────┐
                                │   C++ kpu-sim         │
                                │  (existing simulator) │
                                └───────────────────────┘
```

---

## Unified Phases

### Phase 1: MNIST MLP End-to-End ✅ COMPLETE

**Goal:** Execute multi-layer MLP with decorator-based Python interface.

| Task | Status |
|------|--------|
| Python package (`kpu/`) | ✅ |
| Tensor with operator overloading | ✅ |
| Tracing-based graph capture | ✅ |
| DFX IR emission | ✅ |
| BEHAVIORAL runtime (NumPy) | ✅ |
| MNIST MLP example | ✅ |
| Unit test suite | ✅ |

---

### Phase 2: Extend Operators (Current Focus)

**Goal:** Add operators needed for CNNs.

| Operator | Python API | Implementation |
|----------|------------|----------------|
| **Softmax** | `kpu.softmax(x)` | NumPy: `exp(x-max) / sum(exp)` |
| **LayerNorm** | `kpu.layer_norm(x)` | NumPy: `(x-mean)/std * gamma + beta` |
| **Conv2D** | `kpu.conv2d(x, w, ...)` | im2col + matmul |
| **MaxPool2D** | `kpu.max_pool2d(x, ...)` | NumPy strided view |
| **AvgPool2D** | `kpu.avg_pool2d(x, ...)` | NumPy strided view |
| **Concat** | `kpu.concat([a,b], dim)` | NumPy concatenate |
| **Reshape/Flatten** | `x.reshape(...)` | Tensor metadata only |

**Files to modify:**
- `python/kpu/ops.py` - Add new operator functions
- `python/kpu/tensor.py` - Add `reshape()` method
- `python/kpu/graph.py` - Add OpType enum values
- `python/kpu/dfx_emitter.py` - Emit new op types
- `python/kpu/runtime.py` - Execute new ops in BEHAVIORAL mode

**Test cases to add:**
- `test_softmax`, `test_layer_norm`
- `test_conv2d_basic`, `test_conv2d_with_padding`
- `test_maxpool2d`, `test_avgpool2d`
- `test_concat`, `test_reshape`

---

### Phase 3: Simple CNN

**Goal:** Execute Conv→ReLU→Pool→FC architecture.

```python
@kpu.compile
def simple_cnn(x, conv_w, conv_b, fc_w, fc_b):
    # x: [batch, 1, 28, 28] for MNIST
    h = kpu.relu(kpu.conv2d(x, conv_w, stride=1, padding=1) + conv_b)
    h = kpu.max_pool2d(h, kernel_size=2, stride=2)  # [batch, 32, 14, 14]
    h = h.reshape(h.shape[0], -1)                    # [batch, 32*14*14]
    return h @ fc_w + fc_b                           # [batch, 10]
```

**Test:** Validate output against PyTorch reference on MNIST.

---

### Phase 4: Residual Connections

**Goal:** Support skip connections for ResNet-style architectures.

```python
@kpu.compile
def residual_block(x, conv1_w, conv2_w):
    residual = x
    h = kpu.relu(kpu.conv2d(x, conv1_w, padding=1))
    h = kpu.conv2d(h, conv2_w, padding=1)
    return kpu.relu(h + residual)  # Skip connection
```

**Required:** Elementwise add already works (`+` operator).

---

### Phase 5: Model Loading

**Goal:** Load pretrained models (JSON format first, then ONNX).

**JSON format (custom):**
```json
{
  "format": "kpu-model-v1",
  "inputs": [{"name": "input", "shape": [1, 3, 224, 224]}],
  "nodes": [
    {"name": "conv1", "op": "conv2d", "params": {...}, "weights": ["conv1.weight"]}
  ],
  "weights_file": "model_weights.npz"
}
```

**Python API:**
```python
model = kpu.load_model("squeezenet.json")
output = model(input_tensor)
```

---

### Phase 6: SqueezeNet 1.0

**Goal:** Execute first real torchvision model.

**SqueezeNet architecture:**
- Conv1 (96 filters, 7×7, stride 2) → ReLU → MaxPool
- Fire2-9 modules (squeeze 1×1 → expand 1×1 + 3×3 → concat)
- Conv10 (1000 classes) → GlobalAvgPool → Softmax

**Fire module implementation:**
```python
def fire_module(x, squeeze_w, expand1x1_w, expand3x3_w):
    squeeze = kpu.relu(kpu.conv2d(x, squeeze_w, kernel_size=1))
    e1 = kpu.relu(kpu.conv2d(squeeze, expand1x1_w, kernel_size=1))
    e3 = kpu.relu(kpu.conv2d(squeeze, expand3x3_w, kernel_size=3, padding=1))
    return kpu.concat([e1, e3], dim=1)
```

**Validation:** Compare output to `torchvision.models.squeezenet1_0(pretrained=True)`.

---

### Phase 7: C++ Backend Integration (TRANSACTIONAL)

**Goal:** Connect Python to C++ simulator for timing.

| Task | Description |
|------|-------------|
| Build pybind11 bindings | CMake integration |
| DFX JSON → C++ parser | Parse in `_native` module |
| Connect to KernelCompiler | Use existing C++ compiler |
| Execute on simulator | TRANSACTIONAL fidelity |
| Return timing stats | Cycles, memory access counts |

---

### Phase 8: Codon Integration (Optional)

**Goal:** Native compilation via Exaloop/Codon.

| Task | Description |
|------|-------------|
| Codon DSL plugin | `codon-kpu` package |
| CIR pass | Intercept NumPy ops |
| Direct DFX emission | From Codon IR |
| Ahead-of-time compilation | `.kpu` object files |

---

## Immediate Next Steps (Phase 2)

The next work items, in order:

1. **Add softmax to Python kpu**
   - `kpu.softmax(x, dim=-1)` in `ops.py`
   - NumPy implementation in `runtime.py`
   - Test case

2. **Add layer_norm to Python kpu**
   - `kpu.layer_norm(x, normalized_shape)` in `ops.py`
   - NumPy implementation
   - Test case

3. **Add conv2d to Python kpu**
   - `kpu.conv2d(x, weight, bias, stride, padding)`
   - Use `numpy` (or `scipy.signal.correlate2d`) for BEHAVIORAL
   - Test case with MNIST

4. **Add pooling to Python kpu**
   - `kpu.max_pool2d(x, kernel_size, stride)`
   - `kpu.avg_pool2d(x, kernel_size, stride)`
   - NumPy strided view implementation

5. **Create simple CNN example**
   - `examples/mnist_cnn.py`
   - Validate against PyTorch

---

## File Changes Summary

### Files to Create (Phase 2-6)

| File | Purpose |
|------|---------|
| `python/kpu/nn.py` | Higher-level nn module (optional) |
| `python/kpu/model_loader.py` | Load JSON/ONNX models |
| `python/examples/mnist_cnn.py` | CNN example |
| `python/examples/squeezenet.py` | SqueezeNet example |
| `python/models/squeezenet.json` | Model definition |
| `python/models/squeezenet_weights.npz` | Pretrained weights |

### Files to Modify (Phase 2)

| File | Changes |
|------|---------|
| `python/kpu/ops.py` | Add conv2d, pool, softmax, layer_norm, concat |
| `python/kpu/tensor.py` | Add reshape, view methods |
| `python/kpu/graph.py` | Add CONV2D, POOL, SOFTMAX, LAYER_NORM, CONCAT to OpType |
| `python/kpu/dfx_emitter.py` | Emit new op types to DFX IR |
| `python/kpu/runtime.py` | Implement new ops in BEHAVIORAL mode |
| `python/tests/test_kpu.py` | Add tests for new ops |

---

## Related Documents

- `api-gaps-roadmap.md` - Original C++ gaps analysis (superseded by this document)
- `exaloop-integration-design.md` - Original Python interface design (Phase 1 reference)
- `FUNCTIONAL_SIMULATION_GAP_ANALYSIS.md` - Analysis of timing vs functional simulation
- `../06-compiler/dfx-specification.md` - DFX IR specification

---

*Document created: 2026-01-16*
*Consolidates: api-gaps-roadmap.md + exaloop-integration-design.md*
