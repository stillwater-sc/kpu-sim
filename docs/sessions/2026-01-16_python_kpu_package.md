# Session Log: Python KPU Package Implementation

**Date:** 2026-01-16
**Duration:** ~2 hours
**Focus:** Implementation of high-level Python API for KPU simulator with decorator-based compilation

## Summary

Created a complete Python package (`python/kpu/`) that provides a PyTorch/NumPy-like interface for defining and executing neural network computations on the KPU simulator. The package uses a `@kpu.compile` decorator to trace Python functions and generate DFX IR (Domain Flow Execution Intermediate Representation) for the compiler team.

## Context

The compiler team uses an Exaloop/Codon-based toolchain that compiles Python/NumPy code directly to x86/ARM executables. They needed a functional simulator interface to:
1. Validate correctness of compiled kernels against NumPy reference
2. Generate DFX IR for the KPU compiler pipeline
3. Eventually integrate with C++ kpu-sim for timing simulation

## Architecture

```
Python Code with @kpu.compile
        ↓
    Tracing (build OpGraph)
        ↓
    DFX IR Emission
        ↓
    Runtime Execution
    ├── BEHAVIORAL (pure Python, computes values)
    ├── TRANSACTIONAL (C++ bindings, statistical)
    └── CYCLE_ACCURATE (C++ bindings, full timing)
```

## Files Created

### Core Package (`python/kpu/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports, version info |
| `tensor.py` | `Tensor` class with tracing support for `@`, `+`, `-`, `*`, `/` operators |
| `ops.py` | Operator functions: `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `softmax`, `sum`, `mean`, `matmul`, `linear` |
| `graph.py` | `OpGraph` class for operation DAG with topological ordering |
| `dfx_emitter.py` | `DFXProgram` generation from OpGraph, JSON serialization/deserialization |
| `compiler.py` | `@kpu.compile` decorator, `CompiledFunction` wrapper with lazy compilation |
| `runtime.py` | `KPURuntime` with BEHAVIORAL execution using NumPy |

### Native Bindings (`python/kpu/_native/`)

| File | Purpose |
|------|---------|
| `kpu_native.cpp` | pybind11 bindings for C++ execution (optional acceleration) |
| `CMakeLists.txt` | Build configuration for native module |
| `__init__.py` | Package init with fallback detection |

### Examples and Tests

| File | Purpose |
|------|---------|
| `examples/mnist_mlp.py` | Complete MNIST MLP example (784→128→64→10) |
| `tests/test_kpu.py` | Test suite: 20 tests covering tensors, operators, compiler, DFX |
| `pyproject.toml` | Package configuration for pip install |
| `README.md` | Quick start guide and API documentation |

### Build System Changes

| File | Change |
|------|--------|
| `CMakeLists.txt` | Added section to build `python/kpu/_native` when pybind11 available |

## Key Design Decisions

### 1. Tracing-Based Compilation
The `@kpu.compile` decorator uses Python's dynamic nature to trace operations:
```python
@kpu.compile
def mlp(x, w1, w2):
    h = kpu.relu(x @ w1)
    return h @ w2

# First call traces the function, subsequent calls execute cached graph
result = mlp(x, w1, w2)
```

### 2. DFX IR as Interchange Format
The DFX IR provides a JSON-serializable representation that:
- Captures tensor shapes, dtypes, and memory levels
- Records operation sequence with inputs/outputs
- Includes operation-specific attributes (M, N, K for matmul)
- Can be loaded by the C++ compiler pipeline

### 3. Pure Python BEHAVIORAL Mode
The BEHAVIORAL runtime computes actual values using NumPy, enabling:
- Functional correctness verification
- No C++ dependencies for initial development
- Easy debugging and testing

### 4. Optional Native Bindings
The `_native` module is optional:
- Package works in pure Python mode
- Native bindings provide potential performance boost
- Graceful fallback if bindings not built

## Test Results

All 20 tests pass:

```
TestTensor:           3 tests (creation, factories)
TestOperators:        8 tests (matmul, relu, gelu, sigmoid, softmax, elementwise)
TestCompiler:         5 tests (simple matmul, single layer, two-layer MLP, graph/DFX generation)
TestMNISTMLP:         2 tests (full MNIST MLP, XOR classifier)
TestDFXEmitter:       2 tests (serialization, deserialization)
```

MNIST MLP verification:
```
Max difference from NumPy reference: 0.00e+00
MatMul FLOPs: 6,987,776
Operations: 8 (3 matmul, 3 add, 2 relu)
```

## Example Usage

```python
import kpu
import numpy as np

# Set fidelity (BEHAVIORAL computes actual values)
kpu.set_fidelity(kpu.BEHAVIORAL)

# Define network
@kpu.compile
def mnist_mlp(x, w1, b1, w2, b2, w3, b3):
    h1 = kpu.relu(x @ w1 + b1)
    h2 = kpu.relu(h1 @ w2 + b2)
    return h2 @ w3 + b3

# Create tensors
x = kpu.Tensor(np.random.randn(32, 784).astype(np.float32))
# ... weights and biases ...

# Execute
logits = mnist_mlp(x, w1, b1, w2, b2, w3, b3)

# Inspect DFX IR
print(mnist_mlp.get_dfx().to_json())
```

## DFX IR Output Format

```json
{
  "name": "mnist_mlp",
  "version": "1.0",
  "tensors": {
    "input": {"shape": [32, 784], "dtype": "f32", ...},
    "w1": {"shape": [784, 128], "dtype": "f32", ...},
    ...
  },
  "ops": [
    {"opcode": "matmul", "inputs": ["input", "w1"], "outputs": ["t0"], "attrs": {"M": 32, "K": 784, "N": 128}},
    {"opcode": "add", "inputs": ["t0", "b1"], "outputs": ["t1"]},
    {"opcode": "relu", "inputs": ["t1"], "outputs": ["t2"]},
    ...
  ],
  "inputs": ["input", "w1", "b1", ...],
  "outputs": ["t7"]
}
```

## Files to Check In

All files in `python/kpu/` should be checked in:
- Source files: `*.py`, `*.cpp`, `CMakeLists.txt`
- `.gitignore` already excludes `*.so`, `*.pyd`, `__pycache__/`

## Next Steps

1. **Exaloop/Codon Integration**: Connect DFX IR output to Codon's compilation pipeline
2. **Native Bindings Build**: Build `_native` module when CMake 4.2+ available
3. **C++ Simulator Integration**: Connect native bindings to full kpu-sim for timing
4. **Additional Operators**: Add conv2d, attention, layer normalization as needed
5. **PyTorch Compatibility**: Consider torch.Tensor interoperability

---

## Continuation Session (Later Same Day)

**Focus:** Phase 2 (CNN Operators) and Phase 3 (Validation)

### Summary

Extended the Python KPU package with comprehensive CNN support, including convolution, pooling, normalization operators, and an MNIST data loader. Fixed a critical bug where traced reshape operations captured batch sizes at trace time, causing failures with dynamic batch sizes at runtime.

### Phase 2: CNN Operators Added

| Operator | Description |
|----------|-------------|
| `conv2d` | 2D convolution with stride, padding, dilation support |
| `max_pool2d` | Max pooling with configurable kernel and stride |
| `avg_pool2d` | Average pooling with configurable kernel and stride |
| `adaptive_avg_pool2d` | Adaptive average pooling to target output size |
| `layer_norm` | Layer normalization with optional scale/bias |
| `batch_norm2d` | Batch normalization for 4D tensors (N,C,H,W) |
| `concat` | Concatenate tensors along a dimension |
| `flatten` | Flatten tensor dimensions |
| `Tensor.reshape()` | Reshape tensor (instance method) |
| `Tensor.view()` | Alias for reshape (PyTorch compatibility) |

### Phase 3: Validation and Bug Fixes

#### Bug Fixed: Dynamic Batch Size in Reshape

**Problem:** When a CNN was traced with batch_size=32, the reshape operation captured `(32, 800)` as the target shape. Running with a different batch size (e.g., 4) caused:
```
ValueError: cannot reshape array of size 3200 into shape (32,800)
```

**Root Cause:** The expression `h.reshape(h.shape[0], -1)` evaluates `h.shape[0]` at trace time, capturing the tracing batch size.

**Fix:** Modified `runtime.py` to dynamically recompute the first dimension when there's a size mismatch:
```python
elif op.opcode == DFXOpCode.RESHAPE:
    shape = list(op.attrs.get('shape'))
    x = inputs[0]
    total_size = x.size

    # Handle -1 dimensions
    if -1 in shape:
        neg_idx = shape.index(-1)
        other_size = 1
        for i, s in enumerate(shape):
            if i != neg_idx:
                other_size *= s
        if other_size > 0:
            shape[neg_idx] = total_size // other_size

    # If size mismatch, recompute first dimension (batch)
    target_size = 1
    for s in shape:
        target_size *= s
    if target_size != total_size and len(shape) > 1:
        other_size = 1
        for s in shape[1:]:
            other_size *= s
        if other_size > 0 and total_size % other_size == 0:
            shape[0] = total_size // other_size

    result = x.reshape(tuple(shape))
```

#### Bug Fixed: MNIST URLs (404 Error)

**Problem:** The original MNIST URLs from yann.lecun.com returned 404 errors.

**Fix:** Updated `datasets.py` to use PyTorch's S3 mirror:
```python
URLS = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    ...
}
```

### Files Created/Modified

| File | Changes |
|------|---------|
| `kpu/ops.py` | Added ~450 lines: CNN operators (conv2d, pooling, normalization) |
| `kpu/tensor.py` | Added reshape(), flatten(), view() methods |
| `kpu/graph.py` | Added OpTypes: CONV2D, LAYER_NORM, BATCH_NORM, MAXPOOL2D, AVGPOOL2D, etc. |
| `kpu/dfx_emitter.py` | Added DFXOpCodes for new operations |
| `kpu/runtime.py` | Added behavioral execution for CNN ops, fixed reshape |
| `kpu/__init__.py` | Exported new operators and MNIST loader |
| `kpu/datasets.py` | **NEW**: MNIST data loader with S3 mirror URLs |
| `tests/test_cnn_validation.py` | **NEW**: Comprehensive CNN validation suite |
| `examples/mnist_cnn.py` | **NEW**: MNIST CNN example with validation |
| `examples/mnist_real_validation.py` | **NEW**: Real MNIST data validation |

### Test Results

**CNN Validation Suite:**
```
conv2d vs NumPy: PASS (max diff ~1e-6)
pooling vs NumPy: PASS (diff=0.00)
layer_norm vs NumPy: PASS (diff=0.00)
full CNN pipeline: PASS (max diff=1.19e-06)
traced vs direct: PASS (diff=0.00)
ALL TESTS PASSED
```

**Dynamic Batch Test:**
```
Batch 8: output shape = (8, 10)
Batch 4: output shape = (4, 10)
Batch 1: output shape = (1, 10)
Batch 16: output shape = (16, 10)
Dynamic batch size test: PASSED
```

**Real MNIST Test:**
```
Loaded 20 test images: (20, 1, 28, 28)
Accuracy: 20.00% (random weights expected ~10%)
Validation complete!
```

### MNIST CNN Architecture

```python
@kpu.compile
def mnist_cnn(x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b):
    # Conv1 + ReLU + Pool: (N,1,28,28) -> (N,16,13,13)
    h = kpu.relu(kpu.conv2d(x, conv1_w) + conv1_b.reshape(1, -1, 1, 1))
    h = kpu.max_pool2d(h, kernel_size=2, stride=2)

    # Conv2 + ReLU + Pool: (N,16,13,13) -> (N,32,5,5)
    h = kpu.relu(kpu.conv2d(h, conv2_w) + conv2_b.reshape(1, -1, 1, 1))
    h = kpu.max_pool2d(h, kernel_size=2, stride=2)

    # Flatten + FC: (N,32,5,5) -> (N,800) -> (N,128) -> (N,10)
    h = h.reshape(h.shape[0], -1)
    h = kpu.relu(h @ fc1_w + fc1_b)
    return h @ fc2_w + fc2_b
```

### Remaining Work (Future Sessions)

1. **Performance Optimization**: Current conv2d uses nested Python loops - need Cython/NumPy optimization
2. **PyTorch Validation**: Install PyTorch for cross-validation testing
3. **More Operators**: Attention mechanisms, grouped convolution, depthwise separable conv
4. **Native Bindings**: Connect to C++ kpu-sim for timing simulation

---

## Related Documents

- `docs/09-virtual-platform/exaloop-integration-design.md` - Exaloop/Codon integration design
- `docs/09-virtual-platform/unified-dnn-roadmap.md` - Consolidated DNN implementation roadmap
- `docs/09-virtual-platform/qemu-vs-userspace-runtime.md` - Runtime architecture analysis
- `docs/06-compiler/dfx-specification.md` - DFX IR specification
