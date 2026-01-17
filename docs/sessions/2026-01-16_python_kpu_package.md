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

## Related Documents

- `docs/09-virtual-platform/exaloop-integration-design.md` - Exaloop/Codon integration design
- `docs/09-virtual-platform/qemu-vs-userspace-runtime.md` - Runtime architecture analysis
- `docs/06-compiler/dfx-specification.md` - DFX IR specification
