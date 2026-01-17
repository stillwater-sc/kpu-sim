# KPU Python Package

High-level Python API for the KPU (Knowledge Processing Unit) simulator with decorator-based compilation.

## Quick Start

```python
import kpu
import numpy as np

# Define a neural network with @kpu.compile
@kpu.compile
def mlp(x, w1, w2):
    h = kpu.relu(x @ w1)
    return h @ w2

# Create tensors
x = kpu.Tensor(np.random.randn(32, 784).astype(np.float32))
w1 = kpu.Tensor(np.random.randn(784, 128).astype(np.float32))
w2 = kpu.Tensor(np.random.randn(128, 10).astype(np.float32))

# Execute (computes actual values in BEHAVIORAL mode)
result = mlp(x, w1, w2)
print(result.shape)  # (32, 10)

# Inspect generated DFX IR
print(mlp.get_dfx().to_json())
```

## Features

- **Decorator-based compilation**: Use `@kpu.compile` to compile Python functions to KPU programs
- **NumPy-compatible tensors**: `kpu.Tensor` wraps NumPy arrays with compilation support
- **Multi-fidelity simulation**:
  - `BEHAVIORAL`: Computes actual values (functional correctness)
  - `TRANSACTIONAL`: Statistical timing model
  - `CYCLE_ACCURATE`: Full timing simulation
- **DFX IR generation**: Inspectable intermediate representation

## Supported Operations

### Matrix Operations
- `@` (matmul): Matrix multiplication
- `kpu.linear`: Linear layer (y = x @ W^T + b)

### Activation Functions
- `kpu.relu`: Rectified Linear Unit
- `kpu.gelu`: Gaussian Error Linear Unit
- `kpu.silu`: Sigmoid Linear Unit (Swish)
- `kpu.sigmoid`: Sigmoid
- `kpu.tanh`: Hyperbolic tangent
- `kpu.softmax`: Softmax

### Elementwise Operations
- `+`, `-`, `*`, `/`: Arithmetic operations
- `kpu.exp`, `kpu.log`, `kpu.sqrt`: Math functions

### Reduction Operations
- `kpu.sum`, `kpu.mean`: Aggregations

## Installation

```bash
# From the kpu-sim repository
cd python
pip install -e .

# Run tests
pytest tests/ -v
```

## Examples

See `examples/mnist_mlp.py` for a complete MNIST classifier example.

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
