# KPU Python Package

High-level Python API for the KPU (Knowledge Processing Unit) simulator - a multi-fidelity neural network accelerator simulator with decorator-based compilation and comprehensive quantization support.

## Installation

```bash
pip install stillwater-kpu
```

Optional dependencies:
```bash
pip install stillwater-kpu[torch]     # PyTorch integration (torch.compile backend)
pip install stillwater-kpu[bfloat16]  # Native bfloat16 support via ml_dtypes
```

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

# Execute
result = mlp(x, w1, w2)
print(result.shape)  # (32, 10)
```

## Key Features

### Multi-Fidelity Simulation
- **BEHAVIORAL**: Computes actual values for functional correctness
- **TRANSACTIONAL**: Statistical timing model for performance estimation
- **CYCLE_ACCURATE**: Full timing simulation with C++ backend

### Comprehensive Operator Support
- **Matrix**: matmul, linear
- **Convolution**: conv2d with stride/padding
- **Attention**: scaled_dot_product_attention, multi_head_attention
- **Pooling**: max_pool2d, avg_pool2d, adaptive_avg_pool2d
- **Activation**: relu, gelu, silu, sigmoid, tanh, softmax
- **Normalization**: layer_norm, batch_norm2d
- **Elementwise**: exp, log, sqrt, +, -, *, /
- **Shape**: reshape, transpose, concat, flatten

### Quantization Support (v0.7.x)
Full quantization infrastructure for simulating low-precision inference:

| Type | Bits | Memory Reduction |
|------|------|------------------|
| FP16 | 16 | 2x |
| BF16 | 16 | 2x |
| INT8 | 8 | 4x |
| FP8 (E4M3/E5M2) | 8 | 4x |
| INT4 | 4 | 8x |
| FP4 | 4 | 8x |

```python
# INT8 quantization
from kpu import quantize, dequantize, compute_scale_zero_point

scale, zp = compute_scale_zero_point(weights)
w_int8 = quantize(weights, scale, zp)

# Calibration for post-training quantization
from kpu import CalibrationObserver, CalibrationMethod

observer = CalibrationObserver(method=CalibrationMethod.PERCENTILE)
for batch in calibration_data:
    observer.observe(activations)
params = observer.compute_params()
```

### Kernel Fusion
Automatic fusion of common patterns for reduced memory traffic:
- MatMul + Bias + ReLU/GELU/SiLU
- Conv2D + BatchNorm + Activation

```python
@kpu.compile(optimize=True)  # Fusion enabled by default
def fused_layer(x, w, b):
    return kpu.relu(x @ w + b)  # Fused into single operation
```

### PyTorch Integration
Use KPU as a `torch.compile` backend:

```python
import torch
model = torch.compile(my_model, backend="kpu")
output = model(input)

# With timing statistics
model = torch.compile(my_model, backend="kpu_transactional")
stats = kpu.get_torch_compile_stats()
print(f"Estimated cycles: {stats.cycles}")
```

## Simulation Modes

```python
import kpu

# Functional simulation (default)
kpu.set_fidelity(kpu.BEHAVIORAL)

# Performance estimation
kpu.set_fidelity(kpu.TRANSACTIONAL)
kpu.set_clock_frequency(1.0)  # 1 GHz

# Execute and get timing
result = model(input)
stats = model.get_stats()
print(f"Cycles: {stats.cycles}, GFLOPS: {stats.gflops:.1f}")
```

## Architecture

```
Python Code with @kpu.compile
        ↓
    Tracing (build OpGraph)
        ↓
    Fusion Optimization (optional)
        ↓
    DFX IR Emission
        ↓
    Runtime Execution
    ├── BEHAVIORAL (pure Python, computes values)
    ├── TRANSACTIONAL (C++ bindings, statistical timing)
    └── CYCLE_ACCURATE (C++ bindings, full timing)
```

## Examples

```python
# CNN for image classification
@kpu.compile
def cnn(x, conv_w, fc_w):
    h = kpu.relu(kpu.conv2d(x, conv_w, padding=1))
    h = kpu.max_pool2d(h, kernel_size=2)
    h = h.reshape(h.shape[0], -1)
    return h @ fc_w

# Transformer attention
@kpu.compile
def attention(q, k, v):
    return kpu.scaled_dot_product_attention(q, k, v)

# Quantized inference
from kpu import int4_linear, calibrate_percentile

params = calibrate_percentile(weights)
output = int4_linear(x, weights, params)
```

## Links

- [GitHub Repository](https://github.com/stillwater-sc/kpu-sim)
- [Documentation](https://github.com/stillwater-sc/kpu-sim/tree/main/docs)
- [Issue Tracker](https://github.com/stillwater-sc/kpu-sim/issues)

## License

Apache-2.0
