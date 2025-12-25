# KPU Python Examples

This directory contains Python examples demonstrating the Stillwater KPU simulator.

## Setup

Before running examples, ensure the Python bindings are built:

```bash
# From repository root
cmake --build build --target stillwater_kpu
```

The examples automatically add the build directory to the Python path.

## Kernel and Graph Examples

### Main Examples

| Script | Description |
|--------|-------------|
| `kernel_graph_example.py` | Comprehensive demo of kernels, compilation, and graphs |
| `address_based_dma_example.py` | DMA transfers using unified address space |

### Running the Kernel Graph Example

```bash
python3 examples/python/kernel_graph_example.py
```

This demonstrates:
- Creating matmul and MLP kernels
- Using the KernelCompiler with auto-optimization
- Building multi-kernel computation graphs
- Analyzing execution order and parallelism
- DOT graph visualization
- Configuring the concurrent executor

## Test Suites

### Validation and Regression Tests

| Script | Description |
|--------|-------------|
| `test_kernel_validation.py` | Validates kernel creation and metadata |
| `test_graph_validation.py` | Validates graph construction and analysis |
| `test_compiler_regression.py` | Regression tests for the compiler |

### Running Tests

```bash
# Run all test suites
python3 examples/python/test_kernel_validation.py
python3 examples/python/test_graph_validation.py
python3 examples/python/test_compiler_regression.py

# Or run a specific test
python3 -c "
import sys; sys.path.insert(0, 'build/src/bindings/python/Release')
import stillwater_kpu as kpu
kernel = kpu.Kernel.create_matmul(64, 64, 64)
print(f'Kernel: {kernel.name()}, FLOPs: {kernel.total_flops():,}')
"
```

## Legacy Examples

### Basic Examples (`baseline/`)
- `simple_kpu.py` - First steps with KPU
- `core_features_demo.py` - Core module utilities
- `system_info_demo.py` - System diagnostics
- `timing_demo.py` - Performance measurement

### Educational Examples (`educational/`)
- `neural_network_demo.py` - Neural network layer computation
- `performance_analysis.py` - Scaling analysis
- `blocked_matmul.py` - Memory-efficient algorithms

## Quick Reference

### Import the Module

```python
import sys
sys.path.insert(0, 'build/src/bindings/python/Release')
import stillwater_kpu as kpu
```

### Create Kernels

```python
# Matrix multiplication
matmul = kpu.Kernel.create_matmul(M=1024, N=1024, K=1024)

# MLP with activation
mlp = kpu.Kernel.create_mlp(
    M=64, N=512, K=256,
    activation=kpu.ActivationType.GELU,
    has_bias=True
)
```

### Compile with Optimization

```python
compiler = kpu.KernelCompiler()
kernel = compiler.compile_matmul(1024, 1024, 1024)

stats = compiler.last_stats()
print(f"Tiles: {stats.total_tiles}")
print(f"Arithmetic Intensity: {stats.estimated_arithmetic_intensity:.2f}")
```

### Build Kernel Graphs

```python
graph = kpu.KernelGraph("my_network")

fc1 = graph.add_kernel(kpu.Kernel.create_mlp(64, 256, 128, kpu.ActivationType.RELU, True), "fc1")
fc2 = graph.add_kernel(kpu.Kernel.create_mlp(64, 64, 256, kpu.ActivationType.NONE, True), "fc2")
graph.add_edge(fc1, fc2, "C", "A")

result = graph.compile()
```

## Documentation

For comprehensive documentation, see:
- `docs/python-integration-guide.md` - Complete Python API guide
- `docs/multi-kernel-fusion-support.md` - Multi-kernel graph documentation

## Directory Structure

```
examples/python/
├── README.md                      # This file
├── kernel_graph_example.py        # Main kernel/graph demo
├── address_based_dma_example.py   # DMA example
├── test_kernel_validation.py      # Kernel tests
├── test_graph_validation.py       # Graph tests
├── test_compiler_regression.py    # Compiler tests
├── generate.py                    # Code generation utilities
├── baseline/                      # Basic examples
│   ├── simple_kpu.py
│   └── ...
└── educational/                   # Learning examples
    ├── neural_network_demo.py
    └── ...
```
