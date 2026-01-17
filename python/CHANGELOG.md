# Changelog

All notable changes to the KPU Python package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-01-17

### Added

#### torch.compile Backend
- `torch.compile(model, backend="kpu")` - Full integration with PyTorch's compiler infrastructure
- `KPUBackend` class - Custom backend that converts FX graphs to KPU operations
- `FXToKPUConverter` - Comprehensive FX graph to KPU operation converter (~1000 lines)
- Support for all common DNN operations via Dynamo-captured graphs

#### FX Converter Operations
- **Activations**: relu, gelu, silu, sigmoid, tanh, softmax
- **Convolutions**: conv2d with stride/padding from positional args or kwargs
- **Pooling**: max_pool2d, avg_pool2d, adaptive_avg_pool2d
- **Normalization**: batch_norm, layer_norm
- **Linear**: linear (handles both F.linear and torch._C._nn.linear)
- **Matrix**: matmul, mm, bmm
- **Elementwise**: add, sub, mul, div
- **Shape**: reshape, view, flatten, transpose, permute
- **Reductions**: mean, sum
- **Other**: concat, getitem, identity (for contiguous, dropout)

#### Backend Variants
- `torch.compile(model, backend="kpu")` - Default behavioral mode
- `torch.compile(model, backend="kpu_behavioral")` - Explicit behavioral
- `torch.compile(model, backend="kpu_transactional")` - Transactional mode
- `torch.compile(model, backend="kpu_cycle_accurate")` - Cycle-accurate mode

#### Examples
- `examples/torch_compile_demo.py` - Demo with MLP, CNN, and function compilation

### Fixed

- **Dynamo placeholder handling**: Correctly handles Dynamo-captured graphs where ALL parameters become placeholders. At runtime, Dynamo passes both parameters and inputs as arguments.
- **Output tuple format**: Dynamo expects outputs as a tuple, not a single tensor. Fixed `_build_executable` to always return `tuple(outputs)`.
- **conv2d positional args**: Fixed `_emit_conv2d` to read stride and padding from positional args (not just kwargs), as Dynamo passes them as `torch.conv2d(input, weight, bias, stride, padding, dilation, groups)`.

### Changed

- `__init__.py` - Added torch_backend imports with proper torch detection
- `kpu/torch_backend.py` - **NEW**: Backend registration and KPUBackend class
- `kpu/fx_converter.py` - **NEW**: FX graph to KPU converter

## [0.2.0] - 2026-01-16

### Added

#### CNN Operators
- `conv2d` - 2D convolution with stride, padding, dilation support
- `max_pool2d` - Max pooling with configurable kernel and stride
- `avg_pool2d` - Average pooling with configurable kernel and stride
- `adaptive_avg_pool2d` - Adaptive average pooling to target output size
- `layer_norm` - Layer normalization with optional scale/bias
- `batch_norm2d` - Batch normalization for 4D tensors (N,C,H,W)
- `concat` - Concatenate tensors along a dimension
- `flatten` - Flatten tensor dimensions

#### Tensor Methods
- `Tensor.reshape(*shape)` - Reshape tensor to new dimensions
- `Tensor.flatten(start_dim, end_dim)` - Flatten range of dimensions
- `Tensor.view(*shape)` - Alias for reshape (PyTorch compatibility)

#### Data Loading
- `kpu.datasets.MNIST` - MNIST dataset loader class
- `kpu.load_mnist(split, normalize, flatten, limit)` - Convenience function

#### Examples
- `examples/mnist_cnn.py` - Complete MNIST CNN with validation
- `examples/mnist_real_validation.py` - Real MNIST data validation

#### Tests
- `tests/test_cnn_validation.py` - Comprehensive CNN operator validation suite

### Fixed

- **Dynamic batch size in reshape**: Runtime now recomputes first dimension when traced shape doesn't match input size. This allows CNNs traced with one batch size to run with different batch sizes.
- **MNIST download URLs**: Updated from yann.lecun.com (404) to PyTorch's S3 mirror (ossci-datasets.s3.amazonaws.com)

### Changed

- `runtime.py` - Added behavioral execution for all new CNN operators
- `graph.py` - Added OpTypes for CNN operations
- `dfx_emitter.py` - Added DFXOpCodes for CNN operations
- `__init__.py` - Exported new operators and MNIST loader

## [0.1.0] - 2026-01-16

### Added

#### Core Package
- `kpu.Tensor` - Tensor class with tracing support for `@`, `+`, `-`, `*`, `/` operators
- `kpu.TensorMeta` - Metadata class for tensor shape, dtype, memory level
- `@kpu.compile` - Decorator for tracing functions to DFX IR
- `@kpu.jit` - Alias for compile decorator

#### Operators
- `kpu.relu` - ReLU activation
- `kpu.gelu` - GELU activation
- `kpu.silu` - SiLU/Swish activation
- `kpu.sigmoid` - Sigmoid activation
- `kpu.tanh` - Tanh activation
- `kpu.softmax` - Softmax normalization
- `kpu.matmul` - Matrix multiplication
- `kpu.linear` - Linear layer (matmul + bias)
- `kpu.sum` - Sum reduction
- `kpu.mean` - Mean reduction
- `kpu.exp`, `kpu.log`, `kpu.sqrt` - Elementwise math

#### Runtime
- `KPURuntime` - Runtime executor with fidelity levels
- `BEHAVIORAL` - Pure Python execution (computes actual values)
- `TRANSACTIONAL` - Statistical timing model (requires C++ bindings)
- `CYCLE_ACCURATE` - Full timing simulation (requires C++ bindings)
- `set_fidelity()` / `get_fidelity()` - Fidelity control

#### DFX IR
- `DFXProgram` - IR program representation
- `DFXOp` - Individual operation in IR
- `DFXEmitter` - Generate DFX from OpGraph
- JSON serialization/deserialization

#### Examples
- `examples/mnist_mlp.py` - MNIST MLP with 784->128->64->10 architecture

#### Tests
- `tests/test_kpu.py` - 20 tests covering tensors, operators, compiler, DFX
