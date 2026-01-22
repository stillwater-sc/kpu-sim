# python/kpu/__init__.py
"""
KPU - Knowledge Processing Unit Simulator Python API

A high-level Python interface for the KPU simulator with decorator-based
compilation for neural network execution.

Example:
    >>> import kpu
    >>> import numpy as np

    >>> @kpu.compile
    ... def mlp(x, w1, w2):
    ...     h = kpu.relu(x @ w1)
    ...     return h @ w2

    >>> x = kpu.Tensor(np.random.randn(32, 784).astype(np.float32))
    >>> w1 = kpu.Tensor(np.random.randn(784, 128).astype(np.float32))
    >>> w2 = kpu.Tensor(np.random.randn(128, 10).astype(np.float32))

    >>> result = mlp(x, w1, w2)
    >>> print(result.shape)  # (32, 10)

Fidelity Levels:
    - BEHAVIORAL: Functional correctness, computes actual values
    - TRANSACTIONAL: Performance estimation, statistical timing
    - CYCLE_ACCURATE: Full timing simulation
"""

# Suppress irrelevant PyTorch C++ warnings (like NNPACK) before any imports
# These warnings are written directly to stderr from C++ code
import sys as _sys
import os as _os

def _install_cpp_warning_filter():
    """Install a filter to suppress irrelevant PyTorch C++ warnings.

    NNPACK warnings are common when running on systems without NNPACK support,
    but they're irrelevant when using the KPU backend.
    """
    _suppressed = [b'NNPACK', b'Could not initialize NNPACK']

    # Save original stderr fd
    _stderr_fd = _sys.stderr.fileno()
    _saved_fd = _os.dup(_stderr_fd)

    # Create pipe to capture stderr
    _read_fd, _write_fd = _os.pipe()
    _os.dup2(_write_fd, _stderr_fd)
    _os.close(_write_fd)

    def _flush_filtered_stderr():
        """Read captured stderr and write filtered content."""
        try:
            _os.set_blocking(_read_fd, False)
            captured = _os.read(_read_fd, 1024 * 1024)
            for line in captured.split(b'\n'):
                if line and not any(s in line for s in _suppressed):
                    _sys.stderr.buffer.write(line + b'\n')
                    _sys.stderr.buffer.flush()
        except (BlockingIOError, OSError):
            pass

    def _restore_stderr():
        """Restore original stderr."""
        _flush_filtered_stderr()
        _os.dup2(_saved_fd, _stderr_fd)
        _os.close(_saved_fd)
        _os.close(_read_fd)

    # Register cleanup at exit
    import atexit
    atexit.register(_restore_stderr)

    # Also flush periodically during imports
    return _flush_filtered_stderr

_flush_stderr = _install_cpp_warning_filter()

__version__ = "0.7.1"
__author__ = "Stillwater Supercomputing, Inc."

# Fidelity levels
from .compiler import BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE

# Core classes
from .tensor import Tensor, TensorMeta

# Compiler decorator
from .compiler import compile, jit, CompiledFunction

# Operators
from .ops import (
    # Activation functions
    relu,
    gelu,
    silu,
    sigmoid,
    tanh,
    softmax,
    # Normalization
    layer_norm,
    batch_norm2d,
    # Convolution
    conv2d,
    # Attention
    scaled_dot_product_attention,
    multi_head_attention,
    # Pooling
    max_pool2d,
    avg_pool2d,
    adaptive_avg_pool2d,
    # Elementwise operations
    exp,
    log,
    sqrt,
    # Reduction operations
    sum,
    mean,
    # Shape operations
    reshape,
    transpose,
    concat,
    flatten,
    # Matrix operations
    matmul,
    linear,
)

# Runtime
from .runtime import (
    KPURuntime,
    ExecutionStats,
    get_runtime,
    set_fidelity,
    get_fidelity,
    get_fidelity_name,
    set_clock_frequency,
    get_clock_frequency,
    is_clock_frequency_set,
)

# Graph and DFX (for advanced users)
from .graph import OpGraph, OpNode, OpType
from .dfx_emitter import DFXProgram, DFXOp, DFXEmitter

# Datasets
from .datasets import MNIST, load_mnist

# Fusion (v0.6.0+)
from .fusion import (
    FusionCompiler,
    FusionPattern,
    FusionGroup,
    estimate_memory_savings,
    # v0.6.1: Fusion analysis
    FusionAnalyzer,
    FusionReport,
    FusionOpportunity,
    RooflineMetrics,
    analyze_fusion_potential,
    # v0.6.2: Conv2D fusion patterns
    Conv2DBatchNormActivation,
    Conv2DActivation,
)

# Quantization (v0.7.0+)
from .quantization import (
    # Dtypes and config
    QuantDtype,
    QuantizationConfig,
    # Calibration
    compute_scale_zero_point,
    compute_per_channel_params,
    # Quantize/dequantize
    quantize,
    dequantize,
    quantize_per_channel,
    dequantize_per_channel,
    # Quantized operations (INT8)
    quantized_matmul_int8,
    quantized_linear_int8,
    # FP16 operations (v0.7.1)
    fp16_matmul,
    fp16_linear,
    fp16_conv2d,
    cast_to_fp16,
    cast_from_fp16,
    fp16_range,
    fp16_precision,
    # Memory traffic
    calculate_memory_bytes,
    calculate_matmul_traffic,
    bandwidth_reduction_factor,
)

# torch.compile backend (optional, requires PyTorch)
try:
    import torch as _torch  # Check if torch is actually importable
    from . import torch_backend
    from .torch_backend import compile as torch_compile
    from .torch_backend import get_last_stats as get_torch_compile_stats
    TORCH_AVAILABLE = True
    del _torch
except ImportError:
    torch_backend = None
    torch_compile = None
    get_torch_compile_stats = None
    TORCH_AVAILABLE = False

__all__ = [
    # Version
    "__version__",

    # Fidelity levels
    "BEHAVIORAL",
    "TRANSACTIONAL",
    "CYCLE_ACCURATE",

    # Core classes
    "Tensor",
    "TensorMeta",

    # Compiler
    "compile",
    "jit",
    "CompiledFunction",

    # Operators - activation
    "relu",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "softmax",

    # Operators - normalization
    "layer_norm",
    "batch_norm2d",

    # Operators - convolution
    "conv2d",

    # Operators - attention
    "scaled_dot_product_attention",
    "multi_head_attention",

    # Operators - pooling
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",

    # Operators - elementwise
    "exp",
    "log",
    "sqrt",

    # Operators - reduction
    "sum",
    "mean",

    # Operators - shape
    "reshape",
    "transpose",
    "concat",
    "flatten",

    # Operators - matrix
    "matmul",
    "linear",

    # Runtime
    "KPURuntime",
    "ExecutionStats",
    "get_runtime",
    "set_fidelity",
    "get_fidelity",
    "get_fidelity_name",
    "set_clock_frequency",
    "get_clock_frequency",
    "is_clock_frequency_set",

    # Graph (advanced)
    "OpGraph",
    "OpNode",
    "OpType",

    # DFX (advanced)
    "DFXProgram",
    "DFXOp",
    "DFXEmitter",

    # Datasets
    "MNIST",
    "load_mnist",

    # Fusion (v0.6.0+)
    "FusionCompiler",
    "FusionPattern",
    "FusionGroup",
    "estimate_memory_savings",
    # v0.6.1: Fusion analysis
    "FusionAnalyzer",
    "FusionReport",
    "FusionOpportunity",
    "RooflineMetrics",
    "analyze_fusion_potential",
    # v0.6.2: Conv2D fusion patterns
    "Conv2DBatchNormActivation",
    "Conv2DActivation",

    # Quantization (v0.7.0+)
    "QuantDtype",
    "QuantizationConfig",
    "compute_scale_zero_point",
    "compute_per_channel_params",
    "quantize",
    "dequantize",
    "quantize_per_channel",
    "dequantize_per_channel",
    "quantized_matmul_int8",
    "quantized_linear_int8",
    # FP16 operations (v0.7.1)
    "fp16_matmul",
    "fp16_linear",
    "fp16_conv2d",
    "cast_to_fp16",
    "cast_from_fp16",
    "fp16_range",
    "fp16_precision",
    # Memory traffic
    "calculate_memory_bytes",
    "calculate_matmul_traffic",
    "bandwidth_reduction_factor",

    # torch.compile backend
    "torch_backend",
    "torch_compile",
    "get_torch_compile_stats",
    "TORCH_AVAILABLE",
]


def version() -> str:
    """Return version string."""
    return __version__


def info() -> str:
    """Return information about the KPU package."""
    from .runtime import get_runtime

    fidelity_names = {
        BEHAVIORAL: "BEHAVIORAL",
        TRANSACTIONAL: "TRANSACTIONAL",
        CYCLE_ACCURATE: "CYCLE_ACCURATE",
    }

    runtime = get_runtime()
    fidelity_name = fidelity_names.get(runtime.fidelity, "UNKNOWN")

    torch_status = "available (torch.compile backend registered)" if TORCH_AVAILABLE else "not available (install PyTorch)"

    return f"""KPU Python Package v{__version__}
  Fidelity: {fidelity_name}
  Native bindings: {'available' if runtime._native_sim is not None else 'not available (using pure Python)'}
  PyTorch integration: {torch_status}

Supported operations:
  - Matrix: matmul, linear
  - Convolution: conv2d
  - Attention: scaled_dot_product_attention, multi_head_attention
  - Pooling: max_pool2d, avg_pool2d, adaptive_avg_pool2d
  - Activation: relu, gelu, silu, sigmoid, tanh, softmax
  - Normalization: layer_norm, batch_norm2d
  - Elementwise: +, -, *, /, exp, log, sqrt
  - Reduction: sum, mean
  - Shape: reshape, transpose, concat, flatten

torch.compile usage:
  import torch
  model = torch.compile(my_model, backend="kpu")
  output = model(input)

torch.compile with timing (TRANSACTIONAL mode):
  model = torch.compile(my_model, backend="kpu_transactional")
  output = model(input)
  stats = kpu.get_torch_compile_stats()
  print(f"Cycles: {{stats.cycles}}, GFLOPS: {{stats.gflops:.1f}}")
"""
