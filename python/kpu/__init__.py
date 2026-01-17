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

__version__ = "0.1.0"
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
)

# Graph and DFX (for advanced users)
from .graph import OpGraph, OpNode, OpType
from .dfx_emitter import DFXProgram, DFXOp, DFXEmitter

# Datasets
from .datasets import MNIST, load_mnist

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

    return f"""KPU Python Package v{__version__}
  Fidelity: {fidelity_name}
  Native bindings: {'available' if runtime._native_sim is not None else 'not available (using pure Python)'}

Supported operations:
  - Matrix: matmul, linear
  - Convolution: conv2d
  - Pooling: max_pool2d, avg_pool2d, adaptive_avg_pool2d
  - Activation: relu, gelu, silu, sigmoid, tanh, softmax
  - Normalization: layer_norm
  - Elementwise: +, -, *, /, exp, log, sqrt
  - Reduction: sum, mean
  - Shape: reshape, transpose, concat, flatten
"""
