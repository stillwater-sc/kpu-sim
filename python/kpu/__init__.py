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

    Note: This filter is disabled when running under pytest to avoid
    breaking pytest's capture mechanism.
    """
    # Skip if running under pytest - the pipe redirection breaks pytest's capture
    if 'pytest' in _sys.modules or '_pytest' in _sys.modules:
        return lambda: None

    # Skip if stderr is not a real file (e.g., already redirected)
    try:
        _stderr_fd = _sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        return lambda: None

    _suppressed = [b'NNPACK', b'Could not initialize NNPACK']

    # Save original stderr fd
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

__version__ = "0.7.12"
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
    # BF16 operations (v0.7.2)
    bf16_matmul,
    bf16_linear,
    bf16_conv2d,
    cast_to_bf16,
    cast_from_bf16,
    bf16_range,
    bf16_precision,
    is_bfloat16_native,
    # FP8 operations (v0.7.3)
    FP8Format,
    FP8_E2M5,
    FP8_E3M4,
    FP8_E4M3,
    FP8_E5M2,
    get_fp8_format,
    is_fp8_native,
    fp8_matmul,
    fp8_linear,
    cast_to_fp8,
    cast_from_fp8,
    fp8_range,
    fp8_precision,
    fp8_info,
    # INT4 operations (v0.7.7)
    pack_int4,
    unpack_int4,
    quantize_int4,
    dequantize_int4,
    compute_int4_scale_zero_point,
    int4_matmul,
    int4_linear,
    int4_packed_size,
    int4_memory_bytes,
    int4_info,
    INT4_SIGNED_MIN,
    INT4_SIGNED_MAX,
    INT4_UNSIGNED_MIN,
    INT4_UNSIGNED_MAX,
    # FP4 operations (v0.7.8)
    FP4Format,
    FP4_E2M1,
    FP4_E1M2,
    get_fp4_format,
    fp4_quantize,
    fp4_dequantize,
    pack_fp4,
    unpack_fp4,
    fp4_matmul,
    fp4_linear,
    fp4_range,
    fp4_values,
    fp4_info,
    # Mixed precision (v0.7.9)
    MixedPrecisionConfig,
    MIXED_INT8_FP16,
    MIXED_INT8_BF16,
    MIXED_INT4_FP16,
    MIXED_FP8_FP16,
    MIXED_FP8_BF16,
    mixed_precision_linear,
    mixed_precision_matmul,
    mixed_precision_conv2d,
    calculate_mixed_precision_traffic,
    mixed_precision_info,
    # Q/DQ operations (v0.7.10)
    QDQParams,
    Q,
    DQ,
    fake_quantize,
    qdq_linear,
    qdq_matmul,
    qdq_conv2d,
    create_qdq_params,
    quantize_error,
    # Calibration (v0.7.11)
    CalibrationMethod,
    CalibrationStats,
    CalibrationObserver,
    calibrate_minmax,
    calibrate_percentile,
    calibrate_mse,
    calibrate_entropy,
    compare_calibration_methods,
    calibration_info,
    # Memory traffic
    calculate_memory_bytes,
    calculate_matmul_traffic,
    bandwidth_reduction_factor,
)

# Model-Level Execution (v0.8.0+)
from .model import (
    Layer,
    Sequential,
    Model,
    Linear,
    Conv2d,
    BatchNorm2d,
    LayerNorm,
    ReLU,
    GELU,
    SiLU,
    Sigmoid,
    Tanh,
    Softmax,
    MaxPool2d,
    AvgPool2d,
    AdaptiveAvgPool2d,
    Flatten,
    Dropout,
    Identity,
    save_state_dict,
    load_state_dict_from_file,
)

from .model_loader import (
    ModelLoader,
    LayerSpec,
    ModelSpec,
    register_layer,
    load_model,
    save_model,
    LAYER_REGISTRY,
)

from .inference import (
    LayerStats,
    InferenceStats,
    InferencePipeline,
    BatchInference,
    run_inference,
    profile_model,
    benchmark_model,
)

from .memory_planner import (
    MemoryLevel,
    TensorAllocation,
    MemoryPool,
    MemoryPlan,
    MemoryPlanner,
    plan_memory,
    estimate_memory,
)

# Pre-built models (v0.8.0+)
from . import models

# XUE Observation Architecture (v0.5.0+)
# Provides access to C++ XUE event collection and operational analysis
def get_xue_summary():
    """Get XUE event summary from C++ EventCollector.

    Returns:
        Dict with event counts organized by category:
        - total_flops: Total floating point operations
        - total_bytes_moved: Total bytes moved through memory hierarchy
        - dram_bytes: External memory traffic
        - arithmetic_intensity: FLOP/byte ratio
        - compute: Compute event category summary
        - compute_breakdown: Per-operation-type breakdown
        - memory: Memory event category summary
        - memory_hierarchy: Per-level (DRAM/L3/L2/L1) breakdown
        - data_movement: Data movement event summary
        - synchronization: Synchronization/stall events
    """
    try:
        from . import _native
        return _native.get_xue_summary()
    except ImportError:
        return {"error": "Native bindings not available"}


def get_operational_analysis(peak_gflops=1024.0, dram_bandwidth_gbs=64.0, clock_ghz=1.0):
    """Run operational analysis using roofline model.

    XUE Methodology: X (Throughput) → U (Utilization) → E (Efficiency)
    Analyzes collected observation data and predicts performance
    using the roofline model (Williams et al., 2009).

    Args:
        peak_gflops: Peak compute throughput (default: 1024 for 16x16 systolic)
        dram_bandwidth_gbs: DRAM bandwidth in GB/s (default: 64)
        clock_ghz: Clock frequency in GHz (default: 1.0)

    Returns:
        Dict with workload characteristics and performance predictions:
        - total_flops, dram_bytes, arithmetic_intensity
        - predicted_gflops, predicted_cycles, predicted_bottleneck
        - matmul_events, elementwise_events, etc.
        - hardware: Hardware model parameters
    """
    try:
        from . import _native
        return _native.get_operational_analysis(peak_gflops, dram_bandwidth_gbs, clock_ghz)
    except ImportError:
        return {"error": "Native bindings not available"}


def validate_operational_analysis(actual_gflops, actual_cycles,
                                   peak_gflops=1024.0, dram_bandwidth_gbs=64.0, clock_ghz=1.0):
    """Validate operational analysis against actual simulation results.

    Compares roofline model predictions against actual achieved performance.

    Args:
        actual_gflops: Achieved GFLOPS from simulation
        actual_cycles: Actual cycles from simulation
        peak_gflops: Peak compute throughput
        dram_bandwidth_gbs: DRAM bandwidth in GB/s
        clock_ghz: Clock frequency in GHz

    Returns:
        Dict with prediction vs actual comparison:
        - predicted_gflops, actual_gflops
        - gflops_error_percent, cycles_error_percent
        - within_10_percent: Boolean, True if within 10% accuracy
        - roofline_efficiency, bottleneck
    """
    try:
        from . import _native
        return _native.validate_operational_analysis(
            actual_gflops, actual_cycles, peak_gflops, dram_bandwidth_gbs, clock_ghz)
    except ImportError:
        return {"error": "Native bindings not available"}


def reset_xue_counters():
    """Reset all XUE event counters.

    Call this before starting a new workload to get fresh event counts.
    This is called automatically before each execution in _execute_native().
    """
    try:
        from . import _native
        _native.reset_xue_counters()
    except ImportError:
        pass


def set_xue_enabled(enabled: bool):
    """Enable or disable XUE event collection.

    When disabled, events are not recorded (zero overhead).
    This is useful for performance-critical code paths.

    Args:
        enabled: True to enable, False to disable
    """
    try:
        from . import _native
        _native.set_xue_enabled(enabled)
    except ImportError:
        pass


def is_xue_enabled() -> bool:
    """Check if XUE event collection is enabled.

    Returns:
        True if enabled, False otherwise (or if native bindings unavailable)
    """
    try:
        from . import _native
        return _native.is_xue_enabled()
    except ImportError:
        return False


# =============================================================================
# Native Backend Availability and Strict Mode (v0.8.0+)
# =============================================================================

def is_native_available() -> bool:
    """Check if C++ native backend is available.

    The native backend provides:
    - C++ BehavioralComputeFabric (computes actual values)
    - C++ TransactionalComputeFabric (timing simulation)
    - XUE event recording (X/U/E Observation Architecture)

    Returns:
        True if native bindings are available and functional
    """
    try:
        from . import _native
        # Actually try to create a runtime to verify it works
        runtime = _native.create_runtime(0)  # BEHAVIORAL
        return runtime is not None
    except (ImportError, Exception):
        return False


def set_strict_native(enabled: bool):
    """Enable or disable strict native mode.

    When strict_native=True, execution raises RuntimeError if the C++
    native backend is unavailable (no Python fallback).

    This is recommended for production use to ensure:
    - Functional computation uses C++ resource models
    - XUE events are properly recorded
    - No silent fallback to Python/NumPy shims

    Args:
        enabled: True to require C++ backend, False to allow Python fallback

    Example:
        >>> kpu.set_strict_native(True)
        >>> result = mlp(x, w1, w2)  # Raises RuntimeError if C++ unavailable
    """
    from .runtime import KPURuntime
    KPURuntime._strict_native = enabled


def get_strict_native() -> bool:
    """Get current strict native mode setting.

    Returns:
        True if strict_native mode is enabled
    """
    from .runtime import KPURuntime
    return KPURuntime._strict_native


def get_execution_backend() -> str:
    """Get the execution backend used by the last execution.

    Returns:
        One of:
        - "cpp_behavioral" - C++ BehavioralComputeFabric
        - "cpp_transactional" - C++ TransactionalComputeFabric
        - "cpp_cycle_accurate" - C++ cycle-accurate model
        - "python_fallback" - Pure Python/NumPy (no XUE recording)
        - "unknown" - No execution has been performed yet
    """
    from .runtime import KPURuntime
    runtime = KPURuntime.get_instance()
    stats = runtime.get_stats()
    if stats is None:
        return "unknown"
    return stats.execution_backend


def verify_native_execution():
    """Verify that the last execution used the C++ native backend.

    Raises:
        RuntimeError: If the last execution used Python fallback or no execution occurred

    Example:
        >>> result = mlp(x, w1, w2)
        >>> kpu.verify_native_execution()  # Raises if Python fallback was used
    """
    backend = get_execution_backend()
    if backend == "unknown":
        raise RuntimeError("No execution has been performed yet")
    if backend == "python_fallback":
        raise RuntimeError(
            "Last execution used Python fallback instead of C++ native backend. "
            "XUE events were NOT recorded. "
            "Ensure the native module is built: cd python && pip install -e ."
        )
    if not backend.startswith("cpp_"):
        raise RuntimeError(f"Unexpected execution backend: {backend}")


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
    # BF16 operations (v0.7.2)
    "bf16_matmul",
    "bf16_linear",
    "bf16_conv2d",
    "cast_to_bf16",
    "cast_from_bf16",
    "bf16_range",
    "bf16_precision",
    "is_bfloat16_native",
    # FP8 operations (v0.7.3)
    "FP8Format",
    "FP8_E2M5",
    "FP8_E3M4",
    "FP8_E4M3",
    "FP8_E5M2",
    "get_fp8_format",
    "is_fp8_native",
    "fp8_matmul",
    "fp8_linear",
    "cast_to_fp8",
    "cast_from_fp8",
    "fp8_range",
    "fp8_precision",
    "fp8_info",
    # INT4 operations (v0.7.7)
    "pack_int4",
    "unpack_int4",
    "quantize_int4",
    "dequantize_int4",
    "compute_int4_scale_zero_point",
    "int4_matmul",
    "int4_linear",
    "int4_packed_size",
    "int4_memory_bytes",
    "int4_info",
    "INT4_SIGNED_MIN",
    "INT4_SIGNED_MAX",
    "INT4_UNSIGNED_MIN",
    "INT4_UNSIGNED_MAX",
    # FP4 operations (v0.7.8)
    "FP4Format",
    "FP4_E2M1",
    "FP4_E1M2",
    "get_fp4_format",
    "fp4_quantize",
    "fp4_dequantize",
    "pack_fp4",
    "unpack_fp4",
    "fp4_matmul",
    "fp4_linear",
    "fp4_range",
    "fp4_values",
    "fp4_info",
    # Mixed precision (v0.7.9)
    "MixedPrecisionConfig",
    "MIXED_INT8_FP16",
    "MIXED_INT8_BF16",
    "MIXED_INT4_FP16",
    "MIXED_FP8_FP16",
    "MIXED_FP8_BF16",
    "mixed_precision_linear",
    "mixed_precision_matmul",
    "mixed_precision_conv2d",
    "calculate_mixed_precision_traffic",
    "mixed_precision_info",
    # Q/DQ operations (v0.7.10)
    "QDQParams",
    "Q",
    "DQ",
    "fake_quantize",
    "qdq_linear",
    "qdq_matmul",
    "qdq_conv2d",
    "create_qdq_params",
    "quantize_error",
    # Calibration (v0.7.11)
    "CalibrationMethod",
    "CalibrationStats",
    "CalibrationObserver",
    "calibrate_minmax",
    "calibrate_percentile",
    "calibrate_mse",
    "calibrate_entropy",
    "compare_calibration_methods",
    "calibration_info",
    # Memory traffic
    "calculate_memory_bytes",
    "calculate_matmul_traffic",
    "bandwidth_reduction_factor",

    # torch.compile backend
    "torch_backend",
    "torch_compile",
    "get_torch_compile_stats",
    "TORCH_AVAILABLE",

    # Model-Level Execution (v0.8.0+)
    # Base classes
    "Layer",
    "Sequential",
    "Model",
    # Layer classes
    "Linear",
    "Conv2d",
    "BatchNorm2d",
    "LayerNorm",
    # Activation layers
    "ReLU",
    "GELU",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Pooling layers
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    # Utility layers
    "Flatten",
    "Dropout",
    "Identity",
    # State dict utilities
    "save_state_dict",
    "load_state_dict_from_file",
    # Model loading
    "ModelLoader",
    "LayerSpec",
    "ModelSpec",
    "register_layer",
    "load_model",
    "save_model",
    "LAYER_REGISTRY",
    # Inference
    "LayerStats",
    "InferenceStats",
    "InferencePipeline",
    "BatchInference",
    "run_inference",
    "profile_model",
    "benchmark_model",
    # Memory planning
    "MemoryLevel",
    "TensorAllocation",
    "MemoryPool",
    "MemoryPlan",
    "MemoryPlanner",
    "plan_memory",
    "estimate_memory",
    # Pre-built models
    "models",

    # XUE Observation Architecture (v0.5.0+)
    "get_xue_summary",
    "get_operational_analysis",
    "validate_operational_analysis",
    "reset_xue_counters",
    "set_xue_enabled",
    "is_xue_enabled",

    # Native backend verification (v0.8.0+)
    "is_native_available",
    "set_strict_native",
    "get_strict_native",
    "get_execution_backend",
    "verify_native_execution",
]


def version() -> str:
    """Return version string."""
    return __version__


def info() -> str:
    """Return information about the KPU package."""
    from .runtime import get_runtime, KPURuntime

    fidelity_names = {
        BEHAVIORAL: "BEHAVIORAL",
        TRANSACTIONAL: "TRANSACTIONAL",
        CYCLE_ACCURATE: "CYCLE_ACCURATE",
    }

    runtime = get_runtime()
    fidelity_name = fidelity_names.get(runtime.fidelity, "UNKNOWN")

    # Check native backend status
    native_available = is_native_available()
    native_initialized = runtime._native_sim is not None
    strict_mode = KPURuntime._strict_native

    if native_available:
        if native_initialized:
            native_status = "available and initialized (C++ execution)"
        else:
            native_status = "available but not yet initialized"
    else:
        native_status = "NOT AVAILABLE (Python fallback will be used)"

    strict_status = "ENABLED (errors if C++ unavailable)" if strict_mode else "disabled (allows Python fallback)"

    torch_status = "available (torch.compile backend registered)" if TORCH_AVAILABLE else "not available (install PyTorch)"

    # Get last execution backend if available
    last_stats = runtime.get_stats()
    last_backend = last_stats.execution_backend if last_stats else "no execution yet"

    return f"""KPU Python Package v{__version__}
  Fidelity: {fidelity_name}
  Native bindings: {native_status}
  Strict native mode: {strict_status}
  Last execution backend: {last_backend}
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
