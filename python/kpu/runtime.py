# python/kpu/runtime.py
"""
KPU Runtime for executing compiled functions.

Provides behavioral (functional) simulation that computes actual values,
as well as hooks for transactional and cycle-accurate simulation via
the C++ kpu-sim library.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .tensor import Tensor
    from .dfx_emitter import DFXProgram, DFXOp, DFXOpCode

# Fidelity levels
BEHAVIORAL = 0
TRANSACTIONAL = 1
CYCLE_ACCURATE = 2


# =============================================================================
# Optimized NumPy implementations for behavioral simulation
# =============================================================================

def _conv2d_im2col(x_padded: np.ndarray, weight: np.ndarray,
                   stride: Tuple[int, int], H_out: int, W_out: int) -> np.ndarray:
    """Optimized Conv2D using im2col + matmul.

    im2col unfolds input patches into columns, enabling convolution via
    a single matrix multiplication per batch element.
    """
    N, C_in, H_padded, W_padded = x_padded.shape
    C_out, C_in_w, K_h, K_w = weight.shape

    # im2col: Extract patches using stride tricks
    # Shape: (N, C_in, K_h, K_w, H_out, W_out)
    shape = (N, C_in, K_h, K_w, H_out, W_out)
    s = x_padded.strides
    strides = (s[0], s[1], s[2], s[3], s[2] * stride[0], s[3] * stride[1])

    patches = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

    # Reshape to (N, C_in * K_h * K_w, H_out * W_out)
    col = patches.reshape(N, C_in * K_h * K_w, H_out * W_out)
    col = np.ascontiguousarray(col)

    # Reshape weight: (C_out, C_in * K_h * K_w)
    weight_col = weight.reshape(C_out, -1)

    # Convolution as matmul
    result = np.zeros((N, C_out, H_out * W_out), dtype=x_padded.dtype)
    for n in range(N):
        result[n] = weight_col @ col[n]

    return result.reshape(N, C_out, H_out, W_out)


def _maxpool2d_fast(x: np.ndarray, kernel_size: Tuple[int, int],
                    stride: Tuple[int, int], H_out: int, W_out: int) -> np.ndarray:
    """Optimized MaxPool2D using stride tricks."""
    N, C, H_in, W_in = x.shape
    K_h, K_w = kernel_size

    # Create windowed view
    shape = (N, C, H_out, W_out, K_h, K_w)
    s = x.strides
    strides = (s[0], s[1], s[2] * stride[0], s[3] * stride[1], s[2], s[3])

    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.max(windows, axis=(4, 5))


def _avgpool2d_fast(x: np.ndarray, kernel_size: Tuple[int, int],
                    stride: Tuple[int, int], H_out: int, W_out: int) -> np.ndarray:
    """Optimized AvgPool2D using stride tricks."""
    N, C, H_in, W_in = x.shape
    K_h, K_w = kernel_size

    # Create windowed view
    shape = (N, C, H_out, W_out, K_h, K_w)
    s = x.strides
    strides = (s[0], s[1], s[2] * stride[0], s[3] * stride[1], s[2], s[3])

    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.mean(windows, axis=(4, 5))


def _adaptive_avgpool2d_fast(x: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Optimized AdaptiveAvgPool2D."""
    N, C, H_in, W_in = x.shape
    H_out, W_out = output_size

    # Global average pooling (very common)
    if H_out == 1 and W_out == 1:
        return np.mean(x, axis=(2, 3), keepdims=True)

    # When output divides input evenly
    if H_in % H_out == 0 and W_in % W_out == 0:
        K_h = H_in // H_out
        K_w = W_in // W_out
        x_reshaped = x.reshape(N, C, H_out, K_h, W_out, K_w)
        return np.mean(x_reshaped, axis=(3, 5))

    # General case - vectorized over batch and channel
    result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    for h_out in range(H_out):
        for w_out in range(W_out):
            h_start = (h_out * H_in) // H_out
            h_end = ((h_out + 1) * H_in) // H_out
            w_start = (w_out * W_in) // W_out
            w_end = ((w_out + 1) * W_in) // W_out
            window = x[:, :, h_start:h_end, w_start:w_end]
            result[:, :, h_out, w_out] = np.mean(window, axis=(2, 3))
    return result


@dataclass
class ExecutionStats:
    """Statistics from kernel execution.

    Extended for v0.5.0+ with C++ XUE Observation Architecture integration.

    XUE Methodology: X (Throughput) → U (Utilization) → E (Efficiency)
      The Observation Architecture provides event hierarchies that aggregate
      cleanly without logic on the datapath, enabling drill-down analysis
      of resource effectiveness.

      - Use kpu.get_xue_summary() for detailed event breakdown
      - Use kpu.get_operational_analysis() for roofline predictions
      - Per-level memory stats: stats.xue_summary['memory_hierarchy']['dram'], etc.
    """
    # Basic timing
    cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    elapsed_cycles: int = 0  # Wall clock cycles (T)

    # Detailed cycle breakdown (TRANSACTIONAL mode)
    busy_cycles: int = 0
    idle_cycles: int = 0
    stall_cycles: int = 0

    # Compute metrics
    matmul_flops: int = 0
    total_macs: int = 0
    matmul_count: int = 0

    # Memory metrics (use xue_summary['memory_hierarchy'] for per-level breakdown)
    memory_bytes: int = 0
    external_bytes: int = 0

    # Memory controller stats (TRANSACTIONAL mode)
    memory_reads: int = 0
    memory_writes: int = 0
    page_hits: int = 0
    page_misses: int = 0
    memory_latency_cycles: int = 0

    # Operation counts
    ops_executed: int = 0

    # Clock frequency used for performance calculations
    clock_frequency_ghz: float = 0.0

    # Performance metrics (computed using clock_frequency_ghz)
    gflops: float = 0.0
    utilization: float = 0.0
    efficiency: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    page_hit_rate: float = 0.0

    # XUE summary from C++ EventCollector (v0.5.0+)
    # Contains detailed breakdowns including memory_hierarchy with dram/l3/l2/l1
    xue_summary: Optional[Dict[str, Any]] = None

    # Execution backend identifier (v0.8.0+)
    # Proves which backend executed the program:
    #   "cpp_behavioral" - C++ BehavioralComputeFabric
    #   "cpp_transactional" - C++ TransactionalComputeFabric
    #   "cpp_cycle_accurate" - C++ cycle-accurate model
    #   "python_fallback" - Pure Python/NumPy fallback (no XUE)
    execution_backend: str = "unknown"


class KPURuntime:
    """
    Runtime for executing KPU programs.

    Supports three fidelity levels:
    - BEHAVIORAL: Pure Python execution, computes actual values
    - TRANSACTIONAL: Statistical timing model (requires C++ bindings)
    - CYCLE_ACCURATE: Full timing simulation (requires C++ bindings)

    Example:
        >>> runtime = KPURuntime(fidelity=BEHAVIORAL)
        >>> result, stats = runtime.execute(dfx_program, inputs)
    """

    _instance: Optional['KPURuntime'] = None
    _strict_native: bool = False  # Class-level default

    def __init__(self, fidelity: int = BEHAVIORAL, strict_native: bool = False):
        """
        Initialize KPU runtime.

        Args:
            fidelity: Simulation fidelity level (BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE)
            strict_native: If True, raise error when C++ backend unavailable (no Python fallback)
        """
        self.fidelity = fidelity
        self.strict_native = strict_native
        self._native_sim = None
        self._last_stats: Optional[ExecutionStats] = None

    @classmethod
    def get_instance(cls) -> 'KPURuntime':
        """Get the singleton runtime instance."""
        if cls._instance is None:
            cls._instance = KPURuntime()
        return cls._instance

    def set_fidelity(self, fidelity: int):
        """Set the simulation fidelity level.

        Args:
            fidelity: One of BEHAVIORAL, TRANSACTIONAL, or CYCLE_ACCURATE

        Raises:
            ValueError: If fidelity is not a valid level
        """
        if fidelity not in (BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE):
            valid = "BEHAVIORAL (0), TRANSACTIONAL (1), CYCLE_ACCURATE (2)"
            raise ValueError(f"Invalid fidelity level {fidelity}. Must be one of: {valid}")
        self.fidelity = fidelity
        self._native_sim = None  # Force re-initialization

    def execute(self,
                program: 'DFXProgram',
                inputs: List['Tensor']) -> Tuple['Tensor', ExecutionStats]:
        """
        Execute a DFX program on the given inputs.

        Args:
            program: Compiled DFX program
            inputs: Input tensors (must match program.inputs)

        Returns:
            Tuple of (output tensor, execution statistics)
        """
        from .tensor import Tensor

        if len(inputs) != len(program.inputs):
            raise ValueError(
                f"Expected {len(program.inputs)} inputs, got {len(inputs)}"
            )

        if self.fidelity == BEHAVIORAL:
            result, stats = self._execute_behavioral(program, inputs)
        elif self.fidelity == TRANSACTIONAL:
            result, stats = self._execute_transactional(program, inputs)
        elif self.fidelity == CYCLE_ACCURATE:
            result, stats = self._execute_cycle_accurate(program, inputs)
        else:
            raise ValueError(f"Unknown fidelity level: {self.fidelity}")

        self._last_stats = stats
        return result, stats

    def _execute_behavioral(self,
                            program: 'DFXProgram',
                            inputs: List['Tensor']) -> Tuple['Tensor', ExecutionStats]:
        """
        Execute program behaviorally (computes actual values).

        Routes through C++ BehavioralComputeFabric when native bindings are
        available, enabling XUE event recording. Falls back to pure Python
        if native bindings are not available (unless strict_native=True).

        Raises:
            RuntimeError: If strict_native=True and C++ backend unavailable
        """
        from .tensor import Tensor
        import warnings

        # Initialize native simulator if not done yet
        if self._native_sim is None:
            self._init_native_sim()

        # Try to use native C++ BehavioralComputeFabric for XUE recording
        if self._native_sim is not None:
            return self._execute_native(program, inputs, "behavioral")

        # C++ unavailable - check strict mode
        if self.strict_native or KPURuntime._strict_native:
            raise RuntimeError(
                "C++ native backend unavailable but strict_native=True. "
                "BEHAVIORAL execution requires C++ BehavioralComputeFabric. "
                "Ensure the native module is built: cd python && pip install -e ."
            )

        # Fallback to pure Python with warning
        warnings.warn(
            "Falling back to Python behavioral execution (no C++ backend). "
            "XUE events will NOT be recorded. Set strict_native=True to make this an error.",
            UserWarning,
            stacklevel=3
        )
        return self._execute_behavioral_python(program, inputs)

    def _execute_behavioral_python(self,
                                   program: 'DFXProgram',
                                   inputs: List['Tensor']) -> Tuple['Tensor', ExecutionStats]:
        """
        Execute program using pure Python (fallback when native unavailable).

        Note: This path does NOT record XUE events.
        """
        from .tensor import Tensor
        from .dfx_emitter import DFXOpCode

        # Map tensor names to numpy arrays
        tensors: Dict[str, np.ndarray] = {}

        # Load inputs
        for name, tensor in zip(program.inputs, inputs):
            if tensor._data is None:
                raise ValueError(f"Input tensor '{name}' has no data")
            tensors[name] = tensor._data

        stats = ExecutionStats()
        # Mark as Python fallback - no C++ resources were used
        stats.execution_backend = "python_fallback"

        # Execute operations in order
        for op in program.ops:
            self._execute_op_behavioral(op, tensors, stats)
            stats.ops_executed += 1

        # Get output
        output_name = program.outputs[0]
        output_data = tensors[output_name]

        # Handle multiple outputs
        if len(program.outputs) > 1:
            outputs = [Tensor(tensors[name]) for name in program.outputs]
            return outputs[0], stats  # Return first for now

        return Tensor(output_data), stats

    def _execute_op_behavioral(self,
                               op: 'DFXOp',
                               tensors: Dict[str, np.ndarray],
                               stats: ExecutionStats):
        """Execute a single DFX operation behaviorally."""
        from .dfx_emitter import DFXOpCode

        # Get input arrays
        inputs = [tensors[name] for name in op.inputs]
        output_name = op.outputs[0]

        if op.opcode == DFXOpCode.MATMUL:
            A, B = inputs
            result = np.matmul(A, B)
            # Track FLOPs
            M, K = A.shape[-2], A.shape[-1]
            N = B.shape[-1]
            stats.matmul_flops += 2 * M * N * K

        elif op.opcode == DFXOpCode.RELU:
            result = np.maximum(inputs[0], 0)

        elif op.opcode == DFXOpCode.GELU:
            x = inputs[0]
            result = x * 0.5 * (1 + np.tanh(
                np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
            ))

        elif op.opcode == DFXOpCode.SILU:
            x = inputs[0]
            result = x * (1 / (1 + np.exp(-x)))

        elif op.opcode == DFXOpCode.SIGMOID:
            result = 1 / (1 + np.exp(-inputs[0]))

        elif op.opcode == DFXOpCode.TANH:
            result = np.tanh(inputs[0])

        elif op.opcode == DFXOpCode.SOFTMAX:
            x = inputs[0]
            axis = op.attrs.get('axis', -1)
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        elif op.opcode == DFXOpCode.ADD:
            result = inputs[0] + inputs[1]

        elif op.opcode == DFXOpCode.SUB:
            result = inputs[0] - inputs[1]

        elif op.opcode == DFXOpCode.MUL:
            result = inputs[0] * inputs[1]

        elif op.opcode == DFXOpCode.DIV:
            result = inputs[0] / inputs[1]

        elif op.opcode == DFXOpCode.NEG:
            result = -inputs[0]

        elif op.opcode == DFXOpCode.EXP:
            result = np.exp(inputs[0])

        elif op.opcode == DFXOpCode.LOG:
            result = np.log(inputs[0])

        elif op.opcode == DFXOpCode.SQRT:
            result = np.sqrt(inputs[0])

        elif op.opcode == DFXOpCode.SUM:
            axis = op.attrs.get('axis')
            keepdims = op.attrs.get('keepdims', False)
            result = np.sum(inputs[0], axis=axis, keepdims=keepdims)
            if result.ndim == 0:
                result = np.atleast_1d(result)

        elif op.opcode == DFXOpCode.MEAN:
            axis = op.attrs.get('axis')
            keepdims = op.attrs.get('keepdims', False)
            result = np.mean(inputs[0], axis=axis, keepdims=keepdims)
            if result.ndim == 0:
                result = np.atleast_1d(result)

        elif op.opcode == DFXOpCode.MAX:
            axis = op.attrs.get('axis')
            keepdims = op.attrs.get('keepdims', False)
            result = np.max(inputs[0], axis=axis, keepdims=keepdims)
            if result.ndim == 0:
                result = np.atleast_1d(result)

        elif op.opcode == DFXOpCode.MIN:
            axis = op.attrs.get('axis')
            keepdims = op.attrs.get('keepdims', False)
            result = np.min(inputs[0], axis=axis, keepdims=keepdims)
            if result.ndim == 0:
                result = np.atleast_1d(result)

        elif op.opcode == DFXOpCode.BATCH_NORM:
            x = inputs[0]
            eps = op.attrs.get('eps', 1e-5)
            # Compute batch statistics
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            result = (x - mean) / np.sqrt(var + eps)
            # Apply scale and bias if provided
            if len(inputs) > 1:
                result = result * inputs[1].reshape(1, -1, 1, 1)
            if len(inputs) > 2:
                result = result + inputs[2].reshape(1, -1, 1, 1)

        elif op.opcode == DFXOpCode.LAYER_NORM:
            x = inputs[0]
            normalized_shape = op.attrs.get('normalized_shape', (x.shape[-1],))
            eps = op.attrs.get('eps', 1e-5)
            ndim = len(normalized_shape)
            axes = tuple(range(-ndim, 0))
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
            result = (x - mean) / np.sqrt(var + eps)
            # Apply scale and bias if provided
            if len(inputs) > 1:
                result = result * inputs[1]
            if len(inputs) > 2:
                result = result + inputs[2]

        elif op.opcode == DFXOpCode.CONV2D:
            x = inputs[0]
            weight = inputs[1]
            stride = op.attrs.get('stride', (1, 1))
            padding = op.attrs.get('padding', (0, 0))
            dilation = op.attrs.get('dilation', (1, 1))

            N, C_in, H_in, W_in = x.shape
            C_out, C_in_per_group, K_h, K_w = weight.shape

            H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) // stride[0] + 1
            W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) // stride[1] + 1

            # Pad input
            if padding[0] > 0 or padding[1] > 0:
                x_padded = np.pad(x, ((0, 0), (0, 0),
                                      (padding[0], padding[0]),
                                      (padding[1], padding[1])), mode='constant')
            else:
                x_padded = x

            # Optimized im2col implementation for dilation=1
            if dilation == (1, 1):
                result = _conv2d_im2col(x_padded, weight, stride, H_out, W_out)
            else:
                # Fallback for dilated convolution
                result = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)
                for n in range(N):
                    for c_out in range(C_out):
                        for h_out in range(H_out):
                            for w_out in range(W_out):
                                h_start = h_out * stride[0]
                                w_start = w_out * stride[1]
                                val = 0.0
                                for c_in in range(C_in_per_group):
                                    for kh in range(K_h):
                                        for kw in range(K_w):
                                            h_in = h_start + kh * dilation[0]
                                            w_in = w_start + kw * dilation[1]
                                            val += x_padded[n, c_in, h_in, w_in] * weight[c_out, c_in, kh, kw]
                                result[n, c_out, h_out, w_out] = val

            # Add bias if provided
            if len(inputs) > 2:
                result = result + inputs[2].reshape(1, -1, 1, 1)

        elif op.opcode == DFXOpCode.MAXPOOL2D:
            x = inputs[0]
            kernel_size = op.attrs.get('kernel_size', (2, 2))
            stride = op.attrs.get('stride', kernel_size)
            padding = op.attrs.get('padding', (0, 0))

            N, C, H_in, W_in = x.shape
            K_h, K_w = kernel_size
            H_out = (H_in + 2 * padding[0] - K_h) // stride[0] + 1
            W_out = (W_in + 2 * padding[1] - K_w) // stride[1] + 1

            if padding[0] > 0 or padding[1] > 0:
                x_padded = np.pad(x, ((0, 0), (0, 0),
                                      (padding[0], padding[0]),
                                      (padding[1], padding[1])),
                                  mode='constant', constant_values=-np.inf)
            else:
                x_padded = x

            result = _maxpool2d_fast(x_padded, kernel_size, stride, H_out, W_out)

        elif op.opcode == DFXOpCode.AVGPOOL2D:
            x = inputs[0]
            kernel_size = op.attrs.get('kernel_size', (2, 2))
            stride = op.attrs.get('stride', kernel_size)
            padding = op.attrs.get('padding', (0, 0))

            N, C, H_in, W_in = x.shape
            K_h, K_w = kernel_size
            H_out = (H_in + 2 * padding[0] - K_h) // stride[0] + 1
            W_out = (W_in + 2 * padding[1] - K_w) // stride[1] + 1

            if padding[0] > 0 or padding[1] > 0:
                x_padded = np.pad(x, ((0, 0), (0, 0),
                                      (padding[0], padding[0]),
                                      (padding[1], padding[1])), mode='constant')
            else:
                x_padded = x

            result = _avgpool2d_fast(x_padded, kernel_size, stride, H_out, W_out)

        elif op.opcode == DFXOpCode.ADAPTIVE_AVGPOOL2D:
            x = inputs[0]
            output_size = op.attrs.get('output_size', (1, 1))
            result = _adaptive_avgpool2d_fast(x, output_size)

        elif op.opcode == DFXOpCode.CONCAT:
            dim = op.attrs.get('dim', 0)
            result = np.concatenate(inputs, axis=dim)

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

            # Check if shape matches input size
            target_size = 1
            for s in shape:
                target_size *= s

            # If size mismatch, try to fix first dimension (batch)
            # This handles dynamic batch sizes traced with a fixed batch
            if target_size != total_size and len(shape) > 1:
                other_size = 1
                for s in shape[1:]:
                    other_size *= s
                if other_size > 0 and total_size % other_size == 0:
                    shape[0] = total_size // other_size

            result = x.reshape(tuple(shape))

        elif op.opcode == DFXOpCode.TRANSPOSE:
            axes = op.attrs.get('axes')
            result = np.transpose(inputs[0], axes)

        elif op.opcode == DFXOpCode.FLATTEN:
            start_dim = op.attrs.get('start_dim', 0)
            end_dim = op.attrs.get('end_dim', -1)
            x = inputs[0]
            ndim = x.ndim
            if start_dim < 0:
                start_dim = ndim + start_dim
            if end_dim < 0:
                end_dim = ndim + end_dim
            new_shape = list(x.shape[:start_dim])
            flat_size = 1
            for i in range(start_dim, end_dim + 1):
                flat_size *= x.shape[i]
            new_shape.append(flat_size)
            new_shape.extend(x.shape[end_dim + 1:])
            result = x.reshape(new_shape)

        elif op.opcode == DFXOpCode.FUSED_MATMUL_BIAS_RELU:
            # Fused: Y = relu(matmul(A, B) + bias)
            A, B, bias = inputs
            result = np.maximum(np.matmul(A, B) + bias, 0)
            # Track FLOPs
            M, K = A.shape[-2], A.shape[-1]
            N = B.shape[-1]
            stats.matmul_flops += 2 * M * N * K

        elif op.opcode == DFXOpCode.FUSED_MATMUL_BIAS_GELU:
            # Fused: Y = gelu(matmul(A, B) + bias)
            A, B, bias = inputs
            x = np.matmul(A, B) + bias
            result = x * 0.5 * (1 + np.tanh(
                np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
            ))
            M, K = A.shape[-2], A.shape[-1]
            N = B.shape[-1]
            stats.matmul_flops += 2 * M * N * K

        elif op.opcode == DFXOpCode.FUSED_MATMUL_BIAS_SILU:
            # Fused: Y = silu(matmul(A, B) + bias)
            A, B, bias = inputs
            x = np.matmul(A, B) + bias
            result = x * (1 / (1 + np.exp(-x)))
            M, K = A.shape[-2], A.shape[-1]
            N = B.shape[-1]
            stats.matmul_flops += 2 * M * N * K

        elif op.opcode == DFXOpCode.FUSED_MATMUL_RELU:
            # Fused: Y = relu(matmul(A, B))
            A, B = inputs
            result = np.maximum(np.matmul(A, B), 0)
            M, K = A.shape[-2], A.shape[-1]
            N = B.shape[-1]
            stats.matmul_flops += 2 * M * N * K

        elif op.opcode == DFXOpCode.FUSED_CONV2D_BN_RELU:
            # Fused: Y = relu(batch_norm(conv2d(X, W)))
            # inputs: [X, W, gamma, beta, running_mean, running_var] (BN params optional)
            x = inputs[0]
            weight = inputs[1]
            stride = op.attrs.get('stride', (1, 1))
            padding = op.attrs.get('padding', (0, 0))
            dilation = op.attrs.get('dilation', (1, 1))
            eps = op.attrs.get('eps', 1e-5)

            N_batch, C_in, H_in, W_in = x.shape
            C_out, C_in_per_group, K_h, K_w = weight.shape

            H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) // stride[0] + 1
            W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) // stride[1] + 1

            # Pad input
            if padding[0] > 0 or padding[1] > 0:
                x_padded = np.pad(x, ((0, 0), (0, 0),
                                      (padding[0], padding[0]),
                                      (padding[1], padding[1])), mode='constant')
            else:
                x_padded = x

            # Optimized Conv2D using im2col
            if dilation == (1, 1):
                conv_result = _conv2d_im2col(x_padded, weight, stride, H_out, W_out)
            else:
                # Fallback for dilated convolution
                conv_result = np.zeros((N_batch, C_out, H_out, W_out), dtype=x.dtype)
                for n in range(N_batch):
                    for c_out in range(C_out):
                        for h_out in range(H_out):
                            for w_out in range(W_out):
                                h_start = h_out * stride[0]
                                w_start = w_out * stride[1]
                                val = 0.0
                                for c_in in range(C_in_per_group):
                                    for kh in range(K_h):
                                        for kw in range(K_w):
                                            h_in = h_start + kh * dilation[0]
                                            w_in = w_start + kw * dilation[1]
                                            val += x_padded[n, c_in, h_in, w_in] * weight[c_out, c_in, kh, kw]
                                conv_result[n, c_out, h_out, w_out] = val

            # BatchNorm (compute batch statistics)
            mean = np.mean(conv_result, axis=(0, 2, 3), keepdims=True)
            var = np.var(conv_result, axis=(0, 2, 3), keepdims=True)
            bn_result = (conv_result - mean) / np.sqrt(var + eps)

            # Apply gamma and beta if provided
            if len(inputs) > 2:
                bn_result = bn_result * inputs[2].reshape(1, -1, 1, 1)
            if len(inputs) > 3:
                bn_result = bn_result + inputs[3].reshape(1, -1, 1, 1)

            # ReLU
            result = np.maximum(bn_result, 0)

        elif op.opcode == DFXOpCode.FUSED_CONV2D_RELU:
            # Fused: Y = relu(conv2d(X, W))
            x = inputs[0]
            weight = inputs[1]
            stride = op.attrs.get('stride', (1, 1))
            padding = op.attrs.get('padding', (0, 0))
            dilation = op.attrs.get('dilation', (1, 1))

            N_batch, C_in, H_in, W_in = x.shape
            C_out, C_in_per_group, K_h, K_w = weight.shape

            H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) // stride[0] + 1
            W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) // stride[1] + 1

            # Pad input
            if padding[0] > 0 or padding[1] > 0:
                x_padded = np.pad(x, ((0, 0), (0, 0),
                                      (padding[0], padding[0]),
                                      (padding[1], padding[1])), mode='constant')
            else:
                x_padded = x

            # Optimized Conv2D using im2col
            if dilation == (1, 1):
                conv_result = _conv2d_im2col(x_padded, weight, stride, H_out, W_out)
            else:
                # Fallback for dilated convolution
                conv_result = np.zeros((N_batch, C_out, H_out, W_out), dtype=x.dtype)
                for n in range(N_batch):
                    for c_out in range(C_out):
                        for h_out in range(H_out):
                            for w_out in range(W_out):
                                h_start = h_out * stride[0]
                                w_start = w_out * stride[1]
                                val = 0.0
                                for c_in in range(C_in_per_group):
                                    for kh in range(K_h):
                                        for kw in range(K_w):
                                            h_in = h_start + kh * dilation[0]
                                            w_in = w_start + kw * dilation[1]
                                            val += x_padded[n, c_in, h_in, w_in] * weight[c_out, c_in, kh, kw]
                                conv_result[n, c_out, h_out, w_out] = val

            # Add bias if provided, then ReLU
            if len(inputs) > 2:
                result = np.maximum(conv_result + inputs[2].reshape(1, -1, 1, 1), 0)
            else:
                result = np.maximum(conv_result, 0)

        elif op.opcode == DFXOpCode.ATTENTION:
            # Multi-head attention with QKV projections
            # inputs[0] = x [B, S, D]
            # inputs[1] = w_q [D, D]
            # inputs[2] = w_k [D, D]
            # inputs[3] = w_v [D, D]
            # inputs[4] = w_o [D, D] (optional)
            x = inputs[0]
            w_q = inputs[1]
            w_k = inputs[2]
            w_v = inputs[3]
            w_o = inputs[4] if len(inputs) > 4 else None

            num_heads = op.attrs.get('num_heads', 1)
            is_causal = op.attrs.get('is_causal', False)

            batch_size, seq_len, d_model = x.shape
            head_dim = d_model // num_heads

            # Project Q, K, V
            q = np.matmul(x, w_q)  # [B, S, D]
            k = np.matmul(x, w_k)  # [B, S, D]
            v = np.matmul(x, w_v)  # [B, S, D]

            # Reshape to [B, S, H, d_k] then transpose to [B, H, S, d_k]
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

            # Scaled dot-product attention
            scale = 1.0 / np.sqrt(head_dim)
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [B, H, S, S]

            # Apply causal mask if requested
            if is_causal:
                causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
                scores = scores + causal_mask

            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Attention @ V -> [B, H, S, d_k]
            attn_output = np.matmul(attn_weights, v)

            # Reshape back to [B, S, D]
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

            # Output projection if provided
            if w_o is not None:
                result = np.matmul(attn_output, w_o)
            else:
                result = attn_output

        else:
            raise NotImplementedError(f"Op {op.opcode} not implemented in behavioral runtime")

        tensors[output_name] = result

    def _execute_transactional(self,
                               program: 'DFXProgram',
                               inputs: List['Tensor']) -> Tuple['Tensor', ExecutionStats]:
        """
        Execute program using transactional model (statistical timing).

        Requires C++ bindings to kpu-sim.
        """
        # Try to use native bindings
        if self._native_sim is None:
            self._init_native_sim()

        if self._native_sim is not None:
            return self._execute_native(program, inputs, "transactional")
        else:
            # Fall back to behavioral with timing estimates
            print("Warning: Native bindings not available, using behavioral simulation")
            return self._execute_behavioral(program, inputs)

    def _execute_cycle_accurate(self,
                                program: 'DFXProgram',
                                inputs: List['Tensor']) -> Tuple['Tensor', ExecutionStats]:
        """
        Execute program using cycle-accurate model.

        Requires C++ bindings to kpu-sim.
        """
        # Try to use native bindings
        if self._native_sim is None:
            self._init_native_sim()

        if self._native_sim is not None:
            return self._execute_native(program, inputs, "cycle_accurate")
        else:
            # Fall back to behavioral
            print("Warning: Native bindings not available, using behavioral simulation")
            return self._execute_behavioral(program, inputs)

    def _init_native_sim(self):
        """Initialize native C++ simulator."""
        try:
            # Try to import native bindings
            from ._native import _native
            self._native_sim = _native.create_runtime(self.fidelity)
        except ImportError:
            # Native bindings not available - this is normal in pure-Python mode
            self._native_sim = None
        except Exception as e:
            # Log any other errors but continue without native support
            import warnings
            warnings.warn(f"Native bindings failed to initialize: {e}")
            self._native_sim = None

    def _execute_native(self,
                        program: 'DFXProgram',
                        inputs: List['Tensor'],
                        mode: str) -> Tuple['Tensor', ExecutionStats]:
        """Execute using native C++ simulator."""
        from .tensor import Tensor
        from . import _native

        # Reset XUE counters before execution (v0.5.0+)
        _native.reset_xue_counters()

        # Convert inputs to numpy arrays
        input_arrays = [t._data for t in inputs]

        # Call native simulator (records XUE events during execution)
        result_data, stats_dict = self._native_sim.execute(
            program.to_dict(),
            input_arrays,
            mode
        )

        # Get XUE summary from C++ EventCollector (v0.5.0+)
        # Per-level memory stats available via xue_summary['memory_hierarchy']
        xue_summary = _native.get_xue_summary()

        stats = ExecutionStats(
            # Basic timing
            cycles=stats_dict.get('cycles', 0),
            compute_cycles=stats_dict.get('compute_cycles', 0),
            memory_cycles=stats_dict.get('memory_cycles', 0),
            elapsed_cycles=stats_dict.get('elapsed_cycles', 0),
            # Detailed cycle breakdown
            busy_cycles=stats_dict.get('busy_cycles', 0),
            idle_cycles=stats_dict.get('idle_cycles', 0),
            stall_cycles=stats_dict.get('stall_cycles', 0),
            # Compute metrics
            matmul_flops=stats_dict.get('matmul_flops', 0),
            total_macs=stats_dict.get('total_macs', 0),
            matmul_count=stats_dict.get('matmul_count', 0),
            # Memory metrics
            memory_bytes=stats_dict.get('memory_bytes', 0),
            external_bytes=stats_dict.get('external_bytes', 0),
            # Memory controller stats
            memory_reads=stats_dict.get('memory_reads', 0),
            memory_writes=stats_dict.get('memory_writes', 0),
            page_hits=stats_dict.get('page_hits', 0),
            page_misses=stats_dict.get('page_misses', 0),
            memory_latency_cycles=stats_dict.get('memory_latency_cycles', 0),
            # Operation counts
            ops_executed=stats_dict.get('ops_executed', 0),
            # Clock frequency
            clock_frequency_ghz=stats_dict.get('clock_frequency_ghz', 0.0),
            # Performance metrics
            gflops=stats_dict.get('gflops', 0.0),
            utilization=stats_dict.get('utilization', 0.0),
            efficiency=stats_dict.get('efficiency', 0.0),
            memory_bandwidth_gbps=stats_dict.get('memory_bandwidth_gbps', 0.0),
            page_hit_rate=stats_dict.get('page_hit_rate', 0.0),
            # XUE summary from C++ (contains memory_hierarchy with dram/l3/l2/l1)
            xue_summary=xue_summary,
            # Execution backend identifier (v0.8.0+)
            execution_backend=f"cpp_{mode}",
        )

        return Tensor(result_data), stats

    def get_stats(self) -> Optional[ExecutionStats]:
        """Get statistics from the last execution."""
        return self._last_stats

    def set_clock_frequency(self, ghz: float):
        """Set the clock frequency in GHz for performance calculations.

        IMPORTANT: Must be called before execution in TRANSACTIONAL or
        CYCLE_ACCURATE mode. This is required to ensure accurate GFLOPS
        and bandwidth calculations.

        Args:
            ghz: Clock frequency in GHz (must be positive)

        Raises:
            ValueError: If ghz <= 0
            RuntimeError: If native simulator not initialized
        """
        if ghz <= 0:
            raise ValueError("Clock frequency must be positive")

        # Initialize native sim if needed
        if self._native_sim is None:
            self._init_native_sim()

        if self._native_sim is not None:
            self._native_sim.set_clock_frequency(ghz)
        else:
            # For behavioral mode, we don't need native sim
            # Store locally for potential future use
            self._clock_frequency_ghz = ghz

    def get_clock_frequency(self) -> float:
        """Get the configured clock frequency in GHz.

        Returns:
            Clock frequency in GHz, or 0.0 if not set
        """
        if self._native_sim is not None:
            return self._native_sim.get_clock_frequency()
        return getattr(self, '_clock_frequency_ghz', 0.0)

    def is_clock_frequency_set(self) -> bool:
        """Check if clock frequency has been explicitly set.

        Returns:
            True if set, False otherwise
        """
        if self._native_sim is not None:
            return self._native_sim.is_clock_frequency_set()
        return hasattr(self, '_clock_frequency_ghz')


# Module-level functions for convenience

_runtime: Optional[KPURuntime] = None


def get_runtime() -> KPURuntime:
    """Get the global runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = KPURuntime()
    return _runtime


def set_fidelity(fidelity: int):
    """Set the global simulation fidelity level.

    Fidelity levels control the trade-off between simulation speed and accuracy:

    - BEHAVIORAL (0): Functional correctness only, computes actual values.
      Fastest mode, suitable for algorithm development and verification.

    - TRANSACTIONAL (1): Statistical timing model with throughput-based
      performance estimation. Requires clock_frequency to be set.
      Use for architecture exploration and performance estimation.

    - CYCLE_ACCURATE (2): Full timing simulation with cycle-by-cycle
      tracking. Most accurate but slowest. Requires clock_frequency.
      Use for detailed performance analysis.

    Args:
        fidelity: One of BEHAVIORAL, TRANSACTIONAL, or CYCLE_ACCURATE

    Raises:
        ValueError: If fidelity is not a valid level

    Example:
        >>> import kpu
        >>> kpu.set_fidelity(kpu.TRANSACTIONAL)
        >>> kpu.set_clock_frequency(1.0)  # Required for TRANSACTIONAL
        >>> result = my_function(x, w)
    """
    if fidelity not in (BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE):
        valid = "BEHAVIORAL (0), TRANSACTIONAL (1), CYCLE_ACCURATE (2)"
        raise ValueError(f"Invalid fidelity level {fidelity}. Must be one of: {valid}")
    get_runtime().set_fidelity(fidelity)


def get_fidelity() -> int:
    """Get the current simulation fidelity level.

    Returns:
        Current fidelity level (BEHAVIORAL, TRANSACTIONAL, or CYCLE_ACCURATE)
    """
    return get_runtime().fidelity


def get_fidelity_name() -> str:
    """Get the name of the current simulation fidelity level.

    Returns:
        String name: "BEHAVIORAL", "TRANSACTIONAL", or "CYCLE_ACCURATE"
    """
    names = {
        BEHAVIORAL: "BEHAVIORAL",
        TRANSACTIONAL: "TRANSACTIONAL",
        CYCLE_ACCURATE: "CYCLE_ACCURATE",
    }
    return names.get(get_runtime().fidelity, "UNKNOWN")


def set_clock_frequency(ghz: float):
    """Set the clock frequency for performance calculations.

    IMPORTANT: Must be called before execution in TRANSACTIONAL or
    CYCLE_ACCURATE mode. Without this, execution will fail to prevent
    silent assumptions about clock speed.

    Args:
        ghz: Clock frequency in GHz (must be positive)

    Example:
        >>> kpu.set_clock_frequency(1.0)  # 1 GHz
        >>> kpu.set_fidelity(kpu.TRANSACTIONAL)
        >>> result = my_function(x, w)  # Now works
    """
    get_runtime().set_clock_frequency(ghz)


def get_clock_frequency() -> float:
    """Get the configured clock frequency in GHz.

    Returns:
        Clock frequency in GHz, or 0.0 if not set
    """
    return get_runtime().get_clock_frequency()


def is_clock_frequency_set() -> bool:
    """Check if clock frequency has been explicitly set.

    Returns:
        True if set, False otherwise
    """
    return get_runtime().is_clock_frequency_set()
