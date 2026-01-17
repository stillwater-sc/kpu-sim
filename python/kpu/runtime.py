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


@dataclass
class ExecutionStats:
    """Statistics from kernel execution."""
    cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    matmul_flops: int = 0
    memory_bytes: int = 0
    ops_executed: int = 0


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

    def __init__(self, fidelity: int = BEHAVIORAL):
        self.fidelity = fidelity
        self._native_sim = None
        self._last_stats: Optional[ExecutionStats] = None

    @classmethod
    def get_instance(cls) -> 'KPURuntime':
        """Get the singleton runtime instance."""
        if cls._instance is None:
            cls._instance = KPURuntime()
        return cls._instance

    def set_fidelity(self, fidelity: int):
        """Set the simulation fidelity level."""
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
        Execute program using pure Python (computes actual values).

        This is the functional simulator that verifies correctness.
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

            result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
            for n in range(N):
                for c in range(C):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            h_start = h_out * stride[0]
                            w_start = w_out * stride[1]
                            window = x_padded[n, c, h_start:h_start+K_h, w_start:w_start+K_w]
                            result[n, c, h_out, w_out] = np.max(window)

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

            result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
            for n in range(N):
                for c in range(C):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            h_start = h_out * stride[0]
                            w_start = w_out * stride[1]
                            window = x_padded[n, c, h_start:h_start+K_h, w_start:w_start+K_w]
                            result[n, c, h_out, w_out] = np.mean(window)

        elif op.opcode == DFXOpCode.ADAPTIVE_AVGPOOL2D:
            x = inputs[0]
            output_size = op.attrs.get('output_size', (1, 1))
            N, C, H_in, W_in = x.shape
            H_out, W_out = output_size

            result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
            for n in range(N):
                for c in range(C):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            h_start = (h_out * H_in) // H_out
                            h_end = ((h_out + 1) * H_in) // H_out
                            w_start = (w_out * W_in) // W_out
                            w_end = ((w_out + 1) * W_in) // W_out
                            window = x[n, c, h_start:h_end, w_start:w_end]
                            result[n, c, h_out, w_out] = np.mean(window)

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

        # Convert inputs to numpy arrays
        input_arrays = [t._data for t in inputs]

        # Call native simulator
        result_data, stats_dict = self._native_sim.execute(
            program.to_dict(),
            input_arrays,
            mode
        )

        stats = ExecutionStats(
            cycles=stats_dict.get('cycles', 0),
            compute_cycles=stats_dict.get('compute_cycles', 0),
            memory_cycles=stats_dict.get('memory_cycles', 0),
            matmul_flops=stats_dict.get('matmul_flops', 0),
            memory_bytes=stats_dict.get('memory_bytes', 0),
        )

        return Tensor(result_data), stats

    def get_stats(self) -> Optional[ExecutionStats]:
        """Get statistics from the last execution."""
        return self._last_stats


# Module-level functions for convenience

_runtime: Optional[KPURuntime] = None


def get_runtime() -> KPURuntime:
    """Get the global runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = KPURuntime()
    return _runtime


def set_fidelity(fidelity: int):
    """Set the global simulation fidelity level."""
    get_runtime().set_fidelity(fidelity)


def get_fidelity() -> int:
    """Get the current simulation fidelity level."""
    return get_runtime().fidelity
