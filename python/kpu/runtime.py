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
