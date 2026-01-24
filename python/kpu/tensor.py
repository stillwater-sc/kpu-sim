# python/kpu/tensor.py
"""
KPU Tensor class with tracing support for compilation.

During tracing (@kpu.compile), operations on Tensors are recorded
rather than executed. During execution, operations run on C++ simulator.

v0.8.1: All Tensor operations now route through C++ BehavioralComputeFabric
        when native module is available. This ensures XUE event recording
        for all operations, not just DFXProgram execution.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .graph import OpGraph

# Try to import native module for C++ operations
_native_available = False
_native = None

def _init_native():
    """Initialize native module on first use."""
    global _native_available, _native
    if _native is not None:
        return _native_available
    try:
        from . import _native as native_module
        if hasattr(native_module, 'native_matmul'):
            _native = native_module
            _native_available = True
        else:
            _native_available = False
    except ImportError:
        _native_available = False
    return _native_available


@dataclass
class TensorMeta:
    """Metadata for shape/type tracking during compilation."""
    shape: Tuple[int, ...]
    dtype: np.dtype
    name: Optional[str] = None
    is_weight: bool = False  # Weights can be pre-loaded to L3


class Tensor:
    """
    KPU Tensor - wraps NumPy array with metadata for compilation.

    During tracing (@kpu.compile), operations on Tensors are recorded
    rather than executed. During execution, operations run on simulator.

    Example:
        >>> import kpu
        >>> x = kpu.Tensor(np.random.randn(32, 784).astype(np.float32))
        >>> w = kpu.Tensor(np.random.randn(784, 128).astype(np.float32))
        >>> y = x @ w  # Records operation during tracing, executes otherwise
    """

    # Class-level tracing state
    _tracing: bool = False
    _trace_graph: Optional['OpGraph'] = None

    def __init__(self,
                 data: Union[np.ndarray, TensorMeta, Tuple[int, ...]],
                 name: Optional[str] = None,
                 dtype: Optional[np.dtype] = None):
        """
        Create a KPU Tensor.

        Args:
            data: NumPy array, TensorMeta for symbolic tensor, or shape tuple
            name: Optional name for debugging/tracing
            dtype: Data type (inferred from data if not specified)
        """
        if isinstance(data, np.ndarray):
            self._data = data
            self._meta = TensorMeta(
                shape=data.shape,
                dtype=data.dtype,
                name=name
            )
        elif isinstance(data, TensorMeta):
            # Symbolic tensor for tracing
            self._data = None
            self._meta = data
            if name is not None:
                self._meta.name = name
        elif isinstance(data, (tuple, list)):
            # Create from shape
            dt = dtype if dtype is not None else np.float32
            self._data = None
            self._meta = TensorMeta(
                shape=tuple(data),
                dtype=np.dtype(dt),
                name=name
            )
        else:
            raise TypeError(f"Expected ndarray, TensorMeta, or shape tuple, got {type(data)}")

    @classmethod
    def from_numpy(cls, arr: np.ndarray, name: Optional[str] = None) -> 'Tensor':
        """Create Tensor from NumPy array."""
        return cls(arr, name=name)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...], dtype=np.float32, name: Optional[str] = None) -> 'Tensor':
        """Create zero-filled Tensor."""
        return cls(np.zeros(shape, dtype=dtype), name=name)

    @classmethod
    def ones(cls, shape: Tuple[int, ...], dtype=np.float32, name: Optional[str] = None) -> 'Tensor':
        """Create one-filled Tensor."""
        return cls(np.ones(shape, dtype=dtype), name=name)

    @classmethod
    def randn(cls, *shape, dtype=np.float32, name: Optional[str] = None) -> 'Tensor':
        """Create random normal Tensor."""
        return cls(np.random.randn(*shape).astype(dtype), name=name)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tensor shape."""
        return self._meta.shape

    @property
    def dtype(self) -> np.dtype:
        """Tensor data type."""
        return self._meta.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._meta.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self._meta.shape:
            result *= dim
        return result

    def numel(self) -> int:
        """Total number of elements (PyTorch-compatible alias for size)."""
        return self.size

    @property
    def nbytes(self) -> int:
        """Total bytes."""
        return self.size * self._meta.dtype.itemsize

    @property
    def name(self) -> Optional[str]:
        """Tensor name."""
        return self._meta.name

    @property
    def data(self) -> Optional[np.ndarray]:
        """Underlying NumPy data (None for symbolic tensors)."""
        return self._data

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        if self._data is None:
            raise ValueError("Cannot convert symbolic tensor to numpy")
        return self._data

    def is_symbolic(self) -> bool:
        """Return True if this is a symbolic tensor (no data)."""
        return self._data is None

    # ========== Shape Operations ==========

    def reshape(self, *shape) -> 'Tensor':
        """
        Reshape tensor to new shape.

        Args:
            shape: New shape (can be a tuple or individual dimensions)

        Returns:
            Reshaped tensor
        """
        from .ops import reshape as ops_reshape

        # Handle both reshape((2, 3)) and reshape(2, 3)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        return ops_reshape(self, shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'Tensor':
        """
        Flatten tensor dimensions.

        Args:
            start_dim: First dimension to flatten
            end_dim: Last dimension to flatten

        Returns:
            Flattened tensor
        """
        from .ops import flatten as ops_flatten
        return ops_flatten(self, start_dim, end_dim)

    def view(self, *shape) -> 'Tensor':
        """
        View tensor with new shape (alias for reshape).

        Args:
            shape: New shape

        Returns:
            Reshaped tensor
        """
        return self.reshape(*shape)

    # ========== Arithmetic Operations ==========

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication: C = A @ B"""
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other))

        if Tensor._tracing:
            return self._trace_matmul(other)
        else:
            return self._execute_matmul(other)

    def __add__(self, other: Union['Tensor', np.ndarray, float]) -> 'Tensor':
        """Element-wise addition."""
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                other = Tensor(np.full(self.shape, other, dtype=self.dtype))
            else:
                other = Tensor(np.asarray(other))

        if Tensor._tracing:
            return self._trace_binary_op('add', other)
        else:
            return self._execute_binary_op('add', other)

    def __radd__(self, other) -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other: Union['Tensor', np.ndarray, float]) -> 'Tensor':
        """Element-wise subtraction."""
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                other = Tensor(np.full(self.shape, other, dtype=self.dtype))
            else:
                other = Tensor(np.asarray(other))

        if Tensor._tracing:
            return self._trace_binary_op('sub', other)
        else:
            return self._execute_binary_op('sub', other)

    def __rsub__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                other = Tensor(np.full(self.shape, other, dtype=self.dtype))
            else:
                other = Tensor(np.asarray(other))
        return other.__sub__(self)

    def __mul__(self, other: Union['Tensor', np.ndarray, float]) -> 'Tensor':
        """Element-wise multiplication."""
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                other = Tensor(np.full(self.shape, other, dtype=self.dtype))
            else:
                other = Tensor(np.asarray(other))

        if Tensor._tracing:
            return self._trace_binary_op('mul', other)
        else:
            return self._execute_binary_op('mul', other)

    def __rmul__(self, other) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other: Union['Tensor', np.ndarray, float]) -> 'Tensor':
        """Element-wise division."""
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                other = Tensor(np.full(self.shape, other, dtype=self.dtype))
            else:
                other = Tensor(np.asarray(other))

        if Tensor._tracing:
            return self._trace_binary_op('div', other)
        else:
            return self._execute_binary_op('div', other)

    def __neg__(self) -> 'Tensor':
        """Unary negation."""
        if Tensor._tracing:
            return self._trace_unary_op('neg')
        else:
            return Tensor(-self._data, name=f"neg({self.name})")

    # ========== Tracing Operations ==========

    def _trace_matmul(self, other: 'Tensor') -> 'Tensor':
        """Record matmul operation during tracing."""
        from .graph import OpType

        # Validate shapes
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError(f"matmul requires 2D+ tensors, got {self.shape} @ {other.shape}")

        M, K1 = self.shape[-2], self.shape[-1]
        K2, N = other.shape[-2], other.shape[-1]

        if K1 != K2:
            raise ValueError(f"Shape mismatch for matmul: {self.shape} @ {other.shape}")

        # Compute output shape (handle batched matmul)
        batch_dims = self.shape[:-2]
        out_shape = (*batch_dims, M, N)

        out_meta = TensorMeta(shape=out_shape, dtype=self.dtype)
        out = Tensor(out_meta)

        # Record operation
        Tensor._trace_graph.add_op(OpType.MATMUL, [self, other], [out])
        return out

    def _trace_binary_op(self, op_name: str, other: 'Tensor') -> 'Tensor':
        """Record binary operation during tracing."""
        from .graph import OpType

        # Broadcast shapes
        out_shape = np.broadcast_shapes(self.shape, other.shape)
        out_meta = TensorMeta(shape=out_shape, dtype=self.dtype)
        out = Tensor(out_meta)

        op_type = OpType(op_name)
        Tensor._trace_graph.add_op(op_type, [self, other], [out])
        return out

    def _trace_unary_op(self, op_name: str) -> 'Tensor':
        """Record unary operation during tracing."""
        from .graph import OpType

        out_meta = TensorMeta(shape=self.shape, dtype=self.dtype)
        out = Tensor(out_meta)

        op_type = OpType(op_name)
        Tensor._trace_graph.add_op(op_type, [self], [out])
        return out

    # ========== Execution Operations ==========
    # v0.8.1: These now route through C++ BehavioralComputeFabric when available

    def _execute_matmul(self, other: 'Tensor') -> 'Tensor':
        """Execute matmul via C++ BehavioralComputeFabric (with NumPy fallback)."""
        if self._data is None or other._data is None:
            raise ValueError("Cannot execute matmul on symbolic tensors")

        # Try native C++ execution first
        if _init_native():
            # Ensure contiguous float32 arrays for C++
            a = np.ascontiguousarray(self._data, dtype=np.float32)
            b = np.ascontiguousarray(other._data, dtype=np.float32)
            result = _native.native_matmul(a, b)
            return Tensor(result)

        # Fallback to NumPy
        result = np.matmul(self._data, other._data)
        return Tensor(result)

    def _execute_binary_op(self, op_name: str, other: 'Tensor') -> 'Tensor':
        """Execute binary op via C++ BehavioralComputeFabric (with NumPy fallback)."""
        if self._data is None or other._data is None:
            raise ValueError(f"Cannot execute {op_name} on symbolic tensors")

        # Try native C++ execution first
        if _init_native():
            native_ops = {
                'add': _native.native_add,
                'sub': _native.native_sub,
                'mul': _native.native_mul,
                'div': _native.native_div,
            }
            if op_name in native_ops:
                # Ensure contiguous float32 arrays for C++
                a = np.ascontiguousarray(self._data, dtype=np.float32)
                b = np.ascontiguousarray(other._data, dtype=np.float32)
                # Handle broadcasting by expanding to same shape
                if a.shape != b.shape:
                    a, b = np.broadcast_arrays(a, b)
                    a = np.ascontiguousarray(a)
                    b = np.ascontiguousarray(b)
                result = native_ops[op_name](a, b)
                return Tensor(result)

        # Fallback to NumPy
        numpy_ops = {
            'add': np.add,
            'sub': np.subtract,
            'mul': np.multiply,
            'div': np.divide,
        }
        result = numpy_ops[op_name](self._data, other._data)
        return Tensor(result)

    # ========== Utility ==========

    def __repr__(self) -> str:
        name_str = f"'{self._meta.name}'" if self._meta.name else "unnamed"
        if self._data is not None:
            return f"Tensor({name_str}, shape={self.shape}, dtype={self.dtype})"
        else:
            return f"Tensor({name_str}, shape={self.shape}, dtype={self.dtype}, symbolic)"

    def __str__(self) -> str:
        return self.__repr__()
