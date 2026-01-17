# python/kpu/ops.py
"""
KPU operator definitions.

These operators work with KPU Tensors and support both tracing (for compilation)
and direct execution (for behavioral simulation).
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor import Tensor


def relu(x: 'Tensor') -> 'Tensor':
    """
    Rectified Linear Unit activation: max(0, x)

    Args:
        x: Input tensor

    Returns:
        Output tensor with ReLU applied element-wise
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.RELU, [x], [out])
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute relu on symbolic tensor")
        result = np.maximum(x._data, 0)
        return Tensor(result)


def gelu(x: 'Tensor') -> 'Tensor':
    """
    Gaussian Error Linear Unit activation.

    GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor

    Returns:
        Output tensor with GELU applied element-wise
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.GELU, [x], [out])
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute gelu on symbolic tensor")
        # Approximate GELU
        result = x._data * 0.5 * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x._data + 0.044715 * np.power(x._data, 3))
        ))
        return Tensor(result)


def silu(x: 'Tensor') -> 'Tensor':
    """
    Sigmoid Linear Unit (SiLU/Swish) activation: x * sigmoid(x)

    Args:
        x: Input tensor

    Returns:
        Output tensor with SiLU applied element-wise
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.SILU, [x], [out])
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute silu on symbolic tensor")
        result = x._data * (1 / (1 + np.exp(-x._data)))
        return Tensor(result)


def sigmoid(x: 'Tensor') -> 'Tensor':
    """
    Sigmoid activation: 1 / (1 + exp(-x))

    Args:
        x: Input tensor

    Returns:
        Output tensor with sigmoid applied element-wise
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.SIGMOID, [x], [out])
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute sigmoid on symbolic tensor")
        result = 1 / (1 + np.exp(-x._data))
        return Tensor(result)


def tanh(x: 'Tensor') -> 'Tensor':
    """
    Hyperbolic tangent activation.

    Args:
        x: Input tensor

    Returns:
        Output tensor with tanh applied element-wise
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.TANH, [x], [out])
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute tanh on symbolic tensor")
        result = np.tanh(x._data)
        return Tensor(result)


def softmax(x: 'Tensor', axis: int = -1) -> 'Tensor':
    """
    Softmax activation: exp(x) / sum(exp(x), axis)

    Args:
        x: Input tensor
        axis: Axis along which to compute softmax (default: -1)

    Returns:
        Output tensor with softmax applied along specified axis
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.SOFTMAX, [x], [out], axis=axis)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute softmax on symbolic tensor")
        # Numerically stable softmax
        exp_x = np.exp(x._data - np.max(x._data, axis=axis, keepdims=True))
        result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return Tensor(result)


def exp(x: 'Tensor') -> 'Tensor':
    """
    Element-wise exponential.

    Args:
        x: Input tensor

    Returns:
        Output tensor with exp applied element-wise
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.EXP, [x], [out])
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute exp on symbolic tensor")
        return Tensor(np.exp(x._data))


def log(x: 'Tensor') -> 'Tensor':
    """
    Element-wise natural logarithm.

    Args:
        x: Input tensor

    Returns:
        Output tensor with log applied element-wise
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.LOG, [x], [out])
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute log on symbolic tensor")
        return Tensor(np.log(x._data))


def sqrt(x: 'Tensor') -> 'Tensor':
    """
    Element-wise square root.

    Args:
        x: Input tensor

    Returns:
        Output tensor with sqrt applied element-wise
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.SQRT, [x], [out])
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute sqrt on symbolic tensor")
        return Tensor(np.sqrt(x._data))


# ========== Reduction Operations ==========

def sum(x: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False) -> 'Tensor':
    """
    Sum of tensor elements over given axis.

    Args:
        x: Input tensor
        axis: Axis or axes along which to sum
        keepdims: Whether to keep reduced dimensions

    Returns:
        Reduced tensor
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        # Compute output shape
        if axis is None:
            out_shape = (1,) if keepdims else ()
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            out_shape = list(x.shape)
            for ax in sorted(axes, reverse=True):
                if keepdims:
                    out_shape[ax] = 1
                else:
                    out_shape.pop(ax)
            out_shape = tuple(out_shape) if out_shape else (1,)

        out_meta = TensorMeta(shape=out_shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.SUM, [x], [out], axis=axis, keepdims=keepdims)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute sum on symbolic tensor")
        result = np.sum(x._data, axis=axis, keepdims=keepdims)
        return Tensor(np.atleast_1d(result))


def mean(x: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]] = None,
         keepdims: bool = False) -> 'Tensor':
    """
    Mean of tensor elements over given axis.

    Args:
        x: Input tensor
        axis: Axis or axes along which to compute mean
        keepdims: Whether to keep reduced dimensions

    Returns:
        Reduced tensor
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        # Compute output shape
        if axis is None:
            out_shape = (1,) if keepdims else ()
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            out_shape = list(x.shape)
            for ax in sorted(axes, reverse=True):
                if keepdims:
                    out_shape[ax] = 1
                else:
                    out_shape.pop(ax)
            out_shape = tuple(out_shape) if out_shape else (1,)

        out_meta = TensorMeta(shape=out_shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.MEAN, [x], [out], axis=axis, keepdims=keepdims)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute mean on symbolic tensor")
        result = np.mean(x._data, axis=axis, keepdims=keepdims)
        return Tensor(np.atleast_1d(result))


# ========== Shape Operations ==========

def reshape(x: 'Tensor', shape: Tuple[int, ...]) -> 'Tensor':
    """
    Reshape tensor to new shape.

    Args:
        x: Input tensor
        shape: New shape (one dimension can be -1)

    Returns:
        Reshaped tensor
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    # Resolve -1 dimension
    new_shape = list(shape)
    neg_idx = None
    known_size = 1
    for i, dim in enumerate(new_shape):
        if dim == -1:
            if neg_idx is not None:
                raise ValueError("Only one dimension can be -1")
            neg_idx = i
        else:
            known_size *= dim

    if neg_idx is not None:
        new_shape[neg_idx] = x.size // known_size

    new_shape = tuple(new_shape)

    if Tensor._tracing:
        out_meta = TensorMeta(shape=new_shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.RESHAPE, [x], [out], shape=new_shape)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute reshape on symbolic tensor")
        return Tensor(x._data.reshape(new_shape))


def transpose(x: 'Tensor', axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
    """
    Transpose tensor dimensions.

    Args:
        x: Input tensor
        axes: Permutation of dimensions (default: reverse all)

    Returns:
        Transposed tensor
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if axes is None:
        axes = tuple(range(x.ndim - 1, -1, -1))

    new_shape = tuple(x.shape[ax] for ax in axes)

    if Tensor._tracing:
        out_meta = TensorMeta(shape=new_shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.TRANSPOSE, [x], [out], axes=axes)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute transpose on symbolic tensor")
        return Tensor(np.transpose(x._data, axes))


# ========== Matrix Operations ==========

def matmul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
    """
    Matrix multiplication: C = A @ B

    This is equivalent to the @ operator but provided as a function.

    Args:
        a: Left matrix [M, K]
        b: Right matrix [K, N]

    Returns:
        Result matrix [M, N]
    """
    return a @ b


def linear(x: 'Tensor', weight: 'Tensor', bias: Optional['Tensor'] = None) -> 'Tensor':
    """
    Linear (fully connected) layer: y = x @ W^T + b

    Note: This follows PyTorch convention where weight is [out_features, in_features].

    Args:
        x: Input [batch, in_features]
        weight: Weight matrix [out_features, in_features]
        bias: Optional bias [out_features]

    Returns:
        Output [batch, out_features]
    """
    # Transpose weight to get [in_features, out_features]
    w_t = transpose(weight)
    y = x @ w_t
    if bias is not None:
        y = y + bias
    return y
