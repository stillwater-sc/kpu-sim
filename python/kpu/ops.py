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


# ========== Normalization Operations ==========

def layer_norm(x: 'Tensor',
               normalized_shape: Union[int, Tuple[int, ...]],
               weight: Optional['Tensor'] = None,
               bias: Optional['Tensor'] = None,
               eps: float = 1e-5) -> 'Tensor':
    """
    Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        x: Input tensor
        normalized_shape: Shape over which to normalize (last N dimensions)
        weight: Optional scale parameter (gamma)
        bias: Optional shift parameter (beta)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        inputs = [x]
        if weight is not None:
            inputs.append(weight)
        if bias is not None:
            inputs.append(bias)
        Tensor._trace_graph.add_op(OpType.LAYER_NORM, inputs, [out],
                                   normalized_shape=normalized_shape, eps=eps)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute layer_norm on symbolic tensor")

        # Compute axes for normalization (last N dimensions)
        ndim = len(normalized_shape)
        axes = tuple(range(-ndim, 0))

        # Compute mean and variance
        mean = np.mean(x._data, axis=axes, keepdims=True)
        var = np.var(x._data, axis=axes, keepdims=True)

        # Normalize
        result = (x._data - mean) / np.sqrt(var + eps)

        # Apply scale and shift
        if weight is not None:
            result = result * weight._data
        if bias is not None:
            result = result + bias._data

        return Tensor(result.astype(x.dtype))


# ========== Convolution Operations ==========

def conv2d(x: 'Tensor',
           weight: 'Tensor',
           bias: Optional['Tensor'] = None,
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[int, Tuple[int, int]] = 0,
           dilation: Union[int, Tuple[int, int]] = 1,
           groups: int = 1) -> 'Tensor':
    """
    2D Convolution: applies a 2D convolution over an input image.

    Args:
        x: Input tensor [N, C_in, H, W]
        weight: Convolution kernels [C_out, C_in/groups, K_h, K_w]
        bias: Optional bias [C_out]
        stride: Stride of the convolution (int or tuple)
        padding: Zero-padding added to both sides (int or tuple)
        dilation: Spacing between kernel elements (int or tuple)
        groups: Number of blocked connections from input to output

    Returns:
        Output tensor [N, C_out, H_out, W_out]
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    # Normalize stride, padding, dilation to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Compute output shape
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_per_group, K_h, K_w = weight.shape

    H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) // stride[1] + 1

    if Tensor._tracing:
        out_shape = (N, C_out, H_out, W_out)
        out_meta = TensorMeta(shape=out_shape, dtype=x.dtype)
        out = Tensor(out_meta)
        inputs = [x, weight]
        if bias is not None:
            inputs.append(bias)
        Tensor._trace_graph.add_op(OpType.CONV2D, inputs, [out],
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=groups)
        return out
    else:
        if x._data is None or weight._data is None:
            raise ValueError("Cannot execute conv2d on symbolic tensor")

        # Pad input if needed
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np.pad(x._data,
                              ((0, 0), (0, 0),
                               (padding[0], padding[0]),
                               (padding[1], padding[1])),
                              mode='constant')
        else:
            x_padded = x._data

        # im2col-based convolution for correctness (not optimized)
        result = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

        for n in range(N):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride[0]
                        w_start = w_out * stride[1]

                        # Sum over input channels and kernel
                        val = 0.0
                        for g in range(groups):
                            c_in_start = (c_out // (C_out // groups)) * C_in_per_group
                            for c_in in range(C_in_per_group):
                                for kh in range(K_h):
                                    for kw in range(K_w):
                                        h_in = h_start + kh * dilation[0]
                                        w_in = w_start + kw * dilation[1]
                                        val += (x_padded[n, c_in_start + c_in, h_in, w_in] *
                                                weight._data[c_out, c_in, kh, kw])
                        result[n, c_out, h_out, w_out] = val

        # Add bias
        if bias is not None:
            result = result + bias._data.reshape(1, -1, 1, 1)

        return Tensor(result.astype(x.dtype))


# ========== Pooling Operations ==========

def max_pool2d(x: 'Tensor',
               kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0) -> 'Tensor':
    """
    2D Max Pooling.

    Args:
        x: Input tensor [N, C, H, W]
        kernel_size: Size of the pooling window
        stride: Stride of the pooling (default: kernel_size)
        padding: Zero-padding added to both sides

    Returns:
        Output tensor [N, C, H_out, W_out]
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    N, C, H_in, W_in = x.shape
    K_h, K_w = kernel_size

    H_out = (H_in + 2 * padding[0] - K_h) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - K_w) // stride[1] + 1

    if Tensor._tracing:
        out_shape = (N, C, H_out, W_out)
        out_meta = TensorMeta(shape=out_shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.MAXPOOL2D, [x], [out],
                                   kernel_size=kernel_size, stride=stride, padding=padding)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute max_pool2d on symbolic tensor")

        # Pad input if needed
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np.pad(x._data,
                              ((0, 0), (0, 0),
                               (padding[0], padding[0]),
                               (padding[1], padding[1])),
                              mode='constant', constant_values=-np.inf)
        else:
            x_padded = x._data

        result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride[0]
                        w_start = w_out * stride[1]
                        window = x_padded[n, c,
                                          h_start:h_start + K_h,
                                          w_start:w_start + K_w]
                        result[n, c, h_out, w_out] = np.max(window)

        return Tensor(result.astype(x.dtype))


def avg_pool2d(x: 'Tensor',
               kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0) -> 'Tensor':
    """
    2D Average Pooling.

    Args:
        x: Input tensor [N, C, H, W]
        kernel_size: Size of the pooling window
        stride: Stride of the pooling (default: kernel_size)
        padding: Zero-padding added to both sides

    Returns:
        Output tensor [N, C, H_out, W_out]
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    N, C, H_in, W_in = x.shape
    K_h, K_w = kernel_size

    H_out = (H_in + 2 * padding[0] - K_h) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - K_w) // stride[1] + 1

    if Tensor._tracing:
        out_shape = (N, C, H_out, W_out)
        out_meta = TensorMeta(shape=out_shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.AVGPOOL2D, [x], [out],
                                   kernel_size=kernel_size, stride=stride, padding=padding)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute avg_pool2d on symbolic tensor")

        # Pad input if needed
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np.pad(x._data,
                              ((0, 0), (0, 0),
                               (padding[0], padding[0]),
                               (padding[1], padding[1])),
                              mode='constant')
        else:
            x_padded = x._data

        result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride[0]
                        w_start = w_out * stride[1]
                        window = x_padded[n, c,
                                          h_start:h_start + K_h,
                                          w_start:w_start + K_w]
                        result[n, c, h_out, w_out] = np.mean(window)

        return Tensor(result.astype(x.dtype))


def adaptive_avg_pool2d(x: 'Tensor', output_size: Union[int, Tuple[int, int]]) -> 'Tensor':
    """
    2D Adaptive Average Pooling - pools to a fixed output size.

    Args:
        x: Input tensor [N, C, H, W]
        output_size: Target output spatial size (H_out, W_out)

    Returns:
        Output tensor [N, C, H_out, W_out]
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    N, C, H_in, W_in = x.shape
    H_out, W_out = output_size

    if Tensor._tracing:
        out_shape = (N, C, H_out, W_out)
        out_meta = TensorMeta(shape=out_shape, dtype=x.dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.ADAPTIVE_AVGPOOL2D, [x], [out],
                                   output_size=output_size)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute adaptive_avg_pool2d on symbolic tensor")

        result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        # Compute input region for this output
                        h_start = (h_out * H_in) // H_out
                        h_end = ((h_out + 1) * H_in) // H_out
                        w_start = (w_out * W_in) // W_out
                        w_end = ((w_out + 1) * W_in) // W_out

                        window = x._data[n, c, h_start:h_end, w_start:w_end]
                        result[n, c, h_out, w_out] = np.mean(window)

        return Tensor(result.astype(x.dtype))


# ========== Tensor Manipulation Operations ==========

def concat(tensors: List['Tensor'], dim: int = 0) -> 'Tensor':
    """
    Concatenate tensors along a dimension.

    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate

    Returns:
        Concatenated tensor
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType
    import builtins

    if len(tensors) == 0:
        raise ValueError("concat requires at least one tensor")
    if len(tensors) == 1:
        return tensors[0]

    # Compute output shape
    shapes = [t.shape for t in tensors]
    out_shape = list(shapes[0])
    out_shape[dim] = builtins.sum(s[dim] for s in shapes)
    out_shape = tuple(out_shape)

    if Tensor._tracing:
        out_meta = TensorMeta(shape=out_shape, dtype=tensors[0].dtype)
        out = Tensor(out_meta)
        Tensor._trace_graph.add_op(OpType.CONCAT, tensors, [out], dim=dim)
        return out
    else:
        arrays = []
        for t in tensors:
            if t._data is None:
                raise ValueError("Cannot execute concat on symbolic tensor")
            arrays.append(t._data)
        result = np.concatenate(arrays, axis=dim)
        return Tensor(result)


def flatten(x: 'Tensor', start_dim: int = 0, end_dim: int = -1) -> 'Tensor':
    """
    Flatten tensor dimensions.

    Args:
        x: Input tensor
        start_dim: First dimension to flatten
        end_dim: Last dimension to flatten

    Returns:
        Flattened tensor
    """
    # Normalize dimensions
    ndim = x.ndim
    if start_dim < 0:
        start_dim = ndim + start_dim
    if end_dim < 0:
        end_dim = ndim + end_dim

    # Compute new shape
    new_shape = list(x.shape[:start_dim])
    flat_size = 1
    for i in range(start_dim, end_dim + 1):
        flat_size *= x.shape[i]
    new_shape.append(flat_size)
    new_shape.extend(x.shape[end_dim + 1:])

    return reshape(x, tuple(new_shape))
