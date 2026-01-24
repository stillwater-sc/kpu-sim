# python/kpu/ops.py
"""
KPU operator definitions.

These operators work with KPU Tensors and support both tracing (for compilation)
and direct execution (for behavioral simulation).

v0.8.1: All operations now route through C++ BehavioralComputeFabric when
        native module is available. This ensures XUE event recording.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor import Tensor

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
        if hasattr(native_module, 'native_relu'):
            _native = native_module
            _native_available = True
        else:
            _native_available = False
    except ImportError:
        _native_available = False
    return _native_available


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
        # Try native C++ execution first
        if _init_native():
            a = np.ascontiguousarray(x._data, dtype=np.float32)
            result = _native.native_relu(a)
            return Tensor(result)
        # Fallback to NumPy
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
        # Try native C++ execution first
        if _init_native():
            a = np.ascontiguousarray(x._data, dtype=np.float32)
            result = _native.native_gelu(a)
            return Tensor(result)
        # Fallback to NumPy - Approximate GELU
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
        # Try native C++ execution first
        if _init_native():
            a = np.ascontiguousarray(x._data, dtype=np.float32)
            result = _native.native_silu(a)
            return Tensor(result)
        # Fallback to NumPy
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
        # Try native C++ execution first
        if _init_native():
            a = np.ascontiguousarray(x._data, dtype=np.float32)
            result = _native.native_sigmoid(a)
            return Tensor(result)
        # Fallback to NumPy
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
        # Try native C++ execution first
        if _init_native():
            a = np.ascontiguousarray(x._data, dtype=np.float32)
            result = _native.native_tanh(a)
            return Tensor(result)
        # Fallback to NumPy
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


# ========== Attention Operations ==========

def scaled_dot_product_attention(
    query: 'Tensor',
    key: 'Tensor',
    value: 'Tensor',
    attn_mask: Optional['Tensor'] = None,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> 'Tensor':
    """
    Scaled Dot-Product Attention.

    Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        query: Query tensor [B, H, S, d_k] or [B, S, d_k]
        key: Key tensor [B, H, S, d_k] or [B, S, d_k]
        value: Value tensor [B, H, S, d_k] or [B, S, d_k]
        attn_mask: Optional attention mask (additive, -inf for masked positions)
        is_causal: If True, apply causal mask (future positions masked)
        scale: Optional scale factor (default: 1/sqrt(d_k))

    Returns:
        Output tensor with same shape as query
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=query.shape, dtype=query.dtype)
        out = Tensor(out_meta)
        inputs = [query, key, value]
        if attn_mask is not None:
            inputs.append(attn_mask)
        Tensor._trace_graph.add_op(
            OpType.ATTENTION, inputs, [out],
            is_causal=is_causal, scale=scale
        )
        return out
    else:
        if query._data is None or key._data is None or value._data is None:
            raise ValueError("Cannot execute attention on symbolic tensor")

        q = query._data
        k = key._data
        v = value._data

        # Get head dimension (last dimension of query)
        d_k = q.shape[-1]

        # Compute scale factor
        if scale is None:
            scale = 1.0 / np.sqrt(d_k)

        # Q @ K^T
        # For [B, H, S, d_k]: transpose last two dims of K -> [B, H, d_k, S]
        k_t = np.swapaxes(k, -2, -1)
        scores = np.matmul(q, k_t) * scale

        # Apply mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask._data

        # Apply causal mask if requested
        if is_causal:
            seq_len = scores.shape[-1]
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores = scores + causal_mask

        # Softmax over last dimension
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Attention @ V
        result = np.matmul(attn_weights, v)

        return Tensor(result.astype(query.dtype))


def multi_head_attention(
    x: 'Tensor',
    w_q: 'Tensor',
    w_k: 'Tensor',
    w_v: 'Tensor',
    w_o: Optional['Tensor'] = None,
    num_heads: int = 1,
    is_causal: bool = False
) -> 'Tensor':
    """
    Multi-Head Attention with QKV and output projections.

    Computes:
        Q, K, V = X @ W_Q, X @ W_K, X @ W_V
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        where head_i = Attention(Q_i, K_i, V_i)

    Args:
        x: Input tensor [B, S, D]
        w_q: Query projection weights [D, D]
        w_k: Key projection weights [D, D]
        w_v: Value projection weights [D, D]
        w_o: Output projection weights [D, D] (optional)
        num_heads: Number of attention heads
        is_causal: If True, apply causal mask

    Returns:
        Output tensor [B, S, D]
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        inputs = [x, w_q, w_k, w_v]
        if w_o is not None:
            inputs.append(w_o)
        Tensor._trace_graph.add_op(
            OpType.ATTENTION, inputs, [out],
            num_heads=num_heads, is_causal=is_causal,
            include_qkv_projection=True,
            include_output_projection=(w_o is not None)
        )
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute attention on symbolic tensor")

        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // num_heads

        # Project Q, K, V
        q = np.matmul(x._data, w_q._data)  # [B, S, D]
        k = np.matmul(x._data, w_k._data)  # [B, S, D]
        v = np.matmul(x._data, w_v._data)  # [B, S, D]

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

        # Output projection
        if w_o is not None:
            attn_output = np.matmul(attn_output, w_o._data)

        return Tensor(attn_output.astype(x.dtype))


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


def batch_norm2d(x: 'Tensor',
                 weight: Optional['Tensor'] = None,
                 bias: Optional['Tensor'] = None,
                 running_mean: Optional['Tensor'] = None,
                 running_var: Optional['Tensor'] = None,
                 training: bool = False,
                 eps: float = 1e-5,
                 momentum: float = 0.1) -> 'Tensor':
    """
    2D Batch Normalization.

    In training mode: normalizes using batch statistics.
    In inference mode: uses running_mean and running_var if provided,
    otherwise computes from batch.

    Args:
        x: Input tensor [N, C, H, W]
        weight: Scale parameter (gamma) [C]
        bias: Shift parameter (beta) [C]
        running_mean: Running mean for inference [C]
        running_var: Running variance for inference [C]
        training: Whether in training mode
        eps: Small constant for numerical stability
        momentum: Momentum for running statistics update

    Returns:
        Normalized tensor [N, C, H, W]
    """
    from .tensor import Tensor, TensorMeta
    from .graph import OpType

    if Tensor._tracing:
        out_meta = TensorMeta(shape=x.shape, dtype=x.dtype)
        out = Tensor(out_meta)
        inputs = [x]
        if weight is not None:
            inputs.append(weight)
        if bias is not None:
            inputs.append(bias)
        Tensor._trace_graph.add_op(OpType.BATCH_NORM, inputs, [out],
                                   eps=eps, training=training)
        return out
    else:
        if x._data is None:
            raise ValueError("Cannot execute batch_norm2d on symbolic tensor")

        # Use running stats in inference mode if available
        if not training and running_mean is not None and running_var is not None:
            mean = running_mean._data.reshape(1, -1, 1, 1)
            var = running_var._data.reshape(1, -1, 1, 1)
        else:
            # Compute from batch
            mean = np.mean(x._data, axis=(0, 2, 3), keepdims=True)
            var = np.var(x._data, axis=(0, 2, 3), keepdims=True)

        # Normalize
        result = (x._data - mean) / np.sqrt(var + eps)

        # Apply scale and shift
        if weight is not None:
            result = result * weight._data.reshape(1, -1, 1, 1)
        if bias is not None:
            result = result + bias._data.reshape(1, -1, 1, 1)

        return Tensor(result.astype(x.dtype))


# ========== Convolution Operations ==========

def _im2col(x: np.ndarray, K_h: int, K_w: int, stride: Tuple[int, int],
            dilation: Tuple[int, int], H_out: int, W_out: int) -> np.ndarray:
    """
    Extract image patches for efficient convolution via matrix multiplication.

    Converts [N, C, H, W] input to [N * H_out * W_out, C * K_h * K_w] matrix
    where each row contains one flattened patch that will be convolved.

    This is the key to fast convolution - instead of 6 nested loops,
    we extract all patches at once and use a single matmul.
    """
    N, C, H, W = x.shape

    # Use stride_tricks to create a view of all patches without copying
    # This is the fastest way to extract patches in numpy
    shape = (N, C, H_out, W_out, K_h, K_w)
    strides = (
        x.strides[0],                          # batch stride
        x.strides[1],                          # channel stride
        x.strides[2] * stride[0],              # output row stride
        x.strides[3] * stride[1],              # output col stride
        x.strides[2] * dilation[0],            # kernel row stride
        x.strides[3] * dilation[1],            # kernel col stride
    )

    # Create view of patches [N, C, H_out, W_out, K_h, K_w]
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Reshape to [N, H_out, W_out, C * K_h * K_w] then [N * H_out * W_out, C * K_h * K_w]
    col = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, -1)

    return col


def _conv2d_im2col(x_padded: np.ndarray, weight: np.ndarray,
                   stride: Tuple[int, int], dilation: Tuple[int, int],
                   groups: int, N: int, C_out: int, C_in_per_group: int,
                   K_h: int, K_w: int, H_out: int, W_out: int) -> np.ndarray:
    """
    Optimized conv2d using im2col + matmul.

    Converts convolution to matrix multiplication:
    - Extract patches: [N * H_out * W_out, C_in * K_h * K_w]
    - Reshape weights: [C_out, C_in * K_h * K_w]
    - Matmul: patches @ weights.T -> [N * H_out * W_out, C_out]
    - Reshape to [N, C_out, H_out, W_out]

    This is ~100-1000x faster than nested Python loops.
    """
    if groups == 1:
        # Standard convolution - most common case
        # Extract all patches at once
        col = _im2col(x_padded, K_h, K_w, stride, dilation, H_out, W_out)

        # Reshape weight to [C_out, C_in * K_h * K_w]
        weight_col = weight.reshape(C_out, -1)

        # Single matmul: [N*H_out*W_out, C_in*K_h*K_w] @ [C_in*K_h*K_w, C_out]
        out = col @ weight_col.T

        # Reshape to [N, H_out, W_out, C_out] then transpose to [N, C_out, H_out, W_out]
        result = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)
    else:
        # Grouped convolution (e.g., depthwise separable)
        C_in = x_padded.shape[1]
        C_in_per_group = C_in // groups
        C_out_per_group = C_out // groups

        result = np.zeros((N, C_out, H_out, W_out), dtype=x_padded.dtype)

        for g in range(groups):
            # Extract input channels for this group
            x_g = x_padded[:, g * C_in_per_group:(g + 1) * C_in_per_group]

            # Extract weight for this group
            w_g = weight[g * C_out_per_group:(g + 1) * C_out_per_group]

            # im2col for this group
            col = _im2col(x_g, K_h, K_w, stride, dilation, H_out, W_out)

            # Reshape weight
            weight_col = w_g.reshape(C_out_per_group, -1)

            # Matmul
            out = col @ weight_col.T

            # Store result
            result[:, g * C_out_per_group:(g + 1) * C_out_per_group] = \
                out.reshape(N, H_out, W_out, C_out_per_group).transpose(0, 3, 1, 2)

    return result


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

        # Use optimized im2col-based convolution (converts conv to matmul)
        # This is ~100-1000x faster than nested loops
        result = _conv2d_im2col(x_padded, weight._data, stride, dilation,
                                groups, N, C_out, C_in_per_group,
                                K_h, K_w, H_out, W_out)

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

        # Optimized pooling using stride_tricks (avoids nested loops)
        # Create view of all pooling windows [N, C, H_out, W_out, K_h, K_w]
        shape = (N, C, H_out, W_out, K_h, K_w)
        strides = (
            x_padded.strides[0],                    # batch stride
            x_padded.strides[1],                    # channel stride
            x_padded.strides[2] * stride[0],        # output row stride
            x_padded.strides[3] * stride[1],        # output col stride
            x_padded.strides[2],                    # kernel row stride
            x_padded.strides[3],                    # kernel col stride
        )
        windows = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

        # Max over the last two dimensions (kernel dimensions)
        result = windows.max(axis=(4, 5))

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

        # Optimized pooling using stride_tricks (avoids nested loops)
        # Create view of all pooling windows [N, C, H_out, W_out, K_h, K_w]
        shape = (N, C, H_out, W_out, K_h, K_w)
        strides = (
            x_padded.strides[0],                    # batch stride
            x_padded.strides[1],                    # channel stride
            x_padded.strides[2] * stride[0],        # output row stride
            x_padded.strides[3] * stride[1],        # output col stride
            x_padded.strides[2],                    # kernel row stride
            x_padded.strides[3],                    # kernel col stride
        )
        windows = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

        # Mean over the last two dimensions (kernel dimensions)
        result = windows.mean(axis=(4, 5))

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

        # Optimize common case: global average pooling (output_size = 1x1)
        if H_out == 1 and W_out == 1:
            # Just average over spatial dimensions - very fast
            result = x._data.mean(axis=(2, 3), keepdims=True)
        elif H_in % H_out == 0 and W_in % W_out == 0:
            # Uniform pooling regions - can use reshape + mean
            pool_h = H_in // H_out
            pool_w = W_in // W_out
            # Reshape to [N, C, H_out, pool_h, W_out, pool_w] and mean over pool dims
            reshaped = x._data.reshape(N, C, H_out, pool_h, W_out, pool_w)
            result = reshaped.mean(axis=(3, 5))
        else:
            # Non-uniform regions - need per-output computation
            result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = (h_out * H_in) // H_out
                    h_end = ((h_out + 1) * H_in) // H_out
                    w_start = (w_out * W_in) // W_out
                    w_end = ((w_out + 1) * W_in) // W_out
                    result[:, :, h_out, w_out] = x._data[:, :, h_start:h_end, w_start:w_end].mean(axis=(2, 3))

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
