"""Quantization support for KPU simulator.

This module provides infrastructure for simulating quantized inference,
including INT8, FP16, BF16, and other low-precision data types.

The approach follows a tiered strategy:
- BEHAVIORAL: Emulation with scale/zero_point for correctness
- TRANSACTIONAL: Dtype info for memory traffic calculation
- CYCLE_ACCURATE: Full Universal library integration in C++
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np


class QuantDtype(Enum):
    """Supported quantization data types."""
    # Full precision (reference)
    FP32 = "fp32"

    # 16-bit types
    FP16 = "fp16"
    BF16 = "bf16"

    # 8-bit float types
    FP8_E4M3 = "fp8_e4m3"  # 4-bit exponent, 3-bit mantissa (NVIDIA/OCP)
    FP8_E5M2 = "fp8_e5m2"  # 5-bit exponent, 2-bit mantissa (wider range)
    FP8_E3M4 = "fp8_e3m4"  # 3-bit exponent, 4-bit mantissa
    FP8_E2M5 = "fp8_e2m5"  # 2-bit exponent, 5-bit mantissa (more precision)

    # Integer types
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"
    UINT4 = "uint4"

    # 4-bit float
    FP4 = "fp4"

    @property
    def bytes_per_element(self) -> float:
        """Return bytes per element for this dtype."""
        byte_sizes = {
            QuantDtype.FP32: 4.0,
            QuantDtype.FP16: 2.0,
            QuantDtype.BF16: 2.0,
            QuantDtype.FP8_E4M3: 1.0,
            QuantDtype.FP8_E5M2: 1.0,
            QuantDtype.FP8_E3M4: 1.0,
            QuantDtype.FP8_E2M5: 1.0,
            QuantDtype.INT8: 1.0,
            QuantDtype.UINT8: 1.0,
            QuantDtype.INT4: 0.5,
            QuantDtype.UINT4: 0.5,
            QuantDtype.FP4: 0.5,
        }
        return byte_sizes.get(self, 4.0)

    @property
    def is_integer(self) -> bool:
        """Return True if this is an integer type."""
        return self in (QuantDtype.INT8, QuantDtype.UINT8,
                       QuantDtype.INT4, QuantDtype.UINT4)

    @property
    def is_signed(self) -> bool:
        """Return True if this is a signed type."""
        return self in (QuantDtype.INT8, QuantDtype.INT4)

    @property
    def qmin(self) -> int:
        """Minimum quantized value for integer types."""
        if self == QuantDtype.INT8:
            return -128
        elif self == QuantDtype.UINT8:
            return 0
        elif self == QuantDtype.INT4:
            return -8
        elif self == QuantDtype.UINT4:
            return 0
        return 0

    @property
    def qmax(self) -> int:
        """Maximum quantized value for integer types."""
        if self == QuantDtype.INT8:
            return 127
        elif self == QuantDtype.UINT8:
            return 255
        elif self == QuantDtype.INT4:
            return 7
        elif self == QuantDtype.UINT4:
            return 15
        return 255


@dataclass
class QuantizationConfig:
    """Configuration for quantized tensor/operation.

    Attributes:
        dtype: Target quantization data type
        scale: Scale factor for affine quantization (float = (int - zero_point) * scale)
        zero_point: Zero point for affine quantization
        per_channel: If True, scale/zero_point are per-channel (axis 0)
        channel_scales: Per-channel scales (if per_channel=True)
        channel_zero_points: Per-channel zero points (if per_channel=True)
        symmetric: If True, use symmetric quantization (zero_point = 0)
    """
    dtype: QuantDtype = QuantDtype.INT8
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    per_channel: bool = False
    channel_scales: Optional[np.ndarray] = None
    channel_zero_points: Optional[np.ndarray] = None
    symmetric: bool = False

    def __post_init__(self):
        if self.symmetric and self.zero_point is not None and self.zero_point != 0:
            raise ValueError("Symmetric quantization requires zero_point=0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "dtype": self.dtype.value,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
        }
        if self.scale is not None:
            d["scale"] = self.scale
        if self.zero_point is not None:
            d["zero_point"] = self.zero_point
        if self.channel_scales is not None:
            d["channel_scales"] = self.channel_scales.tolist()
        if self.channel_zero_points is not None:
            d["channel_zero_points"] = self.channel_zero_points.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QuantizationConfig':
        """Create from dictionary."""
        dtype = QuantDtype(d["dtype"])
        config = cls(
            dtype=dtype,
            scale=d.get("scale"),
            zero_point=d.get("zero_point"),
            per_channel=d.get("per_channel", False),
            symmetric=d.get("symmetric", False),
        )
        if "channel_scales" in d:
            config.channel_scales = np.array(d["channel_scales"], dtype=np.float32)
        if "channel_zero_points" in d:
            config.channel_zero_points = np.array(d["channel_zero_points"], dtype=np.int32)
        return config


# --- Calibration utilities ---

def compute_scale_zero_point(
    tensor: np.ndarray,
    dtype: QuantDtype = QuantDtype.INT8,
    symmetric: bool = False,
    percentile: float = 100.0,
) -> Tuple[float, int]:
    """Compute scale and zero_point for quantizing a tensor.

    Args:
        tensor: Input tensor to analyze
        dtype: Target quantization dtype
        symmetric: If True, use symmetric quantization
        percentile: Percentile for range calculation (for outlier clipping)

    Returns:
        (scale, zero_point) tuple
    """
    if percentile < 100.0:
        # Clip outliers
        min_val = np.percentile(tensor, 100 - percentile)
        max_val = np.percentile(tensor, percentile)
    else:
        min_val = tensor.min()
        max_val = tensor.max()

    qmin = dtype.qmin
    qmax = dtype.qmax

    if symmetric:
        # Symmetric: zero_point = 0, range symmetric around 0
        abs_max = max(abs(min_val), abs(max_val))
        scale = abs_max / max(abs(qmin), qmax)
        zero_point = 0
    else:
        # Affine: map [min_val, max_val] to [qmin, qmax]
        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            scale = 1.0
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))

    return float(scale), int(zero_point)


def compute_per_channel_params(
    tensor: np.ndarray,
    axis: int = 0,
    dtype: QuantDtype = QuantDtype.INT8,
    symmetric: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel scale and zero_point.

    Args:
        tensor: Input tensor (weights typically)
        axis: Channel axis (usually 0 for output channels)
        dtype: Target quantization dtype
        symmetric: If True, use symmetric quantization

    Returns:
        (scales, zero_points) arrays with shape matching axis dimension
    """
    num_channels = tensor.shape[axis]
    scales = np.zeros(num_channels, dtype=np.float32)
    zero_points = np.zeros(num_channels, dtype=np.int32)

    for i in range(num_channels):
        # Extract channel slice
        slices = [slice(None)] * tensor.ndim
        slices[axis] = i
        channel_data = tensor[tuple(slices)]

        scale, zp = compute_scale_zero_point(channel_data, dtype, symmetric)
        scales[i] = scale
        zero_points[i] = zp

    return scales, zero_points


# --- Quantize/Dequantize operations ---

def quantize(
    tensor: np.ndarray,
    scale: float,
    zero_point: int,
    dtype: QuantDtype = QuantDtype.INT8,
) -> np.ndarray:
    """Quantize a float tensor to integer representation.

    Args:
        tensor: Input float tensor
        scale: Scale factor
        zero_point: Zero point offset
        dtype: Target quantization dtype

    Returns:
        Quantized integer tensor
    """
    qmin = dtype.qmin
    qmax = dtype.qmax

    # Quantize: q = round(x / scale) + zero_point
    q = np.round(tensor / scale) + zero_point
    q = np.clip(q, qmin, qmax)

    # Select numpy dtype
    if dtype in (QuantDtype.INT8,):
        np_dtype = np.int8
    elif dtype in (QuantDtype.UINT8,):
        np_dtype = np.uint8
    elif dtype in (QuantDtype.INT4, QuantDtype.UINT4):
        # Store as int8 for now (packed storage TBD)
        np_dtype = np.int8
    else:
        np_dtype = np.int8

    return q.astype(np_dtype)


def dequantize(
    tensor: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Dequantize an integer tensor back to float.

    Args:
        tensor: Quantized integer tensor
        scale: Scale factor
        zero_point: Zero point offset

    Returns:
        Dequantized float32 tensor
    """
    # Dequantize: x = (q - zero_point) * scale
    return (tensor.astype(np.float32) - zero_point) * scale


def quantize_per_channel(
    tensor: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    axis: int = 0,
    dtype: QuantDtype = QuantDtype.INT8,
) -> np.ndarray:
    """Quantize with per-channel parameters.

    Args:
        tensor: Input float tensor
        scales: Per-channel scales
        zero_points: Per-channel zero points
        axis: Channel axis
        dtype: Target quantization dtype

    Returns:
        Quantized integer tensor
    """
    qmin = dtype.qmin
    qmax = dtype.qmax

    # Reshape scales/zero_points for broadcasting
    shape = [1] * tensor.ndim
    shape[axis] = -1
    scales_bc = scales.reshape(shape)
    zp_bc = zero_points.reshape(shape)

    q = np.round(tensor / scales_bc) + zp_bc
    q = np.clip(q, qmin, qmax)

    return q.astype(np.int8)


def dequantize_per_channel(
    tensor: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Dequantize with per-channel parameters."""
    shape = [1] * tensor.ndim
    shape[axis] = -1
    scales_bc = scales.reshape(shape)
    zp_bc = zero_points.reshape(shape)

    return (tensor.astype(np.float32) - zp_bc) * scales_bc


# --- Quantized operations ---

def quantized_matmul_int8(
    a: np.ndarray,
    b: np.ndarray,
    scale_a: float,
    zero_point_a: int,
    scale_b: float,
    zero_point_b: int,
    scale_out: Optional[float] = None,
    zero_point_out: Optional[int] = None,
    output_float: bool = True,
) -> np.ndarray:
    """Perform INT8 quantized matrix multiplication.

    This emulates quantized matmul by:
    1. Dequantizing inputs to float32
    2. Performing matmul in float32
    3. Optionally requantizing output

    For accurate INT8 simulation, the computation should ideally be:
    C = (A - zp_a) @ (B - zp_b) * scale_a * scale_b

    Args:
        a: Quantized input A (int8)
        b: Quantized input B (int8)
        scale_a, zero_point_a: Quantization params for A
        scale_b, zero_point_b: Quantization params for B
        scale_out, zero_point_out: Output quantization params (if requantizing)
        output_float: If True, return float32; else return quantized int8

    Returns:
        Result matrix (float32 or int8)
    """
    # Dequantize inputs
    a_fp = (a.astype(np.int32) - zero_point_a) * scale_a
    b_fp = (b.astype(np.int32) - zero_point_b) * scale_b

    # Perform matmul in float32
    c_fp = np.matmul(a_fp, b_fp)

    if output_float or scale_out is None:
        return c_fp.astype(np.float32)
    else:
        # Requantize output
        return quantize(c_fp, scale_out, zero_point_out, QuantDtype.INT8)


def quantized_linear_int8(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    scale_x: float,
    zero_point_x: int,
    scale_w: float,
    zero_point_w: int,
    scale_out: Optional[float] = None,
    zero_point_out: Optional[int] = None,
    output_float: bool = True,
) -> np.ndarray:
    """Perform INT8 quantized linear layer.

    Args:
        x: Quantized input (int8), shape [..., in_features]
        weight: Quantized weights (int8), shape [out_features, in_features]
        bias: Optional float32 bias, shape [out_features]
        scale_x, zero_point_x: Input quantization params
        scale_w, zero_point_w: Weight quantization params
        scale_out, zero_point_out: Output quantization params
        output_float: If True, return float32

    Returns:
        Result tensor
    """
    # Dequantize
    x_fp = (x.astype(np.int32) - zero_point_x) * scale_x
    w_fp = (weight.astype(np.int32) - zero_point_w) * scale_w

    # Linear operation: y = x @ w.T + bias
    y_fp = np.matmul(x_fp, w_fp.T)

    if bias is not None:
        y_fp = y_fp + bias

    if output_float or scale_out is None:
        return y_fp.astype(np.float32)
    else:
        return quantize(y_fp, scale_out, zero_point_out, QuantDtype.INT8)


# --- FP16 operations (v0.7.1) ---

def fp16_matmul(
    a: np.ndarray,
    b: np.ndarray,
    output_fp32: bool = True,
) -> np.ndarray:
    """Perform FP16 matrix multiplication using NumPy native float16.

    This uses NumPy's native float16 support for actual half-precision
    computation, providing realistic FP16 behavior including reduced
    precision and potential overflow for large values.

    Args:
        a: Input matrix A, shape [..., M, K]
        b: Input matrix B, shape [..., K, N]
        output_fp32: If True, return float32; else return float16

    Returns:
        Result matrix C, shape [..., M, N]
    """
    # Convert to FP16 for computation
    a_fp16 = a.astype(np.float16)
    b_fp16 = b.astype(np.float16)

    # Perform matmul in FP16
    c_fp16 = np.matmul(a_fp16, b_fp16)

    if output_fp32:
        return c_fp16.astype(np.float32)
    return c_fp16


def fp16_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    output_fp32: bool = True,
) -> np.ndarray:
    """Perform FP16 linear layer (y = x @ weight.T + bias).

    Args:
        x: Input tensor, shape [..., in_features]
        weight: Weight matrix, shape [out_features, in_features]
        bias: Optional bias vector, shape [out_features]
        output_fp32: If True, return float32; else return float16

    Returns:
        Output tensor, shape [..., out_features]
    """
    # Convert to FP16
    x_fp16 = x.astype(np.float16)
    w_fp16 = weight.astype(np.float16)

    # Linear operation: y = x @ w.T
    y_fp16 = np.matmul(x_fp16, w_fp16.T)

    if bias is not None:
        bias_fp16 = bias.astype(np.float16)
        y_fp16 = y_fp16 + bias_fp16

    if output_fp32:
        return y_fp16.astype(np.float32)
    return y_fp16


def fp16_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    output_fp32: bool = True,
) -> np.ndarray:
    """Perform FP16 2D convolution.

    Args:
        x: Input tensor, shape [N, C_in, H, W]
        weight: Filter weights, shape [C_out, C_in, kH, kW]
        bias: Optional bias, shape [C_out]
        stride: Stride (sH, sW)
        padding: Padding (pH, pW)
        output_fp32: If True, return float32; else return float16

    Returns:
        Output tensor, shape [N, C_out, H_out, W_out]
    """
    # Convert to FP16
    x_fp16 = x.astype(np.float16)
    w_fp16 = weight.astype(np.float16)

    N, C_in, H, W = x_fp16.shape
    C_out, _, kH, kW = w_fp16.shape
    sH, sW = stride
    pH, pW = padding

    # Pad input
    if pH > 0 or pW > 0:
        x_fp16 = np.pad(x_fp16, ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                        mode='constant', constant_values=0)

    # Output dimensions
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1

    # im2col style convolution in FP16
    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float16)

    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * sH
                    w_start = w * sW
                    patch = x_fp16[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    output[n, c_out, h, w] = np.sum(patch * w_fp16[c_out])

    if bias is not None:
        bias_fp16 = bias.astype(np.float16)
        output = output + bias_fp16.reshape(1, -1, 1, 1)

    if output_fp32:
        return output.astype(np.float32)
    return output


def cast_to_fp16(tensor: np.ndarray) -> np.ndarray:
    """Cast a tensor to FP16.

    Args:
        tensor: Input tensor (any dtype)

    Returns:
        FP16 tensor
    """
    return tensor.astype(np.float16)


def cast_from_fp16(tensor: np.ndarray) -> np.ndarray:
    """Cast an FP16 tensor back to FP32.

    Args:
        tensor: Input FP16 tensor

    Returns:
        FP32 tensor
    """
    return tensor.astype(np.float32)


def fp16_range() -> Tuple[float, float]:
    """Return the representable range of FP16.

    Returns:
        (min_value, max_value) tuple for FP16
    """
    info = np.finfo(np.float16)
    return (float(info.min), float(info.max))


def fp16_precision() -> float:
    """Return the machine epsilon for FP16.

    Returns:
        Machine epsilon (smallest representable difference from 1.0)
    """
    return float(np.finfo(np.float16).eps)


# --- BF16 operations (v0.7.2) ---

# Try to import ml_dtypes for native bfloat16 support
try:
    import ml_dtypes
    _BFLOAT16_DTYPE = ml_dtypes.bfloat16
    _ML_DTYPES_AVAILABLE = True
except ImportError:
    _BFLOAT16_DTYPE = None
    _ML_DTYPES_AVAILABLE = False


def is_bfloat16_native() -> bool:
    """Check if native bfloat16 support is available via ml_dtypes.

    Returns:
        True if ml_dtypes is installed and provides native bfloat16
    """
    return _ML_DTYPES_AVAILABLE


def _emulate_bf16(value: np.ndarray) -> np.ndarray:
    """Emulate bfloat16 by truncating float32 mantissa.

    BF16 has the same exponent range as FP32 (8 bits) but only 7 mantissa
    bits (vs 23 for FP32). This emulation truncates the lower 16 bits
    of the float32 representation.

    Args:
        value: Input float32 array

    Returns:
        Float32 array with precision reduced to bfloat16 equivalent
    """
    # View as uint32 to manipulate bits
    as_float32 = value.astype(np.float32)
    as_uint32 = as_float32.view(np.uint32)
    # Zero out lower 16 bits (truncate mantissa to 7 bits)
    truncated = as_uint32 & 0xFFFF0000
    # View back as float32
    return truncated.view(np.float32)


def bf16_matmul(
    a: np.ndarray,
    b: np.ndarray,
    output_fp32: bool = True,
) -> np.ndarray:
    """Perform BF16 matrix multiplication.

    Uses ml_dtypes.bfloat16 if available, otherwise emulates BF16 precision
    by truncating float32 mantissa bits.

    BF16 has the same dynamic range as FP32 (8-bit exponent) but reduced
    precision (7-bit mantissa vs 23-bit). This makes it ideal for deep
    learning where range matters more than precision.

    Args:
        a: Input matrix A, shape [..., M, K]
        b: Input matrix B, shape [..., K, N]
        output_fp32: If True, return float32; else return bfloat16

    Returns:
        Result matrix C, shape [..., M, N]
    """
    if _ML_DTYPES_AVAILABLE:
        # Use native bfloat16 via ml_dtypes
        a_bf16 = a.astype(_BFLOAT16_DTYPE)
        b_bf16 = b.astype(_BFLOAT16_DTYPE)
        # ml_dtypes may compute in higher precision internally
        c_bf16 = np.matmul(a_bf16.astype(np.float32), b_bf16.astype(np.float32))
        c_bf16 = c_bf16.astype(_BFLOAT16_DTYPE)
        if output_fp32:
            return c_bf16.astype(np.float32)
        return c_bf16
    else:
        # Emulate bfloat16 by truncating precision
        a_bf16 = _emulate_bf16(a)
        b_bf16 = _emulate_bf16(b)
        c = np.matmul(a_bf16, b_bf16)
        c_bf16 = _emulate_bf16(c)
        return c_bf16


def bf16_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    output_fp32: bool = True,
) -> np.ndarray:
    """Perform BF16 linear layer (y = x @ weight.T + bias).

    Args:
        x: Input tensor, shape [..., in_features]
        weight: Weight matrix, shape [out_features, in_features]
        bias: Optional bias vector, shape [out_features]
        output_fp32: If True, return float32; else return bfloat16

    Returns:
        Output tensor, shape [..., out_features]
    """
    if _ML_DTYPES_AVAILABLE:
        x_bf16 = x.astype(_BFLOAT16_DTYPE)
        w_bf16 = weight.astype(_BFLOAT16_DTYPE)
        # Compute in float32 for accumulation, convert back to bf16
        y = np.matmul(x_bf16.astype(np.float32), w_bf16.astype(np.float32).T)
        if bias is not None:
            bias_bf16 = bias.astype(_BFLOAT16_DTYPE)
            y = y + bias_bf16.astype(np.float32)
        y_bf16 = y.astype(_BFLOAT16_DTYPE)
        if output_fp32:
            return y_bf16.astype(np.float32)
        return y_bf16
    else:
        # Emulate
        x_bf16 = _emulate_bf16(x)
        w_bf16 = _emulate_bf16(weight)
        y = np.matmul(x_bf16, w_bf16.T)
        if bias is not None:
            bias_bf16 = _emulate_bf16(bias)
            y = y + bias_bf16
        return _emulate_bf16(y)


def bf16_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    output_fp32: bool = True,
) -> np.ndarray:
    """Perform BF16 2D convolution.

    Args:
        x: Input tensor, shape [N, C_in, H, W]
        weight: Filter weights, shape [C_out, C_in, kH, kW]
        bias: Optional bias, shape [C_out]
        stride: Stride (sH, sW)
        padding: Padding (pH, pW)
        output_fp32: If True, return float32; else return bfloat16

    Returns:
        Output tensor, shape [N, C_out, H_out, W_out]
    """
    if _ML_DTYPES_AVAILABLE:
        x_bf16 = x.astype(_BFLOAT16_DTYPE)
        w_bf16 = weight.astype(_BFLOAT16_DTYPE)
    else:
        x_bf16 = _emulate_bf16(x)
        w_bf16 = _emulate_bf16(weight)

    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    sH, sW = stride
    pH, pW = padding

    # Pad input (work in float32 for computation)
    x_work = x_bf16.astype(np.float32) if _ML_DTYPES_AVAILABLE else x_bf16
    w_work = w_bf16.astype(np.float32) if _ML_DTYPES_AVAILABLE else w_bf16

    if pH > 0 or pW > 0:
        x_work = np.pad(x_work, ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                        mode='constant', constant_values=0)

    # Output dimensions
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1

    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * sH
                    w_start = w * sW
                    patch = x_work[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    output[n, c_out, h, w] = np.sum(patch * w_work[c_out])

    if bias is not None:
        if _ML_DTYPES_AVAILABLE:
            bias_work = bias.astype(_BFLOAT16_DTYPE).astype(np.float32)
        else:
            bias_work = _emulate_bf16(bias)
        output = output + bias_work.reshape(1, -1, 1, 1)

    # Convert to bf16
    if _ML_DTYPES_AVAILABLE:
        output_bf16 = output.astype(_BFLOAT16_DTYPE)
        if output_fp32:
            return output_bf16.astype(np.float32)
        return output_bf16
    else:
        return _emulate_bf16(output)


def cast_to_bf16(tensor: np.ndarray) -> np.ndarray:
    """Cast a tensor to BF16.

    Uses ml_dtypes.bfloat16 if available, otherwise returns emulated
    bfloat16 as float32 with truncated precision.

    Args:
        tensor: Input tensor (any dtype)

    Returns:
        BF16 tensor (ml_dtypes.bfloat16 or emulated float32)
    """
    if _ML_DTYPES_AVAILABLE:
        return tensor.astype(_BFLOAT16_DTYPE)
    return _emulate_bf16(tensor.astype(np.float32))


def cast_from_bf16(tensor: np.ndarray) -> np.ndarray:
    """Cast a BF16 tensor back to FP32.

    Args:
        tensor: Input BF16 tensor

    Returns:
        FP32 tensor
    """
    return tensor.astype(np.float32)


def bf16_range() -> Tuple[float, float]:
    """Return the representable range of BF16.

    BF16 has the same exponent range as FP32 (8 bits), so it can
    represent approximately the same range of values.

    Returns:
        (min_value, max_value) tuple for BF16
    """
    # BF16 uses 8-bit exponent like FP32, so similar range
    # Max: ~3.4e38, Min: ~-3.4e38 (same as FP32)
    # But with reduced precision (7-bit mantissa)
    return (-3.3895313892515355e+38, 3.3895313892515355e+38)


def bf16_precision() -> float:
    """Return the machine epsilon for BF16.

    BF16 has 7 mantissa bits, giving roughly 2-3 decimal digits of precision.
    Epsilon = 2^-7 ≈ 0.0078125

    Returns:
        Machine epsilon for bfloat16
    """
    # BF16: 1 sign + 8 exponent + 7 mantissa
    # Epsilon = 2^(-7) = 0.0078125
    return 0.0078125


# --- FP8 operations (v0.7.3) ---

# FP8 format specifications
# Format: fp8_e{exp}m{mantissa} where exp + mantissa + 1(sign) = 8
#
# | Format  | Exp | Mantissa | Range      | Precision | Use Case |
# |---------|-----|----------|------------|-----------|----------|
# | E2M5    | 2   | 5        | ±6.0       | High      | Gradients |
# | E3M4    | 3   | 4        | ±30        | Medium    | General |
# | E4M3    | 4   | 3        | ±448       | Low       | Weights (NVIDIA) |
# | E5M2    | 5   | 2        | ±57344     | Very Low  | Activations |

# Try to import ml_dtypes FP8 types (only e4m3fn and e5m2 are available)
try:
    import ml_dtypes
    _FP8_E4M3_DTYPE = getattr(ml_dtypes, 'float8_e4m3fn', None)
    _FP8_E5M2_DTYPE = getattr(ml_dtypes, 'float8_e5m2', None)
    _FP8_ML_DTYPES_AVAILABLE = _FP8_E4M3_DTYPE is not None
except ImportError:
    _FP8_E4M3_DTYPE = None
    _FP8_E5M2_DTYPE = None
    _FP8_ML_DTYPES_AVAILABLE = False


class FP8Format:
    """FP8 format specification."""

    def __init__(self, name: str, exp_bits: int, mantissa_bits: int,
                 bias: int, has_inf: bool = False, has_nan: bool = True):
        """Initialize FP8 format.

        Args:
            name: Format name (e.g., "e4m3")
            exp_bits: Number of exponent bits
            mantissa_bits: Number of mantissa bits
            bias: Exponent bias
            has_inf: Whether format supports infinity
            has_nan: Whether format supports NaN
        """
        self.name = name
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias
        self.has_inf = has_inf
        self.has_nan = has_nan

        # Calculate range
        max_exp = (1 << exp_bits) - 1 - (1 if has_nan else 0)
        self.max_normal = (2 - 2**(-mantissa_bits)) * (2 ** (max_exp - bias))
        self.min_normal = 2 ** (1 - bias)
        self.min_subnormal = 2 ** (1 - bias - mantissa_bits)

        # Epsilon (precision)
        self.epsilon = 2 ** (-mantissa_bits)

    @property
    def max_value(self) -> float:
        return self.max_normal

    @property
    def min_value(self) -> float:
        return -self.max_normal


# Standard FP8 formats
FP8_E2M5 = FP8Format("e2m5", exp_bits=2, mantissa_bits=5, bias=1)
FP8_E3M4 = FP8Format("e3m4", exp_bits=3, mantissa_bits=4, bias=3)
FP8_E4M3 = FP8Format("e4m3", exp_bits=4, mantissa_bits=3, bias=7, has_nan=True)  # NVIDIA/OCP
FP8_E5M2 = FP8Format("e5m2", exp_bits=5, mantissa_bits=2, bias=15, has_inf=True, has_nan=True)

# Format lookup
_FP8_FORMATS = {
    "e2m5": FP8_E2M5,
    "e3m4": FP8_E3M4,
    "e4m3": FP8_E4M3,
    "e5m2": FP8_E5M2,
}


def get_fp8_format(format_name: str) -> FP8Format:
    """Get FP8 format specification by name.

    Args:
        format_name: Format name ("e2m5", "e3m4", "e4m3", "e5m2")

    Returns:
        FP8Format specification
    """
    if format_name not in _FP8_FORMATS:
        raise ValueError(f"Unknown FP8 format: {format_name}. "
                        f"Available: {list(_FP8_FORMATS.keys())}")
    return _FP8_FORMATS[format_name]


def is_fp8_native(format_name: str = "e4m3") -> bool:
    """Check if native FP8 support is available via ml_dtypes.

    Args:
        format_name: FP8 format to check

    Returns:
        True if ml_dtypes provides native support for this format
    """
    if not _FP8_ML_DTYPES_AVAILABLE:
        return False
    if format_name in ("e4m3", "e4m3fn"):
        return _FP8_E4M3_DTYPE is not None
    if format_name == "e5m2":
        return _FP8_E5M2_DTYPE is not None
    return False  # e2m5, e3m4 not in ml_dtypes


def _emulate_fp8(value: np.ndarray, fmt: FP8Format) -> np.ndarray:
    """Emulate FP8 by quantizing float32 to FP8 range and precision.

    This emulation:
    1. Clips values to FP8 representable range
    2. Quantizes to FP8 precision (rounds mantissa)

    Args:
        value: Input float32 array
        fmt: FP8 format specification

    Returns:
        Float32 array with values quantized to FP8 precision
    """
    x = value.astype(np.float32)

    # Clip to range
    x = np.clip(x, fmt.min_value, fmt.max_value)

    # Handle zeros
    zero_mask = (x == 0)

    # Quantize to FP8 precision
    # For each value, round to nearest representable FP8 value
    sign = np.sign(x)
    abs_x = np.abs(x)

    # Avoid log of zero
    abs_x = np.maximum(abs_x, fmt.min_subnormal)

    # Get exponent and mantissa
    log2_x = np.log2(abs_x)
    exp = np.floor(log2_x).astype(np.int32)

    # Clamp exponent to valid range
    exp = np.clip(exp, 1 - fmt.bias, (1 << fmt.exp_bits) - 2 - fmt.bias)

    # Compute mantissa and round
    mantissa = abs_x / (2.0 ** exp)  # mantissa in [1, 2)
    mantissa_bits = fmt.mantissa_bits
    mantissa_quant = np.round(mantissa * (2 ** mantissa_bits)) / (2 ** mantissa_bits)

    # Reconstruct value
    result = sign * mantissa_quant * (2.0 ** exp)

    # Restore zeros
    result[zero_mask] = 0.0

    return result.astype(np.float32)


def fp8_matmul(
    a: np.ndarray,
    b: np.ndarray,
    format_name: str = "e4m3",
    output_fp32: bool = True,
) -> np.ndarray:
    """Perform FP8 matrix multiplication.

    Uses ml_dtypes for e4m3/e5m2 if available, otherwise emulates.

    Args:
        a: Input matrix A, shape [..., M, K]
        b: Input matrix B, shape [..., K, N]
        format_name: FP8 format ("e2m5", "e3m4", "e4m3", "e5m2")
        output_fp32: If True, return float32

    Returns:
        Result matrix C, shape [..., M, N]
    """
    fmt = get_fp8_format(format_name)

    # Check for native support
    if format_name in ("e4m3", "e4m3fn") and _FP8_E4M3_DTYPE is not None:
        a_fp8 = a.astype(_FP8_E4M3_DTYPE)
        b_fp8 = b.astype(_FP8_E4M3_DTYPE)
        # Compute in float32 (FP8 accumulation would overflow)
        c = np.matmul(a_fp8.astype(np.float32), b_fp8.astype(np.float32))
        c_fp8 = c.astype(_FP8_E4M3_DTYPE)
        if output_fp32:
            return c_fp8.astype(np.float32)
        return c_fp8
    elif format_name == "e5m2" and _FP8_E5M2_DTYPE is not None:
        a_fp8 = a.astype(_FP8_E5M2_DTYPE)
        b_fp8 = b.astype(_FP8_E5M2_DTYPE)
        c = np.matmul(a_fp8.astype(np.float32), b_fp8.astype(np.float32))
        c_fp8 = c.astype(_FP8_E5M2_DTYPE)
        if output_fp32:
            return c_fp8.astype(np.float32)
        return c_fp8
    else:
        # Emulate
        a_fp8 = _emulate_fp8(a, fmt)
        b_fp8 = _emulate_fp8(b, fmt)
        c = np.matmul(a_fp8, b_fp8)
        return _emulate_fp8(c, fmt)


def fp8_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    format_name: str = "e4m3",
    output_fp32: bool = True,
) -> np.ndarray:
    """Perform FP8 linear layer (y = x @ weight.T + bias).

    Args:
        x: Input tensor, shape [..., in_features]
        weight: Weight matrix, shape [out_features, in_features]
        bias: Optional bias vector, shape [out_features]
        format_name: FP8 format ("e2m5", "e3m4", "e4m3", "e5m2")
        output_fp32: If True, return float32

    Returns:
        Output tensor, shape [..., out_features]
    """
    fmt = get_fp8_format(format_name)

    # Quantize inputs
    if format_name in ("e4m3", "e4m3fn") and _FP8_E4M3_DTYPE is not None:
        x_fp8 = x.astype(_FP8_E4M3_DTYPE).astype(np.float32)
        w_fp8 = weight.astype(_FP8_E4M3_DTYPE).astype(np.float32)
    elif format_name == "e5m2" and _FP8_E5M2_DTYPE is not None:
        x_fp8 = x.astype(_FP8_E5M2_DTYPE).astype(np.float32)
        w_fp8 = weight.astype(_FP8_E5M2_DTYPE).astype(np.float32)
    else:
        x_fp8 = _emulate_fp8(x, fmt)
        w_fp8 = _emulate_fp8(weight, fmt)

    # Linear operation
    y = np.matmul(x_fp8, w_fp8.T)

    if bias is not None:
        # Bias typically kept in higher precision
        y = y + bias.astype(np.float32)

    # Quantize output
    if format_name in ("e4m3", "e4m3fn") and _FP8_E4M3_DTYPE is not None:
        y_fp8 = y.astype(_FP8_E4M3_DTYPE)
        if output_fp32:
            return y_fp8.astype(np.float32)
        return y_fp8
    elif format_name == "e5m2" and _FP8_E5M2_DTYPE is not None:
        y_fp8 = y.astype(_FP8_E5M2_DTYPE)
        if output_fp32:
            return y_fp8.astype(np.float32)
        return y_fp8
    else:
        return _emulate_fp8(y, fmt)


def cast_to_fp8(tensor: np.ndarray, format_name: str = "e4m3") -> np.ndarray:
    """Cast a tensor to FP8.

    Args:
        tensor: Input tensor (any dtype)
        format_name: FP8 format ("e2m5", "e3m4", "e4m3", "e5m2")

    Returns:
        FP8 tensor (ml_dtypes type or emulated float32)
    """
    if format_name in ("e4m3", "e4m3fn") and _FP8_E4M3_DTYPE is not None:
        return tensor.astype(_FP8_E4M3_DTYPE)
    elif format_name == "e5m2" and _FP8_E5M2_DTYPE is not None:
        return tensor.astype(_FP8_E5M2_DTYPE)
    else:
        fmt = get_fp8_format(format_name)
        return _emulate_fp8(tensor.astype(np.float32), fmt)


def cast_from_fp8(tensor: np.ndarray) -> np.ndarray:
    """Cast an FP8 tensor back to FP32.

    Args:
        tensor: Input FP8 tensor

    Returns:
        FP32 tensor
    """
    return tensor.astype(np.float32)


def fp8_range(format_name: str = "e4m3") -> Tuple[float, float]:
    """Return the representable range of an FP8 format.

    Args:
        format_name: FP8 format ("e2m5", "e3m4", "e4m3", "e5m2")

    Returns:
        (min_value, max_value) tuple
    """
    fmt = get_fp8_format(format_name)
    return (fmt.min_value, fmt.max_value)


def fp8_precision(format_name: str = "e4m3") -> float:
    """Return the machine epsilon for an FP8 format.

    Args:
        format_name: FP8 format ("e2m5", "e3m4", "e4m3", "e5m2")

    Returns:
        Machine epsilon
    """
    fmt = get_fp8_format(format_name)
    return fmt.epsilon


def fp8_info(format_name: str = "e4m3") -> Dict[str, Any]:
    """Get detailed information about an FP8 format.

    Args:
        format_name: FP8 format ("e2m5", "e3m4", "e4m3", "e5m2")

    Returns:
        Dictionary with format details
    """
    fmt = get_fp8_format(format_name)
    return {
        "name": fmt.name,
        "exp_bits": fmt.exp_bits,
        "mantissa_bits": fmt.mantissa_bits,
        "bias": fmt.bias,
        "max_value": fmt.max_value,
        "min_value": fmt.min_value,
        "min_normal": fmt.min_normal,
        "min_subnormal": fmt.min_subnormal,
        "epsilon": fmt.epsilon,
        "has_inf": fmt.has_inf,
        "has_nan": fmt.has_nan,
        "native_support": is_fp8_native(format_name),
    }


# --- INT4 operations (v0.7.7) ---

# INT4 specifications:
# - Signed INT4: range [-8, 7], 4 bits
# - Unsigned INT4: range [0, 15], 4 bits
# - Packed storage: 2 values per byte
# - 8x memory reduction vs FP32

INT4_SIGNED_MIN = -8
INT4_SIGNED_MAX = 7
INT4_UNSIGNED_MIN = 0
INT4_UNSIGNED_MAX = 15


def pack_int4(values: np.ndarray, signed: bool = True) -> np.ndarray:
    """Pack INT4 values into bytes (2 values per byte).

    Packing format: low nibble first, then high nibble.
    values[0] -> bits 0-3, values[1] -> bits 4-7

    Args:
        values: INT4 values as int8 array (must have even length in last dim)
        signed: If True, values are signed [-8, 7]; else unsigned [0, 15]

    Returns:
        Packed uint8 array with half the size in last dimension
    """
    values = np.asarray(values)
    original_shape = values.shape

    if original_shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even for packing, got {original_shape[-1]}")

    # Flatten to 1D for packing, then reshape
    flat = values.reshape(-1)

    if signed:
        # Convert signed [-8, 7] to unsigned [0, 15] for packing
        flat = flat.astype(np.int8)
        flat = np.where(flat < 0, flat + 16, flat).astype(np.uint8)
    else:
        flat = flat.astype(np.uint8)

    # Clip to 4-bit range
    flat = flat & 0x0F

    # Pack pairs: low nibble from even indices, high nibble from odd indices
    low = flat[0::2]
    high = flat[1::2]
    packed = (low | (high << 4)).astype(np.uint8)

    # Reshape to original shape with last dim halved
    new_shape = original_shape[:-1] + (original_shape[-1] // 2,)
    return packed.reshape(new_shape)


def unpack_int4(packed: np.ndarray, signed: bool = True) -> np.ndarray:
    """Unpack INT4 values from packed bytes.

    Args:
        packed: Packed uint8 array
        signed: If True, return signed [-8, 7]; else unsigned [0, 15]

    Returns:
        Unpacked values with last dimension doubled
    """
    packed = np.asarray(packed, dtype=np.uint8)
    original_shape = packed.shape

    # Flatten for unpacking
    flat = packed.reshape(-1)

    # Extract low and high nibbles
    low = flat & 0x0F
    high = (flat >> 4) & 0x0F

    # Interleave: [low0, high0, low1, high1, ...]
    unpacked = np.empty(len(flat) * 2, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high

    if signed:
        # Convert unsigned [0, 15] back to signed [-8, 7]
        unpacked = unpacked.astype(np.int8)
        unpacked = np.where(unpacked > 7, unpacked - 16, unpacked)

    # Reshape to original shape with last dim doubled
    new_shape = original_shape[:-1] + (original_shape[-1] * 2,)
    return unpacked.reshape(new_shape)


def quantize_int4(
    tensor: np.ndarray,
    scale: float,
    zero_point: int = 0,
    signed: bool = True,
) -> np.ndarray:
    """Quantize a float tensor to INT4.

    Args:
        tensor: Input float tensor
        scale: Scale factor
        zero_point: Zero point offset (typically 0 for symmetric)
        signed: If True, quantize to [-8, 7]; else [0, 15]

    Returns:
        Quantized int8 tensor (not packed; use pack_int4 for storage)
    """
    qmin = INT4_SIGNED_MIN if signed else INT4_UNSIGNED_MIN
    qmax = INT4_SIGNED_MAX if signed else INT4_UNSIGNED_MAX

    # Quantize
    q = np.round(tensor / scale) + zero_point
    q = np.clip(q, qmin, qmax)

    return q.astype(np.int8)


def dequantize_int4(
    tensor: np.ndarray,
    scale: float,
    zero_point: int = 0,
) -> np.ndarray:
    """Dequantize an INT4 tensor back to float.

    Args:
        tensor: Quantized int tensor (unpacked)
        scale: Scale factor
        zero_point: Zero point offset

    Returns:
        Dequantized float32 tensor
    """
    return (tensor.astype(np.float32) - zero_point) * scale


def compute_int4_scale_zero_point(
    tensor: np.ndarray,
    signed: bool = True,
    symmetric: bool = True,
) -> Tuple[float, int]:
    """Compute scale and zero_point for INT4 quantization.

    Args:
        tensor: Input tensor to analyze
        signed: If True, use signed INT4 [-8, 7]
        symmetric: If True, use symmetric quantization (zero_point=0)

    Returns:
        (scale, zero_point) tuple
    """
    min_val = tensor.min()
    max_val = tensor.max()

    qmin = INT4_SIGNED_MIN if signed else INT4_UNSIGNED_MIN
    qmax = INT4_SIGNED_MAX if signed else INT4_UNSIGNED_MAX

    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        scale = abs_max / max(abs(qmin), qmax)
        zero_point = 0
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            scale = 1.0
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))

    return float(scale), int(zero_point)


def int4_matmul(
    a: np.ndarray,
    b: np.ndarray,
    scale_a: float,
    zero_point_a: int,
    scale_b: float,
    zero_point_b: int,
    output_float: bool = True,
) -> np.ndarray:
    """Perform INT4 quantized matrix multiplication.

    Inputs should be unpacked INT4 values (as int8).
    Computation is done by dequantizing to float32.

    Args:
        a: Quantized input A (int8, unpacked INT4), shape [..., M, K]
        b: Quantized input B (int8, unpacked INT4), shape [..., K, N]
        scale_a, zero_point_a: Quantization params for A
        scale_b, zero_point_b: Quantization params for B
        output_float: If True, return float32

    Returns:
        Result matrix (float32)
    """
    # Dequantize inputs
    a_fp = (a.astype(np.int32) - zero_point_a) * scale_a
    b_fp = (b.astype(np.int32) - zero_point_b) * scale_b

    # Perform matmul in float32
    c_fp = np.matmul(a_fp, b_fp)

    return c_fp.astype(np.float32)


def int4_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    scale_x: float,
    zero_point_x: int,
    scale_w: float,
    zero_point_w: int,
    output_float: bool = True,
) -> np.ndarray:
    """Perform INT4 quantized linear layer.

    Args:
        x: Quantized input (int8, unpacked INT4), shape [..., in_features]
        weight: Quantized weights (int8, unpacked INT4), shape [out_features, in_features]
        bias: Optional float32 bias, shape [out_features]
        scale_x, zero_point_x: Input quantization params
        scale_w, zero_point_w: Weight quantization params
        output_float: If True, return float32

    Returns:
        Result tensor (float32)
    """
    # Dequantize
    x_fp = (x.astype(np.int32) - zero_point_x) * scale_x
    w_fp = (weight.astype(np.int32) - zero_point_w) * scale_w

    # Linear operation: y = x @ w.T + bias
    y_fp = np.matmul(x_fp, w_fp.T)

    if bias is not None:
        y_fp = y_fp + bias

    return y_fp.astype(np.float32)


def int4_packed_size(num_elements: int) -> int:
    """Calculate packed storage size for INT4 values.

    Args:
        num_elements: Number of INT4 values

    Returns:
        Number of bytes needed for packed storage
    """
    return (num_elements + 1) // 2  # Ceiling division


def int4_memory_bytes(shape: Tuple[int, ...]) -> int:
    """Calculate memory bytes for packed INT4 tensor.

    Args:
        shape: Tensor shape

    Returns:
        Total bytes (packed storage)
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return int4_packed_size(num_elements)


def int4_info() -> Dict[str, Any]:
    """Get information about INT4 format.

    Returns:
        Dictionary with INT4 details
    """
    return {
        "signed_range": (INT4_SIGNED_MIN, INT4_SIGNED_MAX),
        "unsigned_range": (INT4_UNSIGNED_MIN, INT4_UNSIGNED_MAX),
        "bits": 4,
        "bytes_per_element": 0.5,
        "packing": "2 values per byte",
        "memory_reduction_vs_fp32": 8.0,
        "memory_reduction_vs_int8": 2.0,
    }


# --- FP4 operations (v0.7.8) ---

# FP4 format specifications:
# Common FP4 formats (4 bits = 1 sign + exponent + mantissa):
# - E2M1: 2-bit exponent, 1-bit mantissa (wider range, less precision)
# - E1M2: 1-bit exponent, 2-bit mantissa (narrow range, more precision)
#
# FP4 E2M1 value table (most common format):
# | Bits | Value    |
# |------|----------|
# | 0000 | +0       |
# | 0001 | +0.5     |
# | 0010 | +1.0     |
# | 0011 | +1.5     |
# | 0100 | +2.0     |
# | 0101 | +3.0     |
# | 0110 | +4.0     |
# | 0111 | +6.0     |
# | 1000 | -0       |
# | 1001 | -0.5     |
# | 1010 | -1.0     |
# | 1011 | -1.5     |
# | 1100 | -2.0     |
# | 1101 | -3.0     |
# | 1110 | -4.0     |
# | 1111 | -6.0     |

class FP4Format:
    """FP4 format specification."""

    def __init__(self, name: str, exp_bits: int, mantissa_bits: int, bias: int):
        """Initialize FP4 format.

        Args:
            name: Format name (e.g., "e2m1")
            exp_bits: Number of exponent bits
            mantissa_bits: Number of mantissa bits
            bias: Exponent bias
        """
        self.name = name
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias

        # Build value lookup table for this format
        self._build_value_table()

    def _build_value_table(self):
        """Build lookup table mapping 4-bit codes to float values."""
        self.code_to_value = {}
        self.value_to_code = {}

        for code in range(16):
            sign = -1 if (code & 0x8) else 1
            unsigned_code = code & 0x7  # Lower 3 bits

            if self.exp_bits == 2 and self.mantissa_bits == 1:
                # E2M1 format
                exp_field = (unsigned_code >> 1) & 0x3
                mant_field = unsigned_code & 0x1

                if exp_field == 0:
                    # Subnormal
                    value = sign * mant_field * 0.5
                else:
                    # Normal
                    mantissa = 1.0 + mant_field * 0.5
                    value = sign * mantissa * (2 ** (exp_field - self.bias))

            elif self.exp_bits == 1 and self.mantissa_bits == 2:
                # E1M2 format
                exp_field = (unsigned_code >> 2) & 0x1
                mant_field = unsigned_code & 0x3

                if exp_field == 0:
                    # Subnormal
                    value = sign * mant_field * 0.25
                else:
                    # Normal
                    mantissa = 1.0 + mant_field * 0.25
                    value = sign * mantissa * (2 ** (exp_field - self.bias))
            else:
                # Generic fallback
                value = sign * unsigned_code * 0.5

            self.code_to_value[code] = float(value)

        # Build reverse mapping (value -> code), handling duplicates
        # Sort by absolute value for nearest-neighbor quantization
        sorted_codes = sorted(self.code_to_value.items(),
                             key=lambda x: (abs(x[1]), x[0]))
        self._sorted_values = [(v, c) for c, v in sorted_codes]

    @property
    def max_value(self) -> float:
        return max(self.code_to_value.values())

    @property
    def min_value(self) -> float:
        return min(self.code_to_value.values())

    @property
    def unique_values(self) -> list:
        """Return sorted list of unique representable values."""
        return sorted(set(self.code_to_value.values()))

    def quantize_value(self, x: float) -> int:
        """Quantize a single float value to nearest FP4 code."""
        best_code = 0
        best_dist = float('inf')
        for value, code in self._sorted_values:
            dist = abs(x - value)
            if dist < best_dist:
                best_dist = dist
                best_code = code
        return best_code

    def dequantize_code(self, code: int) -> float:
        """Convert FP4 code to float value."""
        return self.code_to_value.get(code & 0xF, 0.0)


# Standard FP4 formats
FP4_E2M1 = FP4Format("e2m1", exp_bits=2, mantissa_bits=1, bias=1)
FP4_E1M2 = FP4Format("e1m2", exp_bits=1, mantissa_bits=2, bias=0)

# Format lookup
_FP4_FORMATS = {
    "e2m1": FP4_E2M1,
    "e1m2": FP4_E1M2,
}


def get_fp4_format(format_name: str = "e2m1") -> FP4Format:
    """Get FP4 format specification by name.

    Args:
        format_name: Format name ("e2m1", "e1m2")

    Returns:
        FP4Format specification
    """
    if format_name not in _FP4_FORMATS:
        raise ValueError(f"Unknown FP4 format: {format_name}. "
                        f"Available: {list(_FP4_FORMATS.keys())}")
    return _FP4_FORMATS[format_name]


def fp4_quantize(
    tensor: np.ndarray,
    format_name: str = "e2m1",
    scale: Optional[float] = None,
) -> np.ndarray:
    """Quantize float tensor to FP4 codes.

    Args:
        tensor: Input float tensor
        format_name: FP4 format ("e2m1", "e1m2")
        scale: Optional scale factor (if None, auto-computed)

    Returns:
        Quantized tensor as uint8 (4-bit codes, unpacked)
    """
    fmt = get_fp4_format(format_name)
    tensor = np.asarray(tensor, dtype=np.float32)

    # Auto-compute scale if not provided
    if scale is None:
        abs_max = np.abs(tensor).max()
        if abs_max > 0:
            scale = abs_max / fmt.max_value
        else:
            scale = 1.0

    # Scale input
    scaled = tensor / scale

    # Vectorized quantization using lookup
    flat = scaled.reshape(-1)
    codes = np.zeros(len(flat), dtype=np.uint8)
    for i, val in enumerate(flat):
        codes[i] = fmt.quantize_value(val)

    return codes.reshape(tensor.shape), scale


def fp4_dequantize(
    codes: np.ndarray,
    scale: float,
    format_name: str = "e2m1",
) -> np.ndarray:
    """Dequantize FP4 codes to float tensor.

    Args:
        codes: Quantized tensor (uint8, 4-bit codes)
        scale: Scale factor used during quantization
        format_name: FP4 format ("e2m1", "e1m2")

    Returns:
        Dequantized float32 tensor
    """
    fmt = get_fp4_format(format_name)
    codes = np.asarray(codes, dtype=np.uint8)

    # Vectorized dequantization
    flat = codes.reshape(-1)
    values = np.array([fmt.dequantize_code(c) for c in flat], dtype=np.float32)

    return (values * scale).reshape(codes.shape)


def pack_fp4(codes: np.ndarray) -> np.ndarray:
    """Pack FP4 codes into bytes (2 values per byte).

    Args:
        codes: FP4 codes as uint8 array (must have even length in last dim)

    Returns:
        Packed uint8 array with half the size in last dimension
    """
    codes = np.asarray(codes, dtype=np.uint8)
    original_shape = codes.shape

    if original_shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even for packing, got {original_shape[-1]}")

    flat = codes.reshape(-1)
    flat = flat & 0x0F  # Ensure 4-bit values

    low = flat[0::2]
    high = flat[1::2]
    packed = (low | (high << 4)).astype(np.uint8)

    new_shape = original_shape[:-1] + (original_shape[-1] // 2,)
    return packed.reshape(new_shape)


def unpack_fp4(packed: np.ndarray) -> np.ndarray:
    """Unpack FP4 codes from packed bytes.

    Args:
        packed: Packed uint8 array

    Returns:
        Unpacked codes with last dimension doubled
    """
    packed = np.asarray(packed, dtype=np.uint8)
    original_shape = packed.shape

    flat = packed.reshape(-1)
    low = flat & 0x0F
    high = (flat >> 4) & 0x0F

    unpacked = np.empty(len(flat) * 2, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high

    new_shape = original_shape[:-1] + (original_shape[-1] * 2,)
    return unpacked.reshape(new_shape)


def fp4_matmul(
    a: np.ndarray,
    b: np.ndarray,
    scale_a: float,
    scale_b: float,
    format_name: str = "e2m1",
) -> np.ndarray:
    """Perform FP4 matrix multiplication.

    Inputs should be FP4 codes (unpacked).

    Args:
        a: FP4 codes for matrix A, shape [..., M, K]
        b: FP4 codes for matrix B, shape [..., K, N]
        scale_a: Scale for A
        scale_b: Scale for B
        format_name: FP4 format

    Returns:
        Result matrix (float32)
    """
    # Dequantize inputs
    a_fp = fp4_dequantize(a, scale_a, format_name)
    b_fp = fp4_dequantize(b, scale_b, format_name)

    # Matmul in float32
    return np.matmul(a_fp, b_fp)


def fp4_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    scale_x: float,
    scale_w: float,
    format_name: str = "e2m1",
) -> np.ndarray:
    """Perform FP4 linear layer.

    Args:
        x: FP4 codes for input, shape [..., in_features]
        weight: FP4 codes for weights, shape [out_features, in_features]
        bias: Optional float32 bias
        scale_x: Scale for input
        scale_w: Scale for weights
        format_name: FP4 format

    Returns:
        Result tensor (float32)
    """
    # Dequantize
    x_fp = fp4_dequantize(x, scale_x, format_name)
    w_fp = fp4_dequantize(weight, scale_w, format_name)

    # Linear
    y = np.matmul(x_fp, w_fp.T)
    if bias is not None:
        y = y + bias

    return y


def fp4_range(format_name: str = "e2m1") -> Tuple[float, float]:
    """Return the representable range of an FP4 format.

    Args:
        format_name: FP4 format ("e2m1", "e1m2")

    Returns:
        (min_value, max_value) tuple
    """
    fmt = get_fp4_format(format_name)
    return (fmt.min_value, fmt.max_value)


def fp4_values(format_name: str = "e2m1") -> list:
    """Return all unique representable values for an FP4 format.

    Args:
        format_name: FP4 format ("e2m1", "e1m2")

    Returns:
        Sorted list of all representable values
    """
    fmt = get_fp4_format(format_name)
    return fmt.unique_values


def fp4_info(format_name: str = "e2m1") -> Dict[str, Any]:
    """Get detailed information about an FP4 format.

    Args:
        format_name: FP4 format ("e2m1", "e1m2")

    Returns:
        Dictionary with format details
    """
    fmt = get_fp4_format(format_name)
    return {
        "name": fmt.name,
        "exp_bits": fmt.exp_bits,
        "mantissa_bits": fmt.mantissa_bits,
        "bias": fmt.bias,
        "max_value": fmt.max_value,
        "min_value": fmt.min_value,
        "unique_values": fmt.unique_values,
        "num_values": len(set(fmt.code_to_value.values())),
        "bits": 4,
        "bytes_per_element": 0.5,
        "memory_reduction_vs_fp32": 8.0,
    }


# --- Mixed Precision operations (v0.7.9) ---

# Mixed precision configurations for efficient inference:
# - INT8 weights + FP16 activations: Good accuracy, 4x weight compression
# - INT8 weights + BF16 activations: Good for training, wider range
# - INT4 weights + FP16 activations: Aggressive compression, LLM inference
# - FP8 weights + FP16 activations: Balance of range and precision

class MixedPrecisionConfig:
    """Configuration for mixed precision computation.

    Attributes:
        weight_dtype: Data type for weights
        activation_dtype: Data type for activations
        accumulator_dtype: Data type for accumulation (usually FP32)
        output_dtype: Data type for output
    """

    def __init__(
        self,
        weight_dtype: QuantDtype = QuantDtype.INT8,
        activation_dtype: QuantDtype = QuantDtype.FP16,
        accumulator_dtype: QuantDtype = QuantDtype.FP32,
        output_dtype: Optional[QuantDtype] = None,
    ):
        self.weight_dtype = weight_dtype
        self.activation_dtype = activation_dtype
        self.accumulator_dtype = accumulator_dtype
        self.output_dtype = output_dtype or activation_dtype

    def __repr__(self) -> str:
        return (f"MixedPrecisionConfig(weights={self.weight_dtype.value}, "
                f"activations={self.activation_dtype.value}, "
                f"accumulator={self.accumulator_dtype.value})")

    @property
    def weight_bytes(self) -> float:
        """Bytes per weight element."""
        return self.weight_dtype.bytes_per_element

    @property
    def activation_bytes(self) -> float:
        """Bytes per activation element."""
        return self.activation_dtype.bytes_per_element

    def memory_reduction_factor(self) -> float:
        """Calculate memory reduction vs FP32 for typical inference.

        Assumes weights dominate memory (typical for large models).
        """
        fp32_bytes = QuantDtype.FP32.bytes_per_element
        # Weighted average: weights usually 80%+ of memory
        avg_bytes = 0.8 * self.weight_bytes + 0.2 * self.activation_bytes
        return fp32_bytes / avg_bytes


# Common mixed precision configurations
MIXED_INT8_FP16 = MixedPrecisionConfig(
    weight_dtype=QuantDtype.INT8,
    activation_dtype=QuantDtype.FP16,
)

MIXED_INT8_BF16 = MixedPrecisionConfig(
    weight_dtype=QuantDtype.INT8,
    activation_dtype=QuantDtype.BF16,
)

MIXED_INT4_FP16 = MixedPrecisionConfig(
    weight_dtype=QuantDtype.INT4,
    activation_dtype=QuantDtype.FP16,
)

MIXED_FP8_FP16 = MixedPrecisionConfig(
    weight_dtype=QuantDtype.FP8_E4M3,
    activation_dtype=QuantDtype.FP16,
)

MIXED_FP8_BF16 = MixedPrecisionConfig(
    weight_dtype=QuantDtype.FP8_E4M3,
    activation_dtype=QuantDtype.BF16,
)


def mixed_precision_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    weight_scale: float = 1.0,
    weight_zero_point: int = 0,
    config: MixedPrecisionConfig = MIXED_INT8_FP16,
) -> np.ndarray:
    """Perform mixed precision linear layer.

    Weights are stored in low precision (e.g., INT8) and dequantized.
    Activations are computed in higher precision (e.g., FP16).
    Accumulation is done in FP32 for accuracy.

    Args:
        x: Input activations (float32, will be cast to activation_dtype)
        weight: Quantized weights (int8 or appropriate type)
        bias: Optional bias (float32)
        weight_scale: Scale factor for weight dequantization
        weight_zero_point: Zero point for weight dequantization
        config: Mixed precision configuration

    Returns:
        Output tensor (float32)
    """
    # Cast activations to target dtype
    if config.activation_dtype == QuantDtype.FP16:
        x_act = x.astype(np.float16)
    elif config.activation_dtype == QuantDtype.BF16:
        if _ML_DTYPES_AVAILABLE:
            x_act = x.astype(_BFLOAT16_DTYPE)
        else:
            x_act = _emulate_bf16(x.astype(np.float32))
    else:
        x_act = x.astype(np.float32)

    # Dequantize weights
    if config.weight_dtype in (QuantDtype.INT8, QuantDtype.UINT8):
        w_fp = (weight.astype(np.int32) - weight_zero_point) * weight_scale
    elif config.weight_dtype == QuantDtype.INT4:
        w_fp = (weight.astype(np.int32) - weight_zero_point) * weight_scale
    elif config.weight_dtype in (QuantDtype.FP8_E4M3, QuantDtype.FP8_E5M2):
        # FP8 weights - just cast to float
        w_fp = weight.astype(np.float32) * weight_scale
    else:
        w_fp = weight.astype(np.float32)

    # Accumulate in FP32
    x_fp32 = x_act.astype(np.float32) if hasattr(x_act, 'astype') else np.array(x_act, dtype=np.float32)
    y = np.matmul(x_fp32, w_fp.astype(np.float32).T)

    if bias is not None:
        y = y + bias.astype(np.float32)

    return y.astype(np.float32)


def mixed_precision_matmul(
    a: np.ndarray,
    b: np.ndarray,
    a_scale: float = 1.0,
    a_zero_point: int = 0,
    b_scale: float = 1.0,
    b_zero_point: int = 0,
    a_dtype: QuantDtype = QuantDtype.FP16,
    b_dtype: QuantDtype = QuantDtype.INT8,
) -> np.ndarray:
    """Perform mixed precision matrix multiplication.

    Args:
        a: First matrix (activations, typically FP16)
        b: Second matrix (weights, typically INT8)
        a_scale, a_zero_point: Quantization params for a (if quantized)
        b_scale, b_zero_point: Quantization params for b (if quantized)
        a_dtype: Data type for a
        b_dtype: Data type for b

    Returns:
        Result matrix (float32)
    """
    # Dequantize/cast a
    if a_dtype.is_integer:
        a_fp = (a.astype(np.int32) - a_zero_point) * a_scale
    elif a_dtype == QuantDtype.FP16:
        a_fp = a.astype(np.float16).astype(np.float32)
    elif a_dtype == QuantDtype.BF16:
        a_fp = a.astype(np.float32)  # Already dequantized or emulated
    else:
        a_fp = a.astype(np.float32)

    # Dequantize/cast b
    if b_dtype.is_integer:
        b_fp = (b.astype(np.int32) - b_zero_point) * b_scale
    elif b_dtype == QuantDtype.FP16:
        b_fp = b.astype(np.float16).astype(np.float32)
    elif b_dtype == QuantDtype.BF16:
        b_fp = b.astype(np.float32)
    else:
        b_fp = b.astype(np.float32)

    # Matmul in FP32
    return np.matmul(a_fp, b_fp)


def mixed_precision_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    weight_scale: float = 1.0,
    weight_zero_point: int = 0,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    config: MixedPrecisionConfig = MIXED_INT8_FP16,
) -> np.ndarray:
    """Perform mixed precision 2D convolution.

    Args:
        x: Input tensor [N, C_in, H, W] (will be cast to activation_dtype)
        weight: Quantized weights [C_out, C_in, kH, kW]
        bias: Optional bias [C_out]
        weight_scale: Scale for weight dequantization
        weight_zero_point: Zero point for weight dequantization
        stride: Convolution stride
        padding: Convolution padding
        config: Mixed precision configuration

    Returns:
        Output tensor [N, C_out, H_out, W_out] (float32)
    """
    # Cast activations
    if config.activation_dtype == QuantDtype.FP16:
        x_act = x.astype(np.float16)
    else:
        x_act = x.astype(np.float32)

    # Dequantize weights
    if config.weight_dtype.is_integer:
        w_fp = (weight.astype(np.int32) - weight_zero_point) * weight_scale
    else:
        w_fp = weight.astype(np.float32) * weight_scale

    # Convert to FP32 for computation
    x_fp32 = x_act.astype(np.float32)
    w_fp32 = w_fp.astype(np.float32)

    N, C_in, H, W = x_fp32.shape
    C_out, _, kH, kW = w_fp32.shape
    sH, sW = stride
    pH, pW = padding

    # Pad input
    if pH > 0 or pW > 0:
        x_fp32 = np.pad(x_fp32, ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                        mode='constant', constant_values=0)

    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1

    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * sH
                    w_start = w * sW
                    patch = x_fp32[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    output[n, c_out, h, w] = np.sum(patch * w_fp32[c_out])

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    return output


def calculate_mixed_precision_traffic(
    M: int, K: int, N: int,
    config: MixedPrecisionConfig,
) -> Dict[str, int]:
    """Calculate memory traffic for mixed precision matmul.

    Args:
        M, K, N: Matrix dimensions (A: MxK, B: KxN, C: MxN)
        config: Mixed precision configuration

    Returns:
        Dictionary with traffic breakdown
    """
    # A is activations (M x K)
    a_bytes = int(M * K * config.activation_bytes)
    # B is weights (K x N)
    b_bytes = int(K * N * config.weight_bytes)
    # C is output (M x N) in output dtype
    c_bytes = int(M * N * config.output_dtype.bytes_per_element)

    total = a_bytes + b_bytes + c_bytes

    # Compare to FP32
    fp32_total = (M * K + K * N + M * N) * 4

    return {
        "activations_bytes": a_bytes,
        "weights_bytes": b_bytes,
        "output_bytes": c_bytes,
        "total_bytes": total,
        "fp32_baseline_bytes": fp32_total,
        "reduction_factor": fp32_total / total if total > 0 else 1.0,
    }


def mixed_precision_info(config: MixedPrecisionConfig) -> Dict[str, Any]:
    """Get information about a mixed precision configuration.

    Args:
        config: Mixed precision configuration

    Returns:
        Dictionary with configuration details
    """
    return {
        "weight_dtype": config.weight_dtype.value,
        "activation_dtype": config.activation_dtype.value,
        "accumulator_dtype": config.accumulator_dtype.value,
        "output_dtype": config.output_dtype.value,
        "weight_bytes": config.weight_bytes,
        "activation_bytes": config.activation_bytes,
        "memory_reduction_factor": config.memory_reduction_factor(),
    }


# --- Q/DQ Operations (v0.7.10) ---

# Q/DQ (Quantize/Dequantize) operations provide explicit graph-level
# quantization nodes. This pattern is used for:
# 1. Quantization-aware training (QAT) - fake quantization
# 2. ONNX-style quantization representation
# 3. Fine-grained control over quantization points
#
# The Q/DQ pattern:
#   input -> Q -> DQ -> operation -> Q -> DQ -> output
#
# During training: Q/DQ simulate quantization error (fake quant)
# During inference: Q/DQ pairs can be folded into the operation

@dataclass
class QDQParams:
    """Parameters for quantize/dequantize operations.

    Attributes:
        scale: Scale factor for quantization
        zero_point: Zero point offset
        dtype: Target quantization dtype
        axis: Axis for per-channel quantization (None for per-tensor)
        quant_min: Minimum quantized value (for clipping)
        quant_max: Maximum quantized value (for clipping)
    """
    scale: Union[float, np.ndarray]
    zero_point: Union[int, np.ndarray] = 0
    dtype: QuantDtype = QuantDtype.INT8
    axis: Optional[int] = None
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None

    def __post_init__(self):
        if self.quant_min is None:
            self.quant_min = self.dtype.qmin
        if self.quant_max is None:
            self.quant_max = self.dtype.qmax

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scale": float(self.scale) if np.isscalar(self.scale) else self.scale.tolist(),
            "zero_point": int(self.zero_point) if np.isscalar(self.zero_point) else self.zero_point.tolist(),
            "dtype": self.dtype.value,
            "axis": self.axis,
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QDQParams':
        """Create from dictionary."""
        scale = d["scale"]
        if isinstance(scale, list):
            scale = np.array(scale, dtype=np.float32)
        zero_point = d["zero_point"]
        if isinstance(zero_point, list):
            zero_point = np.array(zero_point, dtype=np.int32)
        return cls(
            scale=scale,
            zero_point=zero_point,
            dtype=QuantDtype(d["dtype"]),
            axis=d.get("axis"),
            quant_min=d.get("quant_min"),
            quant_max=d.get("quant_max"),
        )


def Q(
    x: np.ndarray,
    params: QDQParams,
) -> np.ndarray:
    """Quantize operation (Q node).

    Quantizes input tensor according to parameters.

    Args:
        x: Input float tensor
        params: Quantization parameters

    Returns:
        Quantized integer tensor
    """
    if params.axis is not None:
        # Per-channel quantization
        scale = np.array(params.scale)
        zero_point = np.array(params.zero_point)

        # Reshape for broadcasting
        shape = [1] * x.ndim
        shape[params.axis] = -1
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)

        q = np.round(x / scale) + zero_point
    else:
        # Per-tensor quantization
        q = np.round(x / params.scale) + params.zero_point

    # Clip to valid range
    q = np.clip(q, params.quant_min, params.quant_max)

    # Return appropriate integer type
    if params.dtype in (QuantDtype.INT8,):
        return q.astype(np.int8)
    elif params.dtype in (QuantDtype.UINT8,):
        return q.astype(np.uint8)
    elif params.dtype in (QuantDtype.INT4, QuantDtype.UINT4):
        return q.astype(np.int8)  # Store as int8, pack separately
    else:
        return q.astype(np.int32)


def DQ(
    x: np.ndarray,
    params: QDQParams,
) -> np.ndarray:
    """Dequantize operation (DQ node).

    Dequantizes integer tensor back to float.

    Args:
        x: Quantized integer tensor
        params: Quantization parameters

    Returns:
        Dequantized float32 tensor
    """
    if params.axis is not None:
        # Per-channel dequantization
        scale = np.array(params.scale)
        zero_point = np.array(params.zero_point)

        # Reshape for broadcasting
        shape = [1] * x.ndim
        shape[params.axis] = -1
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)

        return (x.astype(np.float32) - zero_point) * scale
    else:
        # Per-tensor dequantization
        return (x.astype(np.float32) - params.zero_point) * params.scale


def fake_quantize(
    x: np.ndarray,
    params: QDQParams,
) -> np.ndarray:
    """Fake quantization (Q followed by DQ).

    Simulates quantization error while keeping values in float.
    Used for quantization-aware training.

    Args:
        x: Input float tensor
        params: Quantization parameters

    Returns:
        Float tensor with quantization error simulated
    """
    return DQ(Q(x, params), params)


def qdq_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    x_params: QDQParams,
    w_params: QDQParams,
    output_params: Optional[QDQParams] = None,
) -> np.ndarray:
    """Linear layer with Q/DQ operations.

    Applies fake quantization to inputs and weights, performs linear
    operation, and optionally quantizes output.

    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Optional bias
        x_params: Quantization params for input
        w_params: Quantization params for weights
        output_params: Optional params for output quantization

    Returns:
        Output tensor (quantized if output_params provided)
    """
    # Fake quantize inputs
    x_fq = fake_quantize(x, x_params)
    w_fq = fake_quantize(weight, w_params)

    # Linear operation in float
    y = np.matmul(x_fq, w_fq.T)
    if bias is not None:
        y = y + bias

    # Optionally quantize output
    if output_params is not None:
        y = fake_quantize(y, output_params)

    return y


def qdq_matmul(
    a: np.ndarray,
    b: np.ndarray,
    a_params: QDQParams,
    b_params: QDQParams,
    output_params: Optional[QDQParams] = None,
) -> np.ndarray:
    """Matrix multiplication with Q/DQ operations.

    Args:
        a: First matrix
        b: Second matrix
        a_params: Quantization params for a
        b_params: Quantization params for b
        output_params: Optional params for output quantization

    Returns:
        Result matrix
    """
    a_fq = fake_quantize(a, a_params)
    b_fq = fake_quantize(b, b_params)

    c = np.matmul(a_fq, b_fq)

    if output_params is not None:
        c = fake_quantize(c, output_params)

    return c


def qdq_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    x_params: QDQParams,
    w_params: QDQParams,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    output_params: Optional[QDQParams] = None,
) -> np.ndarray:
    """2D convolution with Q/DQ operations.

    Args:
        x: Input tensor [N, C_in, H, W]
        weight: Filter weights [C_out, C_in, kH, kW]
        bias: Optional bias [C_out]
        x_params: Quantization params for input
        w_params: Quantization params for weights
        stride: Convolution stride
        padding: Convolution padding
        output_params: Optional params for output quantization

    Returns:
        Output tensor
    """
    # Fake quantize inputs
    x_fq = fake_quantize(x, x_params)
    w_fq = fake_quantize(weight, w_params)

    N, C_in, H, W = x_fq.shape
    C_out, _, kH, kW = w_fq.shape
    sH, sW = stride
    pH, pW = padding

    # Pad input
    if pH > 0 or pW > 0:
        x_fq = np.pad(x_fq, ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                      mode='constant', constant_values=0)

    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1

    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * sH
                    w_start = w * sW
                    patch = x_fq[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    output[n, c_out, h, w] = np.sum(patch * w_fq[c_out])

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    if output_params is not None:
        output = fake_quantize(output, output_params)

    return output


def create_qdq_params(
    tensor: np.ndarray,
    dtype: QuantDtype = QuantDtype.INT8,
    symmetric: bool = True,
    per_channel: bool = False,
    axis: int = 0,
) -> QDQParams:
    """Create Q/DQ parameters from a tensor (calibration).

    Args:
        tensor: Tensor to calibrate from
        dtype: Target quantization dtype
        symmetric: Use symmetric quantization
        per_channel: Use per-channel quantization
        axis: Channel axis (if per_channel)

    Returns:
        QDQParams for the tensor
    """
    if per_channel:
        scales, zero_points = compute_per_channel_params(
            tensor, axis=axis, dtype=dtype, symmetric=symmetric
        )
        return QDQParams(
            scale=scales,
            zero_point=zero_points,
            dtype=dtype,
            axis=axis,
        )
    else:
        scale, zero_point = compute_scale_zero_point(
            tensor, dtype=dtype, symmetric=symmetric
        )
        return QDQParams(
            scale=scale,
            zero_point=zero_point,
            dtype=dtype,
        )


def quantize_error(
    x: np.ndarray,
    params: QDQParams,
) -> Dict[str, float]:
    """Compute quantization error metrics.

    Args:
        x: Original float tensor
        params: Quantization parameters

    Returns:
        Dictionary with error metrics
    """
    x_fq = fake_quantize(x, params)

    abs_error = np.abs(x - x_fq)
    rel_error = abs_error / (np.abs(x) + 1e-10)

    return {
        "mean_abs_error": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "mean_rel_error": float(np.mean(rel_error)),
        "max_rel_error": float(np.max(rel_error)),
        "snr_db": float(10 * np.log10(np.mean(x**2) / (np.mean(abs_error**2) + 1e-10))),
    }


# --- Calibration (v0.7.11) ---

# Calibration is the process of determining optimal quantization parameters
# (scale and zero_point) by collecting statistics from representative data.
#
# Supported methods:
# - MinMax: Uses observed min/max values (simple, fast)
# - Percentile: Uses percentile values to handle outliers
# - MSE: Minimizes mean squared error between original and quantized
# - Entropy: Minimizes KL divergence (information loss)
#
# Workflow:
# 1. Create a CalibrationObserver for each tensor to calibrate
# 2. Run representative data through the model, calling observe() on each tensor
# 3. Call compute_params() to get the optimal QDQParams


class CalibrationMethod(Enum):
    """Calibration methods for determining quantization parameters."""
    MINMAX = "minmax"           # Simple min/max (fast but sensitive to outliers)
    PERCENTILE = "percentile"   # Percentile-based (handles outliers)
    MSE = "mse"                 # Minimize mean squared error
    ENTROPY = "entropy"         # Minimize KL divergence


@dataclass
class CalibrationStats:
    """Statistics collected during calibration.

    Attributes:
        min_val: Minimum observed value
        max_val: Maximum observed value
        num_samples: Number of samples observed
        histogram: Histogram of values (optional, for entropy/mse)
        bin_edges: Histogram bin edges
    """
    min_val: float = float('inf')
    max_val: float = float('-inf')
    num_samples: int = 0
    histogram: Optional[np.ndarray] = None
    bin_edges: Optional[np.ndarray] = None

    def update(self, tensor: np.ndarray) -> None:
        """Update statistics with new tensor values."""
        self.min_val = min(self.min_val, float(np.min(tensor)))
        self.max_val = max(self.max_val, float(np.max(tensor)))
        self.num_samples += tensor.size

    def reset(self) -> None:
        """Reset all statistics."""
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.num_samples = 0
        self.histogram = None
        self.bin_edges = None


class CalibrationObserver:
    """Observer that collects statistics for quantization calibration.

    Example:
        >>> observer = CalibrationObserver(method=CalibrationMethod.PERCENTILE)
        >>> for batch in calibration_data:
        ...     activations = model.forward(batch)
        ...     observer.observe(activations)
        >>> params = observer.compute_params()
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.MINMAX,
        dtype: QuantDtype = QuantDtype.INT8,
        symmetric: bool = True,
        percentile: float = 99.99,
        num_bins: int = 2048,
        per_channel: bool = False,
        channel_axis: int = 0,
    ):
        """Initialize calibration observer.

        Args:
            method: Calibration method to use
            dtype: Target quantization dtype
            symmetric: Use symmetric quantization (zero_point=0)
            percentile: Percentile value for PERCENTILE method (default 99.99)
            num_bins: Number of histogram bins for entropy/mse methods
            per_channel: Collect per-channel statistics
            channel_axis: Axis for per-channel calibration
        """
        self.method = method
        self.dtype = dtype
        self.symmetric = symmetric
        self.percentile = percentile
        self.num_bins = num_bins
        self.per_channel = per_channel
        self.channel_axis = channel_axis

        # Per-tensor statistics
        self.stats = CalibrationStats()

        # Per-channel statistics (list of CalibrationStats)
        self._channel_stats: Optional[list] = None

        # For histogram-based methods
        self._all_values: list = []  # Collect values for histogram building
        self._histogram_built = False

    def observe(self, tensor: np.ndarray) -> None:
        """Observe a tensor and update calibration statistics.

        Args:
            tensor: Tensor to observe (numpy array)
        """
        tensor = np.asarray(tensor, dtype=np.float32)

        if self.per_channel:
            num_channels = tensor.shape[self.channel_axis]
            if self._channel_stats is None:
                self._channel_stats = [CalibrationStats() for _ in range(num_channels)]

            # Update per-channel stats
            for c in range(num_channels):
                # Extract channel slice
                slices = [slice(None)] * tensor.ndim
                slices[self.channel_axis] = c
                channel_data = tensor[tuple(slices)]
                self._channel_stats[c].update(channel_data)

                # Collect values for histogram methods
                if self.method in (CalibrationMethod.MSE, CalibrationMethod.ENTROPY):
                    # Store summarized values (subsample for memory efficiency)
                    if channel_data.size > 10000:
                        indices = np.random.choice(channel_data.size, 10000, replace=False)
                        self._all_values.append((c, channel_data.flatten()[indices]))
                    else:
                        self._all_values.append((c, channel_data.flatten()))
        else:
            # Update global stats
            self.stats.update(tensor)

            # Collect values for histogram methods
            if self.method in (CalibrationMethod.MSE, CalibrationMethod.ENTROPY):
                flat = tensor.flatten()
                # Subsample for memory efficiency
                if flat.size > 10000:
                    indices = np.random.choice(flat.size, 10000, replace=False)
                    self._all_values.append(flat[indices])
                else:
                    self._all_values.append(flat)

    def _build_histogram(self) -> None:
        """Build histogram from collected values."""
        if self._histogram_built or not self._all_values:
            return

        if self.per_channel and self._channel_stats is not None:
            # Build per-channel histograms
            for c, stats in enumerate(self._channel_stats):
                channel_values = [v for ch, v in self._all_values if ch == c]
                if channel_values:
                    all_data = np.concatenate(channel_values)
                    stats.histogram, stats.bin_edges = np.histogram(
                        all_data, bins=self.num_bins
                    )
        else:
            # Build global histogram
            all_data = np.concatenate(self._all_values)
            self.stats.histogram, self.stats.bin_edges = np.histogram(
                all_data, bins=self.num_bins
            )

        self._histogram_built = True
        self._all_values = []  # Free memory

    def compute_params(self) -> QDQParams:
        """Compute quantization parameters from collected statistics.

        Returns:
            QDQParams with optimal scale and zero_point
        """
        if self.method == CalibrationMethod.MINMAX:
            return self._compute_minmax()
        elif self.method == CalibrationMethod.PERCENTILE:
            return self._compute_percentile()
        elif self.method == CalibrationMethod.MSE:
            return self._compute_mse()
        elif self.method == CalibrationMethod.ENTROPY:
            return self._compute_entropy()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def _compute_minmax(self) -> QDQParams:
        """Compute params using min/max values."""
        if self.per_channel and self._channel_stats is not None:
            scales = []
            zero_points = []
            for stats in self._channel_stats:
                scale, zp = _calibrate_minmax_range(
                    stats.min_val, stats.max_val,
                    self.dtype, self.symmetric
                )
                scales.append(scale)
                zero_points.append(zp)
            return QDQParams(
                scale=np.array(scales, dtype=np.float32),
                zero_point=np.array(zero_points, dtype=np.int32),
                dtype=self.dtype,
                axis=self.channel_axis,
            )
        else:
            scale, zp = _calibrate_minmax_range(
                self.stats.min_val, self.stats.max_val,
                self.dtype, self.symmetric
            )
            return QDQParams(scale=scale, zero_point=zp, dtype=self.dtype)

    def _compute_percentile(self) -> QDQParams:
        """Compute params using percentile values."""
        # Need collected values for percentile
        if not self._all_values:
            # Fall back to minmax if no values collected
            return self._compute_minmax()

        if self.per_channel and self._channel_stats is not None:
            scales = []
            zero_points = []
            for c, stats in enumerate(self._channel_stats):
                channel_values = [v for ch, v in self._all_values if ch == c]
                if channel_values:
                    all_data = np.concatenate(channel_values)
                    low_p = (100 - self.percentile) / 2
                    high_p = 100 - low_p
                    min_val = np.percentile(all_data, low_p)
                    max_val = np.percentile(all_data, high_p)
                else:
                    min_val, max_val = stats.min_val, stats.max_val
                scale, zp = _calibrate_minmax_range(
                    min_val, max_val, self.dtype, self.symmetric
                )
                scales.append(scale)
                zero_points.append(zp)
            self._all_values = []  # Free memory
            return QDQParams(
                scale=np.array(scales, dtype=np.float32),
                zero_point=np.array(zero_points, dtype=np.int32),
                dtype=self.dtype,
                axis=self.channel_axis,
            )
        else:
            all_data = np.concatenate(self._all_values)
            low_p = (100 - self.percentile) / 2
            high_p = 100 - low_p
            min_val = np.percentile(all_data, low_p)
            max_val = np.percentile(all_data, high_p)
            self._all_values = []  # Free memory
            scale, zp = _calibrate_minmax_range(
                min_val, max_val, self.dtype, self.symmetric
            )
            return QDQParams(scale=scale, zero_point=zp, dtype=self.dtype)

    def _compute_mse(self) -> QDQParams:
        """Compute params by minimizing MSE."""
        if not self._all_values:
            return self._compute_minmax()

        if self.per_channel and self._channel_stats is not None:
            scales = []
            zero_points = []
            for c, stats in enumerate(self._channel_stats):
                channel_values = [v for ch, v in self._all_values if ch == c]
                if channel_values:
                    all_data = np.concatenate(channel_values)
                    scale, zp = _calibrate_mse_search(
                        all_data, self.dtype, self.symmetric
                    )
                else:
                    scale, zp = _calibrate_minmax_range(
                        stats.min_val, stats.max_val,
                        self.dtype, self.symmetric
                    )
                scales.append(scale)
                zero_points.append(zp)
            self._all_values = []
            return QDQParams(
                scale=np.array(scales, dtype=np.float32),
                zero_point=np.array(zero_points, dtype=np.int32),
                dtype=self.dtype,
                axis=self.channel_axis,
            )
        else:
            all_data = np.concatenate(self._all_values)
            scale, zp = _calibrate_mse_search(all_data, self.dtype, self.symmetric)
            self._all_values = []
            return QDQParams(scale=scale, zero_point=zp, dtype=self.dtype)

    def _compute_entropy(self) -> QDQParams:
        """Compute params by minimizing KL divergence."""
        self._build_histogram()

        if self.per_channel and self._channel_stats is not None:
            scales = []
            zero_points = []
            for stats in self._channel_stats:
                if stats.histogram is not None and stats.bin_edges is not None:
                    scale, zp = _calibrate_entropy_search(
                        stats.histogram, stats.bin_edges,
                        self.dtype, self.symmetric
                    )
                else:
                    scale, zp = _calibrate_minmax_range(
                        stats.min_val, stats.max_val,
                        self.dtype, self.symmetric
                    )
                scales.append(scale)
                zero_points.append(zp)
            return QDQParams(
                scale=np.array(scales, dtype=np.float32),
                zero_point=np.array(zero_points, dtype=np.int32),
                dtype=self.dtype,
                axis=self.channel_axis,
            )
        else:
            if self.stats.histogram is not None and self.stats.bin_edges is not None:
                scale, zp = _calibrate_entropy_search(
                    self.stats.histogram, self.stats.bin_edges,
                    self.dtype, self.symmetric
                )
            else:
                scale, zp = _calibrate_minmax_range(
                    self.stats.min_val, self.stats.max_val,
                    self.dtype, self.symmetric
                )
            return QDQParams(scale=scale, zero_point=zp, dtype=self.dtype)

    def reset(self) -> None:
        """Reset observer to initial state."""
        self.stats.reset()
        self._channel_stats = None
        self._all_values = []
        self._histogram_built = False


def _calibrate_minmax_range(
    min_val: float,
    max_val: float,
    dtype: QuantDtype,
    symmetric: bool,
) -> Tuple[float, int]:
    """Compute scale and zero_point from min/max range.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        dtype: Target dtype
        symmetric: Use symmetric quantization

    Returns:
        (scale, zero_point) tuple
    """
    qmin, qmax = dtype.qmin, dtype.qmax

    if symmetric:
        # Symmetric: scale = max(|min|, |max|) / qmax
        abs_max = max(abs(min_val), abs(max_val))
        if abs_max == 0:
            abs_max = 1.0
        scale = abs_max / max(abs(qmin), abs(qmax))
        zero_point = 0
    else:
        # Asymmetric: map [min, max] to [qmin, qmax]
        if max_val == min_val:
            scale = 1.0
            zero_point = qmin
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = int(round(qmin - min_val / scale))
            zero_point = max(qmin, min(qmax, zero_point))

    return float(scale), int(zero_point)


def _calibrate_mse_search(
    values: np.ndarray,
    dtype: QuantDtype,
    symmetric: bool,
    num_steps: int = 100,
) -> Tuple[float, int]:
    """Search for scale that minimizes MSE.

    Args:
        values: Calibration data
        dtype: Target dtype
        symmetric: Use symmetric quantization
        num_steps: Number of scale candidates to try

    Returns:
        (scale, zero_point) tuple
    """
    qmin, qmax = dtype.qmin, dtype.qmax

    # Get range from data
    data_min, data_max = float(np.min(values)), float(np.max(values))

    if symmetric:
        abs_max = max(abs(data_min), abs(data_max))
        if abs_max == 0:
            return 1.0, 0

        # Search over different percentile clipping
        best_scale = abs_max / max(abs(qmin), abs(qmax))
        best_mse = float('inf')

        for p in np.linspace(0.9, 1.0, num_steps):
            clip_val = abs_max * p
            scale = clip_val / max(abs(qmin), abs(qmax))

            # Quantize and compute MSE
            clipped = np.clip(values, -clip_val, clip_val)
            q = np.round(clipped / scale).astype(np.int32)
            q = np.clip(q, qmin, qmax)
            dq = q.astype(np.float32) * scale
            mse = np.mean((values - dq) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_scale = scale

        return float(best_scale), 0
    else:
        if data_max == data_min:
            return 1.0, qmin

        best_scale = (data_max - data_min) / (qmax - qmin)
        best_zp = int(round(qmin - data_min / best_scale))
        best_mse = float('inf')

        # Search over different clipping ranges
        for p_low in np.linspace(0.0, 0.1, 10):
            for p_high in np.linspace(0.9, 1.0, 10):
                clip_min = data_min + (data_max - data_min) * p_low
                clip_max = data_min + (data_max - data_min) * p_high

                if clip_max <= clip_min:
                    continue

                scale = (clip_max - clip_min) / (qmax - qmin)
                zp = int(round(qmin - clip_min / scale))
                zp = max(qmin, min(qmax, zp))

                # Quantize and compute MSE
                clipped = np.clip(values, clip_min, clip_max)
                q = np.round(clipped / scale + zp).astype(np.int32)
                q = np.clip(q, qmin, qmax)
                dq = (q.astype(np.float32) - zp) * scale
                mse = np.mean((values - dq) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_scale = scale
                    best_zp = zp

        return float(best_scale), int(best_zp)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: Reference distribution (normalized)
        q: Approximation distribution (normalized)

    Returns:
        KL divergence value
    """
    # Add small epsilon for numerical stability
    p = p + 1e-10
    q = q + 1e-10
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Only compute where p > 0
    mask = p > 1e-10
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _calibrate_entropy_search(
    histogram: np.ndarray,
    bin_edges: np.ndarray,
    dtype: QuantDtype,
    symmetric: bool,
) -> Tuple[float, int]:
    """Search for scale that minimizes KL divergence.

    This implementation searches over different clipping thresholds to find
    the one that minimizes information loss when quantizing the distribution.

    Args:
        histogram: Value histogram
        bin_edges: Histogram bin edges
        dtype: Target dtype
        symmetric: Use symmetric quantization

    Returns:
        (scale, zero_point) tuple
    """
    qmin, qmax = dtype.qmin, dtype.qmax
    num_quant_levels = qmax - qmin + 1

    # Reference distribution
    ref_hist = histogram.astype(np.float64)
    total_count = np.sum(ref_hist)
    if total_count == 0:
        return 1.0, 0

    num_bins = len(histogram)
    bin_width = (bin_edges[-1] - bin_edges[0]) / num_bins

    # Get data range from histogram
    nonzero_bins = np.nonzero(histogram)[0]
    if len(nonzero_bins) == 0:
        return 1.0, 0

    data_min = bin_edges[nonzero_bins[0]]
    data_max = bin_edges[nonzero_bins[-1] + 1]
    abs_max = max(abs(data_min), abs(data_max))

    if abs_max == 0:
        return 1.0, 0

    # Initialize with minmax result
    best_scale, best_zp = _calibrate_minmax_range(data_min, data_max, dtype, symmetric)
    best_kl = float('inf')

    if symmetric:
        # For symmetric, search over clip percentages of abs_max
        # Start from a reasonable minimum (keeping most data) to the full range
        for clip_pct in np.linspace(0.5, 1.0, 100):
            clip_val = abs_max * clip_pct
            scale = clip_val / max(abs(qmin), abs(qmax))

            # Compute KL divergence for this clipping
            # Create quantized distribution by mapping bins to quant levels
            quant_hist = np.zeros(num_quant_levels, dtype=np.float64)

            for i in range(num_bins):
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                # Clip and quantize
                clipped = np.clip(bin_center, -clip_val, clip_val)
                q_level = int(round(clipped / scale))
                q_level = max(qmin, min(qmax, q_level))
                quant_hist[q_level - qmin] += histogram[i]

            # Skip if all zeros
            if np.sum(quant_hist) == 0:
                continue

            # Expand quantized distribution back to original bins for KL comparison
            expanded_hist = np.zeros(num_bins, dtype=np.float64)
            for q in range(num_quant_levels):
                dequant_val = (q + qmin) * scale
                # Find which bin this maps to
                bin_idx = int((dequant_val - bin_edges[0]) / bin_width)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                expanded_hist[bin_idx] += quant_hist[q]

            # Compute KL divergence
            kl = _kl_divergence(ref_hist, expanded_hist)

            if kl < best_kl:
                best_kl = kl
                best_scale = scale

        return float(best_scale), 0

    else:
        # For asymmetric, search over clip ranges
        for low_pct in np.linspace(0.0, 0.2, 20):
            for high_pct in np.linspace(0.8, 1.0, 20):
                clip_min = data_min + (data_max - data_min) * low_pct
                clip_max = data_min + (data_max - data_min) * high_pct

                if clip_max <= clip_min:
                    continue

                scale = (clip_max - clip_min) / (qmax - qmin)
                zp = int(round(qmin - clip_min / scale))
                zp = max(qmin, min(qmax, zp))

                # Create quantized distribution
                quant_hist = np.zeros(num_quant_levels, dtype=np.float64)
                for i in range(num_bins):
                    bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                    clipped = np.clip(bin_center, clip_min, clip_max)
                    q_level = int(round(clipped / scale + zp))
                    q_level = max(qmin, min(qmax, q_level))
                    quant_hist[q_level - qmin] += histogram[i]

                if np.sum(quant_hist) == 0:
                    continue

                # Expand back
                expanded_hist = np.zeros(num_bins, dtype=np.float64)
                for q in range(num_quant_levels):
                    val = (q + qmin - zp) * scale
                    bin_idx = int((val - bin_edges[0]) / bin_width)
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                    expanded_hist[bin_idx] += quant_hist[q]

                kl = _kl_divergence(ref_hist, expanded_hist)

                if kl < best_kl:
                    best_kl = kl
                    best_scale = scale
                    best_zp = zp

        return float(best_scale), int(best_zp)


def calibrate_minmax(
    tensor: np.ndarray,
    dtype: QuantDtype = QuantDtype.INT8,
    symmetric: bool = True,
) -> QDQParams:
    """Simple min/max calibration.

    Args:
        tensor: Tensor to calibrate
        dtype: Target quantization dtype
        symmetric: Use symmetric quantization

    Returns:
        QDQParams with calibrated scale/zero_point
    """
    min_val = float(np.min(tensor))
    max_val = float(np.max(tensor))
    scale, zp = _calibrate_minmax_range(min_val, max_val, dtype, symmetric)
    return QDQParams(scale=scale, zero_point=zp, dtype=dtype)


def calibrate_percentile(
    tensor: np.ndarray,
    dtype: QuantDtype = QuantDtype.INT8,
    symmetric: bool = True,
    percentile: float = 99.99,
) -> QDQParams:
    """Percentile-based calibration (handles outliers).

    Args:
        tensor: Tensor to calibrate
        dtype: Target quantization dtype
        symmetric: Use symmetric quantization
        percentile: Percentile to use (default 99.99)

    Returns:
        QDQParams with calibrated scale/zero_point
    """
    low_p = (100 - percentile) / 2
    high_p = 100 - low_p
    min_val = float(np.percentile(tensor, low_p))
    max_val = float(np.percentile(tensor, high_p))
    scale, zp = _calibrate_minmax_range(min_val, max_val, dtype, symmetric)
    return QDQParams(scale=scale, zero_point=zp, dtype=dtype)


def calibrate_mse(
    tensor: np.ndarray,
    dtype: QuantDtype = QuantDtype.INT8,
    symmetric: bool = True,
) -> QDQParams:
    """MSE-minimizing calibration.

    Args:
        tensor: Tensor to calibrate
        dtype: Target quantization dtype
        symmetric: Use symmetric quantization

    Returns:
        QDQParams with calibrated scale/zero_point
    """
    scale, zp = _calibrate_mse_search(
        tensor.flatten(), dtype, symmetric
    )
    return QDQParams(scale=scale, zero_point=zp, dtype=dtype)


def calibrate_entropy(
    tensor: np.ndarray,
    dtype: QuantDtype = QuantDtype.INT8,
    symmetric: bool = True,
    num_bins: int = 2048,
) -> QDQParams:
    """Entropy-minimizing (KL divergence) calibration.

    Args:
        tensor: Tensor to calibrate
        dtype: Target quantization dtype
        symmetric: Use symmetric quantization
        num_bins: Number of histogram bins

    Returns:
        QDQParams with calibrated scale/zero_point
    """
    histogram, bin_edges = np.histogram(tensor.flatten(), bins=num_bins)
    scale, zp = _calibrate_entropy_search(histogram, bin_edges, dtype, symmetric)
    return QDQParams(scale=scale, zero_point=zp, dtype=dtype)


def compare_calibration_methods(
    tensor: np.ndarray,
    dtype: QuantDtype = QuantDtype.INT8,
    symmetric: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compare all calibration methods on a tensor.

    Args:
        tensor: Tensor to calibrate
        dtype: Target quantization dtype
        symmetric: Use symmetric quantization

    Returns:
        Dictionary mapping method name to {params, error_metrics}
    """
    methods = {
        "minmax": calibrate_minmax,
        "percentile": calibrate_percentile,
        "mse": calibrate_mse,
        "entropy": calibrate_entropy,
    }

    results = {}
    for name, calibrate_fn in methods.items():
        if name == "percentile":
            params = calibrate_fn(tensor, dtype, symmetric, percentile=99.99)
        elif name == "entropy":
            params = calibrate_fn(tensor, dtype, symmetric, num_bins=2048)
        else:
            params = calibrate_fn(tensor, dtype, symmetric)

        error = quantize_error(tensor, params)
        results[name] = {
            "params": params,
            "scale": params.scale,
            "zero_point": params.zero_point,
            "snr_db": error["snr_db"],
            "mean_abs_error": error["mean_abs_error"],
            "max_abs_error": error["max_abs_error"],
        }

    return results


def calibration_info() -> str:
    """Return information about calibration methods.

    Returns:
        Multi-line string with calibration method descriptions
    """
    return """Calibration Methods (v0.7.11)
==============================

MinMax (CalibrationMethod.MINMAX)
  - Uses observed min/max values
  - Fast and simple
  - Sensitive to outliers
  - Best for: well-behaved distributions

Percentile (CalibrationMethod.PERCENTILE)
  - Uses percentile values (default 99.99%)
  - Clips outliers
  - Good balance of speed and accuracy
  - Best for: distributions with outliers

MSE (CalibrationMethod.MSE)
  - Minimizes mean squared error
  - Searches over clipping ranges
  - Good for minimizing overall error
  - Best for: accuracy-critical applications

Entropy (CalibrationMethod.ENTROPY)
  - Minimizes KL divergence (information loss)
  - Histogram-based
  - Preserves distribution shape
  - Best for: maintaining statistical properties

Usage:
  observer = CalibrationObserver(
      method=CalibrationMethod.PERCENTILE,
      dtype=QuantDtype.INT8,
      symmetric=True,
  )

  for batch in calibration_data:
      activations = model(batch)
      observer.observe(activations)

  params = observer.compute_params()

Quick calibration:
  params = calibrate_mse(tensor, dtype=QuantDtype.INT8)
  params = calibrate_percentile(tensor, percentile=99.9)
"""


# --- Memory traffic calculation ---

def calculate_memory_bytes(
    shape: Tuple[int, ...],
    dtype: QuantDtype = QuantDtype.FP32,
) -> int:
    """Calculate memory bytes for a tensor with given dtype.

    Args:
        shape: Tensor shape
        dtype: Quantization dtype

    Returns:
        Total bytes
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return int(num_elements * dtype.bytes_per_element)


def calculate_matmul_traffic(
    M: int, K: int, N: int,
    dtype: QuantDtype = QuantDtype.FP32,
) -> int:
    """Calculate memory traffic for matmul operation.

    Memory traffic = read A (M*K) + read B (K*N) + write C (M*N)

    Args:
        M, K, N: Matrix dimensions (A: MxK, B: KxN, C: MxN)
        dtype: Data type for all operands

    Returns:
        Total bytes transferred
    """
    bytes_per_elem = dtype.bytes_per_element
    traffic = (M * K + K * N + M * N) * bytes_per_elem
    return int(traffic)


def bandwidth_reduction_factor(
    dtype: QuantDtype,
    baseline: QuantDtype = QuantDtype.FP32,
) -> float:
    """Calculate bandwidth reduction factor vs baseline.

    Args:
        dtype: Target dtype
        baseline: Reference dtype (default FP32)

    Returns:
        Reduction factor (e.g., 4.0 for INT8 vs FP32)
    """
    return baseline.bytes_per_element / dtype.bytes_per_element
