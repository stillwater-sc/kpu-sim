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
