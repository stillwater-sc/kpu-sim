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
