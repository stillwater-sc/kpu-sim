# python/kpu/dfx_emitter.py
"""
DFX IR (Domain Flow Execution) emitter.

Converts OpGraph to DFX IR, which is the intermediate representation
used by the KPU compiler (similar to PTX for NVIDIA GPUs).

The DFX IR can be:
1. Serialized to JSON for consumption by the C++ compiler
2. Directly interpreted by the Python runtime for behavioral simulation
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .graph import OpGraph, OpNode
    from .tensor import Tensor


class DFXDataType(Enum):
    """Data types supported by DFX IR."""
    # Full precision
    FLOAT32 = "f32"
    FLOAT64 = "f64"

    # 16-bit types
    FLOAT16 = "f16"
    BFLOAT16 = "bf16"

    # 8-bit float types (v0.7.0+)
    FLOAT8_E4M3 = "fp8_e4m3"
    FLOAT8_E5M2 = "fp8_e5m2"
    FLOAT8_E3M4 = "fp8_e3m4"
    FLOAT8_E2M5 = "fp8_e2m5"

    # Integer types
    INT32 = "i32"
    INT16 = "i16"
    INT8 = "i8"
    UINT8 = "u8"
    INT4 = "i4"
    UINT4 = "u4"

    # Other
    BOOL = "bool"

    @property
    def bytes_per_element(self) -> float:
        """Return bytes per element for this dtype."""
        sizes = {
            DFXDataType.FLOAT64: 8.0,
            DFXDataType.FLOAT32: 4.0,
            DFXDataType.FLOAT16: 2.0,
            DFXDataType.BFLOAT16: 2.0,
            DFXDataType.FLOAT8_E4M3: 1.0,
            DFXDataType.FLOAT8_E5M2: 1.0,
            DFXDataType.FLOAT8_E3M4: 1.0,
            DFXDataType.FLOAT8_E2M5: 1.0,
            DFXDataType.INT32: 4.0,
            DFXDataType.INT16: 2.0,
            DFXDataType.INT8: 1.0,
            DFXDataType.UINT8: 1.0,
            DFXDataType.INT4: 0.5,
            DFXDataType.UINT4: 0.5,
            DFXDataType.BOOL: 1.0,
        }
        return sizes.get(self, 4.0)


class DFXMemLevel(Enum):
    """Memory hierarchy levels in KPU."""
    EXTERNAL = 0  # DRAM / Host memory
    L3 = 1        # L3 SRAM (shared across tiles)
    L2 = 2        # L2 SRAM (per-tile)
    L1 = 3        # L1 register file (per-PE)
    REGISTER = 4  # Accumulator registers


class DFXOpCode(Enum):
    """DFX operation codes."""
    # Data movement
    LOAD = "load"
    STORE = "store"
    PREFETCH = "prefetch"
    COPY = "copy"

    # Compute - matrix
    MATMUL = "matmul"
    LINEAR = "linear"  # y = x @ W.T + b (weight transposed)
    CONV2D = "conv2d"

    # Compute - attention
    ATTENTION = "attention"  # Scaled dot-product attention

    # Compute - activation
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"

    # Compute - normalization
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"

    # Compute - elementwise
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    NEG = "neg"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"

    # Compute - reduction
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"

    # Compute - pooling
    MAXPOOL2D = "maxpool2d"
    AVGPOOL2D = "avgpool2d"
    ADAPTIVE_AVGPOOL2D = "adaptive_avgpool2d"

    # Shape operations
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    CONCAT = "concat"
    FLATTEN = "flatten"

    # Control
    BARRIER = "barrier"
    NOP = "nop"

    # Fused operations (v0.6.0+)
    FUSED_MATMUL_BIAS_RELU = "fused_matmul_bias_relu"
    FUSED_MATMUL_BIAS_GELU = "fused_matmul_bias_gelu"
    FUSED_MATMUL_BIAS_SILU = "fused_matmul_bias_silu"
    FUSED_MATMUL_RELU = "fused_matmul_relu"

    # Conv2D fused operations (v0.6.2+)
    FUSED_CONV2D_BN_RELU = "fused_conv2d_bn_relu"
    FUSED_CONV2D_RELU = "fused_conv2d_relu"

    # Quantization operations (v0.7.0+)
    QUANTIZE = "quantize"          # float -> int with scale/zero_point
    DEQUANTIZE = "dequantize"      # int -> float with scale/zero_point

    # Quantized compute operations (v0.7.0+)
    QUANTIZED_MATMUL = "quantized_matmul"  # INT8 matmul
    QUANTIZED_LINEAR = "quantized_linear"  # INT8 linear layer
    QUANTIZED_CONV2D = "quantized_conv2d"  # INT8 conv2d


@dataclass
class DFXTensor:
    """
    Tensor descriptor in DFX IR.

    Represents a named tensor with shape, type, and memory location.
    """
    name: str
    shape: Tuple[int, ...]
    dtype: DFXDataType
    memory_level: DFXMemLevel = DFXMemLevel.EXTERNAL
    is_const: bool = False  # True for weights that don't change

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "memory_level": self.memory_level.value,
            "is_const": self.is_const,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DFXTensor':
        """Create from dictionary."""
        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=DFXDataType(d["dtype"]),
            memory_level=DFXMemLevel(d["memory_level"]),
            is_const=d.get("is_const", False),
        )


@dataclass
class DFXOp:
    """
    A single operation in DFX IR.
    """
    opcode: DFXOpCode
    inputs: List[str]   # Tensor names
    outputs: List[str]  # Tensor names
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "opcode": self.opcode.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DFXOp':
        """Create from dictionary."""
        return cls(
            opcode=DFXOpCode(d["opcode"]),
            inputs=d["inputs"],
            outputs=d["outputs"],
            attrs=d.get("attrs", {}),
        )


@dataclass
class DFXProgram:
    """
    Complete DFX program.

    Contains tensor descriptors and operations in execution order.
    """
    name: str
    tensors: Dict[str, DFXTensor]
    ops: List[DFXOp]
    inputs: List[str]   # Input tensor names
    outputs: List[str]  # Output tensor names
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "version": "1.0",
            "tensors": {name: t.to_dict() for name, t in self.tensors.items()},
            "ops": [op.to_dict() for op in self.ops],
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DFXProgram':
        """Create from dictionary."""
        return cls(
            name=d["name"],
            tensors={name: DFXTensor.from_dict(t) for name, t in d["tensors"].items()},
            ops=[DFXOp.from_dict(op) for op in d["ops"]],
            inputs=d["inputs"],
            outputs=d["outputs"],
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'DFXProgram':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"DFXProgram '{self.name}'",
            f"  Tensors: {len(self.tensors)}",
            f"  Operations: {len(self.ops)}",
            f"  Inputs: {self.inputs}",
            f"  Outputs: {self.outputs}",
            "",
            "  Operations:",
        ]

        for i, op in enumerate(self.ops):
            inputs_str = ", ".join(op.inputs)
            outputs_str = ", ".join(op.outputs)
            attrs_str = f" {op.attrs}" if op.attrs else ""
            lines.append(f"    [{i}] {op.opcode.value}({inputs_str}) -> ({outputs_str}){attrs_str}")

        return "\n".join(lines)


class DFXEmitter:
    """
    Emits DFX IR from an OpGraph.

    This class converts the high-level operation graph built during tracing
    into DFX IR that can be consumed by the KPU compiler.

    Example:
        >>> emitter = DFXEmitter()
        >>> program = emitter.emit(graph)
        >>> print(program.to_json())
    """

    def __init__(self):
        self._tensor_counter = 0
        self._tensors: Dict[int, DFXTensor] = {}  # Python id(tensor) -> DFXTensor
        self._tensor_names: Dict[int, str] = {}   # Python id(tensor) -> name

    def emit(self, graph: 'OpGraph') -> DFXProgram:
        """
        Convert OpGraph to DFX program.

        Args:
            graph: Operation graph from tracing

        Returns:
            DFX program ready for compilation
        """
        from .graph import OpType

        # Reset state
        self._tensor_counter = 0
        self._tensors.clear()
        self._tensor_names.clear()

        dfx_ops = []

        # Register input tensors first
        for i, tensor in enumerate(graph.inputs):
            name = tensor.name or f"input_{i}"
            self._register_tensor(tensor, name, is_input=True)

        # Process nodes in topological order
        for node in graph.topological_order():
            ops = self._emit_node(node)
            dfx_ops.extend(ops)

        # Collect all tensors
        all_tensors = {t.name: t for t in self._tensors.values()}

        # Get input/output names
        input_names = [self._get_tensor_name(t) for t in graph.inputs]
        output_names = [self._get_tensor_name(t) for t in graph.outputs]

        # Compute metadata
        metadata = self._compute_metadata(graph, dfx_ops)

        return DFXProgram(
            name=graph.name,
            tensors=all_tensors,
            ops=dfx_ops,
            inputs=input_names,
            outputs=output_names,
            metadata=metadata,
        )

    def _register_tensor(self, tensor: 'Tensor', name: Optional[str] = None,
                         is_input: bool = False) -> DFXTensor:
        """Register a tensor and return its DFX descriptor."""
        tid = id(tensor)

        if tid in self._tensors:
            return self._tensors[tid]

        if name is None:
            name = tensor.name or f"t{self._tensor_counter}"
            self._tensor_counter += 1

        dtype = self._numpy_to_dfx_dtype(tensor.dtype)

        dfx_tensor = DFXTensor(
            name=name,
            shape=tensor.shape,
            dtype=dtype,
            memory_level=DFXMemLevel.EXTERNAL,
            is_const=tensor._meta.is_weight if hasattr(tensor._meta, 'is_weight') else False,
        )

        self._tensors[tid] = dfx_tensor
        self._tensor_names[tid] = name
        return dfx_tensor

    def _get_tensor_name(self, tensor: 'Tensor') -> str:
        """Get the DFX name for a tensor."""
        tid = id(tensor)
        if tid not in self._tensor_names:
            dfx_tensor = self._register_tensor(tensor)
            return dfx_tensor.name
        return self._tensor_names[tid]

    def _emit_node(self, node: 'OpNode') -> List[DFXOp]:
        """Emit DFX ops for a single graph node."""
        from .graph import OpType

        # Register output tensors
        for out in node.outputs:
            self._register_tensor(out)

        input_names = [self._get_tensor_name(t) for t in node.inputs]
        output_names = [self._get_tensor_name(t) for t in node.outputs]

        # Map OpType to DFXOpCode
        op_map = {
            OpType.MATMUL: DFXOpCode.MATMUL,
            OpType.CONV2D: DFXOpCode.CONV2D,
            OpType.ATTENTION: DFXOpCode.ATTENTION,
            OpType.RELU: DFXOpCode.RELU,
            OpType.GELU: DFXOpCode.GELU,
            OpType.SILU: DFXOpCode.SILU,
            OpType.SIGMOID: DFXOpCode.SIGMOID,
            OpType.TANH: DFXOpCode.TANH,
            OpType.SOFTMAX: DFXOpCode.SOFTMAX,
            OpType.LAYER_NORM: DFXOpCode.LAYER_NORM,
            OpType.BATCH_NORM: DFXOpCode.BATCH_NORM,
            OpType.ADD: DFXOpCode.ADD,
            OpType.SUB: DFXOpCode.SUB,
            OpType.MUL: DFXOpCode.MUL,
            OpType.DIV: DFXOpCode.DIV,
            OpType.NEG: DFXOpCode.NEG,
            OpType.EXP: DFXOpCode.EXP,
            OpType.LOG: DFXOpCode.LOG,
            OpType.SQRT: DFXOpCode.SQRT,
            OpType.SUM: DFXOpCode.SUM,
            OpType.MEAN: DFXOpCode.MEAN,
            OpType.MAX: DFXOpCode.MAX,
            OpType.MIN: DFXOpCode.MIN,
            OpType.MAXPOOL2D: DFXOpCode.MAXPOOL2D,
            OpType.AVGPOOL2D: DFXOpCode.AVGPOOL2D,
            OpType.ADAPTIVE_AVGPOOL2D: DFXOpCode.ADAPTIVE_AVGPOOL2D,
            OpType.RESHAPE: DFXOpCode.RESHAPE,
            OpType.TRANSPOSE: DFXOpCode.TRANSPOSE,
            OpType.CONCAT: DFXOpCode.CONCAT,
            OpType.FLATTEN: DFXOpCode.FLATTEN,
            # Fused operations
            OpType.FUSED_MATMUL_BIAS_RELU: DFXOpCode.FUSED_MATMUL_BIAS_RELU,
            OpType.FUSED_MATMUL_BIAS_GELU: DFXOpCode.FUSED_MATMUL_BIAS_GELU,
            OpType.FUSED_MATMUL_BIAS_SILU: DFXOpCode.FUSED_MATMUL_BIAS_SILU,
            OpType.FUSED_MATMUL_RELU: DFXOpCode.FUSED_MATMUL_RELU,
            # Conv2D fused operations (v0.6.2+)
            OpType.FUSED_CONV2D_BN_RELU: DFXOpCode.FUSED_CONV2D_BN_RELU,
            OpType.FUSED_CONV2D_RELU: DFXOpCode.FUSED_CONV2D_RELU,
        }

        if node.op_type not in op_map:
            raise NotImplementedError(f"Op type {node.op_type} not implemented in DFX emitter")

        opcode = op_map[node.op_type]

        # Build attributes
        attrs = dict(node.attrs)

        # Add shape info for matmul and fused ops
        if node.op_type == OpType.MATMUL:
            A_shape = node.inputs[0].shape
            B_shape = node.inputs[1].shape
            attrs["M"] = A_shape[-2]
            attrs["K"] = A_shape[-1]
            attrs["N"] = B_shape[-1]
        elif node.op_type.is_fused_matmul():
            # Fused MatMul ops have A, B as first two inputs
            A_shape = node.inputs[0].shape
            B_shape = node.inputs[1].shape
            attrs["M"] = A_shape[-2]
            attrs["K"] = A_shape[-1]
            attrs["N"] = B_shape[-1]
        elif node.op_type.is_fused_conv():
            # Fused Conv2D ops have input, weight as first two inputs
            # input: (N, C_in, H, W) or (N, H, W, C_in)
            # weight: (C_out, C_in, kH, kW)
            input_shape = node.inputs[0].shape
            weight_shape = node.inputs[1].shape
            attrs["batch_size"] = input_shape[0]
            attrs["in_channels"] = weight_shape[1]
            attrs["out_channels"] = weight_shape[0]
            attrs["kernel_size"] = (weight_shape[2], weight_shape[3])

        return [DFXOp(
            opcode=opcode,
            inputs=input_names,
            outputs=output_names,
            attrs=attrs,
        )]

    def _numpy_to_dfx_dtype(self, dtype) -> DFXDataType:
        """Convert numpy dtype to DFX dtype."""
        import numpy as np

        dtype_map = {
            np.float32: DFXDataType.FLOAT32,
            np.float16: DFXDataType.FLOAT16,
            np.int32: DFXDataType.INT32,
            np.int16: DFXDataType.INT16,
            np.int8: DFXDataType.INT8,
            np.uint8: DFXDataType.UINT8,
            np.bool_: DFXDataType.BOOL,
        }

        np_dtype = np.dtype(dtype)
        return dtype_map.get(np_dtype.type, DFXDataType.FLOAT32)

    def _compute_metadata(self, graph: 'OpGraph', ops: List[DFXOp]) -> Dict[str, Any]:
        """Compute program metadata."""
        from .graph import OpType

        stats = graph.compute_stats()

        return {
            "num_ops": len(ops),
            "num_tensors": len(self._tensors),
            "total_matmul_flops": stats.get("total_matmul_flops", 0),
            "op_counts": stats.get("op_counts", {}),
        }
