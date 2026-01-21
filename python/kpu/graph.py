# python/kpu/graph.py
"""
Operation Graph for KPU compilation.

Represents the DAG of operations to be compiled to DFX IR.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .tensor import Tensor


class OpType(Enum):
    """Operation types supported by the KPU."""

    # Data movement (implicit in graph structure)
    INPUT = "input"
    OUTPUT = "output"

    # Matrix operations
    MATMUL = "matmul"

    # Convolution operations
    CONV2D = "conv2d"

    # Attention operations
    ATTENTION = "attention"

    # Activation functions
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    TANH = "tanh"

    # Normalization
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"

    # Elementwise binary operations
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"

    # Elementwise unary operations
    NEG = "neg"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"

    # Reduction operations
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"

    # Pooling operations
    MAXPOOL2D = "maxpool2d"
    AVGPOOL2D = "avgpool2d"
    ADAPTIVE_AVGPOOL2D = "adaptive_avgpool2d"

    # Reshape operations
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    CONCAT = "concat"
    FLATTEN = "flatten"

    # Fused operations (v0.6.0+)
    FUSED_MATMUL_BIAS_RELU = "fused_matmul_bias_relu"
    FUSED_MATMUL_BIAS_GELU = "fused_matmul_bias_gelu"
    FUSED_MATMUL_BIAS_SILU = "fused_matmul_bias_silu"
    FUSED_MATMUL_RELU = "fused_matmul_relu"

    # Conv2D fused operations (v0.6.2+)
    FUSED_CONV2D_BN_RELU = "fused_conv2d_bn_relu"
    FUSED_CONV2D_RELU = "fused_conv2d_relu"

    def is_compute(self) -> bool:
        """Return True if this is a compute operation (not data movement)."""
        return self not in (OpType.INPUT, OpType.OUTPUT)

    def is_activation(self) -> bool:
        """Return True if this is an activation function."""
        return self in (
            OpType.RELU, OpType.GELU, OpType.SILU,
            OpType.SOFTMAX, OpType.SIGMOID, OpType.TANH
        )

    def is_elementwise(self) -> bool:
        """Return True if this is an elementwise operation."""
        return self in (
            OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
            OpType.NEG, OpType.EXP, OpType.LOG, OpType.SQRT,
            OpType.RELU, OpType.GELU, OpType.SILU, OpType.SIGMOID, OpType.TANH
        )

    def is_reduction(self) -> bool:
        """Return True if this is a reduction operation."""
        return self in (OpType.SUM, OpType.MEAN, OpType.MAX, OpType.MIN, OpType.SOFTMAX)

    def is_pooling(self) -> bool:
        """Return True if this is a pooling operation."""
        return self in (OpType.MAXPOOL2D, OpType.AVGPOOL2D, OpType.ADAPTIVE_AVGPOOL2D)

    def is_conv(self) -> bool:
        """Return True if this is a convolution operation."""
        return self == OpType.CONV2D

    def is_normalization(self) -> bool:
        """Return True if this is a normalization operation."""
        return self in (OpType.LAYER_NORM, OpType.BATCH_NORM)

    def is_fused(self) -> bool:
        """Return True if this is a fused operation."""
        return self in (
            OpType.FUSED_MATMUL_BIAS_RELU,
            OpType.FUSED_MATMUL_BIAS_GELU,
            OpType.FUSED_MATMUL_BIAS_SILU,
            OpType.FUSED_MATMUL_RELU,
            OpType.FUSED_CONV2D_BN_RELU,
            OpType.FUSED_CONV2D_RELU,
        )

    def is_fused_conv(self) -> bool:
        """Return True if this is a fused convolution operation."""
        return self in (
            OpType.FUSED_CONV2D_BN_RELU,
            OpType.FUSED_CONV2D_RELU,
        )

    def is_fused_matmul(self) -> bool:
        """Return True if this is a fused matmul operation."""
        return self in (
            OpType.FUSED_MATMUL_BIAS_RELU,
            OpType.FUSED_MATMUL_BIAS_GELU,
            OpType.FUSED_MATMUL_BIAS_SILU,
            OpType.FUSED_MATMUL_RELU,
        )


@dataclass
class OpNode:
    """A single operation in the computation graph."""

    op_type: OpType
    inputs: List['Tensor']
    outputs: List['Tensor']
    attrs: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    # For scheduling (assigned during graph construction)
    _id: int = -1
    _deps: Set[int] = field(default_factory=set)

    def __post_init__(self):
        if not self._deps:
            self._deps = set()

    @property
    def input_shapes(self) -> List[tuple]:
        """Get input tensor shapes."""
        return [t.shape for t in self.inputs]

    @property
    def output_shapes(self) -> List[tuple]:
        """Get output tensor shapes."""
        return [t.shape for t in self.outputs]

    def __repr__(self) -> str:
        in_shapes = [str(s) for s in self.input_shapes]
        out_shapes = [str(s) for s in self.output_shapes]
        return f"OpNode({self.op_type.value}, inputs={in_shapes}, outputs={out_shapes})"


class OpGraph:
    """
    Computational graph built during tracing.

    Represents the DAG of operations to be compiled to DFX IR.

    Example:
        >>> graph = OpGraph()
        >>> # Operations are added during tracing via Tensor operations
        >>> for node in graph.topological_order():
        ...     print(node)
    """

    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: List[OpNode] = []
        self.inputs: List['Tensor'] = []
        self.outputs: List['Tensor'] = []

        # Internal tracking
        self._tensor_to_producer: Dict[int, int] = {}  # tensor_id -> node_id
        self._next_id = 0

    def add_op(self,
               op_type: OpType,
               inputs: List['Tensor'],
               outputs: List['Tensor'],
               name: str = "",
               **attrs) -> OpNode:
        """
        Add an operation to the graph.

        Args:
            op_type: Type of operation
            inputs: Input tensors
            outputs: Output tensors
            name: Optional name for the operation
            **attrs: Additional attributes (e.g., axis for reduction)

        Returns:
            The created OpNode
        """
        node = OpNode(
            op_type=op_type,
            inputs=list(inputs),
            outputs=list(outputs),
            attrs=attrs,
            name=name,
            _id=self._next_id
        )
        self._next_id += 1

        # Track dependencies based on input tensor producers
        for inp in inputs:
            producer_id = self._tensor_to_producer.get(id(inp))
            if producer_id is not None:
                node._deps.add(producer_id)

        # Track this node as producer of its output tensors
        for out in outputs:
            self._tensor_to_producer[id(out)] = node._id

        self.nodes.append(node)
        return node

    def mark_input(self, tensor: 'Tensor'):
        """Mark a tensor as a graph input."""
        if tensor not in self.inputs:
            self.inputs.append(tensor)

    def mark_output(self, tensor: 'Tensor'):
        """Mark a tensor as a graph output."""
        if tensor not in self.outputs:
            self.outputs.append(tensor)

    def topological_order(self) -> List[OpNode]:
        """
        Return nodes in topological order (dependencies before dependents).

        Returns:
            List of OpNodes in execution order
        """
        visited = set()
        order = []

        def visit(node_id: int):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            for dep_id in node._deps:
                visit(dep_id)
            order.append(node)

        for node in self.nodes:
            visit(node._id)

        return order

    def get_node_by_id(self, node_id: int) -> Optional[OpNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node._id == node_id:
                return node
        return None

    def compute_stats(self) -> Dict[str, Any]:
        """
        Compute statistics about the graph.

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'num_nodes': len(self.nodes),
            'num_inputs': len(self.inputs),
            'num_outputs': len(self.outputs),
            'op_counts': {},
            'total_matmul_flops': 0,
        }

        for node in self.nodes:
            op_name = node.op_type.value
            stats['op_counts'][op_name] = stats['op_counts'].get(op_name, 0) + 1

            # Estimate FLOPs for matmul
            if node.op_type == OpType.MATMUL:
                A_shape, B_shape = node.input_shapes
                M, K = A_shape[-2], A_shape[-1]
                _, N = B_shape[-2], B_shape[-1]
                flops = 2 * M * N * K  # multiply-add
                stats['total_matmul_flops'] += flops

        return stats

    def validate(self) -> List[str]:
        """
        Validate the graph for correctness.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check for cycles (would cause infinite loop in topological sort)
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: int) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            node = self.nodes[node_id]
            for dep_id in node._deps:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node in self.nodes:
            if node._id not in visited:
                if has_cycle(node._id):
                    errors.append("Graph contains a cycle")
                    break

        # Check for dangling tensors (outputs not produced by any node)
        for output in self.outputs:
            if id(output) not in self._tensor_to_producer:
                errors.append(f"Output tensor {output.name} is not produced by any operation")

        return errors

    def summary(self) -> str:
        """
        Generate a human-readable summary of the graph.

        Returns:
            Summary string
        """
        lines = [
            f"OpGraph '{self.name}'",
            f"  Inputs: {len(self.inputs)}",
            f"  Outputs: {len(self.outputs)}",
            f"  Operations: {len(self.nodes)}",
            "",
            "  Operation sequence:",
        ]

        for node in self.topological_order():
            in_str = ", ".join(str(s) for s in node.input_shapes)
            out_str = ", ".join(str(s) for s in node.output_shapes)
            name = f" '{node.name}'" if node.name else ""
            lines.append(f"    [{node._id}] {node.op_type.value}{name}: ({in_str}) -> ({out_str})")

        stats = self.compute_stats()
        lines.extend([
            "",
            "  Statistics:",
            f"    Total MatMul FLOPs: {stats['total_matmul_flops']:,}",
        ])

        for op_name, count in sorted(stats['op_counts'].items()):
            lines.append(f"    {op_name}: {count}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"OpGraph(name='{self.name}', nodes={len(self.nodes)}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"

    def __len__(self) -> int:
        return len(self.nodes)
