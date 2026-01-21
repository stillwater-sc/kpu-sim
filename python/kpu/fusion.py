# python/kpu/fusion.py
"""
Kernel Fusion for KPU.

Provides pattern detection and graph rewriting to fuse common DNN patterns
like MatMul+Bias+Activation into single fused operations, reducing memory
traffic by eliminating intermediate tensor reads/writes.

Target Patterns:
    - MatMul + Bias + ReLU  -> FUSED_MATMUL_BIAS_RELU
    - MatMul + Bias + GELU  -> FUSED_MATMUL_BIAS_GELU
    - MatMul + Bias + SiLU  -> FUSED_MATMUL_BIAS_SILU
    - MatMul + ReLU         -> FUSED_MATMUL_RELU

Example:
    >>> from kpu.fusion import FusionCompiler
    >>> compiler = FusionCompiler()
    >>> optimized_graph = compiler.optimize(graph)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import OpGraph, OpNode, OpType
    from .tensor import Tensor


@dataclass
class FusionGroup:
    """
    A group of nodes that can be fused together.

    Attributes:
        nodes: List of OpNodes to be fused (in topological order)
        pattern_name: Name of the fusion pattern that matched
        fused_op_type: The resulting fused OpType
        external_inputs: Inputs to the fused op (excluding intermediates)
        output: The output tensor of the fused op
    """
    nodes: List['OpNode']
    pattern_name: str
    fused_op_type: 'OpType'
    external_inputs: List['Tensor'] = field(default_factory=list)
    output: Optional['Tensor'] = None


class FusionPattern(ABC):
    """
    Base class for fusion patterns.

    Subclasses implement pattern matching and fusion logic for specific
    operator sequences (e.g., MatMul+Bias+ReLU).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this fusion pattern."""
        pass

    @abstractmethod
    def match(self, graph: 'OpGraph', node: 'OpNode') -> Optional[FusionGroup]:
        """
        Try to match this pattern starting at the given node.

        Args:
            graph: The operation graph
            node: The starting node (typically a MATMUL)

        Returns:
            FusionGroup if pattern matches, None otherwise
        """
        pass

    @abstractmethod
    def fuse(self, graph: 'OpGraph', group: FusionGroup) -> 'OpNode':
        """
        Create a fused node from the matched group.

        Args:
            graph: The operation graph
            group: The matched fusion group

        Returns:
            The new fused OpNode
        """
        pass


class MatMulBiasActivation(FusionPattern):
    """
    Pattern: MatMul -> Add (bias) -> Activation

    Matches sequences like:
        Y = matmul(X, W)
        Z = Y + bias
        out = relu(Z)  # or gelu, silu, etc.

    Fuses to: FUSED_MATMUL_BIAS_RELU (or GELU, SiLU variant)
    """

    @property
    def name(self) -> str:
        return "MatMulBiasActivation"

    def match(self, graph: 'OpGraph', node: 'OpNode') -> Optional[FusionGroup]:
        """Match MatMul -> Add -> Activation pattern."""
        from .graph import OpType

        # Must start with MATMUL
        if node.op_type != OpType.MATMUL:
            return None

        # Find consumers of this node's output
        matmul_output = node.outputs[0]
        consumers = self._find_consumers(graph, matmul_output)

        # MatMul output must have exactly one consumer
        if len(consumers) != 1:
            return None

        add_node = consumers[0]

        # Consumer must be ADD (bias)
        if add_node.op_type != OpType.ADD:
            return None

        # Check if this is a bias pattern:
        # One input should be matmul output, other should be 1D bias
        bias_tensor = None
        for inp in add_node.inputs:
            if id(inp) != id(matmul_output):
                # Check if broadcast-compatible bias (1D with correct size)
                if len(inp.shape) == 1:
                    # Check shape compatibility
                    matmul_out_cols = matmul_output.shape[-1]
                    if inp.shape[0] == matmul_out_cols:
                        bias_tensor = inp
                        break

        if bias_tensor is None:
            return None

        # Find consumers of ADD
        add_output = add_node.outputs[0]
        add_consumers = self._find_consumers(graph, add_output)

        # ADD output must have exactly one consumer
        if len(add_consumers) != 1:
            return None

        act_node = add_consumers[0]

        # Consumer must be an activation
        supported_activations = {
            OpType.RELU: OpType.FUSED_MATMUL_BIAS_RELU,
            OpType.GELU: OpType.FUSED_MATMUL_BIAS_GELU,
            OpType.SILU: OpType.FUSED_MATMUL_BIAS_SILU,
        }

        if act_node.op_type not in supported_activations:
            return None

        fused_op_type = supported_activations[act_node.op_type]

        # Collect external inputs (A, B from matmul, bias from add)
        external_inputs = list(node.inputs) + [bias_tensor]

        return FusionGroup(
            nodes=[node, add_node, act_node],
            pattern_name=self.name,
            fused_op_type=fused_op_type,
            external_inputs=external_inputs,
            output=act_node.outputs[0],
        )

    def fuse(self, graph: 'OpGraph', group: FusionGroup) -> 'OpNode':
        """Create fused MatMul+Bias+Activation node."""
        from .graph import OpNode

        matmul_node = group.nodes[0]

        # Create fused node
        fused_node = OpNode(
            op_type=group.fused_op_type,
            inputs=group.external_inputs,
            outputs=[group.output],
            attrs={
                **matmul_node.attrs,
                'fused_from': [n.op_type.value for n in group.nodes],
            },
            name=f"fused_{matmul_node.name}" if matmul_node.name else "",
        )

        return fused_node

    def _find_consumers(self, graph: 'OpGraph', tensor: 'Tensor') -> List['OpNode']:
        """Find all nodes that consume the given tensor."""
        consumers = []
        tensor_id = id(tensor)

        for node in graph.nodes:
            for inp in node.inputs:
                if id(inp) == tensor_id:
                    consumers.append(node)
                    break

        return consumers


class MatMulActivation(FusionPattern):
    """
    Pattern: MatMul -> Activation (no bias)

    Matches sequences like:
        Y = matmul(X, W)
        out = relu(Y)

    Fuses to: FUSED_MATMUL_RELU
    """

    @property
    def name(self) -> str:
        return "MatMulActivation"

    def match(self, graph: 'OpGraph', node: 'OpNode') -> Optional[FusionGroup]:
        """Match MatMul -> Activation pattern (no bias)."""
        from .graph import OpType

        # Must start with MATMUL
        if node.op_type != OpType.MATMUL:
            return None

        # Find consumers of this node's output
        matmul_output = node.outputs[0]
        consumers = self._find_consumers(graph, matmul_output)

        # MatMul output must have exactly one consumer
        if len(consumers) != 1:
            return None

        consumer = consumers[0]

        # Consumer must be activation directly (not ADD)
        supported_activations = {
            OpType.RELU: OpType.FUSED_MATMUL_RELU,
        }

        if consumer.op_type not in supported_activations:
            return None

        fused_op_type = supported_activations[consumer.op_type]

        return FusionGroup(
            nodes=[node, consumer],
            pattern_name=self.name,
            fused_op_type=fused_op_type,
            external_inputs=list(node.inputs),
            output=consumer.outputs[0],
        )

    def fuse(self, graph: 'OpGraph', group: FusionGroup) -> 'OpNode':
        """Create fused MatMul+Activation node."""
        from .graph import OpNode

        matmul_node = group.nodes[0]

        fused_node = OpNode(
            op_type=group.fused_op_type,
            inputs=group.external_inputs,
            outputs=[group.output],
            attrs={
                **matmul_node.attrs,
                'fused_from': [n.op_type.value for n in group.nodes],
            },
            name=f"fused_{matmul_node.name}" if matmul_node.name else "",
        )

        return fused_node

    def _find_consumers(self, graph: 'OpGraph', tensor: 'Tensor') -> List['OpNode']:
        """Find all nodes that consume the given tensor."""
        consumers = []
        tensor_id = id(tensor)

        for node in graph.nodes:
            for inp in node.inputs:
                if id(inp) == tensor_id:
                    consumers.append(node)
                    break

        return consumers


# Default patterns in priority order (more specific patterns first)
DEFAULT_PATTERNS: List[FusionPattern] = [
    MatMulBiasActivation(),  # Try 3-op fusion first
    MatMulActivation(),       # Then 2-op fusion
]


class FusionCompiler:
    """
    Compiler pass that detects and fuses common operation patterns.

    Reduces memory traffic by eliminating intermediate tensor writes
    between operations that can be executed together.

    Example:
        >>> compiler = FusionCompiler()
        >>> optimized = compiler.optimize(graph)
        >>> print(f"Fused {compiler.num_fusions} patterns")

    Attributes:
        patterns: List of fusion patterns to detect
        num_fusions: Number of fusions performed in last optimize() call
        fusion_stats: Statistics about fusions performed
    """

    def __init__(self, patterns: Optional[List[FusionPattern]] = None):
        """
        Initialize the FusionCompiler.

        Args:
            patterns: List of fusion patterns to use. If None, uses DEFAULT_PATTERNS.
        """
        self.patterns = patterns if patterns is not None else DEFAULT_PATTERNS
        self.num_fusions = 0
        self.fusion_stats: Dict[str, int] = {}

    def optimize(self, graph: 'OpGraph') -> 'OpGraph':
        """
        Optimize the graph by fusing operations.

        Iterates through nodes and attempts to match each fusion pattern.
        When a pattern matches, the matched nodes are replaced with a
        single fused operation.

        Args:
            graph: The operation graph to optimize

        Returns:
            Optimized graph (may be the same object, mutated)
        """
        from .graph import OpGraph, OpType

        self.num_fusions = 0
        self.fusion_stats.clear()

        # Find all fusible groups
        fusion_groups: List[FusionGroup] = []
        fused_node_ids: Set[int] = set()

        # Process nodes in topological order
        for node in graph.topological_order():
            # Skip if already part of a fusion
            if node._id in fused_node_ids:
                continue

            # Try each pattern
            for pattern in self.patterns:
                group = pattern.match(graph, node)
                if group is not None:
                    # Check no overlap with existing fusions
                    group_ids = {n._id for n in group.nodes}
                    if not group_ids & fused_node_ids:
                        fusion_groups.append(group)
                        fused_node_ids.update(group_ids)

                        # Track stats
                        pattern_name = pattern.name
                        self.fusion_stats[pattern_name] = self.fusion_stats.get(pattern_name, 0) + 1
                        break

        if not fusion_groups:
            return graph  # No fusions possible

        # Rewrite graph with fused operations
        return self._rewrite_graph(graph, fusion_groups)

    def _rewrite_graph(self, graph: 'OpGraph',
                       fusion_groups: List[FusionGroup]) -> 'OpGraph':
        """Rewrite the graph replacing fused patterns with fused nodes."""
        from .graph import OpGraph

        # Get IDs of all nodes being fused
        fused_ids = set()
        for group in fusion_groups:
            fused_ids.update(n._id for n in group.nodes)

        # Build new node list
        new_nodes = []
        new_tensor_to_producer = {}

        # Track which tensors are outputs of fused groups
        fused_outputs: Dict[int, 'Tensor'] = {}  # intermediate tensor id -> final output tensor
        for group in fusion_groups:
            final_output = group.output
            for node in group.nodes[:-1]:  # All but the last (which produces final output)
                for out in node.outputs:
                    fused_outputs[id(out)] = final_output

        # Process: keep non-fused nodes, add fused nodes
        fused_group_by_first_id: Dict[int, FusionGroup] = {}
        for group in fusion_groups:
            fused_group_by_first_id[group.nodes[0]._id] = group

        next_id = 0
        for node in graph.nodes:
            if node._id in fused_ids:
                # Check if this is the first node of a fusion group
                if node._id in fused_group_by_first_id:
                    group = fused_group_by_first_id[node._id]
                    pattern = self._find_pattern(group.pattern_name)
                    fused_node = pattern.fuse(graph, group)
                    fused_node._id = next_id
                    next_id += 1

                    # Update dependencies
                    fused_node._deps = set()
                    for inp in fused_node.inputs:
                        producer_id = new_tensor_to_producer.get(id(inp))
                        if producer_id is not None:
                            fused_node._deps.add(producer_id)

                    # Track outputs
                    for out in fused_node.outputs:
                        new_tensor_to_producer[id(out)] = fused_node._id

                    new_nodes.append(fused_node)
                    self.num_fusions += 1
                # else: skip (intermediate node of fusion group)
            else:
                # Keep this node but update its ID
                node._id = next_id
                next_id += 1

                # Remap dependencies - if depending on fused intermediate, depend on fused node
                new_deps = set()
                for inp in node.inputs:
                    # Check if input was produced by a fused intermediate
                    if id(inp) in fused_outputs:
                        inp = fused_outputs[id(inp)]
                    producer_id = new_tensor_to_producer.get(id(inp))
                    if producer_id is not None:
                        new_deps.add(producer_id)
                node._deps = new_deps

                # Track outputs
                for out in node.outputs:
                    new_tensor_to_producer[id(out)] = node._id

                new_nodes.append(node)

        # Update graph
        graph.nodes = new_nodes
        graph._tensor_to_producer = new_tensor_to_producer
        graph._next_id = next_id

        return graph

    def _find_pattern(self, name: str) -> FusionPattern:
        """Find a pattern by name."""
        for pattern in self.patterns:
            if pattern.name == name:
                return pattern
        raise ValueError(f"Pattern '{name}' not found")

    def summary(self) -> str:
        """Get a summary of the last optimization run."""
        lines = [
            f"FusionCompiler Summary",
            f"  Total fusions: {self.num_fusions}",
            f"  Patterns matched:",
        ]
        for pattern_name, count in sorted(self.fusion_stats.items()):
            lines.append(f"    {pattern_name}: {count}")

        return "\n".join(lines)


def estimate_memory_savings(graph: 'OpGraph', optimized: 'OpGraph') -> Dict[str, Any]:
    """
    Estimate memory traffic savings from fusion.

    Args:
        graph: Original graph
        optimized: Fused graph

    Returns:
        Dict with memory statistics:
            - original_ops: Number of ops in original graph
            - fused_ops: Number of ops after fusion
            - original_memory_passes: Estimated memory passes (reads + writes)
            - fused_memory_passes: Estimated memory passes after fusion
            - reduction_factor: Memory reduction factor (original / fused)
    """
    from .graph import OpType

    # Count intermediate tensors (reads + writes)
    def count_memory_passes(g: 'OpGraph') -> int:
        passes = 0
        for node in g.nodes:
            if node.op_type.is_compute():
                # Each input is a read (unless fused)
                passes += len(node.inputs)
                # Each output is a write
                passes += len(node.outputs)
        return passes

    original_passes = count_memory_passes(graph)
    fused_passes = count_memory_passes(optimized)

    return {
        'original_ops': len(graph.nodes),
        'fused_ops': len(optimized.nodes),
        'original_memory_passes': original_passes,
        'fused_memory_passes': fused_passes,
        'reduction_factor': original_passes / fused_passes if fused_passes > 0 else float('inf'),
    }
