# python/kpu/fusion.py
"""
Kernel Fusion for KPU.

Provides pattern detection and graph rewriting to fuse common DNN patterns
like MatMul+Bias+Activation into single fused operations, reducing memory
traffic by eliminating intermediate tensor reads/writes.

Target Patterns:
    MatMul Patterns:
    - MatMul + Bias + ReLU  -> FUSED_MATMUL_BIAS_RELU
    - MatMul + Bias + GELU  -> FUSED_MATMUL_BIAS_GELU
    - MatMul + Bias + SiLU  -> FUSED_MATMUL_BIAS_SILU
    - MatMul + ReLU         -> FUSED_MATMUL_RELU

    Conv2D Patterns (v0.6.2+):
    - Conv2D + BatchNorm + ReLU -> FUSED_CONV2D_BN_RELU
    - Conv2D + ReLU             -> FUSED_CONV2D_RELU

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


class Conv2DBatchNormActivation(FusionPattern):
    """
    Pattern: Conv2D -> BatchNorm -> Activation (v0.6.2+)

    Matches sequences like:
        Y = conv2d(X, W)
        Z = batch_norm(Y)
        out = relu(Z)

    Fuses to: FUSED_CONV2D_BN_RELU

    This is a very common pattern in CNNs like ResNet, VGG, etc.
    """

    @property
    def name(self) -> str:
        return "Conv2DBatchNormActivation"

    def match(self, graph: 'OpGraph', node: 'OpNode') -> Optional[FusionGroup]:
        """Match Conv2D -> BatchNorm -> Activation pattern."""
        from .graph import OpType

        # Must start with CONV2D
        if node.op_type != OpType.CONV2D:
            return None

        # Find consumers of this node's output
        conv_output = node.outputs[0]
        consumers = self._find_consumers(graph, conv_output)

        # Conv output must have exactly one consumer
        if len(consumers) != 1:
            return None

        bn_node = consumers[0]

        # Consumer must be BATCH_NORM
        if bn_node.op_type != OpType.BATCH_NORM:
            return None

        # Find consumers of BatchNorm
        bn_output = bn_node.outputs[0]
        bn_consumers = self._find_consumers(graph, bn_output)

        # BN output must have exactly one consumer
        if len(bn_consumers) != 1:
            return None

        act_node = bn_consumers[0]

        # Consumer must be ReLU (currently only supported activation for conv fusion)
        if act_node.op_type != OpType.RELU:
            return None

        fused_op_type = OpType.FUSED_CONV2D_BN_RELU

        # Collect external inputs:
        # - Conv inputs (input tensor, weight, optional bias)
        # - BN inputs (gamma, beta, running_mean, running_var) - excluding conv output
        external_inputs = list(node.inputs)

        # Add BN parameters (skip the first input which is conv output)
        for inp in bn_node.inputs:
            if id(inp) != id(conv_output):
                external_inputs.append(inp)

        return FusionGroup(
            nodes=[node, bn_node, act_node],
            pattern_name=self.name,
            fused_op_type=fused_op_type,
            external_inputs=external_inputs,
            output=act_node.outputs[0],
        )

    def fuse(self, graph: 'OpGraph', group: FusionGroup) -> 'OpNode':
        """Create fused Conv2D+BatchNorm+Activation node."""
        from .graph import OpNode

        conv_node = group.nodes[0]

        # Create fused node
        fused_node = OpNode(
            op_type=group.fused_op_type,
            inputs=group.external_inputs,
            outputs=[group.output],
            attrs={
                **conv_node.attrs,
                'fused_from': [n.op_type.value for n in group.nodes],
            },
            name=f"fused_{conv_node.name}" if conv_node.name else "",
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


class Conv2DActivation(FusionPattern):
    """
    Pattern: Conv2D -> Activation (no BatchNorm) (v0.6.2+)

    Matches sequences like:
        Y = conv2d(X, W)
        out = relu(Y)

    Fuses to: FUSED_CONV2D_RELU
    """

    @property
    def name(self) -> str:
        return "Conv2DActivation"

    def match(self, graph: 'OpGraph', node: 'OpNode') -> Optional[FusionGroup]:
        """Match Conv2D -> Activation pattern (no BatchNorm)."""
        from .graph import OpType

        # Must start with CONV2D
        if node.op_type != OpType.CONV2D:
            return None

        # Find consumers of this node's output
        conv_output = node.outputs[0]
        consumers = self._find_consumers(graph, conv_output)

        # Conv output must have exactly one consumer
        if len(consumers) != 1:
            return None

        consumer = consumers[0]

        # Consumer must be activation directly (not BATCH_NORM)
        # Only ReLU is currently supported
        if consumer.op_type != OpType.RELU:
            return None

        fused_op_type = OpType.FUSED_CONV2D_RELU

        return FusionGroup(
            nodes=[node, consumer],
            pattern_name=self.name,
            fused_op_type=fused_op_type,
            external_inputs=list(node.inputs),
            output=consumer.outputs[0],
        )

    def fuse(self, graph: 'OpGraph', group: FusionGroup) -> 'OpNode':
        """Create fused Conv2D+Activation node."""
        from .graph import OpNode

        conv_node = group.nodes[0]

        fused_node = OpNode(
            op_type=group.fused_op_type,
            inputs=group.external_inputs,
            outputs=[group.output],
            attrs={
                **conv_node.attrs,
                'fused_from': [n.op_type.value for n in group.nodes],
            },
            name=f"fused_{conv_node.name}" if conv_node.name else "",
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
    MatMulBiasActivation(),    # Try 3-op MatMul fusion first
    Conv2DBatchNormActivation(),  # Try 3-op Conv2D fusion
    MatMulActivation(),        # Then 2-op MatMul fusion
    Conv2DActivation(),        # Then 2-op Conv2D fusion
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


@dataclass
class RooflineMetrics:
    """
    Roofline model metrics for analyzing memory-bound vs compute-bound workloads.

    The roofline model characterizes performance by comparing:
    - Arithmetic intensity: FLOPs / Bytes (higher = more compute per byte)
    - Ridge point: Where memory bandwidth meets compute throughput

    Attributes:
        total_flops: Total floating point operations
        total_bytes: Total memory traffic (reads + writes)
        arithmetic_intensity: FLOPs per byte
        ridge_point: Hardware-specific intensity where memory = compute
        is_memory_bound: True if workload is memory-limited
        is_compute_bound: True if workload is compute-limited
        efficiency: Percentage of peak performance achievable
        bottleneck: Human-readable description of the bottleneck
    """
    total_flops: int
    total_bytes: int
    arithmetic_intensity: float
    ridge_point: float
    is_memory_bound: bool
    is_compute_bound: bool
    efficiency: float
    bottleneck: str


@dataclass
class FusionOpportunity:
    """
    Describes a detected fusion opportunity in the graph.

    Attributes:
        pattern_name: Name of the fusion pattern that would match
        nodes: List of node IDs that would be fused
        ops: List of operation types involved
        memory_savings: Estimated memory traffic reduction factor
        description: Human-readable description
    """
    pattern_name: str
    nodes: List[int]
    ops: List[str]
    memory_savings: float
    description: str


class FusionAnalyzer:
    """
    Analyzes graphs to detect fusion opportunities and roofline characteristics.

    Unlike FusionCompiler which modifies graphs, FusionAnalyzer only reports
    what could be fused without making changes. Useful for understanding
    optimization potential before committing to changes.

    Example:
        >>> analyzer = FusionAnalyzer()
        >>> report = analyzer.analyze(graph)
        >>> print(f"Found {len(report.opportunities)} fusion opportunities")
        >>> print(f"Workload is {report.roofline.bottleneck}")
    """

    def __init__(self,
                 patterns: Optional[List[FusionPattern]] = None,
                 peak_flops_per_cycle: float = 1024.0,
                 peak_bytes_per_cycle: float = 64.0):
        """
        Initialize the FusionAnalyzer.

        Args:
            patterns: Fusion patterns to detect. If None, uses DEFAULT_PATTERNS.
            peak_flops_per_cycle: Hardware peak compute throughput (FLOPs/cycle)
            peak_bytes_per_cycle: Hardware peak memory bandwidth (bytes/cycle)
        """
        self.patterns = patterns if patterns is not None else DEFAULT_PATTERNS
        self.peak_flops_per_cycle = peak_flops_per_cycle
        self.peak_bytes_per_cycle = peak_bytes_per_cycle
        # Ridge point: where compute time = memory time
        # FLOPs/cycle * T = Bytes/cycle * T => FLOPs/Bytes = Bytes_BW / FLOPs_throughput
        self.ridge_point = peak_flops_per_cycle / peak_bytes_per_cycle

    def analyze(self, graph: 'OpGraph') -> 'FusionReport':
        """
        Analyze a graph for fusion opportunities and performance characteristics.

        Args:
            graph: The operation graph to analyze

        Returns:
            FusionReport with opportunities and roofline analysis
        """
        opportunities = self._find_opportunities(graph)
        roofline_unfused = self._compute_roofline(graph, fused=False)

        # Estimate fused roofline
        fused_bytes = roofline_unfused.total_bytes
        for opp in opportunities:
            # Each fusion eliminates intermediate tensor traffic
            # Estimate: each 3-op fusion saves ~2 tensor read/writes
            fused_bytes -= int(fused_bytes * (opp.memory_savings - 1) / opp.memory_savings / len(opportunities))

        roofline_fused = self._compute_roofline_from_values(
            roofline_unfused.total_flops,
            max(fused_bytes, roofline_unfused.total_bytes // 3)  # Conservative estimate
        )

        return FusionReport(
            opportunities=opportunities,
            roofline_unfused=roofline_unfused,
            roofline_fused=roofline_fused,
            total_fusions_possible=len(opportunities),
            estimated_speedup=roofline_fused.efficiency / roofline_unfused.efficiency if roofline_unfused.efficiency > 0 else 1.0,
        )

    def _find_opportunities(self, graph: 'OpGraph') -> List[FusionOpportunity]:
        """Find all fusion opportunities in the graph."""
        opportunities = []
        seen_node_ids: Set[int] = set()

        for node in graph.topological_order():
            if node._id in seen_node_ids:
                continue

            for pattern in self.patterns:
                group = pattern.match(graph, node)
                if group is not None:
                    group_ids = {n._id for n in group.nodes}
                    if not group_ids & seen_node_ids:
                        seen_node_ids.update(group_ids)

                        # Estimate memory savings based on ops eliminated
                        num_ops = len(group.nodes)
                        memory_savings = num_ops / 1.0  # Fused = 1 memory pass

                        opportunities.append(FusionOpportunity(
                            pattern_name=pattern.name,
                            nodes=list(group_ids),
                            ops=[n.op_type.value for n in group.nodes],
                            memory_savings=memory_savings,
                            description=f"{' -> '.join(n.op_type.value for n in group.nodes)} => {group.fused_op_type.value}",
                        ))
                        break

        return opportunities

    def _compute_roofline(self, graph: 'OpGraph', fused: bool = False) -> RooflineMetrics:
        """Compute roofline metrics for a graph."""
        from .graph import OpType

        total_flops = 0
        total_bytes = 0

        for node in graph.nodes:
            # Compute FLOPs
            if node.op_type == OpType.MATMUL or node.op_type.is_fused():
                if len(node.inputs) >= 2:
                    A_shape = node.inputs[0].shape
                    B_shape = node.inputs[1].shape
                    if len(A_shape) >= 2 and len(B_shape) >= 2:
                        M, K = A_shape[-2], A_shape[-1]
                        N = B_shape[-1]
                        total_flops += 2 * M * N * K

            # Compute memory traffic (bytes)
            # Each tensor read/write = size * 4 bytes (float32)
            for inp in node.inputs:
                size = 1
                for dim in inp.shape:
                    size *= dim
                total_bytes += size * 4  # Read

            for out in node.outputs:
                size = 1
                for dim in out.shape:
                    size *= dim
                total_bytes += size * 4  # Write

        return self._compute_roofline_from_values(total_flops, total_bytes)

    def _compute_roofline_from_values(self, total_flops: int, total_bytes: int) -> RooflineMetrics:
        """Compute roofline metrics from raw FLOP and byte counts."""
        if total_bytes == 0:
            arithmetic_intensity = float('inf')
        else:
            arithmetic_intensity = total_flops / total_bytes

        is_memory_bound = arithmetic_intensity < self.ridge_point
        is_compute_bound = not is_memory_bound

        # Efficiency calculation
        if is_memory_bound:
            # Memory-bound: limited by bandwidth
            # Achievable FLOPs = AI * bandwidth
            achievable_flops_per_cycle = arithmetic_intensity * self.peak_bytes_per_cycle
            efficiency = (achievable_flops_per_cycle / self.peak_flops_per_cycle) * 100
            bottleneck = "memory bandwidth"
        else:
            # Compute-bound: can achieve peak
            efficiency = 100.0
            bottleneck = "compute throughput"

        return RooflineMetrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            arithmetic_intensity=arithmetic_intensity,
            ridge_point=self.ridge_point,
            is_memory_bound=is_memory_bound,
            is_compute_bound=is_compute_bound,
            efficiency=min(efficiency, 100.0),
            bottleneck=bottleneck,
        )


@dataclass
class FusionReport:
    """
    Complete analysis report for fusion opportunities.

    Attributes:
        opportunities: List of detected fusion opportunities
        roofline_unfused: Roofline analysis without fusion
        roofline_fused: Estimated roofline analysis with fusion
        total_fusions_possible: Number of fusions that can be applied
        estimated_speedup: Estimated speedup from applying all fusions
    """
    opportunities: List[FusionOpportunity]
    roofline_unfused: RooflineMetrics
    roofline_fused: RooflineMetrics
    total_fusions_possible: int
    estimated_speedup: float

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "Fusion Analysis Report",
            "=" * 50,
            "",
            f"Fusion Opportunities: {self.total_fusions_possible}",
        ]

        if self.opportunities:
            lines.append("")
            for i, opp in enumerate(self.opportunities, 1):
                lines.append(f"  {i}. {opp.description}")
                lines.append(f"     Pattern: {opp.pattern_name}, Savings: {opp.memory_savings:.1f}x")

        lines.extend([
            "",
            "Roofline Analysis (Unfused):",
            f"  Total FLOPs:            {self.roofline_unfused.total_flops:,}",
            f"  Total Memory Traffic:   {self.roofline_unfused.total_bytes:,} bytes",
            f"  Arithmetic Intensity:   {self.roofline_unfused.arithmetic_intensity:.2f} FLOPs/byte",
            f"  Ridge Point:            {self.roofline_unfused.ridge_point:.2f} FLOPs/byte",
            f"  Bottleneck:             {self.roofline_unfused.bottleneck}",
            f"  Efficiency:             {self.roofline_unfused.efficiency:.1f}%",
            "",
            "Roofline Analysis (Fused):",
            f"  Total FLOPs:            {self.roofline_fused.total_flops:,}",
            f"  Total Memory Traffic:   {self.roofline_fused.total_bytes:,} bytes",
            f"  Arithmetic Intensity:   {self.roofline_fused.arithmetic_intensity:.2f} FLOPs/byte",
            f"  Bottleneck:             {self.roofline_fused.bottleneck}",
            f"  Efficiency:             {self.roofline_fused.efficiency:.1f}%",
            "",
            f"Estimated Speedup:        {self.estimated_speedup:.2f}x",
        ])

        return "\n".join(lines)


def analyze_fusion_potential(graph: 'OpGraph',
                            peak_flops_per_cycle: float = 1024.0,
                            peak_bytes_per_cycle: float = 64.0) -> FusionReport:
    """
    Convenience function to analyze a graph's fusion potential.

    Args:
        graph: The operation graph to analyze
        peak_flops_per_cycle: Hardware peak compute throughput
        peak_bytes_per_cycle: Hardware peak memory bandwidth

    Returns:
        FusionReport with complete analysis

    Example:
        >>> report = analyze_fusion_potential(graph)
        >>> print(report.summary())
        >>> if report.roofline_unfused.is_memory_bound:
        ...     print("Unfused workload is memory-bound, fusion will help!")
    """
    analyzer = FusionAnalyzer(
        peak_flops_per_cycle=peak_flops_per_cycle,
        peak_bytes_per_cycle=peak_bytes_per_cycle,
    )
    return analyzer.analyze(graph)
