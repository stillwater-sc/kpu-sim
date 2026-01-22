# python/kpu/memory_planner.py
"""
KPU Memory Planner - Optimal buffer allocation for model execution.

Provides:
- Tensor lifetime analysis
- Memory pool allocation
- In-place operation detection
- Buffer reuse optimization

Memory Hierarchy:
    DRAM (External) -> L3 -> L2 -> L1 -> Registers -> Compute

Example:
    >>> from kpu.memory_planner import MemoryPlanner
    >>> from kpu.model import Model
    >>>
    >>> model = create_model()
    >>> planner = MemoryPlanner(model)
    >>> plan = planner.plan(input_shape=(1, 3, 224, 224))
    >>>
    >>> print(f"Peak memory: {plan.peak_memory_bytes / 1e6:.1f} MB")
    >>> print(f"Buffer reuse: {plan.reuse_factor:.1f}x")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .tensor import Tensor
from .graph import OpGraph, OpNode, OpType


# =============================================================================
# Memory Level Enum
# =============================================================================

class MemoryLevel(Enum):
    """Memory hierarchy levels in KPU."""
    DRAM = 0      # External memory (HBM/DDR)
    L3 = 1        # L3 buffer (shared across tiles)
    L2 = 2        # L2 buffer (per tile)
    L1 = 3        # L1 stream buffer (per compute unit)
    REGISTER = 4  # Register file

    @property
    def capacity_bytes(self) -> int:
        """Default capacity for each memory level."""
        capacities = {
            MemoryLevel.DRAM: 16 * 1024 * 1024 * 1024,  # 16 GB
            MemoryLevel.L3: 8 * 1024 * 1024,             # 8 MB
            MemoryLevel.L2: 256 * 1024,                   # 256 KB
            MemoryLevel.L1: 16 * 1024,                    # 16 KB
            MemoryLevel.REGISTER: 4 * 1024,               # 4 KB
        }
        return capacities[self]

    @property
    def bandwidth_bytes_per_cycle(self) -> int:
        """Default bandwidth for each memory level."""
        bandwidths = {
            MemoryLevel.DRAM: 64,       # 64 B/cycle
            MemoryLevel.L3: 256,        # 256 B/cycle
            MemoryLevel.L2: 512,        # 512 B/cycle
            MemoryLevel.L1: 1024,       # 1 KB/cycle
            MemoryLevel.REGISTER: 4096, # 4 KB/cycle
        }
        return bandwidths[self]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TensorAllocation:
    """Memory allocation for a tensor."""
    tensor_id: int
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    memory_level: MemoryLevel
    offset: int  # Offset within memory pool
    lifetime_start: int  # First op that uses this tensor
    lifetime_end: int    # Last op that uses this tensor
    is_input: bool = False
    is_output: bool = False
    is_weight: bool = False
    reuses_buffer: Optional[int] = None  # tensor_id of buffer being reused

    @property
    def is_alive_at(self) -> callable:
        """Check if tensor is alive at a given op index."""
        def check(op_idx: int) -> bool:
            return self.lifetime_start <= op_idx <= self.lifetime_end
        return check


@dataclass
class MemoryPool:
    """Memory pool at a specific level."""
    level: MemoryLevel
    capacity_bytes: int
    allocations: List[TensorAllocation] = field(default_factory=list)
    peak_usage_bytes: int = 0

    def allocate(self, tensor: TensorAllocation) -> int:
        """Allocate tensor in this pool.

        Returns:
            Offset within pool
        """
        # Simple first-fit allocation
        # Find first gap that fits
        sorted_allocs = sorted(self.allocations, key=lambda a: a.offset)

        offset = 0
        for alloc in sorted_allocs:
            # Check if tensor's lifetime overlaps with existing allocation
            if (tensor.lifetime_start <= alloc.lifetime_end and
                tensor.lifetime_end >= alloc.lifetime_start):
                # Lifetimes overlap, can't reuse this space
                offset = max(offset, alloc.offset + alloc.size_bytes)
            else:
                # Lifetimes don't overlap, could reuse
                # But for simplicity, still try to find a gap
                if alloc.offset >= offset + tensor.size_bytes:
                    break
                offset = alloc.offset + alloc.size_bytes

        tensor.offset = offset
        self.allocations.append(tensor)
        self.peak_usage_bytes = max(self.peak_usage_bytes, offset + tensor.size_bytes)

        return offset

    def can_fit(self, size_bytes: int) -> bool:
        """Check if tensor can fit in pool."""
        return self.peak_usage_bytes + size_bytes <= self.capacity_bytes


@dataclass
class MemoryPlan:
    """Complete memory allocation plan for a model."""
    tensor_allocations: Dict[int, TensorAllocation] = field(default_factory=dict)
    memory_pools: Dict[MemoryLevel, MemoryPool] = field(default_factory=dict)
    op_schedule: List[int] = field(default_factory=list)  # Order of op execution

    # Statistics
    total_tensor_bytes: int = 0
    peak_memory_bytes: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0
    reused_bytes: int = 0

    @property
    def reuse_factor(self) -> float:
        """Memory reuse factor (total / peak)."""
        if self.peak_memory_bytes == 0:
            return 1.0
        return self.total_tensor_bytes / self.peak_memory_bytes

    @property
    def efficiency(self) -> float:
        """Memory efficiency (needed / allocated)."""
        total_allocated = sum(pool.peak_usage_bytes for pool in self.memory_pools.values())
        if total_allocated == 0:
            return 1.0
        return self.total_tensor_bytes / total_allocated

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Memory Plan Summary",
            "=" * 50,
            f"Total tensor memory: {self.total_tensor_bytes / 1e6:.2f} MB",
            f"Peak memory usage: {self.peak_memory_bytes / 1e6:.2f} MB",
            f"Weight memory: {self.weight_bytes / 1e6:.2f} MB",
            f"Activation memory: {self.activation_bytes / 1e6:.2f} MB",
            f"Reused memory: {self.reused_bytes / 1e6:.2f} MB",
            f"Reuse factor: {self.reuse_factor:.2f}x",
            f"Memory efficiency: {self.efficiency * 100:.1f}%",
            "",
            "Per-Level Breakdown:",
            "-" * 50,
        ]

        for level, pool in sorted(self.memory_pools.items(), key=lambda x: x[0].value):
            lines.append(f"  {level.name}:")
            lines.append(f"    Capacity: {pool.capacity_bytes / 1e6:.2f} MB")
            lines.append(f"    Peak usage: {pool.peak_usage_bytes / 1e6:.2f} MB")
            lines.append(f"    Utilization: {100 * pool.peak_usage_bytes / pool.capacity_bytes:.1f}%")
            lines.append(f"    Allocations: {len(pool.allocations)}")

        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# Memory Planner
# =============================================================================

class MemoryPlanner:
    """Memory planner for optimal buffer allocation.

    Analyzes tensor lifetimes and plans memory allocation to minimize
    peak memory usage through buffer reuse.

    Args:
        model: Model to plan memory for
        memory_config: Optional configuration for memory hierarchy

    Example:
        >>> planner = MemoryPlanner(model)
        >>> plan = planner.plan(input_shape=(1, 3, 224, 224))
        >>> print(plan.summary())
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        memory_config: Optional[Dict[MemoryLevel, int]] = None,
    ):
        self.model = model
        self.memory_config = memory_config or {}

        # Default memory capacities
        self.level_capacities = {
            MemoryLevel.DRAM: self.memory_config.get(MemoryLevel.DRAM, 16 * 1024 * 1024 * 1024),
            MemoryLevel.L3: self.memory_config.get(MemoryLevel.L3, 8 * 1024 * 1024),
            MemoryLevel.L2: self.memory_config.get(MemoryLevel.L2, 256 * 1024),
            MemoryLevel.L1: self.memory_config.get(MemoryLevel.L1, 16 * 1024),
        }

    def plan(
        self,
        input_shape: Tuple[int, ...],
        dtype: str = "float32",
    ) -> MemoryPlan:
        """Create memory plan for model execution.

        Args:
            input_shape: Input tensor shape
            dtype: Data type

        Returns:
            Memory allocation plan
        """
        plan = MemoryPlan()

        # Initialize memory pools
        for level in MemoryLevel:
            if level != MemoryLevel.REGISTER:
                plan.memory_pools[level] = MemoryPool(
                    level=level,
                    capacity_bytes=self.level_capacities.get(level, level.capacity_bytes),
                )

        if self.model is None:
            return plan

        # Analyze model to get tensor allocations
        tensors = self._analyze_model(input_shape, dtype)

        # Assign memory levels and allocate
        for tensor in tensors:
            # Determine memory level based on tensor type and size
            level = self._assign_memory_level(tensor)
            tensor.memory_level = level

            # Allocate in appropriate pool
            pool = plan.memory_pools[level]
            pool.allocate(tensor)

            plan.tensor_allocations[tensor.tensor_id] = tensor

            # Update statistics
            plan.total_tensor_bytes += tensor.size_bytes
            if tensor.is_weight:
                plan.weight_bytes += tensor.size_bytes
            else:
                plan.activation_bytes += tensor.size_bytes

        # Calculate peak memory
        plan.peak_memory_bytes = sum(
            pool.peak_usage_bytes for pool in plan.memory_pools.values()
        )

        return plan

    def _analyze_model(
        self,
        input_shape: Tuple[int, ...],
        dtype: str,
    ) -> List[TensorAllocation]:
        """Analyze model to determine tensor allocations.

        Args:
            input_shape: Input tensor shape
            dtype: Data type

        Returns:
            List of tensor allocations
        """
        tensors = []
        tensor_id = 0

        dtype_bytes = {"float32": 4, "float16": 2, "int8": 1, "int4": 0.5}
        bytes_per_elem = dtype_bytes.get(dtype, 4)

        # Add input tensor
        input_size = int(np.prod(input_shape) * bytes_per_elem)
        tensors.append(TensorAllocation(
            tensor_id=tensor_id,
            name="input",
            shape=input_shape,
            dtype=dtype,
            size_bytes=input_size,
            memory_level=MemoryLevel.DRAM,
            offset=0,
            lifetime_start=0,
            lifetime_end=0,
            is_input=True,
        ))
        tensor_id += 1

        # Analyze each layer
        if hasattr(self.model, 'modules'):
            current_shape = input_shape
            op_idx = 1

            for name, layer in self.model.modules():
                if name == "":
                    continue

                # Estimate output shape
                output_shape = self._estimate_output_shape(layer, current_shape)

                # Add weight tensors
                for param_name, param in layer._parameters.items() if hasattr(layer, '_parameters') else []:
                    weight_size = int(param.numel() * bytes_per_elem)
                    tensors.append(TensorAllocation(
                        tensor_id=tensor_id,
                        name=f"{name}.{param_name}",
                        shape=param.shape,
                        dtype=dtype,
                        size_bytes=weight_size,
                        memory_level=MemoryLevel.DRAM,
                        offset=0,
                        lifetime_start=0,
                        lifetime_end=len(list(self.model.modules())) - 1,  # Weights live forever
                        is_weight=True,
                    ))
                    tensor_id += 1

                # Add output tensor (activation)
                output_size = int(np.prod(output_shape) * bytes_per_elem)
                tensors.append(TensorAllocation(
                    tensor_id=tensor_id,
                    name=f"{name}_output",
                    shape=output_shape,
                    dtype=dtype,
                    size_bytes=output_size,
                    memory_level=MemoryLevel.L3,
                    offset=0,
                    lifetime_start=op_idx,
                    lifetime_end=op_idx + 1,  # Used by next layer
                ))
                tensor_id += 1
                op_idx += 1
                current_shape = output_shape

            # Mark last output as model output
            if tensors:
                tensors[-1].is_output = True
                tensors[-1].lifetime_end = op_idx

        return tensors

    def _estimate_output_shape(
        self,
        layer: Any,
        input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Estimate layer output shape.

        Args:
            layer: Layer instance
            input_shape: Input shape

        Returns:
            Estimated output shape
        """
        from .model import Linear, Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Flatten

        if isinstance(layer, Linear):
            # (batch, in_features) -> (batch, out_features)
            return input_shape[:-1] + (layer.out_features,)

        elif isinstance(layer, Conv2d):
            # (N, C, H, W) -> (N, out_channels, H', W')
            N = input_shape[0]
            H, W = input_shape[2], input_shape[3]
            H_out = (H + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
            W_out = (W + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
            return (N, layer.out_channels, H_out, W_out)

        elif isinstance(layer, (MaxPool2d, AvgPool2d)):
            N, C = input_shape[0], input_shape[1]
            H, W = input_shape[2], input_shape[3]
            H_out = (H + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
            W_out = (W + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
            return (N, C, H_out, W_out)

        elif isinstance(layer, AdaptiveAvgPool2d):
            N, C = input_shape[0], input_shape[1]
            return (N, C) + layer.output_size

        elif isinstance(layer, Flatten):
            N = input_shape[0]
            flattened = int(np.prod(input_shape[1:]))
            return (N, flattened)

        # Default: preserve shape (activation layers, etc.)
        return input_shape

    def _assign_memory_level(self, tensor: TensorAllocation) -> MemoryLevel:
        """Assign optimal memory level for a tensor.

        Args:
            tensor: Tensor allocation

        Returns:
            Assigned memory level
        """
        # Weights always in DRAM (loaded into L3/L2 as needed)
        if tensor.is_weight:
            return MemoryLevel.DRAM

        # Model inputs/outputs in DRAM
        if tensor.is_input or tensor.is_output:
            return MemoryLevel.DRAM

        # Activations: try to fit in L3, fall back to DRAM
        if tensor.size_bytes <= self.level_capacities[MemoryLevel.L3]:
            return MemoryLevel.L3
        else:
            return MemoryLevel.DRAM

    def plan_from_graph(
        self,
        graph: OpGraph,
        input_shapes: Dict[str, Tuple[int, ...]],
    ) -> MemoryPlan:
        """Create memory plan from operation graph.

        Args:
            graph: Operation graph
            input_shapes: Dictionary mapping input names to shapes

        Returns:
            Memory allocation plan
        """
        plan = MemoryPlan()

        # Initialize memory pools
        for level in MemoryLevel:
            if level != MemoryLevel.REGISTER:
                plan.memory_pools[level] = MemoryPool(
                    level=level,
                    capacity_bytes=self.level_capacities.get(level, level.capacity_bytes),
                )

        # Analyze graph for tensor lifetimes
        tensor_lifetimes = self._analyze_graph_lifetimes(graph)

        # Create allocations
        tensor_id = 0
        for tensor_name, (first_use, last_use, shape, dtype) in tensor_lifetimes.items():
            dtype_bytes = {"float32": 4, "float16": 2, "int8": 1}
            size = int(np.prod(shape) * dtype_bytes.get(dtype, 4))

            is_input = tensor_name in graph.inputs
            is_output = tensor_name in graph.outputs

            tensor = TensorAllocation(
                tensor_id=tensor_id,
                name=tensor_name,
                shape=shape,
                dtype=dtype,
                size_bytes=size,
                memory_level=MemoryLevel.DRAM,
                offset=0,
                lifetime_start=first_use,
                lifetime_end=last_use,
                is_input=is_input,
                is_output=is_output,
            )

            # Assign level and allocate
            tensor.memory_level = self._assign_memory_level(tensor)
            pool = plan.memory_pools[tensor.memory_level]
            pool.allocate(tensor)

            plan.tensor_allocations[tensor_id] = tensor
            plan.total_tensor_bytes += size

            tensor_id += 1

        # Calculate peak
        plan.peak_memory_bytes = sum(
            pool.peak_usage_bytes for pool in plan.memory_pools.values()
        )

        return plan

    def _analyze_graph_lifetimes(
        self,
        graph: OpGraph,
    ) -> Dict[str, Tuple[int, int, Tuple[int, ...], str]]:
        """Analyze tensor lifetimes in operation graph.

        Args:
            graph: Operation graph

        Returns:
            Dictionary mapping tensor names to (first_use, last_use, shape, dtype)
        """
        lifetimes = {}

        # Get topological order
        topo_order = graph.topological_order()

        for op_idx, node in enumerate(topo_order):
            # Record input tensor usage
            for inp in node.inputs:
                if isinstance(inp, str):  # Tensor reference
                    if inp not in lifetimes:
                        lifetimes[inp] = (op_idx, op_idx, (), "float32")
                    else:
                        first, _, shape, dtype = lifetimes[inp]
                        lifetimes[inp] = (first, op_idx, shape, dtype)

            # Record output tensor creation
            for out in node.outputs:
                if isinstance(out, str):
                    # Get shape from node attributes if available
                    shape = node.attrs.get("output_shape", ())
                    dtype = node.attrs.get("dtype", "float32")
                    lifetimes[out] = (op_idx, op_idx, shape, dtype)

        return lifetimes

    def optimize(
        self,
        plan: MemoryPlan,
        strategy: str = "linear_scan",
    ) -> MemoryPlan:
        """Optimize memory plan using specified strategy.

        Args:
            plan: Initial memory plan
            strategy: Optimization strategy ("linear_scan", "graph_coloring")

        Returns:
            Optimized memory plan
        """
        if strategy == "linear_scan":
            return self._optimize_linear_scan(plan)
        elif strategy == "graph_coloring":
            return self._optimize_graph_coloring(plan)
        else:
            return plan

    def _optimize_linear_scan(self, plan: MemoryPlan) -> MemoryPlan:
        """Optimize using linear scan register allocation.

        Finds opportunities for buffer reuse based on non-overlapping lifetimes.
        """
        # Sort tensors by lifetime start
        tensors = sorted(
            plan.tensor_allocations.values(),
            key=lambda t: t.lifetime_start
        )

        # Track active tensors and their buffer assignments
        active = []  # (end_time, tensor_id, buffer_offset)
        buffer_pool = []  # List of (size, offset) for reusable buffers

        optimized = MemoryPlan()
        optimized.memory_pools = plan.memory_pools

        for tensor in tensors:
            # Remove expired tensors
            active = [(end, tid, offset) for end, tid, offset in active
                     if end >= tensor.lifetime_start]

            # Find reusable buffer
            reused = None
            for i, (buf_size, buf_offset) in enumerate(buffer_pool):
                # Check if any active tensor uses this buffer
                buffer_in_use = any(offset == buf_offset for _, _, offset in active)
                if not buffer_in_use and buf_size >= tensor.size_bytes:
                    reused = (i, buf_offset)
                    tensor.reuses_buffer = buf_offset
                    plan.reused_bytes += tensor.size_bytes
                    break

            if reused:
                tensor.offset = reused[1]
            else:
                # Allocate new buffer
                if buffer_pool:
                    new_offset = max(off + size for size, off in buffer_pool)
                else:
                    new_offset = 0
                tensor.offset = new_offset
                buffer_pool.append((tensor.size_bytes, new_offset))

            active.append((tensor.lifetime_end, tensor.tensor_id, tensor.offset))
            optimized.tensor_allocations[tensor.tensor_id] = tensor

        # Recalculate peak
        optimized.total_tensor_bytes = plan.total_tensor_bytes
        optimized.weight_bytes = plan.weight_bytes
        optimized.activation_bytes = plan.activation_bytes
        optimized.reused_bytes = plan.reused_bytes

        if buffer_pool:
            optimized.peak_memory_bytes = max(off + size for size, off in buffer_pool)
        else:
            optimized.peak_memory_bytes = 0

        return optimized

    def _optimize_graph_coloring(self, plan: MemoryPlan) -> MemoryPlan:
        """Optimize using graph coloring (interference graph).

        More optimal but slower than linear scan.
        """
        # Build interference graph
        tensors = list(plan.tensor_allocations.values())
        n = len(tensors)

        # Adjacency matrix for interference
        interference = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                # Tensors interfere if lifetimes overlap
                ti, tj = tensors[i], tensors[j]
                if (ti.lifetime_start <= tj.lifetime_end and
                    ti.lifetime_end >= tj.lifetime_start):
                    interference[i, j] = True
                    interference[j, i] = True

        # Greedy coloring
        colors = [-1] * n
        color_sizes = {}  # color -> required size

        for i in range(n):
            # Find used colors by neighbors
            used = set()
            for j in range(n):
                if interference[i, j] and colors[j] >= 0:
                    used.add(colors[j])

            # Find smallest available color with sufficient size
            color = 0
            while color in used:
                color += 1

            colors[i] = color
            color_sizes[color] = max(color_sizes.get(color, 0), tensors[i].size_bytes)

        # Assign offsets based on colors
        color_offsets = {}
        offset = 0
        for color in sorted(color_sizes.keys()):
            color_offsets[color] = offset
            offset += color_sizes[color]

        # Update tensor allocations
        optimized = MemoryPlan()
        optimized.memory_pools = plan.memory_pools

        for i, tensor in enumerate(tensors):
            tensor.offset = color_offsets[colors[i]]
            optimized.tensor_allocations[tensor.tensor_id] = tensor

        optimized.total_tensor_bytes = plan.total_tensor_bytes
        optimized.weight_bytes = plan.weight_bytes
        optimized.activation_bytes = plan.activation_bytes
        optimized.peak_memory_bytes = offset

        return optimized


# =============================================================================
# Convenience Functions
# =============================================================================

def plan_memory(
    model: Any,
    input_shape: Tuple[int, ...],
    dtype: str = "float32",
    optimize: bool = True,
) -> MemoryPlan:
    """Create optimized memory plan for a model.

    Args:
        model: Model to plan memory for
        input_shape: Input tensor shape
        dtype: Data type
        optimize: Whether to optimize the plan

    Returns:
        Memory allocation plan
    """
    planner = MemoryPlanner(model)
    plan = planner.plan(input_shape, dtype)

    if optimize:
        plan = planner.optimize(plan)

    return plan


def estimate_memory(
    model: Any,
    input_shape: Tuple[int, ...],
    dtype: str = "float32",
) -> Dict[str, int]:
    """Estimate memory requirements for a model.

    Args:
        model: Model to estimate
        input_shape: Input tensor shape
        dtype: Data type

    Returns:
        Dictionary with memory estimates
    """
    plan = plan_memory(model, input_shape, dtype)

    return {
        "total_bytes": plan.total_tensor_bytes,
        "peak_bytes": plan.peak_memory_bytes,
        "weight_bytes": plan.weight_bytes,
        "activation_bytes": plan.activation_bytes,
        "reuse_factor": plan.reuse_factor,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MemoryLevel",
    "TensorAllocation",
    "MemoryPool",
    "MemoryPlan",
    "MemoryPlanner",
    "plan_memory",
    "estimate_memory",
]
