#!/usr/bin/env python3
"""
Kernel Graph Example

Demonstrates the KPU simulator Python bindings for:
1. Creating kernels (matmul, MLP)
2. Using the KernelCompiler for automatic optimization
3. Building kernel graphs (multi-kernel DAGs)
4. Analyzing and compiling graphs

This example shows the complete workflow from kernel creation
to graph compilation for a simple two-layer neural network.
"""

import sys
import os

# Add the build directory to the path
# Adjust this path based on your build configuration
build_dir = os.path.join(os.path.dirname(__file__), '../../build/src/bindings/python/Release')
sys.path.insert(0, build_dir)

import stillwater_kpu as kpu

def print_separator(title: str = ""):
    """Print a section separator"""
    print()
    if title:
        print(f"=== {title} {'=' * (55 - len(title))}")
    else:
        print("=" * 60)
    print()

def demo_kernel_creation():
    """Demonstrate creating individual kernels"""
    print_separator("Kernel Creation")

    # Create a simple matmul kernel
    print("Creating matmul kernel (64x64x64)...")
    matmul = kpu.Kernel.create_matmul(64, 64, 64)
    print(f"  Name: {matmul.name()}")
    print(f"  Type: {matmul.op_type()}")
    print(f"  Dimensions: M={matmul.M()}, N={matmul.N()}, K={matmul.K()}")
    print(f"  Tile sizes: Ti={matmul.Ti()}, Tj={matmul.Tj()}, Tk={matmul.Tk()}")
    print(f"  Instructions: {matmul.instruction_count()}")
    print(f"  Total FLOPs: {matmul.total_flops()}")
    print(f"  Arithmetic Intensity: {matmul.arithmetic_intensity():.2f} FLOP/byte")
    print()

    # Create an MLP kernel with activation
    print("Creating MLP kernel with GELU activation...")
    mlp = kpu.Kernel.create_mlp(
        M=32, N=128, K=64,
        activation=kpu.ActivationType.GELU,
        has_bias=True
    )
    print(f"  Name: {mlp.name()}")
    print(f"  Type: {mlp.op_type()}")
    print(f"  Activation: {mlp.activation()}")
    print(f"  Has bias: {mlp.has_bias()}")
    print()

    # Show kernel arguments
    print("Kernel arguments:")
    for arg in mlp.arguments():
        output = "(output)" if arg.is_output else ""
        print(f"  {arg.name}: shape={list(arg.shape)}, dtype={kpu.dtype_name(arg.dtype)} {output}")

def demo_kernel_compiler():
    """Demonstrate the kernel compiler"""
    print_separator("Kernel Compiler")

    compiler = kpu.KernelCompiler()

    # Compile a larger matmul with auto-optimization
    print("Compiling 1024x1024x1024 matmul with auto-optimization...")
    kernel = compiler.compile_matmul(1024, 1024, 1024)

    print(f"  Kernel: {kernel.name()}")
    print(f"  Valid: {kernel.is_valid()}")
    print()

    # Get compilation statistics
    stats = compiler.last_stats()
    print("Compilation Statistics:")
    print(f"  Compile time: {stats.compile_time_us:.1f} us")
    print(f"  Auto-tiling used: {stats.used_auto_tiling}")
    print(f"  Tile sizes: Ti={stats.selected_Ti}, Tj={stats.selected_Tj}, Tk={stats.selected_Tk}")
    print(f"  Tile counts: M={stats.num_m_tiles}, N={stats.num_n_tiles}, K={stats.num_k_tiles}")
    print(f"  Total tiles: {stats.total_tiles}")
    print(f"  Instructions: {stats.instruction_count}")
    print(f"  DMA ops: {stats.dma_ops}")
    print(f"  Block mover ops: {stats.block_mover_ops}")
    print(f"  Streamer ops: {stats.streamer_ops}")
    print(f"  Estimated DRAM traffic: {stats.estimated_external_bytes / 1024:.1f} KB")
    print(f"  Arithmetic intensity: {stats.estimated_arithmetic_intensity:.2f} FLOP/byte")
    print()

    # Compile with explicit tile sizes
    print("Compiling with explicit tile sizes (32, 32, 64)...")
    opts = kpu.CompileOptions.with_tiles(32, 32, 64)
    kernel2 = compiler.compile_matmul(256, 256, 256, opts)

    stats2 = compiler.last_stats()
    print(f"  Tile sizes: Ti={stats2.selected_Ti}, Tj={stats2.selected_Tj}, Tk={stats2.selected_Tk}")
    print(f"  Total tiles: {stats2.total_tiles}")

def demo_kernel_graph():
    """Demonstrate kernel graphs for multi-kernel execution"""
    print_separator("Kernel Graph (Multi-Kernel DAG)")

    # Create a two-layer neural network graph:
    # Input -> FC1 (with ReLU) -> FC2 -> Output
    #
    # Layer 1: [batch=64, in=256] @ [256, 512] -> [64, 512]
    # Layer 2: [batch=64, in=512] @ [512, 128] -> [64, 128]

    print("Creating a two-layer MLP graph...")
    graph = kpu.KernelGraph("two_layer_mlp")

    # Add layer 1: matmul + ReLU
    layer1 = graph.add_kernel(
        kpu.Kernel.create_mlp(64, 512, 256, kpu.ActivationType.RELU, True),
        "fc1_relu"
    )

    # Add layer 2: matmul (no activation, final output)
    layer2 = graph.add_kernel(
        kpu.Kernel.create_mlp(64, 128, 512, kpu.ActivationType.NONE, True),
        "fc2"
    )

    # Connect layers: output of fc1 -> input of fc2
    graph.add_edge(layer1, layer2, "C", "A")

    # Print graph info
    print(f"  Graph name: {graph.name}")
    print(f"  Nodes: {graph.num_nodes()}")
    print(f"  Edges: {graph.num_edges()}")
    print()

    # Validate the graph
    valid, error = graph.validate()
    if valid:
        print("  Graph validation: PASSED")
    else:
        print(f"  Graph validation: FAILED - {error}")
    print()

    # Get execution order
    order = graph.get_execution_order()
    print("Execution Order:")
    for i, node_id in enumerate(order):
        kernel = graph.get_kernel(node_id)
        print(f"  {i+1}. {kernel.name()} (node {node_id})")
    print()

    # Get execution levels (for parallel scheduling)
    levels = graph.get_execution_levels()
    print("Execution Levels (parallel scheduling):")
    for level_idx, level in enumerate(levels):
        kernels = [graph.get_kernel(nid).name() for nid in level]
        print(f"  Level {level_idx}: {', '.join(kernels)}")
    print()

    # Find fusion opportunities
    fusible = graph.find_fusible_pairs()
    print(f"Fusion opportunities: {len(fusible)} pairs found")
    for from_id, to_id in fusible:
        from_name = graph.get_kernel(from_id).name()
        to_name = graph.get_kernel(to_id).name()
        print(f"  {from_name} <-> {to_name}")
    print()

    # Compute statistics
    stats = graph.compute_stats()
    print("Graph Statistics:")
    print(f"  Total nodes: {stats.num_nodes}")
    print(f"  Total edges: {stats.num_edges}")
    print(f"  Max depth: {stats.max_depth}")
    print(f"  Total FLOPs: {stats.total_flops:,}")
    print(f"  Total instructions: {stats.total_instructions}")
    print(f"  Input bytes: {stats.total_input_bytes / 1024:.1f} KB")
    print(f"  Output bytes: {stats.total_output_bytes / 1024:.1f} KB")
    print(f"  Intermediate bytes: {stats.intermediate_bytes / 1024:.1f} KB")
    print(f"  Avg arithmetic intensity: {stats.avg_arithmetic_intensity:.2f} FLOP/byte")
    print()

    # Compile the graph
    print("Compiling graph to single program...")
    result = graph.compile()
    if result.success:
        print(f"  Success!")
        print(f"  Execution order: {result.execution_order}")
        print(f"  Fused pairs: {len(result.fused_pairs)}")
        print(f"  Workspace required: {result.workspace_required / 1024:.1f} KB")
    else:
        print(f"  Failed: {result.error_message}")

def demo_diamond_graph():
    """Demonstrate a diamond-shaped graph with parallel branches"""
    print_separator("Diamond Graph Pattern")

    # Diamond pattern:
    #       input
    #       /   \
    #    left   right
    #       \   /
    #       merge

    print("Creating diamond-pattern graph...")
    graph = kpu.KernelGraph("diamond_pattern")

    # Input layer
    input_node = graph.add_kernel(
        kpu.Kernel.create_matmul(64, 64, 128),
        "input"
    )

    # Left branch
    left = graph.add_kernel(
        kpu.Kernel.create_matmul(64, 128, 64),
        "left_branch"
    )

    # Right branch
    right = graph.add_kernel(
        kpu.Kernel.create_matmul(64, 128, 64),
        "right_branch"
    )

    # Merge layer
    merge = graph.add_kernel(
        kpu.Kernel.create_matmul(64, 64, 128),
        "merge"
    )

    # Connect edges
    graph.add_edge(input_node, left, "C", "A")
    graph.add_edge(input_node, right, "C", "A")
    graph.add_edge(left, merge, "C", "A")
    graph.add_edge(right, merge, "C", "B")

    print(f"  Nodes: {graph.num_nodes()}")
    print(f"  Edges: {graph.num_edges()}")
    print()

    # Show parallel execution opportunities
    levels = graph.get_execution_levels()
    print("Parallel Execution Levels:")
    for level_idx, level in enumerate(levels):
        kernels = [graph.get_kernel(nid).name() for nid in level]
        parallelism = "parallel" if len(kernels) > 1 else "sequential"
        print(f"  Level {level_idx} ({parallelism}): {', '.join(kernels)}")
    print()

    # Show critical path
    critical = graph.get_critical_path()
    print("Critical Path:")
    path_names = [graph.get_kernel(nid).name() for nid in critical]
    print(f"  {' -> '.join(path_names)}")
    print()

    # Get DOT visualization
    print("DOT graph (for Graphviz visualization):")
    print(graph.to_dot(True))

def demo_concurrent_executor():
    """Demonstrate the concurrent executor"""
    print_separator("Concurrent Executor")

    # Configure resources
    config = kpu.ResourceConfig()
    config.num_memory_channels = 4
    config.num_block_movers = 8
    config.num_streamers = 16
    config.systolic_size = 16

    print("Resource Configuration:")
    print(f"  Memory channels: {config.num_memory_channels}")
    print(f"  Block movers: {config.num_block_movers}")
    print(f"  Streamers: {config.num_streamers}")
    print(f"  Systolic array: {config.systolic_size}x{config.systolic_size}")
    print(f"  DMA bandwidth: {config.dma_bandwidth_gb_s} GB/s per channel")
    print(f"  Block mover bandwidth: {config.block_mover_bandwidth_gb_s} GB/s per mover")
    print()

    # Create executor
    executor = kpu.ConcurrentExecutor(config)

    # Compile a kernel
    compiler = kpu.KernelCompiler()
    kernel = compiler.compile_matmul(512, 512, 512)

    print(f"Executing kernel: {kernel.name()}")
    print(f"  Problem size: 512x512x512 matmul")
    print(f"  FLOPs: {kernel.total_flops():,}")

    # Note: The program is accessed through the kernel's internal program
    # This demonstrates the executor API
    # In a full implementation, you would get the DMProgram from the kernel
    print()
    print("Executor is ready for DMProgram execution.")
    print("(Full execution requires the simulator and memory setup)")

def main():
    print()
    print("=" * 60)
    print("  KPU Simulator - Python Kernel Graph Example")
    print("=" * 60)

    try:
        demo_kernel_creation()
        demo_kernel_compiler()
        demo_kernel_graph()
        demo_diamond_graph()
        demo_concurrent_executor()

        print_separator("Example Complete")
        print("All examples executed successfully!")
        print()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
