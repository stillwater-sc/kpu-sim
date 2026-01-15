# KPU Simulator Python Integration Guide

## Overview

The KPU Simulator Python Integration provides a comprehensive Python API for interacting with the Stillwater Knowledge Processing Unit (KPU) simulator. This integration enables rapid prototyping, testing, and verification of neural network workloads on the KPU architecture without requiring C++ development.

### Purpose

The Python integration serves several key purposes:

1. **Rapid Prototyping**: Quickly experiment with kernel configurations, tile sizes, and graph structures
2. **Algorithm Development**: Develop and test scheduling algorithms, fusion strategies, and optimization heuristics
3. **Verification**: Create test suites that verify simulator correctness against reference implementations
4. **Education**: Learn the KPU architecture through interactive exploration
5. **Benchmarking**: Profile performance characteristics across different workload configurations

---

## Core Concepts

### The Knowledge Processing Unit (KPU)

The KPU is a specialized processor architecture designed for efficient execution of neural network inference workloads, particularly matrix operations that dominate transformer models and MLPs.

#### Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │              Host System                │
                    └────────────────┬────────────────────────┘
                                     │ PCIe/DMA
                    ┌────────────────▼────────────────────────┐
                    │         External Memory (HBM)           │
                    │         Multiple Banks, 16 GB/s/ch      │
                    └────────────────┬────────────────────────┘
                                     │ DMA Engines
                    ┌────────────────▼────────────────────────┐
                    │         L3 Tile Cache (2 MB)            │
                    │         Weight/Activation Staging       │
                    └────────────────┬────────────────────────┘
                                     │ Block Movers
                    ┌────────────────▼────────────────────────┐
                    │         L2 Banks (256 KB each)          │
                    │         Tile Working Set                │
                    └────────────────┬────────────────────────┘
                                     │ Streamers
                    ┌────────────────▼────────────────────────┐
                    │       L1 Buffers (Scratchpads)          │
                    │       Double-Buffered for Overlap       │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │         Compute Fabric                  │
                    │   ┌─────────────────────────────────┐   │
                    │   │    16x16 Systolic Array         │   │
                    │   │    2 GHz, 1024 GFLOPS FP32      │   │
                    │   └─────────────────────────────────┘   │
                    │   ┌─────────────────────────────────┐   │
                    │   │    Vector Engine (VE)           │   │
                    │   │    Bias + Activation Fusion     │   │
                    │   └─────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
```

#### Key Components

| Component | Description | Python Class |
|-----------|-------------|--------------|
| **Systolic Array** | 16x16 PE array for matrix multiplication | Part of `ConcurrentExecutor` |
| **Vector Engine** | Fused bias addition and activation functions | Controlled via `ActivationType` |
| **Memory Hierarchy** | 5-level: Host → External → L3 → L2 → L1 | Configured via `ResourceConfig` |
| **DMA Engines** | Async data transfer between memory levels | Part of `ResourceConfig` |
| **Block Movers** | L3↔L2 tile movement | Part of `ResourceConfig` |
| **Streamers** | L2↔L1 streaming with compute overlap | Part of `ResourceConfig` |

### The Simulator

The KPU Simulator is a cycle-accurate model of the KPU architecture that:

- Models the complete memory hierarchy with realistic latencies
- Simulates concurrent execution across DMA, data movement, and compute resources
- Tracks resource utilization and generates performance metrics
- Supports the Data Movement ISA (DM-ISA) instruction set

---

## Python API Capabilities

### Module Structure

```python
import stillwater_kpu as kpu

# Core types
kpu.DataType          # Numeric data types (FLOAT32, INT8, etc.)
kpu.ActivationType    # Activation functions (RELU, GELU, etc.)
kpu.KernelOpType      # Kernel operation types (MATMUL, MLP, etc.)

# Kernel creation and compilation
kpu.Kernel            # Kernel abstraction
kpu.KernelCompiler    # Automatic tile optimization
kpu.CompileOptions    # Compilation configuration

# Multi-kernel graphs
kpu.KernelGraph       # DAG of kernels
kpu.KernelEdge        # Tensor flow between kernels
kpu.FusionStrategy    # Kernel fusion options

# Execution
kpu.ResourceConfig    # Hardware configuration
kpu.ConcurrentExecutor # Cycle-accurate execution

# Runtime (CUDA-like API)
kpu.Runtime           # Memory and kernel management
kpu.GraphExecutor     # High-level execution API

# Serialization
kpu.ProgramSerializer # Save/load compiled programs
kpu.KernelSerializer  # Save/load kernels
```

### Data Types

```python
# Supported numeric types
kpu.DataType.FLOAT32   # 4 bytes, IEEE 754 single precision
kpu.DataType.FLOAT16   # 2 bytes, IEEE 754 half precision
kpu.DataType.BFLOAT16  # 2 bytes, brain floating point
kpu.DataType.INT32     # 4 bytes, signed 32-bit (accumulators)
kpu.DataType.INT8      # 1 byte, quantized inference
kpu.DataType.UINT8     # 1 byte, unsigned
kpu.DataType.INT4      # 0.5 bytes, aggressive quantization

# Utility functions
kpu.dtype_size(kpu.DataType.FLOAT32)  # Returns 4
kpu.dtype_name(kpu.DataType.FLOAT32)  # Returns "float32"
```

### Activation Functions

```python
# Supported activations (fused in Vector Engine)
kpu.ActivationType.NONE        # Pass-through
kpu.ActivationType.RELU        # max(0, x)
kpu.ActivationType.GELU        # Gaussian Error Linear Unit
kpu.ActivationType.SIGMOID     # 1 / (1 + exp(-x))
kpu.ActivationType.TANH        # Hyperbolic tangent
kpu.ActivationType.SILU        # x * sigmoid(x) (Swish)
kpu.ActivationType.LEAKY_RELU  # max(alpha*x, x)
```

---

## How-To Guides

### 1. Creating Kernels

Kernels are the fundamental compute units in the KPU. Each kernel represents a single operation that can be compiled to DM-ISA instructions.

#### Matrix Multiplication Kernel

```python
import stillwater_kpu as kpu

# Create a simple matmul: C[M,N] = A[M,K] @ B[K,N]
matmul = kpu.Kernel.create_matmul(
    M=1024,    # Rows of A and C
    N=1024,    # Columns of B and C
    K=1024,    # Columns of A, rows of B
    dtype=kpu.DataType.FLOAT32
)

# Inspect kernel properties
print(f"Kernel: {matmul.name()}")
print(f"Operation: {matmul.op_type()}")
print(f"Dimensions: {matmul.M()}x{matmul.N()}x{matmul.K()}")
print(f"Tile sizes: {matmul.Ti()}x{matmul.Tj()}x{matmul.Tk()}")
print(f"Total FLOPs: {matmul.total_flops():,}")
print(f"Arithmetic Intensity: {matmul.arithmetic_intensity():.2f} FLOP/byte")

# Get kernel arguments
for arg in matmul.arguments():
    direction = "output" if arg.is_output else "input"
    print(f"  {arg.name}: {list(arg.shape)} ({direction})")
```

#### MLP Kernel with Activation

```python
# Create fused MLP: C = activation(A @ B + bias)
mlp = kpu.Kernel.create_mlp(
    M=64,           # Batch size
    N=512,          # Output features
    K=256,          # Input features
    activation=kpu.ActivationType.GELU,
    has_bias=True,
    dtype=kpu.DataType.FLOAT32
)

print(f"Kernel: {mlp.name()}")
print(f"Activation: {mlp.activation()}")
print(f"Has bias: {mlp.has_bias()}")
```

### 2. Using the Kernel Compiler

The KernelCompiler automatically optimizes tile sizes for maximum performance.

#### Auto-Optimization

```python
compiler = kpu.KernelCompiler()

# Compile with automatic tile optimization
kernel = compiler.compile_matmul(2048, 2048, 2048)

# Get compilation statistics
stats = compiler.last_stats()
print(f"Compilation time: {stats.compile_time_us:.1f} us")
print(f"Tile sizes: Ti={stats.selected_Ti}, Tj={stats.selected_Tj}, Tk={stats.selected_Tk}")
print(f"Total tiles: {stats.total_tiles}")
print(f"DMA operations: {stats.dma_ops}")
print(f"Block mover ops: {stats.block_mover_ops}")
print(f"Streamer ops: {stats.streamer_ops}")
print(f"Arithmetic intensity: {stats.estimated_arithmetic_intensity:.2f} FLOP/byte")
```

#### Manual Tile Configuration

```python
# Create options with explicit tile sizes
opts = kpu.CompileOptions.with_tiles(ti=64, tj=64, tk=128)

# Or configure options manually
opts = kpu.CompileOptions()
opts.Ti = 32
opts.Tj = 32
opts.Tk = 64
opts.double_buffer = True
opts.systolic_size = 16

kernel = compiler.compile_matmul(1024, 1024, 1024, opts)
```

#### Inference-Optimized Compilation

```python
# Weight-stationary dataflow for inference
opts = kpu.CompileOptions.for_inference()
kernel = compiler.compile_matmul(1024, 1024, 1024, opts)
```

### 3. Building Kernel Graphs

Kernel graphs represent multi-kernel computations as directed acyclic graphs (DAGs).

#### Two-Layer MLP Network

```python
# Create graph for: input -> FC1(ReLU) -> FC2 -> output
graph = kpu.KernelGraph("two_layer_mlp")

# Add layers
fc1 = graph.add_kernel(
    kpu.Kernel.create_mlp(64, 512, 256, kpu.ActivationType.RELU, True),
    "fc1_relu"
)

fc2 = graph.add_kernel(
    kpu.Kernel.create_mlp(64, 128, 512, kpu.ActivationType.NONE, True),
    "fc2"
)

# Connect layers: output of fc1 feeds input of fc2
graph.add_edge(fc1, fc2, output_name="C", input_name="A")

# Validate the graph
valid, error = graph.validate()
if not valid:
    raise RuntimeError(f"Invalid graph: {error}")
```

#### Parallel Branch Architecture (Diamond Pattern)

```python
# Create graph with parallel branches:
#       input
#       /   \
#    left   right
#       \   /
#       merge

graph = kpu.KernelGraph("residual_block")

input_layer = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 128), "input")
left_branch = graph.add_kernel(kpu.Kernel.create_matmul(64, 128, 64), "left")
right_branch = graph.add_kernel(kpu.Kernel.create_matmul(64, 128, 64), "right")
merge_layer = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 128), "merge")

graph.add_edge(input_layer, left_branch, "C", "A")
graph.add_edge(input_layer, right_branch, "C", "A")
graph.add_edge(left_branch, merge_layer, "C", "A")
graph.add_edge(right_branch, merge_layer, "C", "B")
```

#### Analyzing Graph Structure

```python
# Get execution order (topological sort)
order = graph.get_execution_order()
print("Execution order:", [graph.get_kernel(nid).name() for nid in order])

# Get parallel execution levels
levels = graph.get_execution_levels()
for i, level in enumerate(levels):
    names = [graph.get_kernel(nid).name() for nid in level]
    parallel = len(level) > 1
    print(f"Level {i} ({'parallel' if parallel else 'sequential'}): {names}")

# Get critical path
critical = graph.get_critical_path()
print("Critical path:", [graph.get_kernel(nid).name() for nid in critical])

# Find fusion opportunities
fusible = graph.find_fusible_pairs()
for from_id, to_id in fusible:
    print(f"Can fuse: {graph.get_kernel(from_id).name()} -> {graph.get_kernel(to_id).name()}")
```

#### Graph Statistics

```python
stats = graph.compute_stats()
print(f"Nodes: {stats.num_nodes}")
print(f"Edges: {stats.num_edges}")
print(f"Max depth: {stats.max_depth}")
print(f"Total FLOPs: {stats.total_flops:,}")
print(f"Input bytes: {stats.total_input_bytes / 1024:.1f} KB")
print(f"Output bytes: {stats.total_output_bytes / 1024:.1f} KB")
print(f"Intermediate bytes: {stats.intermediate_bytes / 1024:.1f} KB")
print(f"Avg arithmetic intensity: {stats.avg_arithmetic_intensity:.2f} FLOP/byte")
```

#### Compiling Graphs

```python
# Compile with default options
result = graph.compile()

if result.success:
    print(f"Compilation successful!")
    print(f"Execution order: {result.execution_order}")
    print(f"Workspace required: {result.workspace_required / 1024:.1f} KB")
else:
    print(f"Compilation failed: {result.error_message}")

# Compile with custom options
opts = kpu.KernelGraphCompileOptions()
opts.fusion_strategy = kpu.FusionStrategy.PRODUCER_CONSUMER
opts.enable_double_buffering = True
opts.optimize_memory_allocation = True

result = graph.compile(opts)
```

#### Visualization with Graphviz

```python
# Generate DOT format for visualization
dot_code = graph.to_dot(show_tensor_sizes=True)
print(dot_code)

# Save to file and render
with open("graph.dot", "w") as f:
    f.write(dot_code)

# Render with: dot -Tpng graph.dot -o graph.png
```

### 4. Configuring the Executor

The ConcurrentExecutor models the parallel execution of DM-ISA programs.

```python
# Create resource configuration
config = kpu.ResourceConfig()

# Configure memory channels and data movers
config.num_memory_channels = 4      # DMA engines
config.num_block_movers = 8         # L3↔L2 movers
config.num_streamers = 16           # L2↔L1 streamers

# Configure clock frequencies (MHz)
config.dma_clock_mhz = 250.0
config.block_mover_clock_mhz = 500.0
config.streamer_clock_mhz = 500.0
config.compute_clock_mhz = 2000.0

# Configure bus widths (bytes per cycle)
config.dma_bus_width_bytes = 64     # 512-bit bus
config.block_mover_bus_width_bytes = 64
config.streamer_bus_width_bytes = 64

# Configure compute fabric
config.systolic_size = 16           # 16x16 PE array
config.compute_throughput_gflops = 1024.0

# Create executor
executor = kpu.ConcurrentExecutor(config)

# Execute a compiled program (via kernel)
# cycles = executor.execute(program)
```

### 5. Serialization

Save and load compiled kernels and programs for deployment.

```python
# Serialize kernels
serializer = kpu.KernelSerializer()

# Save to binary format
serializer.save(kernel, "model.kpubin")

# Save to JSON (human-readable)
serializer.save_json(kernel, "model.json", pretty=True)

# Load kernel
loaded = serializer.load("model.kpubin")

# Validate loaded kernel
if serializer.validate(loaded):
    print("Kernel is valid")

# Auto-detect format
kernel = serializer.load_auto("model.kpubin")  # or .json
```

---

## Example Programs

### Demo 1: Transformer Feed-Forward Network

```python
#!/usr/bin/env python3
"""
Transformer FFN Demo
Demonstrates the typical FFN pattern: up-project -> activation -> down-project
"""
import stillwater_kpu as kpu

def create_transformer_ffn(batch: int, hidden: int, intermediate: int):
    """
    Create transformer FFN graph:
    x -> FC1 (up-project) -> GELU -> FC2 (down-project) -> output
    """
    graph = kpu.KernelGraph("transformer_ffn")

    # Up-projection: [batch, hidden] -> [batch, intermediate] with GELU
    fc1 = graph.add_kernel(
        kpu.Kernel.create_mlp(batch, intermediate, hidden,
                              kpu.ActivationType.GELU, has_bias=True),
        "fc1_gelu"
    )

    # Down-projection: [batch, intermediate] -> [batch, hidden]
    fc2 = graph.add_kernel(
        kpu.Kernel.create_mlp(batch, hidden, intermediate,
                              kpu.ActivationType.NONE, has_bias=True),
        "fc2"
    )

    graph.add_edge(fc1, fc2, "C", "A")
    return graph

# Create GPT-2 style FFN (768 hidden, 3072 intermediate)
graph = create_transformer_ffn(batch=32, hidden=768, intermediate=3072)

# Analyze
stats = graph.compute_stats()
print(f"Transformer FFN Statistics:")
print(f"  Total FLOPs: {stats.total_flops:,}")
print(f"  Arithmetic Intensity: {stats.avg_arithmetic_intensity:.2f} FLOP/byte")

# Compile
result = graph.compile()
print(f"  Workspace: {result.workspace_required / 1024:.1f} KB")
```

### Demo 2: Deep MLP Benchmark

```python
#!/usr/bin/env python3
"""
Deep MLP Benchmark
Creates and benchmarks a configurable-depth MLP network
"""
import stillwater_kpu as kpu

def create_deep_mlp(batch: int, layer_sizes: list, activation=kpu.ActivationType.RELU):
    """Create deep MLP with configurable layers"""
    graph = kpu.KernelGraph("deep_mlp")

    node_ids = []
    for i in range(len(layer_sizes) - 1):
        in_features = layer_sizes[i]
        out_features = layer_sizes[i + 1]

        # Use ReLU for hidden layers, no activation for output
        act = activation if i < len(layer_sizes) - 2 else kpu.ActivationType.NONE

        kernel = kpu.Kernel.create_mlp(batch, out_features, in_features, act, True)
        node_id = graph.add_kernel(kernel, f"layer{i+1}")
        node_ids.append(node_id)

    # Connect layers
    for i in range(len(node_ids) - 1):
        graph.add_edge(node_ids[i], node_ids[i + 1], "C", "A")

    return graph

# Create MNIST classifier: 784 -> 512 -> 256 -> 128 -> 10
layer_sizes = [784, 512, 256, 128, 10]
graph = create_deep_mlp(batch=64, layer_sizes=layer_sizes)

# Print layer info
print(f"Deep MLP: {len(layer_sizes)-1} layers")
for i, nid in enumerate(graph.get_execution_order()):
    kernel = graph.get_kernel(nid)
    print(f"  Layer {i+1}: {kernel.name()}")
    print(f"    FLOPs: {kernel.total_flops():,}")

# Compute totals
stats = graph.compute_stats()
print(f"\nTotal Statistics:")
print(f"  FLOPs: {stats.total_flops:,}")
print(f"  Instructions: {stats.total_instructions}")
print(f"  Arithmetic Intensity: {stats.avg_arithmetic_intensity:.2f}")
```

### Demo 3: Comparison of Tile Configurations

```python
#!/usr/bin/env python3
"""
Tile Configuration Comparison
Demonstrates how different tile sizes affect compilation results
"""
import stillwater_kpu as kpu

def benchmark_tiles(M, N, K, tile_configs):
    """Compare different tile configurations"""
    compiler = kpu.KernelCompiler()

    results = []
    for ti, tj, tk in tile_configs:
        opts = kpu.CompileOptions.with_tiles(ti, tj, tk)
        kernel = compiler.compile_matmul(M, N, K, opts)
        stats = compiler.last_stats()

        results.append({
            'tiles': (ti, tj, tk),
            'total_tiles': stats.total_tiles,
            'dma_ops': stats.dma_ops,
            'instructions': stats.instruction_count,
            'external_bytes': stats.estimated_external_bytes,
            'arith_intensity': stats.estimated_arithmetic_intensity
        })

    return results

# Test different tile sizes for 1024x1024x1024 matmul
tile_configs = [
    (32, 32, 32),
    (32, 32, 64),
    (64, 64, 64),
    (64, 64, 128),
    (128, 128, 128),
]

print("Tile Configuration Comparison (1024x1024x1024 matmul)")
print("-" * 80)
print(f"{'Tiles':<20} {'Total':<10} {'DMA':<10} {'Instructions':<15} {'Arith Int':<10}")
print("-" * 80)

results = benchmark_tiles(1024, 1024, 1024, tile_configs)
for r in results:
    ti, tj, tk = r['tiles']
    print(f"{ti}x{tj}x{tk:<13} {r['total_tiles']:<10} {r['dma_ops']:<10} "
          f"{r['instructions']:<15} {r['arith_intensity']:.2f}")

# Also test auto-optimization
compiler = kpu.KernelCompiler()
kernel = compiler.compile_matmul(1024, 1024, 1024)
stats = compiler.last_stats()
print("-" * 80)
print(f"{'Auto-optimized':<20} {stats.total_tiles:<10} {stats.dma_ops:<10} "
      f"{stats.instruction_count:<15} {stats.estimated_arithmetic_intensity:.2f}")
```

---

## Test and Verification Programs

### Test 1: Kernel Validation Suite

```python
#!/usr/bin/env python3
"""
Kernel Validation Suite
Verifies kernel creation and metadata correctness
"""
import stillwater_kpu as kpu

def test_matmul_kernel():
    """Test matmul kernel creation"""
    test_cases = [
        (64, 64, 64),
        (128, 256, 512),
        (1024, 1024, 1024),
        (32, 128, 64),
    ]

    for M, N, K in test_cases:
        kernel = kpu.Kernel.create_matmul(M, N, K)

        # Verify dimensions
        assert kernel.M() == M, f"M mismatch: {kernel.M()} != {M}"
        assert kernel.N() == N, f"N mismatch: {kernel.N()} != {N}"
        assert kernel.K() == K, f"K mismatch: {kernel.K()} != {K}"

        # Verify validity
        assert kernel.is_valid(), f"Kernel should be valid"

        # Verify FLOPs calculation (2*M*N*K for matmul)
        expected_flops = 2 * M * N * K
        assert kernel.total_flops() == expected_flops, \
            f"FLOPs mismatch: {kernel.total_flops()} != {expected_flops}"

        # Verify arguments
        args = kernel.arguments()
        assert len(args) == 3, f"Expected 3 arguments, got {len(args)}"

        arg_names = {arg.name for arg in args}
        assert arg_names == {"A", "B", "C"}, f"Unexpected arguments: {arg_names}"

        print(f"  PASS: matmul {M}x{N}x{K}")

    return True

def test_mlp_kernel():
    """Test MLP kernel creation with various activations"""
    activations = [
        kpu.ActivationType.NONE,
        kpu.ActivationType.RELU,
        kpu.ActivationType.GELU,
        kpu.ActivationType.SIGMOID,
        kpu.ActivationType.TANH,
        kpu.ActivationType.SILU,
    ]

    M, N, K = 64, 128, 256

    for act in activations:
        kernel = kpu.Kernel.create_mlp(M, N, K, act, has_bias=True)

        assert kernel.is_valid()
        assert kernel.op_type() == kpu.KernelOpType.MLP
        assert kernel.activation() == act
        assert kernel.has_bias() == True

        # MLP with bias has 4 arguments: A, B, bias, C
        args = kernel.arguments()
        assert len(args) == 4

        print(f"  PASS: MLP with {act}")

    return True

def test_kernel_validation():
    """Test kernel validation"""
    kernel = kpu.Kernel.create_matmul(64, 64, 64)
    valid, error = kernel.validate()
    assert valid, f"Valid kernel failed validation: {error}"
    print("  PASS: Kernel validation")
    return True

if __name__ == "__main__":
    print("Running Kernel Validation Suite")
    print("=" * 50)

    print("\nTest: Matmul Kernel Creation")
    test_matmul_kernel()

    print("\nTest: MLP Kernel Creation")
    test_mlp_kernel()

    print("\nTest: Kernel Validation")
    test_kernel_validation()

    print("\n" + "=" * 50)
    print("All tests passed!")
```

### Test 2: Graph Validation Suite

```python
#!/usr/bin/env python3
"""
Graph Validation Suite
Verifies graph construction, validation, and analysis
"""
import stillwater_kpu as kpu

def test_graph_construction():
    """Test basic graph construction"""
    graph = kpu.KernelGraph("test_graph")

    assert graph.empty()
    assert graph.num_nodes() == 0
    assert graph.num_edges() == 0

    # Add nodes
    n1 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "node1")
    n2 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "node2")

    assert graph.num_nodes() == 2
    assert graph.has_node(n1)
    assert graph.has_node(n2)

    # Add edge
    graph.add_edge(n1, n2, "C", "A")
    assert graph.num_edges() == 1

    print("  PASS: Graph construction")
    return True

def test_cycle_detection():
    """Test that cycles are detected"""
    graph = kpu.KernelGraph("cycle_test")

    n1 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "a")
    n2 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "b")
    n3 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "c")

    graph.add_edge(n1, n2, "C", "A")
    graph.add_edge(n2, n3, "C", "A")

    # This would create a cycle
    would_cycle = graph.would_create_cycle(n3, n1)
    assert would_cycle, "Cycle should be detected"

    print("  PASS: Cycle detection")
    return True

def test_topological_sort():
    """Test execution order (topological sort)"""
    graph = kpu.KernelGraph("topo_test")

    # Create chain: a -> b -> c
    a = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "a")
    b = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "b")
    c = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "c")

    graph.add_edge(a, b, "C", "A")
    graph.add_edge(b, c, "C", "A")

    order = graph.get_execution_order()

    # a must come before b, b must come before c
    assert order.index(a) < order.index(b)
    assert order.index(b) < order.index(c)

    print("  PASS: Topological sort")
    return True

def test_parallel_levels():
    """Test parallel execution level detection"""
    graph = kpu.KernelGraph("parallel_test")

    # Diamond pattern
    top = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "top")
    left = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "left")
    right = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "right")
    bottom = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "bottom")

    graph.add_edge(top, left, "C", "A")
    graph.add_edge(top, right, "C", "A")
    graph.add_edge(left, bottom, "C", "A")
    graph.add_edge(right, bottom, "C", "B")

    levels = graph.get_execution_levels()

    # Should have 3 levels: [top], [left, right], [bottom]
    assert len(levels) == 3
    assert len(levels[0]) == 1  # top alone
    assert len(levels[1]) == 2  # left and right parallel
    assert len(levels[2]) == 1  # bottom alone

    print("  PASS: Parallel level detection")
    return True

def test_graph_validation():
    """Test graph validation"""
    # Valid graph
    valid_graph = kpu.KernelGraph("valid")
    n1 = valid_graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "a")
    n2 = valid_graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "b")
    valid_graph.add_edge(n1, n2, "C", "A")

    is_valid, error = valid_graph.validate()
    assert is_valid, f"Valid graph should pass: {error}"

    print("  PASS: Graph validation")
    return True

if __name__ == "__main__":
    print("Running Graph Validation Suite")
    print("=" * 50)

    print("\nTest: Graph Construction")
    test_graph_construction()

    print("\nTest: Cycle Detection")
    test_cycle_detection()

    print("\nTest: Topological Sort")
    test_topological_sort()

    print("\nTest: Parallel Level Detection")
    test_parallel_levels()

    print("\nTest: Graph Validation")
    test_graph_validation()

    print("\n" + "=" * 50)
    print("All tests passed!")
```

### Test 3: Compiler Regression Suite

```python
#!/usr/bin/env python3
"""
Compiler Regression Suite
Verifies compiler output consistency and correctness
"""
import stillwater_kpu as kpu

def test_auto_tiling_consistency():
    """Test that auto-tiling produces consistent results"""
    compiler = kpu.KernelCompiler()

    # Compile same kernel twice
    k1 = compiler.compile_matmul(512, 512, 512)
    stats1 = compiler.last_stats()

    k2 = compiler.compile_matmul(512, 512, 512)
    stats2 = compiler.last_stats()

    # Results should be identical
    assert stats1.selected_Ti == stats2.selected_Ti
    assert stats1.selected_Tj == stats2.selected_Tj
    assert stats1.selected_Tk == stats2.selected_Tk
    assert stats1.total_tiles == stats2.total_tiles
    assert stats1.instruction_count == stats2.instruction_count

    print("  PASS: Auto-tiling consistency")
    return True

def test_tile_bounds():
    """Test that tile sizes respect bounds"""
    compiler = kpu.KernelCompiler()

    test_cases = [
        (64, 64, 64),
        (100, 100, 100),  # Non-power-of-2
        (1024, 512, 256),
        (32, 128, 64),
    ]

    for M, N, K in test_cases:
        kernel = compiler.compile_matmul(M, N, K)
        stats = compiler.last_stats()

        # Tiles should not exceed problem dimensions
        assert stats.selected_Ti <= M, f"Ti {stats.selected_Ti} > M {M}"
        assert stats.selected_Tj <= N, f"Tj {stats.selected_Tj} > N {N}"
        assert stats.selected_Tk <= K, f"Tk {stats.selected_Tk} > K {K}"

        # Tiles should be positive
        assert stats.selected_Ti > 0
        assert stats.selected_Tj > 0
        assert stats.selected_Tk > 0

    print("  PASS: Tile bounds")
    return True

def test_explicit_tiles():
    """Test compilation with explicit tile sizes"""
    compiler = kpu.KernelCompiler()

    Ti, Tj, Tk = 32, 64, 128
    kernel = compiler.compile_matmul_tiled(256, 256, 256, Ti, Tj, Tk)
    stats = compiler.last_stats()

    assert stats.selected_Ti == Ti
    assert stats.selected_Tj == Tj
    assert stats.selected_Tk == Tk

    # Verify tile counts
    expected_m_tiles = (256 + Ti - 1) // Ti
    expected_n_tiles = (256 + Tj - 1) // Tj
    expected_k_tiles = (256 + Tk - 1) // Tk

    assert stats.num_m_tiles == expected_m_tiles
    assert stats.num_n_tiles == expected_n_tiles
    assert stats.num_k_tiles == expected_k_tiles

    print("  PASS: Explicit tiles")
    return True

def test_instruction_count_scaling():
    """Test that instruction count scales reasonably with problem size"""
    compiler = kpu.KernelCompiler()

    # Fix tile sizes to isolate scaling
    opts = kpu.CompileOptions.with_tiles(64, 64, 64)

    sizes = [128, 256, 512]
    instruction_counts = []

    for size in sizes:
        kernel = compiler.compile_matmul(size, size, size, opts)
        instruction_counts.append(compiler.last_stats().instruction_count)

    # Instruction count should increase with problem size
    for i in range(1, len(instruction_counts)):
        assert instruction_counts[i] > instruction_counts[i-1], \
            f"Instructions should increase: {instruction_counts}"

    print("  PASS: Instruction count scaling")
    return True

if __name__ == "__main__":
    print("Running Compiler Regression Suite")
    print("=" * 50)

    print("\nTest: Auto-tiling Consistency")
    test_auto_tiling_consistency()

    print("\nTest: Tile Bounds")
    test_tile_bounds()

    print("\nTest: Explicit Tiles")
    test_explicit_tiles()

    print("\nTest: Instruction Count Scaling")
    test_instruction_count_scaling()

    print("\n" + "=" * 50)
    print("All tests passed!")
```

---

## Best Practices

### Performance Optimization

1. **Use auto-tiling for production**: The compiler's tile optimizer considers cache sizes and bandwidth constraints
2. **Prefer larger tiles when memory permits**: Reduces instruction overhead and improves data reuse
3. **Enable double buffering**: Overlaps compute with data movement
4. **Fuse operations when possible**: MLP kernels fuse matmul + bias + activation

### Memory Efficiency

1. **Minimize intermediate tensors**: Use in-place operations where possible
2. **Consider workspace requirements**: Graph compilation reports workspace needs
3. **Profile arithmetic intensity**: Higher is better (compute-bound vs memory-bound)

### Testing Guidelines

1. **Validate all kernels**: Use `kernel.validate()` before compilation
2. **Check graph structure**: Use `graph.validate()` before compilation
3. **Verify execution order**: Ensure dependencies are respected
4. **Compare against reference**: Implement CPU reference for numerical verification

---

## Troubleshooting

### Common Issues

**Import Error: Module not found**
```python
# Add build directory to path
import sys
sys.path.insert(0, 'build/src/bindings/python/Release')
import stillwater_kpu as kpu
```

**Unregistered type error**
```
TypeError: Unregistered type : sw::kpu::isa::DMProgram
```
Some internal types (like DMProgram) are not exposed to Python. Use the higher-level APIs instead of accessing internal structures directly.

**Graph validation failure**
```python
valid, error = graph.validate()
if not valid:
    print(f"Error: {error}")
    # Check tensor shape compatibility between connected kernels
```

---

## API Reference Summary

| Category | Classes/Functions |
|----------|-------------------|
| **Data Types** | `DataType`, `dtype_size()`, `dtype_name()` |
| **Activation** | `ActivationType` |
| **Kernels** | `Kernel`, `KernelArgument`, `KernelOpType` |
| **Compilation** | `KernelCompiler`, `CompileOptions`, `CompilationStats`, `DataflowStrategy` |
| **Graphs** | `KernelGraph`, `KernelEdge`, `KernelGraphStats`, `KernelGraphCompileResult` |
| **Fusion** | `FusionStrategy` |
| **Execution** | `ResourceConfig`, `ConcurrentExecutor`, `UtilizationStats` |
| **Runtime** | `Runtime`, `GraphExecutor`, `Stream`, `Event` |
| **Serialization** | `KernelSerializer`, `ProgramSerializer` |

---

## Version History

- **v0.1.0**: Initial Python integration with kernel, graph, and execution APIs
