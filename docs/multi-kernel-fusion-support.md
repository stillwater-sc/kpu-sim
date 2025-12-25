# Multi-Kernel and Fusion Support

This document describes the multi-kernel execution facility in the KPU simulator, which enables building, analyzing, and executing computational graphs containing multiple interconnected kernels.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Usage Guide](#usage-guide)
- [Application Development Patterns](#application-development-patterns)
- [Performance Considerations](#performance-considerations)
- [Future Directions](#future-directions)

---

## Problem Statement

### The Single-Kernel Limitation

The initial KPU kernel abstraction (`Kernel` class) represents a single computational operation—typically a matrix multiplication or MLP layer. While sufficient for benchmarking individual operations, real-world neural network inference requires executing **sequences of dependent operations**:

```
Input -> Conv1 -> ReLU -> Conv2 -> ReLU -> FC1 -> FC2 -> Output
```

Without multi-kernel support, developers must:
1. Manually sequence kernel executions
2. Manage intermediate data transfers between kernels
3. Miss optimization opportunities like kernel fusion
4. Lose visibility into the overall computation graph

### Key Challenges

| Challenge | Description |
|-----------|-------------|
| **Dependency Management** | Ensuring kernels execute in correct order based on data dependencies |
| **Memory Allocation** | Efficiently allocating workspace for intermediate tensors |
| **Fusion Opportunities** | Identifying and exploiting producer-consumer pairs that can be merged |
| **Parallelism** | Finding independent kernels that can execute concurrently |
| **Compilation** | Generating a single optimized program from multiple kernels |

---

## Solution Overview

The **KernelGraph** class provides a directed acyclic graph (DAG) abstraction where:
- **Nodes** represent individual kernels (matmul, MLP, etc.)
- **Edges** represent data dependencies between kernel outputs and inputs

This abstraction enables:
1. **Automatic Scheduling**: Topological sort determines valid execution order
2. **Parallel Analysis**: Execution levels identify concurrent execution opportunities
3. **Fusion Detection**: Automatic identification of fusible kernel pairs
4. **Unified Compilation**: All kernels compiled into a single DMProgram
5. **Visualization**: DOT export for graph visualization tools

### Design Principles

1. **Compositional**: Build complex networks from simple kernel primitives
2. **Analyzable**: Query graph structure before execution
3. **Optimizable**: Enable fusion and parallelism optimizations
4. **Executable**: Compile to efficient DMProgram for simulation

---

## Architecture

### Class Hierarchy

```
KernelGraph
├── nodes_: map<size_t, KernelNode>
│   └── KernelNode
│       ├── id: size_t
│       ├── kernel: unique_ptr<Kernel>
│       ├── name: string
│       ├── input_edges: vector<size_t>
│       └── output_edges: vector<size_t>
│
├── edges_: vector<KernelEdge>
│   └── KernelEdge
│       ├── from_node, to_node: size_t
│       ├── output_name, input_name: string
│       └── tensor_size_bytes: Size
│
└── Methods
    ├── add_kernel(), add_edge()
    ├── get_execution_order()
    ├── get_execution_levels()
    ├── find_fusible_pairs()
    ├── compile()
    └── to_dot()
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Code                          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  KernelGraph                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                      │
│  │ Kernel1 │───▶│ Kernel2 │───▶│ Kernel3 │                      │
│  └─────────┘    └─────────┘    └─────────┘                      │
│       │              │              │                            │
│       ▼              ▼              ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               Graph Analysis                                 ││
│  │  • Topological Sort    • Fusion Detection                   ││
│  │  • Execution Levels    • Critical Path                      ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼ compile()
┌─────────────────────────────────────────────────────────────────┐
│  KernelGraphCompileResult                                        │
│  ├── program: DMProgram (unified instruction stream)            │
│  ├── execution_order: vector<size_t>                            │
│  ├── fused_pairs: vector<pair<size_t, size_t>>                  │
│  └── workspace_required: Size                                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ConcurrentExecutor                                              │
│  └── execute(program) -> Cycle                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Reference

### KernelGraph Class

#### Construction

```cpp
// Create empty graph
KernelGraph graph;

// Create named graph
KernelGraph graph("transformer_block");
```

#### Node Management

```cpp
// Add kernel by value (moved)
size_t node_id = graph.add_kernel(
    Kernel::create_matmul(M, N, K),
    "layer_name"  // optional
);

// Add kernel by unique_ptr
size_t node_id = graph.add_kernel(
    std::make_unique<Kernel>(std::move(kernel)),
    "layer_name"
);

// Access nodes
const KernelNode& node = graph.get_node(node_id);
const Kernel& kernel = graph.get_kernel(node_id);
bool exists = graph.has_node(node_id);
std::vector<size_t> ids = graph.node_ids();
```

#### Edge Management

```cpp
// Connect kernels: producer.output_name -> consumer.input_name
size_t edge_id = graph.add_edge(
    producer_id,
    consumer_id,
    "C",    // output argument name (default: "C")
    "A"     // input argument name (default: "A")
);

// Query edges
const KernelEdge& edge = graph.get_edge(edge_id);
std::vector<size_t> out_edges = graph.outgoing_edges(node_id);
std::vector<size_t> in_edges = graph.incoming_edges(node_id);

// Cycle detection
bool would_cycle = graph.would_create_cycle(from_id, to_id);
```

#### Graph Analysis

```cpp
// Validation
std::string error;
bool valid = graph.validate(error);

// Execution order (topological sort)
std::vector<size_t> order = graph.get_execution_order();

// Parallel execution levels
std::vector<std::vector<size_t>> levels = graph.get_execution_levels();
// levels[0] = input nodes (no dependencies)
// levels[1] = nodes depending only on level 0
// etc.

// Critical path (longest dependency chain)
std::vector<size_t> critical = graph.get_critical_path();

// Statistics
KernelGraphStats stats = graph.compute_stats();
// stats.num_nodes, stats.num_edges
// stats.total_flops, stats.total_instructions
// stats.intermediate_bytes, stats.avg_arithmetic_intensity
```

#### Fusion

```cpp
// Find all fusible pairs
std::vector<std::pair<size_t, size_t>> pairs = graph.find_fusible_pairs();

// Check specific pair
bool can = graph.can_fuse(producer_id, consumer_id);

// Mark for fusion
graph.mark_for_fusion(producer_id, consumer_id);
graph.clear_fusion_marks();
```

#### Compilation

```cpp
// Compile with options
KernelGraphCompileOptions opts;
opts.fusion_strategy = FusionStrategy::PRODUCER_CONSUMER;
opts.enable_double_buffering = true;
opts.insert_global_barriers = true;

KernelGraphCompileResult result = graph.compile(opts);

if (result.success) {
    DMProgram& program = result.program;
    // Execute with ConcurrentExecutor
}

// Simple sequential compilation
KernelGraphCompileResult result = graph.compile_sequential();
```

#### Visualization

```cpp
// Human-readable summary
std::string summary = graph.summary();

// DOT format for Graphviz
std::string dot = graph.to_dot(true);  // true = show tensor sizes
// Save to file, then: dot -Tpng graph.dot -o graph.png
```

### Supporting Types

```cpp
// Fusion strategies
enum class FusionStrategy {
    NONE,               // Execute kernels separately
    PRODUCER_CONSUMER,  // Fuse direct producer-consumer pairs
    HORIZONTAL,         // Fuse independent parallel kernels
    PIPELINE            // Pipeline with overlapping data movement
};

// Compilation result
struct KernelGraphCompileResult {
    DMProgram program;
    std::vector<size_t> execution_order;
    std::vector<std::pair<size_t, size_t>> fused_pairs;
    Size workspace_required;
    bool success;
    std::string error_message;
};

// Graph statistics
struct KernelGraphStats {
    size_t num_nodes, num_edges;
    size_t num_input_nodes, num_output_nodes;
    size_t max_depth;
    size_t total_instructions;
    Size total_flops;
    Size total_input_bytes, total_output_bytes;
    Size intermediate_bytes;
    double avg_arithmetic_intensity;
};
```

---

## Usage Guide

### Example 1: Two-Layer Network

```cpp
#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/kernel.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>

using namespace sw::kpu;

int main() {
    // Create graph
    KernelGraph graph("two_layer_mlp");

    // Add layers
    size_t fc1 = graph.add_kernel(
        Kernel::create_mlp(64, 512, 256, ActivationType::RELU, true),
        "fc1_relu");

    size_t fc2 = graph.add_kernel(
        Kernel::create_mlp(64, 10, 512, ActivationType::NONE, true),
        "fc2_output");

    // Connect: fc1.C -> fc2.A
    graph.add_edge(fc1, fc2, "C", "A");

    // Compile
    auto result = graph.compile();
    if (!result.success) {
        std::cerr << "Compilation failed: " << result.error_message << "\n";
        return 1;
    }

    // Execute
    isa::ResourceConfig config;
    isa::ConcurrentExecutor executor(config);
    Cycle cycles = executor.execute(result.program);

    std::cout << "Executed in " << cycles << " cycles\n";
    return 0;
}
```

### Example 2: Transformer Feed-Forward Network

```cpp
KernelGraph graph("transformer_ffn");

Size batch = 32;
Size hidden = 768;
Size intermediate = 3072;  // 4x expansion

// Up-projection with GELU
size_t up = graph.add_kernel(
    Kernel::create_mlp(batch, intermediate, hidden,
                       ActivationType::GELU, true),
    "up_project");

// Down-projection
size_t down = graph.add_kernel(
    Kernel::create_mlp(batch, hidden, intermediate,
                       ActivationType::NONE, true),
    "down_project");

graph.add_edge(up, down);

// Check for fusion opportunities
auto fusible = graph.find_fusible_pairs();
for (auto [from, to] : fusible) {
    std::cout << "Can fuse: " << graph.get_node(from).name
              << " -> " << graph.get_node(to).name << "\n";
}
```

### Example 3: Parallel Branches (ResNet-style)

```cpp
KernelGraph graph("residual_block");

// Main path
size_t input = graph.add_kernel(Kernel::create_matmul(64, 256, 256), "input");
size_t conv1 = graph.add_kernel(Kernel::create_matmul(64, 256, 256), "conv1");
size_t conv2 = graph.add_kernel(Kernel::create_matmul(64, 256, 256), "conv2");

// Skip connection (identity in this simplified example)
size_t skip = graph.add_kernel(Kernel::create_matmul(64, 256, 256), "skip");

// Merge point
size_t merge = graph.add_kernel(Kernel::create_matmul(64, 256, 256), "merge");

// Main path: input -> conv1 -> conv2 -> merge
graph.add_edge(input, conv1);
graph.add_edge(conv1, conv2);
graph.add_edge(conv2, merge, "C", "A");

// Skip path: input -> skip -> merge
graph.add_edge(input, skip);
graph.add_edge(skip, merge, "C", "B");

// Analyze parallel execution opportunities
auto levels = graph.get_execution_levels();
// Level 0: [input]
// Level 1: [conv1, skip]  <- These can run in parallel!
// Level 2: [conv2]
// Level 3: [merge]
```

### Example 4: Visualizing the Graph

```cpp
KernelGraph graph("my_network");
// ... add nodes and edges ...

// Generate DOT file
std::ofstream dot_file("network.dot");
dot_file << graph.to_dot(true);
dot_file.close();

// Then run: dot -Tpng network.dot -o network.png
// Or paste into https://dreampuf.github.io/GraphvizOnline/
```

---

## Application Development Patterns

### Pattern 1: Layer-by-Layer Construction

Build networks by adding layers sequentially:

```cpp
class NetworkBuilder {
    KernelGraph graph_;
    size_t last_layer_ = SIZE_MAX;
    Size batch_size_;
    Size current_features_;

public:
    NetworkBuilder(const std::string& name, Size batch, Size input_features)
        : graph_(name), batch_size_(batch), current_features_(input_features) {}

    NetworkBuilder& add_linear(Size output_features,
                               ActivationType act = ActivationType::NONE,
                               bool bias = true) {
        auto kernel = Kernel::create_mlp(
            batch_size_, output_features, current_features_, act, bias);

        std::string name = "linear_" + std::to_string(graph_.num_nodes());
        size_t id = graph_.add_kernel(std::move(kernel), name);

        if (last_layer_ != SIZE_MAX) {
            graph_.add_edge(last_layer_, id);
        }

        last_layer_ = id;
        current_features_ = output_features;
        return *this;
    }

    KernelGraph build() { return std::move(graph_); }
};

// Usage:
auto graph = NetworkBuilder("classifier", 32, 784)
    .add_linear(512, ActivationType::RELU)
    .add_linear(256, ActivationType::RELU)
    .add_linear(10)
    .build();
```

### Pattern 2: Graph Templates

Create reusable graph patterns:

```cpp
// Transformer MLP block
KernelGraph create_transformer_mlp(Size batch, Size hidden, Size intermediate) {
    KernelGraph graph("transformer_mlp");

    size_t up = graph.add_kernel(
        Kernel::create_mlp(batch, intermediate, hidden, ActivationType::GELU, true),
        "up_project");

    size_t down = graph.add_kernel(
        Kernel::create_mlp(batch, hidden, intermediate, ActivationType::NONE, true),
        "down_project");

    graph.add_edge(up, down);
    return graph;
}

// Multi-head attention (simplified)
KernelGraph create_attention_block(Size batch, Size seq_len, Size hidden, Size heads) {
    KernelGraph graph("attention");

    Size head_dim = hidden / heads;

    // Q, K, V projections (can run in parallel)
    size_t q_proj = graph.add_kernel(
        Kernel::create_matmul(batch * seq_len, hidden, hidden), "q_proj");
    size_t k_proj = graph.add_kernel(
        Kernel::create_matmul(batch * seq_len, hidden, hidden), "k_proj");
    size_t v_proj = graph.add_kernel(
        Kernel::create_matmul(batch * seq_len, hidden, hidden), "v_proj");

    // Note: Actual attention would need softmax and more complex operations
    // This is a simplified example showing the parallel structure

    return graph;
}
```

### Pattern 3: Performance Analysis Before Execution

Analyze graphs before committing to execution:

```cpp
void analyze_network(const KernelGraph& graph) {
    auto stats = graph.compute_stats();

    std::cout << "Network Analysis:\n";
    std::cout << "  Layers: " << stats.num_nodes << "\n";
    std::cout << "  Total FLOPs: " << stats.total_flops << "\n";
    std::cout << "  Arithmetic Intensity: " << stats.avg_arithmetic_intensity << "\n";

    // Check for parallelism
    auto levels = graph.get_execution_levels();
    size_t max_parallel = 0;
    for (const auto& level : levels) {
        max_parallel = std::max(max_parallel, level.size());
    }
    std::cout << "  Max Parallel Kernels: " << max_parallel << "\n";

    // Check for fusion
    auto fusible = graph.find_fusible_pairs();
    std::cout << "  Fusible Pairs: " << fusible.size() << "\n";

    // Critical path length
    auto critical = graph.get_critical_path();
    std::cout << "  Critical Path Length: " << critical.size() << "\n";

    // Memory requirements
    std::cout << "  Intermediate Data: " << stats.intermediate_bytes / 1024 << " KB\n";
}
```

### Pattern 4: Conditional Graph Construction

Build different graphs based on configuration:

```cpp
KernelGraph build_model(const ModelConfig& config) {
    KernelGraph graph(config.name);

    size_t prev_layer = SIZE_MAX;

    for (const auto& layer_config : config.layers) {
        Kernel kernel;

        switch (layer_config.type) {
            case LayerType::LINEAR:
                kernel = Kernel::create_matmul(
                    config.batch_size,
                    layer_config.output_features,
                    layer_config.input_features);
                break;

            case LayerType::MLP:
                kernel = Kernel::create_mlp(
                    config.batch_size,
                    layer_config.output_features,
                    layer_config.input_features,
                    layer_config.activation,
                    layer_config.use_bias);
                break;

            // ... other layer types
        }

        size_t id = graph.add_kernel(std::move(kernel), layer_config.name);

        if (prev_layer != SIZE_MAX) {
            graph.add_edge(prev_layer, id);
        }

        prev_layer = id;
    }

    return graph;
}
```

### Pattern 5: Graph Serialization for Deployment

Save and load compiled graphs:

```cpp
#include <sw/kpu/kernel_serializer.hpp>

void save_compiled_network(const KernelGraph& graph, const std::string& path) {
    auto result = graph.compile();
    if (!result.success) {
        throw std::runtime_error("Compilation failed: " + result.error_message);
    }

    // Save the compiled program
    isa::ProgramSerializer serializer;
    serializer.save(result.program, path);
}

isa::DMProgram load_compiled_network(const std::string& path) {
    isa::ProgramSerializer serializer;
    return serializer.load(path);
}

// Usage:
KernelGraph graph = build_my_network();
save_compiled_network(graph, "model.kpubin");

// Later, in deployment:
auto program = load_compiled_network("model.kpubin");
executor.execute(program);
```

---

## Performance Considerations

### Fusion Benefits

Kernel fusion eliminates intermediate memory traffic:

| Scenario | Memory Traffic |
|----------|---------------|
| Unfused: FC1 + FC2 | Write C1 to L2, Read C1 from L2 |
| Fused: FC1→FC2 | C1 stays in L1, zero L2 traffic |

**When fusion helps most:**
- Small intermediate tensors that fit in L1
- Producer has single consumer
- Compatible dimensions and data types

### Parallelism Exploitation

Use execution levels to identify parallel opportunities:

```cpp
auto levels = graph.get_execution_levels();

for (size_t i = 0; i < levels.size(); ++i) {
    if (levels[i].size() > 1) {
        std::cout << "Level " << i << " has " << levels[i].size()
                  << " parallel kernels\n";
    }
}
```

### Memory Planning

Check intermediate data requirements:

```cpp
auto stats = graph.compute_stats();

// Intermediate bytes = data passed between kernels
// This must fit in available L2/L3 capacity
if (stats.intermediate_bytes > available_l2_capacity) {
    std::cerr << "Warning: Intermediate data exceeds L2 capacity\n";
    std::cerr << "Consider splitting graph or using streaming\n";
}
```

### Compilation Options

Tune compilation for your workload:

```cpp
KernelGraphCompileOptions opts;

// For memory-bound workloads
opts.fusion_strategy = FusionStrategy::PRODUCER_CONSUMER;
opts.enable_double_buffering = true;

// For compute-bound workloads
opts.fusion_strategy = FusionStrategy::NONE;  // Maximize parallelism
opts.insert_global_barriers = false;  // Reduce sync overhead
```

---

## Future Directions

### Planned Enhancements

1. **True Kernel Fusion**: Merge instruction streams for fused pairs, eliminating intermediate buffers entirely

2. **Automatic Parallelization**: Spawn concurrent executors for independent kernel subgraphs

3. **Memory Optimization**: Intelligent workspace allocation to minimize peak memory usage

4. **Graph Transformations**: Operator fusion, constant folding, dead code elimination

5. **Dynamic Graphs**: Support for control flow (conditionals, loops) within graphs

6. **Distributed Execution**: Partition large graphs across multiple KPU instances

### Current Limitations

- Fusion is detected but not yet fully implemented (falls back to sequential)
- No automatic memory layout optimization across kernels
- Graph must be acyclic (no recurrent connections)
- All kernels must use same data type for fusion

---

## Summary

The KernelGraph facility transforms the KPU simulator from a single-operation executor to a full neural network inference engine. By representing computations as DAGs, we enable:

- **Correct Scheduling**: Automatic topological ordering
- **Optimization Opportunities**: Fusion and parallelism detection
- **Analysis Tools**: Statistics, critical path, visualization
- **Unified Execution**: Single compiled program for entire network

This foundation supports building production-quality neural network compilers and runtimes targeting the KPU architecture.
