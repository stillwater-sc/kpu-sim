// Kernel Graph Demo
// Demonstrates multi-kernel execution using KernelGraph
//
// This example shows how to:
// 1. Create multiple kernels
// 2. Build a computation graph with data dependencies
// 3. Analyze the graph structure
// 4. Compile the graph to a single program
// 5. Execute the compiled program

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/kernel.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>

#include <iostream>
#include <iomanip>

using namespace sw::kpu;
using namespace sw::kpu::isa;

void print_separator(const std::string& title = "") {
    std::cout << "\n";
    if (!title.empty()) {
        std::cout << "=== " << title << " ";
    }
    std::cout << std::string(60 - title.length(), '=') << "\n\n";
}

// Example 1: Simple two-layer network (linear chain)
void demo_linear_chain() {
    print_separator("Example 1: Linear Chain (Two-Layer Network)");

    // Create a simple two-layer network:
    // Input [64, 256] -> FC1 -> [64, 512] -> FC2 -> [64, 128]

    KernelGraph graph("two_layer_fc");

    // Layer 1: [64, 256] @ [256, 512] -> [64, 512]
    size_t fc1 = graph.add_kernel(
        Kernel::create_matmul(64, 512, 256, DataType::FLOAT32),
        "fc1");

    // Layer 2: [64, 512] @ [512, 128] -> [64, 128]
    size_t fc2 = graph.add_kernel(
        Kernel::create_matmul(64, 128, 512, DataType::FLOAT32),
        "fc2");

    // Connect: fc1.C -> fc2.A
    graph.add_edge(fc1, fc2, "C", "A");

    // Print graph summary
    std::cout << graph.summary() << "\n";

    // Get execution order
    auto order = graph.get_execution_order();
    std::cout << "Execution order: ";
    for (size_t i = 0; i < order.size(); ++i) {
        if (i > 0) std::cout << " -> ";
        std::cout << graph.get_node(order[i]).name;
    }
    std::cout << "\n\n";

    // Compile to single program
    auto result = graph.compile();
    if (result.success) {
        std::cout << "Compilation successful!\n";
        std::cout << "  Total instructions: " << result.program.instructions.size() << "\n";
        std::cout << "  Workspace required: " << result.workspace_required / 1024 << " KB\n";
    }
}

// Example 2: MLP with activation (transformer feed-forward)
void demo_transformer_ffn() {
    print_separator("Example 2: Transformer Feed-Forward Network");

    // Transformer FFN pattern:
    // x -> FC1 (up-project) -> GELU -> FC2 (down-project) -> output

    KernelGraph graph("transformer_ffn");

    Size batch = 32;
    Size hidden = 768;
    Size intermediate = 3072;  // 4x hidden is typical

    // Up-projection with GELU: [batch, 768] @ [768, 3072] + bias + GELU
    size_t fc1 = graph.add_kernel(
        Kernel::create_mlp(batch, intermediate, hidden,
                           ActivationType::GELU, true),
        "fc1_gelu");

    // Down-projection: [batch, 3072] @ [3072, 768] + bias
    size_t fc2 = graph.add_kernel(
        Kernel::create_mlp(batch, hidden, intermediate,
                           ActivationType::NONE, true),
        "fc2");

    graph.add_edge(fc1, fc2, "C", "A");

    // Analyze
    auto stats = graph.compute_stats();
    std::cout << "Network Statistics:\n";
    std::cout << "  Total FLOPs:       " << stats.total_flops << "\n";
    std::cout << "  Input bytes:       " << stats.total_input_bytes / 1024 << " KB\n";
    std::cout << "  Output bytes:      " << stats.total_output_bytes / 1024 << " KB\n";
    std::cout << "  Intermediate data: " << stats.intermediate_bytes / 1024 << " KB\n";
    std::cout << "  Avg arith. int.:   " << std::fixed << std::setprecision(2)
              << stats.avg_arithmetic_intensity << " FLOP/byte\n\n";

    // Check for fusion opportunities
    auto fusible = graph.find_fusible_pairs();
    if (!fusible.empty()) {
        std::cout << "Fusion opportunities found:\n";
        for (const auto& [from, to] : fusible) {
            std::cout << "  " << graph.get_node(from).name << " <-> "
                      << graph.get_node(to).name << "\n";
        }
    }
    std::cout << "\n";

    // Compile
    auto result = graph.compile();
    std::cout << "Compiled program: " << result.program.instructions.size()
              << " instructions\n";
}

// Example 3: Diamond pattern (parallel branches)
void demo_diamond_pattern() {
    print_separator("Example 3: Diamond Pattern (Parallel Branches)");

    // Diamond pattern:
    //       input
    //       /   \
    //    left   right
    //       \   /
    //       merge
    //
    // This tests parallel execution opportunities

    KernelGraph graph("diamond_network");

    size_t input = graph.add_kernel(
        Kernel::create_matmul(64, 64, 128),
        "input");

    size_t left = graph.add_kernel(
        Kernel::create_matmul(64, 128, 64),
        "left_branch");

    size_t right = graph.add_kernel(
        Kernel::create_matmul(64, 128, 64),
        "right_branch");

    size_t merge = graph.add_kernel(
        Kernel::create_matmul(64, 64, 128),
        "merge");

    // Note: Both branches take input from the same source
    // In a real scenario, we'd need different outputs, but
    // for demonstration we show the graph structure
    graph.add_edge(input, left, "C", "A");
    graph.add_edge(input, right, "C", "A");
    graph.add_edge(left, merge, "C", "A");
    graph.add_edge(right, merge, "C", "B");

    // Show execution levels
    auto levels = graph.get_execution_levels();
    std::cout << "Execution Levels (nodes at same level can run in parallel):\n";
    for (size_t i = 0; i < levels.size(); ++i) {
        std::cout << "  Level " << i << ": ";
        for (size_t j = 0; j < levels[i].size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << graph.get_node(levels[i][j]).name;
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Show critical path
    auto critical = graph.get_critical_path();
    std::cout << "Critical Path: ";
    for (size_t i = 0; i < critical.size(); ++i) {
        if (i > 0) std::cout << " -> ";
        std::cout << graph.get_node(critical[i]).name;
    }
    std::cout << "\n\n";

    // Generate DOT graph for visualization
    std::cout << "DOT graph (paste into graphviz):\n";
    std::cout << graph.to_dot(true) << "\n";
}

// Example 4: Deep network with many layers
void demo_deep_network() {
    print_separator("Example 4: Deep Network (5-Layer MLP)");

    KernelGraph graph("deep_mlp");

    Size batch = 64;
    std::vector<Size> layer_sizes = {784, 512, 256, 128, 64, 10};

    std::vector<size_t> node_ids;

    // Create layers
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        Size in_features = layer_sizes[i];
        Size out_features = layer_sizes[i + 1];

        std::string name = "layer" + std::to_string(i + 1);

        // Use ReLU for hidden layers, no activation for output
        ActivationType act = (i < layer_sizes.size() - 2)
            ? ActivationType::RELU : ActivationType::NONE;

        size_t id = graph.add_kernel(
            Kernel::create_mlp(batch, out_features, in_features, act, true),
            name);

        node_ids.push_back(id);
    }

    // Connect layers
    for (size_t i = 0; i < node_ids.size() - 1; ++i) {
        graph.add_edge(node_ids[i], node_ids[i + 1], "C", "A");
    }

    // Validate
    std::string error;
    if (!graph.validate(error)) {
        std::cerr << "Validation failed: " << error << "\n";
        return;
    }

    // Stats
    auto stats = graph.compute_stats();
    std::cout << "Deep MLP Statistics:\n";
    std::cout << "  Layers:            " << stats.num_nodes << "\n";
    std::cout << "  Total FLOPs:       " << stats.total_flops << "\n";
    std::cout << "  Total instructions:" << stats.total_instructions << "\n";
    std::cout << "  Max depth:         " << stats.max_depth << "\n\n";

    // Compile and execute
    auto result = graph.compile_sequential();
    if (result.success) {
        std::cout << "Compiled to " << result.program.instructions.size()
                  << " instructions\n";

        // Create executor and run
        ResourceConfig res_config;  // Use default configuration
        ConcurrentExecutor executor(res_config);
        Cycle cycles = executor.execute(result.program);

        std::cout << "Execution completed in " << cycles << " cycles\n";

        // Performance estimate
        double freq_ghz = 1.0;  // Assuming 1 GHz
        double time_ms = static_cast<double>(cycles) / (freq_ghz * 1e6);
        double gflops = static_cast<double>(stats.total_flops) / (time_ms * 1e6);

        std::cout << "  Estimated time:    " << std::fixed << std::setprecision(3)
                  << time_ms << " ms @ " << freq_ghz << " GHz\n";
        std::cout << "  Estimated GFLOPS:  " << std::setprecision(2) << gflops << "\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  KPU Kernel Graph Demo\n";
    std::cout << "========================================\n";
    std::cout << "\nThis demo shows how to build and execute\n";
    std::cout << "multi-kernel computation graphs on the KPU.\n";

    try {
        demo_linear_chain();
        demo_transformer_ffn();
        demo_diamond_pattern();
        demo_deep_network();

        print_separator("Demo Complete");
        std::cout << "All examples executed successfully!\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
