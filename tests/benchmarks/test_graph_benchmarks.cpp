// Kernel Graph Benchmark Tests
// Tests for multi-kernel graph execution performance

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/benchmark/benchmark.hpp>
#include <sw/kpu/kernel_graph.hpp>

#include <iostream>

using namespace sw::benchmark;
using namespace sw::kpu;

TEST_CASE("Two-layer MLP graph benchmark", "[benchmark][graph]") {
    BenchmarkHarness harness;

    // Create two-layer MLP graph
    KernelGraph graph("two_layer_mlp");

    Size batch = 64;
    Size in_features = 256;
    Size hidden = 512;
    Size out_features = 128;

    size_t fc1 = graph.add_kernel(
        Kernel::create_mlp(batch, hidden, in_features, ActivationType::RELU, true),
        "fc1_relu"
    );

    size_t fc2 = graph.add_kernel(
        Kernel::create_mlp(batch, out_features, hidden, ActivationType::NONE, true),
        "fc2"
    );

    graph.add_edge(fc1, fc2, "C", "A");

    auto result = harness.benchmark_graph(graph, "two_layer_mlp");

    std::cout << "Two-Layer MLP Graph:" << std::endl;
    std::cout << result.to_detailed_string() << std::endl;

    REQUIRE(result.cycles > 0);
    REQUIRE(result.gflops > 0);
}

TEST_CASE("Deep MLP graph benchmark", "[benchmark][graph]") {
    BenchmarkHarness harness;

    // Create deep MLP: 784 -> 512 -> 256 -> 128 -> 64 -> 10
    std::vector<Size> layer_sizes = {784, 512, 256, 128, 64, 10};
    Size batch = 64;

    KernelGraph graph("deep_mlp");
    std::vector<size_t> node_ids;

    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        Size in_f = layer_sizes[i];
        Size out_f = layer_sizes[i + 1];

        ActivationType act = (i < layer_sizes.size() - 2)
            ? ActivationType::RELU : ActivationType::NONE;

        size_t id = graph.add_kernel(
            Kernel::create_mlp(batch, out_f, in_f, act, true),
            "layer" + std::to_string(i + 1)
        );
        node_ids.push_back(id);
    }

    // Connect layers
    for (size_t i = 0; i < node_ids.size() - 1; i++) {
        graph.add_edge(node_ids[i], node_ids[i + 1], "C", "A");
    }

    auto result = harness.benchmark_graph(graph, "deep_mlp_5layer");

    std::cout << "Deep MLP (5 layers):" << std::endl;
    std::cout << result.to_detailed_string() << std::endl;

    REQUIRE(result.cycles > 0);
}

TEST_CASE("Transformer FFN block benchmark", "[benchmark][graph][transformer]") {
    BenchmarkHarness harness;

    // Transformer FFN: up-project -> GELU -> down-project
    Size batch_seq = 32 * 512;  // batch=32, seq=512
    Size hidden = 768;
    Size intermediate = 3072;

    KernelGraph graph("transformer_ffn");

    size_t up = graph.add_kernel(
        Kernel::create_mlp(batch_seq, intermediate, hidden, ActivationType::GELU, true),
        "up_project_gelu"
    );

    size_t down = graph.add_kernel(
        Kernel::create_mlp(batch_seq, hidden, intermediate, ActivationType::NONE, true),
        "down_project"
    );

    graph.add_edge(up, down, "C", "A");

    auto result = harness.benchmark_graph(graph, "transformer_ffn");

    std::cout << "Transformer FFN Block:" << std::endl;
    std::cout << result.to_detailed_string() << std::endl;

    REQUIRE(result.cycles > 0);

    // Calculate expected matmul FLOPs (MLP also includes bias and activation)
    uint64_t matmul_flops = 2ULL * batch_seq * intermediate * hidden +  // up
                            2ULL * batch_seq * hidden * intermediate;    // down
    // MLP kernels have slightly more FLOPs due to bias and activation
    // Allow 1% tolerance
    REQUIRE(result.flops >= matmul_flops);
    REQUIRE(result.flops < matmul_flops * 1.01);
}

TEST_CASE("Diamond pattern graph benchmark", "[benchmark][graph]") {
    BenchmarkHarness harness;

    // Diamond pattern with parallel branches
    KernelGraph graph("diamond");

    size_t input = graph.add_kernel(
        Kernel::create_matmul(64, 64, 128),
        "input"
    );

    size_t left = graph.add_kernel(
        Kernel::create_matmul(64, 128, 64),
        "left_branch"
    );

    size_t right = graph.add_kernel(
        Kernel::create_matmul(64, 128, 64),
        "right_branch"
    );

    size_t merge = graph.add_kernel(
        Kernel::create_matmul(64, 64, 128),
        "merge"
    );

    graph.add_edge(input, left, "C", "A");
    graph.add_edge(input, right, "C", "A");
    graph.add_edge(left, merge, "C", "A");
    graph.add_edge(right, merge, "C", "B");

    auto result = harness.benchmark_graph(graph, "diamond_pattern");

    std::cout << "Diamond Pattern Graph:" << std::endl;
    std::cout << result.to_detailed_string() << std::endl;

    REQUIRE(result.cycles > 0);

    // Check parallel execution levels
    auto levels = graph.get_execution_levels();
    REQUIRE(levels.size() == 3);  // input, [left, right], merge
    REQUIRE(levels[1].size() == 2);  // left and right in parallel
}

TEST_CASE("Graph vs individual kernels comparison", "[benchmark][graph]") {
    BenchmarkHarness harness;

    Size batch = 64;
    Size hidden = 256;
    Size intermediate = 512;

    // Run as individual kernels
    auto k1_result = harness.benchmark_mlp(batch, intermediate, hidden,
                                            ActivationType::RELU, true);
    auto k2_result = harness.benchmark_mlp(batch, hidden, intermediate,
                                            ActivationType::NONE, true);

    Cycle individual_cycles = k1_result.cycles + k2_result.cycles;

    // Run as graph
    KernelGraph graph("two_layer");
    size_t n1 = graph.add_kernel(
        Kernel::create_mlp(batch, intermediate, hidden, ActivationType::RELU, true),
        "fc1"
    );
    size_t n2 = graph.add_kernel(
        Kernel::create_mlp(batch, hidden, intermediate, ActivationType::NONE, true),
        "fc2"
    );
    graph.add_edge(n1, n2, "C", "A");

    auto graph_result = harness.benchmark_graph(graph, "graph_two_layer");

    std::cout << "\n=== Graph vs Individual Kernels ===" << std::endl;
    std::cout << "Individual kernel 1: " << k1_result.cycles << " cycles" << std::endl;
    std::cout << "Individual kernel 2: " << k2_result.cycles << " cycles" << std::endl;
    std::cout << "Individual total:    " << individual_cycles << " cycles" << std::endl;
    std::cout << "Graph execution:     " << graph_result.cycles << " cycles" << std::endl;

    double overhead = (static_cast<double>(graph_result.cycles) / individual_cycles - 1.0) * 100;
    std::cout << "Graph overhead:      " << std::fixed << std::setprecision(1)
              << overhead << "%" << std::endl;

    // Graph should not have excessive overhead (< 20%)
    REQUIRE(graph_result.cycles <= individual_cycles * 1.2);
}

TEST_CASE("Graph depth scaling", "[benchmark][graph][scaling]") {
    BenchmarkHarness harness;

    std::vector<int> depths = {2, 4, 6, 8};
    Size batch = 64;
    Size width = 128;

    std::cout << "\n=== Graph Depth Scaling ===" << std::endl;
    std::cout << "Layer width: " << width << std::endl;

    for (int depth : depths) {
        KernelGraph graph("depth_" + std::to_string(depth));

        std::vector<size_t> nodes;
        for (int i = 0; i < depth; i++) {
            ActivationType act = (i < depth - 1) ? ActivationType::RELU : ActivationType::NONE;
            size_t id = graph.add_kernel(
                Kernel::create_mlp(batch, width, width, act, true),
                "layer" + std::to_string(i)
            );
            nodes.push_back(id);
        }

        for (size_t i = 0; i < nodes.size() - 1; i++) {
            graph.add_edge(nodes[i], nodes[i + 1], "C", "A");
        }

        auto result = harness.benchmark_graph(graph, "depth_" + std::to_string(depth));

        double cycles_per_layer = static_cast<double>(result.cycles) / depth;
        std::cout << "Depth " << depth << ": " << result.cycles << " cycles ("
                  << std::fixed << std::setprecision(0) << cycles_per_layer
                  << " per layer), " << result.gflops << " GFLOPS" << std::endl;

        REQUIRE(result.cycles > 0);
    }
}
