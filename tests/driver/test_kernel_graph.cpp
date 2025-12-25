// Test suite for Kernel Graph
// Tests multi-kernel DAG representation and compilation

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/kernel.hpp>

#include <string>
#include <vector>

using namespace sw::kpu;

// ============================================================================
// Node and Edge Management Tests
// ============================================================================

TEST_CASE("KernelGraph node management", "[kernel_graph][nodes]") {
    KernelGraph graph("test_graph");

    SECTION("Add single kernel") {
        Kernel k = Kernel::create_matmul(64, 64, 64, DataType::FLOAT32);
        size_t id = graph.add_kernel(std::move(k), "layer1");

        REQUIRE(graph.num_nodes() == 1);
        REQUIRE(graph.has_node(id));
        REQUIRE(graph.get_node(id).name == "layer1");
        REQUIRE(graph.get_kernel(id).M() == 64);
    }

    SECTION("Add multiple kernels") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer2");
        size_t k3 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "layer3");

        REQUIRE(graph.num_nodes() == 3);
        REQUIRE(graph.has_node(k1));
        REQUIRE(graph.has_node(k2));
        REQUIRE(graph.has_node(k3));

        auto ids = graph.node_ids();
        REQUIRE(ids.size() == 3);
    }

    SECTION("Get non-existent node throws") {
        REQUIRE_THROWS_AS(graph.get_node(999), std::out_of_range);
    }

    SECTION("Add invalid kernel throws") {
        Kernel invalid_kernel;  // Default constructor creates invalid kernel
        REQUIRE_THROWS_AS(graph.add_kernel(std::move(invalid_kernel)), std::invalid_argument);
    }
}

TEST_CASE("KernelGraph edge management", "[kernel_graph][edges]") {
    KernelGraph graph;

    size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "layer1");
    size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer2");
    size_t k3 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "layer3");

    SECTION("Add single edge") {
        size_t edge_id = graph.add_edge(k1, k2, "C", "A");

        REQUIRE(graph.num_edges() == 1);
        REQUIRE(graph.get_edge(edge_id).from_node == k1);
        REQUIRE(graph.get_edge(edge_id).to_node == k2);
        REQUIRE(graph.get_edge(edge_id).output_name == "C");
        REQUIRE(graph.get_edge(edge_id).input_name == "A");
    }

    SECTION("Add chain of edges") {
        graph.add_edge(k1, k2, "C", "A");
        graph.add_edge(k2, k3, "C", "A");

        REQUIRE(graph.num_edges() == 2);
        REQUIRE(graph.outgoing_edges(k1).size() == 1);
        REQUIRE(graph.incoming_edges(k2).size() == 1);
        REQUIRE(graph.outgoing_edges(k2).size() == 1);
        REQUIRE(graph.incoming_edges(k3).size() == 1);
    }

    SECTION("Self-loop throws") {
        REQUIRE_THROWS_AS(graph.add_edge(k1, k1, "C", "A"), std::invalid_argument);
    }

    SECTION("Edge to non-existent node throws") {
        REQUIRE_THROWS_AS(graph.add_edge(k1, 999, "C", "A"), std::invalid_argument);
        REQUIRE_THROWS_AS(graph.add_edge(999, k2, "C", "A"), std::invalid_argument);
    }

    SECTION("Cycle detection - direct cycle") {
        graph.add_edge(k1, k2);
        REQUIRE_THROWS_AS(graph.add_edge(k2, k1), std::invalid_argument);
    }

    SECTION("Cycle detection - indirect cycle") {
        graph.add_edge(k1, k2);
        graph.add_edge(k2, k3);
        REQUIRE_THROWS_AS(graph.add_edge(k3, k1), std::invalid_argument);
    }

    SECTION("Would create cycle check") {
        graph.add_edge(k1, k2);
        graph.add_edge(k2, k3);

        REQUIRE_FALSE(graph.would_create_cycle(k1, k3));  // k1 -> k3 is fine
        REQUIRE(graph.would_create_cycle(k3, k1));        // k3 -> k1 would create cycle
        REQUIRE(graph.would_create_cycle(k2, k1));        // k2 -> k1 would create cycle
    }
}

// ============================================================================
// Graph Properties Tests
// ============================================================================

TEST_CASE("KernelGraph properties", "[kernel_graph][properties]") {
    KernelGraph graph("test");

    SECTION("Empty graph") {
        REQUIRE(graph.empty());
        REQUIRE(graph.num_nodes() == 0);
        REQUIRE(graph.num_edges() == 0);

        std::string error;
        REQUIRE_FALSE(graph.validate(error));
        REQUIRE(error == "Graph is empty");
    }

    SECTION("Input and output nodes") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "input1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "input2");
        size_t k3 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "middle");
        size_t k4 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "output");

        graph.add_edge(k1, k3, "C", "A");
        graph.add_edge(k2, k3, "C", "B");
        graph.add_edge(k3, k4, "C", "A");

        auto inputs = graph.input_nodes();
        REQUIRE(inputs.size() == 2);
        REQUIRE((inputs[0] == k1 || inputs[0] == k2));

        auto outputs = graph.output_nodes();
        REQUIRE(outputs.size() == 1);
        REQUIRE(outputs[0] == k4);
    }

    SECTION("Graph validation") {
        graph.add_kernel(Kernel::create_matmul(64, 64, 64), "layer1");

        std::string error;
        REQUIRE(graph.validate(error));
    }

    SECTION("Graph stats") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(128, 128, 128), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(128, 256, 128), "layer2");
        graph.add_edge(k1, k2, "C", "A");

        auto stats = graph.compute_stats();

        REQUIRE(stats.num_nodes == 2);
        REQUIRE(stats.num_edges == 1);
        REQUIRE(stats.num_input_nodes == 1);
        REQUIRE(stats.num_output_nodes == 1);
        REQUIRE(stats.max_depth == 1);
        REQUIRE(stats.total_instructions > 0);
        REQUIRE(stats.total_flops > 0);
    }
}

// ============================================================================
// Execution Order Tests
// ============================================================================

TEST_CASE("KernelGraph execution order", "[kernel_graph][execution]") {
    KernelGraph graph;

    SECTION("Linear chain order") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer2");
        size_t k3 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "layer3");

        graph.add_edge(k1, k2);
        graph.add_edge(k2, k3);

        auto order = graph.get_execution_order();

        REQUIRE(order.size() == 3);
        // k1 must come before k2, k2 must come before k3
        auto pos_k1 = std::find(order.begin(), order.end(), k1);
        auto pos_k2 = std::find(order.begin(), order.end(), k2);
        auto pos_k3 = std::find(order.begin(), order.end(), k3);

        REQUIRE(pos_k1 < pos_k2);
        REQUIRE(pos_k2 < pos_k3);
    }

    SECTION("Diamond pattern order") {
        //     k1
        //    /  \
        //   k2   k3
        //    \  /
        //     k4
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "top");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "left");
        size_t k3 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "right");
        size_t k4 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "bottom");

        graph.add_edge(k1, k2);
        graph.add_edge(k1, k3);
        graph.add_edge(k2, k4);
        graph.add_edge(k3, k4);

        auto order = graph.get_execution_order();

        REQUIRE(order.size() == 4);

        // k1 must be first, k4 must be last
        REQUIRE(order.front() == k1);
        REQUIRE(order.back() == k4);
    }

    SECTION("Execution levels") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "input1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "input2");
        size_t k3 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "merge");
        size_t k4 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "output");

        graph.add_edge(k1, k3);
        graph.add_edge(k2, k3);
        graph.add_edge(k3, k4);

        auto levels = graph.get_execution_levels();

        REQUIRE(levels.size() == 3);
        REQUIRE(levels[0].size() == 2);  // k1, k2 at level 0
        REQUIRE(levels[1].size() == 1);  // k3 at level 1
        REQUIRE(levels[2].size() == 1);  // k4 at level 2
    }

    SECTION("Critical path") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "input");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "branch1");
        size_t k3 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "branch2a");
        size_t k4 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "branch2b");
        size_t k5 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "merge");

        graph.add_edge(k1, k2);
        graph.add_edge(k1, k3);
        graph.add_edge(k3, k4);
        graph.add_edge(k2, k5);
        graph.add_edge(k4, k5);

        auto critical = graph.get_critical_path();

        // Critical path: k1 -> k3 -> k4 -> k5 (length 4)
        REQUIRE(critical.size() == 4);
        REQUIRE(critical.front() == k1);
        REQUIRE(critical.back() == k5);
    }
}

// ============================================================================
// Fusion Tests
// ============================================================================

TEST_CASE("KernelGraph fusion", "[kernel_graph][fusion]") {
    KernelGraph graph;

    SECTION("Find fusible pairs - simple chain") {
        // Two matmuls: C1 = A1 @ B1, then C2 = C1 @ B2
        // For fusion: M must match, and N of first must match K of second
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "layer2");

        graph.add_edge(k1, k2, "C", "A");

        auto fusible = graph.find_fusible_pairs();
        REQUIRE(fusible.size() == 1);
        REQUIRE(fusible[0].first == k1);
        REQUIRE(fusible[0].second == k2);
    }

    SECTION("Can fuse check") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "layer2");

        graph.add_edge(k1, k2);

        REQUIRE(graph.can_fuse(k1, k2));
    }

    SECTION("Cannot fuse - dimension mismatch") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(32, 256, 128), "layer2");  // M mismatch

        graph.add_edge(k1, k2);

        REQUIRE_FALSE(graph.can_fuse(k1, k2));
    }

    SECTION("Cannot fuse - multiple inputs") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "input1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "input2");
        size_t k3 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "merge");

        graph.add_edge(k1, k3);
        graph.add_edge(k2, k3);

        // k3 has two inputs, so it cannot be fused
        REQUIRE_FALSE(graph.can_fuse(k1, k3));
        REQUIRE_FALSE(graph.can_fuse(k2, k3));
    }

    SECTION("Mark for fusion") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "layer2");

        graph.add_edge(k1, k2);

        REQUIRE(graph.mark_for_fusion(k1, k2));
        REQUIRE(graph.get_node(k1).is_fused);
        REQUIRE(graph.get_node(k2).is_fused);

        graph.clear_fusion_marks();
        REQUIRE_FALSE(graph.get_node(k1).is_fused);
        REQUIRE_FALSE(graph.get_node(k2).is_fused);
    }
}

// ============================================================================
// Compilation Tests
// ============================================================================

TEST_CASE("KernelGraph compilation", "[kernel_graph][compile]") {
    KernelGraph graph("test_network");

    SECTION("Compile single kernel") {
        graph.add_kernel(Kernel::create_matmul(128, 128, 128), "single");

        auto result = graph.compile();

        REQUIRE(result.success);
        REQUIRE(!result.program.instructions.empty());
        REQUIRE(result.execution_order.size() == 1);
    }

    SECTION("Compile linear chain") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer2");

        graph.add_edge(k1, k2);

        auto result = graph.compile();

        REQUIRE(result.success);
        REQUIRE(result.execution_order.size() == 2);
        REQUIRE(result.execution_order[0] == k1);
        REQUIRE(result.execution_order[1] == k2);

        // Should have instructions from both kernels
        REQUIRE(result.program.instructions.size() > 0);

        // Should have a HALT at the end
        REQUIRE(result.program.instructions.back().opcode == isa::DMOpcode::HALT);
    }

    SECTION("Compile with options") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "layer2");

        graph.add_edge(k1, k2);

        KernelGraphCompileOptions opts;
        opts.fusion_strategy = FusionStrategy::NONE;
        opts.insert_global_barriers = true;

        auto result = graph.compile(opts);

        REQUIRE(result.success);
        REQUIRE(result.program.name == "test_network");
    }

    SECTION("Compile sequential") {
        size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "layer1");
        size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "layer2");

        graph.add_edge(k1, k2);

        auto result = graph.compile_sequential();

        REQUIRE(result.success);
        REQUIRE(result.execution_order.size() == 2);
    }

    SECTION("Compile empty graph fails") {
        KernelGraph empty_graph;
        auto result = empty_graph.compile();

        REQUIRE_FALSE(result.success);
        REQUIRE(!result.error_message.empty());
    }
}

// ============================================================================
// Visualization Tests
// ============================================================================

TEST_CASE("KernelGraph visualization", "[kernel_graph][viz]") {
    KernelGraph graph("mlp_network");

    size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "fc1");
    size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "fc2");

    graph.add_edge(k1, k2, "C", "A");

    SECTION("Summary output") {
        std::string summary = graph.summary();

        REQUIRE(summary.find("mlp_network") != std::string::npos);
        REQUIRE(summary.find("Nodes: 2") != std::string::npos);
        REQUIRE(summary.find("Edges: 1") != std::string::npos);
        REQUIRE(summary.find("fc1") != std::string::npos);
        REQUIRE(summary.find("fc2") != std::string::npos);
    }

    SECTION("DOT output") {
        std::string dot = graph.to_dot(true);

        REQUIRE(dot.find("digraph KernelGraph") != std::string::npos);
        REQUIRE(dot.find("fc1") != std::string::npos);
        REQUIRE(dot.find("fc2") != std::string::npos);
        REQUIRE(dot.find("->") != std::string::npos);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_CASE("KernelGraph MLP network", "[kernel_graph][integration]") {
    KernelGraph graph("two_layer_mlp");

    // Create a two-layer MLP: input -> hidden -> output
    // Layer 1: [batch, 768] @ [768, 3072] + bias + GELU -> [batch, 3072]
    // Layer 2: [batch, 3072] @ [3072, 768] + bias -> [batch, 768]

    Size batch = 32;
    Size hidden = 768;
    Size intermediate = 3072;

    size_t fc1 = graph.add_kernel(
        Kernel::create_mlp(batch, intermediate, hidden,
                           ActivationType::GELU, true),
        "fc1_gelu");

    size_t fc2 = graph.add_kernel(
        Kernel::create_mlp(batch, hidden, intermediate,
                           ActivationType::NONE, true),
        "fc2");

    graph.add_edge(fc1, fc2, "C", "A");

    // Validate
    std::string error;
    REQUIRE(graph.validate(error));

    // Check stats
    auto stats = graph.compute_stats();
    REQUIRE(stats.num_nodes == 2);
    REQUIRE(stats.num_edges == 1);
    REQUIRE(stats.total_flops > 0);

    // Compile
    auto result = graph.compile();
    REQUIRE(result.success);
    REQUIRE(result.execution_order.size() == 2);
    REQUIRE(result.execution_order[0] == fc1);
    REQUIRE(result.execution_order[1] == fc2);

    // Check program has content
    REQUIRE(result.program.instructions.size() > 10);
}

TEST_CASE("KernelGraph residual connection pattern", "[kernel_graph][integration]") {
    KernelGraph graph("residual_block");

    // Residual pattern:
    //   input -> fc1 -> fc2
    //         \       /
    //          +-----+ (conceptual residual, but we can't add tensors yet)

    size_t input = graph.add_kernel(Kernel::create_matmul(64, 64, 64), "input");
    size_t fc1 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "fc1");
    size_t fc2 = graph.add_kernel(Kernel::create_matmul(64, 64, 128), "fc2");

    graph.add_edge(input, fc1, "C", "A");
    graph.add_edge(fc1, fc2, "C", "A");

    auto levels = graph.get_execution_levels();
    REQUIRE(levels.size() == 3);

    auto critical = graph.get_critical_path();
    REQUIRE(critical.size() == 3);

    auto result = graph.compile();
    REQUIRE(result.success);
}
