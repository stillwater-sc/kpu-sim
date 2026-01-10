// Test suite for ComputeGraph and GraphRunner
// Tests multi-layer graph execution API

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/runtime/graph.hpp>
#include <sw/kpu/kernel.hpp>

#include <vector>
#include <cmath>

using namespace sw::runtime;
using namespace sw::kpu;

// ============================================================================
// Test Fixture
// ============================================================================

class GraphTestFixture {
protected:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> simulator;
    std::unique_ptr<KPURuntime> runtime;

    GraphTestFixture() {
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 64;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 128;
        config.l2_bank_count = 8;
        config.l2_bank_capacity_kb = 64;
        config.page_buffer_count = 2;
        config.page_buffer_capacity_kb = 64;
        config.l1_buffer_count = 4;
        config.l1_buffer_capacity_kb = 64;
        config.dma_engine_count = 2;
        config.block_mover_count = 4;
        config.streamer_count = 8;
        config.processor_array_rows = 16;
        config.processor_array_cols = 16;
        config.use_systolic_array_mode = true;

        simulator = std::make_unique<KPUSimulator>(config);
        runtime = std::make_unique<KPURuntime>(simulator.get());
    }
};

// ============================================================================
// ComputeGraph Construction Tests
// ============================================================================

TEST_CASE("ComputeGraph construction", "[graph]") {
    SECTION("Empty graph") {
        ComputeGraph graph("test_graph");
        REQUIRE(graph.name() == "test_graph");
        REQUIRE(graph.num_nodes() == 0);
        REQUIRE(graph.num_tensors() == 0);
        REQUIRE_FALSE(graph.is_finalized());
    }

    SECTION("Add input tensor") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        REQUIRE(graph.num_tensors() == 1);
        REQUIRE(graph.input_names().size() == 1);
        REQUIRE(graph.input_names()[0] == "input");

        const auto* tensor = graph.find_tensor("input");
        REQUIRE(tensor != nullptr);
        REQUIRE(tensor->is_graph_input);
        REQUIRE(tensor->shape == std::vector<Size>{64, 784});
    }

    SECTION("Duplicate input throws") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});
        REQUIRE_THROWS_AS(graph.add_input("input", {64, 784}), std::runtime_error);
    }
}

// ============================================================================
// ComputeGraph Node Tests
// ============================================================================

TEST_CASE("ComputeGraph nodes", "[graph][nodes]") {
    SECTION("Add single node") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        auto kernel = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
        size_t idx = graph.add_node("fc1", kernel, {"input"}, {"h1"});

        REQUIRE(idx == 0);
        REQUIRE(graph.num_nodes() == 1);

        const auto* node = graph.find_node("fc1");
        REQUIRE(node != nullptr);
        REQUIRE(node->name == "fc1");
        REQUIRE(node->inputs.size() == 1);
        REQUIRE(node->outputs.size() == 1);
    }

    SECTION("Add multiple nodes") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        auto layer1 = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
        auto layer2 = Kernel::create_mlp(64, 128, 256, ActivationType::RELU, true);
        auto layer3 = Kernel::create_mlp(64, 10, 128, ActivationType::NONE, true);

        graph.add_node("fc1", layer1, {"input"}, {"h1"});
        graph.add_node("fc2", layer2, {"h1"}, {"h2"});
        graph.add_node("fc3", layer3, {"h2"}, {"output"});

        REQUIRE(graph.num_nodes() == 3);

        // Check tensor creation
        REQUIRE(graph.find_tensor("h1") != nullptr);
        REQUIRE(graph.find_tensor("h2") != nullptr);
        REQUIRE(graph.find_tensor("output") != nullptr);
    }

    SECTION("Unknown input tensor throws") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        auto kernel = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
        REQUIRE_THROWS_AS(
            graph.add_node("fc1", kernel, {"unknown"}, {"h1"}),
            std::runtime_error);
    }

    SECTION("Duplicate node name throws") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        auto kernel = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
        graph.add_node("fc1", kernel, {"input"}, {"h1"});
        REQUIRE_THROWS_AS(
            graph.add_node("fc1", kernel, {"input"}, {"h2"}),
            std::runtime_error);
    }
}

// ============================================================================
// ComputeGraph Finalization Tests
// ============================================================================

TEST_CASE("ComputeGraph finalization", "[graph][finalize]") {
    SECTION("Finalize valid graph") {
        ComputeGraph graph("mlp_3layer");
        graph.add_input("input", {64, 784});

        auto layer1 = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
        auto layer2 = Kernel::create_mlp(64, 128, 256, ActivationType::RELU, true);
        auto layer3 = Kernel::create_mlp(64, 10, 128, ActivationType::NONE, true);

        graph.add_node("fc1", layer1, {"input"}, {"h1"});
        graph.add_node("fc2", layer2, {"h1"}, {"h2"});
        graph.add_node("fc3", layer3, {"h2"}, {"output"});
        graph.add_output("output");

        REQUIRE_NOTHROW(graph.finalize());
        REQUIRE(graph.is_finalized());
    }

    SECTION("Execution order is topological") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        auto layer1 = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
        auto layer2 = Kernel::create_mlp(64, 128, 256, ActivationType::RELU, true);
        auto layer3 = Kernel::create_mlp(64, 10, 128, ActivationType::NONE, true);

        graph.add_node("fc1", layer1, {"input"}, {"h1"});
        graph.add_node("fc2", layer2, {"h1"}, {"h2"});
        graph.add_node("fc3", layer3, {"h2"}, {"output"});
        graph.add_output("output");
        graph.finalize();

        const auto& order = graph.execution_order();
        REQUIRE(order.size() == 3);

        // Verify topological order: fc1 before fc2 before fc3
        size_t fc1_pos = std::find(order.begin(), order.end(), 0) - order.begin();
        size_t fc2_pos = std::find(order.begin(), order.end(), 1) - order.begin();
        size_t fc3_pos = std::find(order.begin(), order.end(), 2) - order.begin();

        REQUIRE(fc1_pos < fc2_pos);
        REQUIRE(fc2_pos < fc3_pos);
    }

    SECTION("Empty graph fails validation") {
        ComputeGraph graph;
        REQUIRE_THROWS_AS(graph.finalize(), std::runtime_error);
    }

    SECTION("Graph without output fails validation") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        auto kernel = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
        graph.add_node("fc1", kernel, {"input"}, {"h1"});

        REQUIRE_THROWS_AS(graph.finalize(), std::runtime_error);
    }

    SECTION("Cannot modify finalized graph") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
        graph.add_node("fc1", kernel, {"input"}, {"output"});
        graph.add_output("output");
        graph.finalize();

        REQUIRE_THROWS_AS(graph.add_input("x", {64, 32}), std::runtime_error);
        REQUIRE_THROWS_AS(graph.add_node("fc2", kernel, {"output"}, {"y"}), std::runtime_error);
    }
}

// ============================================================================
// ComputeGraph Memory Analysis Tests
// ============================================================================

TEST_CASE("ComputeGraph memory analysis", "[graph][memory]") {
    SECTION("Total memory calculation") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});  // 64 * 784 * 4 = 200,704 bytes

        auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
        graph.add_node("fc1", kernel, {"input"}, {"output"});
        graph.add_output("output");
        graph.finalize();

        Size total = graph.total_memory_bytes();
        REQUIRE(total > 0);

        // Should include input and output at minimum
        Size input_bytes = 64 * 784 * sizeof(float);
        Size output_bytes = 64 * 10 * sizeof(float);
        REQUIRE(total >= input_bytes + output_bytes);
    }

    SECTION("Summary output") {
        ComputeGraph graph("test_mlp");
        graph.add_input("input", {64, 784});

        auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
        graph.add_node("fc1", kernel, {"input"}, {"output"});
        graph.add_output("output");
        graph.finalize();

        std::string summary = graph.summary();
        REQUIRE(summary.find("test_mlp") != std::string::npos);
        REQUIRE(summary.find("Nodes: 1") != std::string::npos);
        REQUIRE(summary.find("Finalized: yes") != std::string::npos);
    }
}

// ============================================================================
// GraphRunner Construction Tests
// ============================================================================

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner construction", "[runner]") {
    SECTION("Basic construction") {
        GraphRunner runner(runtime.get());
        REQUIRE(runner.runtime() == runtime.get());
        REQUIRE(runner.graph() == nullptr);
        REQUIRE_FALSE(runner.is_allocated());
    }

    SECTION("Null runtime throws") {
        REQUIRE_THROWS_AS(GraphRunner(nullptr), std::invalid_argument);
    }
}

// ============================================================================
// GraphRunner Setup Tests
// ============================================================================

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner setup", "[runner][setup]") {
    SECTION("Set finalized graph") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});
        auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
        graph.add_node("fc1", kernel, {"input"}, {"output"});
        graph.add_output("output");
        graph.finalize();

        GraphRunner runner(runtime.get());
        REQUIRE_NOTHROW(runner.set_graph(graph));
        REQUIRE(runner.graph() != nullptr);
    }

    SECTION("Set unfinalized graph throws") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});

        GraphRunner runner(runtime.get());
        REQUIRE_THROWS_AS(runner.set_graph(graph), std::invalid_argument);
    }

    SECTION("Allocate memory") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});
        auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
        graph.add_node("fc1", kernel, {"input"}, {"output"});
        graph.add_output("output");
        graph.finalize();

        GraphRunner runner(runtime.get());
        runner.set_graph(graph);
        REQUIRE_NOTHROW(runner.allocate());
        REQUIRE(runner.is_allocated());
        REQUIRE(runner.total_allocated_bytes() > 0);
    }

    SECTION("Tensor addresses available after allocation") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});
        auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
        graph.add_node("fc1", kernel, {"input"}, {"output"});
        graph.add_output("output");
        graph.finalize();

        GraphRunner runner(runtime.get());
        runner.set_graph(graph);
        runner.allocate();

        REQUIRE(runner.get_tensor_address("input") != 0);
        REQUIRE(runner.get_tensor_address("output") != 0);
        REQUIRE(runner.get_tensor_address("unknown") == 0);
    }
}

// ============================================================================
// GraphRunner Input/Output Tests
// ============================================================================

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner input/output", "[runner][io]") {
    // Setup a simple graph
    ComputeGraph graph;
    graph.add_input("input", {64, 784});
    auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
    graph.add_node("fc1", kernel, {"input"}, {"output"});
    graph.add_output("output");
    graph.finalize();

    GraphRunner runner(runtime.get());
    runner.set_graph(graph);
    runner.allocate();

    SECTION("Set input data") {
        std::vector<float> input(64 * 784, 1.0f);
        REQUIRE_NOTHROW(runner.set_input("input", input.data(), input.size() * sizeof(float)));
    }

    SECTION("Set input with shape validation") {
        std::vector<float> input(64 * 784, 1.0f);
        REQUIRE_NOTHROW(runner.set_input("input", input.data(), {64, 784}));
    }

    SECTION("Set input with wrong shape throws") {
        std::vector<float> input(128 * 784, 1.0f);
        REQUIRE_THROWS_AS(runner.set_input("input", input.data(), {128, 784}), std::invalid_argument);
    }

    SECTION("Set non-input tensor throws") {
        std::vector<float> data(64 * 10, 1.0f);
        REQUIRE_THROWS_AS(runner.set_input("output", data.data(), data.size() * sizeof(float)),
                          std::invalid_argument);
    }

    SECTION("Get output data") {
        std::vector<float> input(64 * 784, 1.0f);
        runner.set_input("input", input.data(), input.size() * sizeof(float));

        // Execute first
        runner.run();

        std::vector<float> output(64 * 10);
        REQUIRE_NOTHROW(runner.get_output("output", output.data()));
    }

    SECTION("Get non-output tensor throws") {
        std::vector<float> data(64 * 784);
        REQUIRE_THROWS_AS(runner.get_output("input", data.data()), std::invalid_argument);
    }
}

// ============================================================================
// GraphRunner Execution Tests
// ============================================================================

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner single layer execution", "[runner][exec]") {
    ComputeGraph graph;
    graph.add_input("input", {64, 784});
    auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::RELU, true);
    graph.add_node("fc1", kernel, {"input"}, {"output"});
    graph.add_output("output");
    graph.finalize();

    GraphRunner runner(runtime.get());
    runner.set_graph(graph);
    runner.allocate();

    std::vector<float> input(64 * 784, 0.5f);
    runner.set_input("input", input.data(), input.size() * sizeof(float));

    auto result = runner.run();

    REQUIRE(result.success);
    REQUIRE(result.total_cycles > 0);
    REQUIRE(result.total_time_ms > 0.0);
    REQUIRE(result.layer_profiles.size() == 1);
    REQUIRE(result.layer_profiles[0].name == "fc1");
}

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner multi-layer MLP execution", "[runner][exec][mlp]") {
    // 3-layer MLP: 784 -> 256 -> 128 -> 10
    ComputeGraph graph("mlp_3layer");
    graph.add_input("input", {64, 784});

    auto layer1 = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
    auto layer2 = Kernel::create_mlp(64, 128, 256, ActivationType::RELU, true);
    auto layer3 = Kernel::create_mlp(64, 10, 128, ActivationType::NONE, true);

    graph.add_node("fc1", layer1, {"input"}, {"h1"});
    graph.add_node("fc2", layer2, {"h1"}, {"h2"});
    graph.add_node("fc3", layer3, {"h2"}, {"output"});
    graph.add_output("output");
    graph.finalize();

    GraphRunner runner(runtime.get());
    runner.set_graph(graph);
    runner.allocate();

    SECTION("Execute full graph") {
        std::vector<float> input(64 * 784, 0.5f);
        runner.set_input("input", input.data(), input.size() * sizeof(float));

        auto result = runner.run();

        REQUIRE(result.success);
        REQUIRE(result.total_cycles > 0);
        REQUIRE(result.layer_profiles.size() == 3);

        // Verify layer names in order
        REQUIRE(result.layer_profiles[0].name == "fc1");
        REQUIRE(result.layer_profiles[1].name == "fc2");
        REQUIRE(result.layer_profiles[2].name == "fc3");

        // Each layer should have non-zero cycles
        for (const auto& profile : result.layer_profiles) {
            REQUIRE(profile.cycles > 0);
        }

        // Total should equal sum of layers
        Cycle sum = 0;
        for (const auto& profile : result.layer_profiles) {
            sum += profile.cycles;
        }
        REQUIRE(result.total_cycles == sum);
    }

    SECTION("Get output after execution") {
        std::vector<float> input(64 * 784, 0.5f);
        runner.set_input("input", input.data(), input.size() * sizeof(float));

        runner.run();

        std::vector<float> output(64 * 10);
        REQUIRE_NOTHROW(runner.get_output("output", output.data()));
    }
}

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner execution errors", "[runner][exec][error]") {
    SECTION("Execute without graph") {
        GraphRunner runner(runtime.get());
        auto result = runner.run();
        REQUIRE_FALSE(result.success);
        REQUIRE(result.error.find("No graph") != std::string::npos);
    }

    SECTION("Execute without allocation") {
        ComputeGraph graph;
        graph.add_input("input", {64, 784});
        auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
        graph.add_node("fc1", kernel, {"input"}, {"output"});
        graph.add_output("output");
        graph.finalize();

        GraphRunner runner(runtime.get());
        runner.set_graph(graph);

        auto result = runner.run();
        REQUIRE_FALSE(result.success);
        REQUIRE(result.error.find("not allocated") != std::string::npos);
    }
}

// ============================================================================
// GraphRunner Release Tests
// ============================================================================

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner release", "[runner][release]") {
    ComputeGraph graph;
    graph.add_input("input", {64, 784});
    auto kernel = Kernel::create_mlp(64, 10, 784, ActivationType::NONE, true);
    graph.add_node("fc1", kernel, {"input"}, {"output"});
    graph.add_output("output");
    graph.finalize();

    GraphRunner runner(runtime.get());
    runner.set_graph(graph);
    runner.allocate();

    Size allocated = runner.total_allocated_bytes();
    REQUIRE(allocated > 0);

    runner.release();

    REQUIRE_FALSE(runner.is_allocated());
    REQUIRE(runner.total_allocated_bytes() == 0);
}

// ============================================================================
// GraphRunner Multiple Executions Tests
// ============================================================================

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner multiple executions", "[runner][multi]") {
    ComputeGraph graph;
    graph.add_input("input", {32, 64});
    auto kernel = Kernel::create_mlp(32, 32, 64, ActivationType::RELU, true);
    graph.add_node("fc1", kernel, {"input"}, {"output"});
    graph.add_output("output");
    graph.finalize();

    GraphRunner runner(runtime.get());
    runner.set_graph(graph);
    runner.allocate();

    std::vector<float> input(32 * 64, 0.5f);
    runner.set_input("input", input.data(), input.size() * sizeof(float));

    // Execute multiple times
    for (int i = 0; i < 5; ++i) {
        auto result = runner.run();
        REQUIRE(result.success);
    }

    // Update input and execute again
    std::fill(input.begin(), input.end(), 0.8f);
    runner.set_input("input", input.data(), input.size() * sizeof(float));

    auto result = runner.run();
    REQUIRE(result.success);
}

// ============================================================================
// Deep Network Tests
// ============================================================================

TEST_CASE_METHOD(GraphTestFixture, "GraphRunner deep network", "[runner][deep]") {
    // 5-layer MLP
    ComputeGraph graph("mlp_5layer");
    graph.add_input("input", {32, 256});

    auto layer1 = Kernel::create_mlp(32, 128, 256, ActivationType::RELU, true);
    auto layer2 = Kernel::create_mlp(32, 64, 128, ActivationType::RELU, true);
    auto layer3 = Kernel::create_mlp(32, 64, 64, ActivationType::RELU, true);
    auto layer4 = Kernel::create_mlp(32, 32, 64, ActivationType::RELU, true);
    auto layer5 = Kernel::create_mlp(32, 10, 32, ActivationType::NONE, true);

    graph.add_node("fc1", layer1, {"input"}, {"h1"});
    graph.add_node("fc2", layer2, {"h1"}, {"h2"});
    graph.add_node("fc3", layer3, {"h2"}, {"h3"});
    graph.add_node("fc4", layer4, {"h3"}, {"h4"});
    graph.add_node("fc5", layer5, {"h4"}, {"output"});
    graph.add_output("output");
    graph.finalize();

    GraphRunner runner(runtime.get());
    runner.set_graph(graph);
    runner.allocate();

    std::vector<float> input(32 * 256, 0.5f);
    runner.set_input("input", input.data(), input.size() * sizeof(float));

    auto result = runner.run();

    REQUIRE(result.success);
    REQUIRE(result.layer_profiles.size() == 5);

    // Print timing breakdown
    INFO("5-layer MLP execution time: " << result.total_time_ms << " ms");
    for (const auto& profile : result.layer_profiles) {
        INFO("  " << profile.name << ": " << profile.cycles << " cycles");
    }
}
