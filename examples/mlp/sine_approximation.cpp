// Sine Wave Approximation Example
// Demonstrates a 3-layer MLP learning to approximate sin(x)
//
// Network architecture: 1 -> 32 -> 32 -> 1
// - Input: x value in range [-pi, pi]
// - Output: approximation of sin(x)
//
// The weights below are illustrative, not trained: run on the host, they do not
// approximate sin(x).
//
// TIMING MODEL ONLY: the runtime reports the network's cycles per layer but computes
// no values, so this example checks no outputs (#257).

#include <sw/runtime/graph.hpp>
#include <sw/kpu/kpu_simulator.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

using namespace sw::runtime;
using namespace sw::kpu;

// Illustrative weights for the network's shape (they are not trained)
// Network: 1 -> 32 (ReLU) -> 32 (ReLU) -> 1

// First hidden layer: [1, 32]
const float LAYER1_WEIGHTS[1 * 32] = {
    // Fourier-like basis functions at different frequencies
     0.5f,  0.7f,  0.9f,  1.1f,  1.3f,  1.5f,  1.7f,  1.9f,
    -0.5f, -0.7f, -0.9f, -1.1f, -1.3f, -1.5f, -1.7f, -1.9f,
     0.3f,  0.4f,  0.5f,  0.6f,  0.8f,  1.0f,  1.2f,  1.4f,
    -0.3f, -0.4f, -0.5f, -0.6f, -0.8f, -1.0f, -1.2f, -1.4f,
};

const float LAYER1_BIAS[32] = {
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, 0.5f, 1.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, 0.5f, 1.0f,
    1.5f, 2.0f, 2.5f, -0.2f, -0.8f, -1.2f, 0.2f, 0.8f,
    1.5f, 2.0f, 2.5f, -0.2f, -0.8f, -1.2f, 0.2f, 0.8f,
};

// Second hidden layer: [32, 32]
// Initialize with small random-like values for feature combination
const float LAYER2_WEIGHTS[32 * 32] = {
    // Row 0
    0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f,
    0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f,
    0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f,
    0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f,
    // Rows 1-31 (simplified pattern for demo)
    0.15f, -0.15f, 0.25f, -0.25f, 0.15f, -0.15f, 0.25f, -0.25f,
    0.15f, -0.15f, 0.25f, -0.25f, 0.15f, -0.15f, 0.25f, -0.25f,
    0.15f, -0.15f, 0.25f, -0.25f, 0.15f, -0.15f, 0.25f, -0.25f,
    0.15f, -0.15f, 0.25f, -0.25f, 0.15f, -0.15f, 0.25f, -0.25f,
    // Continue pattern...
    0.12f, -0.12f, 0.22f, -0.22f, 0.12f, -0.12f, 0.22f, -0.22f,
    0.12f, -0.12f, 0.22f, -0.22f, 0.12f, -0.12f, 0.22f, -0.22f,
    0.12f, -0.12f, 0.22f, -0.22f, 0.12f, -0.12f, 0.22f, -0.22f,
    0.12f, -0.12f, 0.22f, -0.22f, 0.12f, -0.12f, 0.22f, -0.22f,
    0.08f, -0.08f, 0.18f, -0.18f, 0.08f, -0.08f, 0.18f, -0.18f,
    0.08f, -0.08f, 0.18f, -0.18f, 0.08f, -0.08f, 0.18f, -0.18f,
    0.08f, -0.08f, 0.18f, -0.18f, 0.08f, -0.08f, 0.18f, -0.18f,
    0.08f, -0.08f, 0.18f, -0.18f, 0.08f, -0.08f, 0.18f, -0.18f,
    // More rows
    0.05f, -0.05f, 0.15f, -0.15f, 0.05f, -0.05f, 0.15f, -0.15f,
    0.05f, -0.05f, 0.15f, -0.15f, 0.05f, -0.05f, 0.15f, -0.15f,
    0.05f, -0.05f, 0.15f, -0.15f, 0.05f, -0.05f, 0.15f, -0.15f,
    0.05f, -0.05f, 0.15f, -0.15f, 0.05f, -0.05f, 0.15f, -0.15f,
    0.18f, -0.18f, 0.28f, -0.28f, 0.18f, -0.18f, 0.28f, -0.28f,
    0.18f, -0.18f, 0.28f, -0.28f, 0.18f, -0.18f, 0.28f, -0.28f,
    0.18f, -0.18f, 0.28f, -0.28f, 0.18f, -0.18f, 0.28f, -0.28f,
    0.18f, -0.18f, 0.28f, -0.28f, 0.18f, -0.18f, 0.28f, -0.28f,
    // Final rows
    0.2f, -0.2f, 0.3f, -0.3f, 0.2f, -0.2f, 0.3f, -0.3f,
    0.2f, -0.2f, 0.3f, -0.3f, 0.2f, -0.2f, 0.3f, -0.3f,
    0.2f, -0.2f, 0.3f, -0.3f, 0.2f, -0.2f, 0.3f, -0.3f,
    0.2f, -0.2f, 0.3f, -0.3f, 0.2f, -0.2f, 0.3f, -0.3f,
    0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f,
    0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f,
    0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f,
    0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.2f, -0.2f,
    // Fill remaining
    0.11f, -0.11f, 0.21f, -0.21f, 0.11f, -0.11f, 0.21f, -0.21f,
    0.11f, -0.11f, 0.21f, -0.21f, 0.11f, -0.11f, 0.21f, -0.21f,
    0.11f, -0.11f, 0.21f, -0.21f, 0.11f, -0.11f, 0.21f, -0.21f,
    0.11f, -0.11f, 0.21f, -0.21f, 0.11f, -0.11f, 0.21f, -0.21f,
    0.14f, -0.14f, 0.24f, -0.24f, 0.14f, -0.14f, 0.24f, -0.24f,
    0.14f, -0.14f, 0.24f, -0.24f, 0.14f, -0.14f, 0.24f, -0.24f,
    0.14f, -0.14f, 0.24f, -0.24f, 0.14f, -0.14f, 0.24f, -0.24f,
    0.14f, -0.14f, 0.24f, -0.24f, 0.14f, -0.14f, 0.24f, -0.24f,
    // More filler
    0.09f, -0.09f, 0.19f, -0.19f, 0.09f, -0.09f, 0.19f, -0.19f,
    0.09f, -0.09f, 0.19f, -0.19f, 0.09f, -0.09f, 0.19f, -0.19f,
    0.09f, -0.09f, 0.19f, -0.19f, 0.09f, -0.09f, 0.19f, -0.19f,
    0.09f, -0.09f, 0.19f, -0.19f, 0.09f, -0.09f, 0.19f, -0.19f,
    0.16f, -0.16f, 0.26f, -0.26f, 0.16f, -0.16f, 0.26f, -0.26f,
    0.16f, -0.16f, 0.26f, -0.26f, 0.16f, -0.16f, 0.26f, -0.26f,
    0.16f, -0.16f, 0.26f, -0.26f, 0.16f, -0.16f, 0.26f, -0.26f,
    0.16f, -0.16f, 0.26f, -0.26f, 0.16f, -0.16f, 0.26f, -0.26f,
    // Final block
    0.13f, -0.13f, 0.23f, -0.23f, 0.13f, -0.13f, 0.23f, -0.23f,
    0.13f, -0.13f, 0.23f, -0.23f, 0.13f, -0.13f, 0.23f, -0.23f,
    0.13f, -0.13f, 0.23f, -0.23f, 0.13f, -0.13f, 0.23f, -0.23f,
    0.13f, -0.13f, 0.23f, -0.23f, 0.13f, -0.13f, 0.23f, -0.23f,
    0.07f, -0.07f, 0.17f, -0.17f, 0.07f, -0.07f, 0.17f, -0.17f,
    0.07f, -0.07f, 0.17f, -0.17f, 0.07f, -0.07f, 0.17f, -0.17f,
    0.07f, -0.07f, 0.17f, -0.17f, 0.07f, -0.07f, 0.17f, -0.17f,
    0.07f, -0.07f, 0.17f, -0.17f, 0.07f, -0.07f, 0.17f, -0.17f,
    // Remainder
    0.19f, -0.19f, 0.29f, -0.29f, 0.19f, -0.19f, 0.29f, -0.29f,
    0.19f, -0.19f, 0.29f, -0.29f, 0.19f, -0.19f, 0.29f, -0.29f,
    0.19f, -0.19f, 0.29f, -0.29f, 0.19f, -0.19f, 0.29f, -0.29f,
    0.19f, -0.19f, 0.29f, -0.29f, 0.19f, -0.19f, 0.29f, -0.29f,
    0.06f, -0.06f, 0.16f, -0.16f, 0.06f, -0.06f, 0.16f, -0.16f,
    0.06f, -0.06f, 0.16f, -0.16f, 0.06f, -0.06f, 0.16f, -0.16f,
    0.06f, -0.06f, 0.16f, -0.16f, 0.06f, -0.06f, 0.16f, -0.16f,
    0.06f, -0.06f, 0.16f, -0.16f, 0.06f, -0.06f, 0.16f, -0.16f,
};

const float LAYER2_BIAS[32] = {
    0.0f, 0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.0f,
    0.05f, -0.05f, 0.15f, -0.15f, 0.05f, -0.05f, 0.15f, -0.15f,
    0.0f, 0.1f, -0.1f, 0.2f, -0.2f, 0.1f, -0.1f, 0.0f,
    0.05f, -0.05f, 0.15f, -0.15f, 0.05f, -0.05f, 0.15f, -0.15f,
};

// Output layer: [32, 1]
const float LAYER3_WEIGHTS[32 * 1] = {
    0.15f, -0.15f, 0.12f, -0.12f, 0.18f, -0.18f, 0.1f, -0.1f,
    0.14f, -0.14f, 0.11f, -0.11f, 0.17f, -0.17f, 0.09f, -0.09f,
    0.16f, -0.16f, 0.13f, -0.13f, 0.19f, -0.19f, 0.08f, -0.08f,
    0.2f, -0.2f, 0.1f, -0.1f, 0.15f, -0.15f, 0.12f, -0.12f,
};

const float LAYER3_BIAS[1] = {0.0f};

int main() {
    std::cout << "============================================\n";
    std::cout << "Sine Wave Approximation on KPU Simulator\n";
    std::cout << "============================================\n\n";

    std::cout << "Network architecture: 1 -> 32 (ReLU) -> 32 (ReLU) -> 1\n";
    std::cout << "Task: Approximate sin(x) for x in [-pi, pi]\n\n";

    // Configure KPU simulator
    KPUSimulator::Config config;
    config.memory_bank_count = 2;
    config.memory_bank_capacity_mb = 64;
    config.l3_layer.tile_groups = { {"l3", {128}, 4} };
    config.l2_layer.bank_groups = { {"l2", {64}, 8} };
    config.l1_layer.buffer_groups = { {"l1", {64 * 1024}, 4} };
    config.processor_array_rows = 16;
    config.processor_array_cols = 16;
    config.use_systolic_array_mode = true;

    KPUSimulator simulator(config);
    KPURuntime runtime(&simulator);

    // Generate test points
    const Size NUM_SAMPLES = 32;
    const float PI = 3.14159265358979f;
    std::vector<float> x_values(NUM_SAMPLES);

    for (Size i = 0; i < NUM_SAMPLES; ++i) {
        float t = static_cast<float>(i) / (NUM_SAMPLES - 1);
        x_values[i] = -PI + 2.0f * PI * t;
    }

    // Build computational graph
    const Size BATCH_SIZE = NUM_SAMPLES;
    const Size INPUT_DIM = 1;
    const Size HIDDEN1_DIM = 32;
    const Size HIDDEN2_DIM = 32;
    const Size OUTPUT_DIM = 1;

    ComputeGraph graph("sine_approximation");

    // Input: [32, 1]
    graph.add_input("input", {BATCH_SIZE, INPUT_DIM});

    // Layer 1: [32, 1] @ [1, 32] + [32] -> [32, 32]
    auto layer1 = Kernel::create_mlp(
        BATCH_SIZE, HIDDEN1_DIM, INPUT_DIM,
        ActivationType::RELU, true);
    graph.add_node("layer1", layer1, {"input"}, {"h1"});

    // Layer 2: [32, 32] @ [32, 32] + [32] -> [32, 32]
    auto layer2 = Kernel::create_mlp(
        BATCH_SIZE, HIDDEN2_DIM, HIDDEN1_DIM,
        ActivationType::RELU, true);
    graph.add_node("layer2", layer2, {"h1"}, {"h2"});

    // Layer 3: [32, 32] @ [32, 1] + [1] -> [32, 1]
    auto layer3 = Kernel::create_mlp(
        BATCH_SIZE, OUTPUT_DIM, HIDDEN2_DIM,
        ActivationType::NONE, true);
    graph.add_node("layer3", layer3, {"h2"}, {"output"});

    graph.add_output("output");
    graph.finalize();

    std::cout << graph.summary() << "\n";

    // Create runner and allocate
    GraphRunner runner(&runtime);
    runner.set_graph(graph);
    runner.allocate();

    std::cout << "Memory allocated: " << runner.total_allocated_bytes() << " bytes\n\n";

    // Prepare input (reshape x_values to [32, 1])
    runner.set_input("input", x_values.data(), x_values.size() * sizeof(float));

    // Set layer 1 weights
    Address l1_B = runner.get_tensor_address("layer1_B");
    Address l1_bias = runner.get_tensor_address("layer1_bias");
    if (l1_B) runtime.memcpy_h2d(l1_B, LAYER1_WEIGHTS, sizeof(LAYER1_WEIGHTS));
    if (l1_bias) runtime.memcpy_h2d(l1_bias, LAYER1_BIAS, sizeof(LAYER1_BIAS));

    // Set layer 2 weights
    Address l2_B = runner.get_tensor_address("layer2_B");
    Address l2_bias = runner.get_tensor_address("layer2_bias");
    if (l2_B) runtime.memcpy_h2d(l2_B, LAYER2_WEIGHTS, sizeof(LAYER2_WEIGHTS));
    if (l2_bias) runtime.memcpy_h2d(l2_bias, LAYER2_BIAS, sizeof(LAYER2_BIAS));

    // Set layer 3 weights
    Address l3_B = runner.get_tensor_address("layer3_B");
    Address l3_bias = runner.get_tensor_address("layer3_bias");
    if (l3_B) runtime.memcpy_h2d(l3_B, LAYER3_WEIGHTS, sizeof(LAYER3_WEIGHTS));
    if (l3_bias) runtime.memcpy_h2d(l3_bias, LAYER3_BIAS, sizeof(LAYER3_BIAS));

    std::cout << "Running sine approximation on KPU...\n";

    // Execute
    auto result = runner.run();

    if (!result.success) {
        std::cerr << "Execution failed: " << result.error << "\n";
        return 1;
    }

    std::cout << "\nExecution complete!\n";
    std::cout << "  Total cycles: " << result.total_cycles << "\n";
    std::cout << "  Total time: " << result.total_time_ms << " ms\n\n";

    // Per-layer timing
    std::cout << "Per-layer timing:\n";
    for (const auto& profile : result.layer_profiles) {
        std::cout << "  " << profile.name << ": "
                  << profile.cycles << " cycles\n";
    }
    std::cout << "\n";

    // No error table: the runtime is timing only. runner.run() reports each layer's
    // cycles, but computes no values, so the output is never written (#257 makes the
    // host API return simulator-computed values).
    std::cout << "============================================\n";
    std::cout << "Note: the runtime is a timing model. It computes\n";
    std::cout << "no outputs, so sin(x) is not compared here (#257).\n";
    std::cout << "============================================\n";

    return 0;
}
