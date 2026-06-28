// XOR Classifier Example
// Demonstrates a 2-layer MLP solving the classic XOR problem
//
// XOR truth table:
//   (0,0) -> 0
//   (0,1) -> 1
//   (1,0) -> 1
//   (1,1) -> 0
//
// Network architecture: 2 -> 4 -> 1
// - Input layer: 2 neurons (x1, x2)
// - Hidden layer: 4 neurons with ReLU activation
// - Output layer: 1 neuron with linear output
//
// IMPORTANT: TIMING MODEL ONLY
// ============================
// The KPU simulator is a PERFORMANCE/TIMING model, not a functional simulator.
// - It accurately models cycle counts, memory latencies, and resource contention
// - It does NOT compute actual matrix multiplication results
// - The weights are loaded but never used in computation
// - Output tensors are NOT populated with computed values
//
// This example demonstrates:
// 1. Graph construction for multi-layer MLP
// 2. Weight tensor allocation and loading
// 3. Performance estimation for small operators on large arrays
// 4. Reference implementation for numerical verification
//
// For actual XOR computation, see reference_xor_forward() below.
//
// Small Operator on Large Array:
// - XOR MLP: 2x4 and 4x1 matmuls (8 and 4 MACs)
// - KPU config: 16x16 systolic array (256 PEs)
// - Array utilization: ~3% for largest operation
// - The simulator models timing for full array execution

#include <sw/runtime/graph.hpp>
#include <sw/kpu/kpu_simulator.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace sw::runtime;
using namespace sw::kpu;

// Pre-trained weights for XOR classifier
// Network: 2 -> 4 (ReLU) -> 1
// These weights were trained to solve the XOR problem

// Hidden layer weights: [2, 4] (input_dim, hidden_dim)
// Transposed for matmul: A[batch, 2] @ B[2, 4] = H[batch, 4]
const float HIDDEN_WEIGHTS[2 * 4] = {
    // Column 0: neuron that activates for (1,0) and (1,1)
     2.5f, -2.5f,
    // Column 1: neuron that activates for (0,1) and (1,1)
    -2.5f,  2.5f,
    // Column 2: neuron that activates for neither input high
     2.0f,  2.0f,
    // Column 3: neuron that activates for both inputs high
    -2.0f, -2.0f,
};

// Hidden layer bias: [4]
const float HIDDEN_BIAS[4] = {
    -0.5f,  // Bias for neuron 0
    -0.5f,  // Bias for neuron 1
    -3.0f,  // Bias for neuron 2 (needs both low)
     3.0f,  // Bias for neuron 3 (needs both high)
};

// Output layer weights: [4, 1] (hidden_dim, output_dim)
const float OUTPUT_WEIGHTS[4 * 1] = {
     2.0f,   // Weight from hidden 0 (x1 and not x2)
     2.0f,   // Weight from hidden 1 (x2 and not x1)
    -1.0f,   // Weight from hidden 2 (neither)
    -1.0f,   // Weight from hidden 3 (both)
};

// Output layer bias: [1]
const float OUTPUT_BIAS[1] = {
    -0.5f,
};

// XOR test cases
struct XORTestCase {
    float x1, x2;
    float expected;
};

const XORTestCase XOR_TESTS[] = {
    {0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 1.0f},
    {1.0f, 0.0f, 1.0f},
    {1.0f, 1.0f, 0.0f},
};

// Reference implementation for verification
float reference_xor_forward(float x1, float x2) {
    // Hidden layer with ReLU
    float h[4];
    for (int i = 0; i < 4; ++i) {
        float sum = HIDDEN_BIAS[i];
        sum += x1 * HIDDEN_WEIGHTS[i * 2 + 0];
        sum += x2 * HIDDEN_WEIGHTS[i * 2 + 1];
        h[i] = std::max(0.0f, sum);  // ReLU
    }

    // Output layer (linear)
    float out = OUTPUT_BIAS[0];
    for (int i = 0; i < 4; ++i) {
        out += h[i] * OUTPUT_WEIGHTS[i];
    }

    return out;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "XOR Classifier on KPU Simulator\n";
    std::cout << "(TIMING MODEL - See header for details)\n";
    std::cout << "========================================\n\n";

    // First, verify the reference implementation
    std::cout << "Reference implementation verification:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& test : XOR_TESTS) {
        float ref = reference_xor_forward(test.x1, test.x2);
        std::cout << "  XOR(" << test.x1 << ", " << test.x2 << ") = "
                  << ref << " (expected: " << test.expected << ")\n";
    }
    std::cout << "\n";

    // Configure KPU simulator
    KPUSimulator::Config config;
    config.memory_bank_count = 2;
    config.memory_bank_capacity_mb = 64;
    config.l3_layer.num_tiles = 4;
    config.l3_layer.capacity_kb = 128;
    config.l2_layer.num_banks = 8;
    config.l2_layer.capacity_kb = 64;
    config.l1_layer.num_buffers = 4;
    config.l1_layer.capacity_kb = 64;
    config.processor_array_rows = 16;
    config.processor_array_cols = 16;
    config.use_systolic_array_mode = true;

    KPUSimulator simulator(config);
    KPURuntime runtime(&simulator);

    std::cout << "Building XOR classifier graph...\n";

    // Build the computational graph
    // We'll process all 4 test cases as a batch
    const Size BATCH_SIZE = 4;
    const Size INPUT_DIM = 2;
    const Size HIDDEN_DIM = 4;
    const Size OUTPUT_DIM = 1;

    ComputeGraph graph("xor_classifier");

    // Input: [4, 2] - 4 test cases, 2 features each
    graph.add_input("input", {BATCH_SIZE, INPUT_DIM});

    // Hidden layer: [4, 2] @ [2, 4] + [4] -> [4, 4] with ReLU
    auto hidden_layer = Kernel::create_mlp(
        BATCH_SIZE, HIDDEN_DIM, INPUT_DIM,
        ActivationType::RELU, true);
    graph.add_node("hidden", hidden_layer, {"input"}, {"h1"});

    // Output layer: [4, 4] @ [4, 1] + [1] -> [4, 1] (no activation for regression)
    auto output_layer = Kernel::create_mlp(
        BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM,
        ActivationType::NONE, true);
    graph.add_node("output", output_layer, {"h1"}, {"output"});

    graph.add_output("output");
    graph.finalize();

    std::cout << "\n" << graph.summary() << "\n";

    // Create runner and allocate memory
    GraphRunner runner(&runtime);
    runner.set_graph(graph);
    runner.allocate();

    std::cout << "Memory allocated: " << runner.total_allocated_bytes() << " bytes\n\n";

    // Prepare input data (all 4 XOR test cases)
    std::vector<float> input_data(BATCH_SIZE * INPUT_DIM);
    for (size_t i = 0; i < 4; ++i) {
        input_data[i * 2 + 0] = XOR_TESTS[i].x1;
        input_data[i * 2 + 1] = XOR_TESTS[i].x2;
    }

    // Set input
    runner.set_input("input", input_data.data(), input_data.size() * sizeof(float));

    // Set hidden layer weights
    // Note: Weight tensors are named "nodename_B" and "nodename_bias"
    Address hidden_B_addr = runner.get_tensor_address("hidden_B");
    Address hidden_bias_addr = runner.get_tensor_address("hidden_bias");
    if (hidden_B_addr) {
        runtime.memcpy_h2d(hidden_B_addr, HIDDEN_WEIGHTS, sizeof(HIDDEN_WEIGHTS));
    }
    if (hidden_bias_addr) {
        runtime.memcpy_h2d(hidden_bias_addr, HIDDEN_BIAS, sizeof(HIDDEN_BIAS));
    }

    // Set output layer weights
    Address output_B_addr = runner.get_tensor_address("output_B");
    Address output_bias_addr = runner.get_tensor_address("output_bias");
    if (output_B_addr) {
        runtime.memcpy_h2d(output_B_addr, OUTPUT_WEIGHTS, sizeof(OUTPUT_WEIGHTS));
    }
    if (output_bias_addr) {
        runtime.memcpy_h2d(output_bias_addr, OUTPUT_BIAS, sizeof(OUTPUT_BIAS));
    }

    std::cout << "Running XOR classifier on KPU...\n";

    // Execute the graph
    auto result = runner.run();

    if (!result.success) {
        std::cerr << "Execution failed: " << result.error << "\n";
        return 1;
    }

    std::cout << "\nExecution complete!\n";
    std::cout << "  Total cycles: " << result.total_cycles << "\n";
    std::cout << "  Total time: " << result.total_time_ms << " ms\n\n";

    // Print per-layer timing
    std::cout << "Per-layer timing:\n";
    for (const auto& profile : result.layer_profiles) {
        std::cout << "  " << profile.name << ": "
                  << profile.cycles << " cycles ("
                  << profile.time_ms << " ms)\n";
    }
    std::cout << "\n";

    // Get output
    std::vector<float> output_data(BATCH_SIZE * OUTPUT_DIM);
    runner.get_output("output", output_data.data());

    // KPU simulator is a performance model - it simulates timing, not computation.
    // For numerical verification, we use the reference implementation.
    std::cout << "Verification using reference implementation:\n";
    std::cout << "(KPU simulator models timing; reference computes actual values)\n\n";
    std::cout << "  x1    x2    Reference   Expected   Status\n";
    std::cout << "  ----  ----  ----------  --------   ------\n";

    int correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        float expected = XOR_TESTS[i].expected;
        float reference = reference_xor_forward(XOR_TESTS[i].x1, XOR_TESTS[i].x2);

        // Threshold at 0.5 for classification
        bool ref_class = reference > 0.0f;  // Output is centered around 0
        bool expected_class = expected > 0.5f;
        bool match = (ref_class == expected_class);
        if (match) correct++;

        std::cout << "  " << XOR_TESTS[i].x1 << "   "
                  << XOR_TESTS[i].x2 << "   "
                  << std::setw(10) << reference << "  "
                  << std::setw(8) << expected << "   "
                  << (match ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\nReference accuracy: " << correct << "/4 ("
              << (correct * 100 / 4) << "%)\n";

    std::cout << "\n========================================\n";
    std::cout << "Summary:\n";
    std::cout << "- Graph execution infrastructure: WORKING\n";
    std::cout << "- Memory allocation: " << runner.total_allocated_bytes() << " bytes\n";
    std::cout << "- Layer scheduling: " << graph.num_nodes() << " layers executed\n";
    std::cout << "- Total simulated cycles: " << result.total_cycles << "\n";
    std::cout << "\n";
    std::cout << "Array Utilization Analysis:\n";
    std::cout << "- Hidden layer: 2x4 matmul = 8 MACs\n";
    std::cout << "- Output layer: 4x1 matmul = 4 MACs\n";
    std::cout << "- KPU array: 16x16 = 256 PEs\n";
    std::cout << "- Peak utilization: 8/256 = 3.1%\n";
    std::cout << "(Small operators are padded to full array)\n";
    std::cout << "\n";
    std::cout << "IMPORTANT: This is a TIMING MODEL only.\n";
    std::cout << "- Cycle counts reflect real hardware behavior\n";
    std::cout << "- Actual values computed by reference implementation\n";
    std::cout << "- See docs/FUNCTIONAL_SIMULATION_GAP_ANALYSIS.md\n";
    std::cout << "========================================\n";

    return (correct == 4) ? 0 : 1;
}
