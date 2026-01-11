/**
 * @file xor_behavioral.cpp
 * @brief Behavioral XOR classifier test - validates MLP execution
 *
 * This example tests the behavioral MLP execution path with a simple
 * XOR classifier network:
 *   - Input: 2 neurons (x1, x2)
 *   - Hidden: 4 neurons with ReLU activation
 *   - Output: 1 neuron (XOR result)
 *
 * The network uses pre-computed weights that solve the XOR problem.
 * Expected output: XOR(0,0)=0, XOR(0,1)=1, XOR(1,0)=1, XOR(1,1)=0
 */

#include <sw/kpu/models/behavioral/mlp_executor.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace sw::kpu::behavioral;
using sw::kpu::ActivationType;

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << " Behavioral XOR Classifier Test" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::endl;

    // Create MLP executor
    BehavioralMLPExecutor executor;

    // Define network architecture: 2 -> 4 (ReLU) -> 1 (linear)
    executor.add_layer(2, 4, ActivationType::RELU, true, "hidden");
    executor.add_layer(4, 1, ActivationType::NONE, true, "output");

    // Set batch size to 4 (all XOR cases at once)
    executor.set_batch_size(4);

    // Pre-trained weights for XOR
    // Hidden layer: input_dim=2, output_dim=4
    // These weights implement the classic XOR solution:
    //   h1 = ReLU(x1 + x2 - 0.5)  -> detects x1 OR x2
    //   h2 = ReLU(x1 + x2 - 1.5)  -> detects x1 AND x2
    //   h3 = ReLU(-x1 - x2 + 0.5) -> detects NOT(x1 OR x2) - not needed but adds capacity
    //   h4 = ReLU(-x1 - x2 + 1.5) -> detects NOT(x1 AND x2) - not needed but adds capacity
    std::vector<float> w1 = {
        // Columns are outputs h1, h2, h3, h4 for each input row
        1.0f,  1.0f, -1.0f, -1.0f,  // from x1
        1.0f,  1.0f, -1.0f, -1.0f   // from x2
    };
    std::vector<float> b1 = {-0.5f, -1.5f, 0.5f, 1.5f};

    // Output layer: input_dim=4, output_dim=1
    // XOR = (x1 OR x2) AND NOT(x1 AND x2)
    //     = 2*h1 - 6*h2 (scaled because ReLU doesn't saturate at 1)
    // For (0,1)/(1,0): h1=0.5, h2=0 -> output = 1.0
    // For (1,1): h1=1.5, h2=0.5 -> output = 3.0 - 3.0 = 0
    // For (0,0): h1=0, h2=0 -> output = 0
    std::vector<float> w2 = {
        2.0f,   // from h1 (OR) - scaled up
        -6.0f,  // from h2 (AND) - scaled to cancel h1 when both active
        0.0f,   // from h3 (unused)
        0.0f    // from h4 (unused)
    };
    std::vector<float> b2 = {0.0f};

    // Load weights
    std::cout << "Loading network weights..." << std::endl;
    executor.load_weights({w1, w2}, {b1, b2});

    // Prepare input: all 4 XOR cases [batch=4, features=2]
    std::vector<float> input = {
        0.0f, 0.0f,  // Case 1: XOR(0,0) = 0
        0.0f, 1.0f,  // Case 2: XOR(0,1) = 1
        1.0f, 0.0f,  // Case 3: XOR(1,0) = 1
        1.0f, 1.0f   // Case 4: XOR(1,1) = 0
    };

    std::vector<float> expected = {0.0f, 1.0f, 1.0f, 0.0f};

    // Execute forward pass
    std::cout << "Executing forward pass..." << std::endl;
    std::vector<float> output(4);
    executor.forward(input, output);

    // Display results
    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "---------" << std::endl;

    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        float x1 = input[i * 2];
        float x2 = input[i * 2 + 1];
        float pred = output[i];
        float exp = expected[i];

        // Binary prediction using 0.5 threshold
        bool pred_binary = pred > 0.5f;
        bool exp_binary = exp > 0.5f;
        bool match = (pred_binary == exp_binary);

        std::cout << "  XOR(" << x1 << ", " << x2 << ") = " << pred
                  << " (expected " << exp << ") "
                  << (match ? "OK" : "FAIL") << std::endl;

        if (match) correct++;
    }

    // Summary
    std::cout << std::endl;
    std::cout << "Accuracy: " << correct << "/4 = " << (correct * 100 / 4) << "%" << std::endl;

    // Statistics
    const auto& stats = executor.stats();
    std::cout << std::endl;
    std::cout << "Statistics:" << std::endl;
    std::cout << "  Forward passes: " << stats.forward_passes << std::endl;
    std::cout << "  Total FLOPs: " << stats.total_flops << std::endl;
    std::cout << "  Network FLOPs/pass: " << executor.compute_network_flops() << std::endl;

    // Orchestrator stats
    const auto& orch_stats = executor.orchestrator().stats();
    std::cout << "  Matmul operations: " << orch_stats.matmul_count << std::endl;
    std::cout << "  Host->L3 bytes: " << orch_stats.host_to_l3_bytes << std::endl;
    std::cout << "  L3->Host bytes: " << orch_stats.l3_to_host_bytes << std::endl;

    std::cout << std::endl;
    std::cout << "==================================================" << std::endl;
    if (correct == 4) {
        std::cout << " XOR TEST PASSED" << std::endl;
    } else {
        std::cout << " XOR TEST FAILED" << std::endl;
    }
    std::cout << "==================================================" << std::endl;

    return (correct == 4) ? 0 : 1;
}
