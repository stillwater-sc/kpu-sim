/**
 * @file unified_xor_mlp.cpp
 * @brief Value-producing XOR MLP on the CSP concurrent timing executor.
 */

#include <sw/kpu/timing/functional_mlp_executor.hpp>

#include <cmath>
#include <iostream>
#include <vector>

using sw::kpu::timing::ConcurrentTimingExecutor;
using sw::kpu::timing::FunctionalMLPExecutor;

int main() {
    ConcurrentTimingExecutor::Config config;
    config.num_memory_controllers = 1;
    config.num_dma_engines = 1;
    config.num_block_movers = 1;
    config.num_row_streamers = 1;
    config.num_col_streamers = 1;
    config.l3_buffer_count = 1;
    config.l2_bank_count = 1;
    config.compute_latency = 8;
    config.max_cycles = 100000;
    config.enable_livelock_detection = false;

    FunctionalMLPExecutor mlp(config);
    mlp.add_layer(2, 4,
        {1.0f, 1.0f, -1.0f, -1.0f,
         1.0f, 1.0f, -1.0f, -1.0f},
        {-0.5f, -1.5f, 0.5f, 1.5f},
        ConcurrentTimingExecutor::FunctionalActivation::RELU,
        "hidden");
    mlp.add_layer(4, 1,
        {2.0f, -6.0f, 0.0f, 0.0f}, {0.0f},
        ConcurrentTimingExecutor::FunctionalActivation::NONE,
        "output");

    const std::vector<float> input = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    const std::vector<float> expected = {0.0f, 1.0f, 1.0f, 0.0f};
    const auto output = mlp.forward(input, 4);

    bool values_pass = output.size() == expected.size();
    for (size_t i = 0; i < output.size() && i < expected.size(); ++i) {
        values_pass = values_pass && std::fabs(output[i] - expected[i]) < 1e-6f;
        std::cout << "XOR case " << i << ": " << output[i]
                  << " (expected " << expected[i] << ")\n";
    }

    const auto& stats = mlp.statistics();
    const bool ordered_pass = stats.layers_completed == 2 &&
                              stats.total_cycles > 0 &&
                              stats.total_stall_cycles > 0;
    std::cout << "cycles: " << stats.total_cycles << '\n';
    std::cout << "credit/tag stall cycles: " << stats.total_stall_cycles << '\n';
    std::cout << "Numerical result: " << (values_pass ? "PASS" : "FAIL") << '\n';
    std::cout << "Transaction-ordered execution: "
              << (ordered_pass ? "PASS" : "FAIL") << '\n';
    return values_pass && ordered_pass ? 0 : 1;
}
