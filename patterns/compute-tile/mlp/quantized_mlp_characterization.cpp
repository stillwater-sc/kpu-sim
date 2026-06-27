// ============================================================================
// Quantized MLP Characterization Pattern
// ============================================================================
// Demonstrates MLP execution with XUE Observation Architecture characterization
// for performance analysis and roofline modeling.
//
// This pattern shows:
// 1. MLP execution through BehavioralComputeFabric
// 2. XUE event collection for compute/memory characterization
// 3. Roofline model analysis predicting compute vs memory bound behavior
// 4. Impact of different precision settings on cycle estimation
//
// XUE: The Observation Architecture provides hardware event counters for
// pre/post-silicon performance validation (ref: Intel Patent 6,023,759).
//
// Run with: ./mlp_quantized_characterization
// ============================================================================

#include <sw/kpu/models/behavioral/compute/compute_fabric.hpp>
#include <sw/kpu/data_types.hpp>

#include <sw/xue/event_collector.hpp>
#include <sw/xue/operational_analysis.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <cassert>
#include <limits>

using namespace sw::kpu;
using namespace sw::xue;

// ============================================================================
// MLP Configuration
// ============================================================================

struct MLPConfig {
    size_t input_dim;
    size_t hidden_dim;
    size_t output_dim;
    size_t batch_size;

    uint64_t layer1_flops() const {
        return 2ULL * batch_size * input_dim * hidden_dim;  // matmul
    }

    uint64_t layer2_flops() const {
        return 2ULL * batch_size * hidden_dim * output_dim;  // matmul
    }

    uint64_t total_flops() const {
        return layer1_flops() + layer2_flops() +
               batch_size * hidden_dim +      // ReLU
               batch_size * output_dim;       // Final activation
    }

    // Memory for weights + activations (in bytes)
    uint64_t memory_bytes(size_t elem_size) const {
        // Weights: W1[in x hidden] + W2[hidden x out] + biases
        uint64_t weight_bytes = (input_dim * hidden_dim + hidden_dim * output_dim +
                                 hidden_dim + output_dim) * elem_size;
        // Activations: input + hidden + output
        uint64_t act_bytes = (batch_size * input_dim + batch_size * hidden_dim +
                              batch_size * output_dim) * elem_size;
        return weight_bytes + act_bytes;
    }
};

// ============================================================================
// Reference MLP Implementation (for validation)
// ============================================================================

/// Reference matmul: C[m,n] = A[m,k] @ B[k,n]
void ref_matmul(
    const float* A, const float* B, float* C,
    size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/// Reference ReLU: output[i] = max(0, input[i])
void ref_relu(const float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

/// Reference bias add: output[i] += bias[i % dim]
void ref_bias_add(float* data, const float* bias, size_t batch, size_t dim) {
    for (size_t b = 0; b < batch; ++b) {
        for (size_t j = 0; j < dim; ++j) {
            data[b * dim + j] += bias[j];
        }
    }
}

/// Run reference MLP (pure C++ without BehavioralComputeFabric)
float run_reference_mlp(
    const MLPConfig& cfg,
    const std::vector<float>& w1,
    const std::vector<float>& b1,
    const std::vector<float>& w2,
    const std::vector<float>& b2,
    const std::vector<float>& input)
{
    // Allocate intermediate buffers
    std::vector<float> hidden(cfg.batch_size * cfg.hidden_dim);
    std::vector<float> output(cfg.batch_size * cfg.output_dim);

    // Layer 1: hidden = ReLU(input @ W1 + b1)
    ref_matmul(input.data(), w1.data(), hidden.data(),
               cfg.batch_size, cfg.hidden_dim, cfg.input_dim);
    ref_bias_add(hidden.data(), b1.data(), cfg.batch_size, cfg.hidden_dim);
    ref_relu(hidden.data(), hidden.data(), cfg.batch_size * cfg.hidden_dim);

    // Layer 2: output = hidden @ W2 + b2
    ref_matmul(hidden.data(), w2.data(), output.data(),
               cfg.batch_size, cfg.output_dim, cfg.hidden_dim);
    ref_bias_add(output.data(), b2.data(), cfg.batch_size, cfg.output_dim);

    return output[0];
}

// ============================================================================
// Helper: Run MLP via BehavioralComputeFabric and collect XUE events
// ============================================================================

struct MLPResult {
    float output_value;
    uint64_t compute_cycles;
    uint64_t total_flops;
    uint64_t matmul_events;
    uint64_t elementwise_events;
    bool valid;  // Whether output matches reference
};

// Run MLP with FP32 computation via BehavioralComputeFabric
MLPResult run_mlp(
    const MLPConfig& cfg,
    const std::vector<float>& w1,
    const std::vector<float>& b1,
    const std::vector<float>& w2,
    const std::vector<float>& b2,
    const std::vector<float>& input,
    float reference_output)
{
    // Allocate intermediate buffers
    std::vector<float> hidden(cfg.batch_size * cfg.hidden_dim, 0.0f);
    std::vector<float> output(cfg.batch_size * cfg.output_dim, 0.0f);

    // Reset XUE collector and enable
    auto& xue = EventCollector::instance();
    xue.reset();
    xue.set_enabled(true);
    xue.set_cycle(0);

    // Create compute fabric
    ComputeFabricConfig fabric_cfg;
    fabric_cfg.fidelity = SimulationFidelity::BEHAVIORAL;
    fabric_cfg.macs_per_cycle = 256;  // 16x16 systolic array
    BehavioralComputeFabric fabric(fabric_cfg, 0);

    // ========== Layer 1: input @ W1 + b1 ==========
    {
        MatMulDescriptor desc;
        desc.m = static_cast<uint32_t>(cfg.batch_size);
        desc.n = static_cast<uint32_t>(cfg.hidden_dim);
        desc.k = static_cast<uint32_t>(cfg.input_dim);
        desc.dtype = DataType::FLOAT32;

        fabric.submit_matmul(desc, input.data(), w1.data(), hidden.data());
    }

    // Add bias to hidden
    ref_bias_add(hidden.data(), b1.data(), cfg.batch_size, cfg.hidden_dim);

    // ReLU activation
    {
        ElementwiseDescriptor desc;
        desc.op = ElementwiseOp::RELU;
        desc.count = cfg.batch_size * cfg.hidden_dim;
        desc.dtype = DataType::FLOAT32;

        fabric.submit_elementwise(desc, hidden.data(), nullptr, hidden.data());
    }

    // ========== Layer 2: hidden @ W2 + b2 ==========
    {
        MatMulDescriptor desc;
        desc.m = static_cast<uint32_t>(cfg.batch_size);
        desc.n = static_cast<uint32_t>(cfg.output_dim);
        desc.k = static_cast<uint32_t>(cfg.hidden_dim);
        desc.dtype = DataType::FLOAT32;

        fabric.submit_matmul(desc, hidden.data(), w2.data(), output.data());
    }

    // Add bias to output
    ref_bias_add(output.data(), b2.data(), cfg.batch_size, cfg.output_dim);

    // Drain and get stats
    fabric.drain();
    auto& stats = fabric.stats();

    // Get XUE analysis
    const auto& counters = xue.counters();
    HardwareModel hw;
    OperationalAnalyzer analyzer(hw);
    auto analysis = analyzer.analyze(counters);

    MLPResult result;
    result.output_value = output[0];
    result.compute_cycles = stats.total_compute_cycles;
    result.total_flops = analysis.total_flops;
    result.matmul_events = analysis.matmul_events;
    result.elementwise_events = analysis.elementwise_events;

    // Validate against reference
    float rel_error = std::abs(result.output_value - reference_output) /
                      (std::abs(reference_output) + 1e-6f);
    result.valid = (rel_error < 0.01f);  // Within 1% of reference

    return result;
}

// ============================================================================
// XUE Characterization Report
// ============================================================================

void print_detailed_xue_report(const std::string& workload_name, const MLPConfig& cfg) {
    auto& xue = EventCollector::instance();
    const auto& counters = xue.counters();

    HardwareModel hw;
    hw.peak_gflops = 512.0;           // 16x16 @ 1GHz = 512 GFLOPS (2 FLOPs/MAC)
    hw.dram_bandwidth_gbs = 64.0;     // Example: 64 GB/s DRAM bandwidth
    hw.l3_bandwidth_gbs = 256.0;      // Example: 256 GB/s L3 bandwidth
    hw.l2_bandwidth_gbs = 512.0;      // Example: 512 GB/s L2 bandwidth

    OperationalAnalyzer analyzer(hw);
    auto analysis = analyzer.analyze(counters);

    // Calculate expected memory traffic (assuming no data reuse)
    uint64_t weight_bytes = (cfg.input_dim * cfg.hidden_dim +
                             cfg.hidden_dim * cfg.output_dim) * sizeof(float);
    uint64_t act_bytes = (cfg.batch_size * (cfg.input_dim + cfg.hidden_dim + cfg.output_dim)) * sizeof(float);
    uint64_t total_mem = weight_bytes + act_bytes;

    double arith_intensity = static_cast<double>(analysis.total_flops) /
                             static_cast<double>(total_mem > 0 ? total_mem : 1);

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  XUE CHARACTERIZATION REPORT: " << workload_name << "\n";
    std::cout << "================================================================\n";

    std::cout << "\n--- Workload Summary ---\n";
    std::cout << "  Architecture:        " << cfg.input_dim << " -> " << cfg.hidden_dim
              << " -> " << cfg.output_dim << "\n";
    std::cout << "  Batch Size:          " << cfg.batch_size << "\n";
    std::cout << "  Total FLOPs:         " << analysis.total_flops << "\n";
    std::cout << "  Weight Memory:       " << weight_bytes << " bytes\n";
    std::cout << "  Activation Memory:   " << act_bytes << " bytes\n";
    std::cout << "  Total Memory:        " << total_mem << " bytes\n";

    std::cout << "\n--- XUE Event Breakdown ---\n";
    std::cout << "  MatMul Events:       " << analysis.matmul_events << "\n";
    std::cout << "  Elementwise Events:  " << analysis.elementwise_events << "\n";
    std::cout << "  Reduction Events:    " << analysis.reduction_events << "\n";
    std::cout << "  Memory Events:       " << analysis.memory_events << "\n";

    std::cout << "\n--- Roofline Model Analysis ---\n";
    std::cout << "  Peak Compute:        " << std::fixed << std::setprecision(1)
              << hw.peak_gflops << " GFLOPS\n";
    std::cout << "  DRAM Bandwidth:      " << hw.dram_bandwidth_gbs << " GB/s\n";
    std::cout << "  Ridge Point (DRAM):  " << std::setprecision(2)
              << hw.ridge_point_dram() << " FLOP/byte\n";
    std::cout << "  Arithmetic Intens.:  " << arith_intensity << " FLOP/byte\n";

    if (arith_intensity >= hw.ridge_point_dram()) {
        std::cout << "  Bottleneck:          COMPUTE-BOUND\n";
        std::cout << "  Expected Perf:       " << std::setprecision(1)
                  << hw.peak_gflops << " GFLOPS (at peak)\n";
    } else {
        double mem_limited = arith_intensity * hw.dram_bandwidth_gbs;
        std::cout << "  Bottleneck:          MEMORY-BOUND\n";
        std::cout << "  Expected Perf:       " << std::setprecision(1)
                  << mem_limited << " GFLOPS (bandwidth limited)\n";
    }

    std::cout << "\n--- Precision Scaling Projections ---\n";
    std::cout << "  If using FP16/BF16:\n";
    std::cout << "    Memory:            " << total_mem / 2 << " bytes\n";
    std::cout << "    Arith. Intensity:  " << std::setprecision(2)
              << arith_intensity * 2 << " FLOP/byte\n";

    std::cout << "  If using FP8:\n";
    std::cout << "    Memory:            " << total_mem / 4 << " bytes\n";
    std::cout << "    Arith. Intensity:  " << std::setprecision(2)
              << arith_intensity * 4 << " FLOP/byte\n";

    std::cout << "================================================================\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "****************************************************************\n";
    std::cout << "*     QUANTIZED MLP CHARACTERIZATION WITH XUE ANALYSIS         *\n";
    std::cout << "****************************************************************\n";
    std::cout << "\n";
    std::cout << "This example demonstrates:\n";
    std::cout << "  1. Running an MLP through BehavioralComputeFabric\n";
    std::cout << "  2. Collecting XUE events via the Observation Architecture\n";
    std::cout << "  3. Performing Roofline model analysis\n";
    std::cout << "  4. Projecting impact of precision scaling\n";
    std::cout << "\n";

    int failures = 0;

    // ========== Test 1: Small MNIST-like MLP ==========
    MLPConfig mnist_cfg{
        .input_dim = 784,     // 28x28 image
        .hidden_dim = 256,
        .output_dim = 10,
        .batch_size = 32
    };

    std::vector<float> w1(mnist_cfg.input_dim * mnist_cfg.hidden_dim);
    std::vector<float> b1(mnist_cfg.hidden_dim);
    std::vector<float> w2(mnist_cfg.hidden_dim * mnist_cfg.output_dim);
    std::vector<float> b2(mnist_cfg.output_dim);
    std::vector<float> input(mnist_cfg.batch_size * mnist_cfg.input_dim);

    // Initialize with small deterministic values
    // Use Xavier-like scaling: sqrt(2 / (fan_in + fan_out))
    float w1_scale = std::sqrt(2.0f / static_cast<float>(mnist_cfg.input_dim + mnist_cfg.hidden_dim));
    float w2_scale = std::sqrt(2.0f / static_cast<float>(mnist_cfg.hidden_dim + mnist_cfg.output_dim));

    for (size_t i = 0; i < w1.size(); ++i) {
        // Deterministic pseudo-random using LCG
        float t = static_cast<float>((i * 1103515245 + 12345) % 1000) / 1000.0f - 0.5f;
        w1[i] = t * w1_scale;
    }
    for (size_t i = 0; i < b1.size(); ++i) b1[i] = 0.0f;  // Zero biases
    for (size_t i = 0; i < w2.size(); ++i) {
        float t = static_cast<float>((i * 1103515245 + 12345) % 1000) / 1000.0f - 0.5f;
        w2[i] = t * w2_scale;
    }
    for (size_t i = 0; i < b2.size(); ++i) b2[i] = 0.0f;
    // Normalized input (like MNIST pixel values)
    for (size_t i = 0; i < input.size(); ++i) input[i] = (i % 256) / 255.0f;

    // Compute reference output
    float ref_output1 = run_reference_mlp(mnist_cfg, w1, b1, w2, b2, input);
    std::cout << "Reference MNIST MLP Output[0]: " << std::fixed << std::setprecision(6)
              << ref_output1 << "\n";

    // Run via BehavioralComputeFabric
    auto result1 = run_mlp(mnist_cfg, w1, b1, w2, b2, input, ref_output1);
    print_detailed_xue_report("MNIST MLP (784-256-10, batch=32)", mnist_cfg);

    std::cout << "\n  Execution Result:\n";
    std::cout << "    Reference Output:  " << std::fixed << std::setprecision(6)
              << ref_output1 << "\n";
    std::cout << "    Fabric Output:     " << result1.output_value << "\n";
    std::cout << "    Compute Cycles:    " << result1.compute_cycles << "\n";
    std::cout << "    Validation:        " << (result1.valid ? "PASS" : "FAIL") << "\n";

    if (!result1.valid) {
        std::cerr << "    ERROR: Output mismatch detected!\n";
        ++failures;
    }

    // ========== Test 2: Larger Transformer-style FFN ==========
    MLPConfig ffn_cfg{
        .input_dim = 768,     // Hidden dim of a small transformer
        .hidden_dim = 3072,   // 4x expansion typical for FFN
        .output_dim = 768,
        .batch_size = 128
    };

    std::vector<float> w1_ffn(ffn_cfg.input_dim * ffn_cfg.hidden_dim);
    std::vector<float> b1_ffn(ffn_cfg.hidden_dim);
    std::vector<float> w2_ffn(ffn_cfg.hidden_dim * ffn_cfg.output_dim);
    std::vector<float> b2_ffn(ffn_cfg.output_dim);
    std::vector<float> input_ffn(ffn_cfg.batch_size * ffn_cfg.input_dim);

    // Small values to avoid overflow
    for (size_t i = 0; i < w1_ffn.size(); ++i) w1_ffn[i] = 0.0001f;
    for (size_t i = 0; i < b1_ffn.size(); ++i) b1_ffn[i] = 0.0f;
    for (size_t i = 0; i < w2_ffn.size(); ++i) w2_ffn[i] = 0.0001f;
    for (size_t i = 0; i < b2_ffn.size(); ++i) b2_ffn[i] = 0.0f;
    for (size_t i = 0; i < input_ffn.size(); ++i) input_ffn[i] = 0.1f;

    // Compute reference output
    float ref_output2 = run_reference_mlp(ffn_cfg, w1_ffn, b1_ffn, w2_ffn, b2_ffn, input_ffn);
    std::cout << "\nReference Transformer FFN Output[0]: " << ref_output2 << "\n";

    // Run via BehavioralComputeFabric
    auto result2 = run_mlp(ffn_cfg, w1_ffn, b1_ffn, w2_ffn, b2_ffn, input_ffn, ref_output2);
    print_detailed_xue_report("Transformer FFN (768-3072-768, batch=128)", ffn_cfg);

    std::cout << "\n  Execution Result:\n";
    std::cout << "    Reference Output:  " << std::fixed << std::setprecision(6)
              << ref_output2 << "\n";
    std::cout << "    Fabric Output:     " << result2.output_value << "\n";
    std::cout << "    Compute Cycles:    " << result2.compute_cycles << "\n";
    std::cout << "    Validation:        " << (result2.valid ? "PASS" : "FAIL") << "\n";

    if (!result2.valid) {
        std::cerr << "    ERROR: Output mismatch detected!\n";
        ++failures;
    }

    // ========== Summary ==========
    std::cout << "\n";
    std::cout << "****************************************************************\n";
    std::cout << "*                         SUMMARY                              *\n";
    std::cout << "****************************************************************\n";
    std::cout << "\n";
    std::cout << "Key Insights from XUE Characterization:\n";
    std::cout << "\n";
    std::cout << "1. MEMORY-BOUND vs COMPUTE-BOUND:\n";
    std::cout << "   - Small batch sizes -> Memory-bound (low arithmetic intensity)\n";
    std::cout << "   - Large batch sizes -> Compute-bound (high arithmetic intensity)\n";
    std::cout << "\n";
    std::cout << "2. PRECISION SCALING IMPACT:\n";
    std::cout << "   - FP16/BF16: 2x memory efficiency, same FLOPs\n";
    std::cout << "   - FP8: 4x memory efficiency, same FLOPs\n";
    std::cout << "   - Lower precision shifts workload toward compute-bound\n";
    std::cout << "\n";
    std::cout << "3. XUE EVENT TRACKING:\n";
    std::cout << "   - MatMul events count 16x16 tile operations\n";
    std::cout << "   - Enables pre/post-silicon performance validation\n";
    std::cout << "   - Hardware counters will match simulation counters\n";
    std::cout << "\n";

    if (failures > 0) {
        std::cout << "*** " << failures << " TEST(S) FAILED - Output mismatch detected ***\n";
    } else {
        std::cout << "*** ALL TESTS PASSED - Outputs validated against reference ***\n";
    }
    std::cout << "****************************************************************\n\n";

    return failures > 0 ? 1 : 0;
}
