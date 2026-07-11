/**
 * @file m1_mlp_baseline.cpp
 * @brief DNN Milestone M1: MLP baseline demo + benchmark on the CSP executor.
 *
 * The first rung of the DNN milestone ladder (issue #129, plan Section 6):
 * MLP inference with real values flowing through the credit-based dataflow
 * pipeline (DRAM -> L3 -> L2 -> compute -> L2 -> L3 -> DRAM).
 *
 * Three-tier definition of done, all exercised here:
 *  - Demonstrate: end-to-end CSP execution; --trace-dir exports Chrome
 *    traces of the credit dataflow (view at https://ui.perfetto.dev)
 *  - Validate: elementwise comparison against a host reference oracle
 *    (exact expected outputs for XOR; computed oracle for the MNIST-shape
 *    network with deterministic synthetic weights)
 *  - Benchmark: cycles, cycles/sample, stall breakdown across batch sizes
 *    and pipeline configurations
 *
 * Usage:
 *   m1_mlp_baseline [--trace-dir DIR]
 *
 * Writeup: docs/milestones/M1_mlp_baseline.md
 */

#include <sw/kpu/timing/functional_mlp_executor.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using sw::kpu::timing::ConcurrentTimingExecutor;
using sw::kpu::timing::FunctionalMLPExecutor;
using sw::kpu::timing::Size;
using Activation = ConcurrentTimingExecutor::FunctionalActivation;

namespace {

// ============================================================================
// Deterministic synthetic weights (fixed-seed LCG - reproducible everywhere)
// ============================================================================

class Lcg {
public:
    explicit Lcg(uint64_t seed) : state_(seed) {}
    float next_weight() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        // Map the top bits to [-0.1, 0.1] - small weights keep activations
        // in a numerically comfortable range through three layers
        auto bits = static_cast<uint32_t>(state_ >> 33);
        return (static_cast<float>(bits) / static_cast<float>(0x7FFFFFFF) - 1.0f) * 0.1f;
    }

private:
    uint64_t state_;
};

std::vector<float> synthetic(size_t count, uint64_t seed) {
    Lcg lcg(seed);
    std::vector<float> v(count);
    for (auto& x : v) x = lcg.next_weight();
    return v;
}

// ============================================================================
// Host reference oracle: plain-loop forward pass over the same layer specs
// ============================================================================

std::vector<float> reference_forward(const std::vector<FunctionalMLPExecutor::Layer>& layers,
                                     const std::vector<float>& input, Size batch) {
    std::vector<float> current = input;
    for (const auto& layer : layers) {
        std::vector<float> next(static_cast<size_t>(batch) * layer.output_dim, 0.0f);
        for (Size b = 0; b < batch; ++b) {
            for (Size o = 0; o < layer.output_dim; ++o) {
                float acc = layer.bias.empty() ? 0.0f : layer.bias[o];
                for (Size i = 0; i < layer.input_dim; ++i) {
                    acc += current[b * layer.input_dim + i] *
                           layer.weights[i * layer.output_dim + o];
                }
                if (layer.activation == Activation::RELU && acc < 0.0f) acc = 0.0f;
                next[b * layer.output_dim + o] = acc;
            }
        }
        current = std::move(next);
    }
    return current;
}

// ============================================================================
// Benchmark harness
// ============================================================================

struct RunResult {
    std::string name;
    Size batch = 0;
    bool pass = false;
    float max_abs_err = 0.0f;
    FunctionalMLPExecutor::Statistics stats;
};

ConcurrentTimingExecutor::Config pipeline_config(bool minimal) {
    ConcurrentTimingExecutor::Config config;
    if (minimal) {
        config.num_memory_controllers = 1;
        config.num_dma_engines = 1;
        config.num_block_movers = 1;
        config.num_row_streamers = 1;
        config.num_col_streamers = 1;
        config.l3_buffer_count = 4;
        config.l2_bank_count = 4;
    }
    // Defaults otherwise: 1 MC, 1 DMA, 4 BM, 2+2 streamers, 32 L3, 64 L2
    config.max_cycles = 10'000'000;
    return config;
}

RunResult run_mlp(const std::string& name,
                  FunctionalMLPExecutor& mlp,
                  const std::vector<float>& input, Size batch,
                  const std::vector<float>& expected,
                  const std::string& trace_file) {
    if (!trace_file.empty()) mlp.set_trace_file(trace_file);

    RunResult r;
    r.name = name;
    r.batch = batch;
    const auto output = mlp.forward(input, batch);

    r.pass = output.size() == expected.size();
    for (size_t i = 0; i < output.size() && i < expected.size(); ++i) {
        float err = std::fabs(output[i] - expected[i]);
        if (err > r.max_abs_err) r.max_abs_err = err;
    }
    // MNIST-shape magnitudes stay O(1) with the synthetic weight scale, so a
    // uniform absolute tolerance is appropriate for fp32 accumulation depths
    // up to 784
    r.pass = r.pass && r.max_abs_err < 1e-4f;
    r.stats = mlp.statistics();
    return r;
}

RunResult run_xor(bool minimal_pipeline, const std::string& trace_file) {
    FunctionalMLPExecutor mlp(pipeline_config(minimal_pipeline));
    mlp.add_layer(2, 4,
        {1.0f, 1.0f, -1.0f, -1.0f,
         1.0f, 1.0f, -1.0f, -1.0f},
        {-0.5f, -1.5f, 0.5f, 1.5f},
        Activation::RELU, "hidden");
    mlp.add_layer(4, 1, {2.0f, -6.0f, 0.0f, 0.0f}, {0.0f},
        Activation::NONE, "output");

    const std::vector<float> input = {0, 0, 0, 1, 1, 0, 1, 1};
    const std::vector<float> expected = {0.0f, 1.0f, 1.0f, 0.0f};
    std::string name = std::string("XOR 2-4-1 ") +
                       (minimal_pipeline ? "(minimal pipeline)" : "(default pipeline)");
    return run_mlp(name, mlp, input, 4, expected, trace_file);
}

RunResult run_mnist_shape(Size batch, bool minimal_pipeline,
                          const std::string& trace_file) {
    // The canonical MNIST MLP shape: 784 -> 128 -> 64 -> 10, synthetic
    // deterministic weights, oracle-validated. Real trained weights are a
    // drop-in once a weight-loading path lands (E5/E15 scope).
    FunctionalMLPExecutor mlp(pipeline_config(minimal_pipeline));
    mlp.add_layer(784, 128, synthetic(784 * 128, 1), synthetic(128, 2),
                  Activation::RELU, "fc1");
    mlp.add_layer(128, 64, synthetic(128 * 64, 3), synthetic(64, 4),
                  Activation::RELU, "fc2");
    mlp.add_layer(64, 10, synthetic(64 * 10, 5), synthetic(10, 6),
                  Activation::NONE, "logits");

    const auto input = synthetic(static_cast<size_t>(batch) * 784, 7 + batch);
    const auto expected = reference_forward(mlp.layers(), input, batch);

    std::string name = "MNIST-shape 784-128-64-10 " +
                       std::string(minimal_pipeline ? "(minimal pipeline)" : "(default pipeline)");
    return run_mlp(name, mlp, input, batch, expected, trace_file);
}

void print_row(const RunResult& r) {
    const auto& t = r.stats.timing;
    double per_sample = r.batch > 0
        ? static_cast<double>(r.stats.total_cycles) / r.batch : 0.0;
    std::cout << "  " << std::left << std::setw(44) << r.name
              << std::right
              << std::setw(6) << r.batch
              << std::setw(10) << r.stats.total_cycles
              << std::setw(12) << std::fixed << std::setprecision(1) << per_sample
              << std::setw(9) << t.dma_credit_stalls
              << std::setw(9) << (t.bm_tag_stalls + t.bm_credit_stalls)
              << std::setw(9) << (t.str_tag_stalls + t.str_credit_stalls)
              << std::setw(12) << std::scientific << std::setprecision(1)
              << r.max_abs_err
              << std::setw(7) << (r.pass ? "PASS" : "FAIL") << "\n"
              << std::defaultfloat;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string trace_dir;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace-dir") == 0 && i + 1 < argc) {
            trace_dir = argv[++i];
        } else {
            std::cout << "Usage: " << argv[0] << " [--trace-dir DIR]\n";
            return std::strcmp(argv[i], "--help") == 0 ? 0 : 1;
        }
    }
    if (!trace_dir.empty()) {
        // The trace exporter opens the path directly; create the directory
        // so a fresh checkout does not silently produce no files
        std::error_code ec;
        std::filesystem::create_directories(trace_dir, ec);
        if (ec) {
            std::cerr << "Cannot create trace directory '" << trace_dir
                      << "': " << ec.message() << "\n";
            return 1;
        }
    }
    auto trace = [&](const std::string& base) {
        return trace_dir.empty() ? std::string{} : trace_dir + "/" + base + ".json";
    };

    std::cout << "\n"
        "======================================================================\n"
        "DNN Milestone M1: MLP baseline on the CSP concurrent timing executor\n"
        "Values flow through the real credit-dataflow pipeline; results are\n"
        "validated elementwise against a host reference oracle.\n"
        "======================================================================\n\n";

    std::vector<RunResult> results;
    results.push_back(run_xor(true, trace("m1_xor_minimal")));
    results.push_back(run_xor(false, trace("m1_xor_default")));
    results.push_back(run_mnist_shape(16, false, trace("m1_mnist_b16")));
    results.push_back(run_mnist_shape(64, false, trace("m1_mnist_b64")));
    results.push_back(run_mnist_shape(256, false, trace("m1_mnist_b256")));
    results.push_back(run_mnist_shape(64, true, trace("m1_mnist_b64_minimal")));

    std::cout << "  " << std::left << std::setw(44) << "configuration"
              << std::right << std::setw(6) << "batch"
              << std::setw(10) << "cycles" << std::setw(12) << "cyc/sample"
              << std::setw(9) << "dmaStl" << std::setw(9) << "bmStl"
              << std::setw(9) << "strStl" << std::setw(12) << "maxAbsErr"
              << std::setw(7) << "check" << "\n";
    std::cout << "  " << std::string(116, '-') << "\n";

    bool all_pass = true;
    for (const auto& r : results) {
        print_row(r);
        all_pass = all_pass && r.pass;
    }

    std::cout << "\n  Validation: " << (all_pass ? "ALL PASS" : "FAILURES PRESENT")
              << " (oracle: host reference forward pass, tolerance 1e-4)\n";
    if (!trace_dir.empty()) {
        std::cout << "  Chrome traces written to " << trace_dir
                  << "/ - view at https://ui.perfetto.dev\n";
    }
    std::cout << "\n";
    return all_pass ? 0 : 1;
}
