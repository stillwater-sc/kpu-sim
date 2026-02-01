// ============================================================================
// verification/kernels/class1_dense_linear/verify_fused_ops.cpp
// Class 1 Dense Linear — fused operation verification
// 4 fusion patterns × 3 sizes
//
// Verifies that running matmul followed by elementwise ops through the
// behavioral compute fabric matches an unfused reference composition.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../common/compute_harness.hpp"

using namespace sw::kpu;
using namespace sw::kpu::verification;

// ============================================================================
// Test configuration
// ============================================================================

struct FusedConfig {
    uint32_t M, N, K;
};

static const FusedConfig sizes[] = {
    { 32, 128,  64 },
    {  1,  64,  10 },
    { 16, 256, 128 },
};

enum class FusionPattern {
    MATMUL_RELU,
    MATMUL_BIAS_RELU,
    MATMUL_BIAS_GELU,
    MATMUL_BIAS_SILU,
};

struct PatternConfig {
    FusionPattern pattern;
    const char* name;
    bool has_bias;
    ElementwiseOp activation;
};

static const PatternConfig patterns[] = {
    { FusionPattern::MATMUL_RELU,      "matmul+relu",      false, ElementwiseOp::RELU },
    { FusionPattern::MATMUL_BIAS_RELU, "matmul+bias+relu", true,  ElementwiseOp::RELU },
    { FusionPattern::MATMUL_BIAS_GELU, "matmul+bias+gelu", true,  ElementwiseOp::GELU },
    { FusionPattern::MATMUL_BIAS_SILU, "matmul+bias+silu", true,  ElementwiseOp::SILU },
};

static constexpr float TOLERANCE = 1e-5f;

// ============================================================================
// Run a fused pattern through fabric (unfused: matmul, then bias add, then act)
// ============================================================================

static float run_fused_test(ComputeHarness& harness,
                            const PatternConfig& pat,
                            const FusedConfig& sz,
                            std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t sizeA = static_cast<size_t>(sz.M) * sz.K;
    size_t sizeB = static_cast<size_t>(sz.K) * sz.N;
    size_t sizeC = static_cast<size_t>(sz.M) * sz.N;

    std::vector<float> A(sizeA), B(sizeB);
    std::vector<float> bias(sz.N, 0.0f);

    for (size_t i = 0; i < sizeA; ++i) A[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) B[i] = dist(rng);
    if (pat.has_bias) {
        for (size_t i = 0; i < sz.N; ++i) bias[i] = dist(rng);
    }

    // --- Fabric path: matmul, then optional bias add, then activation ---
    std::vector<float> fabric_out(sizeC, 0.0f);
    {
        MatMulDescriptor desc;
        desc.m = sz.M; desc.n = sz.N; desc.k = sz.K;
        desc.dtype = DataType::FLOAT32;
        harness.fabric().submit_matmul(desc, A.data(), B.data(), fabric_out.data());
        harness.fabric().drain();
    }

    if (pat.has_bias) {
        // Broadcast bias: add bias[j] to each row
        // Create a full bias buffer matching output shape
        std::vector<float> bias_full(sizeC);
        for (uint32_t i = 0; i < sz.M; ++i) {
            for (uint32_t j = 0; j < sz.N; ++j) {
                bias_full[i * sz.N + j] = bias[j];
            }
        }

        ElementwiseDescriptor edesc;
        edesc.op = ElementwiseOp::ADD;
        edesc.count = sizeC;
        edesc.dtype = DataType::FLOAT32;

        std::vector<float> biased(sizeC);
        harness.fabric().submit_elementwise(edesc, fabric_out.data(),
                                            bias_full.data(), biased.data());
        harness.fabric().drain();
        fabric_out = std::move(biased);
    }

    {
        ElementwiseDescriptor edesc;
        edesc.op = pat.activation;
        edesc.count = sizeC;
        edesc.dtype = DataType::FLOAT32;

        std::vector<float> activated(sizeC);
        harness.fabric().submit_elementwise(edesc, fabric_out.data(),
                                            nullptr, activated.data());
        harness.fabric().drain();
        fabric_out = std::move(activated);
    }

    // --- Reference path: pure C++ ---
    std::vector<float> ref_out(sizeC);
    ComputeHarness::ref_matmul(sz.M, sz.N, sz.K, A.data(), B.data(), ref_out.data());

    if (pat.has_bias) {
        for (uint32_t i = 0; i < sz.M; ++i) {
            for (uint32_t j = 0; j < sz.N; ++j) {
                ref_out[i * sz.N + j] += bias[j];
            }
        }
    }

    for (size_t i = 0; i < sizeC; ++i) {
        ref_out[i] = ComputeHarness::ref_elementwise_scalar(
            pat.activation, ref_out[i], 0.0f);
    }

    return ComputeHarness::max_abs_diff(fabric_out.data(), ref_out.data(), sizeC);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    ComputeHarness harness;
    std::vector<TestResult> results;
    std::mt19937 rng(42);

    ComputeHarness::print_results_header("Class 1: Fused Ops Verification");

    for (const auto& pat : patterns) {
        for (const auto& sz : sizes) {
            float max_diff = run_fused_test(harness, pat, sz, rng);

            char shape_buf[64];
            std::snprintf(shape_buf, sizeof(shape_buf), "(%u,%u,%u)", sz.M, sz.N, sz.K);

            TestResult r;
            r.op_name = pat.name;
            r.shape = shape_buf;
            r.max_diff = max_diff;
            r.tolerance = TOLERANCE;
            r.passed = (max_diff <= TOLERANCE);

            if (!r.passed) {
                harness.add_violation(
                    std::string(pat.name) + "/" + shape_buf,
                    "max_diff exceeds tolerance",
                    max_diff, TOLERANCE);
            }

            ComputeHarness::print_result(r);
            results.push_back(r);
        }
    }

    harness.print_stats();
    return ComputeHarness::print_summary(results);
}
