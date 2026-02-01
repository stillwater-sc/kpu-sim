// ============================================================================
// verification/kernels/class1_dense_linear/verify_matmul.cpp
// Class 1 Dense Linear kernel verification — 10 dimension configs
//
// Standalone verification program for BehavioralComputeFabric matmul.
// Compares fabric output against triple-loop reference implementation.
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

struct MatmulConfig {
    uint32_t M, N, K;
    const char* category;
};

static const MatmulConfig configs[] = {
    // Aligned square
    {  16,  16,  16, "aligned" },
    {  32,  32,  32, "aligned" },
    { 128, 128, 128, "aligned" },
    { 256, 256, 256, "aligned" },
    // Rectangular
    {   1, 128,  10, "rectangular" },
    {  32, 784, 128, "rectangular" },
    {  32, 128,  10, "rectangular" },
    // Non-aligned
    {   7,  13,   5, "non-aligned" },
    {  33,  65,  17, "non-aligned" },
    // Large
    { 512, 512, 512, "large" },
};

static constexpr float TOLERANCE = 1e-5f;

// ============================================================================
// Main
// ============================================================================

int main() {
    ComputeHarness harness;
    std::vector<TestResult> results;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ComputeHarness::print_results_header("Class 1: Matmul Verification");

    for (const auto& cfg : configs) {
        size_t sizeA = static_cast<size_t>(cfg.M) * cfg.K;
        size_t sizeB = static_cast<size_t>(cfg.K) * cfg.N;
        size_t sizeC = static_cast<size_t>(cfg.M) * cfg.N;

        std::vector<float> A(sizeA), B(sizeB), C(sizeC, 0.0f);
        for (size_t i = 0; i < sizeA; ++i) A[i] = dist(rng);
        for (size_t i = 0; i < sizeB; ++i) B[i] = dist(rng);

        float max_diff = harness.run_matmul(cfg.M, cfg.N, cfg.K,
                                            A.data(), B.data(), C.data());

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "(%u,%u,%u)", cfg.M, cfg.N, cfg.K);

        TestResult r;
        r.op_name = cfg.category;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = TOLERANCE;
        r.passed = (max_diff <= TOLERANCE);

        if (!r.passed) {
            harness.add_violation(
                std::string("matmul/") + shape_buf,
                "max_diff exceeds tolerance",
                max_diff, TOLERANCE);
        }

        ComputeHarness::print_result(r);
        results.push_back(r);
    }

    harness.print_stats();
    return ComputeHarness::print_summary(results);
}
