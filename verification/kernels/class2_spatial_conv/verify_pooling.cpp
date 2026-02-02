// ============================================================================
// verification/kernels/class2_spatial_conv/verify_pooling.cpp
// Class 2 Spatial Convolution — Pooling verification
//
// Tests MaxPool2D, AvgPool2D, AdaptiveAvgPool2D through BehavioralComputeFabric.
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

struct PoolConfig {
    const char* name;
    Pool2DDescriptor::PoolType pool_type;
    uint32_t batch, channels, in_h, in_w;
    uint32_t k_h, k_w;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
    // For adaptive pooling
    uint32_t target_h, target_w;
};

static const PoolConfig configs[] = {
    // MaxPool 2x2 stride 2
    { "maxpool_2x2",     Pool2DDescriptor::PoolType::MAX, 1, 16, 8, 8,   2, 2,  2, 2,  0, 0,  0, 0 },
    // MaxPool 3x3 stride 2 pad 1
    { "maxpool_3x3",     Pool2DDescriptor::PoolType::MAX, 1, 32, 16, 16, 3, 3,  2, 2,  1, 1,  0, 0 },
    // AvgPool 2x2 stride 2
    { "avgpool_2x2",     Pool2DDescriptor::PoolType::AVG, 1, 16, 8, 8,   2, 2,  2, 2,  0, 0,  0, 0 },
    // AvgPool 3x3 stride 1 pad 1
    { "avgpool_3x3_s1",  Pool2DDescriptor::PoolType::AVG, 1, 8, 8, 8,    3, 3,  1, 1,  1, 1,  0, 0 },
    // Batched MaxPool
    { "batch_maxpool",   Pool2DDescriptor::PoolType::MAX, 4, 16, 16, 16, 2, 2,  2, 2,  0, 0,  0, 0 },
    // Adaptive AvgPool to 1x1
    { "adaptive_1x1",    Pool2DDescriptor::PoolType::ADAPTIVE_AVG, 1, 64, 7, 7,   0, 0,  0, 0,  0, 0,  1, 1 },
    // Adaptive AvgPool to 4x4
    { "adaptive_4x4",    Pool2DDescriptor::PoolType::ADAPTIVE_AVG, 1, 32, 16, 16, 0, 0,  0, 0,  0, 0,  4, 4 },
    // Large maxpool
    { "maxpool_large",   Pool2DDescriptor::PoolType::MAX, 1, 64, 32, 32, 2, 2,  2, 2,  0, 0,  0, 0 },
    // Rectangular input
    { "maxpool_rect",    Pool2DDescriptor::PoolType::MAX, 1, 8, 12, 16,  2, 2,  2, 2,  0, 0,  0, 0 },
};

// MaxPool should be exact, AvgPool can have small FP rounding
static float tolerance_for(Pool2DDescriptor::PoolType pt) {
    return (pt == Pool2DDescriptor::PoolType::MAX) ? 0.0f : 1e-6f;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    ComputeHarness harness;
    std::vector<TestResult> results;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    ComputeHarness::print_results_header("Class 2: Pooling Verification");

    for (const auto& cfg : configs) {
        Pool2DDescriptor desc;
        desc.pool_type = cfg.pool_type;
        desc.batch_size = cfg.batch;
        desc.channels = cfg.channels;
        desc.in_height = cfg.in_h;
        desc.in_width = cfg.in_w;
        desc.kernel_height = cfg.k_h;
        desc.kernel_width = cfg.k_w;
        desc.stride_h = cfg.stride_h;
        desc.stride_w = cfg.stride_w;
        desc.padding_h = cfg.pad_h;
        desc.padding_w = cfg.pad_w;
        desc.target_out_height = cfg.target_h;
        desc.target_out_width = cfg.target_w;
        desc.dtype = DataType::FLOAT32;

        size_t in_size = static_cast<size_t>(cfg.batch) * cfg.channels * cfg.in_h * cfg.in_w;
        uint32_t H_out = desc.out_height();
        uint32_t W_out = desc.out_width();
        size_t out_size = static_cast<size_t>(cfg.batch) * cfg.channels * H_out * W_out;

        std::vector<float> input(in_size), output(out_size, 0.0f);
        for (size_t i = 0; i < in_size; ++i) input[i] = dist(rng);

        float max_diff = harness.run_pool2d(desc, input.data(), output.data());
        float tol = tolerance_for(cfg.pool_type);

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "%ux%ux%ux%u",
                      cfg.batch, cfg.channels, cfg.in_h, cfg.in_w);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = tol;
        r.passed = (max_diff <= tol);

        if (!r.passed) {
            harness.add_violation(cfg.name, "max_diff exceeds tolerance", max_diff, tol);
        }

        ComputeHarness::print_result(r);
        results.push_back(r);
    }

    harness.print_stats();
    return ComputeHarness::print_summary(results);
}
