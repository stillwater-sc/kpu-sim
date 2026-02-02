// ============================================================================
// verification/kernels/class2_spatial_conv/verify_conv2d.cpp
// Class 2 Spatial Convolution — Conv2D verification
//
// Tests Conv2D through BehavioralComputeFabric against inline reference.
// Covers: standard conv, strided, padded, dilated, with/without bias.
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

struct Conv2DConfig {
    const char* name;
    uint32_t batch, in_ch, in_h, in_w;
    uint32_t out_ch, k_h, k_w;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
    uint32_t dilation_h, dilation_w;
    bool has_bias;
};

static const Conv2DConfig configs[] = {
    // Basic 3x3
    { "basic_3x3",       1, 1, 8, 8,    4, 3, 3,   1, 1,  0, 0,  1, 1, false },
    // 3x3 with padding (same)
    { "3x3_pad1",        1, 3, 16, 16,  8, 3, 3,   1, 1,  1, 1,  1, 1, true },
    // 3x3 stride 2 (downsampling)
    { "3x3_stride2",     1, 3, 32, 32, 16, 3, 3,   2, 2,  1, 1,  1, 1, true },
    // 1x1 pointwise
    { "1x1_pointwise",   1, 16, 8, 8,  32, 1, 1,   1, 1,  0, 0,  1, 1, true },
    // 5x5 kernel
    { "5x5_pad2",        1, 3, 16, 16, 16, 5, 5,   1, 1,  2, 2,  1, 1, true },
    // Dilated 3x3
    { "3x3_dilated2",    1, 8, 16, 16, 16, 3, 3,   1, 1,  2, 2,  2, 2, false },
    // Batched
    { "batch4_3x3",      4, 3, 16, 16, 16, 3, 3,   1, 1,  1, 1,  1, 1, true },
    // Small (LeNet-style)
    { "lenet_conv1",     1, 1, 28, 28,  6, 5, 5,   1, 1,  0, 0,  1, 1, true },
    // VGG-style 3x3 stack input
    { "vgg_3x3",         1, 64, 8, 8, 128, 3, 3,   1, 1,  1, 1,  1, 1, true },
    // Rectangular input
    { "rect_input",      1, 3, 24, 32, 16, 3, 3,   1, 1,  1, 1,  1, 1, false },
};

static constexpr float TOLERANCE = 1e-4f;

// ============================================================================
// Main
// ============================================================================

int main() {
    ComputeHarness harness;
    std::vector<TestResult> results;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ComputeHarness::print_results_header("Class 2: Conv2D Verification");

    for (const auto& cfg : configs) {
        Conv2DDescriptor desc;
        desc.batch_size = cfg.batch;
        desc.in_channels = cfg.in_ch;
        desc.in_height = cfg.in_h;
        desc.in_width = cfg.in_w;
        desc.out_channels = cfg.out_ch;
        desc.kernel_height = cfg.k_h;
        desc.kernel_width = cfg.k_w;
        desc.stride_h = cfg.stride_h;
        desc.stride_w = cfg.stride_w;
        desc.padding_h = cfg.pad_h;
        desc.padding_w = cfg.pad_w;
        desc.dilation_h = cfg.dilation_h;
        desc.dilation_w = cfg.dilation_w;
        desc.has_bias = cfg.has_bias;
        desc.dtype = DataType::FLOAT32;

        size_t in_size = static_cast<size_t>(cfg.batch) * cfg.in_ch * cfg.in_h * cfg.in_w;
        size_t w_size = static_cast<size_t>(cfg.out_ch) * cfg.in_ch * cfg.k_h * cfg.k_w;
        size_t out_size = static_cast<size_t>(cfg.batch) * cfg.out_ch * desc.out_height() * desc.out_width();

        std::vector<float> input(in_size), weight(w_size), bias_vec(cfg.out_ch, 0.0f);
        std::vector<float> output(out_size, 0.0f);

        for (size_t i = 0; i < in_size; ++i) input[i] = dist(rng);
        for (size_t i = 0; i < w_size; ++i) weight[i] = dist(rng);
        if (cfg.has_bias) {
            for (size_t i = 0; i < cfg.out_ch; ++i) bias_vec[i] = dist(rng);
        }

        float max_diff = harness.run_conv2d(desc, input.data(), weight.data(),
                                            cfg.has_bias ? bias_vec.data() : nullptr,
                                            output.data());

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "%ux%ux%ux%u→%u",
                      cfg.batch, cfg.in_ch, cfg.in_h, cfg.in_w, cfg.out_ch);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = TOLERANCE;
        r.passed = (max_diff <= TOLERANCE);

        if (!r.passed) {
            harness.add_violation(cfg.name, "max_diff exceeds tolerance", max_diff, TOLERANCE);
        }

        ComputeHarness::print_result(r);
        results.push_back(r);
    }

    harness.print_stats();
    return ComputeHarness::print_summary(results);
}
