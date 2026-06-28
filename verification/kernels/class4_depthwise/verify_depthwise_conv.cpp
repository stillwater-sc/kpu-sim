// ============================================================================
// verification/kernels/class4_depthwise/verify_depthwise_conv.cpp
// Class 4 Depthwise Separable — depthwise conv, pointwise conv, grouped conv,
// and MobileNet-style inverted residual block verification.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../common/compute_harness.hpp"

using namespace sw::kpu;
using namespace sw::kpu::verification;

static constexpr float TOLERANCE = 1e-4f;

// ============================================================================
// Test 1: Pure depthwise 3x3 (groups = C_in)
// ============================================================================

struct DepthwiseConfig {
    const char* name;
    uint32_t batch, channels, in_h, in_w;
    uint32_t k_h, k_w;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
};

static const DepthwiseConfig dw_configs[] = {
    { "dw_3x3",       1, 16, 16, 16,  3, 3,  1, 1,  1, 1 },
    { "dw_3x3_s2",    1, 32, 16, 16,  3, 3,  2, 2,  1, 1 },
    { "dw_5x5",       1, 16, 8, 8,    5, 5,  1, 1,  2, 2 },
    { "dw_batch",     4, 16, 8, 8,    3, 3,  1, 1,  1, 1 },
    { "dw_large_ch",  1, 96, 8, 8,    3, 3,  1, 1,  1, 1 },
};

// ============================================================================
// Test 2: Depthwise separable = depthwise 3x3 + pointwise 1x1
// ============================================================================

struct SeparableConfig {
    const char* name;
    uint32_t batch, C_in, H, W, C_out;
};

static const SeparableConfig sep_configs[] = {
    { "sep_16→32",  1, 16, 16, 16, 32 },
    { "sep_32→64",  1, 32, 8, 8,   64 },
    { "sep_96→24",  1, 96, 8, 8,   24 },
};

// ============================================================================
// Test 3: Grouped convolution (intermediate between standard and depthwise)
// ============================================================================

struct GroupedConfig {
    const char* name;
    uint32_t batch, C_in, H, W, C_out, groups;
    uint32_t k_h, k_w;
};

static const GroupedConfig grp_configs[] = {
    { "grp2_3x3",  1, 16, 8, 8, 16, 2,  3, 3 },
    { "grp4_3x3",  1, 32, 8, 8, 32, 4,  3, 3 },
    { "grp8_1x1",  1, 64, 4, 4, 64, 8,  1, 1 },
};

// ============================================================================
// Test 4: Inverted residual block (MobileNetV2 pattern)
// x → expand(1x1) → relu6 → depthwise(3x3) → relu6 → project(1x1) → + → out
//  └──────────────────────────────────────────────────────────────────┘
// (residual only if input/output channels match and stride=1)
// ============================================================================

struct InvertedResConfig {
    const char* name;
    uint32_t C_in, H, W, C_out, expand_ratio, stride;
    bool has_residual;
};

static const InvertedResConfig ir_configs[] = {
    { "ir_16→16_e6",   16, 8, 8, 16, 6, 1, true  },
    { "ir_16→24_e6",   16, 8, 8, 24, 6, 2, false },
    { "ir_24→24_e6",   24, 4, 4, 24, 6, 1, true  },
};

// relu6 helper
static float relu6(float x) {
    return std::min(std::max(0.0f, x), 6.0f);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    ComputeHarness harness;
    std::vector<TestResult> results;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ComputeHarness::print_results_header("Class 4: Depthwise Separable Verification");

    // --- Depthwise conv tests ---
    for (const auto& cfg : dw_configs) {
        Conv2DDescriptor desc;
        desc.batch_size = cfg.batch;
        desc.in_channels = cfg.channels;
        desc.in_height = cfg.in_h;
        desc.in_width = cfg.in_w;
        desc.out_channels = cfg.channels;  // depthwise: out_ch = in_ch
        desc.kernel_height = cfg.k_h;
        desc.kernel_width = cfg.k_w;
        desc.stride_h = cfg.stride_h;
        desc.stride_w = cfg.stride_w;
        desc.padding_h = cfg.pad_h;
        desc.padding_w = cfg.pad_w;
        desc.groups = cfg.channels;  // depthwise: groups = channels
        desc.dtype = DataType::FLOAT32;

        size_t in_size = static_cast<size_t>(cfg.batch) * cfg.channels * cfg.in_h * cfg.in_w;
        // Depthwise weight: [C, 1, K_h, K_w]
        size_t w_size = static_cast<size_t>(cfg.channels) * 1 * cfg.k_h * cfg.k_w;
        size_t out_size = static_cast<size_t>(cfg.batch) * cfg.channels * desc.out_height() * desc.out_width();

        std::vector<float> input(in_size), weight(w_size), output(out_size, 0.0f);
        for (auto& v : input) v = dist(rng);
        for (auto& v : weight) v = dist(rng);

        float max_diff = harness.run_conv2d(desc, input.data(), weight.data(), nullptr, output.data());

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "%ux%ux%ux%u",
                      cfg.batch, cfg.channels, cfg.in_h, cfg.in_w);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = TOLERANCE;
        r.passed = (max_diff <= TOLERANCE);
        ComputeHarness::print_result(r);
        results.push_back(r);
    }

    // --- Depthwise separable tests ---
    for (const auto& cfg : sep_configs) {
        size_t sp = static_cast<size_t>(cfg.H) * cfg.W;

        // Depthwise 3x3
        Conv2DDescriptor dw;
        dw.batch_size = cfg.batch; dw.in_channels = cfg.C_in;
        dw.in_height = cfg.H; dw.in_width = cfg.W;
        dw.out_channels = cfg.C_in; dw.kernel_height = 3; dw.kernel_width = 3;
        dw.stride_h = 1; dw.stride_w = 1; dw.padding_h = 1; dw.padding_w = 1;
        dw.groups = cfg.C_in; dw.dtype = DataType::FLOAT32;

        // Pointwise 1x1
        Conv2DDescriptor pw;
        pw.batch_size = cfg.batch; pw.in_channels = cfg.C_in;
        pw.in_height = cfg.H; pw.in_width = cfg.W;
        pw.out_channels = cfg.C_out; pw.kernel_height = 1; pw.kernel_width = 1;
        pw.stride_h = 1; pw.stride_w = 1; pw.dtype = DataType::FLOAT32;

        size_t in_size = cfg.batch * cfg.C_in * sp;
        size_t dw_w_size = static_cast<size_t>(cfg.C_in) * 1 * 3 * 3;
        size_t pw_w_size = static_cast<size_t>(cfg.C_out) * cfg.C_in;
        size_t out_size = cfg.batch * cfg.C_out * sp;

        std::vector<float> input(in_size), dw_w(dw_w_size), pw_w(pw_w_size);
        for (auto& v : input) v = dist(rng);
        for (auto& v : dw_w) v = dist(rng);
        for (auto& v : pw_w) v = dist(rng);

        // Fabric: dw → relu → pw
        std::vector<float> dw_out(in_size), relu_out(in_size), pw_out(out_size);
        harness.fabric().submit_conv2d(dw, input.data(), dw_w.data(), nullptr, dw_out.data());
        harness.fabric().drain();

        ElementwiseDescriptor edesc;
        edesc.op = ElementwiseOp::RELU; edesc.count = in_size; edesc.dtype = DataType::FLOAT32;
        harness.fabric().submit_elementwise(edesc, dw_out.data(), nullptr, relu_out.data());
        harness.fabric().drain();

        harness.fabric().submit_conv2d(pw, relu_out.data(), pw_w.data(), nullptr, pw_out.data());
        harness.fabric().drain();

        // Reference
        std::vector<float> ref_dw(in_size), ref_relu(in_size), ref_pw(out_size);
        ComputeHarness::ref_conv2d(dw, input.data(), dw_w.data(), nullptr, ref_dw.data());
        for (size_t i = 0; i < in_size; ++i) ref_relu[i] = std::max(0.0f, ref_dw[i]);
        ComputeHarness::ref_conv2d(pw, ref_relu.data(), pw_w.data(), nullptr, ref_pw.data());

        float max_diff = ComputeHarness::max_abs_diff(pw_out.data(), ref_pw.data(), out_size);

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "%u→%u@%ux%u",
                      cfg.C_in, cfg.C_out, cfg.H, cfg.W);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = TOLERANCE;
        r.passed = (max_diff <= TOLERANCE);
        ComputeHarness::print_result(r);
        results.push_back(r);
    }

    // --- Grouped conv tests ---
    for (const auto& cfg : grp_configs) {
        Conv2DDescriptor desc;
        desc.batch_size = cfg.batch; desc.in_channels = cfg.C_in;
        desc.in_height = cfg.H; desc.in_width = cfg.W;
        desc.out_channels = cfg.C_out; desc.kernel_height = cfg.k_h; desc.kernel_width = cfg.k_w;
        desc.stride_h = 1; desc.stride_w = 1;
        desc.padding_h = cfg.k_h / 2; desc.padding_w = cfg.k_w / 2;
        desc.groups = cfg.groups; desc.dtype = DataType::FLOAT32;

        size_t in_size = static_cast<size_t>(cfg.batch) * cfg.C_in * cfg.H * cfg.W;
        uint32_t C_in_per_g = cfg.C_in / cfg.groups;
        size_t w_size = static_cast<size_t>(cfg.C_out) * C_in_per_g * cfg.k_h * cfg.k_w;
        size_t out_size = static_cast<size_t>(cfg.batch) * cfg.C_out * desc.out_height() * desc.out_width();

        std::vector<float> input(in_size), weight(w_size), output(out_size, 0.0f);
        for (auto& v : input) v = dist(rng);
        for (auto& v : weight) v = dist(rng);

        float max_diff = harness.run_conv2d(desc, input.data(), weight.data(), nullptr, output.data());

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "g%u %ux%ux%u",
                      cfg.groups, cfg.C_in, cfg.H, cfg.W);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = TOLERANCE;
        r.passed = (max_diff <= TOLERANCE);
        ComputeHarness::print_result(r);
        results.push_back(r);
    }

    // --- Inverted residual block tests ---
    for (const auto& cfg : ir_configs) {
        uint32_t C_exp = cfg.C_in * cfg.expand_ratio;
        uint32_t H_out = cfg.H / cfg.stride;
        uint32_t W_out = cfg.W / cfg.stride;
        size_t sp_in = static_cast<size_t>(cfg.H) * cfg.W;
        size_t sp_out = static_cast<size_t>(H_out) * W_out;

        // Weights
        size_t exp_w = static_cast<size_t>(C_exp) * cfg.C_in;          // expand 1x1
        size_t dw_w = static_cast<size_t>(C_exp) * 1 * 3 * 3;         // depthwise 3x3
        size_t proj_w = static_cast<size_t>(cfg.C_out) * C_exp;       // project 1x1

        std::vector<float> input(cfg.C_in * sp_in);
        std::vector<float> w_exp(exp_w), w_dw(dw_w), w_proj(proj_w);
        for (auto& v : input) v = dist(rng);
        for (auto& v : w_exp) v = dist(rng);
        for (auto& v : w_dw) v = dist(rng);
        for (auto& v : w_proj) v = dist(rng);

        // Expand 1x1
        Conv2DDescriptor d_exp;
        d_exp.batch_size = 1; d_exp.in_channels = cfg.C_in;
        d_exp.in_height = cfg.H; d_exp.in_width = cfg.W;
        d_exp.out_channels = C_exp; d_exp.kernel_height = 1; d_exp.kernel_width = 1;
        d_exp.stride_h = 1; d_exp.stride_w = 1; d_exp.dtype = DataType::FLOAT32;

        // Depthwise 3x3
        Conv2DDescriptor d_dw;
        d_dw.batch_size = 1; d_dw.in_channels = C_exp;
        d_dw.in_height = cfg.H; d_dw.in_width = cfg.W;
        d_dw.out_channels = C_exp; d_dw.kernel_height = 3; d_dw.kernel_width = 3;
        d_dw.stride_h = cfg.stride; d_dw.stride_w = cfg.stride;
        d_dw.padding_h = 1; d_dw.padding_w = 1;
        d_dw.groups = C_exp; d_dw.dtype = DataType::FLOAT32;

        // Project 1x1
        Conv2DDescriptor d_proj;
        d_proj.batch_size = 1; d_proj.in_channels = C_exp;
        d_proj.in_height = H_out; d_proj.in_width = W_out;
        d_proj.out_channels = cfg.C_out; d_proj.kernel_height = 1; d_proj.kernel_width = 1;
        d_proj.stride_h = 1; d_proj.stride_w = 1; d_proj.dtype = DataType::FLOAT32;

        auto apply_relu6 = [&](std::vector<float>& data) {
            // relu6 = min(relu(x), 6) = relu(x) then min(x, 6)
            // Implement as: relu, then clamp via MUL_SCALAR trick...
            // Actually, just do it manually since fabric doesn't have relu6
            for (auto& v : data) v = relu6(v);
        };

        // Fabric path
        std::vector<float> exp_out(C_exp * sp_in);
        harness.fabric().submit_conv2d(d_exp, input.data(), w_exp.data(), nullptr, exp_out.data());
        harness.fabric().drain();
        apply_relu6(exp_out);

        std::vector<float> dw_out(C_exp * sp_out);
        harness.fabric().submit_conv2d(d_dw, exp_out.data(), w_dw.data(), nullptr, dw_out.data());
        harness.fabric().drain();
        apply_relu6(dw_out);

        std::vector<float> proj_out(cfg.C_out * sp_out);
        harness.fabric().submit_conv2d(d_proj, dw_out.data(), w_proj.data(), nullptr, proj_out.data());
        harness.fabric().drain();

        // Residual add
        if (cfg.has_residual) {
            ElementwiseDescriptor edesc;
            edesc.op = ElementwiseOp::ADD; edesc.count = proj_out.size(); edesc.dtype = DataType::FLOAT32;
            std::vector<float> res_out(proj_out.size());
            harness.fabric().submit_elementwise(edesc, proj_out.data(), input.data(), res_out.data());
            harness.fabric().drain();
            proj_out = std::move(res_out);
        }

        // Reference
        std::vector<float> ref_exp(C_exp * sp_in);
        ComputeHarness::ref_conv2d(d_exp, input.data(), w_exp.data(), nullptr, ref_exp.data());
        for (auto& v : ref_exp) v = relu6(v);

        std::vector<float> ref_dw(C_exp * sp_out);
        ComputeHarness::ref_conv2d(d_dw, ref_exp.data(), w_dw.data(), nullptr, ref_dw.data());
        for (auto& v : ref_dw) v = relu6(v);

        std::vector<float> ref_proj(cfg.C_out * sp_out);
        ComputeHarness::ref_conv2d(d_proj, ref_dw.data(), w_proj.data(), nullptr, ref_proj.data());

        if (cfg.has_residual) {
            for (size_t i = 0; i < ref_proj.size(); ++i) ref_proj[i] += input[i];
        }

        float max_diff = ComputeHarness::max_abs_diff(proj_out.data(), ref_proj.data(), proj_out.size());

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "%u→%u e%u s%u",
                      cfg.C_in, cfg.C_out, cfg.expand_ratio, cfg.stride);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = TOLERANCE;
        r.passed = (max_diff <= TOLERANCE);
        ComputeHarness::print_result(r);
        results.push_back(r);
    }

    harness.print_stats();
    return ComputeHarness::print_summary(results);
}
