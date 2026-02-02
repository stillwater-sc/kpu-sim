// ============================================================================
// verification/kernels/class3_multi_branch/verify_residual.cpp
// Class 3 Multi-Branch — Residual add, skip connections, concat verification
//
// Tests multi-branch patterns: residual add (ResNet-style), concat (SqueezeNet
// Fire module), and skip connections with shape transformations.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../common/compute_harness.hpp"

using namespace sw::kpu;
using namespace sw::kpu::verification;

// ============================================================================
// Helper: channel-wise concat (NCHW) — pure data movement, no fabric op
// ============================================================================

static void concat_channels(const float* a, uint32_t a_channels,
                            const float* b, uint32_t b_channels,
                            uint32_t batch, uint32_t H, uint32_t W,
                            float* output) {
    size_t spatial = static_cast<size_t>(H) * W;
    for (uint32_t n = 0; n < batch; ++n) {
        // Copy a's channels
        std::memcpy(output + (n * (a_channels + b_channels)) * spatial,
                    a + (n * a_channels) * spatial,
                    a_channels * spatial * sizeof(float));
        // Copy b's channels
        std::memcpy(output + (n * (a_channels + b_channels) + a_channels) * spatial,
                    b + (n * b_channels) * spatial,
                    b_channels * spatial * sizeof(float));
    }
}

// ============================================================================
// Test 1: Residual Add (ResNet basic block)
// x → conv → relu → conv → + → relu
//  └──────────────────────────┘
// ============================================================================

static TestResult test_residual_add(ComputeHarness& harness, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const uint32_t N = 1, C = 16, H = 8, W = 8;
    size_t spatial = static_cast<size_t>(C) * H * W;
    size_t w_size = static_cast<size_t>(C) * C * 3 * 3;

    std::vector<float> x(N * spatial), w1(w_size), w2(w_size);
    for (auto& v : x) v = dist(rng);
    for (auto& v : w1) v = dist(rng);
    for (auto& v : w2) v = dist(rng);

    // Fabric path: conv1 → relu → conv2 → residual add → relu
    Conv2DDescriptor cdesc;
    cdesc.batch_size = N; cdesc.in_channels = C; cdesc.in_height = H; cdesc.in_width = W;
    cdesc.out_channels = C; cdesc.kernel_height = 3; cdesc.kernel_width = 3;
    cdesc.stride_h = 1; cdesc.stride_w = 1; cdesc.padding_h = 1; cdesc.padding_w = 1;
    cdesc.dtype = DataType::FLOAT32;

    std::vector<float> conv1_out(N * spatial), relu1_out(N * spatial);
    std::vector<float> conv2_out(N * spatial), add_out(N * spatial), relu2_out(N * spatial);

    // conv1
    harness.fabric().submit_conv2d(cdesc, x.data(), w1.data(), nullptr, conv1_out.data());
    harness.fabric().drain();
    // relu
    ElementwiseDescriptor edesc;
    edesc.op = ElementwiseOp::RELU; edesc.count = N * spatial; edesc.dtype = DataType::FLOAT32;
    harness.fabric().submit_elementwise(edesc, conv1_out.data(), nullptr, relu1_out.data());
    harness.fabric().drain();
    // conv2
    harness.fabric().submit_conv2d(cdesc, relu1_out.data(), w2.data(), nullptr, conv2_out.data());
    harness.fabric().drain();
    // residual add
    edesc.op = ElementwiseOp::ADD;
    harness.fabric().submit_elementwise(edesc, conv2_out.data(), x.data(), add_out.data());
    harness.fabric().drain();
    // final relu
    edesc.op = ElementwiseOp::RELU;
    harness.fabric().submit_elementwise(edesc, add_out.data(), nullptr, relu2_out.data());
    harness.fabric().drain();

    // Reference path
    std::vector<float> ref_c1(N * spatial), ref_r1(N * spatial);
    std::vector<float> ref_c2(N * spatial), ref_add(N * spatial), ref_r2(N * spatial);

    ComputeHarness::ref_conv2d(cdesc, x.data(), w1.data(), nullptr, ref_c1.data());
    for (size_t i = 0; i < N * spatial; ++i) ref_r1[i] = std::max(0.0f, ref_c1[i]);
    ComputeHarness::ref_conv2d(cdesc, ref_r1.data(), w2.data(), nullptr, ref_c2.data());
    for (size_t i = 0; i < N * spatial; ++i) ref_add[i] = ref_c2[i] + x[i];
    for (size_t i = 0; i < N * spatial; ++i) ref_r2[i] = std::max(0.0f, ref_add[i]);

    float max_diff = ComputeHarness::max_abs_diff(relu2_out.data(), ref_r2.data(), N * spatial);

    TestResult r;
    r.op_name = "residual_add";
    r.shape = "1x16x8x8";
    r.max_diff = max_diff;
    r.tolerance = 1e-4f;
    r.passed = (max_diff <= r.tolerance);
    return r;
}

// ============================================================================
// Test 2: Channel Concat (SqueezeNet Fire module pattern)
// x → squeeze(1x1) → expand1x1 ─┐
//                   → expand3x3 ─┤→ concat
// ============================================================================

static TestResult test_fire_module(ComputeHarness& harness, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const uint32_t N = 1, C_in = 16, H = 8, W = 8;
    const uint32_t squeeze_ch = 4, expand_ch = 8;  // each branch

    // Weights
    size_t sq_w_size = static_cast<size_t>(squeeze_ch) * C_in * 1 * 1;
    size_t e1_w_size = static_cast<size_t>(expand_ch) * squeeze_ch * 1 * 1;
    size_t e3_w_size = static_cast<size_t>(expand_ch) * squeeze_ch * 3 * 3;

    std::vector<float> x(N * C_in * H * W);
    std::vector<float> sq_w(sq_w_size), e1_w(e1_w_size), e3_w(e3_w_size);
    for (auto& v : x) v = dist(rng);
    for (auto& v : sq_w) v = dist(rng);
    for (auto& v : e1_w) v = dist(rng);
    for (auto& v : e3_w) v = dist(rng);

    // Fabric path
    std::vector<float> sq_out(N * squeeze_ch * H * W);
    std::vector<float> e1_out(N * expand_ch * H * W);
    std::vector<float> e3_out(N * expand_ch * H * W);
    std::vector<float> cat_out(N * 2 * expand_ch * H * W);

    // squeeze 1x1
    Conv2DDescriptor sq_desc;
    sq_desc.batch_size = N; sq_desc.in_channels = C_in; sq_desc.in_height = H; sq_desc.in_width = W;
    sq_desc.out_channels = squeeze_ch; sq_desc.kernel_height = 1; sq_desc.kernel_width = 1;
    sq_desc.stride_h = 1; sq_desc.stride_w = 1; sq_desc.dtype = DataType::FLOAT32;
    harness.fabric().submit_conv2d(sq_desc, x.data(), sq_w.data(), nullptr, sq_out.data());
    harness.fabric().drain();

    // relu after squeeze
    ElementwiseDescriptor edesc;
    edesc.op = ElementwiseOp::RELU; edesc.count = sq_out.size(); edesc.dtype = DataType::FLOAT32;
    std::vector<float> sq_relu(sq_out.size());
    harness.fabric().submit_elementwise(edesc, sq_out.data(), nullptr, sq_relu.data());
    harness.fabric().drain();

    // expand 1x1
    Conv2DDescriptor e1_desc;
    e1_desc.batch_size = N; e1_desc.in_channels = squeeze_ch; e1_desc.in_height = H; e1_desc.in_width = W;
    e1_desc.out_channels = expand_ch; e1_desc.kernel_height = 1; e1_desc.kernel_width = 1;
    e1_desc.stride_h = 1; e1_desc.stride_w = 1; e1_desc.dtype = DataType::FLOAT32;
    harness.fabric().submit_conv2d(e1_desc, sq_relu.data(), e1_w.data(), nullptr, e1_out.data());
    harness.fabric().drain();

    // expand 3x3
    Conv2DDescriptor e3_desc = e1_desc;
    e3_desc.kernel_height = 3; e3_desc.kernel_width = 3;
    e3_desc.padding_h = 1; e3_desc.padding_w = 1;
    harness.fabric().submit_conv2d(e3_desc, sq_relu.data(), e3_w.data(), nullptr, e3_out.data());
    harness.fabric().drain();

    // relu on both branches
    edesc.count = e1_out.size();
    std::vector<float> e1_relu(e1_out.size()), e3_relu(e3_out.size());
    harness.fabric().submit_elementwise(edesc, e1_out.data(), nullptr, e1_relu.data());
    harness.fabric().drain();
    harness.fabric().submit_elementwise(edesc, e3_out.data(), nullptr, e3_relu.data());
    harness.fabric().drain();

    // concat channels
    concat_channels(e1_relu.data(), expand_ch, e3_relu.data(), expand_ch,
                    N, H, W, cat_out.data());

    // Reference path
    std::vector<float> ref_sq(sq_out.size()), ref_sq_r(sq_out.size());
    std::vector<float> ref_e1(e1_out.size()), ref_e3(e3_out.size());
    std::vector<float> ref_e1_r(e1_out.size()), ref_e3_r(e3_out.size());
    std::vector<float> ref_cat(cat_out.size());

    ComputeHarness::ref_conv2d(sq_desc, x.data(), sq_w.data(), nullptr, ref_sq.data());
    for (size_t i = 0; i < ref_sq.size(); ++i) ref_sq_r[i] = std::max(0.0f, ref_sq[i]);
    ComputeHarness::ref_conv2d(e1_desc, ref_sq_r.data(), e1_w.data(), nullptr, ref_e1.data());
    ComputeHarness::ref_conv2d(e3_desc, ref_sq_r.data(), e3_w.data(), nullptr, ref_e3.data());
    for (size_t i = 0; i < ref_e1.size(); ++i) ref_e1_r[i] = std::max(0.0f, ref_e1[i]);
    for (size_t i = 0; i < ref_e3.size(); ++i) ref_e3_r[i] = std::max(0.0f, ref_e3[i]);
    concat_channels(ref_e1_r.data(), expand_ch, ref_e3_r.data(), expand_ch,
                    N, H, W, ref_cat.data());

    float max_diff = ComputeHarness::max_abs_diff(cat_out.data(), ref_cat.data(), cat_out.size());

    TestResult r;
    r.op_name = "fire_module";
    r.shape = "1x16x8x8";
    r.max_diff = max_diff;
    r.tolerance = 1e-4f;
    r.passed = (max_diff <= r.tolerance);
    return r;
}

// ============================================================================
// Test 3: Residual with 1x1 projection (when dimensions change)
// x → conv3x3(stride2) → relu → conv3x3 → + → relu
//  └── conv1x1(stride2) ───────────────────┘
// ============================================================================

static TestResult test_residual_projection(ComputeHarness& harness, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const uint32_t N = 1, C_in = 8, C_out = 16, H = 16, W = 16;
    const uint32_t H_out = 8, W_out = 8;  // stride 2

    std::vector<float> x(N * C_in * H * W);
    for (auto& v : x) v = dist(rng);

    // Main branch weights
    size_t w1_size = static_cast<size_t>(C_out) * C_in * 3 * 3;
    size_t w2_size = static_cast<size_t>(C_out) * C_out * 3 * 3;
    size_t wp_size = static_cast<size_t>(C_out) * C_in * 1 * 1;
    std::vector<float> w1(w1_size), w2(w2_size), wp(wp_size);
    for (auto& v : w1) v = dist(rng);
    for (auto& v : w2) v = dist(rng);
    for (auto& v : wp) v = dist(rng);

    size_t out_spatial = static_cast<size_t>(C_out) * H_out * W_out;

    // Main branch: conv3x3_s2 → relu → conv3x3
    Conv2DDescriptor d1;
    d1.batch_size = N; d1.in_channels = C_in; d1.in_height = H; d1.in_width = W;
    d1.out_channels = C_out; d1.kernel_height = 3; d1.kernel_width = 3;
    d1.stride_h = 2; d1.stride_w = 2; d1.padding_h = 1; d1.padding_w = 1;
    d1.dtype = DataType::FLOAT32;

    Conv2DDescriptor d2;
    d2.batch_size = N; d2.in_channels = C_out; d2.in_height = H_out; d2.in_width = W_out;
    d2.out_channels = C_out; d2.kernel_height = 3; d2.kernel_width = 3;
    d2.stride_h = 1; d2.stride_w = 1; d2.padding_h = 1; d2.padding_w = 1;
    d2.dtype = DataType::FLOAT32;

    // Projection: conv1x1_s2
    Conv2DDescriptor dp;
    dp.batch_size = N; dp.in_channels = C_in; dp.in_height = H; dp.in_width = W;
    dp.out_channels = C_out; dp.kernel_height = 1; dp.kernel_width = 1;
    dp.stride_h = 2; dp.stride_w = 2;
    dp.dtype = DataType::FLOAT32;

    // Fabric path
    std::vector<float> c1(N * out_spatial), r1(N * out_spatial);
    std::vector<float> c2(N * out_spatial), proj(N * out_spatial);
    std::vector<float> add_out(N * out_spatial), relu_out(N * out_spatial);

    harness.fabric().submit_conv2d(d1, x.data(), w1.data(), nullptr, c1.data());
    harness.fabric().drain();
    ElementwiseDescriptor edesc;
    edesc.op = ElementwiseOp::RELU; edesc.count = N * out_spatial; edesc.dtype = DataType::FLOAT32;
    harness.fabric().submit_elementwise(edesc, c1.data(), nullptr, r1.data());
    harness.fabric().drain();
    harness.fabric().submit_conv2d(d2, r1.data(), w2.data(), nullptr, c2.data());
    harness.fabric().drain();
    harness.fabric().submit_conv2d(dp, x.data(), wp.data(), nullptr, proj.data());
    harness.fabric().drain();
    edesc.op = ElementwiseOp::ADD;
    harness.fabric().submit_elementwise(edesc, c2.data(), proj.data(), add_out.data());
    harness.fabric().drain();
    edesc.op = ElementwiseOp::RELU;
    harness.fabric().submit_elementwise(edesc, add_out.data(), nullptr, relu_out.data());
    harness.fabric().drain();

    // Reference
    std::vector<float> rc1(N * out_spatial), rr1(N * out_spatial);
    std::vector<float> rc2(N * out_spatial), rproj(N * out_spatial);
    std::vector<float> radd(N * out_spatial), rrelu(N * out_spatial);

    ComputeHarness::ref_conv2d(d1, x.data(), w1.data(), nullptr, rc1.data());
    for (size_t i = 0; i < N * out_spatial; ++i) rr1[i] = std::max(0.0f, rc1[i]);
    ComputeHarness::ref_conv2d(d2, rr1.data(), w2.data(), nullptr, rc2.data());
    ComputeHarness::ref_conv2d(dp, x.data(), wp.data(), nullptr, rproj.data());
    for (size_t i = 0; i < N * out_spatial; ++i) radd[i] = rc2[i] + rproj[i];
    for (size_t i = 0; i < N * out_spatial; ++i) rrelu[i] = std::max(0.0f, radd[i]);

    float max_diff = ComputeHarness::max_abs_diff(relu_out.data(), rrelu.data(), N * out_spatial);

    TestResult r;
    r.op_name = "resid_proj";
    r.shape = "1x8x16x16→16";
    r.max_diff = max_diff;
    r.tolerance = 1e-4f;
    r.passed = (max_diff <= r.tolerance);
    return r;
}

// ============================================================================
// Test 4: Multi-branch parallel conv (Inception-style)
// x → 1x1 ─────┐
// x → 3x3 ─────┤→ concat
// x → 5x5 ─────┘
// ============================================================================

static TestResult test_inception_branch(ComputeHarness& harness, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const uint32_t N = 1, C_in = 16, H = 8, W = 8;
    const uint32_t ch_1x1 = 8, ch_3x3 = 8, ch_5x5 = 4;

    std::vector<float> x(N * C_in * H * W);
    for (auto& v : x) v = dist(rng);

    auto make_w = [&](size_t s) {
        std::vector<float> w(s);
        for (auto& v : w) v = dist(rng);
        return w;
    };

    auto w1 = make_w(static_cast<size_t>(ch_1x1) * C_in);
    auto w3 = make_w(static_cast<size_t>(ch_3x3) * C_in * 3 * 3);
    auto w5 = make_w(static_cast<size_t>(ch_5x5) * C_in * 5 * 5);

    Conv2DDescriptor d1;
    d1.batch_size = N; d1.in_channels = C_in; d1.in_height = H; d1.in_width = W;
    d1.out_channels = ch_1x1; d1.kernel_height = 1; d1.kernel_width = 1;
    d1.stride_h = 1; d1.stride_w = 1; d1.dtype = DataType::FLOAT32;

    Conv2DDescriptor d3 = d1;
    d3.out_channels = ch_3x3; d3.kernel_height = 3; d3.kernel_width = 3;
    d3.padding_h = 1; d3.padding_w = 1;

    Conv2DDescriptor d5 = d1;
    d5.out_channels = ch_5x5; d5.kernel_height = 5; d5.kernel_width = 5;
    d5.padding_h = 2; d5.padding_w = 2;

    size_t sp = static_cast<size_t>(H) * W;
    std::vector<float> o1(N * ch_1x1 * sp), o3(N * ch_3x3 * sp), o5(N * ch_5x5 * sp);

    // Fabric
    harness.fabric().submit_conv2d(d1, x.data(), w1.data(), nullptr, o1.data());
    harness.fabric().drain();
    harness.fabric().submit_conv2d(d3, x.data(), w3.data(), nullptr, o3.data());
    harness.fabric().drain();
    harness.fabric().submit_conv2d(d5, x.data(), w5.data(), nullptr, o5.data());
    harness.fabric().drain();

    // Concat: ch_1x1 + ch_3x3 + ch_5x5
    uint32_t total_ch = ch_1x1 + ch_3x3 + ch_5x5;
    std::vector<float> cat_12(N * (ch_1x1 + ch_3x3) * sp);
    concat_channels(o1.data(), ch_1x1, o3.data(), ch_3x3, N, H, W, cat_12.data());
    std::vector<float> cat_all(N * total_ch * sp);
    concat_channels(cat_12.data(), ch_1x1 + ch_3x3, o5.data(), ch_5x5, N, H, W, cat_all.data());

    // Reference
    std::vector<float> r1(o1.size()), r3(o3.size()), r5(o5.size());
    ComputeHarness::ref_conv2d(d1, x.data(), w1.data(), nullptr, r1.data());
    ComputeHarness::ref_conv2d(d3, x.data(), w3.data(), nullptr, r3.data());
    ComputeHarness::ref_conv2d(d5, x.data(), w5.data(), nullptr, r5.data());

    std::vector<float> ref_12(cat_12.size()), ref_all(cat_all.size());
    concat_channels(r1.data(), ch_1x1, r3.data(), ch_3x3, N, H, W, ref_12.data());
    concat_channels(ref_12.data(), ch_1x1 + ch_3x3, r5.data(), ch_5x5, N, H, W, ref_all.data());

    float max_diff = ComputeHarness::max_abs_diff(cat_all.data(), ref_all.data(), cat_all.size());

    TestResult r;
    r.op_name = "inception_3br";
    r.shape = "1x16x8x8→20";
    r.max_diff = max_diff;
    r.tolerance = 1e-4f;
    r.passed = (max_diff <= r.tolerance);
    return r;
}

// ============================================================================
// Test 5: Dense connection (DenseNet-style) — concat with growing channels
// ============================================================================

static TestResult test_dense_block(ComputeHarness& harness, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const uint32_t N = 1, H = 8, W = 8;
    const uint32_t init_ch = 8, growth = 4, layers = 3;
    size_t sp = static_cast<size_t>(H) * W;

    // Initial input
    uint32_t cur_ch = init_ch;
    std::vector<float> cur(N * cur_ch * sp);
    for (auto& v : cur) v = dist(rng);

    std::vector<float> ref_cur = cur;

    for (uint32_t l = 0; l < layers; ++l) {
        // 3x3 conv: cur_ch → growth channels
        size_t w_size = static_cast<size_t>(growth) * cur_ch * 3 * 3;
        std::vector<float> w(w_size);
        for (auto& v : w) v = dist(rng);

        Conv2DDescriptor cdesc;
        cdesc.batch_size = N; cdesc.in_channels = cur_ch; cdesc.in_height = H; cdesc.in_width = W;
        cdesc.out_channels = growth; cdesc.kernel_height = 3; cdesc.kernel_width = 3;
        cdesc.stride_h = 1; cdesc.stride_w = 1; cdesc.padding_h = 1; cdesc.padding_w = 1;
        cdesc.dtype = DataType::FLOAT32;

        // Fabric: conv → relu
        std::vector<float> conv_out(N * growth * sp), relu_out(N * growth * sp);
        harness.fabric().submit_conv2d(cdesc, cur.data(), w.data(), nullptr, conv_out.data());
        harness.fabric().drain();
        ElementwiseDescriptor edesc;
        edesc.op = ElementwiseOp::RELU; edesc.count = N * growth * sp; edesc.dtype = DataType::FLOAT32;
        harness.fabric().submit_elementwise(edesc, conv_out.data(), nullptr, relu_out.data());
        harness.fabric().drain();

        // Concat: cur + relu_out
        uint32_t new_ch = cur_ch + growth;
        std::vector<float> new_cur(N * new_ch * sp);
        concat_channels(cur.data(), cur_ch, relu_out.data(), growth, N, H, W, new_cur.data());

        // Reference
        std::vector<float> ref_conv(N * growth * sp), ref_relu(N * growth * sp);
        ComputeHarness::ref_conv2d(cdesc, ref_cur.data(), w.data(), nullptr, ref_conv.data());
        for (size_t i = 0; i < ref_relu.size(); ++i) ref_relu[i] = std::max(0.0f, ref_conv[i]);
        std::vector<float> ref_new(N * new_ch * sp);
        concat_channels(ref_cur.data(), cur_ch, ref_relu.data(), growth, N, H, W, ref_new.data());

        cur = std::move(new_cur);
        ref_cur = std::move(ref_new);
        cur_ch = new_ch;
    }

    float max_diff = ComputeHarness::max_abs_diff(cur.data(), ref_cur.data(), cur.size());

    TestResult r;
    r.op_name = "dense_block";
    r.shape = "1x8x8x8→20";
    r.max_diff = max_diff;
    r.tolerance = 1e-4f;
    r.passed = (max_diff <= r.tolerance);
    return r;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    ComputeHarness harness;
    std::vector<TestResult> results;
    std::mt19937 rng(42);

    ComputeHarness::print_results_header("Class 3: Multi-Branch Verification");

    results.push_back(test_residual_add(harness, rng));
    ComputeHarness::print_result(results.back());

    results.push_back(test_fire_module(harness, rng));
    ComputeHarness::print_result(results.back());

    results.push_back(test_residual_projection(harness, rng));
    ComputeHarness::print_result(results.back());

    results.push_back(test_inception_branch(harness, rng));
    ComputeHarness::print_result(results.back());

    results.push_back(test_dense_block(harness, rng));
    ComputeHarness::print_result(results.back());

    harness.print_stats();
    return ComputeHarness::print_summary(results);
}
