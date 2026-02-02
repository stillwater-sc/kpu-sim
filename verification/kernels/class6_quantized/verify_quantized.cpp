// ============================================================================
// verification/kernels/class6_quantized/verify_quantized.cpp
// Class 6 Quantized Inference — INT8/INT4 matmul, quantize/dequantize,
// mixed-precision pipeline verification.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../common/compute_harness.hpp"

using namespace sw::kpu;
using namespace sw::kpu::verification;

// ============================================================================
// Quantization helpers
// ============================================================================

struct QuantParams {
    float scale;
    int32_t zero_point;
};

// Symmetric quantization: q = clamp(round(x / scale), qmin, qmax)
static QuantParams compute_symmetric_params(const float* data, size_t n, int bits) {
    float absmax = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        absmax = std::max(absmax, std::abs(data[i]));
    }
    if (absmax == 0.0f) absmax = 1.0f;
    int qmax = (1 << (bits - 1)) - 1;  // 127 for INT8, 7 for INT4
    float scale = absmax / qmax;
    return { scale, 0 };
}

static void quantize_symmetric(const float* input, int8_t* output, size_t n,
                                float scale, int qmin, int qmax) {
    for (size_t i = 0; i < n; ++i) {
        int q = static_cast<int>(std::round(input[i] / scale));
        q = std::max(qmin, std::min(qmax, q));
        output[i] = static_cast<int8_t>(q);
    }
}

static void dequantize_symmetric(const int32_t* input, float* output, size_t n,
                                  float scale_a, float scale_b) {
    // INT32 accumulator result: dequantized = acc * scale_a * scale_b
    float combined_scale = scale_a * scale_b;
    for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<float>(input[i]) * combined_scale;
    }
}

// Reference INT8 matmul with INT32 accumulator
static void ref_int8_matmul(uint32_t M, uint32_t N, uint32_t K,
                            const int8_t* A, const int8_t* B, int32_t* C) {
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (uint32_t p = 0; p < K; ++p) {
                sum += static_cast<int32_t>(A[i * K + p]) * static_cast<int32_t>(B[p * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Test 1: INT8 matmul — direct integer verification
// ============================================================================

struct INT8MatmulConfig {
    const char* name;
    uint32_t M, N, K;
};

static const INT8MatmulConfig int8_configs[] = {
    { "int8_16x16x16",   16,  16,  16 },
    { "int8_32x32x32",   32,  32,  32 },
    { "int8_128x128x128",128, 128, 128 },
    { "int8_1x128x10",    1, 128,  10 },
    { "int8_32x64x128",  32,  64, 128 },
    { "int8_7x13x5",      7,  13,   5 },
};

static void run_int8_matmul_tests(ComputeHarness& harness, std::mt19937& rng,
                                  std::vector<TestResult>& results) {
    std::uniform_int_distribution<int> dist(-50, 50);

    for (const auto& cfg : int8_configs) {
        size_t sizeA = static_cast<size_t>(cfg.M) * cfg.K;
        size_t sizeB = static_cast<size_t>(cfg.K) * cfg.N;
        size_t sizeC = static_cast<size_t>(cfg.M) * cfg.N;

        std::vector<int8_t> A(sizeA), B(sizeB);
        std::vector<int32_t> C(sizeC, 0), ref_C(sizeC, 0);
        for (size_t i = 0; i < sizeA; ++i) A[i] = static_cast<int8_t>(dist(rng));
        for (size_t i = 0; i < sizeB; ++i) B[i] = static_cast<int8_t>(dist(rng));

        MatMulDescriptor desc;
        desc.m = cfg.M; desc.n = cfg.N; desc.k = cfg.K;
        desc.dtype = DataType::INT8;

        harness.fabric().submit_matmul(desc, A.data(), B.data(), C.data());
        harness.fabric().drain();

        ref_int8_matmul(cfg.M, cfg.N, cfg.K, A.data(), B.data(), ref_C.data());

        // Check exact match for integer computation
        int32_t max_diff = 0;
        for (size_t i = 0; i < sizeC; ++i) {
            int32_t d = std::abs(C[i] - ref_C[i]);
            if (d > max_diff) max_diff = d;
        }

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "(%u,%u,%u)", cfg.M, cfg.N, cfg.K);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = static_cast<float>(max_diff);
        r.tolerance = 0.0f;  // Exact for integers
        r.passed = (max_diff == 0);
        ComputeHarness::print_result(r);
        results.push_back(r);
    }
}

// ============================================================================
// Test 2: Quantize → INT8 matmul → Dequantize pipeline
// Verify end-to-end: FP32 input → quantize → INT8 matmul → dequantize → FP32
// Compare against direct FP32 matmul within quantization error bounds.
// ============================================================================

struct QDQConfig {
    const char* name;
    uint32_t M, N, K;
    float expected_max_error;  // Depends on scale and K
};

static const QDQConfig qdq_configs[] = {
    { "qdq_16x16x16",   16,  16,  16, 0.1f  },
    { "qdq_32x32x32",   32,  32,  32, 0.2f  },
    { "qdq_32x64x128",  32,  64, 128, 0.5f  },
    { "qdq_1x128x10",    1, 128,  10, 0.3f  },
};

static void run_qdq_pipeline_tests(ComputeHarness& harness, std::mt19937& rng,
                                   std::vector<TestResult>& results) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const auto& cfg : qdq_configs) {
        size_t sizeA = static_cast<size_t>(cfg.M) * cfg.K;
        size_t sizeB = static_cast<size_t>(cfg.K) * cfg.N;
        size_t sizeC = static_cast<size_t>(cfg.M) * cfg.N;

        // Generate FP32 data
        std::vector<float> A_fp(sizeA), B_fp(sizeB);
        for (auto& v : A_fp) v = dist(rng);
        for (auto& v : B_fp) v = dist(rng);

        // FP32 reference
        std::vector<float> ref_fp(sizeC);
        ComputeHarness::ref_matmul(cfg.M, cfg.N, cfg.K, A_fp.data(), B_fp.data(), ref_fp.data());

        // Quantize
        auto params_a = compute_symmetric_params(A_fp.data(), sizeA, 8);
        auto params_b = compute_symmetric_params(B_fp.data(), sizeB, 8);

        std::vector<int8_t> A_q(sizeA), B_q(sizeB);
        quantize_symmetric(A_fp.data(), A_q.data(), sizeA, params_a.scale, -128, 127);
        quantize_symmetric(B_fp.data(), B_q.data(), sizeB, params_b.scale, -128, 127);

        // INT8 matmul through fabric
        std::vector<int32_t> C_int(sizeC, 0);
        MatMulDescriptor desc;
        desc.m = cfg.M; desc.n = cfg.N; desc.k = cfg.K;
        desc.dtype = DataType::INT8;
        harness.fabric().submit_matmul(desc, A_q.data(), B_q.data(), C_int.data());
        harness.fabric().drain();

        // Dequantize
        std::vector<float> C_fp(sizeC);
        dequantize_symmetric(C_int.data(), C_fp.data(), sizeC, params_a.scale, params_b.scale);

        // Compare: quantization error is bounded by scale_a * scale_b * K
        float max_diff = ComputeHarness::max_abs_diff(C_fp.data(), ref_fp.data(), sizeC);

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "(%u,%u,%u)", cfg.M, cfg.N, cfg.K);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = cfg.expected_max_error;
        r.passed = (max_diff <= cfg.expected_max_error);
        ComputeHarness::print_result(r);
        results.push_back(r);
    }
}

// ============================================================================
// Test 3: Quantize/Dequantize round-trip error
// ============================================================================

struct RoundTripConfig {
    const char* name;
    size_t count;
    int bits;
    float range;
};

static const RoundTripConfig rt_configs[] = {
    { "rt_int8_small",   64,  8, 1.0f },
    { "rt_int8_large",  1024, 8, 2.0f },
    { "rt_int4_small",   64,  4, 1.0f },
    { "rt_int4_large",  1024, 4, 2.0f },
};

static void run_roundtrip_tests(std::mt19937& rng,
                                std::vector<TestResult>& results) {
    for (const auto& cfg : rt_configs) {
        std::uniform_real_distribution<float> dist(-cfg.range, cfg.range);

        std::vector<float> original(cfg.count);
        for (auto& v : original) v = dist(rng);

        auto params = compute_symmetric_params(original.data(), cfg.count, cfg.bits);
        int qmax = (1 << (cfg.bits - 1)) - 1;
        int qmin = -(1 << (cfg.bits - 1));

        std::vector<int8_t> quantized(cfg.count);
        quantize_symmetric(original.data(), quantized.data(), cfg.count,
                          params.scale, qmin, qmax);

        // Dequantize back
        std::vector<float> reconstructed(cfg.count);
        for (size_t i = 0; i < cfg.count; ++i) {
            reconstructed[i] = static_cast<float>(quantized[i]) * params.scale;
        }

        float max_diff = ComputeHarness::max_abs_diff(original.data(), reconstructed.data(), cfg.count);
        // Round-trip error should be bounded by scale / 2
        float expected_max = params.scale / 2.0f + 1e-7f;

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "n=%zu %dbit", cfg.count, cfg.bits);

        TestResult r;
        r.op_name = cfg.name;
        r.shape = shape_buf;
        r.max_diff = max_diff;
        r.tolerance = expected_max;
        r.passed = (max_diff <= expected_max);
        ComputeHarness::print_result(r);
        results.push_back(r);
    }
}

// ============================================================================
// Test 4: INT8 elementwise (ADD, MUL)
// ============================================================================

static void run_int8_elementwise_tests(ComputeHarness& harness, std::mt19937& rng,
                                       std::vector<TestResult>& results) {
    std::uniform_int_distribution<int> dist(-50, 50);

    struct { const char* name; ElementwiseOp op; } ops[] = {
        { "int8_add", ElementwiseOp::ADD },
        { "int8_mul", ElementwiseOp::MUL },
    };

    size_t counts[] = { 64, 1024 };

    for (const auto& op : ops) {
        for (size_t count : counts) {
            std::vector<int8_t> a(count), b(count), out(count, 0);
            std::vector<int8_t> ref(count);

            for (size_t i = 0; i < count; ++i) {
                a[i] = static_cast<int8_t>(dist(rng));
                b[i] = static_cast<int8_t>(dist(rng));
            }

            ElementwiseDescriptor desc;
            desc.op = op.op;
            desc.count = count;
            desc.dtype = DataType::INT8;
            harness.fabric().submit_elementwise(desc, a.data(), b.data(), out.data());
            harness.fabric().drain();

            // Reference: match fabric behavior (Scalar * Scalar with native truncation)
            for (size_t i = 0; i < count; ++i) {
                if (op.op == ElementwiseOp::ADD) {
                    ref[i] = static_cast<int8_t>(static_cast<int8_t>(a[i]) + static_cast<int8_t>(b[i]));
                } else {
                    // Fabric does: C[i] = A[i] * B[i] as int8_t, which truncates
                    ref[i] = static_cast<int8_t>(static_cast<int>(a[i]) * static_cast<int>(b[i]));
                }
            }

            int max_diff = 0;
            for (size_t i = 0; i < count; ++i) {
                int d = std::abs(static_cast<int>(out[i]) - static_cast<int>(ref[i]));
                if (d > max_diff) max_diff = d;
            }

            char shape_buf[64];
            std::snprintf(shape_buf, sizeof(shape_buf), "n=%zu", count);

            TestResult r;
            r.op_name = op.name;
            r.shape = shape_buf;
            r.max_diff = static_cast<float>(max_diff);
            r.tolerance = 0.0f;
            r.passed = (max_diff == 0);
            ComputeHarness::print_result(r);
            results.push_back(r);
        }
    }
}

// ============================================================================
// Test 5: Quantized MLP (FP32 → Q → INT8 matmul → DQ → ReLU → Q → INT8 matmul → DQ)
// End-to-end two-layer quantized MLP vs FP32 baseline.
// ============================================================================

static TestResult test_quantized_mlp(ComputeHarness& harness, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const uint32_t batch = 4, D_in = 32, D_hidden = 64, D_out = 10;

    // FP32 weights and input
    std::vector<float> x(batch * D_in);
    std::vector<float> w1(D_in * D_hidden), w2(D_hidden * D_out);
    for (auto& v : x) v = dist(rng);
    for (auto& v : w1) v = dist(rng);
    for (auto& v : w2) v = dist(rng);

    // FP32 reference: x → matmul(w1) → relu → matmul(w2)
    std::vector<float> h1(batch * D_hidden), h1_relu(batch * D_hidden);
    std::vector<float> fp_out(batch * D_out);
    ComputeHarness::ref_matmul(batch, D_hidden, D_in, x.data(), w1.data(), h1.data());
    for (size_t i = 0; i < h1.size(); ++i) h1_relu[i] = std::max(0.0f, h1[i]);
    ComputeHarness::ref_matmul(batch, D_out, D_hidden, h1_relu.data(), w2.data(), fp_out.data());

    // Quantized path
    auto qp_x = compute_symmetric_params(x.data(), x.size(), 8);
    auto qp_w1 = compute_symmetric_params(w1.data(), w1.size(), 8);

    std::vector<int8_t> x_q(x.size()), w1_q(w1.size());
    quantize_symmetric(x.data(), x_q.data(), x.size(), qp_x.scale, -128, 127);
    quantize_symmetric(w1.data(), w1_q.data(), w1.size(), qp_w1.scale, -128, 127);

    // Layer 1: INT8 matmul
    std::vector<int32_t> h1_int(batch * D_hidden, 0);
    MatMulDescriptor md1;
    md1.m = batch; md1.n = D_hidden; md1.k = D_in; md1.dtype = DataType::INT8;
    harness.fabric().submit_matmul(md1, x_q.data(), w1_q.data(), h1_int.data());
    harness.fabric().drain();

    // Dequantize → ReLU → Requantize
    std::vector<float> h1_dq(batch * D_hidden);
    dequantize_symmetric(h1_int.data(), h1_dq.data(), h1_dq.size(), qp_x.scale, qp_w1.scale);
    for (auto& v : h1_dq) v = std::max(0.0f, v);

    auto qp_h1 = compute_symmetric_params(h1_dq.data(), h1_dq.size(), 8);
    auto qp_w2 = compute_symmetric_params(w2.data(), w2.size(), 8);

    std::vector<int8_t> h1_rq(h1_dq.size()), w2_q(w2.size());
    quantize_symmetric(h1_dq.data(), h1_rq.data(), h1_dq.size(), qp_h1.scale, -128, 127);
    quantize_symmetric(w2.data(), w2_q.data(), w2.size(), qp_w2.scale, -128, 127);

    // Layer 2: INT8 matmul
    std::vector<int32_t> out_int(batch * D_out, 0);
    MatMulDescriptor md2;
    md2.m = batch; md2.n = D_out; md2.k = D_hidden; md2.dtype = DataType::INT8;
    harness.fabric().submit_matmul(md2, h1_rq.data(), w2_q.data(), out_int.data());
    harness.fabric().drain();

    // Final dequantize
    std::vector<float> q_out(batch * D_out);
    dequantize_symmetric(out_int.data(), q_out.data(), q_out.size(), qp_h1.scale, qp_w2.scale);

    float max_diff = ComputeHarness::max_abs_diff(q_out.data(), fp_out.data(), q_out.size());

    TestResult r;
    r.op_name = "quant_mlp";
    r.shape = "4x32→64→10";
    r.max_diff = max_diff;
    r.tolerance = 1.0f;  // Quantization introduces significant error through 2 layers
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

    ComputeHarness::print_results_header("Class 6: Quantized Inference Verification");

    run_int8_matmul_tests(harness, rng, results);
    run_qdq_pipeline_tests(harness, rng, results);
    run_roundtrip_tests(rng, results);
    run_int8_elementwise_tests(harness, rng, results);

    auto mlp_r = test_quantized_mlp(harness, rng);
    ComputeHarness::print_result(mlp_r);
    results.push_back(mlp_r);

    harness.print_stats();
    return ComputeHarness::print_summary(results);
}
