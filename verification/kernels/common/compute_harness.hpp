// ============================================================================
// verification/kernels/common/compute_harness.hpp
// RAII harness for behavioral compute fabric kernel verification
//
// Follows the lpddr5_harness.hpp pattern - wraps BehavioralComputeFabric
// with convenience methods for reference computation and comparison.
// ============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <sw/kpu/models/behavioral/compute/compute_fabric.hpp>
#include <sw/kpu/models/interfaces/compute_fabric_interface.hpp>
#include <sw/kpu/fidelity/component_config.hpp>

namespace sw::kpu::verification {

// ============================================================================
// Structured violation tracking
// ============================================================================

struct Violation {
    std::string test_name;
    std::string message;
    float max_diff = 0.0f;
    float tolerance = 0.0f;
};

// ============================================================================
// Test result tracking
// ============================================================================

struct TestResult {
    std::string op_name;
    std::string shape;
    bool passed = false;
    float max_diff = 0.0f;
    float tolerance = 0.0f;
};

// ============================================================================
// ComputeHarness
// ============================================================================

class ComputeHarness {
public:
    explicit ComputeHarness(uint32_t macs_per_cycle = 256)
        : fabric_(make_config(macs_per_cycle), /*tile_id=*/0) {}

    // ========================================================================
    // Fabric access
    // ========================================================================

    BehavioralComputeFabric& fabric() { return fabric_; }
    const BehavioralComputeFabric& fabric() const { return fabric_; }

    // ========================================================================
    // Convenience: run matmul through fabric, return max abs diff vs reference
    // ========================================================================

    float run_matmul(uint32_t M, uint32_t N, uint32_t K,
                     const float* A, const float* B, float* C) {
        MatMulDescriptor desc;
        desc.m = M;
        desc.n = N;
        desc.k = K;
        desc.dtype = DataType::FLOAT32;

        fabric_.submit_matmul(desc, A, B, C);
        fabric_.drain();

        // Compute reference
        std::vector<float> ref(M * N);
        ref_matmul(M, N, K, A, B, ref.data());

        return max_abs_diff(C, ref.data(), M * N);
    }

    // ========================================================================
    // Convenience: run elementwise through fabric, return max abs diff
    // ========================================================================

    float run_elementwise(ElementwiseOp op, size_t count,
                          const float* a, const float* b, float* out) {
        ElementwiseDescriptor desc;
        desc.op = op;
        desc.count = static_cast<uint64_t>(count);
        desc.dtype = DataType::FLOAT32;

        fabric_.submit_elementwise(desc, a, b, out);
        fabric_.drain();

        // Compute reference
        std::vector<float> ref(count);
        ref_elementwise(op, count, a, b, ref.data());

        return max_abs_diff(out, ref.data(), count);
    }

    // ========================================================================
    // Reference implementations
    // ========================================================================

    static void ref_matmul(uint32_t M, uint32_t N, uint32_t K,
                           const float* A, const float* B, float* C) {
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (uint32_t p = 0; p < K; ++p) {
                    sum += A[i * K + p] * B[p * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    static void ref_elementwise(ElementwiseOp op, size_t count,
                                const float* a, const float* b, float* out) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = ref_elementwise_scalar(op, a[i], b ? b[i] : 0.0f);
        }
    }

    static float ref_elementwise_scalar(ElementwiseOp op, float a, float b) {
        switch (op) {
            case ElementwiseOp::RELU:    return std::max(0.0f, a);
            case ElementwiseOp::GELU: {
                // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                constexpr float sqrt_2_over_pi = 0.7978845608028654f;
                float x3 = a * a * a;
                return a * 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (a + 0.044715f * x3)));
            }
            case ElementwiseOp::SIGMOID: return 1.0f / (1.0f + std::exp(-a));
            case ElementwiseOp::TANH:    return std::tanh(a);
            case ElementwiseOp::NEG:     return -a;
            case ElementwiseOp::ABS:     return std::abs(a);
            case ElementwiseOp::SQRT:    return std::sqrt(a);
            case ElementwiseOp::EXP:     return std::exp(a);
            case ElementwiseOp::LOG:     return std::log(a);
            case ElementwiseOp::SILU:    return a / (1.0f + std::exp(-a));
            case ElementwiseOp::ADD:     return a + b;
            case ElementwiseOp::MUL:     return a * b;
            default:                     return 0.0f;
        }
    }

    // ========================================================================
    // Convenience: run conv2d through fabric, return max abs diff vs reference
    // ========================================================================

    float run_conv2d(const Conv2DDescriptor& desc,
                     const float* input, const float* weight,
                     const float* bias, float* output) {
        fabric_.submit_conv2d(desc, input, weight, bias, output);
        fabric_.drain();

        uint32_t H_out = desc.out_height();
        uint32_t W_out = desc.out_width();
        size_t out_size = static_cast<size_t>(desc.batch_size) * desc.out_channels * H_out * W_out;

        std::vector<float> ref(out_size);
        ref_conv2d(desc, input, weight, bias, ref.data());
        return max_abs_diff(output, ref.data(), out_size);
    }

    // ========================================================================
    // Convenience: run pool2d through fabric, return max abs diff vs reference
    // ========================================================================

    float run_pool2d(const Pool2DDescriptor& desc,
                     const float* input, float* output) {
        fabric_.submit_pool2d(desc, input, output);
        fabric_.drain();

        uint32_t H_out = desc.out_height();
        uint32_t W_out = desc.out_width();
        size_t out_size = static_cast<size_t>(desc.batch_size) * desc.channels * H_out * W_out;

        std::vector<float> ref(out_size);
        ref_pool2d(desc, input, ref.data());
        return max_abs_diff(output, ref.data(), out_size);
    }

    // ========================================================================
    // Reference: Conv2D (NCHW layout, groups support)
    // ========================================================================

    static void ref_conv2d(const Conv2DDescriptor& desc,
                           const float* input, const float* weight,
                           const float* bias, float* output) {
        const uint32_t N = desc.batch_size;
        const uint32_t C_in = desc.in_channels;
        const uint32_t H_in = desc.in_height;
        const uint32_t W_in = desc.in_width;
        const uint32_t C_out = desc.out_channels;
        const uint32_t K_h = desc.kernel_height;
        const uint32_t K_w = desc.kernel_width;
        const uint32_t H_out = desc.out_height();
        const uint32_t W_out = desc.out_width();
        const uint32_t groups = desc.groups;
        const uint32_t C_in_per_group = C_in / groups;
        const uint32_t C_out_per_group = C_out / groups;

        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t g = 0; g < groups; ++g) {
                for (uint32_t co = 0; co < C_out_per_group; ++co) {
                    uint32_t oc = g * C_out_per_group + co;
                    for (uint32_t h = 0; h < H_out; ++h) {
                        for (uint32_t w = 0; w < W_out; ++w) {
                            float sum = 0.0f;
                            for (uint32_t ci = 0; ci < C_in_per_group; ++ci) {
                                uint32_t ic = g * C_in_per_group + ci;
                                for (uint32_t kh = 0; kh < K_h; ++kh) {
                                    for (uint32_t kw = 0; kw < K_w; ++kw) {
                                        int32_t ih = static_cast<int32_t>(h * desc.stride_h + kh * desc.dilation_h)
                                                     - static_cast<int32_t>(desc.padding_h);
                                        int32_t iw = static_cast<int32_t>(w * desc.stride_w + kw * desc.dilation_w)
                                                     - static_cast<int32_t>(desc.padding_w);
                                        if (ih >= 0 && ih < static_cast<int32_t>(H_in) &&
                                            iw >= 0 && iw < static_cast<int32_t>(W_in)) {
                                            uint64_t in_idx = ((static_cast<uint64_t>(n) * C_in + ic) * H_in + ih) * W_in + iw;
                                            uint64_t w_idx = ((static_cast<uint64_t>(oc) * C_in_per_group + ci) * K_h + kh) * K_w + kw;
                                            sum += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                            if (bias) sum += bias[oc];
                            uint64_t out_idx = ((static_cast<uint64_t>(n) * C_out + oc) * H_out + h) * W_out + w;
                            output[out_idx] = sum;
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Reference: Pool2D (max, avg, adaptive_avg)
    // ========================================================================

    static void ref_pool2d(const Pool2DDescriptor& desc,
                           const float* input, float* output) {
        const uint32_t N = desc.batch_size;
        const uint32_t C = desc.channels;
        const uint32_t H_in = desc.in_height;
        const uint32_t W_in = desc.in_width;
        const uint32_t H_out = desc.out_height();
        const uint32_t W_out = desc.out_width();

        if (desc.pool_type == Pool2DDescriptor::PoolType::ADAPTIVE_AVG) {
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t c = 0; c < C; ++c) {
                    for (uint32_t ho = 0; ho < H_out; ++ho) {
                        for (uint32_t wo = 0; wo < W_out; ++wo) {
                            uint32_t h_start = (ho * H_in) / H_out;
                            uint32_t h_end = ((ho + 1) * H_in) / H_out;
                            uint32_t w_start = (wo * W_in) / W_out;
                            uint32_t w_end = ((wo + 1) * W_in) / W_out;
                            float sum = 0.0f;
                            uint32_t cnt = 0;
                            for (uint32_t h = h_start; h < h_end; ++h) {
                                for (uint32_t w = w_start; w < w_end; ++w) {
                                    sum += input[((static_cast<uint64_t>(n) * C + c) * H_in + h) * W_in + w];
                                    ++cnt;
                                }
                            }
                            output[((static_cast<uint64_t>(n) * C + c) * H_out + ho) * W_out + wo] = sum / static_cast<float>(cnt);
                        }
                    }
                }
            }
        } else {
            const uint32_t K_h = desc.kernel_height;
            const uint32_t K_w = desc.kernel_width;
            const uint32_t stride_h = desc.stride_h > 0 ? desc.stride_h : K_h;
            const uint32_t stride_w = desc.stride_w > 0 ? desc.stride_w : K_w;

            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t c = 0; c < C; ++c) {
                    for (uint32_t ho = 0; ho < H_out; ++ho) {
                        for (uint32_t wo = 0; wo < W_out; ++wo) {
                            float result;
                            if (desc.pool_type == Pool2DDescriptor::PoolType::MAX) {
                                result = -std::numeric_limits<float>::infinity();
                            } else {
                                result = 0.0f;
                            }
                            uint32_t cnt = 0;
                            for (uint32_t kh = 0; kh < K_h; ++kh) {
                                for (uint32_t kw = 0; kw < K_w; ++kw) {
                                    int32_t ih = static_cast<int32_t>(ho * stride_h + kh) - static_cast<int32_t>(desc.padding_h);
                                    int32_t iw = static_cast<int32_t>(wo * stride_w + kw) - static_cast<int32_t>(desc.padding_w);
                                    if (ih >= 0 && ih < static_cast<int32_t>(H_in) &&
                                        iw >= 0 && iw < static_cast<int32_t>(W_in)) {
                                        float val = input[((static_cast<uint64_t>(n) * C + c) * H_in + ih) * W_in + iw];
                                        if (desc.pool_type == Pool2DDescriptor::PoolType::MAX) {
                                            result = std::max(result, val);
                                        } else {
                                            result += val;
                                        }
                                        ++cnt;
                                    }
                                }
                            }
                            if (desc.pool_type == Pool2DDescriptor::PoolType::AVG && cnt > 0) {
                                result /= static_cast<float>(cnt);
                            }
                            output[((static_cast<uint64_t>(n) * C + c) * H_out + ho) * W_out + wo] = result;
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Comparison utilities
    // ========================================================================

    static float max_abs_diff(const float* a, const float* b, size_t n) {
        float max_d = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float d = std::abs(a[i] - b[i]);
            if (d > max_d) max_d = d;
        }
        return max_d;
    }

    // ========================================================================
    // Violation tracking
    // ========================================================================

    void add_violation(const std::string& test_name, const std::string& msg,
                       float max_diff, float tolerance) {
        violations_.push_back({test_name, msg, max_diff, tolerance});
    }

    bool verify_no_violations() const { return violations_.empty(); }

    const std::vector<Violation>& violations() const { return violations_; }

    // ========================================================================
    // Statistics
    // ========================================================================

    void print_stats() const {
        const auto& s = fabric_.stats();
        std::printf("  Matmuls: %lu  Elementwise: %lu  Total MACs: %lu\n",
                    static_cast<unsigned long>(s.matmuls),
                    static_cast<unsigned long>(s.elementwise_ops),
                    static_cast<unsigned long>(s.total_macs));
    }

    // ========================================================================
    // Results table printing
    // ========================================================================

    static void print_results_header(const char* title) {
        std::printf("\n=== %s ===\n", title);
        std::printf("%-14s %-18s %-8s %s\n", "Op", "Shape", "Status", "Max Diff");
        std::printf("%-14s %-18s %-8s %s\n", "──────────────", "──────────────────",
                    "────────", "────────────");
    }

    static void print_result(const TestResult& r) {
        std::printf("%-14s %-18s %-8s %.2e\n",
                    r.op_name.c_str(), r.shape.c_str(),
                    r.passed ? "PASS" : "FAIL", r.max_diff);
    }

    static int print_summary(const std::vector<TestResult>& results) {
        int pass = 0, fail = 0;
        for (const auto& r : results) {
            if (r.passed) ++pass; else ++fail;
        }
        std::printf("\nTotal: %d  PASS: %d  FAIL: %d\n",
                    pass + fail, pass, fail);
        return fail > 0 ? 1 : 0;
    }

private:
    BehavioralComputeFabric fabric_;
    std::vector<Violation> violations_;

    static ComputeFabricConfig make_config(uint32_t macs_per_cycle) {
        ComputeFabricConfig cfg;
        cfg.fidelity = SimulationFidelity::BEHAVIORAL;
        cfg.macs_per_cycle = macs_per_cycle;
        cfg.technology = ComputeTechnology::IDEAL;
        return cfg;
    }
};

} // namespace sw::kpu::verification
