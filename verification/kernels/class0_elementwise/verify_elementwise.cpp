// ============================================================================
// verification/kernels/class0_elementwise/verify_elementwise.cpp
// Class 0 Elementwise kernel verification — 12 ops × 4 shapes
//
// Standalone verification program for BehavioralComputeFabric elementwise ops.
// Compares fabric output against inline C++ reference implementations.
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
// Test configuration
// ============================================================================

struct OpConfig {
    ElementwiseOp op;
    const char* name;
    bool is_unary;
    bool is_exact;      // true for ops that should match exactly (no transcendentals)
    float tolerance;
    float input_min;    // input range for valid domain
    float input_max;
};

static const OpConfig ops[] = {
    { ElementwiseOp::RELU,    "relu",    true,  true,  0.0f,   -2.0f, 2.0f },
    { ElementwiseOp::GELU,    "gelu",    true,  false, 1e-6f,  -3.0f, 3.0f },
    { ElementwiseOp::SIGMOID, "sigmoid", true,  false, 1e-6f,  -5.0f, 5.0f },
    { ElementwiseOp::TANH,    "tanh",    true,  false, 1e-6f,  -3.0f, 3.0f },
    { ElementwiseOp::NEG,     "neg",     true,  true,  0.0f,   -2.0f, 2.0f },
    { ElementwiseOp::ABS,     "abs",     true,  true,  0.0f,   -2.0f, 2.0f },
    { ElementwiseOp::SQRT,    "sqrt",    true,  false, 1e-6f,   0.01f, 10.0f },
    { ElementwiseOp::EXP,     "exp",     true,  false, 1e-5f,  -3.0f, 3.0f },
    { ElementwiseOp::LOG,     "log",     true,  false, 1e-6f,   0.01f, 10.0f },
    { ElementwiseOp::SILU,    "silu",    true,  false, 1e-6f,  -3.0f, 3.0f },
    { ElementwiseOp::ADD,     "add",     false, true,  0.0f,   -2.0f, 2.0f },
    { ElementwiseOp::MUL,     "mul",     false, false, 1e-7f,  -2.0f, 2.0f },
};

struct ShapeConfig {
    std::vector<size_t> dims;
    std::string name;

    size_t count() const {
        size_t c = 1;
        for (auto d : dims) c *= d;
        return c;
    }
};

static std::vector<ShapeConfig> get_shapes() {
    return {
        { {32},            "32" },
        { {32, 128},       "32x128" },
        { {4, 32, 128},    "4x32x128" },
        { {1, 3, 64, 64},  "1x3x64x64" },
    };
}

// ============================================================================
// Main
// ============================================================================

int main() {
    ComputeHarness harness;
    std::vector<TestResult> results;
    std::mt19937 rng(42);

    auto shapes = get_shapes();

    ComputeHarness::print_results_header("Class 0: Elementwise Verification");

    for (const auto& op : ops) {
        for (const auto& shape : shapes) {
            size_t count = shape.count();

            // Generate input data in valid domain
            std::uniform_real_distribution<float> dist(op.input_min, op.input_max);
            std::vector<float> a(count), b(count), out(count, 0.0f);

            for (size_t i = 0; i < count; ++i) a[i] = dist(rng);
            if (!op.is_unary) {
                for (size_t i = 0; i < count; ++i) b[i] = dist(rng);
            }

            float max_diff = harness.run_elementwise(
                op.op, count, a.data(),
                op.is_unary ? nullptr : b.data(),
                out.data());

            TestResult r;
            r.op_name = op.name;
            r.shape = shape.name;
            r.max_diff = max_diff;
            r.tolerance = op.tolerance;
            r.passed = (max_diff <= op.tolerance);

            if (!r.passed) {
                harness.add_violation(
                    std::string(op.name) + "/" + shape.name,
                    "max_diff exceeds tolerance",
                    max_diff, op.tolerance);
            }

            ComputeHarness::print_result(r);
            results.push_back(r);
        }
    }

    harness.print_stats();
    return ComputeHarness::print_summary(results);
}
