// ============================================================================
// Fused batched-MLP layer as a single SURE — "fusion in action".
//
//   Y[i,j] = activation( sum_k X[i,k] * W[k,j] + b[j] )
//
// Thin client of the reusable FusedMlpSure library API
// (include/sw/kpu/dataflow/fused_mlp_sure.hpp). The library models the fused
// matmul+bias+activation operator as ONE SURE over a single (i,j,k) domain,
// with the bias+activation as boundary recurrences on the terminal face k=K-1
// (built on domain_flow's polyhedral primitives; see docs/design/fused-mlp-sure.md).
//
// This demo prints the fused domain / recurrences / schedule and validates the
// fused execution against a direct reference. (issue #46, epic #45)
// ============================================================================

#include <sw/kpu/dataflow/fused_mlp_sure.hpp>

#include <cmath>
#include <iostream>
#include <vector>

using sw::kpu::dataflow::Activation;
using sw::kpu::dataflow::FusedMlpSure;
using sw::kpu::dataflow::FusedMlpSureConfig;
using sw::kpu::dataflow::apply_activation;

int main() {
    FusedMlpSureConfig cfg;
    cfg.batch = 2;          // B
    cfg.in_features = 3;    // K
    cfg.out_features = 2;   // N
    cfg.activation = Activation::ReLU;

    FusedMlpSure sure(cfg);

    // [1]-[3]: the fused domain, recurrence system, and wavefront schedule.
    std::cout << sure.describe() << "\n";

    // ---- inputs (deterministic, easy to read) ------------------------------
    const std::size_t B = cfg.batch, K = cfg.in_features, N = cfg.out_features;
    std::vector<float> X(B * K), W(K * N), b(N);
    for (std::size_t i = 0; i < B; ++i)
        for (std::size_t k = 0; k < K; ++k) X[i * K + k] = 0.5f * (i + 1) + k;
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) W[k * N + j] = (static_cast<float>(k) - j) * 0.25f;
    for (std::size_t j = 0; j < N; ++j) b[j] = (j == 0) ? 1.0f : -1.0f;

    // ---- reference (plain math, no fusion) ---------------------------------
    std::vector<float> Yref(B * N, 0.0f);
    for (std::size_t i = 0; i < B; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) acc += X[i * K + k] * W[k * N + j];
            Yref[i * N + j] = apply_activation(cfg.activation, acc + b[j]);
        }

    // ---- fused SURE execution (library) ------------------------------------
    std::vector<float> Y = sure.evaluate(X, W, b);

    float max_abs_err = 0.0f;
    for (std::size_t t = 0; t < Y.size(); ++t)
        max_abs_err = std::max(max_abs_err, std::fabs(Y[t] - Yref[t]));

    std::cout << "[4] Fused SURE execution vs reference:  max |error| = " << max_abs_err << "\n";
    std::cout << "      Y (fused) =";
    for (float v : Y) std::cout << " " << v;
    std::cout << "\n      intermediate tensors materialized: 0 (no A=X.W, no Z=A+b)\n\n";

    const bool ok = max_abs_err < 1e-5f;
    std::cout << (ok ? "RESULT: PASS — fused matmul+bias+activation SURE validated.\n"
                     : "RESULT: FAIL — fused result does not match reference.\n");
    return ok ? 0 : 1;
}
