#pragma once

// ============================================================================
// FusedMlpSure — the fused batched-MLP operator as a single SURE.
//
//   Y[i,j] = activation( sum_k X[i,k] * W[k,j] + b[j] )
//
// Models the fused matmul+bias+activation operator as ONE System of Uniform
// Recurrence Equations over a single iteration domain D = {(i,j,k)}: the
// matmul accumulates along k, and the bias+activation are BOUNDARY recurrences
// on the terminal face k=K-1, so the matmul result is consumed in place and
// never materializes as an intermediate tensor.
//
// This is an OWNERSHIP / OBSERVATION + lowering-input structure, not a dataflow
// control API. Internally it uses domain_flow's standalone polyhedral/affine
// primitives as the math kernel when available (KPU_HAS_DOMAIN_FLOW), with a
// direct fallback otherwise; the public API below is domain_flow-free so
// consumers (the demo, the #47 tiled-DMProgram lowering) need not depend on it.
//
// See docs/design/fused-mlp-sure.md. (issue #46, epic #45)
// ============================================================================

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace sw::kpu::dataflow {

/// Output-face activation applied by the fused epilogue.
enum class Activation { Identity, ReLU, GELU, SiLU, Sigmoid };

/// Human-readable name, e.g. for `.dfg` attributes / debug.
const char* to_string(Activation a);

/// Apply a single activation value (used by the epilogue and reference checks).
float apply_activation(Activation a, float x);

/// Problem shape + epilogue kind for a fused MLP layer.
struct FusedMlpSureConfig {
    std::size_t batch = 1;         ///< B — rows of X / Y
    std::size_t in_features = 1;   ///< K — reduction (accumulation) length
    std::size_t out_features = 1;  ///< N — columns of W / Y
    Activation activation = Activation::ReLU;
};

/// A point (i,j,k) in the fused iteration domain D.
struct Index3 {
    int i = 0;
    int j = 0;
    int k = 0;
};

/// The fused matmul+bias+activation operator as a single SURE.
class FusedMlpSure {
public:
    explicit FusedMlpSure(const FusedMlpSureConfig& config);

    const FusedMlpSureConfig& config() const noexcept { return config_; }

    // --- Iteration domain D = {(i,j,k): 0<=i<B, 0<=j<N, 0<=k<K} -------------
    const std::vector<Index3>& domain_points() const noexcept { return points_; }
    std::size_t domain_size() const noexcept { return points_.size(); }
    /// True if the domain was enumerated by domain_flow's IndexSpace (vs the
    /// direct fallback used when domain_flow is unavailable).
    bool enumerated_by_domain_flow() const noexcept { return dfa_enumerated_; }
    /// Number of polyhedral constraints used to define D (2 per axis = 6).
    std::size_t constraint_count() const noexcept { return constraint_count_; }

    // --- SURE structure ------------------------------------------------------
    /// Uniform dependence vectors of the recurrence system.
    static constexpr Index3 x_reuse_vector()      { return {0, 1, 0}; } ///< X reused across columns
    static constexpr Index3 w_reuse_vector()      { return {1, 0, 0}; } ///< W reused across batch
    static constexpr Index3 c_accumulate_vector() { return {0, 0, 1}; } ///< C accumulates along k
    /// Terminal accumulation face k = K-1 where the fused epilogue fires.
    int terminal_k() const noexcept { return static_cast<int>(config_.in_features) - 1; }
    /// True if the bias+activation epilogue is applied at this point (k == K-1).
    bool is_epilogue_point(const Index3& p) const noexcept { return p.k == terminal_k(); }

    // --- Output-stationary schedule tau = (0,0,1): time(i,j,k) = k ----------
    std::array<int, 3> schedule_tau() const noexcept { return {0, 0, 1}; }
    long time_of(const Index3& p) const noexcept;
    /// Points grouped by schedule time (wavefronts); the epilogue rides t=K-1.
    std::map<long, std::vector<Index3>> wavefronts() const;

    // --- Behavioral execution of the fused SURE ------------------------------
    /// Execute the fused recurrence system. `X` row-major [B,K], `W` row-major
    /// [K,N], `bias` [N]; returns `Y` row-major [B,N]. No intermediate tensors
    /// are materialized — the epilogue rides the accumulation's terminal face.
    std::vector<float> evaluate(const std::vector<float>& X,
                                const std::vector<float>& W,
                                const std::vector<float>& bias) const;

    // --- Human-readable description (domain, recurrences, wavefronts) --------
    std::string describe() const;

private:
    FusedMlpSureConfig config_;
    std::vector<Index3> points_;
    bool dfa_enumerated_ = false;
    std::size_t constraint_count_ = 0;
};

} // namespace sw::kpu::dataflow
