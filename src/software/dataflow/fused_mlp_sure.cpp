#include <sw/kpu/dataflow/fused_mlp_sure.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef KPU_HAS_DOMAIN_FLOW
#include <dfa/dfa.hpp>
#endif

namespace sw::kpu::dataflow {

const char* to_string(Activation a) {
    switch (a) {
        case Activation::Identity: return "identity";
        case Activation::ReLU:     return "relu";
        case Activation::GELU:     return "gelu";
        case Activation::SiLU:     return "silu";
        case Activation::Sigmoid:  return "sigmoid";
    }
    return "unknown";
}

float apply_activation(Activation a, float x) {
    switch (a) {
        case Activation::Identity: return x;
        case Activation::ReLU:     return x > 0.0f ? x : 0.0f;
        case Activation::Sigmoid:  return 1.0f / (1.0f + std::exp(-x));
        case Activation::SiLU:     return x / (1.0f + std::exp(-x));
        case Activation::GELU: {
            // tanh approximation of GELU
            constexpr float kSqrt2OverPi = 0.7978845608028654f;
            const float u = kSqrt2OverPi * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + std::tanh(u));
        }
    }
    return x;
}

namespace {

// Direct (fallback) enumeration of the box domain, used when domain_flow is
// unavailable or its IndexSpace did not yield the expected lattice.
std::vector<Index3> enumerate_box(std::size_t B, std::size_t N, std::size_t K) {
    std::vector<Index3> pts;
    pts.reserve(B * N * K);
    for (int i = 0; i < static_cast<int>(B); ++i)
        for (int j = 0; j < static_cast<int>(N); ++j)
            for (int k = 0; k < static_cast<int>(K); ++k)
                pts.push_back(Index3{i, j, k});
    return pts;
}

} // namespace

FusedMlpSure::FusedMlpSure(const FusedMlpSureConfig& config) : config_(config) {
    const std::size_t B = config_.batch;
    const std::size_t K = config_.in_features;
    const std::size_t N = config_.out_features;

    // Reject degenerate / oversized shapes before deriving the domain. With
    // K == 0 there is no terminal accumulation face (the epilogue cannot fire);
    // a 0 in any dimension is an empty operator, not a valid fused MLP layer.
    // Bounding each dimension by INT_MAX keeps the (i,j,k) index coordinates and
    // the constraint right-hand sides within the int domain used by the polyhedron.
    if (B == 0 || K == 0 || N == 0) {
        throw std::invalid_argument(
            "FusedMlpSure: batch, in_features, and out_features must all be >= 1");
    }
    constexpr std::size_t kMaxDim = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (B > kMaxDim || K > kMaxDim || N > kMaxDim) {
        throw std::invalid_argument(
            "FusedMlpSure: a dimension exceeds the supported range (INT_MAX)");
    }

    const std::size_t expected = B * N * K;
    constraint_count_ = 6;

#ifdef KPU_HAS_DOMAIN_FLOW
    // Build the fused domain D = {(i,j,k)} as a single polyhedron via
    // domain_flow's ConstraintSet -> IndexSpace (the math kernel). Bias and
    // activation add no iteration dimensions, so D is exactly the matmul domain.
    if (B && K && N) {
        using namespace sw::dfa;
        ConstraintSet<int> cs;
        cs.add(Hyperplane<int>({1, 0, 0}, 0, ConstraintType::GreaterOrEqual));
        cs.add(Hyperplane<int>({1, 0, 0}, static_cast<int>(B) - 1, ConstraintType::LessOrEqual));
        cs.add(Hyperplane<int>({0, 1, 0}, 0, ConstraintType::GreaterOrEqual));
        cs.add(Hyperplane<int>({0, 1, 0}, static_cast<int>(N) - 1, ConstraintType::LessOrEqual));
        cs.add(Hyperplane<int>({0, 0, 1}, 0, ConstraintType::GreaterOrEqual));
        cs.add(Hyperplane<int>({0, 0, 1}, static_cast<int>(K) - 1, ConstraintType::LessOrEqual));

        IndexSpace<int> domain(cs);  // ctor builds bounding box and enumerates
        const auto& dfa_points = domain.getPoints();
        if (dfa_points.size() == expected) {
            points_.reserve(expected);
            for (const auto& p : dfa_points) {
                points_.push_back(Index3{p[0], p[1], p[2]});
            }
            dfa_enumerated_ = true;
        }
    }
#endif

    if (!dfa_enumerated_) {
        points_ = enumerate_box(B, N, K);
    }
}

long FusedMlpSure::time_of(const Index3& p) const noexcept {
    const auto tau = schedule_tau();  // (0,0,1)
    return static_cast<long>(tau[0]) * p.i +
           static_cast<long>(tau[1]) * p.j +
           static_cast<long>(tau[2]) * p.k;
}

std::map<long, std::vector<Index3>> FusedMlpSure::wavefronts() const {
    std::map<long, std::vector<Index3>> wf;
    for (const auto& p : points_) {
        wf[time_of(p)].push_back(p);
    }
    return wf;
}

std::vector<float> FusedMlpSure::evaluate(const std::vector<float>& X,
                                          const std::vector<float>& W,
                                          const std::vector<float>& bias) const {
    const std::size_t B = config_.batch;
    const std::size_t K = config_.in_features;
    const std::size_t N = config_.out_features;
    if (X.size() != B * K) throw std::invalid_argument("FusedMlpSure::evaluate: X must be [B,K]");
    if (W.size() != K * N) throw std::invalid_argument("FusedMlpSure::evaluate: W must be [K,N]");
    if (bias.size() != N)  throw std::invalid_argument("FusedMlpSure::evaluate: bias must be [N]");

    const int term = terminal_k();
    std::vector<float> acc(B * N, 0.0f);  // C(i,j,.) accumulator, one per output (output-stationary)
    std::vector<float> Y(B * N, 0.0f);

    // Execute the fused recurrence system in schedule (wavefront) order. The
    // accumulation runs along k; the epilogue (bias + activation) fires exactly
    // when a point reaches the terminal face k=K-1 — in place, no intermediate.
    for (const auto& [t, wf] : wavefronts()) {
        (void)t;
        for (const auto& p : wf) {
            const std::size_t out = static_cast<std::size_t>(p.i) * N + static_cast<std::size_t>(p.j);
            acc[out] += X[static_cast<std::size_t>(p.i) * K + static_cast<std::size_t>(p.k)] *
                        W[static_cast<std::size_t>(p.k) * N + static_cast<std::size_t>(p.j)];
            if (p.k == term) {
                Y[out] = apply_activation(config_.activation,
                                          acc[out] + bias[static_cast<std::size_t>(p.j)]);
            }
        }
    }
    return Y;
}

std::string FusedMlpSure::describe() const {
    const std::size_t B = config_.batch;
    const std::size_t K = config_.in_features;
    const std::size_t N = config_.out_features;

    std::ostringstream os;
    os << "Fused batched-MLP SURE:  Y = " << to_string(config_.activation)
       << "( X[" << B << "x" << K << "] . W[" << K << "x" << N << "] + b[" << N << "] )\n\n";

    os << "[1] Single fused domain D: " << domain_size() << " index points ("
       << B << "x" << N << "x" << K << "), from " << constraint_count_ << " constraints"
       << (dfa_enumerated_ ? " [enumerated by domain_flow IndexSpace]\n\n"
                           : " [direct enumeration]\n\n");

    os << "[2] Recurrence system over D:\n";
#ifdef KPU_HAS_DOMAIN_FLOW
    // Declare the SURE with domain_flow's RecurrenceVariable + AffineMap and
    // recover the dependence vectors from the affine "reads-from" maps.
    {
        using namespace sw::dfa;
        RecurrenceVariable X("X", 3), W("W", 3), C("C", 3);
        AffineMap<int> readX({{1,0,0},{0,1,0},{0,0,1}}, {0, -1, 0});  // X(i,j,k)=X(i,j-1,k)
        AffineMap<int> readW({{1,0,0},{0,1,0},{0,0,1}}, {-1, 0, 0});  // W(i,j,k)=W(i-1,j,k)
        AffineMap<int> readC({{1,0,0},{0,1,0},{0,0,1}}, {0, 0, -1});  // C(i,j,k)=C(i,j,k-1)+X*W
        X.dependsOn(&X, readX);
        W.dependsOn(&W, readW);
        C.dependsOn(&C, readC);
        auto dep = [](const AffineMap<int>& m) {
            VectorX<int> c = m.apply(VectorX<int>(static_cast<size_t>(3), 0));
            return Index3{-c[0], -c[1], -c[2]};
        };
        const Index3 dx = dep(readX), dw = dep(readW), dc = dep(readC);
        os << "      X(i,j,k) = X(i,j-1,k)   dep (" << dx.i << "," << dx.j << "," << dx.k << ")  input reuse\n";
        os << "      W(i,j,k) = W(i-1,j,k)   dep (" << dw.i << "," << dw.j << "," << dw.k << ")  weight reuse\n";
        os << "      C(i,j,k) = C(i,j,k-1)+X*W  dep (" << dc.i << "," << dc.j << "," << dc.k << ")  accumulate\n";
    }
#else
    os << "      X(i,j,k) = X(i,j-1,k)   dep (0,1,0)  input reuse\n";
    os << "      W(i,j,k) = W(i-1,j,k)   dep (1,0,0)  weight reuse\n";
    os << "      C(i,j,k) = C(i,j,k-1)+X*W  dep (0,0,1)  accumulate\n";
#endif
    os << "      EPILOGUE (boundary recurrence on terminal face k=" << terminal_k() << "):\n";
    os << "      Y(i,j) = " << to_string(config_.activation)
       << "( C(i,j,K-1) + b(j) )   <-- bias+activation in place; only Y leaves D\n\n";

    os << "[3] Output-stationary schedule tau=(0,0,1) -> wavefronts:\n";
    for (const auto& [t, wf] : wavefronts()) {
        os << "      t=" << t << " : ";
        for (const auto& p : wf) os << "(" << p.i << "," << p.j << "," << p.k << ") ";
        if (t == terminal_k()) os << " <-- fused epilogue rides this wavefront";
        os << "\n";
    }
    return os.str();
}

} // namespace sw::kpu::dataflow
