// ============================================================================
// Fused batched-MLP layer as a single SURE — "fusion in action"
//
//   Y[i,j] = activation( sum_k X[i,k] * W[k,j] + b[j] )
//
// This demo builds the FUSED operator as ONE System of Uniform Recurrence
// Equations (SURE) over a single iteration domain D = {(i,j,k)}, using
// domain_flow's standalone polyhedral/affine primitives (option "B3"):
//   ConstraintSet, IndexSpace, RecurrenceVariable, AffineMap, ScheduleVector.
// domain_flow itself is left unmodified — we use it as the math kernel and own
// the fused-domain construction here.
//
// The point of the demo is to SHOW why fusion is a property of the recurrence
// system, not a label:
//   * the matmul, bias-add, and activation live in ONE domain;
//   * the bias+activation are BOUNDARY recurrences on the terminal face k=K-1;
//   * under the output-stationary schedule tau=(0,0,1) the epilogue rides the
//     last wavefront — the matmul result is consumed in place and never leaves
//     the domain as an intermediate tensor.
//
// See docs/design/fused-mlp-sure.md for the formalism. (issue #46, epic #45)
// ============================================================================

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <string>

#ifdef KPU_HAS_DOMAIN_FLOW

#include <dfa/dfa.hpp>

using namespace sw::dfa;

namespace {

// Extract the uniform dependence vector d from a "reads-from" affine map
// f(p) = p + c  (so the value at p depends on the value at p+c, i.e. d = -c).
std::vector<int> dependence_vector(const AffineMap<int>& reads_from, int dim) {
    // Apply the map to the origin to recover the constant translation c, then d = -c.
    VectorX<int> origin(static_cast<size_t>(dim), 0);
    VectorX<int> c = reads_from.apply(origin);
    std::vector<int> d(dim);
    for (int t = 0; t < dim; ++t) d[t] = -c[t];
    return d;
}

float relu(float x) { return x > 0.0f ? x : 0.0f; }

} // namespace

int main() {
    // ---- problem dimensions (small, single-tile, easy to read) -------------
    const int B = 2;   // batch
    const int K = 3;   // in-features
    const int N = 2;   // out-features

    std::cout << "Fused batched-MLP SURE:  Y = relu(X[" << B << "x" << K
              << "] . W[" << K << "x" << N << "] + b[" << N << "])\n\n";

    // ======================================================================
    // 1) THE FUSED DOMAIN  D = { (i,j,k) : 0<=i<B, 0<=j<N, 0<=k<K }
    //    Bias and activation add NO iteration dimensions (they are pointwise on
    //    the (i,j) output face), so the fused domain IS the matmul domain.
    // ======================================================================
    ConstraintSet<int> cs;
    cs.add(Hyperplane<int>({1, 0, 0}, 0,     ConstraintType::GreaterOrEqual)); // i >= 0
    cs.add(Hyperplane<int>({1, 0, 0}, B - 1, ConstraintType::LessOrEqual));    // i <= B-1
    cs.add(Hyperplane<int>({0, 1, 0}, 0,     ConstraintType::GreaterOrEqual)); // j >= 0
    cs.add(Hyperplane<int>({0, 1, 0}, N - 1, ConstraintType::LessOrEqual));    // j <= N-1
    cs.add(Hyperplane<int>({0, 0, 1}, 0,     ConstraintType::GreaterOrEqual)); // k >= 0
    cs.add(Hyperplane<int>({0, 0, 1}, K - 1, ConstraintType::LessOrEqual));    // k <= K-1

    IndexSpace<int> domain(cs);  // ctor builds the bounding box and enumerates D

    // Use domain_flow's enumerated points if it produced the expected lattice;
    // otherwise fall back to a direct enumeration (and say so), so the demo is
    // robust to enumerator quirks while still exercising the ConstraintSet path.
    std::vector<IndexPoint> points = domain.getPoints();
    const size_t expected = static_cast<size_t>(B) * N * K;
    bool used_dfa_enum = (points.size() == expected);
    if (!used_dfa_enum) {
        points.clear();
        for (int i = 0; i < B; ++i)
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < K; ++k)
                    points.push_back(IndexPoint({i, j, k}));
    }
    std::cout << "[1] Single fused domain D: " << points.size()
              << " index points (" << B << "x" << N << "x" << K << "), built from "
              << cs.getConstraints().size() << " constraints"
              << (used_dfa_enum ? " [enumerated by domain_flow IndexSpace]\n\n"
                                : " [direct enumeration fallback]\n\n");

    // ======================================================================
    // 2) THE RECURRENCE SYSTEM (the SURE). Uniform dependences expressed as
    //    affine "reads-from" maps f(p) = p + c.  d = -c is the dependence vector.
    // ======================================================================
    RecurrenceVariable X("X", 3), W("W", 3), C("C", 3), Y("Y", 3), bias("b", 3);

    // X(i,j,k) = X(i,j-1,k)   -> reuse across output columns j     dep (0,1,0)
    AffineMap<int> readX({{1,0,0},{0,1,0},{0,0,1}}, {0, -1, 0});
    // W(i,j,k) = W(i-1,j,k)   -> reuse across the batch i          dep (1,0,0)
    AffineMap<int> readW({{1,0,0},{0,1,0},{0,0,1}}, {-1, 0, 0});
    // C(i,j,k) = C(i,j,k-1) + X*W   -> accumulation along k        dep (0,0,1)
    AffineMap<int> readC({{1,0,0},{0,1,0},{0,0,1}}, {0, 0, -1});
    X.dependsOn(&X, readX);
    W.dependsOn(&W, readW);
    C.dependsOn(&C, readC);

    auto print_dep = [&](const char* name, const AffineMap<int>& m) {
        std::vector<int> d = dependence_vector(m, 3);
        std::cout << "      " << name << " dependence vector = ("
                  << d[0] << "," << d[1] << "," << d[2] << ")\n";
    };
    std::cout << "[2] Recurrence system over D:\n";
    std::cout << "      X(i,j,k) = X(i,j-1,k)                 (input reuse)\n";
    print_dep("X", readX);
    std::cout << "      W(i,j,k) = W(i-1,j,k)                 (weight reuse)\n";
    print_dep("W", readW);
    std::cout << "      C(i,j,k) = C(i,j,k-1) + X(i,j,k)*W(i,j,k)   (accumulate)\n";
    print_dep("C", readC);
    std::cout << "      EPILOGUE (boundary recurrence on terminal face k=K-1):\n";
    std::cout << "      Y(i,j)   = relu( C(i,j,K-1) + b(j) )  <-- bias+activation, in place\n";
    std::cout << "      (only Y leaves D; the matmul result C never becomes a tensor)\n\n";
    (void)Y; (void)bias;

    // ======================================================================
    // 3) OUTPUT-STATIONARY SCHEDULE  tau = (0,0,1):  time(i,j,k) = k.
    //    Each (i,j) PE accumulates over k in time; the epilogue fires at k=K-1.
    // ======================================================================
    ScheduleVector<int> tau({0, 0, 1});
    std::map<long, std::vector<IndexPoint>> wavefronts;
    for (const auto& p : points) wavefronts[tau.dot(p)].push_back(p);

    std::cout << "[3] Output-stationary schedule tau=(0,0,1) -> wavefronts:\n";
    for (const auto& [t, wf] : wavefronts) {
        std::cout << "      t=" << t << " : ";
        for (const auto& p : wf) std::cout << p;
        if (t == K - 1) std::cout << "  <-- fused epilogue (bias+activation) rides this wavefront";
        std::cout << "\n";
    }
    std::cout << "\n";

    // ======================================================================
    // 4) EXECUTE THE SURE in schedule order and validate vs a direct reference.
    //    This proves the fused recurrence system computes the right answer
    //    while materializing ZERO intermediate tensors (no A=X.W, no Z=A+b).
    // ======================================================================
    // Inputs
    std::vector<std::vector<float>> Xd(B, std::vector<float>(K));
    std::vector<std::vector<float>> Wd(K, std::vector<float>(N));
    std::vector<float> bd(N);
    for (int i = 0; i < B; ++i) for (int k = 0; k < K; ++k) Xd[i][k] = 0.5f * (i + 1) + k;
    for (int k = 0; k < K; ++k) for (int j = 0; j < N; ++j) Wd[k][j] = (k - j) * 0.25f;
    for (int j = 0; j < N; ++j) bd[j] = (j == 0) ? 1.0f : -1.0f;

    // Reference: plain math, no fusion.
    std::vector<std::vector<float>> Yref(B, std::vector<float>(N));
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += Xd[i][k] * Wd[k][j];
            Yref[i][j] = relu(acc + bd[j]);
        }

    // Fused SURE execution: accumulate C(i,j,.) wavefront by wavefront; apply the
    // boundary epilogue exactly when a point reaches the terminal face k=K-1.
    std::vector<std::vector<float>> Cacc(B, std::vector<float>(N, 0.0f));
    std::vector<std::vector<float>> Yout(B, std::vector<float>(N, 0.0f));
    for (const auto& [t, wf] : wavefronts) {
        (void)t;
        for (const auto& p : wf) {
            int i = p[0], j = p[1], k = p[2];
            Cacc[i][j] += Xd[i][k] * Wd[k][j];        // accumulation recurrence
            if (k == K - 1)                            // terminal face -> fused epilogue
                Yout[i][j] = relu(Cacc[i][j] + bd[j]);
        }
    }

    float max_abs_err = 0.0f;
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < N; ++j)
            max_abs_err = std::max(max_abs_err, std::fabs(Yout[i][j] - Yref[i][j]));

    std::cout << "[4] Fused SURE execution vs reference:  max |error| = "
              << max_abs_err << "\n";
    std::cout << "      Y (fused) =";
    for (int i = 0; i < B; ++i) for (int j = 0; j < N; ++j) std::cout << " " << Yout[i][j];
    std::cout << "\n      intermediate tensors materialized: 0 (no A=X.W, no Z=A+b)\n\n";

    const bool ok = max_abs_err < 1e-5f;
    std::cout << (ok ? "RESULT: PASS — fused matmul+bias+activation SURE validated.\n"
                     : "RESULT: FAIL — fused result does not match reference.\n");
    return ok ? 0 : 1;
}

#else  // !KPU_HAS_DOMAIN_FLOW

int main() {
    std::cout << "fused_mlp_sure_demo requires domain_flow (KPU_USE_DOMAIN_FLOW=ON).\n";
    return 0;
}

#endif
