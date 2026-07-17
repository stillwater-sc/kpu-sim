/**
 * @file m3_efficientnet.cpp
 * @brief DNN Milestone M3: EfficientNet-B0 as a dataflow graph on the CSP executor.
 *
 * The EfficientNet half of M3 (issue #131). The full EfficientNet-B0 (stem +
 * MBConv+SE bottleneck stack + 1x1 head conv + global-average-pool + FC) is
 * expressed as a KernelGraph DFG and executed end-to-end on the credit-based CSP
 * value path through GraphCspExecutor. Beyond MobileNetV2 each block adds a
 * squeeze-and-excitation gate (GAP -> FC_reduce -> ReLU -> FC_expand -> sigmoid
 * -> channel-broadcast multiply) and per-stage depthwise kernel sizes (3 or 5).
 *
 * Three-tier definition of done:
 *  - Demonstrate: the whole network runs end-to-end; `--dot FILE` emits the graph.
 *  - Validate: output compared elementwise against a composed whole-network host
 *    oracle (conv/depthwise/BN/ReLU6/SE/add/GAP/FC).
 *  - Benchmark: cycles, CSP ops, cyc/op, and DMA/BM/STR stall breakdown.
 *
 * SiLU/swish is approximated by ReLU6 for the M3 subset (per the design).
 *
 * Usage: m3_efficientnet [--dot FILE]
 * Writeup: docs/milestones/M3_efficientnet.md
 */

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/efficientnet.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sw::kpu::timing::graph;

namespace {

struct Row {
    std::string name;
    std::size_t nodes = 0, ops = 0;
    bool pass = false;
    bool dot_ok = true;
    float max_err = 0.0f;
    RunStats stats;
};

Row run_spec(const std::string& name, const EfficientNetB0Spec& spec,
             const std::string& dot_path) {
    sw::kpu::KernelGraph g;
    auto net = build_efficientnet_b0(g, spec);
    bool dot_ok = true;
    if (!dot_path.empty()) {
        std::ofstream f(dot_path);
        f << g.to_dot(true);
        dot_ok = static_cast<bool>(f);
    }

    GraphCspExecutor exec;
    auto result = exec.run(g, net.input, net.node_data, spec.tile);

    Row r;
    r.name = name;
    r.nodes = net.num_nodes;
    r.ops = result.stats.ops;
    r.stats = result.stats;
    r.dot_ok = dot_ok;
    r.pass = result.output.size() == net.oracle.size();
    bool finite = true;
    for (std::size_t i = 0; i < result.output.size() && i < net.oracle.size(); ++i) {
        const float d = std::abs(result.output[i] - net.oracle[i]);
        if (!std::isfinite(result.output[i]) || !std::isfinite(d)) finite = false;
        r.max_err = std::max(r.max_err, d);
    }
    r.pass = r.pass && finite && r.max_err < 5e-3f;
    return r;
}

void print_row(const Row& r) {
    double cyc_per_op = r.ops ? static_cast<double>(r.stats.total_cycles) / static_cast<double>(r.ops) : 0.0;
    std::cout << "  " << std::left << std::setw(26) << r.name << std::right
              << std::setw(7) << r.nodes
              << std::setw(6) << r.ops
              << std::setw(11) << r.stats.total_cycles
              << std::setw(10) << std::fixed << std::setprecision(0) << cyc_per_op
              << std::setw(9) << r.stats.dma_stalls
              << std::setw(9) << r.stats.bm_stalls
              << std::setw(9) << r.stats.str_stalls
              << std::setw(11) << std::scientific << std::setprecision(1) << r.max_err
              << std::setw(7) << (r.pass ? "PASS" : "FAIL") << "\n"
              << std::defaultfloat;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string dot_path;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dot") == 0 && i + 1 < argc) {
            dot_path = argv[++i];
        } else {
            std::cout << "Usage: " << argv[0] << " [--dot FILE]\n";
            return std::strcmp(argv[i], "--help") == 0 ? 0 : 1;
        }
    }

    std::cout << "\n"
        "======================================================================\n"
        "DNN Milestone M3: EfficientNet-B0 as a DFG on the CSP concurrent executor\n"
        "The whole network runs through the credit-based dataflow (pointwise conv\n"
        "im2col->GEMM, depthwise conv (3x3/5x5) via pooling-window unfold, folded\n"
        "BN, ReLU6, squeeze-and-excitation gate, inverted-residual adds, global-\n"
        "avg-pool, FC); outputs are validated against a whole-network host oracle.\n"
        "======================================================================\n\n";

    std::vector<Row> rows;

    // Compact base (fast), plus a variant adding a 5x5 MBConv stage.
    EfficientNetB0Spec base;
    base.stages = { {1, 16, 1, 1, 3, 16}, {2, 32, 1, 2, 3, 16} };
    rows.push_back(run_spec("efficientnet-b0 (base)", base, dot_path));

    EfficientNetB0Spec k5 = base;
    k5.stages.push_back({2, 32, 1, 1, 5, 16});   // extra 5x5 MBConv stage
    rows.push_back(run_spec("efficientnet-b0 (+5x5)", k5, {}));

    std::cout << "  " << std::left << std::setw(26) << "configuration" << std::right
              << std::setw(7) << "nodes" << std::setw(6) << "ops"
              << std::setw(11) << "cycles" << std::setw(10) << "cyc/op"
              << std::setw(9) << "dmaStl" << std::setw(9) << "bmStl"
              << std::setw(9) << "strStl" << std::setw(11) << "maxErr"
              << std::setw(7) << "check" << "\n";
    std::cout << "  " << std::string(103, '-') << "\n";

    bool all_pass = true;
    for (const auto& r : rows) { print_row(r); all_pass = all_pass && r.pass; }

    std::cout << "\n  Fusion: BatchNorm folded into each conv; the base network's "
              << rows.front().nodes << " graph nodes\n  execute as "
              << rows.front().ops << " CSP ops (depthwise -> pooling-window path;"
                 " SE gate = GAP/FC/sigmoid/broadcast-mul).\n";
    std::cout << "  Validation: " << (all_pass ? "ALL PASS" : "FAILURES PRESENT")
              << " (oracle: composed whole-network host reference, tol 5e-3)\n";
    if (!dot_path.empty()) {
        if (rows.front().dot_ok) {
            std::cout << "  KernelGraph written to " << dot_path << " (Graphviz dot)\n";
        } else {
            std::cerr << "  error: could not write KernelGraph to " << dot_path << "\n";
            all_pass = false;
        }
    }
    std::cout << "\n";
    return all_pass ? 0 : 1;
}
