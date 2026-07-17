/**
 * @file m3_mobilenet.cpp
 * @brief DNN Milestone M3: MobileNetV2 as a dataflow graph on the CSP executor.
 *
 * The third rung of the DNN milestone ladder (issue #131). The full MobileNetV2
 * (stem + inverted-residual bottleneck stack + 1x1 head conv + global-average-
 * pool + FC) is expressed as a KernelGraph DFG and executed end-to-end on the
 * credit-based CSP value path through GraphCspExecutor, which folds BatchNorm
 * into its conv, lowers pointwise conv via im2col->GEMM and depthwise conv via
 * the pooling-window unfold + per-channel filter reduce, applies ReLU6, and
 * joins the identity residuals.
 *
 * Three-tier definition of done:
 *  - Demonstrate: the whole network runs end-to-end on the CSP executor;
 *    `--dot FILE` emits the KernelGraph for visualization.
 *  - Validate: the classification output is compared elementwise against a
 *    composed whole-network host oracle (conv/depthwise/BN/ReLU6/add/GAP/FC).
 *  - Benchmark: cycles, CSP ops (post-fusion), cyc/op, and DMA/BM/STR stall
 *    breakdown across a small spec sweep.
 *
 * Usage: m3_mobilenet [--dot FILE]
 * Writeup: docs/milestones/M3_mobilenet.md
 */

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/mobilenetv2.hpp>

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

Row run_spec(const std::string& name, const MobileNetV2Spec& spec,
             const std::string& dot_path) {
    sw::kpu::KernelGraph g;
    auto net = build_mobilenetv2(g, spec);
    bool dot_ok = true;
    if (!dot_path.empty()) {
        std::ofstream f(dot_path);
        f << g.to_dot(true);
        dot_ok = static_cast<bool>(f);
    }

    GraphCspExecutor exec;
    auto result = exec.run(g, net.input, net.node_data, /*T*/16);

    Row r;
    r.name = name;
    r.nodes = net.num_nodes;
    r.ops = result.stats.ops;
    r.stats = result.stats;
    r.dot_ok = dot_ok;
    r.pass = result.output.size() == net.oracle.size();
    for (std::size_t i = 0; i < result.output.size() && i < net.oracle.size(); ++i)
        r.max_err = std::max(r.max_err, std::abs(result.output[i] - net.oracle[i]));
    r.pass = r.pass && r.max_err < 5e-3f;
    return r;
}

void print_row(const Row& r) {
    double cyc_per_op = r.ops ? static_cast<double>(r.stats.total_cycles) / static_cast<double>(r.ops) : 0.0;
    std::cout << "  " << std::left << std::setw(24) << r.name << std::right
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
        "DNN Milestone M3: MobileNetV2 as a DFG on the CSP concurrent executor\n"
        "The whole network runs through the credit-based dataflow (pointwise conv\n"
        "im2col->GEMM, depthwise conv via pooling-window unfold, folded BN, ReLU6,\n"
        "inverted-residual adds, global-avg-pool, FC); outputs are validated\n"
        "elementwise against a composed whole-network host oracle.\n"
        "======================================================================\n\n";

    std::vector<Row> rows;

    MobileNetV2Spec base;                              // default fast scaled topology
    rows.push_back(run_spec("mobilenetv2 (base)", base, dot_path));

    MobileNetV2Spec deeper = base;                     // an extra bottleneck stage
    deeper.stages.push_back({3, 32, 2, 1});
    rows.push_back(run_spec("mobilenetv2 (deeper)", deeper, {}));

    MobileNetV2Spec batch32 = base; batch32.batch = 32;
    rows.push_back(run_spec("mobilenetv2 (batch 32)", batch32, {}));

    std::cout << "  " << std::left << std::setw(24) << "configuration" << std::right
              << std::setw(7) << "nodes" << std::setw(6) << "ops"
              << std::setw(11) << "cycles" << std::setw(10) << "cyc/op"
              << std::setw(9) << "dmaStl" << std::setw(9) << "bmStl"
              << std::setw(9) << "strStl" << std::setw(11) << "maxErr"
              << std::setw(7) << "check" << "\n";
    std::cout << "  " << std::string(101, '-') << "\n";

    bool all_pass = true;
    for (const auto& r : rows) { print_row(r); all_pass = all_pass && r.pass; }

    std::cout << "\n  Fusion: BatchNorm folded into each conv; the base network's "
              << rows.front().nodes << " graph nodes\n  execute as "
              << rows.front().ops << " CSP ops (depthwise convs dispatch to the "
                 "pooling-window path).\n";
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
