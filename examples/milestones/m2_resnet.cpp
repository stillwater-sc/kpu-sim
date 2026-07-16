/**
 * @file m2_resnet.cpp
 * @brief DNN Milestone M2: ResNet-18 as a dataflow graph on the CSP executor.
 *
 * The second rung of the DNN milestone ladder (issue #130). The full ResNet-18
 * (stem + four stages of BasicBlocks + global-average-pool + FC) is expressed as
 * a KernelGraph DFG and executed end-to-end on the credit-based CSP value path
 * through GraphCspExecutor, which folds BatchNorm into its conv, fuses the ReLU
 * epilogue, lowers conv via im2col, and joins the residual branches.
 *
 * Three-tier definition of done:
 *  - Demonstrate: the whole network runs end-to-end on the CSP executor;
 *    `--dot FILE` emits the KernelGraph for visualization.
 *  - Validate: the classification output is compared elementwise against a
 *    composed whole-network host oracle (conv/BN/ReLU/add/GAP/FC references).
 *  - Benchmark: cycles, CSP ops (post-fusion), cyc/op, and DMA/BM/STR stall
 *    breakdown across a small spec sweep.
 *
 * Usage: m2_resnet [--dot FILE]
 * Writeup: docs/milestones/M2_resnet.md
 */

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/resnet18.hpp>

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
    float max_err = 0.0f;
    RunStats stats;
};

Row run_spec(const std::string& name, const ResNet18Spec& spec,
             const std::string& dot_path) {
    sw::kpu::KernelGraph g;
    auto net = build_resnet18(g, spec);
    if (!dot_path.empty()) {
        std::ofstream f(dot_path);
        if (f) f << g.to_dot(true);
    }

    GraphCspExecutor exec;
    auto result = exec.run(g, net.input, net.node_data, /*T*/16);

    Row r;
    r.name = name;
    r.nodes = net.num_nodes;
    r.ops = result.stats.ops;
    r.stats = result.stats;
    r.pass = result.output.size() == net.oracle.size();
    for (std::size_t i = 0; i < result.output.size() && i < net.oracle.size(); ++i)
        r.max_err = std::max(r.max_err, std::abs(result.output[i] - net.oracle[i]));
    r.pass = r.pass && r.max_err < 5e-3f;
    return r;
}

void print_row(const Row& r) {
    double cyc_per_op = r.ops ? static_cast<double>(r.stats.total_cycles) / static_cast<double>(r.ops) : 0.0;
    std::cout << "  " << std::left << std::setw(22) << r.name << std::right
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
        "DNN Milestone M2: ResNet-18 as a DFG on the CSP concurrent executor\n"
        "The whole network runs through the credit-based dataflow (conv im2col->\n"
        "GEMM, folded BN, fused ReLU epilogue, residual adds, global-avg-pool,\n"
        "FC); outputs are validated elementwise against a host oracle.\n"
        "======================================================================\n\n";

    std::vector<Row> rows;

    // Default scaled demo, plus two sweeps: wider batch, and deeper channels.
    ResNet18Spec base;                                   // batch 16, 16ch 8x8
    rows.push_back(run_spec("resnet18 (base)", base, dot_path));

    ResNet18Spec deeper = base; deeper.blocks_per_stage = 2;   // [2,2,2,2]
    rows.push_back(run_spec("resnet18 [2,2,2,2]", deeper, {}));

    ResNet18Spec batch32 = base; batch32.batch = 32;
    rows.push_back(run_spec("resnet18 (batch 32)", batch32, {}));

    std::cout << "  " << std::left << std::setw(22) << "configuration" << std::right
              << std::setw(7) << "nodes" << std::setw(6) << "ops"
              << std::setw(11) << "cycles" << std::setw(10) << "cyc/op"
              << std::setw(9) << "dmaStl" << std::setw(9) << "bmStl"
              << std::setw(9) << "strStl" << std::setw(11) << "maxErr"
              << std::setw(7) << "check" << "\n";
    std::cout << "  " << std::string(97, '-') << "\n";

    bool all_pass = true;
    for (const auto& r : rows) { print_row(r); all_pass = all_pass && r.pass; }

    std::cout << "\n  Fusion: BatchNorm folded into conv, ReLU fused as the conv"
                 " epilogue -\n  the base network's " << rows.front().nodes
              << " graph nodes execute as " << rows.front().ops << " CSP ops.\n";
    std::cout << "  Validation: " << (all_pass ? "ALL PASS" : "FAILURES PRESENT")
              << " (oracle: composed whole-network host reference, tol 5e-3)\n";
    if (!dot_path.empty())
        std::cout << "  KernelGraph written to " << dot_path << " (Graphviz dot)\n";
    std::cout << "\n";
    return all_pass ? 0 : 1;
}
