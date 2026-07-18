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
 *  - Benchmark: cycles, CSP ops (post-fusion), cyc/op, DMA/BM/STR stall
 *    breakdown, and movement-fabric utilization (busy/total, tiles moved,
 *    effective DRAM bandwidth) across a small spec sweep.
 *
 * Utilization is the executor's own busy/total ratio
 * (ConcurrentTimingExecutor::get_statistics()), summed per-op through RunStats and
 * reported as Sum(busy)/Sum(total_cycles). Nodes execute sequentially, so these
 * reflect within-op pipeline activity, not cross-branch overlap - a relative
 * metric for comparing configs. See docs/benchmarking/resnet-benchmarking-guide.md.
 *
 * Usage: m2_resnet [--dot FILE] [--occupancy]
 * Writeup: docs/milestones/M2_resnet.md
 */

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/resnet18.hpp>
#include <sw/kpu/timing/tile_tracker.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sw::kpu::timing::graph;
using namespace sw::kpu::timing;   // ConcurrentTimingExecutor, TileTracker, MemoryLevel

namespace {

struct Row {
    std::string name;
    std::size_t nodes = 0, ops = 0;
    bool pass = false;
    bool dot_ok = true;   // false if a requested --dot write failed
    float max_err = 0.0f;
    RunStats stats;
};

Row run_spec(const std::string& name, const ResNet18Spec& spec,
             const std::string& dot_path) {
    sw::kpu::KernelGraph g;
    auto net = build_resnet18(g, spec);
    bool dot_ok = true;
    if (!dot_path.empty()) {
        std::ofstream f(dot_path);
        f << g.to_dot(true);
        dot_ok = static_cast<bool>(f);   // open + write succeeded
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

// Assumed clock for the effective-bandwidth column. The CSP model is unit-clock;
// at 1.0 GHz, GB/s == bytes/cycle, so this only sets the reported unit.
constexpr double kAssumedClockGHz = 1.0;

// Compute-roofline hardware constants (sw::benchmark::HardwareSpec convention):
// a 16x16 systolic array at 2 FLOP/MAC gives 512 FLOP/cycle = 512 GFLOP/s @ 1 GHz;
// external DRAM bandwidth 64 GB/s. Ridge point = 512/64 = 8 FLOP/byte.
constexpr double kPeakFlopsPerCycle = 512.0;   // 16*16*2
constexpr double kPeakGflops        = 512.0;   // at kAssumedClockGHz
constexpr double kExtBwGbs          = 64.0;
constexpr double kRidgeAI           = kPeakGflops / kExtBwGbs;   // 8 FLOP/byte

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

void print_util_row(const Row& r) {
    auto pct = [](double u) { return 100.0 * u; };
    std::cout << "  " << std::left << std::setw(22) << r.name << std::right
              << std::fixed << std::setprecision(1)
              << std::setw(8) << pct(r.stats.dma_utilization())
              << std::setw(8) << pct(r.stats.bm_utilization())
              << std::setw(8) << pct(r.stats.str_utilization())
              << std::setw(9) << r.stats.tiles_loaded
              << std::setw(9) << r.stats.tiles_moved
              << std::setw(9) << r.stats.tiles_fed
              << std::setw(10) << std::setprecision(1) << r.stats.effective_load_bandwidth(kAssumedClockGHz)
              << std::setw(10) << r.stats.effective_store_bandwidth(kAssumedClockGHz)
              << "\n" << std::defaultfloat;
}

void print_compute_row(const Row& r) {
    const double ai = r.stats.arithmetic_intensity();
    std::cout << "  " << std::left << std::setw(22) << r.name << std::right
              << std::fixed
              << std::setw(10) << std::setprecision(2) << r.stats.total_flops() / 1e6   // MFLOP
              << std::setw(10) << std::setprecision(1) << r.stats.achieved_gflops(kAssumedClockGHz)
              << std::setw(10) << std::setprecision(1) << 100.0 * r.stats.compute_efficiency(kPeakFlopsPerCycle)
              << std::setw(10) << std::setprecision(2) << ai
              << std::setw(10) << std::setprecision(1)
              << 100.0 * r.stats.roofline_efficiency(kPeakGflops, kExtBwGbs, kAssumedClockGHz)
              << std::setw(8) << (ai < kRidgeAI ? "mem" : "cmp")
              << "\n" << std::defaultfloat;
}

// Occupancy timeline: run one representative ResNet conv (the stem im2col GEMM)
// cycle-by-cycle and render the L3 | L2 | L1/array tile bands via TileTracker,
// plus the peak simultaneous occupancy per level - i.e. which level saturates.
int run_occupancy() {
    // A small but representative ResNet layer: a 1x1 projection conv (batch 2,
    // 32->16 channels, 4x4), im2col GEMM M=32 x N=16 x K=32, tiled 16^3 - few
    // enough tiles to read the L3->L2->L1->array flow, while still exercising the
    // full credit-managed hierarchy.
    Conv2DGeometry geom;
    geom.N = 2; geom.C_in = 32; geom.H_in = 4; geom.W_in = 4;
    geom.C_out = 16; geom.Kh = 1; geom.Kw = 1;
    geom.stride_h = geom.stride_w = 1; geom.pad_h = geom.pad_w = 0;
    const Size T = 16;

    // Values are irrelevant to occupancy; fill with constants.
    std::vector<float> input(geom.input_elems(), 0.1f);
    std::vector<float> filter(
        static_cast<std::size_t>(geom.C_out) * geom.C_in * geom.Kh * geom.Kw, 0.01f);

    TileTracker tracker;
    std::size_t pk_l3 = 0, pk_l2 = 0, pk_l1 = 0, pk_arr = 0;
    auto observe = [&](const ConcurrentTimingExecutor& e) {
        tracker.observe(e);
        pk_l3  = std::max(pk_l3,  e.tiles_at(MemoryLevel::L3).size());
        pk_l2  = std::max(pk_l2,  e.tiles_at(MemoryLevel::L2).size());
        pk_l1  = std::max(pk_l1,  e.tiles_at(MemoryLevel::L1).size());
        pk_arr = std::max(pk_arr, e.tiles_at(MemoryLevel::COMPUTE).size());
    };

    RunStats stats;
    (void)run_conv2d_fused(input, filter, geom, /*bn*/ nullptr, /*bias*/ {},
                           /*relu*/ false, T, stats, observe);

    const ConcurrentTimingExecutor::Config cap;   // capacities the runner used (defaults)
    std::cout <<
        "\n======================================================================\n"
        "Occupancy timeline - 1x1 projection conv im2col GEMM (M=" << geom.M()
        << " x N=" << geom.Ncols() << " x K=" << geom.K() << ", tile " << T << ")\n"
        "L3 | L2 | L1/array tile bands, one row per occupancy transition ('*' = in\n"
        "array). Credit-managed BUFFERS: tiles arrive, stay resident, return credit\n"
        "- never hit/miss/evict.\n"
        "======================================================================\n\n";
    std::cout << tracker.log() << "\n";
    std::cout << "  peak simultaneous occupancy:  L3 " << pk_l3 << "/" << cap.l3_buffer_count
              << "   L2 " << pk_l2 << "/" << cap.l2_bank_count
              << "   L1 " << pk_l1 << "   array " << pk_arr << "\n"
              << "  (L3/L2 peaks are shown against their buffer/bank credit counts; a\n"
              << "   peak at capacity means that level is the pipeline's binding buffer.)\n"
              << "  total cycles " << stats.total_cycles << ", DMA util "
              << std::fixed << std::setprecision(1) << 100.0 * stats.dma_utilization()
              << "%\n" << std::defaultfloat;
    return 0;
}

// The three benchmark tables (timing+stalls, movement utilization, compute FLOP
// efficiency) for a set of already-run configurations.
void print_all_tables(const std::vector<Row>& rows) {
    std::cout << "  " << std::left << std::setw(22) << "configuration" << std::right
              << std::setw(7) << "nodes" << std::setw(6) << "ops"
              << std::setw(11) << "cycles" << std::setw(10) << "cyc/op"
              << std::setw(9) << "dmaStl" << std::setw(9) << "bmStl"
              << std::setw(9) << "strStl" << std::setw(11) << "maxErr"
              << std::setw(7) << "check" << "\n";
    std::cout << "  " << std::string(97, '-') << "\n";
    for (const auto& r : rows) print_row(r);

    std::cout << "\n  " << std::left << std::setw(22) << "utilization" << std::right
              << std::setw(8) << "dmaU%" << std::setw(8) << "bmU%" << std::setw(8) << "strU%"
              << std::setw(9) << "tilesLd" << std::setw(9) << "tilesMv" << std::setw(9) << "tilesFd"
              << std::setw(10) << "ldGB/s" << std::setw(10) << "stGB/s" << "\n";
    std::cout << "  " << std::string(93, '-') << "\n";
    for (const auto& r : rows) print_util_row(r);
    std::cout << "\n  Utilization = Sum(active)/Sum(total) per mover: directly"
                 " measured cycles a\n  transfer occupied each component (excludes"
                 " stalled + idle), summed over the\n  sequentially executed ops;"
                 " GB/s at "
              << std::fixed << std::setprecision(1) << kAssumedClockGHz
              << " GHz assumed clock. See\n"
                 "  docs/benchmarking/resnet-benchmarking-guide.md.\n"
              << std::defaultfloat;

    std::cout << "\n  " << std::left << std::setw(22) << "compute" << std::right
              << std::setw(10) << "MFLOP" << std::setw(10) << "GFLOP/s"
              << std::setw(10) << "peakEff%" << std::setw(10) << "AI(F/B)"
              << std::setw(10) << "roofEff%" << std::setw(8) << "bound" << "\n";
    std::cout << "  " << std::string(80, '-') << "\n";
    for (const auto& r : rows) print_compute_row(r);
    std::cout << "\n  GEMM FLOPs (conv im2col + FC, 2/MAC) vs a "
              << std::fixed << std::setprecision(0) << kPeakGflops
              << " GFLOP/s peak (16x16 PE @ "
              << std::setprecision(1) << kAssumedClockGHz << " GHz) and "
              << std::setprecision(0) << kExtBwGbs << " GB/s DRAM. peakEff = achieved/peak;\n"
                 "  roofEff = achieved/min(AI*bw, peak); bound = mem below the "
              << std::setprecision(0) << kRidgeAI << " FLOP/byte ridge, else cmp.\n"
              << std::defaultfloat;

    // Concurrency headroom: sequential cycles vs the idealized branch-overlap
    // critical path (upper bound on what concurrent branch scheduling could buy).
    std::cout << "\n  " << std::left << std::setw(22) << "concurrency" << std::right
              << std::setw(11) << "seqCyc" << std::setw(11) << "critCyc"
              << std::setw(9) << "ovlp x" << "\n";
    std::cout << "  " << std::string(51, '-') << "\n";
    for (const auto& r : rows)
        std::cout << "  " << std::left << std::setw(22) << r.name << std::right
                  << std::setw(11) << r.stats.total_cycles
                  << std::setw(11) << r.stats.critical_path_cycles
                  << std::setw(9) << std::fixed << std::setprecision(2) << r.stats.overlap_speedup()
                  << "\n" << std::defaultfloat;
    std::cout << "\n  Nodes execute sequentially today; critCyc is the DAG critical"
                 " path if\n  execution-level-independent branches (e.g. a residual's"
                 " 1x1 projection\n  skip vs. its main path) overlapped with unbounded"
                 " resources. ovlp = seqCyc/\n  critCyc is the resource-free UPPER"
                 " bound on that speedup.\n";
}

// Full-scale offline run: realistic channel growth (64->512) + [2,2,2,2] depth +
// larger spatial. Slow (cycle-by-cycle over many more tiles) - not in the CI
// smoke test, which runs the default no-arg scaled sweep.
int run_full() {
    // Representative-scale (not full 224x224, which is intractable cycle-by-cycle):
    // realistic channel GROWTH (16->128) across four stages with stride-2
    // downsampling + 1x1 projections, at the true [2,2,2,2] depth. The scaled
    // spatial (8x8) keeps a whole forward pass to a few million cycles.
    ResNet18Spec full;
    full.batch = 16;
    full.in_channels = 16;
    full.height = full.width = 8;
    full.stage_channels = {16, 32, 64, 128};
    full.blocks_per_stage = 2;
    full.num_classes = 16;

    std::cout << "\n"
        "======================================================================\n"
        "Representative-scale ResNet-18 (offline) - batch " << full.batch << ", stem "
        << full.in_channels << "ch " << full.height << "x" << full.width
        << ", stages {16,32,64,128} x2\n"
        "([2,2,2,2] with stride-2 downsampling + 1x1 projections), "
        << full.num_classes << " classes. Channel growth + real depth; slow.\n"
        "======================================================================\n\n";

    std::vector<Row> rows{ run_spec("resnet18 (full)", full, {}) };
    print_all_tables(rows);
    std::cout << "\n  Validation: " << (rows.front().pass ? "PASS" : "FAIL")
              << " (oracle: composed whole-network host reference, tol 5e-3)\n\n";
    return rows.front().pass ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string dot_path;
    bool occupancy = false, full = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dot") == 0 && i + 1 < argc) {
            dot_path = argv[++i];
        } else if (std::strcmp(argv[i], "--occupancy") == 0) {
            occupancy = true;
        } else if (std::strcmp(argv[i], "--full") == 0) {
            full = true;
        } else {
            std::cout << "Usage: " << argv[0] << " [--dot FILE | --occupancy | --full]\n";
            return std::strcmp(argv[i], "--help") == 0 ? 0 : 1;
        }
    }

    // --occupancy and --full are focused modes that do not build the whole-network
    // graph the way the default sweep does, so --dot has nothing to emit alongside
    // them; reject the combination rather than silently ignore --dot.
    if (occupancy || full) {
        if (!dot_path.empty()) {
            std::cerr << "error: --" << (occupancy ? "occupancy" : "full")
                      << " and --dot are mutually exclusive\n";
            return 2;
        }
        return occupancy ? run_occupancy() : run_full();
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

    print_all_tables(rows);

    bool all_pass = true;
    for (const auto& r : rows) all_pass = all_pass && r.pass;

    std::cout << "\n  Fusion: BatchNorm folded into conv, ReLU fused as the conv"
                 " epilogue -\n  the base network's " << rows.front().nodes
              << " graph nodes execute as " << rows.front().ops << " CSP ops.\n";
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
