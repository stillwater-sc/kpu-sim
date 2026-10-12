// ============================================================================
// examples/characterize/tile_characterize.cpp
// Design-of-experiments harness for characterizing L0 TilePrograms.
//
// Sweeps (algorithm x size x tile-shape x compute-tiles x topology), for each cell:
//   1. takes the cell's L0 -- the trace of kpu-csp-gen's program for the operator, as L-B
//      records it -- so you can SEE the tile sequence (--disasm/--trace),
//   2. runs the L0 on the platform and VALIDATES its tiles against L-B's values,
//   3. characterizes structural + first-order modeled performance/energy metrics,
// and emits a table (+ CSV/JSON). The compute-tiles sweep is the domain-flow analogue
// of CUDA's occupancy-vs-resources question: how does achievable concurrency (and
// thus makespan/energy) scale with the hardware you give the program?
//
//   tile_characterize --algo lu --sizes 64,128,256 --tiles 16,32
//       --compute-tiles 1,4,16,64 --topology single,checkerboard
//       --csv out.csv --trace first.json --disasm
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_check.hpp>
#include <sw/kpu/program/characterize/characterization.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/stream/derive/matmul_streams.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/virtual_platform.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::characterize;
namespace csp = sw::kpu::program::csp;
namespace driver = sw::kpu::program::driver;

namespace {

// Argument helpers, the program/device mapping and the operand fills all come from
// sw/kpu/program/driver/program_spec.hpp, shared with kpu-run (#285 §D1). Two copies
// of "what --tiles means" drift, and then a bug reproduces in one tool and not the
// other.
using sw::kpu::program::driver::arg;
using sw::kpu::program::driver::has_flag;
using sw::kpu::program::driver::parse_ints;
using sw::kpu::program::driver::parse_strs;

// ---- the cell's program, and its validation --------------------------------
// EACH CELL'S L0 IS A TRACE: kpu-csp-gen writes the operator's CSP program for the cell's L3,
// and L-B runs it and records the trace (docs/plans/retire-l0-derivations.md §3.7). L-B is the
// oracle, so it gives the cell's values too. A C++ derivation used to write these L0s, which
// made this harness a second source of one.
struct Cell {
    TileProgram trace;               // the L0, on its inputs, before it runs
    TileProgram oracle;              // every operand as L-B leaves it
};

// The deployment's L3 sizes the schedule, as kpu-csp-gen's --target takes it. An unbounded L3
// (capacity 0) gets the L3 the schedule needs.
csp::gen::Options cell_options(const std::string& algo, Dim size, Dim tile, const platform::DeploymentSpec& spec) {
    csp::gen::Options o;
    o.algo = algo;
    o.size = size;
    o.tile = tile;
    o.l3 = spec.device(0).l3.capacity_tiles;
    if (o.l3 == 0) o.l3 = csp::gen::slots_needed(o);
    return o;
}

// Any refusal here is an error: the one refusal that skips a cell (the schedule does not fit
// the L3) is checked before this is called.
Cell cell_program(const csp::gen::Options& o, const platform::DeploymentSpec& spec) {
    const csp::lang::Target target = csp::lang::target_from(spec.device(0));
    const csp::lang::Program ast = csp::lang::parse(csp::gen::generate(o, &target));
    driver::CspReference r = driver::csp_reference(ast, driver::csp_inputs(ast),
                                                   std::numeric_limits<std::size_t>::max());
    if (!r.trace) throw std::runtime_error(ast.name + ": " + r.note);
    if (!r.replay_mismatch.empty()) throw std::runtime_error(ast.name + ": " + r.replay_mismatch);
    return Cell{std::move(*r.trace), std::move(*r.values)};
}

// EVERY EXECUTION GOES THROUGH THE PLATFORM (#282 increment 4). The characterization is a
// first-order ANALYTICAL model and is deliberately NOT routed through a level: giving an
// estimate a fidelity level would claim something it does not have. It is the VALIDATION that
// executes. A fresh platform per cell, because restore() is platform-wide.
//
// The values are checked per tile against L-B's, interpreted as float32, within the default
// proximity bound (§3.8). Returns the worst absolute error over every operand; `ok` is false
// when a tile is out of bound.
float validate(const platform::DeploymentSpec& spec, const Cell& c, bool& ok) {
    platform::VirtualPlatform vp(spec);
    const platform::ProgramHandle h = vp.load_program(c.trace);
    vp.run(h, platform::ExecutionLevel::Behavioral, vp.snapshot());
    const TileProgram& done = vp.program(h);
    double max_err = 0.0;
    ok = true;
    for (const auto& name : c.oracle.operand_order()) {
        const TileCheck t = check_tiles(done, c.oracle, name);
        if (!t.pass) {
            ok = false;
            std::cerr << "error: " << t.summary() << "\n";
        }
        max_err = std::max(max_err, t.max_abs);
    }
    return static_cast<float>(max_err);
}

// The DEPLOYMENT for one sweep cell. A --deploy spec supplies the machine and the sweep
// varies only what it is asked to vary: topology and compute-tiles come from the sweep
// lists, so they OVERRIDE the spec's -- which is what makes a sweep a sweep. Everything else
// the spec says is kept, which is how a spec's resource fields (L3 banks, L1 vectors) reach
// a cell at all.
platform::DeploymentSpec cell_deployment(const platform::DeploymentSpec& base,
                                         const std::string& topo, Dim cf) {
    platform::DeploymentSpec spec = base;
    spec.device(0).topology = topo;
    spec.device(0).compute_tiles = cf;
    const std::string bad = spec.validate();
    if (!bad.empty()) throw std::invalid_argument(bad);
    return spec;
}

// The dataflow name -> space-time mapping is shared with kpu-run (#285 §D1).
using sw::kpu::program::driver::known_dataflow;
using sw::kpu::program::driver::map_for;

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    if (has_flag(a, "--help") || has_flag(a, "-h")) {
        std::cout <<
            "tile_characterize — DoE harness for L0 tile programs\n"
            "  --algo matmul|lu           (default matmul)\n"
            "  --sizes N[,N...]           square problem size(s) (default 128)\n"
            "  --tiles T[,T...]           tile dimension(s)      (default 32)\n"
            "  --compute-tiles C[,C...]   CF tile count(s)       (default 1,4,16)\n"
            "  --topology single|news|checkerboard[,...] (default single)\n"
            "  --dataflow output-stationary|weight-stationary|a-stationary|fully-streaming[,...]\n"
            "             (matmul only; aliases os/ws/as/hex; systolic L1 timing + network)\n"
            "  --macs-per-cycle F  --bytes-per-cycle F  --pj-per-mac F  --pj-per-byte F\n"
            "  --l3-tiles N               L3 capacity in tiles: the schedule kpu-csp-gen writes\n"
            "                             (0 = unbounded: the L3 the schedule needs)\n"
            "  --deploy FILE              the machine as a deployment spec (ADR 0002 §3.5);\n"
            "                             mutually exclusive with the machine flags above.\n"
            "                             --topology/--compute-tiles remain the SWEEP axes\n"
            "  --csv FILE   --json FILE   --trace FILE (first cell)  --disasm  --no-validate\n"
            "  --dot FILE                 Graphviz tile-dependency DAG of the first cell\n";
        return 0;
    }

    // EVERY numeric flag goes through the shared checked parse (program_spec.hpp). These were
    // bare std::stod / std::stoul: "abc" threw uncaught (SIGABRT, which CI cannot tell from a
    // crash in the model), "-2" wrapped to an enormous sweep count, and "inf" was accepted as
    // a bandwidth. kpu-run fixed this for its own flags; this tool still had it.
    std::string perr;
    std::vector<std::uint32_t> sizes, tiles, cfs;
    double macs_pc = 256.0, bytes_pc = 64.0, pj_mac = 1.0, pj_byte = 20.0;
    std::uint32_t l3_tiles = 0;
    const std::string algo = arg(a, "--algo", "matmul");
    auto topos = parse_strs(arg(a, "--topology", "single"));
    const auto dataflows = parse_strs(arg(a, "--dataflow", "output-stationary"));  // matmul only
    try {
        sizes = parse_ints(arg(a, "--sizes", "128"));
        tiles = parse_ints(arg(a, "--tiles", "32"));
        cfs   = parse_ints(arg(a, "--compute-tiles", "1,4,16"));
        for (const auto v : sizes)
            if (v == 0) throw std::invalid_argument("a size must be positive");
        for (const auto v : tiles)
            if (v == 0) throw std::invalid_argument("a tile must be positive");
    } catch (const std::exception& e) {
        std::cerr << "error: --sizes/--tiles/--compute-tiles: " << e.what() << "\n";
        return 2;
    }
    using sw::kpu::program::driver::parse_double;
    using sw::kpu::program::driver::parse_dim;
    if (!parse_double(a, "--macs-per-cycle", macs_pc, true, macs_pc, perr) ||
        !parse_double(a, "--bytes-per-cycle", bytes_pc, true, bytes_pc, perr) ||
        // Energy coefficients may be ZERO -- "ignore compute energy" is a modelling choice.
        !parse_double(a, "--pj-per-mac", pj_mac, false, pj_mac, perr) ||
        !parse_double(a, "--pj-per-byte", pj_byte, false, pj_byte, perr) ||
        !parse_dim(a, "--l3-tiles", l3_tiles, l3_tiles, perr)) {
        std::cerr << "error: " << perr << "\n";
        return 2;
    }

    // THE MACHINE IS A DEPLOYMENT, from a spec file or from the flags -- the same one
    // description kpu-run uses (#282 increment 1). The sweep then varies topology and
    // compute-tiles per cell; everything else the spec says is kept, which is how a spec's
    // resource fields reach a cell at all.
    // arg_required, NOT arg(): a terminal `--deploy` returns the fallback "" from arg(), and
    // this function would then have swept the flag-built default machine and exited 0 -- the
    // silent machine mismatch the comment below calls the failure this change removes, in the
    // change that removes it. kpu-run already used arg_required for exactly this.
    std::string deploy_path;
    if (!sw::kpu::program::driver::arg_required(a, "--deploy", deploy_path, perr)) {
        std::cerr << "error: " << perr << "\n";
        return 2;
    }
    platform::DeploymentSpec base;
    try {
        if (!deploy_path.empty()) {
            for (const char* k : {"--topology", "--compute-tiles", "--macs-per-cycle",
                                  "--bytes-per-cycle", "--pj-per-mac", "--pj-per-byte",
                                  "--l3-tiles"})
                if (sw::kpu::program::driver::arg_present(a, k)) {
                    std::cerr << "error: --deploy and " << k << " cannot be combined: the "
                                 "spec already says what machine to sweep\n";
                    return 2;
                }
            base = platform::read_spec_file(deploy_path);
        } else {
            sw::kpu::program::driver::DeviceSpec ds;
            ds.macs_per_cycle = macs_pc;
            ds.bytes_per_cycle = bytes_pc;
            ds.pj_per_mac = pj_mac;
            ds.pj_per_byte = pj_byte;
            ds.l3_tiles = l3_tiles;
            base = sw::kpu::program::driver::make_deployment(ds);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    if (base.device_count() != 1) {
        std::cerr << "error: --deploy names " << base.device_count()
                  << " devices; this harness sweeps one machine at a time\n";
        return 2;
    }
    // WITH --deploy, THE MACHINE AXES COME FROM THE SPEC. Leaving them at their flag
    // defaults meant a spec saying "checkerboard, 16 compute tiles" was swept as
    // "single, 1/4/16" without a word -- the machine described and the machine measured
    // being different, which is the exact failure this issue exists to remove. The axis
    // flags are refused alongside --deploy (above), so there is nothing to reconcile:
    // ADR 0002 §3.5 sweeps MACHINES as a list of deployments, not as axes over one spec.
    if (!deploy_path.empty()) {
        topos = {base.device(0).topology};
        cfs = {base.device(0).compute_tiles};
    }
    if (algo != "matmul" && algo != "lu") {
        std::cerr << "error: unknown --algo '" << algo << "' (matmul | lu)\n";
        return 2;
    }
    const bool do_validate = !has_flag(a, "--no-validate");
    const std::string csv_path = arg(a, "--csv", "");
    const std::string json_path = arg(a, "--json", "");
    const std::string trace_path = arg(a, "--trace", "");
    const std::string dot_path = arg(a, "--dot", "");
    const bool disasm = has_flag(a, "--disasm");

    // dataflow sweep applies to matmul (L1 systolic timing); LU has no stream deriver.
    if (algo != "lu")
        for (const auto& df : dataflows)
            if (!known_dataflow(df)) {
                std::cerr << "error: unknown --dataflow '" << df << "' "
                             "(expected output-stationary|weight-stationary|a-stationary|"
                             "fully-streaming, or os/ws/as/hex)\n";
                return 2;
            }
    const std::vector<std::string> dfs = (algo == "lu")
        ? std::vector<std::string>{"-"} : dataflows;

    std::ostringstream csv, json;
    // The DEPLOYMENT DIGEST on every row, so a sweep result says which machine produced it.
    // ADR 0002 §3.5 wants results cacheable; a row that names only the axes it swept cannot
    // be told apart from a row measured on a different spec with the same topology.
    csv << "algo,size,tile,compute_tiles,topology,dataflow,stationary,c_bubble,network,"
        << "func_max_err,deployment," << metrics_csv_header() << "\n";
    json << "[\n";

    std::cout << "algo   size tile  CF topology    dataflow          stat bub network            "
                 "makespan  cmp_util  bound  energy_pJ    err\n";
    std::cout << "------------------------------------------------------------------------------------"
                 "-------------------------------------\n";

    bool did_trace = false, did_disasm = false, did_dot = false, first_json = true, failed = false;
    for (const auto& topo : topos)
        for (Dim size : sizes)
            for (Dim tile : tiles) {
                if (tile > size) continue;
                for (Dim cf : cfs) {
                    platform::DeploymentSpec cell;
                    try {
                        cell = cell_deployment(base, topo, cf);
                    } catch (const std::exception& e) {
                        std::cerr << "error: " << e.what() << "\n";
                        return 2;
                    }
                    const csp::gen::Options opts = cell_options(algo, size, tile, cell);
                    if (csp::gen::slots_needed(opts) > opts.l3) {
                        // The schedule does not fit this machine's L3: the cell is refused, not
                        // characterized as a program written for another machine.
                        std::cout << algo << " " << size << " t" << tile << " CF " << cf << " " << topo
                                  << ": skipped, the schedule needs " << csp::gen::slots_needed(opts)
                                  << " L3 slots; the L3 holds " << opts.l3 << "\n";
                        continue;
                    }
                    Cell program;
                    try {
                        program = cell_program(opts, cell);
                    } catch (const std::exception& e) {
                        std::cerr << "error: " << e.what() << "\n";
                        return 1;
                    }
                    const TileProgram& prog = program.trace;
                    float err = -1.0f;
                    if (do_validate) {
                        bool ok = true;
                        err = validate(cell, program, ok);
                        if (!ok) failed = true;
                    }

                    const DeviceDescriptor dev = cell.device_view();

                    for (const auto& df : dfs) {
                        stream::StreamProgram l1;
                        const stream::StreamProgram* l1p = nullptr;
                        if (algo != "lu") { l1 = stream::derive_matmul_streams(prog, map_for(df)); l1p = &l1; }
                        Metrics m = characterize_program(prog, dev, l1p);

                        if (disasm && !did_disasm) { std::cout << "\n" << prog.disassemble() << "\n"; did_disasm = true; }
                        if (!trace_path.empty() && !did_trace) {
                            write_chrome_trace(prog, dev, trace_path, l1p);
                            std::cout << "[trace] wrote " << trace_path << " (chrome://tracing)\n";
                            did_trace = true;
                        }
                        if (!dot_path.empty() && !did_dot) {
                            // schedule first, so the DOT carries start/finish/worker too
                            TileDag dag(prog, dev, l1p);
                            dag.list_schedule();
                            std::ofstream f(dot_path);
                            if (!f) {
                                std::cerr << "error: cannot write --dot file '" << dot_path << "'\n";
                                return 2;
                            }
                            f << dag.to_dot(prog, prog.name() + "  [" + dev.label() + "]");
                            f.close();   // a buffered write or close can fail after a good open
                            if (!f) {
                                std::cerr << "error: failed writing --dot file '" << dot_path << "'\n";
                                return 2;
                            }
                            std::cout << "[dot] wrote " << dot_path
                                      << " (dot -Tsvg " << dot_path << " -o dag.svg)\n";
                            did_dot = true;
                        }

                        char line[320];
                        std::snprintf(line, sizeof(line),
                            "%-6s %4u %4u %3u %-10s %-17s %-4s %3d %-18s %9.1f %8.2f %5s %10.3g %8.1g\n",
                            algo.c_str(), size, tile, cf, topo.c_str(), df.c_str(),
                            m.stationary.empty() ? "-" : m.stationary.c_str(), m.c_bubble,
                            m.network.empty() ? "-" : m.network.c_str(),
                            m.makespan_cycles, m.compute_util, m.compute_bound ? "cmp" : "mov",
                            m.energy_total_pj, err);
                        std::cout << line;

                        const std::string cell_digest = platform::deployment_digest(cell);
                        csv << algo << ',' << size << ',' << tile << ',' << cf << ',' << topo << ','
                            << df << ',' << (m.stationary.empty() ? "-" : m.stationary) << ','
                            << m.c_bubble << ',' << (m.network.empty() ? "-" : m.network) << ','
                            << err << ',' << cell_digest << ',' << metrics_csv_row(m) << "\n";
                        if (!first_json) json << ",\n";
                        first_json = false;
                        json << "  {\"algo\":\"" << algo << "\",\"size\":" << size << ",\"tile\":" << tile
                             << ",\"compute_tiles\":" << cf << ",\"topology\":\"" << topo
                             << "\",\"dataflow\":\"" << df << "\",\"stationary\":\"" << m.stationary
                             << "\",\"c_bubble\":" << m.c_bubble << ",\"network\":\"" << m.network
                             << "\",\"func_max_err\":" << err
                             << ",\"deployment\":\"" << cell_digest << "\""
                             << ",\"macs\":" << m.total_macs << ",\"arith_intensity\":" << m.arithmetic_intensity
                             << ",\"critical_path\":" << m.critical_path_cycles
                             << ",\"makespan\":" << m.makespan_cycles
                             << ",\"lower_bound\":" << m.lower_bound_cycles
                             << ",\"compute_util\":" << m.compute_util
                             << ",\"movement_util\":" << m.movement_util
                             << ",\"compute_bound\":" << (m.compute_bound ? "true" : "false")
                             << ",\"peak_live_tiles\":" << m.peak_live_tiles
                             << ",\"energy_total_pj\":" << m.energy_total_pj
                             << ",\"feasible\":" << (m.feasible ? "true" : "false") << "}";
                    }
                }
            }
    json << "\n]\n";

    if (!csv_path.empty()) { std::ofstream(csv_path) << csv.str(); std::cout << "[csv] wrote " << csv_path << "\n"; }
    if (!json_path.empty()) { std::ofstream(json_path) << json.str(); std::cout << "[json] wrote " << json_path << "\n"; }
    if (failed) {
        std::cerr << "FAILED: a cell's values are out of bound of L-B's\n";
        return 1;
    }
    return 0;
}
