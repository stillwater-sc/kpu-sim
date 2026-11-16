// ============================================================================
// tools/kpu-run/kpu_run.cpp
// kpu-run — execute a Domain Flow Program and compare the simulation models.
//
// Increment 1 of #285. The point is DIFFERENTIAL TESTING, not convenience:
// values are level-invariant (ADR 0002 §2), so a disagreement between two levels
// is a bug signal by construction. A level that returns plausible TIMING while
// computing wrong VALUES is the failure mode this simulator is most exposed to,
// because timing is what everyone looks at.
//
// Exit codes: 0 agreement, 1 value disagreement, 2 usage/refusal.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/driver/csp_timeline.hpp>
#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/tile_check.hpp>
#include <sw/kpu/program/platform/virtual_platform.hpp>
#include <sw/kpu/program/driver/step_cursor.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/program/record/domain_flow_record.hpp>
#include <sw/kpu/program/record/tile_flow_record.hpp>
#include <sw/kpu/program/csp/bubbles.hpp>
#include <sw/kpu/program/csp/tile_commands.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/version.hpp>
#include <sw/trace/trace_exporter.hpp>

#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;

namespace {

void usage() {
    std::cout <<
R"(kpu-run — the simulator: execute a program at one or more levels and compare them.
kpu-run runs a program; it does not write one. kpu-csp-gen writes CSP programs.

  --program <file.csp>      execute a CSP PROGRAM (docs/plans/kpu-run-csp-programs.md): the
                            program configures the data path, and every level executes it
  --deploy <file.json>      the machine, as a deployment spec (ADR 0002 §3.5). L-CA builds
                            its machine from it; without it L-CA is skipped. REPEATED, it is a
                            sweep: the program runs on each deployment, every level is checked
                            against that deployment's L-B, L-B must agree bit for bit across
                            them, and each run is reported with its deployment's label and
                            digest. A sweep takes --level, --window, --inputs, --trace-limit,
                            --bubbles and --no-compare; what records one run is refused
  --level <name|all>        behavioral | block-sequential | resource-transactional
                            | cycle-accurate | all       (default all)
  --window <W>              L-CA's driver window         (default 256)
  --inputs <values.l0>      the inputs, from an L0 file's values (default: synthesized)
  --trace-limit <n>         trace (and replay) at most this many actions (default 1000000)
  --emit-l0 <file.l0>       write the program's L0 TRACE, inputs inline, once every level
                            has agreed with it: a test case, and how the golden corpus is
                            made (kpu-sim is the one source of an L0)
  --emit-l0-kernel <file.l0>  the trace, structure only (VALUES none)
  --emit-l0-result <file.l0>  the agreed values: the corpus's expected-output half
  --tflow <dir>             the L-T1 run's tile-flow record (.tflow version 6: residencies at
                            their L3 addresses, staged L2 copies, and fabric chains)
  --dflow <dir>             the L-B run's domain-flow record (.dflow version 6, the format the
                            KPU RTL shares): every action as a CSP state transition, written by
                            the oracle's own run
  --dflow-limit <n>         record at most n steps (0 = all; default 1000000); residencies and
                            the iteration summary are always complete
  --dflow-values <tiles>    also record these tiles' values after every action that leaves
                            them somewhere: "C[2,3],A[2,0]", an operand name for all its tiles,
                            or "all". Every operand's final values are recorded regardless
  --dflow-values-cap <MiB>  the values' byte cap (default 64): finals first, then snapshots
                            until the cap, and the record says where they stopped
                            When L-T1 runs too (--level all, or block-sequential), the record
                            carries its overlay: every action's legs in L-T1 cycles, each
                            final tile's proximity to L-B, and (with a declared fabric) each
                            accumulator's gemm chain as a systolic stream
  --tile-commands <file.json>  the program as the KPU RTL's tile command stream (kpu-tile-cmds
                            version 1, #397): LOAD / STORE / EXEC with their sync and chain bits,
                            for one output-stationary int8 tile, with the inputs and L-B's results
  --timeline <file.json>    the L-T1 run as Chrome-trace intervals, one per leg
  --bubbles                 the L-T1 run's reuse and bubble report: each compute lane's idle time
                            before a call and what it waited for (an operand's Feed, Move or
                            Load, a slot's credit, a busy lane), the reuse each operand's
                            program states, and each process's load (#425)
  --step                    single-step (L-B: one action per step; L-T1: one record)
  --step-limit <n>          stop after n steps, 0 = all      (default 40)
  --no-compare              run the levels, do not diff values

  --program <file.l0>       execute an L0 trace FROM A FILE (#265). An L0 program is the
                            oracle CSP runs answer to: it runs at L-B only. With it:
                            --fill-inputs (synthesize the inputs of a file that carries
                            none), --deploy, --emit-l0 (the program before it runs),
                            --emit-l0-result, --step, --step-limit
  -h, --help

Values are checked against L-B, the oracle (the other levels refine the behavioral model),
over EVERY operand rather than just the result: a level that scribbled on an input is a model
bug too. Each tile is read as float32 and every element must lie within |a - o| <= 1e-6 +
1e-4|o| of L-B's: a data path may reorder a reduction, so bit equality is not the bar
(retire-l0-derivations §3.8). A tile out of bound exits 1. L-B also writes the program's L0
trace, and that trace's replay must reproduce L-B bit for bit before it is written.
)";
}

void print_run(const RunOutcome& o) {
    std::cout << "  " << std::left << std::setw(24) << to_string(o.level)
              << std::setw(6) << short_name(o.level);
    if (!o.has_timing) std::cout << "timing: not modelled at this level";
    else std::cout << "makespan " << std::right << std::setw(8) << o.makespan;
    std::cout << "   " << o.ops << " ops\n";
}

// ---- a CSP program (docs/plans/kpu-run-csp-programs.md step 2) ------------------------------
//
// The program is the input. L0 is its oracle: the trace's ops, run by TileProgramReference on
// the same inputs, when the program is small enough to trace.
bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Each level's lines: its makespan, traffic, occupancy and what it does not model.
void print_levels(const std::vector<CspLevelOutcome>& levels, const std::vector<std::string>& run_ids,
                  const csp::lang::Program& prog, std::uint32_t window) {
    for (std::size_t li = 0; li < levels.size(); ++li) {
        const auto& o = levels[li];
        std::cout << "  " << std::left << std::setw(24) << to_string(o.level) << std::setw(6) << short_name(o.level);
        if (o.skipped) {
            std::cout << "skipped: " << *o.skipped << "\n";
            continue;
        }
        if (!o.has_timing) {
            std::cout << "timing: not modelled at this level   " << o.actions << " actions, peak L3 " << o.peak_l3
                      << "\n        run id " << run_ids[li] << "\n";
            continue;
        }
        std::cout << "makespan " << o.makespan << " cycles   " << o.actions << " actions\n";
        std::cout << "        run id " << run_ids[li] << "\n";
        std::cout << "        DRAM  " << o.dram_loads << " loads, " << o.dram_stores << " stores, " << o.dram_bytes
                  << " bytes      cf busy " << o.cf_busy << "\n";
        if (o.level == ExecutionLevel::BlockSequential) {
            std::cout << "        L3    peak " << o.peak_l3 << " of " << prog.l3 << " (in time), credit stalls "
                      << o.credit_stalls << ", slot-bound " << o.slot_bound << " (" << o.slot_wait_cycles
                      << " cycles)\n";
            // L2 staging's credits (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 4).
            if (prog.l2 > 0)
                std::cout << "        L2    peak " << o.peak_l2 << " of " << prog.l2 << " staged (in time), "
                          << o.l2_slots.size() << " stagings, slot-bound " << o.l2_bound << " (" << o.l2_wait_cycles
                          << " cycles)\n";
            std::cout << "        busy ";
            for (const auto& [p, b] : o.busy)
                std::cout << " " << p << " " << b << "/" << (o.lanes.count(p) ? o.lanes.at(p) : 0);
            std::cout << "   (lane-cycles/lanes)   UNCALIBRATED\n";
            // The fabric pipeline (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 2).
            if (o.pe_rows) {
                std::cout << "        fabric " << o.pe_rows << "x" << o.pe_cols << " PEs, output-stationary: "
                          << o.chains.size() << " chains as systolic streams, results waited " << o.link_delay_cycles
                          << " cycles for a top link";
            } else {
                std::cout << "        fabric not declared (fabric.pe_rows/pe_cols): every call is one leg";
            }
            for (const auto& [why, n] : o.one_leg) std::cout << "; one leg: " << n << " (" << why << ")";
            std::cout << "\n";
            for (const auto& u : o.unmodelled) std::cout << "        unmodelled: " << u << "\n";
            continue;
        }
        std::cout << "        vector  combiner " << o.ve.combiner_busy << "/" << o.ve.combiner_bound << "  bm " << o.ve.bm_busy << "/"
                  << o.ve.bm_bound << "  fabric " << o.ve.fabric << "   (busy/bound)   window " << window << "\n";
        for (const auto& u : o.unmodelled) std::cout << "        unmodelled: " << u << "\n";
    }
}

// Every level that ran, against L-B's values, printed; true when all are within bound.
bool print_value_check(const std::vector<CspLevelOutcome>& levels, const TileProgram& oracle) {
    // Every operand, tile by tile, within the proximity bound of L-B's: the levels refine the
    // behavioral model, and a data path may round differently (§3.8).
    const TileProgram* authority = &oracle;
    // How each operand is compared: in its own number system (tile_check), integers exactly.
    std::string bounds;
    for (const auto& name : authority->operand_order()) {
        const TensorOperand& t = authority->operand(name);
        const std::string b = t.fp32() ? "fp32 within 1e-6 + 1e-4|o|" : t.type + " exact";
        if (bounds.find(b) == std::string::npos) bounds += (bounds.empty() ? "" : "; ") + b;
    }
    std::cout << "\nvalues vs L-B, the oracle (every operand, per tile, " << bounds << ")\n";
    bool all_agree = true;
    for (const auto& o : levels) {
        if (o.skipped) continue;
        std::cout << "  " << std::left << std::setw(24) << to_string(o.level);
        const OperandsCheck c = check_operands(o.values, *authority);
        all_agree = all_agree && c.pass;
        if (c.pass) {
            std::cout << "within bound (" << c.operands.size() << " operands, max abs " << c.max_abs << ")\n";
            continue;
        }
        std::cout << "\n";
        for (const TileCheck& t : c.operands)
            if (!t.pass) std::cout << "      " << t.summary() << "\n";
    }
    return all_agree;
}

// The values of two runs, bit for bit: every operand, in the same order, the same shape and type,
// the same words. L-B is the same computation on every deployment, so here bit equality is the bar.
bool same_values(const TileProgram& a, const TileProgram& b, std::string& first_difference) {
    if (a.operand_order() != b.operand_order()) {
        first_difference = "the operands differ";
        return false;
    }
    for (const auto& name : a.operand_order()) {
        const TensorOperand& x = a.operand(name);
        const TensorOperand& y = b.operand(name);
        if (x.rows != y.rows || x.cols != y.cols || x.type != y.type || x.values.size() != y.values.size() ||
            (!x.values.empty() &&
             std::memcmp(x.values.data(), y.values.data(), x.values.size() * sizeof(x.values[0])) != 0)) {
            first_difference = "operand " + name;
            return false;
        }
    }
    return true;
}

// A SWEEP (#285): the same program, on the same inputs, over several deployments. Each deployment
// runs the requested levels through its own platform, and every level is checked against that
// deployment's L-B; L-B must then compute the same values on every deployment. Each run is
// reported with its provenance (the deployment's label and digest, and the run's identity), so two
// deployments' results cannot be mistaken for each other.
//
// Exit: 1 when a level disagrees with L-B, a trace's replay does not reproduce L-B, or L-B differs
// across deployments (each is a kpu-sim defect); otherwise 2 when a deployment refused the program
// or no level ran on it; otherwise 0.
int run_csp_sweep(const std::vector<std::string>& a, const std::string& path, const csp::lang::Program& ast,
                  const TileProgram& inputs, const std::string& inputs_note,
                  const std::vector<std::string>& deploy_paths, const std::vector<ExecutionLevel>& levels,
                  std::uint32_t window, std::uint32_t trace_limit) {
    // Every deployment is read before anything runs: a file that cannot be read is a usage error,
    // not a row of the sweep.
    std::vector<platform::DeploymentSpec> specs;
    for (const auto& dp : deploy_paths) {
        try {
            specs.push_back(platform::read_spec_file(dp));
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: " << e.what() << "\n";
            return 2;
        }
        if (specs.back().devices.size() != 1) {
            std::cerr << "kpu-run: " << dp << ": the deployment has " << specs.back().devices.size()
                      << " devices; multi-device execution is not implemented\n";
            return 2;
        }
    }
    const bool compare = !has_flag(a, "--no-compare");
    // --bubbles reads each deployment's L-T1 run, as a single run's does.
    if (has_flag(a, "--bubbles") &&
        std::find(levels.begin(), levels.end(), ExecutionLevel::BlockSequential) == levels.end()) {
        std::cerr << "kpu-run: --bubbles reads the L-T1 run, which a sweep at --level " << arg(a, "--level", "all")
                  << " does not make (add block-sequential to --level)\n";
        return 2;
    }

    struct Row {
        std::string digest;
        std::optional<std::string> refused;             // why this deployment did not run the program
        std::vector<CspLevelOutcome> levels;
        std::vector<std::string> run_ids;
        std::optional<TileProgram> oracle;               // L-B's values on this deployment
        bool agree = true;                               // every level within bound of L-B
        bool replay_ok = true;
        bool no_bubbles = false;                         // --bubbles, and L-T1 did not run here
    };
    std::vector<Row> rows(specs.size());

    std::cout << "program  " << path << "   \"" << ast.name << "\"  csp " << ast.version << ", " << ast.decls.size()
              << " operands, L3 " << ast.l3 << "   values: " << inputs_note << "\n";
    std::cout << "sweep    " << specs.size() << " deployments, the same program and inputs on each\n";

    for (std::size_t d = 0; d < specs.size(); ++d) {
        const platform::DeploymentSpec& spec = specs[d];
        Row& row = rows[d];
        row.digest = platform::deployment_digest(spec);
        std::cout << "\n== deployment " << d + 1 << " of " << specs.size() << ": " << deploy_paths[d] << "\n"
                  << "device   " << spec.label() << "   digest " << row.digest << "\n";
        for (std::size_t e = 0; e < d; ++e)
            if (rows[e].digest == row.digest)
                std::cout << "         the same deployment as " << e + 1 << " (" << deploy_paths[e] << ")\n";
        try {
            platform::VirtualPlatform vp(spec);
            const csp::lang::Target target = csp::lang::target_from(spec.device(0));
            const csp::lang::Validation v = csp::lang::validate(ast, &target);
            const platform::CspProgramHandle h = vp.load_csp(ast, inputs, true);
            const driver::CspReference ref = vp.csp_reference(h, trace_limit, nullptr);
            std::cout << "         validated: peak L3 " << v.peak_l3 << (ast.l2 > 0 ? ", peak L2 " + std::to_string(v.peak_l2) : "")
                      << "\noracle   L-B, " << ref.actions << " actions; "
                      << (ref.trace ? "L0 trace of " + std::to_string(ref.trace->ops().size()) + " ops, replayed bit-exact"
                                    : ref.note)
                      << "\n\n";
            row.oracle = ref.values;
            row.replay_ok = ref.replay_mismatch.empty();
            for (ExecutionLevel l : levels) {
                auto r = vp.run_csp(h, l, window);
                row.run_ids.push_back(r.outcome.skipped ? std::string() : r.identity.str());
                row.levels.push_back(std::move(r.outcome));
            }
        } catch (const std::exception& e) {
            row.refused = e.what();
            std::cout << "refused  " << e.what() << "\n";
            continue;
        }
        print_levels(row.levels, row.run_ids, ast, window);
        if (!row.replay_ok) std::cout << "\n  FAILED: the trace does not reproduce L-B on this deployment\n";
        if (has_flag(a, "--bubbles")) {
            bool reported = false;
            for (const auto& o : row.levels)
                if (o.level == ExecutionLevel::BlockSequential && !o.skipped) {
                    std::cout << "\n" << sw::kpu::program::csp::bubble_report(o.records, o.makespan, o.busy, o.lanes).text();
                    reported = true;
                }
            // L-T1 skipped on this deployment: the report it asked for is missing, which is a refusal.
            if (!reported) row.no_bubbles = true;
        }
        if (compare && row.oracle) row.agree = print_value_check(row.levels, *row.oracle);
    }

    // The sweep: one line per deployment, every level's makespan, and its verdict.
    std::cout << "\nsweep    " << path << "\n  " << std::left << std::setw(4) << "#" << std::setw(18) << "digest";
    for (ExecutionLevel l : levels) std::cout << std::setw(12) << short_name(l);
    std::cout << "values\n";
    bool failed = false, refused = false;
    const TileProgram* first = nullptr;
    std::size_t first_at = 0;
    std::vector<std::string> across;                 // deployments whose L-B differs from the first's
    for (std::size_t d = 0; d < rows.size(); ++d) {
        const Row& row = rows[d];
        std::cout << "  " << std::left << std::setw(4) << d + 1 << std::setw(18) << row.digest;
        if (row.refused) {
            refused = true;
            std::cout << "refused: " << *row.refused << "\n";
            continue;
        }
        bool ran = false;
        for (const auto& o : row.levels) {
            const std::string cell = o.skipped ? "skipped" : o.has_timing ? std::to_string(o.makespan) : "untimed";
            ran = ran || !o.skipped;
            std::cout << std::setw(12) << cell;
        }
        if (!ran || row.no_bubbles) refused = true;
        failed = failed || !row.agree || !row.replay_ok;
        std::cout << (!row.replay_ok ? "FAILED (replay)" : !compare ? "not compared" : row.agree ? "agree" : "FAILED")
                  << "   " << deploy_paths[d] << "\n";
        if (!row.oracle || !compare) continue;
        std::string diff;
        if (!first) {
            first = &*row.oracle;
            first_at = d;
        } else if (!same_values(*first, *row.oracle, diff)) {
            across.push_back(std::to_string(d + 1) + " (" + diff + ")");
        }
    }
    std::cout << "  (makespans in cycles; each row's levels ran on that deployment and are named by its run ids above)\n";
    if (first) {
        if (across.empty()) {
            std::cout << "\nL-B computes the same values, bit for bit, on every deployment that ran it\n";
        } else {
            failed = true;
            std::cout << "\nFAILED: L-B's values differ from deployment " << first_at + 1 << "'s on deployment";
            for (const auto& x : across) std::cout << " " << x;
            std::cout << "\n";
        }
    }
    if (failed) {
        std::cout << "\nFAILED: the sweep does not compute the program's values on every deployment.\n";
        return 1;
    }
    if (refused) {
        std::cout << "\nrefused: the program did not run on every deployment.\n";
        return 2;
    }
    std::cout << "\nOK: every level that ran, on every deployment, " << (compare ? "computes the program's values.\n" : "ran (values not compared).\n");
    return 0;
}

int run_csp_program(const std::vector<std::string>& a, const std::string& path) {
    // What a CSP program run accepts. Everything else is the L0 path's (or the generator's:
    // --algo and friends are kpu-csp-gen's), and is refused by name rather than ignored.
    static const std::set<std::string> kOptions = {"--program", "--deploy", "--level", "--window", "--inputs",
                                                   "--trace-limit", "--no-compare", "--emit-l0-result",
                                                   "--emit-l0", "--emit-l0-kernel",
                                                   "--tflow", "--step", "--step-limit", "--timeline",
                                                   "--dflow", "--dflow-limit", "--dflow-values", "--dflow-values-cap",
                                                   "--tile-commands", "--bubbles"};
    for (const std::string& t : a)
        if (t.rfind("--", 0) == 0 && !kOptions.count(t)) {
            std::cerr << "kpu-run: " << t << " does not apply to a CSP program"
                      << (t == "--algo" || t == "--size" || t == "--tile" || t == "--act"
                              ? " (the program already says what it computes; kpu-csp-gen writes programs)"
                              : "")
                      << "\n";
            return 2;
        }
    std::string err, deploy_path, inputs_path, emit_path, tflow_path, timeline_path, trace_path, kernel_path, dflow_path,
        dflow_values, tile_cmds_path;
    std::vector<std::string> deploy_paths;
    if (!arg_all(a, "--deploy", deploy_paths, err) || !arg_required(a, "--inputs", inputs_path, err) ||
        !arg_required(a, "--emit-l0-result", emit_path, err) || !arg_required(a, "--tflow", tflow_path, err) ||
        !arg_required(a, "--timeline", timeline_path, err) || !arg_required(a, "--emit-l0", trace_path, err) ||
        !arg_required(a, "--emit-l0-kernel", kernel_path, err) || !arg_required(a, "--dflow", dflow_path, err) ||
        !arg_required(a, "--dflow-values", dflow_values, err) ||
        !arg_required(a, "--tile-commands", tile_cmds_path, err)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
    std::uint32_t window = 256, trace_limit = 1000000, step_limit = 40, dflow_limit = 1000000, dflow_cap_mib = 64;
    if (!parse_dim(a, "--window", 256, window, err) || !parse_dim(a, "--trace-limit", 1000000, trace_limit, err) ||
        !parse_dim(a, "--step-limit", 40, step_limit, err) || !parse_dim(a, "--dflow-limit", 1000000, dflow_limit, err) ||
        !parse_dim(a, "--dflow-values-cap", 64, dflow_cap_mib, err)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
    // Every emitted file is written only after the levels agreed, so a run that does not compare
    // writes none. Refused here rather than skipped at the end: a regeneration script that asked
    // for a file must not get exit 0 and no file.
    if (has_flag(a, "--no-compare") && (!emit_path.empty() || !trace_path.empty() || !kernel_path.empty())) {
        std::cerr << "kpu-run: --emit-l0, --emit-l0-kernel and --emit-l0-result are written only after the "
                     "levels agree, so they cannot be combined with --no-compare\n";
        return 2;
    }
    if (window == 0) {
        std::cerr << "kpu-run: --window must be positive\n";
        return 2;
    }
    // A SWEEP: the same program over several deployments (--deploy repeated, #285). Each run is
    // reported with its deployment's provenance; what records ONE run is refused by name.
    const bool sweep = deploy_paths.size() > 1;
    if (deploy_paths.size() == 1) deploy_path = deploy_paths.front();
    if (sweep) {
        for (const char* k : {"--tflow", "--dflow", "--dflow-values", "--timeline", "--tile-commands", "--emit-l0",
                              "--emit-l0-kernel", "--emit-l0-result", "--step"})
            if (arg_present(a, k)) {
                std::cerr << "kpu-run: " << k << " records one run, and " << deploy_paths.size()
                          << " deployments make a sweep: run it with one --deploy\n";
                return 2;
            }
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "kpu-run: cannot read '" << path << "'\n";
        return 2;
    }
    std::ostringstream text;
    text << in.rdbuf();
    std::optional<csp::lang::Program> ast;
    try {
        ast.emplace(csp::lang::parse(text.str()));
    } catch (const std::exception& e) {
        std::cerr << "kpu-run: " << path << ": " << e.what() << "\n";
        return 2;
    }

    std::optional<platform::DeploymentSpec> deployment;
    if (!deploy_path.empty()) {
        try {
            deployment.emplace(platform::read_spec_file(deploy_path));
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: " << e.what() << "\n";
            return 2;
        }
        if (deployment->devices.size() != 1) {
            std::cerr << "kpu-run: the deployment has " << deployment->devices.size()
                      << " devices; multi-device execution is not implemented\n";
            return 2;
        }
    }
    const platform::DeviceSpecification* device = deployment ? &deployment->devices.front() : nullptr;

    // Levels: "all" is every level that can run this program here; a named one that cannot is
    // a usage error with its reason.
    const std::string level_arg = arg(a, "--level", "all");
    std::vector<ExecutionLevel> levels;
    if (level_arg == "all") {
        levels = all_levels();
    } else {
        const auto parsed = parse_level(level_arg);
        if (!parsed) {
            std::cerr << "kpu-run: unknown --level '" << level_arg << "'\n";
            return 2;
        }
        // In a sweep each deployment says whether it can run the level (its run reports why not).
        if (const auto why = sweep ? std::nullopt : csp_level_supported(*parsed, *ast, device)) {
            std::cerr << "kpu-run: " << to_string(*parsed) << ": " << *why << "\n";
            return 2;
        }
        levels.push_back(*parsed);
    }

    // Inputs: synthesized from the operand names, or the values an L0 file carries.
    TileProgram inputs;
    std::string inputs_note = "synthesized (by operand name)";
    try {
        inputs = csp_inputs(*ast);
        if (!inputs_path.empty()) {
            std::ifstream vin(inputs_path, std::ios::binary);
            if (!vin) {
                std::cerr << "kpu-run: cannot read '" << inputs_path << "'\n";
                return 2;
            }
            serialize::LoadInfo vinfo;
            const TileProgram values = serialize::read_l0(vin, &vinfo);
            if (!vinfo.has_values) {
                std::cerr << "kpu-run: " << inputs_path << " carries no values (VALUES none)\n";
                return 2;
            }
            for (const auto& name : inputs.operand_order()) {
                auto& t = inputs.operand(name);
                if (!values.has_operand(name)) {
                    std::cerr << "kpu-run: " << inputs_path << " has no operand '" << name << "'\n";
                    return 2;
                }
                const auto& v = values.operand(name);
                if (v.rows != t.rows || v.cols != t.cols) {
                    std::cerr << "kpu-run: " << inputs_path << ": operand '" << name << "' is " << v.rows << "x"
                              << v.cols << "; the program declares " << t.rows << "x" << t.cols << "\n";
                    return 2;
                }
                if (v.type != t.type) {
                    std::cerr << "kpu-run: " << inputs_path << ": operand '" << name << "' is " << v.type
                              << "; the program declares " << t.type << "\n";
                    return 2;
                }
                t.values = v.values;
            }
            inputs_note = "from " + inputs_path;
        }
    } catch (const std::exception& e) {
        std::cerr << "kpu-run: " << e.what() << "\n";
        return 2;
    }

    if (sweep) return run_csp_sweep(a, path, *ast, inputs, inputs_note, deploy_paths, levels, window, trace_limit);

    // THROUGH THE PLATFORM, the one execution path (kpu-run-csp-programs step 4a). Without
    // --deploy the platform holds the default device, and L-CA -- which builds its machine from
    // a deployment -- is skipped with that reason.
    platform::DeploymentSpec spec = deployment ? *deployment : make_deployment(DeviceSpec{});
    std::optional<platform::VirtualPlatform> vp;
    platform::CspProgramHandle handle;
    csp::lang::Validation validation;
    driver::CspReference reference;
    std::optional<record::DomainFlowRecorder> dflow;
    auto write_dflow_bundle = [&](const std::string& run_id, const TileProgram* finals,
                                  const CspLevelOutcome* lt1 = nullptr, const std::string& lt1_id = "") {
        try {
            record::DomainFlowRecord rec = dflow->finish(finals);
            rec.run_id = run_id;
            // The L-T1 overlay, when L-T1 ran too: its legs joined on the action, and its values
            // checked per tile against L-B's (domain-flow step 5).
            if (lt1 && finals) record::attach_lt1(rec, *lt1, *finals, lt1_id);
            record::write_dflow(rec, dflow_path);
            std::cout << "wrote  " << dflow_path << "  (" << rec.steps.size() << " of " << rec.actions
                      << " steps, " << rec.residencies.size() << " residencies, " << rec.iterations.size()
                      << " iterations, from L-B" << (rec.cut ? "; steps cut at --dflow-limit" : "")
                      << (rec.refusal ? "; L-B refused action " + std::to_string(rec.refusal->action) : "")
                      << "; " << rec.snapshots.size() << " tile values"
                      << (rec.values_cut_at ? ", cut at the cap at action " + std::to_string(*rec.values_cut_at) : "")
                      << (rec.lt1 ? "; L-T1 overlay, " + std::to_string(rec.legs.size()) + " legs" : "")
                      << ")\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: --dflow: " << e.what() << "\n";
            return false;
        }
    };
    // A program L-B refuses is recorded up to the refusal, which the record names (§6 invariant 4).
    auto write_refused_dflow = [&]() {
        if (dflow && dflow->refused_at()) (void)write_dflow_bundle("", nullptr);
    };
    std::vector<CspLevelOutcome> outcomes;
    std::vector<std::string> run_ids;
    try {
        vp.emplace(spec);
        const csp::lang::Target target = csp::lang::target_from(spec.device(0));
        validation = csp::lang::validate(*ast, deployment ? &target : nullptr);
        handle = vp->load_csp(*ast, inputs, deployment.has_value());
        if (!dflow_path.empty()) {
            record::ValueSelection sel;
            if (!dflow_values.empty()) sel = record::parse_value_selection(dflow_values);
            sel.cap_bytes = static_cast<std::uint64_t>(dflow_cap_mib) << 20;
            dflow.emplace(*ast, text.str(), dflow_limit, sel);
        }
        reference = vp->csp_reference(handle, trace_limit, dflow ? &*dflow : nullptr);
        for (ExecutionLevel l : levels) {
            if (!deployment && l == ExecutionLevel::CycleAccurate) {
                CspLevelOutcome o;
                o.level = l;
                o.skipped = std::string("L-CA builds its machine from a deployment spec (--deploy)");
                outcomes.push_back(std::move(o));
                run_ids.emplace_back();
                continue;
            }
            auto r = vp->run_csp(handle, l, window);
            run_ids.push_back(r.outcome.skipped ? std::string() : r.identity.str());
            outcomes.push_back(std::move(r.outcome));
        }
    } catch (const csp::lang::CompileError& e) {
        std::cerr << "kpu-run: " << path << ": " << e.what() << "\n";
        write_refused_dflow();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "kpu-run: " << e.what() << "\n";
        write_refused_dflow();
        return 2;
    }
    // --dflow: the domain-flow record of the oracle's run (domain-flow step 2).
    if (dflow) {
        std::string run_id, lt1_id;
        const CspLevelOutcome* lt1 = nullptr;
        for (std::size_t li = 0; li < outcomes.size(); ++li) {
            if (outcomes[li].level == ExecutionLevel::Behavioral) run_id = run_ids[li];
            if (outcomes[li].level == ExecutionLevel::BlockSequential && !outcomes[li].skipped) {
                lt1 = &outcomes[li];
                lt1_id = run_ids[li];
            }
        }
        if (!write_dflow_bundle(run_id, reference.values ? &*reference.values : nullptr, lt1, lt1_id)) return 2;
    }
    // --tile-commands: the program as the RTL's tile command stream (#397), with the inputs and
    // L-B's results in it, for the RTL's runtime to execute.
    if (!tile_cmds_path.empty()) {
        try {
            const csp::CspProgram p = csp::lang::compile(*ast);
            csp::TileTarget t;
            if (!p.source.operand_order().empty()) t.dim = static_cast<std::uint32_t>(p.source.operand(p.source.operand_order()[0]).tile_rows);
            const auto s = csp::lower_tile_commands(p, t, &inputs, reference.values ? &*reference.values : nullptr);
            std::ofstream out(tile_cmds_path);
            out << csp::to_json(s, "kpu-sim " + std::string(sw::kpu::VERSION_FULL));
            out.close();                                // flushes: a failed tail write is a failure
            if (!out) throw std::runtime_error("cannot write " + tile_cmds_path);
            std::cout << "wrote  " << tile_cmds_path << "  (kpu-tile-cmds version " << csp::tilecmd::kVersion << ": "
                      << s.commands.size() << " commands, " << s.l3_init.size() << " input tiles, " << s.results.size()
                      << " result tiles" << (reference.values ? ", with L-B's values" : "") << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: --tile-commands: " << e.what() << "\n";
            return 2;
        }
    }
    CspRunResult res;
    res.validation = validation;
    res.reference = reference.values;
    res.reference_note = reference.note;
    res.actions = reference.actions;
    res.levels = outcomes;

    std::cout << "program  " << path << "   \"" << ast->name << "\"  csp " << ast->version << ", "
              << ast->decls.size() << " operands, L3 " << ast->l3 << "   values: " << inputs_note << "\n";
    std::cout << "         validated: peak L3 " << res.validation.peak_l3;
    if (res.validation.loads) std::cout << ", " << *res.validation.loads << " loads";
    if (res.validation.calls) std::cout << ", " << *res.validation.calls << " calls";
    if (res.validation.stores) std::cout << ", " << *res.validation.stores << " stores";
    if (res.validation.inherited) std::cout << ", " << res.validation.inherited << " inherited";
    if (res.validation.retained) std::cout << ", " << res.validation.retained << " retained";
    if (ast->l2 > 0) std::cout << ", peak L2 " << res.validation.peak_l2 << " of " << ast->l2 << " staged";
    std::cout << "\n";
    // L-B is the oracle; its trace, replayed, must reproduce it before it can be written.
    std::cout << "oracle   L-B, " << res.actions << " actions; "
              << (reference.trace ? std::string("L0 trace of ") + std::to_string(reference.trace->ops().size()) +
                                        " ops, replayed bit-exact"
                                  : res.reference_note) << "\n";
    std::cout << "device   " << (deployment ? deployment->label() + "   digest " + platform::deployment_digest(*deployment)
                                            : std::string("none (no --deploy): the default device")) << "\n\n";

    print_levels(res.levels, run_ids, *ast, window);

    // --tflow: the tile-flow record (version 6) of the program's L-T1 run (step 4b; its residencies
    // carry their L3 address since l3-placement-at-lb step 5b, and its staged L2 copies and fabric
    // chains since l2-reuse-and-fabric-pipeline-view step 4).
    if (!tflow_path.empty()) {
        const CspLevelOutcome* lt1 = nullptr;
        for (const auto& o : res.levels)
            if (o.level == ExecutionLevel::BlockSequential && !o.skipped) lt1 = &o;
        if (!lt1) {
            std::cerr << "kpu-run: --tflow records the L-T1 run, which did not run (add block-sequential to --level)\n";
            return 2;
        }
        try {
            const auto rec = record::build_csp_record(*ast, *lt1, inputs, spec);
            record::write_tflow(rec, tflow_path);
            std::cout << "wrote  " << tflow_path << "  (" << rec.residency.size() << " residencies, "
                      << rec.transits.size() << " transits, " << rec.computes.size()
                      << " computes, " << rec.l2_residency.size() << " staged L2 copies, " << rec.chains.size()
                      << " fabric chains from L-T1; ops are the program's actions, residencies at their L3 "
                      << "addresses, version 6)\n";
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: --tflow: " << e.what() << "\n";
            return 2;
        }
    }

    // --timeline: the L-T1 run's records as Chrome-trace intervals, one per leg (step 4c).
    if (!timeline_path.empty()) {
        const CspLevelOutcome* lt1 = nullptr;
        for (const auto& o : res.levels)
            if (o.level == ExecutionLevel::BlockSequential && !o.skipped) lt1 = &o;
        if (!lt1) {
            std::cerr << "kpu-run: --timeline records the L-T1 run, which did not run (add block-sequential to "
                         "--level)\n";
            return 2;
        }
        const auto entries = csp_trace_entries(lt1->records, inputs, spec.device_view().element_bytes);
        if (!sw::trace::ChromeTraceExporter::export_traces(timeline_path, entries)) {
            std::cerr << "kpu-run: could not write '" << timeline_path << "'\n";
            return 2;
        }
        std::cout << "wrote  " << timeline_path << "  (" << entries.size()
                  << " events, one per leg, from L-T1; per process and lane)\n";
    }

    // --bubbles: the L-T1 run's reuse and bubble report (#425): where the compute lanes waited for
    // data, why, and the reuse the program stated.
    if (has_flag(a, "--bubbles")) {
        const CspLevelOutcome* lt1 = nullptr;
        for (const auto& o : res.levels)
            if (o.level == ExecutionLevel::BlockSequential && !o.skipped) lt1 = &o;
        if (!lt1) {
            std::cerr << "kpu-run: --bubbles reads the L-T1 run, which did not run (add block-sequential to --level)\n";
            return 2;
        }
        std::cout << sw::kpu::program::csp::bubble_report(lt1->records, lt1->makespan, lt1->busy, lt1->lanes).text();
    }

    // --step: walk the program at the finest of L-B and L-T1 that ran, through the platform
    // (step 4c). L-B applies one action per step; L-T1 replays its records in start order.
    if (has_flag(a, "--step")) {
        std::optional<ExecutionLevel> target;
        for (const auto& o : res.levels)
            if (!o.skipped && (o.level == ExecutionLevel::Behavioral || o.level == ExecutionLevel::BlockSequential))
                target = o.level;
        if (!target) {
            std::cerr << "kpu-run: --step walks L-B (actions) or L-T1 (records); neither ran (L-CA's step is a "
                         "cycle, #283)\n";
            return 2;
        }
        std::optional<platform::VirtualPlatform::CspCursor> cur;
        try {
            cur.emplace(vp->step_begin(handle, *target));
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: " << e.what() << "\n";
            return 2;
        }
        std::cout << "\nstepping " << short_name(*target) << ": " << cur->size()
                  << (cur->executes() ? " actions (each applied as it is stepped)"
                                      : " records (a replay of the run, in start order)")
                  << (step_limit && cur->size() > step_limit ? ", showing the first " + std::to_string(step_limit)
                                                              : "")
                  << "\n";
        std::size_t shown = 0;
        // The limit is checked before advancing: at L-B a step applies an action.
        while (!step_limit || shown < step_limit) {
            if (!cur->step()) break;
            const driver::CspStep& st = cur->current();
            std::cout << "  " << std::setw(6) << std::right << cur->position() << "  " << std::left
                      << describe(st, !cur->executes()) << "   L3 " << st.l3_held << "/" << ast->l3;
            if (cur->executes())
                std::cout << "  L2 " << st.l2_held << "  fabric " << st.cf_held;
            else
                for (const auto& [p, n] : cur->stepper().lanes_busy()) std::cout << "  [" << p << " " << n << "]";
            std::cout << "\n";
            ++shown;
        }
        std::cout << "  stopped after " << cur->position() << " of " << cur->size() << " steps\n";
    }

    // A trace that does not reproduce the run it records is a kpu-sim defect, whatever was asked.
    if (!reference.replay_mismatch.empty()) {
        std::cout << "\nFAILED: " << reference.note << "\n";
        return 1;
    }
    if (has_flag(a, "--no-compare")) return 0;
    const bool all_agree = print_value_check(res.levels, *res.reference);
    if (!all_agree) {
        std::cout << "\nFAILED: the levels do not compute the program's values.\n";
        return 1;
    }
    std::cout << "\nOK: every level that ran computes the program's values.\n";
    // The agreed values, written only after the comparison passed: a file from a run whose
    // levels disagreed must never look like an expected output.
    if (!emit_path.empty()) {
        std::ofstream out(emit_path, std::ios::binary);
        serialize::WriteOptions opt;
        opt.include_values = true;
        TileProgram result = *res.reference;
        if (!(out << serialize::to_string(result, opt))) {
            std::cerr << "kpu-run: cannot write '" << emit_path << "'\n";
            return 2;
        }
        std::cout << "wrote  " << emit_path << "  (every operand, as the program leaves them)\n";
    }
    // The program's L0 trace, written by the simulator that ran it (ADR 0004 §4: kpu-sim is
    // the one source of an L0, and L-B the one place it is made), only after its replay reproduced
    // L-B and every level agreed with L-B: this is how a test reference is made. --emit-l0 writes a test case (the inputs inline); --emit-l0-kernel the
    // structure alone. A program too long to trace has no trace to write, which is refused.
    for (const auto& [file, with_values] : {std::pair<const std::string&, bool>{trace_path, true},
                                            std::pair<const std::string&, bool>{kernel_path, false}}) {
        if (file.empty()) continue;
        if (!reference.trace) {
            std::cerr << "kpu-run: " << (with_values ? "--emit-l0" : "--emit-l0-kernel")
                      << ": the program has no L0 trace (" << reference.note << ")\n";
            return 2;
        }
        std::ofstream out(file, std::ios::binary);
        serialize::WriteOptions opt;
        opt.include_values = with_values;
        if (!(out << serialize::to_string(*reference.trace, opt))) {
            std::cerr << "kpu-run: cannot write '" << file << "'\n";
            return 2;
        }
        std::cout << "wrote  " << file << "  (the program's L0 trace, " << reference.trace->ops().size()
                  << " ops" << (with_values ? ", inputs inline)" : ", structure only)") << "\n";
    }
    return 0;
}

// ---- an L0 program: the trace, run at L-B as the oracle ---------------------------------------
int run_l0_program(const std::vector<std::string>& a, const std::string& program_path) {
    // What an L0 program run accepts. An L0 program is a trace (ADR 0004 §4) and runs at L-B,
    // which models no machine, so the timing options are refused by name, with where their
    // job went, rather than ignored.
    static const std::set<std::string> kOptions = {"--program", "--fill-inputs", "--deploy", "--level",
                                                   "--emit-l0", "--emit-l0-result", "--step", "--step-limit"};
    static const std::set<std::string> kDeviceFlags = {
        "--topology", "--compute-tiles", "--l3-tiles", "--dma-engines", "--block-movers", "--streamers",
        "--noc-links", "--dma-bytes-per-cycle", "--bm-bytes-per-cycle", "--str-bytes-per-cycle",
        "--noc-bytes-per-cycle", "--macs-per-cycle"};
    for (const std::string& t : a) {
        if (t.rfind("--", 0) != 0 || kOptions.count(t)) continue;
        if (t == "--streams" || t == "--timeline" || t == "--tflow")
            std::cerr << "kpu-run: " << t << " times an L0 program's L-T1 run, which retired: an L0 "
                         "program is a trace and runs at L-B. Time its CSP program instead "
                         "(--program file.csp; write one with kpu-csp-gen)\n";
        else if (kDeviceFlags.count(t))
            std::cerr << "kpu-run: " << t << " left kpu-run (kpu-run-csp-programs step 5): the machine "
                         "is a deployment spec (--deploy), and an L0 program runs at L-B, which "
                         "models none of it\n";
        else
            std::cerr << "kpu-run: " << t << " does not apply to an L0 program\n";
        return 2;
    }
    std::string err;

    // Load it. A refusal must be a DIAGNOSTIC and an exit code -- never a variant crash
    // and never a silent mis-parse -- which is why read_l0 throws FormatError with a cause
    // rather than returning a half-built program.
    std::optional<TileProgram> loaded;
    serialize::LoadInfo info;
    {
        std::ifstream in(program_path, std::ios::binary);   // see --emit-l0 on binary mode
        if (!in) {
            std::cerr << "kpu-run: cannot read '" << program_path << "'\n";
            return 2;
        }
        try {
            loaded.emplace(serialize::read_l0(in, &info));
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: " << program_path << ": " << e.what() << "\n";
            return 2;
        }
    }

    // Inputs. A program that carries no values would execute on zeros and report success,
    // which is the one outcome worse than refusing.
    const bool want_fill = has_flag(a, "--fill-inputs");
    if (!info.has_values && !want_fill) {
        std::cerr << "kpu-run: " << program_path << " carries no values (VALUES none), "
                     "so its run would compute zeros. Pass --fill-inputs to "
                     "synthesize inputs, or use a file written as a test case\n";
        return 2;
    }
    if (info.has_values && want_fill) {
        std::cerr << "kpu-run: " << program_path << " carries its own values; "
                     "--fill-inputs would replace them and the run would no longer be "
                     "the one the file describes\n";
        return 2;
    }
    if (want_fill) fill_inputs(*loaded);

    // A file may carry a STREAMS record, a dataflow choice. Nothing here consumes it any more,
    // but --emit-l0 and --emit-l0-result keep it, so the file a run writes still says what the
    // file it read said.
    const std::string dataflow = info.has_streams ? info.dataflow : std::string();
    std::string emit_path, emit_result_path, deploy_path;
    if (!arg_required(a, "--emit-l0", emit_path, err) ||
        !arg_required(a, "--deploy", deploy_path, err) ||
        !arg_required(a, "--emit-l0-result", emit_result_path, err)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
    // A repeated --deploy is a sweep, which runs a CSP program: an L0 trace runs at L-B only, where
    // every deployment computes the same values. Refused rather than reading the first.
    if (std::count(a.begin(), a.end(), std::string("--deploy")) > 1) {
        std::cerr << "kpu-run: a repeated --deploy is a sweep, which runs a CSP program (file.csp); an L0 "
                     "trace runs at L-B on one deployment\n";
        return 2;
    }
    const bool do_step = has_flag(a, "--step");
    std::uint32_t step_limit = 40;
    if (!parse_dim(a, "--step-limit", 40, step_limit, err)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }

    // The level: L-B, the only one an L0 program runs at. Naming another is an error with the
    // reason, not a silent substitution.
    const std::string level_arg = arg(a, "--level", "all");
    if (level_arg != "all") {
        const auto parsed = parse_level(level_arg);
        if (!parsed) {
            std::cerr << "kpu-run: unknown --level '" << level_arg << "'\n";
            return 2;
        }
        if (!level_implemented(*parsed)) {
            std::cerr << "kpu-run: " << to_string(*parsed) << ": "
                      << not_implemented_reason(*parsed) << "\n";
            return 2;
        }
    }
    const ExecutionLevel level = ExecutionLevel::Behavioral;

    // The machine, from a spec file or the default device. L-B schedules on none of it, but a
    // run's identity names its deployment, and a multi-device one is refused by the platform.
    std::optional<platform::DeploymentSpec> deployment;
    try {
        deployment.emplace(deploy_path.empty() ? make_deployment(DeviceSpec{})
                                               : platform::read_spec_file(deploy_path));
    } catch (const std::exception& e) {
        std::cerr << "kpu-run: " << e.what() << "\n";
        return 2;
    }

    std::cout << "program  " << program_path << "\n";
    // What the FILE turned out to carry, stated rather than assumed: a test case or a kernel
    // filled here.
    std::cout << "         \"" << loaded->name() << "\"  "
              << loaded->operand_order().size() << " operands, "
              << loaded->ops().size() << " ops   values: "
              << (info.has_values ? "from file" : "synthesized (--fill-inputs)")
              << (dataflow.empty() ? "" : "   streams: " + dataflow + " (carried, not run)")
              << "\n";
    std::cout << "device   " << deployment->label()
              << "   digest " << platform::deployment_digest(*deployment) << "\n";
    std::cout << "levels   " << short_name(level);
    // Say out loud what is NOT being run, so a clean report is not mistaken for
    // full coverage.
    for (ExecutionLevel l : all_levels())
        if (!level_implemented(l))
            std::cout << "\n         (" << short_name(l) << " not run: "
                      << not_implemented_reason(l) << ")";
    // ...and what of the MACHINE is not being modelled, on one line.
    if (const auto fields = unmodelled(level, *deployment); !fields.empty()) {
        std::cout << "\n         (" << short_name(level) << " ignores: ";
        for (std::size_t i = 0; i < fields.size(); ++i)
            std::cout << (i ? ", " : "") << platform::to_string(fields[i]);
        std::cout << ")";
    }
    std::cout << "\n\n";

    // One place decides what an emitted file records, so the two emit paths cannot drift.
    auto emit_options = [&]() {
        serialize::WriteOptions opt;
        opt.include_values = true;
        if (!dataflow.empty()) opt.dataflow = dataflow;
        return opt;
    };
    auto write_file = [&](const std::string& path, const TileProgram& prog, const char* what) -> int {
        // BINARY, so the bytes do not depend on the platform that wrote them: a text-mode
        // stream on Windows would translate every \n into \r\n, and a format with a
        // byte-stability check cannot have a platform-dependent encoding.
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            std::cerr << "kpu-run: cannot write '" << path << "'\n";
            return 2;
        }
        serialize::write_l0(out, prog, emit_options());
        out.close();
        if (!out) {
            std::cerr << "kpu-run: failed while writing '" << path << "'\n";
            return 2;
        }
        std::cout << "wrote  " << path << "  (" << what << ")\n";
        return 0;
    };

    // --emit-l0 writes the program BEFORE it executes, so the file is an input rather than a
    // snapshot of a finished run.
    if (!emit_path.empty())
        if (const int rc = write_file(emit_path, *loaded, "test case, inputs inline")) return rc;

    // EVERY RUN GOES THROUGH THE PLATFORM (#282 increment 4): it takes the initial state as an
    // argument, so the run identity is complete.
    std::optional<platform::VirtualPlatform> vp;
    platform::ProgramHandle handle;
    std::optional<platform::StateSnapshot> initial;
    TileProgram result;
    try {
        vp.emplace(*deployment);
        handle = vp->load_program(*loaded);
        initial.emplace(vp->snapshot());
        const auto r = vp->run(handle, level, *initial);
        result = vp->program(handle);
        std::cout << "run id   " << r.identity.str() << "\n\n";
        print_run(r.outcome);
    } catch (const std::exception& e) {
        std::cerr << "kpu-run: " << e.what() << "\n";
        return 2;
    }

    // --step: walk the program at L-B, THROUGH THE PLATFORM (#282 increment 5). step_begin()
    // restores the snapshot first, so a stepped walk has the same identity a run does, and
    // each step EXECUTES one op.
    if (do_step) {
        std::optional<platform::VirtualPlatform::Cursor> cur;
        try {
            cur.emplace(vp->step_begin(handle, level, *initial));
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: " << e.what() << "\n";
            return 2;
        }
        std::cout << "\nstepping " << short_name(level) << ": " << cur->size()
                  << " steps (re-executed one op at a time)"
                  << (step_limit && cur->size() > step_limit
                          ? ", showing the first " + std::to_string(step_limit)
                          : "")
                  << "\n";
        std::size_t shown = 0;
        // The limit is checked BEFORE advancing, because a step is not free: step() APPLIES
        // the next op, so testing afterwards executed one more op than it showed.
        while (!step_limit || shown < step_limit) {
            if (!cur->step()) break;
            std::cout << "  " << std::setw(5) << std::right << cur->position() << "  "
                      << std::left << describe(cur->current(), false) << "\n";
            ++shown;
        }
        // Report where it stopped, so "stop after n steps" is checkable from outside.
        std::cout << "  stopped after " << cur->position() << " of " << cur->size()
                  << " steps\n";
    }

    // An L0 program is the oracle: its L-B values are the reference other levels answer to, so
    // there is nothing here to compare them against, and the result is written as it stands.
    std::cout << "\nL-B ran the oracle; an L0 program has no other level to compare with\n";
    if (!emit_result_path.empty())
        if (const int rc = write_file(emit_result_path, result, "results, from L-B")) return rc;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> a(argv + 1, argv + argc);
    if (has_flag(a, "--help") || has_flag(a, "-h")) { usage(); return 0; }

    // kpu-run is the simulator: it runs a program, and does not write one
    // (kpu-run-csp-programs step 5). The derivation flags are refused by name, with where
    // their job went.
    for (const char* k : {"--algo", "--size", "--tile", "--act"})
        if (arg_present(a, k)) {
            std::cerr << "kpu-run: " << k << " left kpu-run (kpu-run-csp-programs step 5): kpu-run runs a "
                         "program and does not write one. Write it with kpu-csp-gen, then run it:\n"
                         "  kpu-csp-gen --algo matmul --size 64 --tile 16 --l3 16 -o matmul.csp\n"
                         "  kpu-run --program matmul.csp\n";
            return 2;
        }
    std::string err, program_path;
    if (!arg_required(a, "--program", program_path, err)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
    if (program_path.empty()) {
        std::cerr << "kpu-run: --program is required: a CSP program (file.csp; write one with "
                     "kpu-csp-gen), or an L0 trace (file.l0), which runs at L-B as the oracle\n";
        return 2;
    }
    if (ends_with(program_path, ".csp")) return run_csp_program(a, program_path);
    return run_l0_program(a, program_path);
}
