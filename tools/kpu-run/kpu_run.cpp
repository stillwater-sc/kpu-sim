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
#include <sw/kpu/program/record/tile_flow_record.hpp>
#include <sw/trace/trace_exporter.hpp>

#include <cmath>
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
                            its machine from it; without it L-CA is skipped
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
  --tflow <dir>             the L-T1 run's tile-flow record (.tflow version 4)
  --timeline <file.json>    the L-T1 run as Chrome-trace intervals, one per leg
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

int run_csp_program(const std::vector<std::string>& a, const std::string& path) {
    // What a CSP program run accepts. Everything else is the L0 path's (or the generator's:
    // --algo and friends are kpu-csp-gen's), and is refused by name rather than ignored.
    static const std::set<std::string> kOptions = {"--program", "--deploy", "--level", "--window", "--inputs",
                                                   "--trace-limit", "--no-compare", "--emit-l0-result",
                                                   "--emit-l0", "--emit-l0-kernel",
                                                   "--tflow", "--step", "--step-limit", "--timeline"};
    for (const std::string& t : a)
        if (t.rfind("--", 0) == 0 && !kOptions.count(t)) {
            std::cerr << "kpu-run: " << t << " does not apply to a CSP program"
                      << (t == "--algo" || t == "--size" || t == "--tile" || t == "--act"
                              ? " (the program already says what it computes; kpu-csp-gen writes programs)"
                              : "")
                      << "\n";
            return 2;
        }
    std::string err, deploy_path, inputs_path, emit_path, tflow_path, timeline_path, trace_path, kernel_path;
    if (!arg_required(a, "--deploy", deploy_path, err) || !arg_required(a, "--inputs", inputs_path, err) ||
        !arg_required(a, "--emit-l0-result", emit_path, err) || !arg_required(a, "--tflow", tflow_path, err) ||
        !arg_required(a, "--timeline", timeline_path, err) || !arg_required(a, "--emit-l0", trace_path, err) ||
        !arg_required(a, "--emit-l0-kernel", kernel_path, err)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
    std::uint32_t window = 256, trace_limit = 1000000, step_limit = 40;
    if (!parse_dim(a, "--window", 256, window, err) || !parse_dim(a, "--trace-limit", 1000000, trace_limit, err) ||
        !parse_dim(a, "--step-limit", 40, step_limit, err)) {
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
        if (const auto why = csp_level_supported(*parsed, *ast, device)) {
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
                t.values = v.values;
            }
            inputs_note = "from " + inputs_path;
        }
    } catch (const std::exception& e) {
        std::cerr << "kpu-run: " << e.what() << "\n";
        return 2;
    }

    // THROUGH THE PLATFORM, the one execution path (kpu-run-csp-programs step 4a). Without
    // --deploy the platform holds the default device, and L-CA -- which builds its machine from
    // a deployment -- is skipped with that reason.
    platform::DeploymentSpec spec = deployment ? *deployment : make_deployment(DeviceSpec{});
    std::optional<platform::VirtualPlatform> vp;
    platform::CspProgramHandle handle;
    csp::lang::Validation validation;
    driver::CspReference reference;
    std::vector<CspLevelOutcome> outcomes;
    std::vector<std::string> run_ids;
    try {
        vp.emplace(spec);
        const csp::lang::Target target = csp::lang::target_from(spec.device(0));
        validation = csp::lang::validate(*ast, deployment ? &target : nullptr);
        handle = vp->load_csp(*ast, inputs, deployment.has_value());
        reference = vp->csp_reference(handle, trace_limit);
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
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "kpu-run: " << e.what() << "\n";
        return 2;
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
    std::cout << "\n";
    // L-B is the oracle; its trace, replayed, must reproduce it before it can be written.
    std::cout << "oracle   L-B, " << res.actions << " actions; "
              << (reference.trace ? std::string("L0 trace of ") + std::to_string(reference.trace->ops().size()) +
                                        " ops, replayed bit-exact"
                                  : res.reference_note) << "\n";
    std::cout << "device   " << (deployment ? deployment->label() + "   digest " + platform::deployment_digest(*deployment)
                                            : std::string("none (no --deploy): the default device")) << "\n\n";

    for (std::size_t li = 0; li < res.levels.size(); ++li) {
        const auto& o = res.levels[li];
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
            std::cout << "        L3    peak " << o.peak_l3 << " of " << ast->l3 << " (in time), credit stalls "
                      << o.credit_stalls << ", slot-bound " << o.slot_bound << " (" << o.slot_wait_cycles
                      << " cycles)\n        busy ";
            for (const auto& [p, b] : o.busy)
                std::cout << " " << p << " " << b << "/" << (o.lanes.count(p) ? o.lanes.at(p) : 0);
            std::cout << "   (lane-cycles/lanes)   UNCALIBRATED\n";
            for (const auto& u : o.unmodelled) std::cout << "        unmodelled: " << u << "\n";
            continue;
        }
        std::cout << "        VE    str " << o.ve.str_busy << "/" << o.ve.str_bound << "  bm " << o.ve.bm_busy << "/"
                  << o.ve.bm_bound << "  fabric " << o.ve.fabric << "   (busy/bound)   window " << window << "\n";
        for (const auto& u : o.unmodelled) std::cout << "        unmodelled: " << u << "\n";
    }

    // --tflow: the tile-flow record (version 4) of the program's L-T1 run (step 4b).
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
                      << " computes from L-T1; ops are the program's actions, version 4)\n";
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
    // Every operand, tile by tile, within the proximity bound of L-B's: the levels refine the
    // behavioral model, and a data path may round differently (§3.8).
    const TileProgram* authority = &*res.reference;
    std::cout << "\nvalues vs L-B, the oracle (every operand, per tile, float32 within 1e-6 + 1e-4|o|)\n";
    bool all_agree = true;
    for (const auto& o : res.levels) {
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
        TileProgram result = *authority;
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
