// ============================================================================
// include/sw/kpu/program/driver/csp_run.hpp
// Run a CSP program at each level (docs/plans/kpu-run-csp-programs.md step 2).
//
// The program is the input: it configures the data path -- what is resident, the order tiles
// meet the fabric, what is fused on a move -- and every level executes it.
//
//   L-B   csp::BehavioralInterpreter, from the program's stream
//   L-T1  csp::TransactionalInterpreter: one tile move per transaction, under the program's
//         own L3 credits (its Loads and Releases), timed per process from the device
//   L-CA  timing::CspDriver, from the stream, on the machine csp_config_from(device) builds
//
// L0 is the ORACLE, not the input: when the program is small enough to trace, its trace's
// L0 ops (calls, and the arithmetic of its tile contexts) run by TileProgramReference on the
// same inputs give the values every level must equal, bit for bit, on every operand.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/csp_driver.hpp>

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::program::driver {

// The program's declared operands, with values: every operand the program READS before it
// writes it is filled deterministically by name (fill_operand); the rest are zero.
//
// `in` is read; `out` is written. An `inout` operand can be either, and the difference matters
// to the oracle: an accumulator starts from zero in the fabric (`acc ... in fabric`), but L0's
// MatMulAccum adds onto its operand's buffer, so a result tile the program writes first --
// stored from an accumulator, then read back (the unfused linear epilogue) -- must start at
// zero for the trace's L0 to compute what the program computes. LU's A, loaded first, is an
// input. The first touch decides, found by streaming until every `inout` operand has been
// touched (it ends early: operands are touched in the first actions of any sane program).
inline TileProgram csp_inputs(const csp::lang::Program& ast) {
    TileProgram t = csp::lang::declared_operands(ast);
    std::map<std::string, bool> read_first;             // inout operand -> its first touch is a read
    std::size_t open = 0;
    for (const auto& d : ast.decls)
        if (d.io == "inout") ++open;
    if (open > 0) {
        csp::lang::ActionStream s(ast);
        while (open > 0) {
            auto e = s.next();
            if (!e) break;
            const auto& act = e->action;
            const std::string& name = act.tile.operand;
            auto decl = std::find_if(ast.decls.begin(), ast.decls.end(), [&](const auto& d) { return d.name == name; });
            if (decl == ast.decls.end() || decl->io != "inout" || read_first.count(name)) continue;
            if (act.kind == csp::Action::Kind::Load) read_first[name] = true;
            else if (act.kind == csp::Action::Kind::Drain || act.kind == csp::Action::Kind::Writeback)
                read_first[name] = false;
            else continue;
            --open;
        }
    }
    TileProgram out(t.name());
    for (const auto& d : ast.decls) {
        const TensorOperand& shape = t.operand(d.name);
        TensorOperand& o = out.add_operand(TensorOperand(shape.name, shape.rows, shape.cols, shape.tile_rows,
                                                         shape.tile_cols));
        const bool reads = d.io == "in" || (d.io == "inout" && (!read_first.count(d.name) || read_first.at(d.name)));
        if (reads) fill_operand(o);
    }
    return out;
}

// Why `level` cannot run this program on this machine, or nullopt when it can. `device` may be
// null: a run with no deployment.
inline std::optional<std::string> csp_level_supported(ExecutionLevel level, const csp::lang::Program& ast,
                                                      const platform::DeviceSpecification* device) {
    switch (level) {
        case ExecutionLevel::Behavioral:
            return std::nullopt;
        case ExecutionLevel::BlockSequential:
            return std::nullopt;
        case ExecutionLevel::ResourceTransactional:
            return not_implemented_reason(level);
        case ExecutionLevel::CycleAccurate:
            break;
    }
    if (!device) return std::string("L-CA builds its machine from a deployment spec (--deploy)");
    // The executor names three matrices (A, B, C) and a vector operand by its tag; a program
    // with other matrix names is not yet mapped onto them.
    for (const auto& d : ast.decls)
        if (!d.is_vector && d.name != "A" && d.name != "B" && d.name != "C")
            return "L-CA maps a program's matrices onto the executor's A, B and C; '" + d.name +
                   "' is not one of them (csp line " + std::to_string(d.line) + ")";
    // The tile functions L-CA computes: gemm (alpha 1) and the epilogue.
    std::optional<std::string> refused;
    auto scan = [&](const std::vector<csp::lang::Stmt>& body, auto&& self) -> void {
        for (const auto& s : body) {
            if (refused) return;
            if (s.kind == csp::lang::Stmt::Kind::Call) {
                if (s.fn == "getrf" || s.fn == "laswp" || s.fn == "trsm_ll" || s.fn == "trsm_ur")
                    refused = "L-CA runs matmul and the linear operator; " + s.fn + " (LU) at L-CA is a later "
                              "increment (csp line " + std::to_string(s.line) + ")";
                else if (s.alpha && *s.alpha != 1.0)
                    refused = "the fabric's matmul computes alpha 1 (csp line " + std::to_string(s.line) + ")";
            }
            self(s.body, self);
        }
    };
    scan(ast.body, scan);
    if (refused) return refused;
    std::string why;
    const auto cfg = timing::csp_config_from(*device, &why);
    if (!cfg) return why;
    if (cfg->config.l3_tiles != 1)
        return "the machine has " + std::to_string(cfg->config.l3_tiles) +
               " L3 tiles; a level-1 program addresses one (level 2 distributes)";
    if (static_cast<std::size_t>(ast.l3) > device->l3.capacity_tiles)
        return "the program was written for an L3 of " + std::to_string(ast.l3) + " tiles; the machine's holds " +
               std::to_string(device->l3.capacity_tiles);
    return std::nullopt;
}

struct CspLevelOutcome {
    ExecutionLevel level = ExecutionLevel::Behavioral;
    std::optional<std::string> skipped;     // why the level did not run
    TileProgram values;                     // every operand, as DRAM holds it at the end
    bool has_timing = false;
    std::uint64_t makespan = 0;
    std::size_t actions = 0;
    std::size_t peak_l3 = 0;                // L-B: as executed; L-T1: slots held at once, in time
    std::size_t credit_stalls = 0;          // L-T1: Loads that waited for a credit
    std::map<std::string, std::uint64_t> busy;      // L-T1: lane-cycles per process
    std::map<std::string, std::size_t> lanes;
    // L-T1: every leg of every action, every residency's slot, and the compute tiles it was
    // timed on -- what the tile-flow record (record::build_csp_record) is built from.
    std::vector<csp::TransactionalInterpreter::Record> records;
    std::vector<csp::TransactionalInterpreter::Slot> slots;
    std::size_t compute_tiles = 0;
    std::uint64_t dram_loads = 0, dram_stores = 0, dram_bytes = 0;
    std::uint64_t cf_busy = 0;
    timing::ConcurrentTimingExecutor::VectorStats ve;
    bool livelock = false;
    std::vector<std::string> unmodelled;    // spec fields the level keeps a default for
};

struct CspRunRequest {
    const csp::lang::Program& ast;
    const platform::DeviceSpecification* device = nullptr;
    TileProgram inputs;                                 // csp_inputs(ast), or supplied values
    std::vector<ExecutionLevel> levels;
    std::size_t window = 256;                           // L-CA's driver window (a schedule option)
    std::size_t trace_limit = 1'000'000;                // the oracle: trace at most this many actions
};

struct CspRunResult {
    csp::lang::Validation validation;
    std::optional<TileProgram> reference;               // the trace's L0, run on the inputs
    std::string reference_note;                         // why there is none
    std::size_t actions = 0;                            // the program's actions (when counted)
    std::vector<CspLevelOutcome> levels;
};

// The oracle: the trace's L0, run on `inputs`, when the program has at most `trace_limit`
// actions (counted without keeping them).
struct CspReference {
    std::optional<TileProgram> values;      // every operand, as the program leaves them
    std::string note;                       // why there is none
    std::size_t actions = 0;
};

inline CspReference csp_reference(const csp::lang::Program& ast, const TileProgram& inputs, std::size_t trace_limit) {
    CspReference out;
    csp::lang::ActionStream s(ast);
    std::size_t n = 0;
    while (n <= trace_limit && s.next()) ++n;
    if (n > trace_limit) {
        out.note = "more than " + std::to_string(trace_limit) +
                   " actions: no L0 reference (raise --trace-limit); the levels are compared with each other";
        return out;
    }
    out.actions = n;
    csp::CspProgram trace = csp::lang::compile(ast);
    for (const auto& name : trace.source.operand_order())
        trace.source.operand(name).values = inputs.operand(name).values;
    TileProgramReference().run(trace.source);
    TileProgram ref = inputs;
    for (const auto& name : ref.operand_order()) ref.operand(name).values = trace.source.operand(name).values;
    out.values = std::move(ref);
    return out;
}

// One level's run of the program on `inputs` (skipped, with the reason, when it cannot run).
inline CspLevelOutcome csp_run_level(ExecutionLevel level, const csp::lang::Program& ast,
                                     const platform::DeviceSpecification* device, const TileProgram& inputs,
                                     std::size_t window = 256) {
    CspLevelOutcome o;
    o.level = level;
    o.skipped = csp_level_supported(level, ast, device);
    if (o.skipped) return o;
    std::optional<csp::lang::Target> target;
    if (device) target = csp::lang::target_from(*device);
    if (level == ExecutionLevel::Behavioral) {
        csp::lang::ActionStream s(ast);
        csp::BehavioralInterpreter lb;
        lb.begin(inputs);
        while (auto e = s.next()) lb.step(e->action, e->op ? &*e->op : nullptr);
        const auto sum = lb.finish();
        o.values = lb.result();
        o.actions = sum.actions;
        o.peak_l3 = sum.peak_l3;
    } else if (level == ExecutionLevel::BlockSequential) {
        // The device's lanes and rates; with no deployment, the default single-site device.
        characterize::DeviceDescriptor dev = characterize::DeviceDescriptor::single();
        if (device) {
            platform::DeploymentSpec one;
            one.devices = {*device};
            dev = one.device_view(0);
        }
        csp::lang::ActionStream s(ast);
        csp::TransactionalInterpreter lt(dev, static_cast<std::size_t>(ast.l3));
        lt.begin(inputs);
        while (auto e = s.next()) lt.step(e->action, e->op ? &*e->op : nullptr);
        const auto st = lt.finish();
        o.values = lt.result();
        o.has_timing = true;
        o.makespan = st.makespan;
        o.actions = st.actions;
        o.peak_l3 = st.peak_l3;
        o.dram_loads = st.dram_loads;
        o.dram_stores = st.dram_stores;
        o.dram_bytes = st.dram_bytes;
        o.credit_stalls = st.credit_stalls;
        o.busy = st.busy;
        o.lanes = st.lanes;
        o.cf_busy = st.busy.count("cf") ? st.busy.at("cf") : 0;
        o.records = lt.records();
        o.slots = lt.slots();
        o.compute_tiles = lt.lanes_of(csp::TransactionalInterpreter::Proc::Cf);
        o.unmodelled.push_back("tile contexts' vector-unit time (movers.vector): L-T1 applies the stages' "
                               "values; their time is L-CA's");
    } else {                                        // CycleAccurate
        const auto cfg = timing::csp_config_from(*device);
        o.unmodelled = cfg->unmapped;
        timing::ConcurrentTimingExecutor exec(cfg->config);
        csp::lang::ActionStream s(ast, target);
        const auto r = timing::CspDriver(exec, s, inputs, window).run();
        if (!r.completed) {
            o.livelock = r.livelock;
            o.skipped = r.livelock ? std::string("L-CA livelocked at cycle ") + std::to_string(r.cycles)
                                   : std::string("L-CA did not complete in ") + std::to_string(r.cycles) +
                                         " cycles (max_cycles)";
            return o;
        }
        o.values = r.values;
        o.has_timing = true;
        o.makespan = r.cycles;
        o.actions = r.actions;
        o.dram_loads = r.dram_loads;
        o.dram_stores = r.dram_stores;
        o.dram_bytes = r.dram_bytes;
        o.cf_busy = r.cf_busy;
        o.ve = r.ve;
    }
    return o;
}

// Validate, build the oracle, and run each requested level. Throws CompileError for a program
// the validator refuses (against the device's sites when there is one).
inline CspRunResult run_csp(const CspRunRequest& req) {
    CspRunResult out;
    std::optional<csp::lang::Target> target;
    if (req.device) target = csp::lang::target_from(*req.device);
    out.validation = csp::lang::validate(req.ast, target ? &*target : nullptr);
    CspReference ref = csp_reference(req.ast, req.inputs, req.trace_limit);
    out.reference = std::move(ref.values);
    out.reference_note = std::move(ref.note);
    out.actions = ref.actions;
    for (ExecutionLevel level : req.levels)
        out.levels.push_back(csp_run_level(level, req.ast, req.device, req.inputs, req.window));
    return out;
}

}  // namespace sw::kpu::program::driver
