// ============================================================================
// examples/program/run_levels.cpp
// One CSP program, run at every level, checked against L-B (#503).
//
// The program is the input (ADR 0002): a .csp file says what is resident in L3, the order tiles
// meet the fabric, and what is staged in L2. Every level executes that one program. They differ
// in how finely they decompose a transaction, never in what they compute:
//
//   L-B   the behavioral model: whole block moves, atomic. It computes the values, and it is the
//         ORACLE every other level is checked against.
//   L-T1  one tile move per transaction, under the program's own L3 credits.
//   L-T2  per-resource read/write. Not implemented yet (#283): reported as skipped.
//   L-CA  protocol events, per cycle, on the machine the deployment spec describes. It is the
//         timing reference.
//
// For each level this prints the makespan, then checks every operand, per tile, against L-B
// (sw/kpu/program/tile_check.hpp: |a - o| <= atol + rtol*|o| for fp32, exact for integers).
//
// With --placement it also shows the L3 placement: the walker gives every residency an L3
// address (module, slot, bank), and every level executes that address rather than allocating
// its own (ADR 0004 §5). Likewise each staging's L2 slot (`stage ... in l2`). The example checks
// that L-T1 and L-CA hold each tile in exactly the slot the walker assigned.
//
//   program_run_levels [--program <file.csp>] [--deploy <spec.json>] [--placement]
//
// Exit codes: 0 = every level that ran agrees with L-B (and, with --placement, executes the
// walker's slots); 1 = a level disagrees; 2 = bad arguments, program or spec.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/virtual_platform.hpp>
#include <sw/kpu/program/tile_check.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef KPU_SOURCE_DIR
#define KPU_SOURCE_DIR "."
#endif

using namespace sw::kpu::program;
using driver::CspLevelOutcome;
using driver::ExecutionLevel;

namespace {

// A residency, as "tile -> slot": what placement is compared on.
using Residencies = std::vector<std::pair<std::string, std::uint32_t>>;

Residencies sorted(Residencies p) {
    std::sort(p.begin(), p.end());
    return p;
}

void print_placement(const char* what, const char* noun, const Residencies& p, std::size_t limit = 6) {
    std::cout << "    " << std::left << std::setw(4) << what << p.size() << " " << noun << ":";
    for (std::size_t i = 0; i < p.size() && i < limit; ++i) std::cout << "  " << p[i].first << "@" << p[i].second;
    if (p.size() > limit) std::cout << "  ...";
    std::cout << "\n";
}

bool same(const char* level, const char* buffer, const Residencies& walker, const Residencies& executed) {
    const bool ok = sorted(walker) == sorted(executed);
    std::cout << "    " << level << " " << buffer << ": " << (ok ? "executes the walker's slots" : "DIFFERS from the walker")
              << "\n";
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string program_path = std::string(KPU_SOURCE_DIR) + "/tests/program/corpus/matmul_48x48x48_t16.csp";
    std::string deploy_path = std::string(KPU_SOURCE_DIR) + "/tests/program/deploy/kpu_s1.json";
    bool show_placement = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--program" && i + 1 < argc) program_path = argv[++i];
        else if (a == "--deploy" && i + 1 < argc) deploy_path = argv[++i];
        else if (a == "--placement") show_placement = true;
        else {
            std::cerr << "usage: program_run_levels [--program <file.csp>] [--deploy <spec.json>] [--placement]\n";
            return 2;
        }
    }

    // 1. The program and the machine. The .csp text parses to an AST; the deployment spec
    //    describes the device L-CA builds its machine from.
    std::ifstream in(program_path);
    if (!in) {
        std::cerr << "cannot read " << program_path << "\n";
        return 2;
    }
    std::ostringstream text;
    text << in.rdbuf();
    std::optional<csp::lang::Program> ast;
    std::optional<platform::DeploymentSpec> spec;
    std::optional<platform::VirtualPlatform> vp;
    platform::CspProgramHandle handle;
    driver::CspReference reference;
    try {
        ast.emplace(csp::lang::parse(text.str()));
        spec.emplace(platform::read_spec_file(deploy_path));
        vp.emplace(*spec);

        // 2. Load it on the virtual platform, with inputs synthesized from the operand names.
        handle = vp->load_csp(*ast, driver::csp_inputs(*ast));

        // 3. The oracle: L-B's run. Its trace (the program's L0) is replayed to validate it, so
        //    a replay mismatch is a simulator defect, never a property of the program.
        reference = vp->csp_reference(handle);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    if (!reference.values) {
        std::cerr << "L-B produced no values: " << reference.note << "\n";
        return 1;
    }
    if (!reference.replay_mismatch.empty()) {
        std::cerr << "L-B's trace does not replay: " << reference.replay_mismatch << "\n";
        return 1;
    }
    std::cout << "program  " << program_path << "\n"
              << "deploy   " << deploy_path << "\n"
              << "actions  " << reference.actions << "\n\n";

    // 4. Every level, timed, then checked per tile against the oracle.
    bool ok = true;
    std::vector<CspLevelOutcome> outcomes;
    for (ExecutionLevel level : driver::all_levels()) {
        CspLevelOutcome o = vp->run_csp(handle, level).outcome;
        std::cout << std::left << std::setw(24) << driver::to_string(level);
        if (o.skipped) {
            std::cout << "skipped: " << *o.skipped << "\n";
            continue;
        }
        if (o.has_timing) std::cout << "makespan " << std::setw(8) << o.makespan;
        else std::cout << std::setw(17) << "(values only)";
        const OperandsCheck c = check_operands(o.values, *reference.values);
        std::cout << (c.pass ? "  values within bound of L-B" : "  VALUES DIFFER from L-B") << " (max abs "
                  << c.max_abs << ")\n";
        if (!c.pass) {
            ok = false;
            for (const TileCheck& t : c.operands)
                if (!t.pass) std::cout << "      " << t.summary() << "\n";
        }
        outcomes.push_back(std::move(o));
    }

    // 5. L3 placement and L2 staging, as the walker assigns them and as each level executes them.
    if (show_placement) {
        Residencies walker_l3, walker_l2, lt1_l3, lt1_l2, lca_l3, lca_l2;
        csp::lang::ActionStream stream(*ast, csp::lang::target_from(spec->device(0)));
        while (auto e = stream.next()) {
            const csp::Action& a = e->action;
            if (e->opens && a.place) walker_l3.emplace_back(tile_key(a.tile), a.place->slot);
            // A staging the program states (`stage ... in l2`). The per-call copies the walker
            // also places are not yet the timed levels' (they run unplaced there), so they are
            // left out (docs/plans/slot-credits-at-lca.md §4).
            if (e->opens_staging && a.l2 && !a.copy) walker_l2.emplace_back(tile_key(a.tile), a.l2->slot);
        }
        for (const CspLevelOutcome& o : outcomes) {
            if (o.level == ExecutionLevel::BlockSequential) {
                for (const auto& s : o.slots) lt1_l3.emplace_back(tile_key(s.tile), s.place.slot);
                for (const auto& s : o.l2_slots) lt1_l2.emplace_back(tile_key(s.tile), s.place.slot);
            }
            if (o.level == ExecutionLevel::CycleAccurate) {
                for (const auto& [t, slot] : o.l3_placement) lca_l3.emplace_back(tile_key(t), slot);
                for (const auto& [t, slot] : o.l2_placement) lca_l2.emplace_back(tile_key(t), slot);
            }
        }
        std::cout << "\nplacement (tile@slot, in the walker's order)\n";
        print_placement("L3", "residencies", walker_l3);
        print_placement("L2", "stagings", walker_l2);
        for (const CspLevelOutcome& o : outcomes) {
            if (o.level == ExecutionLevel::BlockSequential) {
                ok = same("L-T1", "L3", walker_l3, lt1_l3) && ok;
                ok = same("L-T1", "L2", walker_l2, lt1_l2) && ok;
            }
            if (o.level == ExecutionLevel::CycleAccurate) {
                ok = same("L-CA", "L3", walker_l3, lca_l3) && ok;
                ok = same("L-CA", "L2", walker_l2, lca_l2) && ok;
            }
        }
    }

    std::cout << "\n" << (ok ? "OK: every level that ran computes L-B's values" : "FAILED")
              << (ok && show_placement ? ", in the walker's slots" : "") << "\n";
    return ok ? 0 : 1;
}
