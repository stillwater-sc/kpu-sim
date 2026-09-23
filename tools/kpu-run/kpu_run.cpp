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
#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;

namespace {

void usage() {
    std::cout <<
R"(kpu-run — execute a Domain Flow Program at one or more levels and compare them.

  --algo <matmul|lu>        program to derive            (default matmul)
  --size <n>                square problem size          (default 64)
  --tile <n>                tile size                    (default 16)
  --level <name|all>        behavioral | block-sequential | resource-transactional
                            | cycle-accurate | all       (default all)
  --topology <t>            single | news | checkerboard  (default single)
  --compute-tiles <n>       compute fabric tiles         (default 1)
  --l3-tiles <n>            L3 capacity in tiles, 0=unbounded (default 0)
  --dma-engines <n>         DMA engines   (DRAM<->L3)    (default 1)
  --block-movers <n>        BlockMovers   (L3<->L2)      (default 1)
  --streamers <n>           Streamers     (L2<->L1)      (default 1)
  --no-compare              run the levels, do not diff values
  -h, --help

Values are compared bit-exactly against the behavioral level, which is the
authority (ADR 0001 D5). A disagreement exits 1.
)";
}

// Bit-exact, not a tolerance: L-B/L-T1/L-T2 must agree exactly, and a tolerance
// here would hide precisely the reordering bugs this comparison exists to find.
struct Diff {
    bool identical = true;
    std::size_t first_index = 0;
    float expected = 0.0f, actual = 0.0f;
    std::size_t differing = 0;
};

Diff compare_bitwise(const std::vector<float>& authority, const std::vector<float>& other) {
    Diff d;
    if (authority.size() != other.size()) {
        d.identical = false;
        d.differing = authority.size() > other.size() ? authority.size() - other.size()
                                                      : other.size() - authority.size();
        return d;
    }
    for (std::size_t i = 0; i < authority.size(); ++i) {
        if (std::memcmp(&authority[i], &other[i], sizeof(float)) != 0) {
            if (d.identical) {
                d.identical = false;
                d.first_index = i;
                d.expected = authority[i];
                d.actual = other[i];
            }
            ++d.differing;
        }
    }
    return d;
}

void print_run(const RunOutcome& o) {
    std::cout << "  " << std::left << std::setw(24) << to_string(o.level)
              << std::setw(6) << short_name(o.level);
    if (!o.has_timing) {
        std::cout << "timing: not modelled at this level";
    } else {
        std::cout << "makespan " << std::right << std::setw(8) << o.makespan
                  << "   floor " << std::setw(8) << std::fixed << std::setprecision(0)
                  << o.lower_bound;
    }
    std::cout << "\n";
    if (o.stats) {
        const auto& s = *o.stats;
        std::cout << "        ops " << s.ops << " (" << s.computes << " compute, "
                  << s.movements << " movement)"
                  << ", L3 peak " << s.peak_l3_residency
                  << ", credit stalls " << s.l3_credit_stalls
                  << ", reuse chains " << s.resident_feeds << "\n";
        for (const auto& kv : s.hop_busy_cycles)
            std::cout << "        " << std::left << std::setw(16) << to_string(kv.first)
                      << "busy " << std::right << std::setw(8) << kv.second
                      << "   transfers " << std::setw(6)
                      << (s.hop_transfers.count(kv.first) ? s.hop_transfers.at(kv.first) : 0)
                      << "\n";
        for (const auto& kv : s.mover_utilization)
            std::cout << "        [" << std::left << std::setw(13) << to_string(kv.first)
                      << "util " << std::fixed << std::setprecision(3) << kv.second
                      << "  lanes " << (s.mover_lanes.count(kv.first)
                                        ? s.mover_lanes.at(kv.first) : 0) << "]\n";
    }
    if (o.provenance)
        std::cout << "        device " << o.provenance->device
                  << "   placement " << o.provenance->placement
                  << (o.provenance->calibrated ? "   calibrated" : "   UNCALIBRATED")
                  << (o.provenance->extrapolated ? "   extrapolated" : "") << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> a(argv + 1, argv + argc);
    if (has_flag(a, "--help") || has_flag(a, "-h")) { usage(); return 0; }

    ProgramSpec ps;
    ps.algo = arg(a, "--algo", "matmul");
    ps.size = static_cast<Dim>(std::stoul(arg(a, "--size", "64")));
    ps.tile = static_cast<Dim>(std::stoul(arg(a, "--tile", "16")));
    if (!known_algo(ps.algo)) {
        std::cerr << "kpu-run: unknown --algo '" << ps.algo << "' (matmul | lu)\n";
        return 2;
    }
    if (ps.tile == 0 || ps.size == 0) {
        std::cerr << "kpu-run: --size and --tile must be non-zero\n";
        return 2;
    }

    DeviceSpec ds;
    ds.topology = arg(a, "--topology", "single");
    ds.compute_tiles = static_cast<Dim>(std::stoul(arg(a, "--compute-tiles", "1")));
    ds.l3_tiles = static_cast<Dim>(std::stoul(arg(a, "--l3-tiles", "0")));
    ds.dma_engines = static_cast<Dim>(std::stoul(arg(a, "--dma-engines", "1")));
    ds.block_movers = static_cast<Dim>(std::stoul(arg(a, "--block-movers", "1")));
    ds.streamers = static_cast<Dim>(std::stoul(arg(a, "--streamers", "1")));
    if (!known_topology(ds.topology)) {
        std::cerr << "kpu-run: unknown --topology '" << ds.topology
                  << "' (single | news | checkerboard)\n";
        return 2;
    }

    // Which levels to run. "all" means every level that HAS an interpreter; naming
    // one explicitly that does not is an error, not a silent substitution.
    const std::string level_arg = arg(a, "--level", "all");
    std::vector<ExecutionLevel> levels;
    if (level_arg == "all") {
        for (ExecutionLevel l : all_levels())
            if (level_implemented(l)) levels.push_back(l);
    } else {
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
        levels.push_back(*parsed);
    }

    const auto device = make_device(ds);
    const bool compare = !has_flag(a, "--no-compare");

    std::cout << "program  " << ps.label() << "\n"
              << "device   " << device.label() << "\n"
              << "levels   ";
    for (std::size_t i = 0; i < levels.size(); ++i)
        std::cout << (i ? ", " : "") << short_name(levels[i]);
    // Say out loud what is NOT being run, so a clean report is not mistaken for
    // full coverage.
    for (ExecutionLevel l : all_levels())
        if (!level_implemented(l))
            std::cout << "\n         (" << short_name(l) << " not run: "
                      << not_implemented_reason(l) << ")";
    std::cout << "\n\n";

    // Each level gets its OWN program, filled identically, so the comparison is of
    // the models rather than of leftover state.
    std::vector<TileProgram> programs;
    std::vector<RunOutcome> outcomes;
    programs.reserve(levels.size());
    for (ExecutionLevel l : levels) {
        programs.push_back(derive(ps));
        fill(programs.back(), ps);
        try {
            outcomes.push_back(run_at(l, programs.back(), device,
                                     Placement::single(device.compute_tiles)));
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: " << e.what() << "\n";
            return 2;
        }
        print_run(outcomes.back());
    }

    if (!compare || levels.size() < 2) {
        if (compare && levels.size() < 2)
            std::cout << "\nonly one level ran, so there is nothing to compare\n";
        return 0;
    }

    // L-B is the authority for values (ADR 0001 D5).
    std::size_t authority = 0;
    for (std::size_t i = 0; i < levels.size(); ++i)
        if (levels[i] == ExecutionLevel::Behavioral) authority = i;

    const char* operand = result_operand(ps);
    std::cout << "\nvalues vs " << short_name(levels[authority])
              << " (bit-exact, operand " << operand << ")\n";
    bool all_agree = true;
    for (std::size_t i = 0; i < levels.size(); ++i) {
        if (i == authority) continue;
        const Diff d = compare_bitwise(programs[authority].operand(operand).values,
                                       programs[i].operand(operand).values);
        std::cout << "  " << std::left << std::setw(24) << to_string(levels[i]);
        if (d.identical) {
            std::cout << "identical\n";
        } else {
            all_agree = false;
            std::cout << "DISAGREES: " << d.differing << " element(s), first at ["
                      << d.first_index << "] expected " << d.expected
                      << " got " << d.actual << "\n";
        }
        // LU carries a permutation and a swap count that a value diff would miss.
        if (std::string(operand) == "A") {
            if (outcomes[i].summary.row_swaps != outcomes[authority].summary.row_swaps) {
                all_agree = false;
                std::cout << "        DISAGREES on row swaps: "
                          << outcomes[authority].summary.row_swaps << " vs "
                          << outcomes[i].summary.row_swaps << "\n";
            }
            if (outcomes[i].summary.permutation != outcomes[authority].summary.permutation) {
                all_agree = false;
                std::cout << "        DISAGREES on the row permutation\n";
            }
        }
    }

    if (!all_agree) {
        std::cout << "\nFAILED: the levels do not compute the same values. Decomposition "
                     "changes WHEN, never WHAT (ADR 0002 §2), so this is a model bug.\n";
        return 1;
    }
    std::cout << "\nOK: every level computes identical values.\n";
    return 0;
}
