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
#include <sw/kpu/program/driver/timeline_trace.hpp>
#include <sw/trace/trace_exporter.hpp>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;

namespace {

// Bandwidths are doubles, and std::stod has the same hazards as std::stoul: it throws
// on junk, accepts trailing characters, and would abort the process rather than exit 2.
bool parse_rate(const std::vector<std::string>& a, const char* key, double fallback,
                double& out, std::string& error) {
    const std::string raw = arg(a, key, "");
    if (raw.empty()) { out = fallback; return true; }
    try {
        std::size_t consumed = 0;
        const double v = std::stod(raw, &consumed);
        if (consumed != raw.size()) {
            error = std::string(key) + ": '" + raw + "' has trailing characters";
            return false;
        }
        if (!(v > 0.0)) {
            error = std::string(key) + ": '" + raw + "' must be positive";
            return false;
        }
        out = v;
        return true;
    } catch (const std::exception&) {
        error = std::string(key) + ": '" + raw + "' is not a number";
        return false;
    }
}

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
  --noc-links <n>           NoC links     (L3->L3 reuse) (default 0 = none)
  --dma-bytes-per-cycle <b> per DMA engine               (default 64)
  --bm-bytes-per-cycle <b>  per BlockMover               (default 128)
  --str-bytes-per-cycle <b> per Streamer                 (default 256)
  --noc-bytes-per-cycle <b> per NoC link                 (default 128)
  --macs-per-cycle <m>      one compute tile's throughput (default 256)
  --streams <dataflow>      derive an L1 stream program: output-stationary|os,
                            weight-stationary|ws, a-stationary|as,
                            fully-streaming|hex  (matmul only)
  --timeline <file.json>    Chrome Trace Event Format, one event PER HOP
  --no-compare              run the levels, do not diff values
  -h, --help

Every *-bytes-per-cycle is PER LANE, never aggregate: lanes give concurrency, never
speed-up (design note §6.3).

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

    // Every numeric option goes through the checked parse: an uncaught std::stoul throw
    // would abort with SIGABRT, which a CI job cannot tell apart from a crash in the
    // model, and the usage contract above promises exit 2.
    std::string err;
    auto dim_opt = [&](const char* key, std::uint32_t fallback, Dim& out) {
        std::uint32_t v = 0;
        if (!parse_dim(a, key, fallback, v, err)) return false;
        out = static_cast<Dim>(v);
        return true;
    };

    ProgramSpec ps;
    ps.algo = arg(a, "--algo", "matmul");
    if (!dim_opt("--size", 64, ps.size) || !dim_opt("--tile", 16, ps.tile)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
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
    if (!dim_opt("--compute-tiles", 1, ds.compute_tiles) ||
        !dim_opt("--l3-tiles", 0, ds.l3_tiles) ||
        !dim_opt("--dma-engines", 1, ds.dma_engines) ||
        !dim_opt("--block-movers", 1, ds.block_movers) ||
        !dim_opt("--streamers", 1, ds.streamers)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
    if (!dim_opt("--noc-links", 0, ds.noc_links)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
    if (!parse_rate(a, "--dma-bytes-per-cycle", ds.dma_bytes_per_cycle,
                    ds.dma_bytes_per_cycle, err) ||
        !parse_rate(a, "--bm-bytes-per-cycle", ds.bm_bytes_per_cycle,
                    ds.bm_bytes_per_cycle, err) ||
        !parse_rate(a, "--str-bytes-per-cycle", ds.str_bytes_per_cycle,
                    ds.str_bytes_per_cycle, err) ||
        !parse_rate(a, "--noc-bytes-per-cycle", ds.noc_bytes_per_cycle,
                    ds.noc_bytes_per_cycle, err) ||
        !parse_rate(a, "--macs-per-cycle", ds.macs_per_cycle, ds.macs_per_cycle, err)) {
        std::cerr << "kpu-run: " << err << "\n";
        return 2;
    }
    if (ds.compute_tiles == 0) {
        std::cerr << "kpu-run: --compute-tiles must be non-zero\n";
        return 2;
    }

    // An L1 stream program is matmul-only, and saying so beats deriving an empty one and
    // reporting timing that silently ignored the flag.
    const std::string dataflow = arg(a, "--streams", "");
    if (!dataflow.empty()) {
        if (!known_dataflow(dataflow)) {
            std::cerr << "kpu-run: unknown --streams '" << dataflow
                      << "' (output-stationary|os, weight-stationary|ws, "
                         "a-stationary|as, fully-streaming|hex)\n";
            return 2;
        }
        if (ps.algo != "matmul") {
            std::cerr << "kpu-run: --streams is derived for matmul only, not '"
                      << ps.algo << "'\n";
            return 2;
        }
    }
    const std::string timeline_path = arg(a, "--timeline", "");
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
    std::vector<stream::StreamProgram> streams;
    programs.reserve(levels.size());
    streams.reserve(levels.size());
    for (ExecutionLevel l : levels) {
        try {
            // derive() and fill() throw too -- on an operand a spec does not declare, for
            // one -- so they belong inside the guard rather than beside it.
            programs.push_back(derive(ps));
            fill(programs.back(), ps);
            // The stream program is derived PER PROGRAM: it indexes ops of the program it
            // was derived from, so sharing one across levels would alias the wrong ops.
            streams.push_back(dataflow.empty()
                                  ? stream::StreamProgram{}
                                  : stream::derive_matmul_streams(programs.back(),
                                                                  map_for(dataflow)));
            outcomes.push_back(run_at(l, programs.back(), device,
                                     Placement::single(device.compute_tiles),
                                     dataflow.empty() ? nullptr : &streams.back()));
        } catch (const std::exception& e) {
            std::cerr << "kpu-run: " << e.what() << "\n";
            return 2;
        }
        print_run(outcomes.back());
    }

    // --timeline: one event per hop, from the level that models resources. L-B has no
    // intervals to report, so there is nothing to write from it.
    if (!timeline_path.empty()) {
        std::size_t src = levels.size();
        for (std::size_t i = 0; i < levels.size(); ++i)
            if (!outcomes[i].timeline.empty()) src = i;
        if (src == levels.size()) {
            std::cerr << "kpu-run: --timeline needs a level that models resources; "
                      << "L-B reports no intervals\n";
            return 2;
        }
        const auto entries = to_trace_entries(programs[src], outcomes[src].timeline,
                                              device.element_bytes);
        if (!sw::trace::ChromeTraceExporter::export_traces(timeline_path, entries)) {
            std::cerr << "kpu-run: could not write '" << timeline_path << "'\n";
            return 2;
        }
        std::cout << "\ntimeline  " << entries.size() << " events (one per hop) from "
                  << short_name(levels[src]) << " -> " << timeline_path << "\n";
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
            // print_run() leaves std::fixed and a small precision on the stream, which
            // would render a one-bit disagreement as "expected 1.234 got 1.234" -- the one
            // diagnostic line for a failure, hiding the failure. Hex float is exact.
            std::cout << "DISAGREES: " << d.differing << " element(s), first at ["
                      << d.first_index << "] expected "
                      << std::defaultfloat << std::setprecision(9) << d.expected
                      << " got " << d.actual
                      << "  (" << std::hexfloat << d.expected << " vs " << d.actual
                      << std::defaultfloat << ")\n";
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
