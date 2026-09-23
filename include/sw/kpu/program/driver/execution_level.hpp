// ============================================================================
// include/sw/kpu/program/driver/execution_level.hpp
// The four levels of transaction granularity, and the ONE place that maps a
// level to an interpreter.
//
// ADR 0002: CSP is the program layer, not a fidelity. One program is executed by
// every level; what differs is how finely the level decomposes a transaction.
// Values are level-invariant — decomposition changes WHEN things happen, never
// WHAT is computed — which is what makes a disagreement between two levels a bug
// signal by construction rather than a judgement call.
//
// run_at() is deliberately the only function that names a level (design note
// §D2). VirtualPlatform (#282) takes this seam over, and the L-T2 interpreter
// (#283) adds a case here; nothing else in the driver learns an executor type.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/characterize/device_model.hpp>
#include <sw/kpu/program/placement.hpp>
#include <sw/kpu/program/stream/stream_signature.hpp>
#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/program/tile_transaction_executor.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::driver {

// ----------------------------------------------------------------------------
// The levels, in increasing decomposition of one transaction.
// ----------------------------------------------------------------------------
enum class ExecutionLevel {
    Behavioral,              // L-B:  a whole block move, atomic
    BlockSequential,         // L-T1: one tile move, under credits and capacity
    ResourceTransactional,   // L-T2: read/write per resource, push into the fabric
    CycleAccurate,           // L-CA: protocol events, per cycle
};

inline const char* to_string(ExecutionLevel l) {
    switch (l) {
        case ExecutionLevel::Behavioral:            return "behavioral";
        case ExecutionLevel::BlockSequential:       return "block-sequential";
        case ExecutionLevel::ResourceTransactional: return "resource-transactional";
        case ExecutionLevel::CycleAccurate:         return "cycle-accurate";
    }
    return "?";
}

inline const char* short_name(ExecutionLevel l) {
    switch (l) {
        case ExecutionLevel::Behavioral:            return "L-B";
        case ExecutionLevel::BlockSequential:       return "L-T1";
        case ExecutionLevel::ResourceTransactional: return "L-T2";
        case ExecutionLevel::CycleAccurate:         return "L-CA";
    }
    return "?";
}

// Accepts the long name, the short name, or an unambiguous alias.
inline std::optional<ExecutionLevel> parse_level(const std::string& s) {
    if (s == "behavioral" || s == "L-B" || s == "l-b" || s == "lb")
        return ExecutionLevel::Behavioral;
    if (s == "block-sequential" || s == "L-T1" || s == "l-t1" || s == "lt1")
        return ExecutionLevel::BlockSequential;
    if (s == "resource-transactional" || s == "L-T2" || s == "l-t2" || s == "lt2")
        return ExecutionLevel::ResourceTransactional;
    if (s == "cycle-accurate" || s == "L-CA" || s == "l-ca" || s == "lca")
        return ExecutionLevel::CycleAccurate;
    return std::nullopt;
}

inline std::vector<ExecutionLevel> all_levels() {
    return {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential,
            ExecutionLevel::ResourceTransactional, ExecutionLevel::CycleAccurate};
}

// Which levels have an interpreter today. Stated as data rather than as a comment
// so a caller can enumerate what it may run instead of discovering it by failing.
inline bool level_implemented(ExecutionLevel l) {
    return l == ExecutionLevel::Behavioral || l == ExecutionLevel::BlockSequential;
}

// Why a level is missing, and where to follow it up. A driver that silently fell
// back to a level that works would answer a different question than it was asked,
// which is worse than refusing.
inline std::string not_implemented_reason(ExecutionLevel l) {
    switch (l) {
        case ExecutionLevel::ResourceTransactional:
            return "the L-T2 resource-transactional interpreter is not implemented (#283)";
        case ExecutionLevel::CycleAccurate:
            return "L-CA exists in substance (ConcurrentTimingExecutor) but consumes a "
                   "ScheduleResult rather than this program, so it cannot run it yet (#283)";
        default:
            return "";
    }
}

// ----------------------------------------------------------------------------
// One run's outcome, level-agnostic.
// ----------------------------------------------------------------------------
struct RunOutcome {
    ExecutionLevel level{};
    bool has_timing = false;         // L-B computes values but models no time
    Cycle makespan = 0;
    double lower_bound = 0.0;
    std::size_t ops = 0;
    TileProgramReference::RunSummary summary{};
    // Present only for levels that model resources.
    std::optional<TileRunStats> stats;
    std::optional<TileRunProvenance> provenance;
    // Per-op records, including each movement op's per-hop intervals. Empty at L-B,
    // which models no time and therefore has no intervals to report.
    std::vector<TileOpRecord> timeline;
};

// ----------------------------------------------------------------------------
// Execute `prog` at `level`, in place. The program carries its own operand
// values, so the caller reads results back out of `prog` afterwards — which is
// what makes a value comparison between two levels a comparison of two programs.
//
// Throws std::invalid_argument for a level with no interpreter. It does not fall
// back.
// ----------------------------------------------------------------------------
inline RunOutcome run_at(ExecutionLevel level, TileProgram& prog,
                         const characterize::DeviceDescriptor& device,
                         const Placement& placement,
                         const stream::StreamProgram* streams = nullptr,
                         std::uint64_t seed = 0) {
    RunOutcome out;
    out.level = level;

    switch (level) {
        case ExecutionLevel::Behavioral: {
            // L-B applies every op in program order and models no time at all.
            // Reporting makespan 0 here would read as "instant"; has_timing says
            // "not modelled", which is a different claim.
            TileProgramReference ref;
            out.summary = ref.run(prog);
            out.ops = out.summary.ops;
            out.has_timing = false;
            return out;
        }
        case ExecutionLevel::BlockSequential: {
            TileTransactionExecutor exec;
            TileExecutionRequest req{prog, placement, device, streams, seed};
            const TileRunResult r = exec.run(req);
            out.summary = r.summary;
            out.ops = r.stats.ops;
            out.makespan = r.stats.makespan;
            out.lower_bound = r.stats.lower_bound;
            out.stats = r.stats;
            out.provenance = r.provenance;
            out.timeline = r.timeline;
            out.has_timing = true;
            return out;
        }
        case ExecutionLevel::ResourceTransactional:
        case ExecutionLevel::CycleAccurate:
            break;
    }
    throw std::invalid_argument(std::string("run_at: ") + to_string(level) + ": " +
                               not_implemented_reason(level));
}

} // namespace sw::kpu::program::driver
