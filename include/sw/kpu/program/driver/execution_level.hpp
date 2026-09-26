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
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/stream/stream_signature.hpp>
#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/program/tile_transaction_executor.hpp>

#include <optional>
#include <set>
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
// What a level does NOT model, out of what the deployment declared
// ----------------------------------------------------------------------------
// A DeploymentSpec is a superset of what any level schedules on (§3.3 belongs to L-T2),
// so device_view() drops fields. Dropping them quietly is the failure this answers: a
// spec that says `"l3": {"banks": 8}` and a run that schedules as though L3 had no banks
// agree on nothing, and the report would not say so.
//
// It is the same statement as not_implemented_reason(), one layer down. That one keeps a
// clean report from being mistaken for full coverage across LEVELS; this one keeps it from
// being mistaken for full coverage of the MACHINE.
//
// The table says what the IMPLEMENTATION models, not what a level is supposed to model.
// L-T2 rows therefore read false until #283 makes each one true as it implements it —
// claiming otherwise would describe code that does not exist.
inline bool level_models(ExecutionLevel l, platform::SpecField f) {
    using platform::SpecField;
    switch (f) {
        case SpecField::L3Capacity:
            // Enforced by TileTransactionExecutor (#264 increment 4) and by the
            // cycle-accurate credit model; L-B has no buffers to bound.
            return l == ExecutionLevel::BlockSequential ||
                   l == ExecutionLevel::CycleAccurate;
        case SpecField::L3Tiles:
            // The MODULE count, which nothing schedules on yet -- it exists for the
            // naming map (#282 increment 3). Not the capacity: see DeviceSpecification::L3.
            return false;
        case SpecField::L3Banks:
        case SpecField::L2BanksPerTile:
        case SpecField::L1Vectors:
        case SpecField::DmaBurst:
            return false;                        // §3.3 resource vocabulary -- #283
    }
    return false;
}

// The fields a deployment DECLARED that this level does not model. The fields themselves,
// so a caller can render them however it needs -- one sentence each for a provenance record,
// or one compact line per level for a header -- without a second function deciding which
// fields those are. Two renderings of one list drift; two lists of one thing drift worse.
//
// Bandwidths are NOT listed at L-B even though it models no time at all: the outcome already
// states "timing: not modelled at this level", and repeating it per field would bury the
// resource-model fields this exists to surface.
inline std::vector<platform::SpecField> unmodelled(ExecutionLevel l,
                                                   const platform::DeploymentSpec& spec,
                                                   Dim device = 0) {
    std::vector<platform::SpecField> out;
    if (spec.devices.empty()) return out;
    const platform::DeviceSpecification& d = spec.device(device);
    for (platform::SpecField f : platform::all_spec_fields())
        if (platform::declared(d, f) && !level_models(l, f)) out.push_back(f);
    return out;
}

// One sentence per field, ready to print or to assert on.
inline std::vector<std::string> unmodelled_fields(ExecutionLevel l,
                                                  const platform::DeploymentSpec& spec,
                                                  Dim device = 0) {
    std::vector<std::string> out;
    if (spec.devices.empty()) return out;
    const platform::DeviceSpecification& d = spec.device(device);
    for (platform::SpecField f : unmodelled(l, spec, device))
        out.push_back(std::string(platform::to_string(f)) + " declared (" +
                      platform::declared_value(d, f) + ") but not modelled at " +
                      short_name(l));
    return out;
}

// ----------------------------------------------------------------------------
// One run's outcome, level-agnostic.
// ----------------------------------------------------------------------------
struct RunOutcome {
    ExecutionLevel level{};
    // Inputs this level could not represent, in its own words. Distinct from
    // `unmodelled_fields`, which is about what the DEPLOYMENT declared: this is about what the
    // CALLER passed. Both exist for the same reason -- an input that vanishes silently makes a
    // clean report mean less than it appears to.
    std::vector<std::string> unmodelled_inputs;
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
// Tiles already resident when the run starts, by `tile_key`. Threaded through rather than
// smuggled in, because it changes what a run does: a seeded tile's chain skips the DMA leg, so
// two runs differing only in this produce different makespans and different transfer counts.
// An input that changes the result belongs in the signature -- and, once the platform carries
// it, in the run identity.
//
// `retained_by_caller` is the other half of the same input and travels with it: tiles this run
// must leave resident, which the executor therefore may not release. A caller that seeds without
// retaining is claiming a residency the next run cannot rely on.
//
// `foreign_held_slots` is the third: slots the caller holds for some OTHER program, which this
// one cannot name and must not be allowed to use.
//
// Empty is the old behaviour and the default: a run that begins cold and keeps nothing.
inline RunOutcome run_at(ExecutionLevel level, TileProgram& prog,
                         const characterize::DeviceDescriptor& device,
                         const Placement& placement,
                         const stream::StreamProgram* streams = nullptr,
                         std::uint64_t seed = 0,
                         const std::set<std::string>& initially_resident = {},
                         const std::set<std::string>& retained_by_caller = {},
                         std::size_t foreign_held_slots = 0) {
    RunOutcome out;
    out.level = level;

    switch (level) {
        case ExecutionLevel::Behavioral: {
            // L-B applies every op in program order and models no time at all.
            // Reporting makespan 0 here would read as "instant"; has_timing says
            // "not modelled", which is a different claim.
            // L-B MODELS NO RESOURCES AT ALL, so residency is not merely unmodelled here --
            // it is meaningless. Silently ignoring a non-empty set would let a caller believe
            // L-B honoured a placement decision it cannot represent, which is the same class
            // of quiet lie as a timing-free level advancing a clock. Reported, not ignored.
            if (!initially_resident.empty())
                out.unmodelled_inputs.push_back(
                    "initially_resident (" + std::to_string(initially_resident.size()) +
                    " tiles) declared, but L-B models no buffers, so residency has no meaning "
                    "at this level");
            if (!retained_by_caller.empty())
                out.unmodelled_inputs.push_back(
                    "retained_by_caller (" + std::to_string(retained_by_caller.size()) +
                    " tiles) declared, but L-B models no buffers, so retention has no meaning "
                    "at this level");
            if (foreign_held_slots > 0)
                out.unmodelled_inputs.push_back(
                    "foreign_held_slots (" + std::to_string(foreign_held_slots) +
                    ") declared, but L-B models no buffers, so an occupied slot has no meaning "
                    "at this level");
            TileProgramReference ref;
            out.summary = ref.run(prog);
            out.ops = out.summary.ops;
            out.has_timing = false;
            return out;
        }
        case ExecutionLevel::BlockSequential: {
            TileTransactionExecutor exec;
            TileExecutionRequest req{prog, placement, device, streams, seed};
            req.initially_resident = initially_resident;
            req.retained_by_caller = retained_by_caller;
            req.foreign_held_slots = foreign_held_slots;
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
