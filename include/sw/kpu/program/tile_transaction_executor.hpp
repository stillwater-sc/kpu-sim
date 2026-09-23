// ============================================================================
// include/sw/kpu/program/tile_transaction_executor.hpp
// The TRANSACTIONAL tier: execute an L0 TileProgram, returning exact values AND
// tile-granularity timing from one run.
//
// Decided by ADR 0001 (docs/architecture/adr/0001-program-contract-and-
// transactional-engine.md); designed in docs/plans/tile-transaction-executor.md.
//
// SCOPE (design note §10): the event engine and the firing rule (increment 3), plus
// L3 buffer capacity, credits and residency-based reuse (increment 4). Deliberately
// NOT yet:
//   - L2/L1 capacity, which §5 counts PER COMPUTE TILE and therefore waits for the
//     placement pass to bind tiles to compute tiles (increment 5);
//   - per-hop movement, DRAM→L3 vs on-chip (increment 5);
//   - calibration against the cycle-accurate tier (increment 6);
//   - reachability through the fidelity factory (increment 7).
// Timing here is therefore structurally right but UNCALIBRATED, and says so in
// its provenance rather than pretending otherwise.
//
// What it does guarantee already:
//   - values bit-identical to TileProgramReference, because both drive the same
//     kernels (tile_kernels.hpp) and dependency-respecting orders cannot differ;
//   - deterministic cycles: ready ops fire lowest-index-first, resources are
//     taken lowest-id-first, events are ordered by (time, op index);
//   - aggregate compute cycles that are positive whenever the program computes,
//     and that scale with M, N, K and tile size.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/placement.hpp>
#include <sw/kpu/program/tile_dependencies.hpp>
#include <sw/kpu/program/tile_kernels.hpp>
#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/program/tile_work.hpp>
#include <sw/kpu/program/characterize/device_model.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program {

using Cycle = std::uint64_t;

// ----------------------------------------------------------------------------
// TileExecutionRequest — everything a tier needs, in one struct (ADR D2.1).
// The program is mutated in place: values live in its operand buffers.
// ----------------------------------------------------------------------------
struct TileExecutionRequest {
    TileProgram& program;
    const Placement& placement;
    const characterize::DeviceDescriptor& device;
    const stream::StreamProgram* streams = nullptr;   // optional (design note §7.4)
    std::uint64_t seed = 0;                           // recorded; unused until timing is stochastic
};

enum class ResourceKind { ComputeTile, MoveLane };

inline const char* to_string(ResourceKind r) {
    return r == ResourceKind::ComputeTile ? "CF" : "lane";
}

struct TileOpRecord {
    std::size_t op_index = 0;
    TileOpKind kind{};
    Cycle start = 0, finish = 0;
    ResourceKind resource = ResourceKind::ComputeTile;
    Dim resource_id = 0;
    bool zero_work = false;      // resident/degenerate: real, and visibly free (§7.1)
};

struct TileRunStats {
    Cycle makespan = 0;
    Cycle compute_cycles = 0;     // aggregate over compute ops; 0 only if nothing computes
    Cycle movement_cycles = 0;
    double compute_utilization = 0.0;
    double movement_utilization = 0.0;
    std::size_t ops = 0, computes = 0, movements = 0, zero_work_ops = 0;
    double total_macs = 0.0, total_move_bytes = 0.0;
    // Capacity (increment 4). A stall is an op that was dependency-ready and had a free
    // resource but could not get an L3 slot — the signal that the buffer, not the
    // fabric, is the limit.
    std::size_t l3_credit_stalls = 0;
    std::size_t peak_l3_residency = 0;    // in tiles; compare against DeviceDescriptor::l3_tiles
    std::size_t resident_feeds = 0;       // feeds that cost nothing because the tile was already there
    // Analytical floor under THIS executor's resource model (design note §9).
    double lower_bound = 0.0;
};

struct TileRunProvenance {
    std::string device;             // DeviceDescriptor::label()
    std::string placement;          // Placement::label()
    bool l1_timing = false;         // were systolic latencies used?
    std::uint64_t seed = 0;
    bool calibrated = false;        // increment 6 sets this
    bool extrapolated = false;      // compute_tiles > 1 is uncalibrated (ADR §8)
};

struct TileRunResult {
    TileRunStats stats;
    std::vector<TileOpRecord> timeline;
    TileRunProvenance provenance;
    TileProgramReference::RunSummary summary;   // op counts + LU permutation
};

// ----------------------------------------------------------------------------
// TileTransactionExecutor
// ----------------------------------------------------------------------------
class TileTransactionExecutor {
public:
    TileRunResult run(const TileExecutionRequest& req) {
        const TileProgram& prog = req.program;
        const auto& ops = prog.ops();
        const auto& dev = req.device;

        // Reset the kernel state up front, exactly as TileProgramReference::run does.
        // Without this, reusing one executor leaks a previous run's pivot slots and row
        // permutation into the next, so the second LU run would be WRONG. Every test
        // happened to use a fresh executor, which is precisely why this needed catching.
        state_.clear();

        const TileDependencies deps = build_tile_dependencies(prog);

        TileRunResult result;
        result.timeline.resize(ops.size());
        result.provenance = TileRunProvenance{
            dev.label(), req.placement.label(), req.streams != nullptr, req.seed,
            /*calibrated=*/false,
            /*extrapolated=*/dev.compute_tiles > 1 || req.placement.compute_tiles() > 1};

        // ---- per-op cost, quantized once ----------------------------------
        std::vector<Cycle> duration(ops.size(), 0);
        std::vector<TileWork> work(ops.size());
        for (std::size_t i = 0; i < ops.size(); ++i) {
            work[i] = tile_work_of(prog, ops[i], dev.element_bytes);
            double d = lumped_duration(work[i], dev.fabric_macs_per_cycle, dev.bytes_per_cycle);
            if (req.streams) d = l1_duration(prog, ops[i], i, d, *req.streams);
            const bool has_work = work[i].macs > 0.0 || work[i].bytes > 0.0;
            duration[i] = quantize_cycles(d, has_work);
            result.timeline[i].op_index = i;
            result.timeline[i].kind = ops[i].kind;
            result.timeline[i].zero_work = !has_work;
        }

        // ---- L3 capacity, residency and tile lifetimes (increment 4) --------
        // A tile occupies an L3 slot from the moment it becomes resident — fed in, or
        // materialised as an op's output — until its LAST USE completes. Last use is
        // computed statically here, so credit returns are deterministic and a Drain
        // behaves as a consumer rather than a deallocator: it frees the slot only when
        // it happens to be the last op that touches the tile.
        auto tile_key = [](const TileCoord& c) {
            return c.operand + "#" + std::to_string(c.ti) + "#" + std::to_string(c.tj);
        };
        std::map<std::string, std::size_t> last_use;      // tile -> last op touching it
        std::vector<std::vector<std::string>> op_tiles(ops.size());   // tiles each op touches
        for (std::size_t i = 0; i < ops.size(); ++i) {
            for (const TileCoord& c : ops[i].inputs)  { last_use[tile_key(c)] = i; op_tiles[i].push_back(tile_key(c)); }
            for (const TileCoord& c : ops[i].outputs) { last_use[tile_key(c)] = i; op_tiles[i].push_back(tile_key(c)); }
        }

        const Dim l3_capacity = dev.l3_tiles;             // 0 = unbounded
        std::set<std::string> resident;                   // tiles holding an L3 slot
        std::size_t peak_residency = 0, credit_stalls = 0, resident_feeds = 0;

        // What this op would have to make resident in order to run.
        auto needed_slots = [&](std::size_t op) {
            std::size_t n = 0;
            for (const std::string& k : op_tiles[op])
                if (!resident.count(k)) ++n;
            return n;
        };
        auto has_capacity = [&](std::size_t op) {
            if (l3_capacity == 0) return true;            // unbounded
            const std::size_t need = needed_slots(op);
            return resident.size() + need <= l3_capacity;
        };
        auto acquire = [&](std::size_t op) {
            for (const std::string& k : op_tiles[op]) resident.insert(k);
            peak_residency = std::max(peak_residency, resident.size());
        };
        auto release_after = [&](std::size_t op) {
            for (const std::string& k : op_tiles[op])
                if (last_use.count(k) && last_use[k] == op) resident.erase(k);
        };

        // ---- resources -----------------------------------------------------
        // Compute tiles come from the placement (the JIT's decision); movement
        // lanes from the device. Per-hop movement is increment 5.
        const Dim n_cf = std::max<Dim>(req.placement.compute_tiles(), 1);
        const Dim n_lanes = std::max<Dim>(dev.move_lanes, 1);
        std::vector<bool> cf_busy(n_cf, false), lane_busy(n_lanes, false);

        // Completion events, earliest first; ties broken by op index so a run is
        // reproducible regardless of container order.
        struct Event {
            Cycle at;
            std::size_t op;
            bool operator<(const Event& o) const {     // std::priority_queue is a max-heap
                return at != o.at ? at > o.at : op > o.op;
            }
        };
        std::priority_queue<Event> events;

        Cycle now = 0;
        std::size_t remaining = ops.size();
        Cycle cf_busy_cycles = 0, lane_busy_cycles = 0;

        // ---- dependency bookkeeping ----------------------------------------
        std::vector<std::size_t> pred_remaining(ops.size());
        for (std::size_t i = 0; i < ops.size(); ++i) pred_remaining[i] = deps.preds[i].size();
        std::vector<bool> completed(ops.size(), false);
        std::vector<bool> fired(ops.size(), false);

        // Ready ops live in min-heaps keyed by op index, so the lowest index — the
        // order the compiler emitted — always wins a contended resource, and neither
        // firing nor unblocking ever rescans the ready set. A single sorted vector is
        // O(n^2 log n) at scale: a 1120x1120x1120 GEMM at T=16 emits over 10^6 ops,
        // most of them feeds that sit ready while one lane retires them one at a time.
        // Heaps keep it O(log n) per op, which is what the 10^6-op target needs.
        using ReadyHeap = std::priority_queue<std::size_t, std::vector<std::size_t>,
                                              std::greater<std::size_t>>;
        ReadyHeap ready_move;         // movement lanes are interchangeable
        ReadyHeap ready_compute;      // unpinned: any free compute tile will do
        // Pinned placements need per-tile queues: an op pinned to a busy tile must not
        // block an op pinned to a free one.
        std::vector<ReadyHeap> ready_pinned(req.placement.is_pinned() ? n_cf : 0);

        auto enqueue = [&](std::size_t op) {
            if (!work[op].is_compute) { ready_move.push(op); return; }
            Dim pinned = 0;
            if (req.placement.compute_tile_for(op, pinned) && pinned < n_cf)
                ready_pinned[pinned].push(op);
            else
                ready_compute.push(op);
        };
        for (std::size_t i = 0; i < ops.size(); ++i)
            if (pred_remaining[i] == 0) enqueue(i);


        auto fire = [&](std::size_t op, ResourceKind kind, Dim id) {
            (kind == ResourceKind::ComputeTile ? cf_busy : lane_busy)[id] = true;
            fired[op] = true;

            // Residency is checked BEFORE movement: a Feed of a tile already in L3 moves
            // nothing, so it costs nothing (§5, and the zero-work case of §7.1).
            const bool feed_of_resident =
                ops[op].kind == TileOpKind::Feed && !ops[op].inputs.empty() &&
                resident.count(tile_key(ops[op].inputs[0])) != 0;
            const Cycle dur = feed_of_resident ? 0 : duration[op];
            if (feed_of_resident) ++resident_feeds;

            acquire(op);                       // hold the slots for every tile it touches

            TileOpRecord& rec = result.timeline[op];
            rec.resource = kind;
            rec.resource_id = id;
            rec.zero_work = rec.zero_work || feed_of_resident;
            rec.start = now;
            rec.finish = now + dur;

            // Values are applied at FIRE time, in event order. Dependencies are
            // satisfied by construction, so this is exactly the reference's order for
            // anything that interacts — hence bit-identical results.
            apply(req.program, ops[op], state_);
            events.push(Event{rec.finish, op});
            duration[op] = dur;                // so the stats and bounds see what it cost
        };

        // Credits GATE, they never reorder — and the invariant has to be stronger than
        // "don't promote a later READY op past an earlier one".
        //
        // Measured failure of the weaker rule: a 64^3 T=16 GEMM whose peak live set is 21
        // tiles deadlocked at EVERY finite capacity, 25 included. The feeds are
        // dependency-ready immediately, so they eagerly take every slot; the computes that
        // would retire those tiles are not ready yet (they are waiting on those same
        // feeds), so they never get counted as "earlier waiters", and once the slots are
        // gone the computes cannot obtain the one slot they need for their output. Classic
        // hold-and-wait, and no ordering among READY ops can break it.
        //
        // The invariant that does: NEW SLOTS ARE ACQUIRED IN PROGRAM ORDER. An op may take
        // slots only when no earlier unfired op still needs any. An op needing no new slots
        // — every tile it touches is already resident — fires freely, because it cannot
        // contribute to hold-and-wait.
        //
        // This bounds the live set by the PROGRAM-ORDER live set, so any capacity at or
        // above `peak_live_tiles(prog)` is guaranteed to complete, and a smaller budget
        // stalls and then refuses with a diagnosis rather than wedging silently.
        // Concurrency survives: resident-tile ops still fire in parallel, which is exactly
        // the reuse the tier exists to reward.
        std::size_t slot_cursor = 0;          // monotone: never revisits a fired op
        auto earliest_slot_seeker = [&]() -> long {
            if (l3_capacity == 0) return -1;  // unbounded: no ordering constraint at all
            while (slot_cursor < ops.size() && fired[slot_cursor]) ++slot_cursor;
            for (std::size_t i = slot_cursor; i < ops.size(); ++i)
                if (!fired[i] && needed_slots(i) > 0) return static_cast<long>(i);
            return -1;
        };

        auto try_fire = [&]() {
            const long seeker = earliest_slot_seeker();
            auto admissible = [&](std::size_t op) {
                if (needed_slots(op) == 0) return true;      // takes nothing new
                if (!has_capacity(op)) return false;         // cannot fit
                return seeker < 0 || static_cast<long>(op) == seeker;
            };
            for (Dim l = 0; l < n_lanes && !ready_move.empty(); ++l) {
                if (lane_busy[l]) continue;
                const std::size_t op = ready_move.top();
                if (!admissible(op)) { ++credit_stalls; break; }
                fire(op, ResourceKind::MoveLane, l);
                ready_move.pop();
            }
            for (Dim t = 0; t < n_cf && !ready_compute.empty(); ++t) {
                if (cf_busy[t]) continue;
                const std::size_t op = ready_compute.top();
                if (!admissible(op)) { ++credit_stalls; break; }
                fire(op, ResourceKind::ComputeTile, t);
                ready_compute.pop();
            }
            for (Dim t = 0; t < ready_pinned.size(); ++t) {
                if (cf_busy[t] || ready_pinned[t].empty()) continue;
                const std::size_t op = ready_pinned[t].top();
                if (!admissible(op)) { ++credit_stalls; continue; }
                fire(op, ResourceKind::ComputeTile, t);
                ready_pinned[t].pop();
            }
        };

        try_fire();
        while (remaining > 0) {
            if (events.empty()) {
                // Nothing running, nothing able to fire: a genuine wedge. Diagnose it
                // rather than hang or silently return a short run (design note §4).
                // With capacity modelled this path is reachable, and the usual cause is
                // an L3 budget too small for the program's live set — so say that first,
                // with the numbers, before dumping dependency chains.
                std::string why;
                std::size_t blocked_on_credit = 0;
                for (std::size_t i = 0; i < ops.size(); ++i)
                    if (!completed[i] && !fired[i] && pred_remaining[i] == 0 && !has_capacity(i))
                        ++blocked_on_credit;

                if (blocked_on_credit > 0) {
                    why = "\n  " + std::to_string(blocked_on_credit) +
                          " op(s) are dependency-ready but cannot get an L3 slot: " +
                          std::to_string(resident.size()) + " of " +
                          std::to_string(l3_capacity) + " tiles resident. The program's "
                          "live set does not fit this L3 budget — raise l3_tiles, or "
                          "re-tile so fewer tiles are live at once.";
                }
                for (std::size_t i = 0; i < ops.size() && why.size() < 1200; ++i)
                    if (!completed[i] && !fired[i])
                        why += "\n  " + deps.explain_blocked(prog, i, completed);
                throw std::runtime_error(
                    "TileTransactionExecutor: no op can fire and nothing is in flight; " +
                    std::to_string(remaining) + " ops remain." + why);
            }

            // Advance to the next completion, then release and unblock.
            const Cycle at = events.top().at;
            now = at;
            while (!events.empty() && events.top().at == at) {
                const std::size_t op = events.top().op;
                events.pop();
                completed[op] = true;
                --remaining;

                TileOpRecord& rec = result.timeline[op];
                if (rec.resource == ResourceKind::ComputeTile) {
                    cf_busy[rec.resource_id] = false;
                    cf_busy_cycles += duration[op];
                } else {
                    lane_busy[rec.resource_id] = false;
                    lane_busy_cycles += duration[op];
                }

                release_after(op);          // slots whose last consumer just completed

                for (std::size_t s : deps.succs[op])
                    if (--pred_remaining[s] == 0) enqueue(s);
            }
            try_fire();
        }

        // ---- statistics ----------------------------------------------------
        TileRunStats& st = result.stats;
        st.ops = ops.size();
        for (std::size_t i = 0; i < ops.size(); ++i) {
            st.makespan = std::max(st.makespan, result.timeline[i].finish);
            st.total_macs += work[i].macs;
            st.total_move_bytes += work[i].bytes;
            if (result.timeline[i].zero_work) ++st.zero_work_ops;
            if (work[i].is_compute) { ++st.computes; st.compute_cycles += duration[i]; }
            else                    { ++st.movements; st.movement_cycles += duration[i]; }
        }
        if (st.makespan > 0) {
            st.compute_utilization =
                static_cast<double>(cf_busy_cycles) / (double(n_cf) * double(st.makespan));
            st.movement_utilization =
                static_cast<double>(lane_busy_cycles) / (double(n_lanes) * double(st.makespan));
        }
        st.lower_bound = std::max({critical_path_(deps, duration),
                                   double(st.compute_cycles) / double(n_cf),
                                   double(st.movement_cycles) / double(n_lanes)});

        st.l3_credit_stalls = credit_stalls;
        st.peak_l3_residency = peak_residency;
        st.resident_feeds = resident_feeds;

        result.summary = summarize_(prog);
        return result;
    }

private:
    TileKernelState state_;

    // Longest weighted path — the makespan floor at unlimited resources.
    static double critical_path_(const TileDependencies& deps,
                                 const std::vector<Cycle>& duration) {
        const std::size_t n = deps.op_count();
        std::vector<double> height(n, 0.0);
        double best = 0.0;
        for (std::size_t i = n; i-- > 0;) {            // successors have higher indices
            double tail = 0.0;
            for (std::size_t s : deps.succs[i]) tail = std::max(tail, height[s]);
            height[i] = double(duration[i]) + tail;
            best = std::max(best, height[i]);
        }
        return best;
    }

    // Op counts and the LU permutation, matching the reference's summary so the
    // two tiers can be compared field by field.
    TileProgramReference::RunSummary summarize_(const TileProgram& prog) const {
        TileProgramReference::RunSummary s;
        for (const TileOp& op : prog.ops()) {
            ++s.ops;
            switch (op.kind) {
                case TileOpKind::Feed:           ++s.feeds; break;
                case TileOpKind::Drain:          ++s.drains; break;
                case TileOpKind::MatMulAccum:    ++s.computes; break;
                case TileOpKind::LuDiagFactor:   ++s.diag_factors; break;
                case TileOpKind::PivotApply:     ++s.pivot_applies; break;
                case TileOpKind::TrsmLowerLeft:
                case TileOpKind::TrsmUpperRight: ++s.trsms; break;
            }
        }
        s.row_swaps = state_.swaps_performed;
        s.permutation = state_.perm;
        return s;
    }
};

} // namespace sw::kpu::program
