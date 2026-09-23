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
#include <map>
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

// Which physical stage a movement transfer occupies (design note §6). `Collapsed` is the
// single aggregate pool the descriptor selects by leaving the per-hop lane counts at 0 —
// the pre-increment-5 model, kept as a descriptor setting rather than a code path.
//
// THE COMPUTE FABRIC TALKS ONLY TO L1. No hop terminates at the fabric: movement ends in
// L1, and the fabric is fed from there. `L3ToL1` therefore collapses the BlockMover's
// L3→L2 and the Streamer's L2→L1 into one stage — it is two hops modelled as one, not a
// path from L3 into the fabric. Splitting L2 out later refines this stage; it does not
// extend it past L1.
enum class Hop { Collapsed, DramToL3, L3ToL1 };

inline const char* to_string(Hop h) {
    switch (h) {
        case Hop::Collapsed: return "move";
        case Hop::DramToL3:  return "dram->l3";
        case Hop::L3ToL1:    return "l3->l1";
    }
    return "?";
}

// One transfer of one tile across one hop, on one lane, start to finish with no
// preemption (§6.1).
struct HopRecord {
    Hop hop = Hop::Collapsed;
    Dim lane = 0;
    Cycle start = 0, finish = 0;
};

struct TileOpRecord {
    std::size_t op_index = 0;
    TileOpKind kind{};
    Cycle start = 0, finish = 0;   // the op's whole span: first hop start -> last hop finish
    ResourceKind resource = ResourceKind::ComputeTile;
    Dim resource_id = 0;           // compute tile, or the lane of the FIRST hop
    bool zero_work = false;        // resident/degenerate: real, and visibly free (§7.1)
    // Per-hop occupancy, in chain order. Empty for compute ops and for movement that
    // residency made free. A tile's hops are PIPELINED: hop n+1 starts when hop n
    // completes for that tile, so these intervals abut but the op as a whole can span
    // several lanes over its lifetime.
    std::vector<HopRecord> hops;
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
    // Per-hop movement (increment 5). Populated for whichever hops the descriptor models;
    // with the collapsed descriptor only Hop::Collapsed is non-zero, so a reader can tell
    // which movement model produced these numbers without consulting the descriptor.
    std::map<Hop, Cycle> hop_busy_cycles;         // summed lane occupancy per hop
    std::map<Hop, double> hop_utilization;        // busy / (lanes * makespan)
    std::map<Hop, std::size_t> hop_transfers;     // transfers that actually moved bytes
    std::size_t hop_lane_stalls = 0;              // a transfer ready to advance, no lane free
    // Analytical floor under THIS executor's resource model (design note §9).
    double lower_bound = 0.0;
};

struct TileRunProvenance {
    std::string device;             // DeviceDescriptor::label()
    std::string placement;          // Placement::label()
    // Was the L1 stream timing model used? That model is SYSTOLIC — one realization of a
    // domain flow compute engine, not what the fabric is (see ADR 0002 §3.3).
    bool l1_timing = false;
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
        // materialised as an op's output — until EVERY op that touches it has completed.
        // The slot is freed by counting unfinished users down to zero, which is the
        // credit rule as written: a consumer returns its credit when it is done with the
        // data. The tempting shortcut — release at the highest-indexed user, computed
        // statically — is wrong, because readers of one tile are deliberately NOT ordered
        // against each other. Two macs reading the same A tile on different compute tiles
        // are concurrent, their durations differ once per-op stream costs are applied, and
        // the higher-indexed one can finish first. Releasing on it frees a slot an earlier
        // reader still holds, which undercounts `resident` and lets the run exceed the very
        // capacity this increment exists to enforce.
        //
        // A Drain is therefore a consumer, not a deallocator: it frees the tile only when
        // it happens to be the last user still outstanding.
        auto tile_key = [](const TileCoord& c) {
            return c.operand + "#" + std::to_string(c.ti) + "#" + std::to_string(c.tj);
        };
        std::map<std::string, std::size_t> unfinished_users;           // tile -> users left
        std::vector<std::vector<std::string>> op_tiles(ops.size());    // tiles each op touches
        for (std::size_t i = 0; i < ops.size(); ++i) {
            // Deduplicated: an in-place op naming one tile as both input and output holds
            // ONE slot and is ONE user of it, so it must not be counted twice — neither in
            // `needed_slots` nor in the user count.
            std::set<std::string> unique_tiles;
            for (const TileCoord& c : ops[i].inputs)  unique_tiles.insert(tile_key(c));
            for (const TileCoord& c : ops[i].outputs) unique_tiles.insert(tile_key(c));
            op_tiles[i].assign(unique_tiles.begin(), unique_tiles.end());
            for (const std::string& k : op_tiles[i]) ++unfinished_users[k];
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
            for (const std::string& k : op_tiles[op]) {
                auto it = unfinished_users.find(k);
                if (it == unfinished_users.end()) continue;
                if (--it->second == 0) {          // last user done -> credit returned
                    resident.erase(k);
                    unfinished_users.erase(it);
                }
            }
        };

        // ---- resources -----------------------------------------------------
        // Compute tiles come from the placement (the JIT's decision); movement
        // lanes from the device. Per-hop movement is increment 5.
        const Dim n_cf = std::max<Dim>(req.placement.compute_tiles(), 1);
        const Dim n_lanes = std::max<Dim>(dev.move_lanes, 1);
        std::vector<bool> cf_busy(n_cf, false);

        // ---- movement, per hop (increment 5, §6) ---------------------------
        // A tile does not cross the machine in one step. Each hop is its own pool of
        // lanes with its own per-lane bandwidth, so DRAM bandwidth — usually the real
        // bottleneck — is expressible separately from on-chip movement.
        //
        // The COLLAPSED descriptor is the same mechanism with one hop, not a second code
        // path: that is what §6 means by "the collapse is a descriptor setting rather than
        // a hardcoded assumption", and it is why every pre-increment-5 number is
        // reproduced exactly when the per-hop lane counts are left at 0.
        struct HopSpec {
            Hop hop;
            Dim lanes;
            double bytes_per_cycle;      // PER LANE (§6.1), never aggregate
        };
        std::vector<HopSpec> hop_specs;
        if (dev.per_hop_movement()) {
            hop_specs.push_back({Hop::DramToL3, std::max<Dim>(dev.dram_lanes, 1),
                                 dev.dram_bytes_per_cycle});
            hop_specs.push_back({Hop::L3ToL1, std::max<Dim>(dev.onchip_lanes, 1),
                                 dev.onchip_bytes_per_cycle});
        } else {
            hop_specs.push_back({Hop::Collapsed, n_lanes, dev.bytes_per_cycle});
        }
        std::vector<std::vector<bool>> hop_lane_busy;
        for (const HopSpec& h : hop_specs) hop_lane_busy.emplace_back(h.lanes, false);

        auto hop_index = [&](Hop h) -> std::size_t {
            for (std::size_t i = 0; i < hop_specs.size(); ++i)
                if (hop_specs[i].hop == h) return i;
            return 0;
        };
        auto free_lane_on = [&](std::size_t hi, Dim& out) {
            for (Dim l = 0; l < hop_specs[hi].lanes; ++l)
                if (!hop_lane_busy[hi][l]) { out = l; return true; }
            return false;
        };

        // The chain a movement op traverses, in order, with residency-satisfied hops
        // dropped (§6). Two deliberately different residency semantics, because the
        // descriptor means different things:
        //
        //   - COLLAPSED: one hop stands for the whole DRAM->CF path, so an L3-resident
        //     tile satisfies it entirely and the feed is free. This is increment 4's
        //     behaviour and is preserved bit-for-bit.
        //   - PER-HOP: L3 residency satisfies only DRAM->L3. The tile still has to reach
        //     L1, which is the only layer the fabric reads, so the L3->L1 hop still runs.
        //     This is strictly more faithful, and it is why the two descriptors disagree
        //     on makespan. Making a feed free again needs L1 (and later L2) residency,
        //     which is not modelled yet.
        auto build_chain = [&](std::size_t op, bool resident) {
            std::vector<std::size_t> chain;                  // indices into hop_specs
            if (work[op].bytes <= 0.0) return chain;         // nothing to move
            if (!dev.per_hop_movement()) {
                if (!resident) chain.push_back(0);
                return chain;
            }
            const bool outbound = ops[op].kind == TileOpKind::Drain;
            if (outbound) {                                  // L1 -> L3 -> DRAM
                chain.push_back(hop_index(Hop::L3ToL1));     // same movers, reversed
                chain.push_back(hop_index(Hop::DramToL3));
            } else {                                         // DRAM -> L3 -> L1
                if (!resident) chain.push_back(hop_index(Hop::DramToL3));
                chain.push_back(hop_index(Hop::L3ToL1));
            }
            return chain;
        };
        auto hop_duration = [&](std::size_t op, std::size_t hi) {
            return quantize_cycles(work[op].bytes / std::max(1.0, hop_specs[hi].bytes_per_cycle),
                                   work[op].bytes > 0.0);
        };

        // In-flight movement: which chain an op is on, and how far along it is.
        std::vector<std::vector<std::size_t>> chain_of(ops.size());
        std::vector<std::size_t> stage_of(ops.size(), 0);
        std::size_t hop_lane_stalls = 0;
        std::map<Hop, Cycle> hop_busy;
        std::map<Hop, std::size_t> hop_transfers;

        // Completion events, earliest first; ties broken by op index so a run is
        // reproducible regardless of container order.
        struct Event {
            Cycle at;
            std::size_t op;
            std::size_t stage;                         // which hop of the chain finished
            bool operator<(const Event& o) const {     // std::priority_queue is a max-heap
                return at != o.at ? at > o.at : op > o.op;
            }
        };
        std::priority_queue<Event> events;

        Cycle now = 0;
        std::size_t remaining = ops.size();
        Cycle cf_busy_cycles = 0;

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


        // Start an op. For compute that is the whole story; for movement it starts the
        // FIRST hop of the chain, and later hops are started by advance_hops() as each
        // one completes. `id` is the compute tile, or the lane on that first hop.
        auto fire = [&](std::size_t op, ResourceKind kind, Dim id) {
            fired[op] = true;

            // Residency is checked BEFORE movement: a Feed of a tile already in L3 skips
            // the hops that residency satisfies (§5, §6, and the zero-work case of §7.1).
            const bool feed_of_resident =
                ops[op].kind == TileOpKind::Feed && !ops[op].inputs.empty() &&
                resident.count(tile_key(ops[op].inputs[0])) != 0;

            acquire(op);                       // hold the slots for every tile it touches

            TileOpRecord& rec = result.timeline[op];
            rec.resource = kind;
            rec.resource_id = id;
            rec.start = now;

            if (kind == ResourceKind::ComputeTile) {
                cf_busy[id] = true;
                rec.finish = now + duration[op];
                apply(req.program, ops[op], state_);
                events.push(Event{rec.finish, op, 0});
                return;
            }

            chain_of[op] = build_chain(op, feed_of_resident);
            stage_of[op] = 0;
            if (chain_of[op].empty()) {        // residency made it free, or nothing to move
                if (feed_of_resident) ++resident_feeds;
                rec.zero_work = true;
                rec.finish = now;
                duration[op] = 0;
                apply(req.program, ops[op], state_);
                events.push(Event{rec.finish, op, 0});
                return;
            }
            if (feed_of_resident) ++resident_feeds;   // still counted: it skipped a hop

            // Total cost is the sum over the hops it actually traverses, so the stats and
            // the analytical bound see the whole journey rather than one leg of it.
            Cycle total = 0;
            for (std::size_t hi : chain_of[op]) total += hop_duration(op, hi);
            duration[op] = total;

            // Values are applied at FIRE time, in event order. Dependencies are
            // satisfied by construction, so this is exactly the reference's order for
            // anything that interacts — hence bit-identical results.
            apply(req.program, ops[op], state_);

            const std::size_t hi = chain_of[op][0];
            hop_lane_busy[hi][id] = true;
            const Cycle d = hop_duration(op, hi);
            rec.hops.push_back(HopRecord{hop_specs[hi].hop, id, now, now + d});
            rec.finish = now + d;              // provisional: extended as hops complete
            ++hop_transfers[hop_specs[hi].hop];
            events.push(Event{now + d, op, 0});
        };

        // Advance in-flight transfers whose next hop has a free lane.
        //
        // HOP ADVANCEMENT IS NEVER CREDIT-GATED, and that is load-bearing. The op already
        // took its slots when it fired, so making it re-qualify against the program-order
        // seeker would let an op hold slots while the seeker waits for slots it cannot
        // get — hold-and-wait reintroduced through the movement model, which is exactly
        // the deadlock the acquisition rule exists to prevent. In-flight work is also
        // advanced FIRST in each pass, ahead of starting new ops, so a transfer already
        // holding L3 slots is never starved by newly admitted work that would delay its
        // release.
        //
        // Pending transfers are held in a min-heap PER HOP rather than rescanned: the
        // whole point of the ready heaps is to keep the scheduler off O(n^2), and an
        // O(ops) sweep per pass would put it straight back.
        std::vector<ReadyHeap> hop_pending(hop_specs.size());

        auto start_stage = [&](std::size_t op, std::size_t stage, std::size_t hi, Dim lane) {
            hop_lane_busy[hi][lane] = true;
            stage_of[op] = stage;
            const Cycle d = hop_duration(op, hi);
            result.timeline[op].hops.push_back(HopRecord{hop_specs[hi].hop, lane, now, now + d});
            result.timeline[op].finish = now + d;
            ++hop_transfers[hop_specs[hi].hop];
            events.push(Event{now + d, op, stage});
        };

        auto advance_hops = [&]() {
            bool any = false;
            for (std::size_t hi = 0; hi < hop_specs.size(); ++hi) {
                while (!hop_pending[hi].empty()) {
                    Dim lane = 0;
                    if (!free_lane_on(hi, lane)) { ++hop_lane_stalls; break; }
                    const std::size_t op = hop_pending[hi].top();
                    hop_pending[hi].pop();
                    start_stage(op, stage_of[op] + 1, hi, lane);
                    any = true;
                }
            }
            return any;
        };

        // Credits GATE, they never reorder — and the invariant has to be stronger than
        // "don't promote a later READY op past an earlier one".
        //
        // Measured failure of the weaker rule, on a 64^3 T=16 GEMM whose live set is 21
        // tiles and whose greedy (unbounded) peak residency is 29: it WEDGES at every
        // FINITE POSITIVE capacity below 29 — 25 included, comfortably above the 21 it
        // actually needs, and noting that l3_tiles == 0 means unbounded here, not zero — and
        // succeeds at 29 or more, where it is capacity-equivalent to unbounded and so
        // cannot deadlock. The rule is therefore not universally broken; it simply cannot
        // use a budget between the true live set and its own greedy peak, which is exactly
        // the range this tier exists to model. The feeds are
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
        // above `peak_live_tiles(prog)` is SUFFICIENT — guaranteed to complete. It is not
        // always necessary: reuse and zero-slot ops can reorder enough that a smaller
        // budget still runs. Below the requirement the run stalls and then refuses with a
        // diagnosis rather than wedging silently.
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

        // Only ONE op can be the seeker, so every other slot-needing op is inadmissible
        // by construction. Two consequences, both handled below.
        //
        // (a) The seeker has to be re-read after it fires. Computing it once per try_fire
        //     lets at most one new-slot op start per call, so the remaining free lanes idle
        //     until the next completion even when capacity is sitting there unused — a
        //     makespan and credit_stalls artefact of the bookkeeping, not of the credits.
        //
        // (b) A blocked head must not hide cheap work behind it. An op whose tiles are all
        //     resident needs no slots and is admissible whatever the seeker is, so leaving
        //     it queued behind a slot-starved op idles a resource for nothing. try_fire
        //     therefore takes the lowest-index ADMISSIBLE op, not the head.
        //
        //     The scan is WINDOWED in the hot path and EXHAUSTIVE only where a miss would
        //     be fatal, because neither alone is acceptable:
        //
        //       - window only is unsound. If 33+ consecutive queued ops all need slots the
        //         seeker cannot afford and the one op that would release a slot sits past
        //         the window, nothing fires, nothing is in flight, and a perfectly feasible
        //         program gets a wedge diagnosis. A false refusal is the worst failure this
        //         executor has.
        //       - always exhaustive is unusable. MEASURED: at a tight budget nearly every
        //         queued feed is inadmissible, so the scan walks the whole backlog on every
        //         completion. A 512^3 T=16 GEMM (99,328 ops) went from 559 ms unbounded to
        //         290,473 ms at its peak live set — 520x, growing quadratically (the same
        //         run at 256^3 costs 3.8 s). That is the O(n^2) the ready heaps exist to
        //         avoid, reintroduced through the back door.
        //
        //     So: the window runs per firing decision, and the exhaustive pass runs only as
        //     a precondition of throwing — see the wedge branch below. Progress is
        //     guaranteed whenever anything is in flight, since every completion re-enters
        //     try_fire; the only truly stuck state is an empty event queue, which is
        //     exactly where the exhaustive retry sits. Cost in the common case: zero.
        static constexpr std::size_t kScanWindow = 32;
        bool exhaustive = false;          // set only on the wedge path, never in steady state
        long seeker = -1;
        auto admissible = [&](std::size_t op) {
            if (needed_slots(op) == 0) return true;      // takes nothing new
            if (!has_capacity(op)) return false;         // cannot fit
            return seeker < 0 || static_cast<long>(op) == seeker;
        };
        // Lowest-index admissible op, deferring the entries it looks past and restoring
        // them so heap order (and therefore the run) stays deterministic.
        auto take_admissible = [&](ReadyHeap& h, std::size_t& out) {
            std::vector<std::size_t> deferred;
            bool found = false;
            while (!h.empty() && (exhaustive || deferred.size() < kScanWindow)) {
                const std::size_t op = h.top();
                h.pop();
                if (admissible(op)) { out = op; found = true; break; }
                deferred.push_back(op);
            }
            for (std::size_t d : deferred) h.push(d);                // order is by index
            return found;
        };

        // Passes REPEAT while one of them fires something. Firing changes both the seeker
        // and the residency set, so an op in a queue this pass already walked past can
        // become admissible before the pass ends: the move queue is scanned before compute,
        // and a compute firing can hand the seeker to a movement op. Stopping after one
        // pass leaves that lane idle until the next completion and books a credit stall
        // against capacity that was in fact available.
        //
        // The repeat is amortised, not quadratic: a pass only repeats when it fired at
        // least one op, and the run fires each op exactly once.
        //
        // A stall is therefore counted only on a pass that fires NOTHING anywhere. Counting
        // per failed queue scan would charge a stall to a moment that the very next pass
        // resolves.
        auto try_fire = [&]() {
            bool any = false;
            for (bool pass_progress = true; pass_progress; ) {
                pass_progress = false;
                std::size_t pass_stalls = 0;
                if (advance_hops()) any = pass_progress = true;   // in-flight work first
                seeker = earliest_slot_seeker();
                auto fired_seeker = [&](std::size_t op) {    // advance to the next one
                    if (seeker == static_cast<long>(op)) seeker = earliest_slot_seeker();
                };
                // A movement op enters on the FIRST hop of its chain, so the lane it
                // needs belongs to that hop's pool. Which hop that is depends on residency
                // (an L3-resident feed skips DRAM->L3), so it is resolved here rather than
                // assumed.
                for (;;) {
                    if (ready_move.empty()) break;
                    std::size_t op = 0;
                    if (!take_admissible(ready_move, op)) { ++pass_stalls; break; }
                    const bool resident_feed =
                        ops[op].kind == TileOpKind::Feed && !ops[op].inputs.empty() &&
                        resident.count(tile_key(ops[op].inputs[0])) != 0;
                    const std::vector<std::size_t> c = build_chain(op, resident_feed);
                    Dim lane = 0;
                    if (!c.empty() && !free_lane_on(c[0], lane)) {
                        ready_move.push(op);          // put it back; the lane is the limit
                        ++hop_lane_stalls;
                        break;
                    }
                    fire(op, ResourceKind::MoveLane, lane);
                    fired_seeker(op);
                    any = pass_progress = true;
                }
                for (Dim t = 0; t < n_cf && !ready_compute.empty(); ++t) {
                    if (cf_busy[t]) continue;
                    std::size_t op = 0;
                    if (!take_admissible(ready_compute, op)) { ++pass_stalls; break; }
                    fire(op, ResourceKind::ComputeTile, t);
                    fired_seeker(op);
                    any = pass_progress = true;
                }
                for (Dim t = 0; t < ready_pinned.size(); ++t) {
                    if (cf_busy[t] || ready_pinned[t].empty()) continue;
                    std::size_t op = 0;
                    if (!take_admissible(ready_pinned[t], op)) { ++pass_stalls; continue; }
                    fire(op, ResourceKind::ComputeTile, t);
                    fired_seeker(op);
                    any = pass_progress = true;
                }
                if (!pass_progress && !exhaustive) credit_stalls += pass_stalls;
            }
            return any;
        };

        try_fire();
        while (remaining > 0) {
            if (events.empty()) {
                // Before declaring a wedge, look PAST the scan window. This is the one
                // place the full scan is worth its cost: the alternative is refusing a
                // program that can actually run. It is also the one place it stays cheap,
                // because reaching an empty event queue with work left is rare — every
                // steady-state firing decision took the windowed path.
                exhaustive = true;
                const bool progress = try_fire();
                exhaustive = false;
                if (progress) continue;

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
                const std::size_t stage = events.top().stage;
                events.pop();

                TileOpRecord& rec = result.timeline[op];
                if (rec.resource == ResourceKind::ComputeTile) {
                    cf_busy[rec.resource_id] = false;
                    cf_busy_cycles += duration[op];
                } else if (!chain_of[op].empty()) {
                    // A HOP finished, which is not the same as the OP finishing. Free the
                    // lane, bank its occupancy, and either hand the tile to the next hop
                    // or, if this was the last, complete the op.
                    const std::size_t hi = chain_of[op][stage];
                    const HopRecord& hr = rec.hops[stage];
                    hop_lane_busy[hi][hr.lane] = false;
                    hop_busy[hop_specs[hi].hop] += hr.finish - hr.start;
                    if (stage + 1 < chain_of[op].size()) {
                        hop_pending[chain_of[op][stage + 1]].push(op);
                        continue;            // still in flight: not completed, not counted
                    }
                }

                completed[op] = true;
                --remaining;
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
        Cycle all_hop_busy = 0;
        Dim all_hop_lanes = 0;
        double hop_floor = 0.0;
        for (const HopSpec& h : hop_specs) {
            const Cycle busy = hop_busy.count(h.hop) ? hop_busy[h.hop] : 0;
            all_hop_busy += busy;
            all_hop_lanes = static_cast<Dim>(all_hop_lanes + h.lanes);
            st.hop_busy_cycles[h.hop] = busy;
            st.hop_transfers[h.hop] = hop_transfers.count(h.hop) ? hop_transfers[h.hop] : 0;
            // Each hop is its own bottleneck candidate: the floor is the BUSIEST hop, not
            // the average, because a hop's lanes cannot help another hop's traffic.
            hop_floor = std::max(hop_floor, double(busy) / double(std::max<Dim>(h.lanes, 1)));
        }
        if (st.makespan > 0) {
            st.compute_utilization =
                static_cast<double>(cf_busy_cycles) / (double(n_cf) * double(st.makespan));
            st.movement_utilization =
                static_cast<double>(all_hop_busy) /
                (double(std::max<Dim>(all_hop_lanes, 1)) * double(st.makespan));
            for (const HopSpec& h : hop_specs)
                st.hop_utilization[h.hop] =
                    double(st.hop_busy_cycles[h.hop]) /
                    (double(std::max<Dim>(h.lanes, 1)) * double(st.makespan));
        }
        st.lower_bound = std::max({critical_path_(deps, duration),
                                   double(st.compute_cycles) / double(n_cf),
                                   hop_floor});
        st.hop_lane_stalls = hop_lane_stalls;

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
