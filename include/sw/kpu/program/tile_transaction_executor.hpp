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
// THE NAME OF A TILE, and the one spelling of it.
//
// This was a lambda inside run(), which was fine while nothing outside the executor needed
// to name a tile. `initially_resident` below changes that: an orchestrator that seeds a
// resident tile has to produce exactly the key the executor compares against, and a caller
// copying a format out of a lambda is the "two spellings drift" failure waiting to happen --
// it cost this seam its first test, which seeded "A[0,0]" and silently matched nothing.
//
// Not TileCoord::str(): that is for humans and uses "A[0,1]". This is a key, and the two
// have different jobs, so they are allowed to differ -- but each has exactly one definition.
inline std::string tile_key(const TileCoord& c) {
    return c.operand + "#" + std::to_string(c.ti) + "#" + std::to_string(c.tj);
}

struct TileExecutionRequest {
    TileProgram& program;
    const Placement& placement;
    const characterize::DeviceDescriptor& device;
    const stream::StreamProgram* streams = nullptr;   // optional (design note §7.4)
    std::uint64_t seed = 0;                           // recorded; unused until timing is stochastic

    // TILES THAT ARE ALREADY IN L3 WHEN THIS RUN STARTS, by tile key.
    //
    // Spell the key with tile_key() above -- "A#0#1", not TileCoord::str()'s "A[0,1]". A
    // caller that writes the human spelling by hand matches nothing and is told nothing,
    // which is how this seam lost its first test; naming the format here and pointing at
    // the one function that produces it is the whole reason tile_key() is public.
    //
    // Empty is the old behaviour and stays the default: a run that begins with nothing
    // resident. A non-empty set is what makes the KPU STATEFUL ACROSS RUNS (#305): an
    // orchestrator that placed a tile for one operator can tell the next operator it is
    // still there, and the observable consequence is that the tile's chain SKIPS THE DMA
    // LEG -- visible in stats.hop_transfers[Hop::DmaDramToL3] and in resident_feeds.
    //
    // THEIR LIFETIME BELONGS TO THE CALLER, and that takes an explicit exemption rather
    // than falling out of the bookkeeping. A seeded tile the program also touches DOES get
    // a consumer count like any other, so the completion rule would free it the moment its
    // last reader finished -- handing a slot the orchestrator still believes it holds to an
    // unrelated tile. The release rule therefore skips seeded keys: they are released by an
    // orchestrator decision (a RELEASE descriptor), not by this executor guessing. An
    // executor that freed them would be freeing what it does not own -- the same ownership
    // mistake the per-tile refcount fixed within a run.
    //
    // So a seeded tile holds its slot for the WHOLE run, which is a real cost: the budget
    // this program needs is its own live set PLUS what the caller is holding, and a run can
    // refuse at a capacity the same program accepts cold. That is the honest answer, not a
    // regression.
    //
    // They count against L3 capacity, because they occupy slots. A caller that seeds more
    // than the capacity is refused BEFORE anything is scheduled, with both numbers -- see
    // run(), which cannot leave that to the ordinary per-op check.
    //
    // `= {}` is not decoration: every field above carries a default so a caller can
    // brace-initialize the first three and omit the rest, and without one this addition
    // breaks every such caller under -Werror=missing-field-initializers. Adding a field to
    // an aggregate others initialize positionally is a compatibility event, and the type's
    // own convention is how it stays a compatible one.
    std::set<std::string> initially_resident = {};

    // TILES THE CALLER WILL STILL BE HOLDING WHEN THIS RUN ENDS, by tile_key.
    //
    // `initially_resident` says a tile is already there; this says a tile must still be there
    // afterwards. They are different claims and both are needed, because an orchestrator that
    // places a tile for THIS run and intends to reuse it in the NEXT one can say neither with
    // the other: the tile is not resident at the start (it has to be fetched), and the
    // completion rule would return its credit at its last reader -- so a later run that
    // seeded it would be claiming a residency the machine did not provide, skipping a DMA leg
    // for a tile whose slot had been handed to something else. That is a timing result
    // credited to a reuse that never happened, which is worse than a slow model.
    //
    // So a retained tile occupies its slot from the moment it becomes resident to the end of
    // the run, exempt from release exactly as a seeded tile is. It does NOT skip the DMA leg:
    // it is not there yet, and this run is what puts it there.
    //
    // The cost is real and is the point: `initially_resident` plus `retained_by_caller` is
    // what the caller holds at the end, and a run whose union exceeds the L3 is refused up
    // front, because it cannot finish.
    std::set<std::string> retained_by_caller = {};
};

enum class ResourceKind { ComputeTile, MoveLane };

inline const char* to_string(ResourceKind r) {
    return r == ResourceKind::ComputeTile ? "CF" : "lane";
}

// One leg of the movement chain, named by its governing CSP process and the memories it
// connects (design note §6.1).
//
//   DMA         DRAM <-> L3
//   BlockMover  L3 <-> L2, and L3 -> L3 across the NoC (reuse)
//   Streamer    L2 <-> L1
//
// HOPS DO NOT COLLAPSE. Each leg is a distinct process over a distinct physical pathway,
// so one stage standing for two describes a machine that cannot be built. A span always
// contains all of its hops; residency changes only where the chain STARTS. There is
// deliberately no `Collapsed` member and no single-pool mode — an earlier revision had
// both, which was the premise of this increment and it was wrong.
//
// The fabric is not a hop endpoint: it reads only L1, and the L1 stream buffers push
// elements into it, which is not a mover and owns no lanes (ADR 0002 §3.3).
enum class Hop {
    DmaDramToL3,        // 1. inbound: DMA picks the data out of DRAM
    BlockMoverL3ToL2,   // 2. inbound: BlockMover, may restructure/reshape
    StreamerL2ToL1,     // 3. inbound: Streamer writes an L1 stream buffer
    StreamerL1ToL2,     // 5. outbound: Streamer reads results out of L1
    BlockMoverL2ToL3,   // 6. outbound
    DmaL3ToDram,        // 7. outbound, only if the result must reach DRAM
    BlockMoverL3ToL3,   // reuse: across the NoC
};

inline const char* to_string(Hop h) {
    switch (h) {
        case Hop::DmaDramToL3:      return "dma:dram->l3";
        case Hop::BlockMoverL3ToL2: return "bm:l3->l2";
        case Hop::StreamerL2ToL1:   return "str:l2->l1";
        case Hop::StreamerL1ToL2:   return "str:l1->l2";
        case Hop::BlockMoverL2ToL3: return "bm:l2->l3";
        case Hop::DmaL3ToDram:      return "dma:l3->dram";
        case Hop::BlockMoverL3ToL3: return "bm:l3->l3";
    }
    return "?";
}

// The process that owns the lanes. Inbound and outbound legs of one process SHARE its
// pool: there is one set of BlockMovers, not one per direction.
enum class Mover { Dma, BlockMover, Streamer, Noc };

inline const char* to_string(Mover m) {
    switch (m) {
        case Mover::Dma:        return "dma";
        case Mover::BlockMover: return "block-mover";
        case Mover::Streamer:   return "streamer";
        case Mover::Noc:        return "noc";
    }
    return "?";
}

inline Mover mover_of(Hop h) {
    switch (h) {
        case Hop::DmaDramToL3:
        case Hop::DmaL3ToDram:      return Mover::Dma;
        case Hop::BlockMoverL3ToL2:
        case Hop::BlockMoverL2ToL3: return Mover::BlockMover;
        case Hop::StreamerL2ToL1:
        case Hop::StreamerL1ToL2:   return Mover::Streamer;
        case Hop::BlockMoverL3ToL3: return Mover::Noc;
    }
    return Mover::Dma;
}

// One transfer of one tile across one hop, on one lane, start to finish with no
// preemption (§6.3).
struct HopRecord {
    Hop hop = Hop::DmaDramToL3;
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
    // Feeds whose tile was already in L3, so the chain started at the BlockMover instead
    // of the DMA. Cheaper by one leg -- NOT free: the tile still has to reach a stream
    // buffer, because L2 and L1 are separate memories reached by separate processes.
    std::size_t resident_feeds = 0;
    // Movement, per leg (increment 5). Every leg a run used appears here, so the chain is
    // reconstructible from the stats alone.
    std::map<Hop, Cycle> hop_busy_cycles;         // cycles spent on each leg
    std::map<Hop, std::size_t> hop_transfers;     // transfers that crossed each leg
    // Occupancy is per PROCESS, not per leg: the lanes belong to the DMA engines,
    // BlockMovers and Streamers, and two legs of one process compete for them. A per-leg
    // utilization would divide by a pool the leg does not own on its own.
    std::map<Mover, Cycle> mover_busy_cycles;
    std::map<Mover, double> mover_utilization;    // busy / (lanes * makespan)
    std::map<Mover, Dim> mover_lanes;
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
        // The stream-derived cost, kept SEPARATELY from the lumped one when an L1 stream
        // program is present. l1_duration() describes what happens at the L1 stream
        // buffers — a Drain costs the C signature's element stride, so an
        // output-stationary drain BUBBLE stretches it, and a Feed costs elements/lanes.
        // That is a property of the STREAMER legs (L2->L1 in, L1->L2 out), which are the
        // hops that touch those buffers; the DMA and BlockMover legs upstream are byte
        // movements and keep the byte model. Charging the whole chain the stream cost, or
        // none of it, would both misplace a real effect.
        std::vector<Cycle> stream_dur(ops.size(), 0);
        for (std::size_t i = 0; i < ops.size(); ++i) {
            work[i] = tile_work_of(prog, ops[i], dev.element_bytes);
            double d = lumped_duration(work[i], dev.fabric_macs_per_cycle, dev.bytes_per_cycle);
            if (req.streams) {
                d = l1_duration(prog, ops[i], i, d, *req.streams);
                if (!work[i].is_compute) stream_dur[i] = quantize_cycles(d, work[i].bytes > 0.0);
            }
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
        // Seeded from the request, so a tile an orchestrator already placed is not placed
        // again. See TileExecutionRequest::initially_resident on why these are never freed
        // here: their lifetime belongs to whoever placed them.
        std::set<std::string> resident = req.initially_resident;
        std::size_t peak_residency = resident.size(), credit_stalls = 0, resident_feeds = 0;

        // SEEDED MORE THAN THE MACHINE HOLDS: refuse here, before anything is scheduled.
        // The per-op check below cannot catch it, because it only runs when an op needs a
        // NEW slot -- so a program whose every tile is already seeded, or one with no ops
        // at all, would run to completion over an L3 that was overfull from cycle zero and
        // report a peak residency above its own capacity.
        if (l3_capacity != 0 && resident.size() > static_cast<std::size_t>(l3_capacity))
            throw std::runtime_error(
                "TileTransactionExecutor: " + std::to_string(resident.size()) +
                " tiles were seeded resident but this L3 holds " +
                std::to_string(l3_capacity) + ". The initial residency does not fit the L3 "
                "budget — raise l3_tiles, or release tiles before this run.");

        // The caller's tiles, exempt from the completion rule. See
        // TileExecutionRequest::initially_resident on why this exemption has to be written
        // down instead of emerging: a seeded tile the program reads has a consumer count
        // like any other, and counting it down to zero would free a slot we do not own.
        //
        // SEEDED and RETAINED differ at the START of the run -- one is already there, the
        // other is not -- and are identical at the END, since both are still held. The
        // release rule is an end-of-life rule, so it is the UNION that matters here.
        std::set<std::string> held = req.initially_resident;
        held.insert(req.retained_by_caller.begin(), req.retained_by_caller.end());

        // A run whose caller-held set does not fit cannot finish: nothing ever releases those
        // slots, so the wedge is certain. Saying so here names the cause; reaching it through
        // the wedge path would report a dependency stall and make the reader find the cause.
        if (l3_capacity != 0 && held.size() > static_cast<std::size_t>(l3_capacity))
            throw std::runtime_error(
                "TileTransactionExecutor: the caller holds " + std::to_string(held.size()) +
                " tiles at the end of this run (" +
                std::to_string(req.initially_resident.size()) + " seeded, " +
                std::to_string(held.size() - req.initially_resident.size()) +
                " newly retained) but this L3 holds " + std::to_string(l3_capacity) +
                ". Nothing releases a held tile, so the run cannot complete — retain "
                "fewer tiles, or raise l3_tiles.");

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
                    if (held.count(k) == 0) resident.erase(k);     // not ours to free
                    unfinished_users.erase(it);
                }
            }
        };

        // ---- resources -----------------------------------------------------
        // Compute tiles come from the placement (the JIT's decision); movement
        // lanes belong to the movement processes below (§6.3).
        const Dim n_cf = std::max<Dim>(req.placement.compute_tiles(), 1);
        std::vector<bool> cf_busy(n_cf, false);

        // ---- movement, per CSP process (increment 5, §6) -------------------
        // Lanes belong to the PROCESS, not the leg: inbound and outbound share one pool
        // because there is one set of BlockMovers, not one per direction.
        struct MoverSpec {
            Mover mover;
            Dim lanes;
            double bytes_per_cycle;      // PER LANE (§6.3), never aggregate
        };
        const std::vector<MoverSpec> movers = {
            {Mover::Dma,        std::max<Dim>(dev.dma_engines, 1),  dev.dma_bytes_per_cycle},
            {Mover::BlockMover, std::max<Dim>(dev.block_movers, 1), dev.bm_bytes_per_cycle},
            {Mover::Streamer,   std::max<Dim>(dev.streamers, 1),    dev.str_bytes_per_cycle},
            {Mover::Noc,        std::max<Dim>(dev.noc_links, 1),    dev.noc_bytes_per_cycle},
        };
        auto pool_of = [&](Hop h) -> std::size_t {
            const Mover m = mover_of(h);
            for (std::size_t i = 0; i < movers.size(); ++i)
                if (movers[i].mover == m) return i;
            return 0;
        };
        std::vector<std::vector<bool>> lane_busy;
        for (const MoverSpec& m : movers) lane_busy.emplace_back(m.lanes, false);

        auto free_lane_on = [&](std::size_t pool, Dim& out) {
            for (Dim l = 0; l < movers[pool].lanes; ++l)
                if (!lane_busy[pool][l]) { out = l; return true; }
            return false;
        };
        // Is there ANY mover with a free lane? If not, no transfer can start whatever its
        // chain, so the scheduler must not pop the ready heap to discover that — doing so
        // popped every queued transfer on every pass, which is the O(n^2) the heaps exist
        // to prevent (measured: 60s vs 0.17s on a 128^3 T=4 GEMM).
        auto any_lane_free = [&]() {
            for (std::size_t i = 0; i < movers.size(); ++i) {
                Dim l = 0;
                if (free_lane_on(i, l)) return true;
            }
            return false;
        };

        // The chain a movement op traverses, in order. ALL of its hops, always: the
        // pathways for a shortcut do not exist. Residency changes where the chain STARTS,
        // never which hops it contains — a tile already in L3 begins at the BlockMover,
        // which is the reuse case of §6.1, not a collapse.
        auto build_chain = [&](std::size_t op, bool resident) {
            std::vector<Hop> chain;
            if (work[op].bytes <= 0.0) return chain;         // nothing to move
            if (ops[op].kind == TileOpKind::Drain) {         // L1 -> L2 -> L3 -> DRAM
                chain.push_back(Hop::StreamerL1ToL2);
                chain.push_back(Hop::BlockMoverL2ToL3);
                chain.push_back(Hop::DmaL3ToDram);
                return chain;
            }
            // Inbound. A resident tile is already past the DMA leg; everything below it
            // still has to happen, because L2 and L1 are separate memories reached by
            // separate processes.
            if (!resident) chain.push_back(Hop::DmaDramToL3);
            chain.push_back(Hop::BlockMoverL3ToL2);
            chain.push_back(Hop::StreamerL2ToL1);
            return chain;
        };
        auto hop_duration = [&](std::size_t op, Hop h) {
            // The Streamer legs carry the stream-derived cost when there is one: they are
            // the hops at the L1 stream buffers, which is what l1_duration() models.
            if (stream_dur[op] > 0 && mover_of(h) == Mover::Streamer) return stream_dur[op];
            return quantize_cycles(
                work[op].bytes / std::max(1.0, movers[pool_of(h)].bytes_per_cycle),
                work[op].bytes > 0.0);
        };

        // In-flight movement: which chain an op is on, and how far along it is.
        std::vector<std::vector<Hop>> chain_of(ops.size());
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
            for (Hop h : chain_of[op]) total += hop_duration(op, h);
            duration[op] = total;

            // Values are applied at FIRE time, in event order. Dependencies are
            // satisfied by construction, so this is exactly the reference's order for
            // anything that interacts — hence bit-identical results.
            apply(req.program, ops[op], state_);

            const Hop h0 = chain_of[op][0];
            lane_busy[pool_of(h0)][id] = true;
            const Cycle d = hop_duration(op, h0);
            rec.hops.push_back(HopRecord{h0, id, now, now + d});
            rec.finish = now + d;              // provisional: extended as hops complete
            ++hop_transfers[h0];
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
        std::vector<ReadyHeap> hop_pending(movers.size());

        auto start_stage = [&](std::size_t op, std::size_t stage, Hop h, Dim lane) {
            lane_busy[pool_of(h)][lane] = true;
            stage_of[op] = stage;
            const Cycle d = hop_duration(op, h);
            result.timeline[op].hops.push_back(HopRecord{h, lane, now, now + d});
            result.timeline[op].finish = now + d;
            ++hop_transfers[h];
            events.push(Event{now + d, op, stage});
        };

        auto advance_hops = [&]() {
            bool any = false;
            for (std::size_t pool = 0; pool < movers.size(); ++pool) {
                while (!hop_pending[pool].empty()) {
                    Dim lane = 0;
                    if (!free_lane_on(pool, lane)) { ++hop_lane_stalls; break; }
                    const std::size_t op = hop_pending[pool].top();
                    hop_pending[pool].pop();
                    start_stage(op, stage_of[op] + 1, chain_of[op][stage_of[op] + 1], lane);
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
            // Movement ops held out because their FIRST hop had no free lane. They are
            // credit-admissible, so ending the scan on them would stop an op whose first
            // hop is a DIFFERENT pool from starting — and §6 requires different tiles to
            // occupy different hops concurrently. Restored once the passes finish, so each
            // is taken out at most once per call and the min-heap puts the order back.
            std::vector<std::size_t> lane_blocked;
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
                for (std::size_t held = 0;;) {
                    if (ready_move.empty()) break;
                    if (!any_lane_free()) break;         // nothing can move: do not pop
                    std::size_t op = 0;
                    if (!take_admissible(ready_move, op)) { ++pass_stalls; break; }
                    const bool resident_feed =
                        ops[op].kind == TileOpKind::Feed && !ops[op].inputs.empty() &&
                        resident.count(tile_key(ops[op].inputs[0])) != 0;
                    const std::vector<Hop> c = build_chain(op, resident_feed);
                    Dim lane = 0;
                    if (!c.empty() && !free_lane_on(pool_of(c[0]), lane)) {
                        // This op's pool is full, but another pool may be free and a
                        // later op may want it, so hold this one out rather than ending
                        // the scan. Bounded by the same window take_admissible uses: an
                        // unbounded hold-out drains the whole heap when every queued
                        // transfer wants the busy pool.
                        lane_blocked.push_back(op);
                        ++hop_lane_stalls;
                        if (++held >= kScanWindow) break;
                        continue;
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
            for (std::size_t op : lane_blocked) ready_move.push(op);
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
                    const Hop h = chain_of[op][stage];
                    const HopRecord& hr = rec.hops[stage];
                    lane_busy[pool_of(h)][hr.lane] = false;
                    hop_busy[h] += hr.finish - hr.start;
                    if (stage + 1 < chain_of[op].size()) {
                        hop_pending[pool_of(chain_of[op][stage + 1])].push(op);
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
        // Per-hop cycles, and per-MOVER occupancy: the lanes are the process's, so a hop's
        // utilization is only meaningful against its process's pool, and two legs of the
        // same process compete with each other.
        Cycle all_busy = 0;
        Dim all_lanes = 0;
        std::map<Mover, Cycle> mover_busy;
        for (const auto& kv : hop_busy) mover_busy[mover_of(kv.first)] += kv.second;
        for (const auto& kv : hop_busy) {
            st.hop_busy_cycles[kv.first] = kv.second;
            all_busy += kv.second;
        }
        for (const auto& kv : hop_transfers) st.hop_transfers[kv.first] = kv.second;

        double mover_floor = 0.0;
        for (const MoverSpec& m : movers) {
            all_lanes = static_cast<Dim>(all_lanes + m.lanes);
            const Cycle busy = mover_busy.count(m.mover) ? mover_busy[m.mover] : 0;
            st.mover_busy_cycles[m.mover] = busy;
            st.mover_lanes[m.mover] = m.lanes;
            // Each PROCESS is its own bottleneck candidate: the floor is the busiest one,
            // not the average, because one process's lanes cannot carry another's traffic.
            mover_floor = std::max(mover_floor,
                                   double(busy) / double(std::max<Dim>(m.lanes, 1)));
        }
        if (st.makespan > 0) {
            st.compute_utilization =
                static_cast<double>(cf_busy_cycles) / (double(n_cf) * double(st.makespan));
            st.movement_utilization =
                static_cast<double>(all_busy) /
                (double(std::max<Dim>(all_lanes, 1)) * double(st.makespan));
            for (const MoverSpec& m : movers)
                st.mover_utilization[m.mover] =
                    double(st.mover_busy_cycles[m.mover]) /
                    (double(std::max<Dim>(m.lanes, 1)) * double(st.makespan));
        }
        st.lower_bound = std::max({critical_path_(deps, duration),
                                   double(st.compute_cycles) / double(n_cf),
                                   mover_floor});
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
