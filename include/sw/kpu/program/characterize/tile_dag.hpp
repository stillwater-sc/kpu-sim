// ============================================================================
// include/sw/kpu/program/characterize/tile_dag.hpp
// Recover the tile-dependency DAG from an L0 TileProgram and analyze concurrency.
//
// Because every TileOp declares its full tile I/O (§3a of kpu-program-model.md),
// the dependency DAG is recoverable from tile RAW/WAR/WAW hazards (plus the pivot-
// slot producer→consumer edge that GETRF→LASWP needs). From the DAG we derive:
//   - the critical path (makespan lower bound at unlimited compute tiles),
//   - a list-scheduled makespan on a finite DeviceDescriptor (compute tiles +
//     movement lanes) — the "concurrency as a function of resources" question, the
//     domain-flow analogue of CUDA's warp/occupancy-vs-resources adaptation.
// The per-op work model is first-order (see device_model.hpp).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/characterize/device_model.hpp>
#include <sw/kpu/program/stream/stream_signature.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <vector>

namespace sw::kpu::program::characterize {

struct TileWork {
    double macs = 0.0;      // multiply-accumulates (compute ops)
    double bytes = 0.0;     // bytes moved (movement ops)
    bool is_compute = false;
};

struct DagNode {
    std::size_t op_index = 0;
    TileOpKind kind{};
    TileWork work{};
    double duration = 0.0;              // cycles on its resource type
    std::vector<std::size_t> preds, succs;
    // schedule outputs (filled by list_schedule):
    double start = 0.0, finish = 0.0;
    int worker = -1;                    // resource id within its pool
};

class TileDag {
public:
    // Without an L1 StreamProgram, compute/movement durations use the first-order
    // lumped model (device_model.hpp). With one, compute ops take their systolic
    // WAVEFRONT latency and Drain ops are stretched by the C-stream bubble — so the
    // schedule becomes systolic and DATAFLOW-sensitive (output-stationary pays the
    // drain-bubble penalty; weight/A-stationary and hex drain densely).
    TileDag(const TileProgram& prog, const DeviceDescriptor& dev,
            const stream::StreamProgram* l1 = nullptr)
        : dev_(dev), l1_(l1) {
        build_(prog);
    }

    const std::vector<DagNode>& nodes() const { return nodes_; }

    double total_macs() const {
        double s = 0; for (auto& n : nodes_) s += n.work.macs; return s;
    }
    double total_move_bytes() const {
        double s = 0; for (auto& n : nodes_) s += n.work.bytes; return s;
    }
    double compute_work_cycles() const {
        double s = 0; for (auto& n : nodes_) if (n.work.is_compute) s += n.duration; return s;
    }
    double movement_work_cycles() const {
        double s = 0; for (auto& n : nodes_) if (!n.work.is_compute) s += n.duration; return s;
    }

    // Longest weighted path (cycles) = makespan at unlimited resources.
    double critical_path_cycles() const {
        std::vector<double> height(nodes_.size(), -1.0);
        double best = 0.0;
        for (std::size_t i = nodes_.size(); i-- > 0;) best = std::max(best, height_(i, height));
        return best;
    }

    struct Schedule {
        double makespan = 0.0;
        double compute_util = 0.0;   // compute work / (compute_tiles * makespan)
        double movement_util = 0.0;  // movement work / (move_lanes * makespan)
        // analytical lower bound = max(critical path, compute_work/C, move_work/M)
        double lower_bound = 0.0;
    };

    // Greedy critical-path-first list schedule over the full DAG, with two resource
    // pools (compute tiles, movement lanes). Fills node.start/finish/worker.
    Schedule list_schedule() {
        const std::size_t n = nodes_.size();
        std::vector<double> height(n, -1.0);
        // descending: successors (always higher index) are memoized first, so
        // height_ recursion never goes deeper than one frame (bounded on long chains)
        for (std::size_t i = n; i-- > 0;) height_(i, height);

        std::vector<std::size_t> pred_left(n);
        for (std::size_t i = 0; i < n; ++i) pred_left[i] = nodes_[i].preds.size();

        std::vector<double> pred_finish(n, 0.0);   // earliest start = max pred finish
        std::vector<double> compute_free(std::max<Dim>(dev_.compute_tiles, 1), 0.0);
        std::vector<double> move_free(std::max<Dim>(dev_.move_lanes, 1), 0.0);

        std::vector<bool> done(n, false);
        std::size_t remaining = n;
        while (remaining) {
            // pick the ready op with the greatest height (critical-path-first);
            // ties → lowest op index for determinism.
            long pick = -1;
            for (std::size_t i = 0; i < n; ++i) {
                if (done[i] || pred_left[i] != 0) continue;
                if (pick < 0 || height[i] > height[static_cast<std::size_t>(pick)])
                    pick = static_cast<long>(i);
            }
            auto& node = nodes_[static_cast<std::size_t>(pick)];
            auto& pool = node.work.is_compute ? compute_free : move_free;
            // earliest-free resource of this type
            int best_w = 0;
            for (int w = 1; w < static_cast<int>(pool.size()); ++w)
                if (pool[static_cast<std::size_t>(w)] < pool[static_cast<std::size_t>(best_w)]) best_w = w;
            const double start = std::max(pred_finish[static_cast<std::size_t>(pick)],
                                          pool[static_cast<std::size_t>(best_w)]);
            const double finish = start + node.duration;
            node.start = start; node.finish = finish; node.worker = best_w;
            pool[static_cast<std::size_t>(best_w)] = finish;

            for (std::size_t s : node.succs) {
                pred_finish[s] = std::max(pred_finish[s], finish);
                --pred_left[s];
            }
            done[static_cast<std::size_t>(pick)] = true;
            --remaining;
        }

        Schedule sch;
        for (auto& nd : nodes_) sch.makespan = std::max(sch.makespan, nd.finish);
        const double C = std::max<Dim>(dev_.compute_tiles, 1);
        const double M = std::max<Dim>(dev_.move_lanes, 1);
        if (sch.makespan > 0) {
            sch.compute_util = compute_work_cycles() / (C * sch.makespan);
            sch.movement_util = movement_work_cycles() / (M * sch.makespan);
        }
        sch.lower_bound = std::max({critical_path_cycles(),
                                    compute_work_cycles() / C,
                                    movement_work_cycles() / M});
        return sch;
    }

private:
    DeviceDescriptor dev_;
    const stream::StreamProgram* l1_ = nullptr;
    std::vector<DagNode> nodes_;

    // L1-timed duration (systolic cycles): compute = wavefront latency; Drain = the
    // C stream drained down its lanes at the C signature's element stride (the bubble
    // stretches it); Feed = the tile filled at one element/cycle/lane. Movement timing
    // uses the op's ACTUAL (clamped) tile extent, not the nominal signature shape, so
    // trailing tiles at non-divisible dimensions are timed correctly.
    double l1_duration_(const TileProgram& prog, const TileOp& op,
                        std::size_t idx, double fallback) const {
        if (op.kind == TileOpKind::MatMulAccum) {
            auto it = l1_->computes.find(idx);
            return it != l1_->computes.end() ? static_cast<double>(it->second.latency()) : fallback;
        }
        // boundary Feed/Drain: numerator = this op's actual element count
        const TileCoord& tile = op.inputs.empty() ? op.outputs[0] : op.inputs[0];
        Dim rows = 0, cols = 0;
        extent(prog, tile, rows, cols);
        const double elements = static_cast<double>(rows) * cols;
        if (op.kind == TileOpKind::Drain) {
            if (const auto* c = l1_->signature("C")) {
                const double lanes = c->lanes > 0 ? static_cast<double>(c->lanes)
                                   : (cols > 0 ? static_cast<double>(cols) : 1.0);
                return c->element_stride * (elements / lanes);   // bubble stretches the drain
            }
        } else if (op.kind == TileOpKind::Feed) {
            // A stationary operand still preloads into the array (lanes==0), so cost it
            // over its rows rather than falling back to the byte model.
            if (const auto* s = l1_->signature(tile.operand)) {
                const double lanes = s->lanes > 0 ? static_cast<double>(s->lanes)
                                   : (rows > 0 ? static_cast<double>(rows) : 1.0);
                return elements / lanes;
            }
        }
        return fallback;
    }

    double height_(std::size_t i, std::vector<double>& memo) const {
        if (memo[i] >= 0) return memo[i];
        double best = 0.0;
        for (std::size_t s : nodes_[i].succs) best = std::max(best, height_(s, memo));
        return memo[i] = nodes_[i].duration + best;
    }

    // Half-open tile extent for a coord.
    static void extent(const TileProgram& p, const TileCoord& c,
                       Dim& rows, Dim& cols) {
        const TensorOperand& op = p.operand(c.operand);
        rows = op.row_end(c.ti) - op.row_begin(c.ti);
        cols = op.col_end(c.tj) - op.col_begin(c.tj);
    }

    TileWork op_work_(const TileProgram& p, const TileOp& op) const {
        TileWork w;
        Dim r = 0, c = 0, r2 = 0, c2 = 0;
        switch (op.kind) {
            case TileOpKind::MatMulAccum: {
                extent(p, op.outputs[0], r, c);        // m x n
                extent(p, op.inputs[0], r2, c2);       // m x k
                w.macs = double(r) * c * c2;           // m*n*k
                w.is_compute = true;
                break;
            }
            case TileOpKind::LuDiagFactor: {
                extent(p, op.outputs[0], r, c);
                const double t = std::min(r, c);
                w.macs = t * t * t / 3.0;               // ~ (1/3) t^3
                w.is_compute = true;
                break;
            }
            case TileOpKind::TrsmLowerLeft: {           // L(m x m) . X(m x w)
                extent(p, op.outputs[0], r, c);         // m x w
                w.macs = double(r) * r * c / 2.0;
                w.is_compute = true;
                break;
            }
            case TileOpKind::TrsmUpperRight: {          // X(m x w) . U(w x w)
                extent(p, op.outputs[0], r, c);         // m x w
                w.macs = double(r) * c * c / 2.0;
                w.is_compute = true;
                break;
            }
            case TileOpKind::PivotApply: {              // row swaps: movement
                extent(p, op.outputs[0], r, c);
                w.bytes = double(r) * c * dev_.element_bytes;
                break;
            }
            case TileOpKind::Feed:
            case TileOpKind::Drain: {
                const TileCoord& t = op.inputs.empty() ? op.outputs[0] : op.inputs[0];
                extent(p, t, r, c);
                w.bytes = double(r) * c * dev_.element_bytes;
                break;
            }
        }
        return w;
    }

    void build_(const TileProgram& prog) {
        const auto& ops = prog.ops();
        nodes_.resize(ops.size());

        // hazard tracking
        std::map<std::string, long> last_writer;                 // tile-key -> op
        std::map<std::string, std::vector<std::size_t>> readers; // tile-key -> readers since last write
        std::map<int, long> slot_writer;                         // pivot slot -> op
        std::map<std::string, long> last_feed;                   // tile-key -> Feed that made it available

        auto key = [](const TileCoord& c) {
            return c.operand + "#" + std::to_string(c.ti) + "#" + std::to_string(c.tj);
        };
        auto add_dep = [&](std::size_t consumer, long producer) {
            if (producer < 0 || static_cast<std::size_t>(producer) == consumer) return;
            auto& preds = nodes_[consumer].preds;
            if (std::find(preds.begin(), preds.end(), static_cast<std::size_t>(producer)) == preds.end()) {
                preds.push_back(static_cast<std::size_t>(producer));
                nodes_[static_cast<std::size_t>(producer)].succs.push_back(consumer);
            }
        };

        for (std::size_t i = 0; i < ops.size(); ++i) {
            const TileOp& op = ops[i];
            nodes_[i].op_index = i;
            nodes_[i].kind = op.kind;
            nodes_[i].work = op_work_(prog, op);
            nodes_[i].duration = nodes_[i].work.is_compute
                ? nodes_[i].work.macs / std::max(1.0, dev_.fabric_macs_per_cycle)
                : nodes_[i].work.bytes / std::max(1.0, dev_.bytes_per_cycle);

            // L1-timed override: systolic wavefront for compute, bubble-scaled drain.
            if (l1_) nodes_[i].duration = l1_duration_(prog, op, i, nodes_[i].duration);

            // classify reads / writes
            std::vector<const TileCoord*> reads, writes;
            switch (op.kind) {
                case TileOpKind::Feed:
                    // A Feed is a movement SOURCE: it makes an input tile available.
                    // Record it so consumers depend on it (RAW below), but do NOT
                    // treat it as a destructive write — repeated feeds of a shared
                    // input tile must stay independent, else independent output-tile
                    // computations would serialize through their shared input.
                    if (!op.inputs.empty()) last_feed[key(op.inputs[0])] = static_cast<long>(i);
                    break;
                case TileOpKind::Drain:         if (!op.outputs.empty()) reads.push_back(&op.outputs[0]); break;
                case TileOpKind::MatMulAccum:
                    reads.push_back(&op.inputs[0]); reads.push_back(&op.inputs[1]);
                    reads.push_back(&op.outputs[0]); writes.push_back(&op.outputs[0]); // RMW accumulate
                    break;
                case TileOpKind::LuDiagFactor:
                    reads.push_back(&op.outputs[0]); writes.push_back(&op.outputs[0]);
                    if (op.pivot_slot >= 0) slot_writer[op.pivot_slot] = static_cast<long>(i);
                    break;
                case TileOpKind::PivotApply:
                    reads.push_back(&op.outputs[0]); writes.push_back(&op.outputs[0]);
                    if (op.pivot_slot >= 0) {
                        auto it = slot_writer.find(op.pivot_slot);
                        if (it != slot_writer.end()) add_dep(i, it->second);
                    }
                    break;
                case TileOpKind::TrsmLowerLeft:
                case TileOpKind::TrsmUpperRight:
                    reads.push_back(&op.inputs[0]);
                    reads.push_back(&op.outputs[0]); writes.push_back(&op.outputs[0]);
                    break;
            }

            for (const TileCoord* t : reads) {
                const std::string k = key(*t);
                auto it = last_writer.find(k);
                if (it != last_writer.end()) add_dep(i, it->second);     // RAW (on-chip producer)
                auto fit = last_feed.find(k);
                if (fit != last_feed.end()) add_dep(i, fit->second);     // wait for the input feed
                readers[k].push_back(i);
            }
            for (const TileCoord* t : writes) {
                const std::string k = key(*t);
                auto it = last_writer.find(k);
                if (it != last_writer.end()) add_dep(i, it->second);     // WAW
                for (std::size_t rdr : readers[k]) add_dep(i, static_cast<long>(rdr)); // WAR
                last_writer[k] = static_cast<long>(i);
                readers[k].clear();
            }
        }
    }
};

} // namespace sw::kpu::program::characterize
