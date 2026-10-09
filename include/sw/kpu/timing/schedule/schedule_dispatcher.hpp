// ============================================================================
// include/sw/kpu/timing/schedule/schedule_dispatcher.hpp
// The schedule-paced dispatcher (docs/plans/system-schedule-debugger.md §3.3, step 3).
//
// Without it, every operation of a schedule is handed to the executor before cycle 0 and a
// load takes an L3 credit as soon as one exists, so a credit can sit reserved and empty for
// thousands of cycles waiting for DRAM. The dispatcher releases operations in schedule order
// against a CURSOR, the oldest compute not yet complete: an operation is released when the
// compute it serves is at most P computes ahead of the cursor. P = 1 is double buffering (load
// the next tile while this one computes); an unlimited P reproduces the release-everything
// behaviour exactly.
//
// P is a SCHEDULE parameter, not a machine parameter (decided, Q2): it is chosen per run and
// never appears in a deployment spec.
//
// Pacing decides WHEN an operation may ask; credits still decide WHETHER there is room. A
// released load pushes only with an L3 credit (credits up, data down).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sw::kpu::timing::schedule {

class ScheduleDispatcher {
public:
    static constexpr std::size_t kUnlimited = std::numeric_limits<std::size_t>::max();

    // One release: which operation, when, and the cursor it was released against.
    struct Release {
        std::size_t op = 0;
        Cycle cycle = 0;
        std::size_t cursor = 0;
        bool forced = false;        // released past P to let the cursor's own compute through
    };

    // `ops` must outlive the dispatcher. `prefetch_depth` is P.
    ScheduleDispatcher(const std::vector<ScheduleOperation>& ops, std::size_t prefetch_depth)
        : ops_(ops), depth_(prefetch_depth) {
        tag();
    }

    // The compute each operation serves (its consumer; for DRAIN/WRITEBACK/STORE, its producer).
    [[nodiscard]] std::size_t serves(std::size_t op) const { return serves_.at(op); }
    [[nodiscard]] std::size_t computes() const { return compute_op_.size(); }
    [[nodiscard]] std::size_t prefetch_depth() const { return depth_; }
    [[nodiscard]] const std::vector<Release>& releases() const { return releases_; }
    [[nodiscard]] bool done() const { return next_ == ops_.size(); }
    // The oldest compute not yet complete (computes() once all are).
    [[nodiscard]] std::size_t cursor() const { return cursor_; }

    // Release every operation now eligible, in order, through `enqueue`. Returns how many.
    std::size_t release(const ConcurrentTimingExecutor& exec,
                        const std::function<void(const ScheduleOperation&)>& enqueue) {
        while (cursor_ < compute_op_.size()) {
            const auto& [tile, instance] = compute_instance_[cursor_];
            if (exec.completed_compute_count(tile) < instance) break;
            ++cursor_;
        }
        std::size_t n = 0;
        while (next_ < ops_.size() && eligible(serves_[next_])) {
            enqueue(ops_[next_]);
            releases_.push_back({next_, exec.current_cycle(), cursor_, false});
            ++next_;
            ++n;
        }
        // Progress: release is in schedule order, so an operation for a later compute that the
        // schedule placed ahead of the cursor's own compute would block that compute forever.
        // The cursor's compute and everything before it are always released; a forced release
        // says the schedule's order prefetches deeper than P.
        while (next_ < ops_.size() && cursor_ < compute_op_.size() && compute_op_[cursor_] >= next_) {
            enqueue(ops_[next_]);
            releases_.push_back({next_, exec.current_cycle(), cursor_, true});
            ++next_;
            ++n;
        }
        return n;
    }

    [[nodiscard]] std::size_t forced() const {
        return static_cast<std::size_t>(std::count_if(releases_.begin(), releases_.end(),
                                                      [](const Release& r) { return r.forced; }));
    }

private:
    const std::vector<ScheduleOperation>& ops_;
    std::size_t depth_;
    std::vector<std::size_t> serves_;                       // by operation
    std::vector<std::size_t> compute_op_;                   // compute index -> operation index
    std::vector<std::pair<TileID, std::size_t>> compute_instance_;   // its tile, and which instance
    std::size_t next_ = 0, cursor_ = 0;
    std::vector<Release> releases_;

    bool eligible(std::size_t c) const {
        return depth_ == kUnlimited || c <= cursor_ + depth_;
    }

    static bool consumes(const ScheduleOperation& op, const TileID& t) {
        auto in = [&](const std::vector<TileID>& v) { return std::find(v.begin(), v.end(), t) != v.end(); };
        return in(op.dependency_tiles) || in(op.resident_tiles) ||
               (op.dependency_tiles.empty() && op.dependency_tile == t);
    }

    // Tag every operation with the compute it serves. A LOAD/MOVE/FEED serves the first later
    // compute that consumes its tile; a DRAIN/WRITEBACK/STORE serves (follows) the latest
    // earlier compute that produced it. An operation with neither is tied to the computes around
    // it: the next one, or the last.
    void tag() {
        std::unordered_map<TileID, std::size_t, TileIDHash> instances;
        for (std::size_t i = 0; i < ops_.size(); ++i)
            if (ops_[i].type == ScheduleOpType::COMPUTE) {
                compute_op_.push_back(i);
                compute_instance_.push_back({ops_[i].tile.tile_id, ++instances[ops_[i].tile.tile_id]});
            }
        const std::size_t none = compute_op_.size() ? compute_op_.size() - 1 : 0;
        serves_.assign(ops_.size(), none);
        std::size_t c = 0;                                   // the first compute after op i
        for (std::size_t i = 0; i < ops_.size(); ++i) {
            while (c < compute_op_.size() && compute_op_[c] < i) ++c;
            const ScheduleOperation& op = ops_[i];
            const TileID& t = op.tile.tile_id;
            switch (op.type) {
                case ScheduleOpType::COMPUTE:
                    serves_[i] = c;
                    break;
                case ScheduleOpType::LOAD:
                case ScheduleOpType::MOVE:
                case ScheduleOpType::FEED: {
                    std::size_t k = c;
                    while (k < compute_op_.size() && !consumes(ops_[compute_op_[k]], t)) ++k;
                    serves_[i] = k < compute_op_.size() ? k : std::min(c, none);
                    break;
                }
                default: {                                   // DRAIN, WRITEBACK, STORE
                    std::size_t k = c;
                    while (k > 0 && !(ops_[compute_op_[k - 1]].tile.tile_id == t)) --k;
                    serves_[i] = k > 0 ? k - 1 : std::min(c, none);
                    break;
                }
            }
        }
    }
};

}  // namespace sw::kpu::timing::schedule
