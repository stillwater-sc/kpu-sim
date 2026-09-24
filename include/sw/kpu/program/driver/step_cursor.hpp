// ============================================================================
// include/sw/kpu/program/driver/step_cursor.hpp
// Single-stepping, at each level's own transaction granularity (design note §D4).
//
//   L-B    one TileOp applied
//   L-T1   one event: an op firing, a hop starting or finishing, an op completing
//   L-T2   one resource transaction   (#283)
//   L-CA   one cycle                  (#283)
//
// The two levels differ in kind, not just in grain, and the steppers reflect that
// honestly:
//
//   - BEHAVIORAL STEPS ARE REAL EXECUTION. `apply()` is public, so the cursor holds
//     the kernel state and applies one op per step. Values are observable as they
//     form, which is what makes it useful for debugging arithmetic.
//   - BLOCK-SEQUENTIAL STEPS ARE A REPLAY of the run's recorded timeline. The
//     executor is an event engine whose schedule depends on the whole program, so
//     stepping it means walking what it did, not pausing it mid-flight. §D4 called
//     this a projection of data the executor already produces, and that is exactly
//     what it is. The consequence worth knowing: values cannot be inspected
//     mid-replay, because the run already finished.
//
// What a step can report is bounded by what the timeline records. Lane occupancy
// per movement process is derivable and is reported. STATION occupancy — how many
// tiles sit in L3, L2, L1 at this cycle — is NOT: the executor keeps its resident
// set internal and publishes only the peak, which is the first gap #286 lists.
// This header does not fake it.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/tile_kernels.hpp>
#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_transaction_executor.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::program::driver {

enum class StepKind {
    OpApplied,      // L-B: the op's arithmetic just happened
    OpFired,        // L-T1: the op acquired its resource and started
    HopStarted,     // L-T1: a leg took a lane
    HopFinished,    // L-T1: a leg released its lane
    OpCompleted,    // L-T1: the last leg finished; credits are returned here
};

inline const char* to_string(StepKind k) {
    switch (k) {
        case StepKind::OpApplied:   return "applied";
        case StepKind::OpFired:     return "fired";
        case StepKind::HopStarted:  return "hop-start";
        case StepKind::HopFinished: return "hop-end";
        case StepKind::OpCompleted: return "completed";
    }
    return "?";
}

struct StepEvent {
    Cycle at = 0;                     // cycle; meaningless at L-B, which models no time
    StepKind kind = StepKind::OpApplied;
    std::size_t op_index = 0;
    TileOpKind op_kind{};
    std::optional<Hop> hop;           // set for hop events
    Dim lane = 0;
    std::size_t leg = 0;              // which leg of the chain
    bool zero_work = false;
};

// ----------------------------------------------------------------------------
// One interface, because the driver should not branch on level to step.
// ----------------------------------------------------------------------------
class Stepper {
public:
    virtual ~Stepper() = default;
    virtual bool step() = 0;                        // false when there is nothing left
    virtual const StepEvent& current() const = 0;
    virtual std::size_t position() const = 0;       // steps taken
    virtual std::size_t size() const = 0;           // total steps, 0 if unknown
    virtual bool models_time() const = 0;
    // Lanes busy per movement process at the current step. Empty where movement is
    // not modelled. NOT station occupancy — see the header comment.
    virtual const std::map<Mover, std::size_t>& lanes_busy() const = 0;
    virtual std::size_t in_flight() const = 0;
};

// ----------------------------------------------------------------------------
// L-B: apply one op per step, for real.
// ----------------------------------------------------------------------------
class BehavioralStepper final : public Stepper {
public:
    explicit BehavioralStepper(TileProgram& prog) : prog_(prog) {}

    bool step() override {
        if (next_ >= prog_.ops().size()) return false;
        const TileOp& op = prog_.ops()[next_];
        apply(prog_, op, state_);            // the arithmetic actually happens here
        cur_ = StepEvent{};
        cur_.at = next_;                     // ordinal, not a cycle: L-B models no time
        cur_.kind = StepKind::OpApplied;
        cur_.op_index = next_;
        cur_.op_kind = op.kind;
        ++next_;
        return true;
    }
    const StepEvent& current() const override { return cur_; }
    std::size_t position() const override { return next_; }
    std::size_t size() const override { return prog_.ops().size(); }
    bool models_time() const override { return false; }
    const std::map<Mover, std::size_t>& lanes_busy() const override { return none_; }
    std::size_t in_flight() const override { return 0; }

private:
    TileProgram& prog_;
    TileKernelState state_;
    StepEvent cur_{};
    std::size_t next_ = 0;
    std::map<Mover, std::size_t> none_;
};

// ----------------------------------------------------------------------------
// L-T1: replay the recorded timeline as an ordered event stream.
// ----------------------------------------------------------------------------
class BlockSequentialStepper final : public Stepper {
public:
    explicit BlockSequentialStepper(const std::vector<TileOpRecord>& timeline) {
        for (const TileOpRecord& rec : timeline) {
            StepEvent fired{};
            fired.at = rec.start;
            fired.kind = StepKind::OpFired;
            fired.op_index = rec.op_index;
            fired.op_kind = rec.kind;
            fired.lane = rec.resource_id;
            fired.zero_work = rec.zero_work;
            events_.push_back(fired);

            for (std::size_t leg = 0; leg < rec.hops.size(); ++leg) {
                const HopRecord& hr = rec.hops[leg];
                for (StepKind k : {StepKind::HopStarted, StepKind::HopFinished}) {
                    StepEvent e{};
                    e.at = (k == StepKind::HopStarted) ? hr.start : hr.finish;
                    e.kind = k;
                    e.op_index = rec.op_index;
                    e.op_kind = rec.kind;
                    e.hop = hr.hop;
                    e.lane = hr.lane;
                    e.leg = leg;
                    events_.push_back(e);
                }
            }

            StepEvent done{};
            done.at = rec.finish;
            done.kind = StepKind::OpCompleted;
            done.op_index = rec.op_index;
            done.op_kind = rec.kind;
            done.leg = rec.hops.empty() ? 0 : rec.hops.size() - 1;
            done.zero_work = rec.zero_work;
            events_.push_back(done);
        }

        // Order matters more than it looks, and it has to MIRROR THE EXECUTOR rather than
        // merely be self-consistent. At one cycle the executor processes completions first
        // and only then fires ready ops, so within a cycle every RELEASE precedes every
        // ACQUIRE — across ops, not just within one.
        //
        // Sorting by (cycle, op, leg, kind) alone got this wrong and the replay reported
        // occupancy ABOVE CAPACITY: a lower-indexed op taking a lane sorted before a
        // higher-indexed op giving one back at the same cycle, so the count incremented
        // before it decremented. Measured on the DEFAULT one-lane-per-process device:
        // 2 BlockMovers busy where the device has 1.
        //
        // Phase first fixes that and keeps the within-op guarantees, because a leg's finish
        // is a release and the next leg's start is an acquire: the fire precedes its first
        // leg, a leg's finish precedes the next leg's start (they share a cycle, which is
        // what pipelining means), and the last finish precedes the op completing.
        std::stable_sort(events_.begin(), events_.end(),
                         [](const StepEvent& a, const StepEvent& b) {
            if (a.at != b.at) return a.at < b.at;
            if (phase(a.kind) != phase(b.kind)) return phase(a.kind) < phase(b.kind);
            if (a.op_index != b.op_index) return a.op_index < b.op_index;
            if (a.leg != b.leg) return a.leg < b.leg;
            return rank(a.kind) < rank(b.kind);
        });
    }

    bool step() override {
        if (next_ >= events_.size()) return false;
        cur_ = events_[next_++];
        switch (cur_.kind) {
            case StepKind::OpFired:     ++in_flight_; break;
            case StepKind::HopStarted:  if (cur_.hop) ++busy_[mover_of(*cur_.hop)]; break;
            case StepKind::HopFinished:
                if (cur_.hop) {
                    auto it = busy_.find(mover_of(*cur_.hop));
                    if (it != busy_.end() && it->second > 0) --it->second;
                }
                break;
            case StepKind::OpCompleted: if (in_flight_ > 0) --in_flight_; break;
            case StepKind::OpApplied:   break;
        }
        return true;
    }
    const StepEvent& current() const override { return cur_; }
    std::size_t position() const override { return next_; }
    std::size_t size() const override { return events_.size(); }
    bool models_time() const override { return true; }
    const std::map<Mover, std::size_t>& lanes_busy() const override { return busy_; }
    std::size_t in_flight() const override { return in_flight_; }

private:
    // Releases before acquires, which is the order the executor itself uses.
    static int phase(StepKind k) {
        switch (k) {
            case StepKind::HopFinished:
            case StepKind::OpCompleted: return 0;      // gives a lane back
            case StepKind::OpFired:
            case StepKind::HopStarted:
            case StepKind::OpApplied:   return 1;      // takes one
        }
        return 1;
    }
    static int rank(StepKind k) {
        switch (k) {
            case StepKind::OpFired:     return 0;
            case StepKind::HopStarted:  return 1;
            case StepKind::HopFinished: return 2;
            case StepKind::OpCompleted: return 3;
            case StepKind::OpApplied:   return 4;
        }
        return 5;
    }
    std::vector<StepEvent> events_;
    StepEvent cur_{};
    std::size_t next_ = 0;
    std::size_t in_flight_ = 0;
    std::map<Mover, std::size_t> busy_;
};

// ----------------------------------------------------------------------------
// The one place that maps a level to a stepper, mirroring run_at(). Behavioral
// stepping EXECUTES, so it takes the program; block-sequential replays, so it takes
// a finished run's timeline.
// ----------------------------------------------------------------------------
inline std::unique_ptr<Stepper> make_stepper(ExecutionLevel level, TileProgram& prog,
                                             const std::vector<TileOpRecord>& timeline) {
    switch (level) {
        case ExecutionLevel::Behavioral:
            return std::make_unique<BehavioralStepper>(prog);
        case ExecutionLevel::BlockSequential:
            return std::make_unique<BlockSequentialStepper>(timeline);
        case ExecutionLevel::ResourceTransactional:
        case ExecutionLevel::CycleAccurate:
            break;
    }
    throw std::invalid_argument(std::string("make_stepper: ") + to_string(level) + ": " +
                               not_implemented_reason(level));
}

// One line per step, for a terminal. `--step` prints these.
inline std::string describe(const StepEvent& e, bool with_cycle) {
    std::string s;
    if (with_cycle) {
        s = "@" + std::to_string(e.at);
        s.append(s.size() < 10 ? 10 - s.size() : 1, ' ');
    }
    s += std::string(to_string(e.kind));
    s.append(s.size() < (with_cycle ? 22u : 12u) ? (with_cycle ? 22u : 12u) - s.size() : 1, ' ');
    s += "op " + std::to_string(e.op_index) + " " + to_string(e.op_kind);
    if (e.hop) s += "  " + std::string(to_string(*e.hop)) + " lane " + std::to_string(e.lane);
    if (e.zero_work && e.kind == StepKind::OpCompleted) s += "  (zero work)";
    return s;
}

} // namespace sw::kpu::program::driver
