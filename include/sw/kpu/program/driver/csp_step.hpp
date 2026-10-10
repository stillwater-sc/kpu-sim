// ============================================================================
// include/sw/kpu/program/driver/csp_step.hpp
// Stepping a CSP program (docs/plans/kpu-run-csp-programs.md step 4c).
//
// A step is the level's transaction, as for an L0 program (step_cursor.hpp), but over the
// program's ACTIONS rather than a trace's ops:
//
//   L-B   one action, applied: the ActionStream feeds csp::BehavioralInterpreter one action per
//         step, so values form as you step. Station occupancy (tiles held in L3, L2, the
//         fabric) is the interpreter's own, after the action.
//   L-T1  one record, replayed in start order: a leg of an action as the one-pass schedule
//         timed it (a Store has two legs, a Release one of zero length). The run is finished
//         before the first step, as for an L0 program at L-T1. Occupancy is AT THE RECORD'S
//         START CYCLE: lanes busy per process, and L3 slots held -- both from the run's
//         records and slots, so a step reports station occupancy, which the L0 executor could
//         not (it publishes only its peak).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/program/csp/transactional.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

namespace sw::kpu::program::driver {

struct CspStep {
    std::size_t action = 0;                 // the action's index in program order
    csp::Action::Kind kind = csp::Action::Kind::Call;
    TileCoord tile;
    std::vector<csp::Stage> context;        // L-B: the tile context the action carries
    // L-T1 only.
    std::optional<csp::TransactionalInterpreter::Proc> proc;   // none for a Release
    std::size_t lane = 0;
    std::uint64_t start = 0, finish = 0;
    std::size_t leg = 0;                    // a Store's second leg is 1
    // Station occupancy: L-B after the action; L-T1 at `start`.
    std::size_t l3_held = 0;
    std::size_t l2_held = 0, cf_held = 0;   // L-B only
};

class CspStepper {
public:
    virtual ~CspStepper() = default;
    virtual bool step() = 0;                        // false when there is nothing left
    virtual const CspStep& current() const = 0;
    virtual std::size_t position() const = 0;       // steps taken
    virtual std::size_t size() const = 0;           // total steps
    virtual bool executes() const = 0;              // true: a step applies work (L-B)
    // L-T1: lanes busy per process at the current step's start. Empty at L-B.
    virtual const std::map<std::string, std::size_t>& lanes_busy() const = 0;
};

// ----------------------------------------------------------------------------
// L-B: one action per step, applied for real.
// ----------------------------------------------------------------------------
class CspBehavioralStepper final : public CspStepper {
public:
    CspBehavioralStepper(const csp::lang::Program& ast, const TileProgram& inputs) {
        {
            csp::lang::ActionStream count(ast);         // the walk is lazy; counting is one pass
            while (count.next()) ++size_;
        }
        stream_ = std::make_unique<csp::lang::ActionStream>(ast);
        lb_.begin(inputs);
    }

    bool step() override {
        auto e = stream_->next();
        if (!e) {
            if (!finished_) {
                (void)lb_.finish();                     // DRAM's tiles back into operands
                finished_ = true;
            }
            return false;
        }
        lb_.step(e->action, e->op ? &*e->op : nullptr);
        cur_ = CspStep{};
        cur_.action = next_++;
        cur_.kind = e->action.kind;
        cur_.tile = e->action.tile;
        cur_.context = e->action.context;
        cur_.l3_held = lb_.held(csp::Chan::L3);
        cur_.l2_held = lb_.held(csp::Chan::L2);
        cur_.cf_held = lb_.held(csp::Chan::Cf);
        if (next_ == size_ && !finished_) {             // the last action: the values are final
            (void)lb_.finish();
            finished_ = true;
        }
        return true;
    }
    const CspStep& current() const override { return cur_; }
    std::size_t position() const override { return next_; }
    std::size_t size() const override { return size_; }
    bool executes() const override { return true; }
    const std::map<std::string, std::size_t>& lanes_busy() const override { return none_; }

    // The operands as the program has left them so far; final once every action has run.
    const TileProgram& values() const { return lb_.result(); }
    bool finished() const { return finished_; }

private:
    std::unique_ptr<csp::lang::ActionStream> stream_;
    csp::BehavioralInterpreter lb_;
    CspStep cur_{};
    std::size_t next_ = 0, size_ = 0;
    bool finished_ = false;
    std::map<std::string, std::size_t> none_;
};

// ----------------------------------------------------------------------------
// L-T1: the run's records, replayed in start order.
// ----------------------------------------------------------------------------
class CspReplayStepper final : public CspStepper {
public:
    using Record = csp::TransactionalInterpreter::Record;
    using Slot = csp::TransactionalInterpreter::Slot;

    CspReplayStepper(std::vector<Record> records, std::vector<Slot> slots)
        : records_(std::move(records)), slots_(std::move(slots)) {
        // A Store's legs are recorded in order, so the leg number is its rank within its action.
        legs_.resize(records_.size());
        std::map<std::size_t, std::size_t> seen;
        for (std::size_t i = 0; i < records_.size(); ++i) legs_[i] = seen[records_[i].action]++;
        order_.resize(records_.size());
        for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = i;
        // Start order; at one cycle, program order (and a Store's legs in order).
        std::stable_sort(order_.begin(), order_.end(), [&](std::size_t a, std::size_t b) {
            if (records_[a].start != records_[b].start) return records_[a].start < records_[b].start;
            return records_[a].action < records_[b].action;
        });
        by_start_ = order_;
        std::sort(slots_.begin(), slots_.end(), [](const Slot& a, const Slot& b) { return a.t0 < b.t0; });
    }

    bool step() override {
        if (next_ >= order_.size()) return false;
        const std::size_t i = order_[next_++];
        const Record& r = records_[i];
        advance(r.start);
        cur_ = CspStep{};
        cur_.action = r.action;
        cur_.kind = r.kind;
        cur_.tile = r.tile;
        if (csp::TransactionalInterpreter::has_process(r.kind)) cur_.proc = r.proc;
        cur_.lane = r.lane;
        cur_.start = r.start;
        cur_.finish = r.finish;
        cur_.leg = legs_[i];
        cur_.l3_held = held_.size();
        busy_.clear();
        for (const auto& [p, q] : busy_q_)
            if (!q.empty()) busy_[csp::TransactionalInterpreter::name(p)] = q.size();
        return true;
    }
    const CspStep& current() const override { return cur_; }
    std::size_t position() const override { return next_; }
    std::size_t size() const override { return order_.size(); }
    bool executes() const override { return false; }
    const std::map<std::string, std::size_t>& lanes_busy() const override { return busy_; }

private:
    using MinHeap = std::priority_queue<std::uint64_t, std::vector<std::uint64_t>, std::greater<std::uint64_t>>;

    // Occupancy is a function of the cycle, not of how many steps were taken: every leg and
    // every slot that has begun by `t` and not ended by it (an end at `t` frees before `t`'s
    // begins, as the schedule's own peak counts it).
    void advance(std::uint64_t t) {
        for (; admitted_ < by_start_.size() && records_[by_start_[admitted_]].start <= t; ++admitted_) {
            const Record& r = records_[by_start_[admitted_]];
            if (csp::TransactionalInterpreter::has_process(r.kind) && r.finish > r.start) busy_q_[r.proc].push(r.finish);
        }
        for (auto& [p, q] : busy_q_)
            while (!q.empty() && q.top() <= t) q.pop();
        for (; slot_next_ < slots_.size() && slots_[slot_next_].t0 <= t; ++slot_next_)
            if (slots_[slot_next_].t1 > slots_[slot_next_].t0) held_.push(slots_[slot_next_].t1);
        while (!held_.empty() && held_.top() <= t) held_.pop();
    }

    std::vector<Record> records_;
    std::vector<Slot> slots_;
    std::vector<std::size_t> legs_, order_, by_start_;
    std::size_t next_ = 0, admitted_ = 0, slot_next_ = 0;
    std::map<csp::TransactionalInterpreter::Proc, MinHeap> busy_q_;
    MinHeap held_;
    std::map<std::string, std::size_t> busy_;
    CspStep cur_{};
};

// One line per step, for a terminal. `--step` prints these.
inline std::string describe(const CspStep& s, bool timed) {
    std::string out;
    if (timed) {
        out = "@" + std::to_string(s.start);
        out.append(out.size() < 10 ? 10 - out.size() : 1, ' ');
    }
    out += "action " + std::to_string(s.action) + " " + csp::to_string(s.kind) + " " + s.tile.to_string();
    if (timed) {
        if (s.proc)
            out += "  " + std::string(csp::TransactionalInterpreter::name(*s.proc)) + " lane " +
                   std::to_string(s.lane) + "  -> @" + std::to_string(s.finish);
        if (s.kind == csp::Action::Kind::Store) out += s.leg == 0 ? "  (l3 -> dma buffer)" : "  (dma buffer -> dram)";
    }
    for (std::size_t i = 0; i < s.context.size(); ++i) {
        out += i ? ", " : "  via ";
        out += std::string(csp::to_string(s.context[i].op)) + " @ " + csp::to_string(s.context[i].place);
    }
    return out;
}

}  // namespace sw::kpu::program::driver
