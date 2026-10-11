// ============================================================================
// include/sw/kpu/program/driver/step_cursor.hpp
// Single-stepping an L0 program (design note §D4).
//
// An L0 program is a trace (ADR 0004 §4) and runs at L-B only, so its one stepper is
// behavioral: it holds the kernel state and applies one op per step, so values are
// observable as they form. The timed levels step the CSP program the trace came from
// (csp_step.hpp): L-T1 replays its records, and L-CA's step is a cycle (#283). The L0 L-T1
// replay stepper retired with the L0 L-T1 executor (kpu-run-csp-programs step 4e).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/tile_kernels.hpp>
#include <sw/kpu/program/tile_program.hpp>

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
};

inline const char* to_string(StepKind k) {
    switch (k) {
        case StepKind::OpApplied:   return "applied";
    }
    return "?";
}

struct StepEvent {
    Cycle at = 0;                     // cycle; meaningless at L-B, which models no time
    StepKind kind = StepKind::OpApplied;
    std::size_t op_index = 0;
    TileOpKind op_kind{};
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

private:
    TileProgram& prog_;
    TileKernelState state_;
    StepEvent cur_{};
    std::size_t next_ = 0;
};

// ----------------------------------------------------------------------------
// The one place that maps a level to a stepper, mirroring run_at(). Behavioral
// stepping EXECUTES, so it takes the program.
// ----------------------------------------------------------------------------
inline std::unique_ptr<Stepper> make_stepper(ExecutionLevel level, TileProgram& prog) {
    if (level == ExecutionLevel::Behavioral) return std::make_unique<BehavioralStepper>(prog);
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
    return s;
}

} // namespace sw::kpu::program::driver
