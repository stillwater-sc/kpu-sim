#pragma once
// ============================================================================
// include/sw/kpu/cycle/counters.hpp
// Operational-analysis counters on the cycle core (docs/plans/cycle-based-kpu-prototype.md §3.3;
// the XUE observation architecture, docs/xue-observation-architecture.md).
//
// Every component and every link counts, exactly, over the T cycles of its domain it was clocked:
//   A   arrivals: work items it accepted (a link: beats pushed)
//   C   completions: work items it finished (a link: beats popped)
//   the occupancy integral: the items it held, summed at the end of every cycle (Little's law)
//   and, for a component, ONE STATE PER CYCLE, so that busy + every stall + idle = T:
//     busy        it made progress, or holds work that is progressing
//     stalled     it has work it cannot advance, by cause (no credit downstream, no data from
//                 upstream on a transfer it started, an arbitration lost, a bank busy, or held by
//                 an outside control)
//     idle        it has nothing to do
//
// From these, the operational laws (no probability, only counts):
//   X = C / T        throughput             U = B / T        utilization (productive)
//   S = B / C        service time           N = Q / T        mean occupancy (Q the integral)
//   R = Q / C        mean residence         N = X R          Little's law, exact at quiescence
//
// The counters are plain integers: the core is single-threaded and deterministic, so XUE's
// atomic EventStats would put a locked instruction in the hottest loop. to_xue() (xue_export.hpp)
// copies them into XUE's EventStats for its tooling.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/time.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sw::kpu::cycle {

enum class Stall : std::uint8_t { NoCredit, NoData, Arbitration, BankBusy, External };
inline constexpr std::size_t kStallCauses = 5;

inline const char* to_string(Stall s) {
    switch (s) {
        case Stall::NoCredit: return "no credit";
        case Stall::NoData: return "no data";
        case Stall::Arbitration: return "arbitration";
        case Stall::BankBusy: return "bank busy";
        case Stall::External: return "held";
    }
    return "?";
}

struct Counters {
    Cycle cycles = 0;                         // T: the cycles counted
    std::uint64_t arrivals = 0;               // A
    std::uint64_t completions = 0;            // C
    std::uint64_t occupancy = 0;              // items held now (A - C for a component that holds)
    std::uint64_t occupancy_integral = 0;     // Q: occupancy summed at the end of every cycle
    std::uint64_t max_occupancy = 0;
    Cycle busy = 0;                           // B
    std::array<Cycle, kStallCauses> stalls{};
    Cycle idle = 0;

    [[nodiscard]] Cycle stalled() const {
        Cycle s = 0;
        for (const Cycle x : stalls) s += x;
        return s;
    }
    [[nodiscard]] Cycle stalled(Stall s) const { return stalls[static_cast<std::size_t>(s)]; }

    // The laws, from the counts. Zero where the denominator is.
    [[nodiscard]] double throughput() const { return cycles ? double(completions) / double(cycles) : 0.0; }
    [[nodiscard]] double utilization() const { return cycles ? double(busy) / double(cycles) : 0.0; }
    [[nodiscard]] double service_time() const { return completions ? double(busy) / double(completions) : 0.0; }
    [[nodiscard]] double mean_occupancy() const { return cycles ? double(occupancy_integral) / double(cycles) : 0.0; }
    [[nodiscard]] double mean_residence() const {
        return completions ? double(occupancy_integral) / double(completions) : 0.0;
    }

    void reset() { *this = Counters{}; }
};

// What a component did this cycle: the one state its evaluate() reports. The last report of a
// cycle stands; none is idle.
enum class State : std::uint8_t { Idle, Busy, Stalled };

class ComponentCounters {
public:
    // The component's evaluate() reports its state for this cycle.
    void busy() { state_ = State::Busy; }
    void stall(Stall cause) {
        state_ = State::Stalled;
        cause_ = cause;
    }
    // A work item accepted (held from now), or finished (no longer held).
    void accept() {
        ++c_.arrivals;
        ++c_.occupancy;
    }
    void complete() {
        if (c_.occupancy == 0) throw std::logic_error("cycle::Counters: a completion with nothing held");
        ++c_.completions;
        --c_.occupancy;
    }
    // A completion of an item the component never held (a source: it creates and sends at once).
    void produce() { ++c_.completions; }

    // The simulator closes every cycle: tally its state and its occupancy.
    void end_cycle() {
        ++c_.cycles;
        switch (state_) {
            case State::Busy: ++c_.busy; break;
            case State::Stalled: ++c_.stalls[static_cast<std::size_t>(cause_)]; break;
            case State::Idle: ++c_.idle; break;
        }
        c_.occupancy_integral += c_.occupancy;
        if (c_.occupancy > c_.max_occupancy) c_.max_occupancy = c_.occupancy;
        state_ = State::Idle;
    }

    [[nodiscard]] const Counters& counts() const { return c_; }
    void reset() {
        c_.reset();
        state_ = State::Idle;
    }

private:
    Counters c_;
    State state_ = State::Idle;
    Stall cause_ = Stall::NoCredit;
};

}  // namespace sw::kpu::cycle
