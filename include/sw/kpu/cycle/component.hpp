#pragma once
// ============================================================================
// include/sw/kpu/cycle/component.hpp
// A clocked component on the cycle core (docs/plans/cycle-based-kpu-prototype.md §3.1-3.2).
//
// On each edge of its domain's clock a component evaluates: it reads what has landed on its input
// links, pops, pushes where it holds a credit, and updates its own state. It never computes a
// future completion time; how long anything takes is the number of edges its state takes to get
// there. commit() runs after every component of the domain has evaluated, for a component that
// keeps registered state of its own (a pipeline register read by its own next evaluate).
//
// COUNTERS (counters.hpp). evaluate() reports the cycle's state -- counters().busy(), or
// counters().stall(cause); nothing reported is idle -- and each work item it accepts and
// finishes (accept(), complete()). The simulator closes the cycle.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/counters.hpp>
#include <sw/kpu/cycle/time.hpp>

#include <string>
#include <utility>

namespace sw::kpu::cycle {

class Component {
public:
    Component(std::string name, Domain& domain) : name_(std::move(name)), domain_(&domain) {}
    virtual ~Component() = default;
    Component(const Component&) = delete;
    Component& operator=(const Component&) = delete;

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] Domain& domain() const { return *domain_; }

    virtual void evaluate(Cycle now) = 0;
    virtual void commit(Cycle /*now*/) {}
    // Nothing left to do: no work held and none to start. The simulator's run_until_idle stops
    // when every component is idle and every link is empty.
    [[nodiscard]] virtual bool idle() const = 0;
    // Its own invariants, checked after every cycle of its domain; raise std::logic_error.
    virtual void check(Cycle /*now*/) const {}
    virtual void reset() {}
    // Its internal state in a line, for the state dump and the deadlock report (none by default).
    [[nodiscard]] virtual std::string describe() const { return {}; }

    [[nodiscard]] ComponentCounters& counters() { return counters_; }
    [[nodiscard]] const ComponentCounters& counters() const { return counters_; }
    [[nodiscard]] const Counters& counts() const { return counters_.counts(); }

private:
    std::string name_;
    Domain* domain_;
    ComponentCounters counters_;
};

}  // namespace sw::kpu::cycle
