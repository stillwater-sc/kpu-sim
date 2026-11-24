#pragma once
// ============================================================================
// include/sw/kpu/cycle/simulator.hpp
// The cycle core's scheduler (docs/plans/cycle-based-kpu-prototype.md §3.2).
//
// The simulator owns the domains, components and links of a model, and clocks them: step() takes
// the earliest clock edge across every domain and clocks that domain's components and links
// (step() explains the cycle, and why it is cycle-based rather than event-driven).
//
// A link joins two components of ONE domain. A link between domains is a clock-domain crossing,
// with its own synchronizer (step 0d), and is refused until then.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/time.hpp>

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

class Simulator {
public:
    Domain& add_domain(Domain d) {
        domains_.push_back(std::make_unique<Domain>(std::move(d)));
        clocked_.push_back({});
        return *domains_.back();
    }

    template <class C, class... Args>
    C& add(Args&&... args) {
        auto c = std::make_unique<C>(std::forward<Args>(args)...);
        C& ref = *c;
        clocked_[require_domain(ref.domain(), "component " + ref.name())].components.push_back(&ref);
        components_.push_back(std::move(c));
        return ref;
    }

    template <class T>
    Link<T>& link(std::string name, const Component& from, const Component& to, std::size_t capacity,
                  Cycle latency = 1, Cycle credit_latency = 1) {
        if (&from.domain() != &to.domain())
            throw std::invalid_argument("cycle::Simulator: link " + name + " joins " + from.name() + " (" +
                                        from.domain().name() + ") and " + to.name() + " (" + to.domain().name() +
                                        "); a link between clock domains is a crossing (step 0d)");
        const std::size_t d = require_domain(from.domain(), "link " + name);   // one this simulator schedules
        auto l = std::make_unique<Link<T>>(std::move(name), from.domain(), capacity, latency, credit_latency);
        Link<T>& ref = *l;
        clocked_[d].links.push_back(&ref);
        links_.push_back(std::move(l));
        return ref;
    }

    // ONE SIMULATION STEP: the earliest clock edge across every domain. False when there is no
    // domain.
    //
    //   1. Find the edge. Each domain knows its next edge in picoseconds (Domain::next_edge(), kept
    //      exactly by advance() with no multiply or divide). The earliest is `t`. Domains are few
    //      (array, L2 banks, NoC, DRAM: three to five), so a linear scan finds it as fast as a heap
    //      would; a priority queue of domains pays off only with many more.
    //   2. Clock every domain whose edge is at `t`, in the order the domains were added (a fixed
    //      order for edges that coincide, so runs are deterministic). For each, over ITS OWN
    //      components and links only (bucketed when they were added, so nothing is filtered here):
    //        a. evaluate: every component reads what has landed on its links, pops, pushes where it
    //           holds a credit, and updates its state. A push lands >= 1 cycle later and a credit
    //           returns >= 1 cycle later, so no component sees another's work of this cycle, and the
    //           order of this loop cannot change the result (link.hpp);
    //        b. commit: components with registered state of their own latch it;
    //        c. check: every link's credit conservation and every component's invariants, after the
    //           cycle has settled. This is the debug guarantee of the plan (§3.3): a violation stops
    //           the run at the cycle it happens;
    //        d. the domain's clock advances to its next edge.
    //   A step costs O(the components and links of the domains clocked at `t`).
    //
    // WHY CYCLE-BASED, NOT AN EVENT QUEUE PER COMPONENT. An event-driven simulator keeps a priority
    // queue of (time, component) wake-ups and runs only components with something to do. That
    // wins when activity is sparse. Here, by design (§3.1), every clocked component is evaluated on
    // every edge of its domain, because:
    //   - in the machine this models, the busy parts (the PE array, the L1 lanes, the L2 banks, the
    //     Streamer) do work nearly every cycle under load, so there is little to skip where the
    //     cycles are;
    //   - skipping a component requires it to declare exactly when it next needs to run. A wrong
    //     declaration changes timing silently -- the class of error this core exists to remove
    //     (the old executor's processes computed their completion times);
    //   - the operational-analysis counters (step 0b) account busy and stalled cycles per cycle,
    //     which is direct here and interval arithmetic under skipping.
    // If profiling a subsystem shows idle components dominating (DRAM between bursts, say), the
    // remedy is an opt-in sleep: a component that reports idle with nothing landed on its inputs
    // is skipped until a link delivers to it -- added then, measured, and tested against this loop
    // for identical results.
    bool step() {
        if (domains_.empty()) return false;
        Picoseconds t = std::numeric_limits<Picoseconds>::max();
        for (const auto& d : domains_) t = std::min(t, d->next_edge());
        now_ = t;
        for (std::size_t i = 0; i < domains_.size(); ++i) {
            Domain& d = *domains_[i];
            if (d.next_edge() != t) continue;
            const Cycle c = d.cycle();
            const Clocked& k = clocked_[i];
            for (Component* x : k.components) x->evaluate(c);
            for (Component* x : k.components) x->commit(c);
            for (LinkBase* l : k.links) l->check(c);
            for (Component* x : k.components) x->check(c);
            d.advance();
        }
        return true;
    }

    // Steps until `done()` holds after an edge, or past `limit` picoseconds; true if done.
    bool run_until(const std::function<bool()>& done, Picoseconds limit) {
        while (!done()) {
            if (!step() || now_ > limit) return false;
        }
        return true;
    }
    // Steps until every component is idle and every link is empty, or past `limit`.
    bool run_until_idle(Picoseconds limit) {
        return run_until([this] { return idle(); }, limit);
    }
    [[nodiscard]] bool idle() const {
        for (const auto& x : components_) if (!x->idle()) return false;
        for (const auto& l : links_) if (!l->empty()) return false;
        return true;
    }

    [[nodiscard]] Picoseconds now() const { return now_; }     // the last edge taken
    void reset() {
        for (const auto& d : domains_) d->reset();
        for (const auto& x : components_) x->reset();
        for (const auto& l : links_) l->reset();
        now_ = 0;
    }

private:
    // The index of `d` among this simulator's domains; refuses one it does not own.
    std::size_t require_domain(const Domain& d, const std::string& what) const {
        for (std::size_t i = 0; i < domains_.size(); ++i) if (domains_[i].get() == &d) return i;
        throw std::invalid_argument("cycle::Simulator: " + what + " is in domain " + d.name() +
                                    ", which this simulator does not own (add_domain first)");
    }

    // Each domain's components and links, in the order added: what one of its edges clocks.
    struct Clocked {
        std::vector<Component*> components;
        std::vector<LinkBase*> links;
    };

    std::vector<std::unique_ptr<Domain>> domains_;
    std::vector<Clocked> clocked_;                       // parallel to domains_
    std::vector<std::unique_ptr<Component>> components_;
    std::vector<std::unique_ptr<LinkBase>> links_;
    Picoseconds now_ = 0;
};

}  // namespace sw::kpu::cycle
