#pragma once
// ============================================================================
// include/sw/kpu/cycle/simulator.hpp
// The cycle core's scheduler (docs/plans/cycle-based-kpu-prototype.md §3.2).
//
// The simulator owns the domains, components and links of a model. step() takes the earliest
// clock edge across every domain (in picoseconds). Domains whose edges fall at the same instant
// are taken in the order they were added. For each such domain:
//   1. every component in it evaluates, in the order added;
//   2. every component commits;
//   3. every link and component in it checks its invariants;
//   4. the domain's clock advances.
// The order of components within a domain does not change the result (link.hpp).
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
        return *domains_.back();
    }

    template <class C, class... Args>
    C& add(Args&&... args) {
        auto c = std::make_unique<C>(std::forward<Args>(args)...);
        C& ref = *c;
        require_domain(ref.domain(), "component " + ref.name());
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
        auto l = std::make_unique<Link<T>>(std::move(name), from.domain(), capacity, latency, credit_latency);
        Link<T>& ref = *l;
        links_.push_back(std::move(l));
        return ref;
    }

    // Takes the earliest edge across every domain; false when there is no domain.
    bool step() {
        if (domains_.empty()) return false;
        Picoseconds t = std::numeric_limits<Picoseconds>::max();
        for (const auto& d : domains_) t = std::min(t, d->next_edge());
        now_ = t;
        for (const auto& d : domains_) {
            if (d->next_edge() != t) continue;
            const Cycle c = d->cycle();
            for (const auto& x : components_) if (&x->domain() == d.get()) x->evaluate(c);
            for (const auto& x : components_) if (&x->domain() == d.get()) x->commit(c);
            for (const auto& l : links_) if (&l->domain() == d.get()) l->check(c);
            for (const auto& x : components_) if (&x->domain() == d.get()) x->check(c);
            d->advance();
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
    void require_domain(const Domain& d, const std::string& what) const {
        for (const auto& x : domains_) if (x.get() == &d) return;
        throw std::invalid_argument("cycle::Simulator: " + what + " is in domain " + d.name() +
                                    ", which this simulator does not own (add_domain first)");
    }

    std::vector<std::unique_ptr<Domain>> domains_;
    std::vector<std::unique_ptr<Component>> components_;
    std::vector<std::unique_ptr<LinkBase>> links_;
    Picoseconds now_ = 0;
};

}  // namespace sw::kpu::cycle
