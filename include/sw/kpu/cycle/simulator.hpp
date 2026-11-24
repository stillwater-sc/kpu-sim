#pragma once
// ============================================================================
// include/sw/kpu/cycle/simulator.hpp
// The cycle core's scheduler (docs/plans/cycle-based-kpu-prototype.md §3.2).
//
// The simulator owns the domains, components and links of a model, and clocks them: step() takes
// the earliest clock edge across every domain and clocks that domain's components and links
// (step() explains the cycle, and why it is cycle-based rather than event-driven).
//
// DEBUG (§3.3, step 0c):
//   - any exception a component or link raises while it is clocked or checked, and any failed
//     invariant hook (add_invariant), stops the run as an InvariantViolation: which component, link
//     or hook, the domain and the cycle, and the state dump at that moment;
//   - the DEADLOCK DETECTOR: a domain is ACTIVE at an edge if a component it clocks was busy or held
//     by an outside control (Stall::External), or a link it clocks pushed, popped, or still has a beat
//     in flight or a credit on its way back (timed transport under way). A domain stays active, or
//     not, until its next edge, so a busy component in a slow domain is not taken for stuck at the
//     fast domains' edges in between. A step makes progress if any domain is active. After
//     deadlock_window() steps in a row without progress while work remains (!idle()), the run stops
//     as a Deadlock, naming every component that is not idle and every link's credits. A long
//     latency is not a deadlock: a component waiting out its own latency reports busy, and a beat
//     on a slow link is in flight. A true credit cycle has every beat buffered and no credit
//     returning, so nothing is active;
//   - dump(): every domain, component and link, now;
//   - attach(Tracer&): the Perfetto trace (trace.hpp), with each domain's clock track.
//
// A link joins two components of ONE domain. Between domains, crossing() makes a clock-domain
// crossing (step 0d): an async FIFO with a synchronizer each way (link.hpp). The crossing is
// clocked, counted and checked with its sender's domain.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/time.hpp>
#include <sw/kpu/cycle/trace.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

// A run stopped by a failed invariant, or by an exception a component or link raised.
class InvariantViolation : public std::logic_error {
public:
    InvariantViolation(std::string where, std::string domain, Cycle cycle, const std::string& what, std::string dump)
        : std::logic_error(where + " (" + domain + " cycle " + std::to_string(cycle) + "): " + what),
          where_(std::move(where)), domain_(std::move(domain)), cycle_(cycle), dump_(std::move(dump)) {}
    [[nodiscard]] const std::string& where() const { return where_; }      // the component, link or hook
    [[nodiscard]] const std::string& domain() const { return domain_; }
    [[nodiscard]] Cycle cycle() const { return cycle_; }
    [[nodiscard]] const std::string& dump() const { return dump_; }      // the state when it stopped

private:
    std::string where_, domain_;
    Cycle cycle_;
    std::string dump_;
};

// A run stopped because nothing made progress for the deadlock window while work remained.
class Deadlock : public std::runtime_error {
public:
    Deadlock(const std::string& what, std::vector<std::string> blocked, std::string dump)
        : std::runtime_error(what), blocked_(std::move(blocked)), dump_(std::move(dump)) {}
    [[nodiscard]] const std::vector<std::string>& blocked() const { return blocked_; }   // components not idle
    [[nodiscard]] const std::string& dump() const { return dump_; }

private:
    std::vector<std::string> blocked_;
    std::string dump_;
};

class Simulator {
public:
    Domain& add_domain(Domain d) {
        domains_.push_back(std::make_unique<Domain>(std::move(d)));
        clocked_.push_back({});
        active_.push_back(false);
        clocks_.push_back(tracer_ ? tracer_->add_clock(domains_.back()->name()) : 0);
        return *domains_.back();
    }

    template <class C, class... Args>
    C& add(Args&&... args) {
        auto c = std::make_unique<C>(std::forward<Args>(args)...);
        C& ref = *c;
        Clocked& k = clocked_[require_domain(ref.domain(), "component " + ref.name())];
        k.components.push_back(&ref);
        k.tracks.push_back(tracer_ ? tracer_->add_track(ref.name(), ref.domain().name(), false) : 0);
        components_.push_back(std::move(c));
        return ref;
    }

    template <class T>
    Link<T>& link(std::string name, const Component& from, const Component& to, std::size_t capacity,
                  Cycle latency = 1, Cycle credit_latency = 1) {
        if (&from.domain() != &to.domain())
            throw std::invalid_argument("cycle::Simulator: link " + name + " joins " + from.name() + " (" +
                                        from.domain().name() + ") and " + to.name() + " (" + to.domain().name() +
                                        "); a link between clock domains is a crossing (crossing())");
        const std::size_t d = require_domain(from.domain(), "link " + name);   // one this simulator schedules
        return own(d, std::make_unique<Link<T>>(std::move(name), from.domain(), capacity, latency, credit_latency));
    }

    // A clock-domain crossing from `from`'s domain to `to`'s (link.hpp): an async FIFO of
    // `capacity` entries; a beat is seen at the receiver's `sync`-th edge after its push, and a
    // credit at the sender's `credit_sync`-th edge after its pop. Two stages each way by default,
    // as in a two-flop synchronizer.
    template <class T>
    Link<T>& crossing(std::string name, const Component& from, const Component& to, std::size_t capacity,
                      Cycle sync = 2, Cycle credit_sync = 2) {
        if (&from.domain() == &to.domain())
            throw std::invalid_argument("cycle::Simulator: crossing " + name + " joins " + from.name() + " and " +
                                        to.name() + ", both in " + from.domain().name() + "; within a domain, use link()");
        const std::size_t d = require_domain(from.domain(), "crossing " + name);
        require_domain(to.domain(), "crossing " + name);
        return own(d, std::make_unique<Link<T>>(std::move(name), from.domain(), to.domain(), capacity, sync, credit_sync));
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
    //        c. count: every component closes the cycle's counters (counters.hpp): its state;
    //        d. check: every component's invariants, after the cycle has settled. This is the debug
    //           guarantee of the plan (§3.3): a violation stops the run at the cycle it happens.
    //           set_checks(false) skips it, to measure its cost;
    //        e. the domain's clock advances to its next edge.
    //   3. Then, for every domain clocked at `t`, its links close the cycle: their counters (the
    //      items they hold into the occupancy integral) and their credit conservation (checked).
    //      After every domain at `t`, not within each: a crossing (link.hpp) is clocked with its
    //      sender, and its receiver, in another domain, may pop at the same instant.
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
        due_.clear();   // a step a failure interrupted leaves its domains here (#553's review)
        for (std::size_t i = 0; i < domains_.size(); ++i) {
            Domain& d = *domains_[i];
            if (d.next_edge() != t) continue;
            const Cycle c = d.cycle();
            const Clocked& k = clocked_[i];
            bool progress = false;
            for (Component* x : k.components) guarded(x->name(), d, c, [&] { x->evaluate(c); });
            for (Component* x : k.components) guarded(x->name(), d, c, [&] { x->commit(c); });
            for (std::size_t j = 0; j < k.components.size(); ++j) {
                ComponentCounters& n = k.components[j]->counters();
                const State s = n.end_cycle();
                if (s == State::Busy || (s == State::Stalled && n.last_cause() == Stall::External)) progress = true;
                if (tracer_) tracer_->state(k.tracks[j], s, n.last_cause(), d.edge(c), d.edge(c + 1));
            }
            active_[i] = progress;
            if (checks_)
                for (Component* x : k.components) guarded(x->name(), d, c, [&] { x->check(c); });
            d.advance();
            if (tracer_) tracer_->clock(clocks_[i], c, t, d.next_edge());
            last_domain_ = &d;
            due_.push_back({i, c});
        }
        // The links' cycles close once every domain with an edge at `t` has been clocked: a
        // crossing's receiver may pop at the instant its sender's cycle ends, and its counts and
        // conservation must not depend on which of the two domains was clocked first.
        for (const auto& [i, c] : due_) {
            const Domain& d = *domains_[i];
            bool moved = false;
            for (LinkBase* l : clocked_[i].links)
                if (l->end_cycle(c) || l->transporting(c)) moved = true;
            if (moved) active_[i] = true;
            if (checks_)
                for (LinkBase* l : clocked_[i].links) guarded(l->name(), d, c, [&] { l->check(c); });
        }
        if (checks_)
            for (const auto& [name, hook] : invariants_)
                guarded("invariant " + name, *last_domain_, last_domain_->cycle() - 1, [&] { hook(); });
        bool progress = false;
        for (const bool a : active_) progress = progress || a;
        if (progress) {
            quiet_ = 0;
        } else if (deadlock_window_ > 0 && ++quiet_ >= deadlock_window_ && !idle()) {
            deadlock();
        }
        return true;
    }

    // An invariant checked after every step (with the checks on): raise to fail it.
    void add_invariant(std::string name, std::function<void()> hook) {
        invariants_.emplace_back(std::move(name), std::move(hook));
    }
    // Edges in a row without progress before the run stops as a Deadlock; 0 turns the detector off.
    void set_deadlock_window(std::uint64_t edges) { deadlock_window_ = edges; }
    [[nodiscard]] std::uint64_t deadlock_window() const { return deadlock_window_; }

    // The trace: attach before adding components and links (each gets a track as it is added).
    // Every domain gets a clock track, the domains added so far included.
    void attach(Tracer& t) {
        if (!components_.empty() || !links_.empty())
            throw std::logic_error("cycle::Simulator: attach the tracer before adding components and links");
        tracer_ = &t;
        for (std::size_t i = 0; i < domains_.size(); ++i) clocks_[i] = t.add_clock(domains_[i]->name());
    }

    // Every domain, component and link, as they are now. While a domain's cycle is being clocked
    // (`interrupted`, a failure in it), its links are shown at that cycle, not the one before.
    [[nodiscard]] std::string dump(const Domain* interrupted = nullptr, Cycle at = 0) const {
        std::ostringstream o;
        o << "cycle core at " << now_ << " ps\n";
        for (std::size_t i = 0; i < domains_.size(); ++i) {
            const Domain& d = *domains_[i];
            o << "domain " << d.name() << ": cycle " << d.cycle() << ", next edge " << d.next_edge() << " ps\n";
            const Cycle c = &d == interrupted ? at : (d.cycle() ? d.cycle() - 1 : 0);   // the last cycle clocked
            for (const Component* x : clocked_[i].components) {
                const Counters& n = x->counts();
                o << "  component " << x->name() << ": " << describe_state(*x) << "; A " << n.arrivals << ", C "
                  << n.completions << ", held " << n.occupancy;
                const std::string more = x->describe();
                if (!more.empty()) o << "; " << more;
                o << "\n";
            }
            for (LinkBase* l : clocked_[i].links) o << "  link " << l->name() << ": " << l->describe(c) << "\n";
        }
        return o.str();
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
    [[nodiscard]] Tracer* tracer() const { return tracer_; }
    // The per-cycle invariant checks, on by default. Off only to measure what they cost.
    void set_checks(bool on) { checks_ = on; }
    [[nodiscard]] bool checks() const { return checks_; }

    // Every domain: its name, period in picoseconds, and the cycles clocked so far.
    template <class F>
    void for_each_domain(F&& f) const {
        for (const auto& d : domains_) f(d->name(), d->period_ps(), d->cycle());
    }

    // The counters of every component and link, in the order added (report.hpp formats them).
    template <class F>
    void for_each_counted(F&& f) const {
        for (const auto& x : components_) f(x->name(), x->domain().name(), false, x->counts());
        for (const auto& l : links_) f(l->name(), l->domain().name(), true, l->counts());
    }

    void reset() {
        due_.clear();
        for (const auto& d : domains_) d->reset();
        for (const auto& x : components_) {
            x->reset();
            x->counters().reset();
        }
        for (const auto& l : links_) l->reset();
        now_ = 0;
        quiet_ = 0;
        active_.assign(active_.size(), false);
    }

private:
    template <class T>
    Link<T>& own(std::size_t d, std::unique_ptr<Link<T>> l) {
        Link<T>& ref = *l;
        clocked_[d].links.push_back(&ref);
        links_.push_back(std::move(l));
        if (tracer_) ref.attach(tracer_, tracer_->add_track(ref.name(), domain_label(ref), true));
        return ref;
    }
    static std::string domain_label(const LinkBase& l) {
        return l.crossing() ? l.domain().name() + " -> " + l.rx_domain().name() : l.domain().name();
    }

    // The index of `d` among this simulator's domains; refuses one it does not own.
    std::size_t require_domain(const Domain& d, const std::string& what) const {
        for (std::size_t i = 0; i < domains_.size(); ++i) if (domains_[i].get() == &d) return i;
        throw std::invalid_argument("cycle::Simulator: " + what + " is in domain " + d.name() +
                                    ", which this simulator does not own (add_domain first)");
    }

    // Each domain's components and links, in the order added: what one of its edges clocks.
    struct Clocked {
        std::vector<Component*> components;
        std::vector<std::uint32_t> tracks;               // each component's trace track (when traced)
        std::vector<LinkBase*> links;
    };

    // Runs f; any exception it raises stops the run as an InvariantViolation naming `where`.
    template <class F>
    void guarded(const std::string& where, const Domain& d, Cycle c, F&& f) const {
        try {
            f();
        } catch (const InvariantViolation&) {
            throw;
        } catch (const std::exception& e) {
            throw InvariantViolation(where, d.name(), c, e.what(), dump(&d, c));
        }
    }
    static std::string describe_state(const Component& x) {
        const ComponentCounters& n = x.counters();
        if (n.last_state() == State::Stalled) return std::string("stalled (") + to_string(n.last_cause()) + ")";
        return to_string(n.last_state());
    }
    [[noreturn]] void deadlock() const {
        std::vector<std::string> blocked;
        std::ostringstream o;
        o << "cycle::Simulator: deadlock: no progress for " << quiet_ << " edges, at " << now_ << " ps, with work left;";
        for (std::size_t i = 0; i < domains_.size(); ++i)
            for (const Component* x : clocked_[i].components) {
                if (x->idle()) continue;
                blocked.push_back(x->name());
                o << " " << x->name() << " " << describe_state(*x) << ";";
            }
        for (std::size_t i = 0; i < domains_.size(); ++i) {
            const Cycle c = domains_[i]->cycle() ? domains_[i]->cycle() - 1 : 0;
            for (LinkBase* l : clocked_[i].links) o << " link " << l->name() << " " << l->describe(c) << ";";
        }
        throw Deadlock(o.str(), std::move(blocked), dump());
    }

    std::vector<std::unique_ptr<Domain>> domains_;
    std::vector<Clocked> clocked_;                       // parallel to domains_
    std::vector<std::uint32_t> clocks_;                  // each domain's clock track (when traced)
    bool checks_ = true;
    std::vector<std::pair<std::string, std::function<void()>>> invariants_;
    std::uint64_t deadlock_window_ = 10'000;            // edges without progress (0: off)
    std::uint64_t quiet_ = 0;                            // steps in a row without progress
    std::vector<bool> active_;                           // each domain's activity at its last edge
    std::vector<std::pair<std::size_t, Cycle>> due_;     // the domains clocked this step, and their cycles
    const Domain* last_domain_ = nullptr;
    Tracer* tracer_ = nullptr;
    std::vector<std::unique_ptr<Component>> components_;
    std::vector<std::unique_ptr<LinkBase>> links_;
    Picoseconds now_ = 0;
};

}  // namespace sw::kpu::cycle
