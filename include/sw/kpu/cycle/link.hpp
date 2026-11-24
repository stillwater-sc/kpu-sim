#pragma once
// ============================================================================
// include/sw/kpu/cycle/link.hpp
// A point-to-point channel with credit-based flow control (docs/plans/cycle-based-kpu-prototype.md
// §3.1-3.2): credits up, data down.
//
// A link has a CAPACITY: the receiver's buffer slots, which the sender holds as credits. The
// sender may push only while it holds a credit, and the push spends it. A pushed item travels
// `latency` cycles and lands in the receiver's buffer. When the receiver pops it, the slot is
// free, and its credit travels `credit_latency` cycles back to the sender.
//
// WHY THE ORDER OF COMPONENTS DOES NOT MATTER. Both latencies are at least one cycle, so an item
// pushed in cycle c can be popped no earlier than c + 1, and a credit returned in cycle c can be
// spent no earlier than c + 1. Nothing one component does in a cycle is seen by another in the
// same cycle, whichever evaluates first. Components meet only through links, so every run is
// deterministic and independent of the order they were added in.
//
// A CLOCK-DOMAIN CROSSING (step 0d) is a link whose sender and receiver are clocked by different
// domains: an async FIFO. The sender pushes on its edges and the receiver pops on its own. A beat
// pushed at the sender's edge t is visible to the receiver at its `sync`-th edge strictly after t
// (the synchronizer that carries the FIFO's write pointer across). A beat popped at the receiver's
// edge t returns its credit at the sender's `credit_sync`-th edge strictly after t (the read
// pointer's synchronizer back). Both are at least 1, so, as on any link, nothing crosses at the
// instant it is sent: two domains whose edges coincide see nothing of each other's work at that
// instant, whichever the scheduler clocks first, and the run stays deterministic. A link within one
// domain is the case where the two domains are one: `latency` edges after, `credit_latency` back.
// The link's own cycles, for its counters, its checks and the arguments of credits(), in_flight()
// and the rest, are the SENDER's: push() and can_push() take the sender's cycle, can_pop(), peek()
// and pop() the receiver's. The simulator closes a link's cycle after every domain with an edge at
// that instant has been clocked, so a crossing's counts include the receiver's pops at an instant
// both clocks share, whichever domain is clocked first. Its occupancy is as of the sender's last
// edge: after a fast receiver's pops, a slow sender's count catches up at its next edge.
//
// CONSERVATION, every cycle: credits held + items in flight + items buffered + credits in flight
// = capacity. check() raises if it does not hold.
//
// COUNTERS (counters.hpp): A = items pushed, C = items popped, the occupancy integral of items in
// the link (in flight or buffered), and busy = the cycles it carried a push (its use as a channel).
// The sender, not the link, reports a stall for want of a credit.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/counters.hpp>
#include <sw/kpu/cycle/time.hpp>
#include <sw/kpu/cycle/trace.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

// One beat of data: real bytes at an address, and what they belong to.
struct Beat {
    std::uint64_t id = 0;               // unique per source, for tracing and loss/duplication checks
    std::uint64_t address = 0;
    std::vector<std::uint8_t> bytes;
    std::uint64_t tag = 0;              // the tile, slot or command it belongs to
    bool first = false, last = false;   // a transfer's first and last beat
};

class LinkBase {
public:
    // `rx`: the receiver's domain, for a crossing (`latency` and `credit_latency` are then its two
    // synchronizers' stages); null for a link within `domain`.
    LinkBase(std::string name, const Domain& domain, std::size_t capacity, Cycle latency, Cycle credit_latency,
             const Domain* rx = nullptr)
        : name_(std::move(name)), domain_(&domain), rx_(rx ? rx : &domain), capacity_(capacity), latency_(latency),
          credit_latency_(credit_latency), credits_(capacity) {
        if (capacity_ == 0) throw std::invalid_argument("cycle::Link " + name_ + ": capacity must be at least 1");
        if (latency_ == 0 || credit_latency_ == 0)
            throw std::invalid_argument("cycle::Link " + name_ + ": latencies must be at least 1 cycle");
    }
    virtual ~LinkBase() = default;
    LinkBase(const LinkBase&) = delete;
    LinkBase& operator=(const LinkBase&) = delete;

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] const Domain& domain() const { return *domain_; }        // the sender's: the link's cycles
    [[nodiscard]] const Domain& rx_domain() const { return *rx_; }         // the receiver's
    [[nodiscard]] bool crossing() const { return rx_ != domain_; }
    [[nodiscard]] std::size_t capacity() const { return capacity_; }
    [[nodiscard]] Cycle latency() const { return latency_; }
    [[nodiscard]] Cycle credit_latency() const { return credit_latency_; }

    // The sender's credits now (returned credits that have arrived by `now` included).
    [[nodiscard]] std::size_t credits(Cycle now) {
        settle(now);
        return credits_;
    }
    [[nodiscard]] virtual std::size_t in_flight(Cycle now) const = 0;   // pushed, not landed by `now`
    [[nodiscard]] virtual std::size_t buffered(Cycle now) const = 0;    // landed by `now`, not popped
    [[nodiscard]] std::size_t credits_returning() const { return returning_.size(); }
    [[nodiscard]] virtual bool empty() const = 0;                // nothing in flight or buffered

    // Conservation; raises std::logic_error naming the link and the cycle.
    void check(Cycle now) {
        settle(now);
        const std::size_t sum = credits_ + in_flight(now) + buffered(now) + returning_.size();
        if (sum != capacity_)
            throw std::logic_error("cycle::Link " + name_ + " at cycle " + std::to_string(now) + ": credits " +
                                   std::to_string(credits_) + " + in flight " + std::to_string(in_flight(now)) +
                                   " + buffered " + std::to_string(buffered(now)) + " + returning " +
                                   std::to_string(returning_.size()) + " != capacity " + std::to_string(capacity_));
        if (counts_.arrivals - counts_.completions != held())
            throw std::logic_error("cycle::Link " + name_ + " at cycle " + std::to_string(now) + ": " +
                                   std::to_string(counts_.arrivals) + " pushed - " + std::to_string(counts_.completions) +
                                   " popped != " + std::to_string(held()) + " held");
    }

    virtual void reset() {
        credits_ = capacity_;
        returning_.clear();
        counts_.reset();
        pushed_this_cycle_ = popped_this_cycle_ = false;
    }

    // The simulator closes every cycle `now` of the link's domain. True if the link moved anything
    // this cycle -- a push or a pop -- which is progress to the deadlock detector.
    bool end_cycle(Cycle now) {
        ++counts_.cycles;
        if (pushed_this_cycle_) ++counts_.busy;
        else ++counts_.idle;
        counts_.occupancy = held();
        counts_.occupancy_integral += counts_.occupancy;
        if (counts_.occupancy > counts_.max_occupancy) counts_.max_occupancy = counts_.occupancy;
        if (tracer_) tracer_->occupancy(track_, domain_->edge(now), counts_.occupancy);
        const bool active = pushed_this_cycle_ || popped_this_cycle_;
        pushed_this_cycle_ = popped_this_cycle_ = false;
        return active;
    }
    [[nodiscard]] const Counters& counts() const { return counts_; }

    void attach(Tracer* t, std::uint32_t track) {
        tracer_ = t;
        track_ = track;
    }
    // Credits, in flight, buffered and returning, at `now`: for the state dump and the deadlock report.
    // Timed transport still under way at `now`: a beat in flight or a credit on its way back. While
    // there is, the link is not stuck, whatever its ends do (the deadlock detector's rule).
    [[nodiscard]] bool transporting(Cycle now) {
        settle(now);
        return in_flight(now) > 0 || !returning_.empty();
    }
    [[nodiscard]] std::string describe(Cycle now) {
        settle(now);
        std::ostringstream o;
        if (crossing()) o << "crossing " << domain_->name() << " -> " << rx_->name() << ", ";
        o << "credits " << credits_ << "/" << capacity_ << ", in flight " << in_flight(now) << ", buffered "
          << buffered(now) << ", returning " << returning_.size();
        return o.str();
    }

protected:
    [[nodiscard]] virtual std::size_t held() const = 0;   // in flight + buffered
    void counted_push() {
        ++counts_.arrivals;
        pushed_this_cycle_ = true;
    }
    void counted_pop() {
        ++counts_.completions;
        popped_this_cycle_ = true;
    }
    // A popped item's stay, from its push at the sender's cycle `pushed` to its pop at the
    // receiver's cycle `now`.
    void traced_pop(std::uint64_t id, Cycle pushed, Cycle now) {
        if (tracer_) tracer_->beat(track_, id, domain_->edge(pushed), rx_->edge(now));
    }
    // The receiver's cycle at which an item pushed at the sender's cycle `now` lands.
    [[nodiscard]] Cycle arrival(Cycle now) const {
        if (!crossing()) return now + latency_;
        return rx_->first_after(domain_->edge(now)) + latency_ - 1;
    }
    // The receiver's first cycle after the sender's cycle `now`: items landing before it have
    // landed by `now`.
    [[nodiscard]] Cycle horizon(Cycle now) const {
        if (!crossing()) return now + 1;
        return rx_->first_after(domain_->edge(now));
    }
    void spend_credit(Cycle now) {
        settle(now);
        if (credits_ == 0)
            throw std::logic_error("cycle::Link " + name_ + " at cycle " + std::to_string(now) +
                                   ": push without a credit");
        --credits_;
    }
    // A credit freed at the receiver's cycle `now` starts back; it arrives at a sender's cycle.
    void return_credit(Cycle now) {
        if (!crossing()) returning_.push_back(now + credit_latency_);
        else returning_.push_back(domain_->first_after(rx_->edge(now)) + credit_latency_ - 1);
    }
    // Credits whose return has arrived by `now` are the sender's again.
    void settle(Cycle now) {
        while (!returning_.empty() && returning_.front() <= now) {
            returning_.pop_front();
            ++credits_;
        }
    }

private:
    std::string name_;
    const Domain* domain_;
    const Domain* rx_;                  // the receiver's domain: domain_ but for a crossing
    std::size_t capacity_;
    Cycle latency_, credit_latency_;
    std::size_t credits_;
    std::deque<Cycle> returning_;       // the sender's cycle each credit on its way back arrives at
    Counters counts_;
    bool pushed_this_cycle_ = false, popped_this_cycle_ = false;
    Tracer* tracer_ = nullptr;
    std::uint32_t track_ = 0;
};

template <class T>
class Link : public LinkBase {
public:
    Link(std::string name, const Domain& domain, std::size_t capacity, Cycle latency = 1, Cycle credit_latency = 1)
        : LinkBase(std::move(name), domain, capacity, latency, credit_latency) {}
    // A clock-domain crossing from `tx` to `rx`: an async FIFO of `capacity` entries, `sync`
    // receiver edges to see a beat and `credit_sync` sender edges to see a credit come back.
    Link(std::string name, const Domain& tx, const Domain& rx, std::size_t capacity, Cycle sync, Cycle credit_sync)
        : LinkBase(std::move(name), tx, capacity, sync, credit_sync, &rx) {}

    // Sender side, in the sender's cycles.
    [[nodiscard]] bool can_push(Cycle now) { return credits(now) > 0; }
    void push(Cycle now, T item) {
        spend_credit(now);
        pipe_.push_back({arrival(now), now, std::move(item)});
        counted_push();
    }

    // Receiver side, in the receiver's cycles: the oldest landed item.
    [[nodiscard]] bool can_pop(Cycle now) const { return !pipe_.empty() && pipe_.front().arrival <= now; }
    [[nodiscard]] const T& peek(Cycle now) const {
        if (!can_pop(now)) throw std::logic_error("cycle::Link " + name() + ": peek with nothing landed");
        return pipe_.front().item;
    }
    // Pops the oldest landed item; its slot is free, so its credit starts back to the sender.
    T pop(Cycle now) {
        if (!can_pop(now)) throw std::logic_error("cycle::Link " + name() + " at cycle " + std::to_string(now) +
                                                  ": pop with nothing landed");
        const Cycle pushed = pipe_.front().pushed;
        T item = std::move(pipe_.front().item);
        pipe_.pop_front();
        return_credit(now);
        counted_pop();
        if constexpr (requires { static_cast<std::uint64_t>(item.id); })
            traced_pop(static_cast<std::uint64_t>(item.id), pushed, now);
        else traced_pop(0, pushed, now);
        return item;
    }

    // In the sender's cycles, as every LinkBase query.
    [[nodiscard]] std::size_t in_flight(Cycle now) const override {
        const Cycle h = horizon(now);
        std::size_t n = 0;
        for (auto it = pipe_.rbegin(); it != pipe_.rend() && it->arrival >= h; ++it) ++n;   // arrivals are in order
        return n;
    }
    [[nodiscard]] std::size_t buffered(Cycle now) const override { return pipe_.size() - in_flight(now); }
    [[nodiscard]] bool empty() const override { return pipe_.empty(); }
    void reset() override {
        LinkBase::reset();
        pipe_.clear();
    }

private:
    [[nodiscard]] std::size_t held() const override { return pipe_.size(); }
    struct Slot {
        Cycle arrival;   // the receiver's cycle it lands at
        Cycle pushed;    // the sender's cycle it was pushed at
        T item;
    };
    std::deque<Slot> pipe_;   // in push order
};

}  // namespace sw::kpu::cycle
