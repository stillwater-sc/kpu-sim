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
// CONSERVATION, every cycle: credits held + items in flight + items buffered + credits in flight
// = capacity. check() raises if it does not hold.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/time.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

class Domain;

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
    LinkBase(std::string name, const Domain& domain, std::size_t capacity, Cycle latency, Cycle credit_latency)
        : name_(std::move(name)), domain_(&domain), capacity_(capacity), latency_(latency),
          credit_latency_(credit_latency), credits_(capacity) {
        if (capacity_ == 0) throw std::invalid_argument("cycle::Link " + name_ + ": capacity must be at least 1");
        if (latency_ == 0 || credit_latency_ == 0)
            throw std::invalid_argument("cycle::Link " + name_ + ": latencies must be at least 1 cycle");
    }
    virtual ~LinkBase() = default;
    LinkBase(const LinkBase&) = delete;
    LinkBase& operator=(const LinkBase&) = delete;

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] const Domain& domain() const { return *domain_; }
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
    }

    virtual void reset() {
        credits_ = capacity_;
        returning_.clear();
    }

protected:
    void spend_credit(Cycle now) {
        settle(now);
        if (credits_ == 0)
            throw std::logic_error("cycle::Link " + name_ + " at cycle " + std::to_string(now) +
                                   ": push without a credit");
        --credits_;
    }
    void return_credit(Cycle now) { returning_.push_back(now + credit_latency_); }
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
    std::size_t capacity_;
    Cycle latency_, credit_latency_;
    std::size_t credits_;
    std::deque<Cycle> returning_;       // arrival cycle of each credit on its way back
};

template <class T>
class Link : public LinkBase {
public:
    Link(std::string name, const Domain& domain, std::size_t capacity, Cycle latency = 1, Cycle credit_latency = 1)
        : LinkBase(std::move(name), domain, capacity, latency, credit_latency) {}

    // Sender side.
    [[nodiscard]] bool can_push(Cycle now) { return credits(now) > 0; }
    void push(Cycle now, T item) {
        spend_credit(now);
        pipe_.push_back({now + latency(), std::move(item)});
    }

    // Receiver side: the oldest landed item.
    [[nodiscard]] bool can_pop(Cycle now) const { return !pipe_.empty() && pipe_.front().first <= now; }
    [[nodiscard]] const T& peek(Cycle now) const {
        if (!can_pop(now)) throw std::logic_error("cycle::Link " + name() + ": peek with nothing landed");
        return pipe_.front().second;
    }
    // Pops the oldest landed item; its slot is free, so its credit starts back to the sender.
    T pop(Cycle now) {
        if (!can_pop(now)) throw std::logic_error("cycle::Link " + name() + " at cycle " + std::to_string(now) +
                                                  ": pop with nothing landed");
        T item = std::move(pipe_.front().second);
        pipe_.pop_front();
        return_credit(now);
        return item;
    }

    [[nodiscard]] std::size_t in_flight(Cycle now) const override {
        std::size_t n = 0;
        for (auto it = pipe_.rbegin(); it != pipe_.rend() && it->first > now; ++it) ++n;   // arrivals are in order
        return n;
    }
    [[nodiscard]] std::size_t buffered(Cycle now) const override { return pipe_.size() - in_flight(now); }
    [[nodiscard]] bool empty() const override { return pipe_.empty(); }
    void reset() override {
        LinkBase::reset();
        pipe_.clear();
    }

private:
    std::deque<std::pair<Cycle, T>> pipe_;   // (arrival cycle, item), in push order
};

}  // namespace sw::kpu::cycle
