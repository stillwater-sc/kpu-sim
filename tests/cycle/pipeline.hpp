#pragma once
// ============================================================================
// tests/cycle/pipeline.hpp
// The cycle core's test structure (docs/plans/cycle-based-kpu-prototype.md §3.5): a producer, N
// stages and a consumer, each with its own service rate, latency and buffering.
//
//   Producer   pushes beats 0..count-1, at most one every `interval` cycles, when it holds a credit
//   Stage      accepts a landed beat at most every `interval` cycles while it holds fewer than
//              `depth`, and offers it downstream `latency` cycles after accepting it. In a cycle it
//              offers first, then accepts, as a registered pipeline does.
//   Consumer   pops a landed beat at most every `interval` cycles, except in its stall windows,
//              and logs (beat id, cycle). check() fails on a beat out of order, lost or duplicated;
//              finish(n) fails if fewer than n arrived (a lost last beat).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/simulator.hpp>

#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle::test {

class Producer : public Component {
public:
    Producer(std::string name, Domain& d, std::uint64_t count, Cycle interval = 1, std::size_t bytes = 4)
        : Component(std::move(name), d), count_(count), interval_(interval), bytes_(bytes) {}
    void connect(Link<Beat>& out) { out_ = &out; }

    void evaluate(Cycle now) override {
        if (next_id_ == count_ || now < next_at_ || !out_->can_push(now)) return;
        Beat b;
        b.id = next_id_;
        b.address = next_id_ * bytes_;
        b.bytes.assign(bytes_, static_cast<std::uint8_t>(next_id_ & 0xFF));
        b.first = next_id_ == 0;
        b.last = next_id_ + 1 == count_;
        out_->push(now, std::move(b));
        pushed_at.push_back(now);
        ++next_id_;
        next_at_ = now + interval_;
    }
    [[nodiscard]] bool idle() const override { return next_id_ == count_; }
    void reset() override {
        next_id_ = 0;
        next_at_ = 0;
        pushed_at.clear();
    }

    std::vector<Cycle> pushed_at;       // the cycle each beat was pushed

private:
    Link<Beat>* out_ = nullptr;
    std::uint64_t count_;
    Cycle interval_;
    std::size_t bytes_;
    std::uint64_t next_id_ = 0;
    Cycle next_at_ = 0;
};

class Stage : public Component {
public:
    Stage(std::string name, Domain& d, Cycle latency, Cycle interval = 1, std::size_t depth = 1)
        : Component(std::move(name), d), latency_(latency), interval_(interval), depth_(depth) {}
    void connect(Link<Beat>& in, Link<Beat>& out) {
        in_ = &in;
        out_ = &out;
    }

    void evaluate(Cycle now) override {
        if (!held_.empty() && held_.front().first <= now && out_->can_push(now)) {
            out_->push(now, std::move(held_.front().second));
            held_.pop_front();
        }
        if (held_.size() < depth_ && now >= next_at_ && in_->can_pop(now)) {
            held_.push_back({now + latency_, in_->pop(now)});
            next_at_ = now + interval_;
        }
    }
    [[nodiscard]] bool idle() const override { return held_.empty(); }
    void check(Cycle now) const override {
        if (held_.size() > depth_)
            throw std::logic_error(name() + " at cycle " + std::to_string(now) + ": holds more than its depth");
    }
    void reset() override {
        held_.clear();
        next_at_ = 0;
    }

private:
    Link<Beat>* in_ = nullptr;
    Link<Beat>* out_ = nullptr;
    Cycle latency_, interval_;
    std::size_t depth_;
    std::deque<std::pair<Cycle, Beat>> held_;   // (ready cycle, beat)
    Cycle next_at_ = 0;
};

class Consumer : public Component {
public:
    Consumer(std::string name, Domain& d, Cycle interval = 1) : Component(std::move(name), d), interval_(interval) {}
    void connect(Link<Beat>& in) { in_ = &in; }
    // No pops in [from, to).
    void stall(Cycle from, Cycle to) { stalls_.emplace_back(from, to); }

    void evaluate(Cycle now) override {
        for (const auto& [a, b] : stalls_) if (now >= a && now < b) return;
        if (now < next_at_ || !in_->can_pop(now)) return;
        const Beat beat = in_->pop(now);
        if (beat.bytes.empty() || beat.bytes.front() != static_cast<std::uint8_t>(beat.id & 0xFF))
            bad_payload_ = true;
        received.emplace_back(beat.id, now);
        next_at_ = now + interval_;
    }
    [[nodiscard]] bool idle() const override { return true; }
    void check(Cycle now) const override {
        if (bad_payload_) throw std::logic_error(name() + " at cycle " + std::to_string(now) + ": a beat's bytes are not its own");
        if (!received.empty() && received.back().first + 1 != received.size())
            throw std::logic_error(name() + " at cycle " + std::to_string(now) + ": beat " +
                                   std::to_string(received.back().first) + " arrived as number " +
                                   std::to_string(received.size() - 1) + " (lost, duplicated or out of order)");
    }
    void reset() override {
        received.clear();
        next_at_ = 0;
        bad_payload_ = false;
    }
    // At quiescence: every one of `expected` beats arrived. check() sees a gap only when a later
    // beat arrives, so a lost last beat (or the only one) is caught here.
    void finish(std::uint64_t expected) const {
        if (received.size() != expected)
            throw std::logic_error(name() + ": " + std::to_string(received.size()) + " of " +
                                   std::to_string(expected) + " beats arrived");
    }

    std::vector<std::pair<std::uint64_t, Cycle>> received;   // (beat id, cycle popped)

private:
    Link<Beat>* in_ = nullptr;
    Cycle interval_;
    std::vector<std::pair<Cycle, Cycle>> stalls_;
    Cycle next_at_ = 0;
    bool bad_payload_ = false;
};

}  // namespace sw::kpu::cycle::test
