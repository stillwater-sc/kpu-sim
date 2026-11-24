#pragma once
// ============================================================================
// tests/cycle/dram_pipeline.hpp
// A requester and a sink for the LPDDR5 channel's tests (#537): the requester pushes a list of
// DRAM requests, each no earlier than its cycle, one a cycle while it holds a credit; the sink
// pops responses, at most one every `interval` cycles, except in its stall windows, and logs them.
//
// COUNTERS (counters.hpp): the requester is busy when it pushes and stalled (no credit) when a
// request is due without a credit; the sink is busy when it pops and stalled (held) in a stall
// window with a response waiting.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle::test {

class Requester : public Component {
public:
    struct Item {
        DramRequest request;
        Cycle at = 0;               // no earlier than this cycle
    };
    Requester(std::string name, Domain& d, std::vector<Item> items)
        : Component(std::move(name), d), items_(std::move(items)) {}
    void connect(Link<DramRequest>& out) { out_ = &out; }

    void evaluate(Cycle now) override {
        if (next_ == items_.size() || now < items_[next_].at) return;
        if (!out_->can_push(now)) {
            counters().stall(Stall::NoCredit);
            return;
        }
        out_->push(now, items_[next_].request);
        pushed_at.push_back(now);
        ++next_;
        counters().busy();
        counters().produce();
    }
    [[nodiscard]] bool idle() const override { return next_ == items_.size(); }
    [[nodiscard]] std::string describe() const override {
        return "pushed " + std::to_string(next_) + " of " + std::to_string(items_.size());
    }
    void reset() override {
        next_ = 0;
        pushed_at.clear();
    }

    std::vector<Cycle> pushed_at;

private:
    Link<DramRequest>* out_ = nullptr;
    std::vector<Item> items_;
    std::size_t next_ = 0;
};

class Sink : public Component {
public:
    Sink(std::string name, Domain& d, Cycle interval = 1) : Component(std::move(name), d), interval_(interval) {}
    void connect(Link<DramResponse>& in) { in_ = &in; }
    void stall(Cycle from, Cycle to) { stalls_.emplace_back(from, to); }

    void evaluate(Cycle now) override {
        for (const auto& [a, b] : stalls_)
            if (now >= a && now < b) {
                if (in_->can_pop(now)) counters().stall(Stall::External);
                return;
            }
        if (now < next_at_ || !in_->can_pop(now)) return;
        received.push_back(in_->pop(now));
        popped_at.push_back(now);
        next_at_ = now + interval_;
        counters().busy();
        counters().accept();
        counters().complete();
    }
    [[nodiscard]] bool idle() const override { return true; }
    void reset() override {
        received.clear();
        popped_at.clear();
        next_at_ = 0;
    }

    std::vector<DramResponse> received;
    std::vector<Cycle> popped_at;

private:
    Link<DramResponse>* in_ = nullptr;
    Cycle interval_;
    std::vector<std::pair<Cycle, Cycle>> stalls_;
    Cycle next_at_ = 0;
};

}  // namespace sw::kpu::cycle::test
