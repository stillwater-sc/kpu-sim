#pragma once
// ============================================================================
// include/sw/kpu/cycle/noc/noc_port.hpp
// A fold-end NoC port on the cycle core (#539; docs/plans/cycle-based-kpu-prototype.md §3.4). The
// port of timing/noc_port_process.hpp, re-expressed on links.
//
// The port sits on the fold link between hub_a and hub_b and owns both directed channels across
// it: hub_a's output on the a->b channel is a link into the port, and the port's link out is
// hub_b's ring input for that channel (and the same the other way). So the port is one more queue
// on each of the two rings that cross it. Three kinds of traffic go through:
//   ring-through  a flit continuing around its ring;
//   injection     a DMA engine's beat entering the NoC, bound for its L3 tile;
//   ejection      a flit leaving the ring for one of the port's DMA engines (a Store's beat).
//
// TWO BUSES, EACH BINARY (timing/noc_port_process.hpp §3.2): at most one injection and one
// ejection a cycle, over both channels.
//
// RING FIRST, AND GREEDY (§3.3). Each channel's head goes first: an ejection to its engine, when
// the engine's link has a credit and the ejection bus is free; ring-through to the next hub, with
// one credit (it continues on its ring). Then the oldest engine head (by the cycle it reached the
// head, then engine index) is injected, through the fold hub nearer its L3 (the topology's
// injection_hub), if no ring flit took that channel this cycle and the bubble rule allows: it
// enters the ring, so it needs two credits.
//
// COUNTERS. Each flit through the port is an arrival and a completion; the port is busy in a
// cycle it moves one, stalled (no credit) when a head waited, idle otherwise. noc() adds the
// flits of each kind, the cycles an injection waited, and of those the ones the bubble rule held.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/noc/noc_hub.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

struct NocPortCounts {
    std::uint64_t ring_through = 0, injected = 0, ejected = 0;
    std::uint64_t injection_waits = 0;   // cycles an engine's head waited to enter
    std::uint64_t bubble_waits = 0;      // of those, the ones with one credit, where the rule wanted two
};

class NocPort : public Component {
public:
    NocPort(std::string name, Domain& d, const NocTopology& topo, NocDim port, bool bubble = true)
        : Component(std::move(name), d), topo_(&topo), port_(port), bubble_(bubble) {
        const auto& p = topo.port(port);
        hub_a_ = p.hub_a;
        hub_b_ = p.hub_b;
        ch_ab_ = topo.fold_from(port, hub_a_);
        ch_ba_ = topo.fold_from(port, hub_b_);
    }

    // Wiring, by the fabric: the a->b channel in from hub_a and out to hub_b, and b->a.
    void connect_ring(Link<NocFlit>& from_a, Link<NocFlit>& to_b, Link<NocFlit>& from_b, Link<NocFlit>& to_a) {
        dir_[0] = {&from_a, &to_b, hub_a_};
        dir_[1] = {&from_b, &to_a, hub_b_};
    }
    // An engine: its beats in (bound for the L3 tile at `l3_hub`), and its ejected beats out.
    // Returns the engine's index on the port.
    NocDim connect_engine(Link<L3Beat>& from_engine, Link<L3Beat>& to_engine, NocDim l3_hub) {
        if (l3_hub >= topo_->hub_count())
            throw std::invalid_argument("cycle::NocPort " + name() + ": there is no hub " + std::to_string(l3_hub));
        engines_.push_back({&from_engine, &to_engine, l3_hub, std::nullopt});
        return static_cast<NocDim>(engines_.size() - 1);
    }

    void evaluate(Cycle now) override {
        if (!dir_[0].in || !dir_[1].in) throw std::logic_error("cycle::NocPort " + name() + ": not connected");
        bool moved = false, waited = false, ejected = false;
        bool used[2] = {false, false};
        // Ring first: each channel's head.
        for (int d = 0; d < 2; ++d) {
            Dir& x = dir_[d];
            if (!x.in->can_pop(now)) continue;
            const NocFlit& f = x.in->peek(now);
            if (f.fresh)
                throw std::logic_error("cycle::NocPort " + name() + " at cycle " + std::to_string(now) + ": flit " +
                                       std::to_string(f.id) + " reached it unrouted: a hub forwards every flit routed");
            if (f.dest == NocFlit::Dest::Engine && f.dst_port == port_ && f.dst_hub == x.from_hub) {
                if (f.dst_engine >= engines_.size())
                    throw std::logic_error("cycle::NocPort " + name() + " at cycle " + std::to_string(now) + ": flit " +
                                           std::to_string(f.id) + " for engine " + std::to_string(f.dst_engine) +
                                           ", which the port does not have");
                Link<L3Beat>& out = *engines_[f.dst_engine].out;
                if (!ejected && out.can_push(now)) {
                    out.push(now, x.in->pop(now).beat);
                    ejected = true;
                    ++noc_.ejected;
                    moved = true;
                    counters().accept();
                    counters().complete();
                } else {
                    waited = true;
                }
            } else if (x.out->can_push(now)) {
                x.out->push(now, x.in->pop(now));
                used[d] = true;
                ++noc_.ring_through;
                moved = true;
                counters().accept();
                counters().complete();
            } else {
                waited = true;
            }
        }
        // Then one injection: the oldest engine head.
        std::optional<std::size_t> best;
        for (std::size_t e = 0; e < engines_.size(); ++e) {
            Engine& g = engines_[e];
            if (!g.in->can_pop(now)) {
                g.head_since.reset();
                continue;
            }
            if (!g.head_since) g.head_since = now;
        }
        std::vector<std::size_t> order;
        bool bubble_flag = false;
        for (std::size_t e = 0; e < engines_.size(); ++e)
            if (engines_[e].head_since) order.push_back(e);
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return *engines_[a].head_since < *engines_[b].head_since; });
        for (const std::size_t e : order) {
            Engine& g = engines_[e];
            const NocDim to = topo_->injection_hub(port_, g.l3_hub);
            const int d = to == hub_b_ ? 0 : 1;
            if (used[d]) continue;
            const std::size_t have = dir_[d].out->credits(now);
            if (have < (bubble_ ? 2u : 1u)) {
                if (bubble_ && have == 1) bubble_flag = true;
                continue;
            }
            NocFlit f;
            f.id = ((++uid_count_) << 16) | 0x8000u | port_;
            f.born = now;
            f.dest = NocFlit::Dest::L3;
            f.dst_hub = g.l3_hub;
            f.fresh = true;
            f.beat = g.in->pop(now);
            g.head_since.reset();
            dir_[d].out->push(now, std::move(f));
            ++noc_.injected;
            moved = true;
            best = e;
            counters().accept();
            counters().complete();
            break;
        }
        if (!best && !order.empty()) {
            waited = true;
            ++noc_.injection_waits;
            if (bubble_flag) ++noc_.bubble_waits;
        }
        if (moved) counters().busy();
        else if (waited) counters().stall(Stall::NoCredit);
    }

    [[nodiscard]] bool idle() const override { return true; }   // its queues are links
    void reset() override {
        noc_ = {};
        uid_count_ = 0;
        for (Engine& g : engines_) g.head_since.reset();
    }
    [[nodiscard]] std::string describe() const override {
        return "port " + std::to_string(port_) + " (" + topo_->port(port_).label() + "): ring-through " +
               std::to_string(noc_.ring_through) + ", injected " + std::to_string(noc_.injected) + ", ejected " +
               std::to_string(noc_.ejected);
    }

    [[nodiscard]] NocDim port() const { return port_; }
    [[nodiscard]] const NocPortCounts& noc() const { return noc_; }

private:
    struct Dir {
        Link<NocFlit>* in = nullptr;
        Link<NocFlit>* out = nullptr;
        NocDim from_hub = 0;
    };
    struct Engine {
        Link<L3Beat>* in;
        Link<L3Beat>* out;
        NocDim l3_hub;
        std::optional<Cycle> head_since;   // when its current head reached the head of its link
    };

    const NocTopology* topo_;
    NocDim port_;
    bool bubble_;
    NocDim hub_a_ = 0, hub_b_ = 0, ch_ab_ = 0, ch_ba_ = 0;
    Dir dir_[2];
    std::vector<Engine> engines_;
    std::uint64_t uid_count_ = 0;
    NocPortCounts noc_;
};

}  // namespace sw::kpu::cycle
