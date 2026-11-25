#pragma once
// ============================================================================
// include/sw/kpu/cycle/noc/noc_hub.hpp
// A NoC hub on the cycle core (#539; docs/plans/cycle-based-kpu-prototype.md §3.4, §4.1). The
// hub of timing/noc_hub_process.hpp, re-expressed on links.
//
// THE HUB lives in its L3 tile, on the folded torus (timing/noc_topology.hpp, reused as it is):
// four ring inputs and four ring outputs, two per torus dimension, plus its L3's side, a link to
// the L3 tile (the Loads' beats it delivers) and one from it (the Stores' beats it ejects).
//
// QUEUES ARE PER INPUT, and they are the links. Each ring input is a link whose capacity is that
// input's queue (`ring_queue` beats); the sender holds its credits. A flit is one 64-byte beat
// (l3_program.hpp), so a packet is one flit and nothing is wormholed.
//
// ROUTING is the topology's: dimension-ordered, the row loop to the turn and the column loop to
// the destination, shorter way round. A flit bound for an L3 tile leaves at its hub to the L3; a
// flit bound for a DMA engine leaves at its exit hub onto the fold channel of the engine's port.
//
// THE BUBBLE RULE (timing/noc_hub_process.hpp, §3.5). A flit that ENTERS a ring -- from the L3, from
// a port's injection, or turning from its row loop onto its column loop -- may take a slot in the
// next queue only if a second slot stays free there: it needs two credits. A flit continuing along
// the ring it is on needs one. So every unidirectional ring always keeps a free slot, a flit on it
// can always move, and with dimension-ordered routing and sinks that drain (the L3 tiles, and DMA
// engines whose store path never waits on reads, dma_engine.hpp), no cycle of waits can form.
// noc_fabric.hpp checks the free slot on every ring every cycle.
//
// ARBITRATION is greedy and stateless: each output, in channel order and then the L3, takes the
// oldest flit (born, then id) at the head of an input that is routed to it and that its credits
// allow. Each input gives at most one flit a cycle, and each output takes at most one.
//
// COUNTERS. Every flit forwarded is an arrival and a completion (the hub holds nothing: its
// queues are its input links, counted there). The hub is busy in a cycle it forwards, stalled (no
// credit) when a flit at an input head waited for its output's credits, idle otherwise. hub()
// (noc()) adds: flits forwarded on the rings, delivered to the L3, taken from the L3, and the cycles a
// flit waited because the bubble rule wanted a second credit.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/timing/noc_topology.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

using NocDim = sw::kpu::timing::NocDim;
using NocTopology = sw::kpu::timing::NocTopology;

// One flit: one beat, and where it goes.
struct NocFlit {
    std::uint64_t id = 0;            // unique over the fabric: (count << 16) | the source's uid
    Cycle born = 0;                  // when it entered the NoC: its age
    enum class Dest : std::uint8_t { L3, Engine } dest = Dest::L3;
    NocDim dst_hub = 0;              // L3: the hub whose L3 receives it; Engine: its exit hub
    NocDim dst_port = 0, dst_engine = 0;   // Engine only
    bool fresh = true;               // not yet on its route's ring: it entered at a port, so its
                                     // first hub routes it from the start, as entering
    L3Beat beat;
};

// Age order: born first, then id. Nothing but the flits decides it.
inline bool older(const NocFlit& a, const NocFlit& b) { return a.born != b.born ? a.born < b.born : a.id < b.id; }

struct NocHubCounts {
    std::uint64_t forwarded = 0;      // onto a ring channel
    std::uint64_t delivered = 0;      // to the L3
    std::uint64_t from_l3 = 0;        // taken from the L3, bound for an engine
    std::uint64_t bubble_waits = 0;   // cycles a flit had one credit but needed two
};

class NocHub : public Component {
public:
    static constexpr std::size_t kL3 = static_cast<std::size_t>(-1);   // the L3 side, as an input or output

    NocHub(std::string name, Domain& d, const NocTopology& topo, NocDim hub, bool bubble = true)
        : Component(std::move(name), d), topo_(&topo), hub_(hub), bubble_(bubble) {
        for (const NocDim c : topo.in_channels(hub)) in_.push_back({c, nullptr});
        for (const NocDim c : topo.out_channels(hub)) out_.push_back({c, nullptr});
    }

    // Wiring, by the fabric.
    void connect_in(NocDim channel, Link<NocFlit>& l) { slot(in_, channel).link = &l; }
    void connect_out(NocDim channel, Link<NocFlit>& l) { slot(out_, channel).link = &l; }
    // The L3 tile: `to_l3` its Loads' beats, `from_l3` its Stores' beats, which go to `engine` of
    // `port`.
    void connect_l3(Link<L3Beat>& to_l3, Link<L3Beat>& from_l3, NocDim port, NocDim engine) {
        if (port >= topo_->port_count())
            throw std::invalid_argument("cycle::NocHub " + name() + ": there is no port " + std::to_string(port));
        to_l3_ = &to_l3;
        from_l3_ = &from_l3;
        store_port_ = port;
        store_engine_ = engine;
    }

    void evaluate(Cycle now) override {
        for (const auto& e : in_)
            if (!e.link) throw std::logic_error("cycle::NocHub " + name() + ": ring input " + std::to_string(e.channel) + " not connected");
        for (const auto& e : out_)
            if (!e.link) throw std::logic_error("cycle::NocHub " + name() + ": ring output " + std::to_string(e.channel) + " not connected");
        // The heads: each input's oldest flit, where it goes, and whether it enters a ring there.
        std::vector<Head> heads;
        for (std::size_t i = 0; i < in_.size(); ++i)
            if (in_[i].link->can_pop(now)) heads.push_back(head(i, in_[i].link->peek(now)));
        std::optional<NocFlit> local;
        if (from_l3_ && from_l3_->can_pop(now)) {
            local = flit_from_l3(now, from_l3_->peek(now));
            heads.push_back(head(kL3, *local));
        }
        std::sort(heads.begin(), heads.end(), [](const Head& a, const Head& b) { return older(*a.flit, *b.flit); });

        bool moved = false, waited = false;
        std::vector<bool> taken(heads.size(), false);
        auto serve = [&](std::size_t out) {   // index into out_, or kL3
            for (std::size_t h = 0; h < heads.size(); ++h) {
                if (taken[h] || heads[h].out != out) continue;
                const std::size_t need = out == kL3 ? 1 : (bubble_ && enters(heads[h], out_[out].channel) ? 2 : 1);
                const std::size_t have = out == kL3 ? (to_l3_ ? to_l3_->credits(now) : 0) : out_[out].link->credits(now);
                if (have < need) {
                    waited = true;
                    if (have + 1 == need) ++noc_.bubble_waits;
                    continue;
                }
                NocFlit f = pop(now, heads[h].in, local);
                f.fresh = false;
                if (out == kL3) {
                    to_l3_->push(now, std::move(f.beat));
                    ++noc_.delivered;
                } else {
                    out_[out].link->push(now, std::move(f));
                    ++noc_.forwarded;
                }
                taken[h] = true;
                counters().accept();
                counters().complete();
                moved = true;
                return;
            }
        };
        for (std::size_t o = 0; o < out_.size(); ++o) serve(o);
        serve(kL3);
        if (moved) counters().busy();
        else if (waited) counters().stall(Stall::NoCredit);
    }

    [[nodiscard]] bool idle() const override { return true; }   // it holds nothing: its queues are links

    [[nodiscard]] std::string describe() const override {
        return "hub " + std::to_string(hub_) + ": forwarded " + std::to_string(noc_.forwarded) + ", delivered " +
               std::to_string(noc_.delivered) + ", from L3 " + std::to_string(noc_.from_l3);
    }
    void reset() override {
        hub_counts_reset();
        uid_count_ = 0;
    }

    [[nodiscard]] NocDim hub() const { return hub_; }
    [[nodiscard]] const NocHubCounts& noc() const { return noc_; }

    // Where a flit goes from hub `at`, having arrived on `in_channel` (nullopt: it entered here).
    static std::size_t route_channel(const NocTopology& t, NocDim at, const NocFlit& f, std::optional<NocDim> in_channel) {
        if (f.dst_hub == at) {
            if (f.dest == NocFlit::Dest::L3) return kL3;
            return t.fold_from(f.dst_port, at);
        }
        const bool on_column = !f.fresh && in_channel && t.channel(*in_channel).column;
        return t.next_hop(at, f.dst_hub, on_column);
    }

private:
    struct End {
        NocDim channel;
        Link<NocFlit>* link;
    };
    struct Head {
        std::size_t in;              // index into in_, or kL3
        const NocFlit* flit;
        std::size_t out;             // index into out_, or kL3
    };

    End& slot(std::vector<End>& v, NocDim channel) {
        for (End& e : v)
            if (e.channel == channel) return e;
        throw std::invalid_argument("cycle::NocHub " + name() + ": channel " + std::to_string(channel) +
                                    " does not meet hub " + std::to_string(hub_));
    }

    Head head(std::size_t in, const NocFlit& f) const {
        const std::optional<NocDim> ch = in == kL3 ? std::nullopt : std::optional<NocDim>(in_[in].channel);
        const std::size_t c = route_channel(*topo_, hub_, f, ch);
        if (c == kL3) return {in, &f, kL3};
        for (std::size_t o = 0; o < out_.size(); ++o)
            if (out_[o].channel == c) return {in, &f, o};
        throw std::logic_error("cycle::NocHub " + name() + ": routed flit " + std::to_string(f.id) + " to channel " +
                               std::to_string(c) + ", which does not leave hub " + std::to_string(hub_));
    }

    // Entering the ring of `out`: from the L3, from a port's injection, or from another ring.
    [[nodiscard]] bool enters(const Head& h, NocDim out) const {
        if (h.in == kL3 || h.flit->fresh) return true;
        return !topo_->channel(in_[h.in].channel).same_ring(topo_->channel(out));
    }

    NocFlit flit_from_l3(Cycle now, const L3Beat& b) const {
        NocFlit f;
        f.id = ((uid_count_ + 1) << 16) | hub_;
        f.born = now;
        f.dest = NocFlit::Dest::Engine;
        f.dst_port = store_port_;
        f.dst_engine = store_engine_;
        f.dst_hub = topo_->exit_hub(store_port_, hub_);
        f.fresh = true;
        f.beat = b;
        return f;
    }

    NocFlit pop(Cycle now, std::size_t in, std::optional<NocFlit>& local) {
        if (in == kL3) {
            from_l3_->pop(now);
            ++uid_count_;
            ++noc_.from_l3;
            return std::move(*local);
        }
        return in_[in].link->pop(now);
    }

    void hub_counts_reset() { noc_ = NocHubCounts{}; }

    const NocTopology* topo_;
    NocDim hub_;
    bool bubble_;
    std::vector<End> in_, out_;
    Link<L3Beat>* to_l3_ = nullptr;
    Link<L3Beat>* from_l3_ = nullptr;
    NocDim store_port_ = 0, store_engine_ = 0;
    std::uint64_t uid_count_ = 0;
    NocHubCounts noc_;
};

}  // namespace sw::kpu::cycle
