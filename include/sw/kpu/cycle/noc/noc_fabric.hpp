#pragma once
// ============================================================================
// include/sw/kpu/cycle/noc/noc_fabric.hpp
// The NoC on the cycle core, built from a deployment's layout (#539;
// docs/plans/cycle-based-kpu-prototype.md §3.4, §4.1, §4.3 rung 3).
//
// One NocHub per L3 tile and one NocPort per fold-end port of the folded torus (ArrayLayout ->
// timing/noc_topology.hpp), all in one clock domain, and a link for every directed channel: hub to
// hub, or, on a fold link, hub to port and port to hub. Every ring link holds `ring_queue` flits:
// the receiver's input queue. The fabric is not a component; it adds the components and links to
// the simulator, and wires what is attached to it:
//   attach_l3      an L3 tile at its hub: the Loads' beats in, its Stores' beats out, the latter
//                  bound for one DMA engine on one port;
//   attach_engine  a DMA engine on a port: its beats in, bound for one L3 tile, and its Stores'
//                  beats out. The engine's slot credits come back on a link of their own, from
//                  the L3 tile, not over the NoC.
//
// THE BUBBLE RULE, CHECKED (noc_hub.hpp): every unidirectional ring -- the links of one loop in one
// direction, through its hubs and ports -- holds fewer flits than its slots, every cycle, and
// every ring link takes at most one flit a cycle. A violation stops the run naming the ring or link. With the rule off (Config::bubble = false, only to show
// what it prevents), the check is what fails, or the deadlock detector.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/noc/noc_hub.hpp>
#include <sw/kpu/cycle/noc/noc_port.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/timing/noc_topology.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

class NocFabric {
public:
    struct Config {
        std::size_t ring_queue = 2;     // flits per ring input; the bubble rule needs two
        std::size_t l3_link = 2;        // the hub <-> L3 tile links
        std::size_t engine_link = 2;    // the port <-> DMA engine links
        bool bubble = true;             // off only to show what it prevents

        // From a device's declared noc: its hub buffers split over the four ring inputs, and its
        // port input queue.
        static Config from(const sw::kpu::program::platform::DeviceSpecification& d) {
            if (!d.noc) throw std::invalid_argument("cycle::NocFabric: the device declares no noc");
            Config c;
            c.ring_queue = d.noc->hub_buffer_blocks / sw::kpu::program::platform::kNocHubInputs;
            c.engine_link = d.noc->port.input_queue_blocks;
            return c;
        }
    };
    struct L3Links {
        Link<L3Beat>& to_l3;
        Link<L3Beat>& from_l3;
    };
    struct EngineLinks {
        Link<L3Beat>& to_noc;
        Link<L3Beat>& from_noc;
        NocDim engine;
    };

    NocFabric(Simulator& sim, Domain& d, const sw::kpu::program::platform::ArrayLayout& layout, Config c)
        : sim_(&sim), domain_(&d), config_(c), topo_(std::make_unique<NocTopology>(layout)) {
        if (c.ring_queue < 2 && c.bubble)
            throw std::invalid_argument("cycle::NocFabric: a ring queue of " + std::to_string(c.ring_queue) +
                                        " flit cannot keep a bubble; it needs two");
        if (c.ring_queue == 0 || c.l3_link == 0 || c.engine_link == 0)
            throw std::invalid_argument("cycle::NocFabric: every link needs a credit");
        const NocTopology& t = *topo_;
        for (NocDim h = 0; h < t.hub_count(); ++h)
            hubs_.push_back(&sim.add<NocHub>("noc hub " + std::to_string(h), d, t, h, c.bubble));
        for (NocDim k = 0; k < t.port_count(); ++k)
            ports_.push_back(&sim.add<NocPort>("noc port " + t.port(k).label(), d, t, k, c.bubble));
        std::vector<std::map<NocDim, std::pair<Link<NocFlit>*, Link<NocFlit>*>>> fold(t.port_count());
        for (const auto& ch : t.channels()) {
            const std::string name = "ch" + std::to_string(ch.id) + " " + std::to_string(ch.from_hub) + "->" +
                                     std::to_string(ch.to_hub);
            Ring& ring = rings_[{ch.loop, ch.forward}];
            ring.label = (ch.column ? "column loop " : "row loop ") + std::to_string(ch.loop) +
                         (ch.forward ? ", forward" : ", backward");
            if (ch.port) {
                NocPort& p = *ports_[*ch.port];
                auto& in = sim.link<NocFlit>(name + " into " + p.name(), *hubs_[ch.from_hub], p, c.ring_queue);
                auto& out = sim.link<NocFlit>(name + " out of " + p.name(), p, *hubs_[ch.to_hub], c.ring_queue);
                hubs_[ch.from_hub]->connect_out(ch.id, in);
                hubs_[ch.to_hub]->connect_in(ch.id, out);
                fold[*ch.port][ch.from_hub] = {&in, &out};
                ring.links.push_back(&in);
                ring.links.push_back(&out);
            } else {
                auto& l = sim.link<NocFlit>(name, *hubs_[ch.from_hub], *hubs_[ch.to_hub], c.ring_queue);
                hubs_[ch.from_hub]->connect_out(ch.id, l);
                hubs_[ch.to_hub]->connect_in(ch.id, l);
                ring.links.push_back(&l);
            }
        }
        for (NocDim k = 0; k < t.port_count(); ++k) {
            const auto& p = t.port(k);
            auto& a = fold[k].at(p.hub_a);   // a -> b: in from a, out to b
            auto& b = fold[k].at(p.hub_b);
            ports_[k]->connect_ring(*a.first, *a.second, *b.first, *b.second);
        }
        for (auto& [key, r] : rings_) r.arrivals.assign(r.links.size(), 0);
        sim.add_invariant("noc bubble: every ring keeps a free slot", [this] { check_rings(); });
    }
    NocFabric(const NocFabric&) = delete;
    NocFabric& operator=(const NocFabric&) = delete;

    // An L3 tile at `hub`; its Stores' beats go to `engine` on `port`. The caller connects the
    // tile: l3.connect(links.to_l3, links.from_l3, its credit link).
    L3Links attach_l3(NocDim hub, const Component& l3, NocDim store_port, NocDim store_engine) {
        NocHub& h = *hubs_.at(hub);
        auto& to = sim_->link<L3Beat>(h.name() + " -> " + l3.name(), h, l3, config_.l3_link);
        auto& from = sim_->link<L3Beat>(l3.name() + " -> " + h.name(), l3, h, config_.l3_link);
        h.connect_l3(to, from, store_port, store_engine);
        return {to, from};
    }

    // A DMA engine on `port`, its Loads bound for the L3 tile at `l3_hub`. The caller connects the
    // engine: dma.connect(to DRAM, from DRAM, links.to_noc, links.from_noc, its credit link).
    EngineLinks attach_engine(NocDim port, const Component& dma, NocDim l3_hub) {
        NocPort& p = *ports_.at(port);
        auto& to = sim_->link<L3Beat>(dma.name() + " -> " + p.name(), dma, p, config_.engine_link);
        auto& from = sim_->link<L3Beat>(p.name() + " -> " + dma.name(), p, dma, config_.engine_link);
        const NocDim e = p.connect_engine(to, from, l3_hub);
        return {to, from, e};
    }

    [[nodiscard]] const NocTopology& topology() const { return *topo_; }
    [[nodiscard]] NocHub& hub(NocDim h) { return *hubs_.at(h); }
    [[nodiscard]] NocPort& port(NocDim k) { return *ports_.at(k); }
    [[nodiscard]] const Config& config() const { return config_; }

    // The most flits each ring has held at once, over the run so far, of its slots.
    struct RingPeak {
        std::string label;
        std::size_t peak = 0, slots = 0;
    };
    [[nodiscard]] std::vector<RingPeak> ring_peaks() const {
        std::vector<RingPeak> v;
        for (const auto& [key, r] : rings_) v.push_back({r.label, r.peak, r.links.size() * config_.ring_queue});
        return v;
    }

private:
    struct Ring {
        std::string label;
        std::vector<Link<NocFlit>*> links;
        std::vector<std::uint64_t> arrivals;   // each link's, at the last check
        std::size_t peak = 0;
    };

    void check_rings() {
        if (domain_->cycle() == 0) return;          // the NoC's domain has not ticked yet
        const Cycle now = domain_->cycle() - 1;   // its cycle just closed
        if (now == checked_) return;                // a step of another domain only
        checked_ = now;
        for (auto& [key, r] : rings_) {
            // One flit a cycle on each link: a channel carries one, whoever drives it.
            for (std::size_t i = 0; i < r.links.size(); ++i) {
                const std::uint64_t a = r.links[i]->counts().arrivals;
                if (a > r.arrivals[i] + 1)
                    throw std::logic_error("link " + r.links[i]->name() + " took " + std::to_string(a - r.arrivals[i]) +
                                           " flits in cycle " + std::to_string(now));
                r.arrivals[i] = a;
            }
            std::size_t held = 0;
            for (Link<NocFlit>* l : r.links) held += l->in_flight(now) + l->buffered(now);
            r.peak = std::max(r.peak, held);
            const std::size_t slots = r.links.size() * config_.ring_queue;
            if (held >= slots)
                throw std::logic_error("the " + r.label + " ring holds " + std::to_string(held) + " flits in its " +
                                       std::to_string(slots) + " slots: no bubble left");
        }
    }

    Simulator* sim_;
    Domain* domain_;
    Config config_;
    std::unique_ptr<NocTopology> topo_;
    std::vector<NocHub*> hubs_;
    std::vector<NocPort*> ports_;
    std::map<std::pair<NocDim, bool>, Ring> rings_;
    Cycle checked_ = ~Cycle{0};
};

}  // namespace sw::kpu::cycle
