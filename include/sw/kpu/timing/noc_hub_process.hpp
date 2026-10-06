// ============================================================================
// include/sw/kpu/timing/noc_hub_process.hpp
// A store-and-forward NoC hub (docs/plans/noc-port-arbitration.md §1.2, §3.5, step 4).
//
// The hub lives in its L3 tile. It has four ring inputs and four ring outputs (two per torus
// dimension; noc_topology.hpp).
//
// BUFFERS ARE PER INPUT. Each ring input has its own queue of `ring_queue_blocks` blocks: a
// block that arrived over a channel waits in that channel's queue. Blocks that enter the NoC
// at this hub -- pushed out of the L3 by a BlockMover, or injected by a port whose fold link
// ends here -- wait in an ENTRY queue, one per source, outside every ring.
//
// THE BUBBLE RULE (§3.5) is what keeps the fabric live. A block that ENTERS a ring -- from an
// entry queue, or turning from its row loop onto a column loop -- may take a slot in the next
// hub's queue only if a second slot stays free there. A block continuing along the ring it is
// on needs one slot. So every unidirectional ring always keeps a free slot somewhere, and a
// block on it can always move. With dimension-ordered routing (a column ring never waits on a
// row ring) and sinks that always drain (the L3, and DMA engines that write their output
// queues to DRAM), no cycle of waiting blocks can form. That is why a ring queue holds at least
// two blocks: one for the block, one for the bubble.
//
// CREDIT, NOT REQUEST. A sender reserves a slot here before it starts a transfer and the block
// lands at the end of it; the slot is released when the block has finished leaving. So a block
// holds two slots while it crosses a link, which is what store-and-forward costs.
//
// The hub drives its own ring outputs, except the ones that are fold links: those belong to the
// port on the link (NocPortProcess), which takes its blocks out of this hub's queues.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/timing/credit_pool.hpp>
#include <sw/kpu/timing/noc_topology.hpp>
#include <sw/kpu/timing/process_interface.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing {

// One block on the NoC. It carries the caller's tag, never a value: the NoC changes when a
// block arrives, not what it holds (ADR 0002).
struct NocBlock {
    std::uint64_t id = 0;
    std::uint64_t tag = 0;              // the caller's handle for the block
    Cycle born = 0;                     // when it entered the NoC: its age (§3.3)
    enum class Dest : std::uint8_t { L3, Engine } dest = Dest::L3;
    NocDim dst_hub = 0;                 // L3: the hub whose L3 receives it; Engine: its exit hub
    NocDim dst_port = 0, dst_engine = 0;  // Engine only: the DMA engine buffer it is pushed to
};

// One block crossing a link, a bus, or a hub's local side.
struct NocTransfer {
    NocBlock block;
    Cycle start = 0, end = 0;
    std::size_t src_queue = 0;          // the sender's queue slot, released at `end`
};

class NocHubProcess : public IProcess {
public:
    struct Config {
        NocDim hub = 0;                     // l3 index
        std::size_t ring_queue_blocks = 2;  // per ring input
        std::size_t entry_queue_blocks = 2; // per entry source: one landing, one leaving
        Cycle block_cycles = 1;             // one block across one link
        Cycle watchdog_cycles = 0;          // TF-HUB-2 bound; 0 = no watchdog
        bool bubble = true;                 // §3.5; off only to show what it prevents
    };

    // A block in one of this hub's queues, waiting for its next hop.
    struct Resident {
        NocBlock block;
        Cycle arrived = 0;
        std::size_t queue = 0;
        std::optional<NocDim> in_channel;   // the channel it arrived on; nullopt = it entered here
        std::optional<NocDim> next;         // the channel it leaves on; nullopt = to the L3
    };

    struct Stats {
        std::uint64_t forwarded = 0;            // ring hops started from this hub
        std::uint64_t delivered = 0;            // blocks handed to this hub's L3
        std::uint64_t accepted_from_l3 = 0;     // BlockMover pushes into the entry queue
        std::size_t peak_ring_queue = 0;        // TF-HUB-1: never above ring_queue_blocks
        std::size_t peak_ring_occupancy = 0;    // over all four ring queues at once
        Cycle max_residence = 0;                // longest wait in a RING queue for a next hop
        Cycle max_entry_wait = 0;               // TF-HUB-3: longest wait to enter a ring
        bool watchdog_fired = false;            // TF-HUB-2: a ring queue stopped moving
        Cycle watchdog_cycle = 0;
        std::uint64_t watchdog_block = 0;
    };

    NocHubProcess(const Config& config, const NocTopology& topo,
                  std::vector<NocHubProcess>& hubs)
        : config_(config), topo_(&topo), hubs_(&hubs) {
        if (config.ring_queue_blocks == 0 || config.entry_queue_blocks == 0)
            throw std::invalid_argument(name() + ": queues must hold at least one block");
        for (NocDim c : topo.out_channels(config.hub))
            if (!topo.channel(c).port) ring_out_.push_back(c);
        out_flight_.assign(ring_out_.size(), std::nullopt);
        in_channels_ = topo.in_channels(config.hub);
        for (NocDim k = 0; k < topo.port_count(); ++k)
            if (topo.port(k).hub_a == config.hub || topo.port(k).hub_b == config.hub)
                entry_ports_.push_back(k);
        for (std::size_t i = 0; i < in_channels_.size(); ++i)
            queues_.emplace_back(config.ring_queue_blocks);
        for (std::size_t i = 0; i < 1 + entry_ports_.size(); ++i)
            queues_.emplace_back(config.entry_queue_blocks);
    }

    // ---- queues ----
    std::size_t ring_queue(NocDim in_channel) const {
        for (std::size_t i = 0; i < in_channels_.size(); ++i)
            if (in_channels_[i] == in_channel) return i;
        throw std::invalid_argument(name() + ": channel " + std::to_string(in_channel) +
                                    " does not arrive here");
    }
    std::size_t local_queue() const { return in_channels_.size(); }
    std::size_t port_queue(NocDim port) const {
        for (std::size_t i = 0; i < entry_ports_.size(); ++i)
            if (entry_ports_[i] == port) return in_channels_.size() + 1 + i;
        throw std::invalid_argument(name() + ": port " + std::to_string(port) +
                                    " does not end here");
    }
    bool is_ring_queue(std::size_t q) const { return q < in_channels_.size(); }

    // `keep_free` slots must stay free after this one is taken (the bubble).
    bool reserve(std::size_t q, std::size_t keep_free = 0) {
        CreditPool& c = queues_.at(q);
        if (c.available() <= keep_free || !c.acquire()) return false;
        if (is_ring_queue(q)) {
            stats_.peak_ring_queue = std::max(stats_.peak_ring_queue, c.outstanding());
            std::size_t ring = 0;
            for (std::size_t i = 0; i < in_channels_.size(); ++i) ring += queues_[i].outstanding();
            stats_.peak_ring_occupancy = std::max(stats_.peak_ring_occupancy, ring);
        }
        return true;
    }
    void release(std::size_t q) { queues_.at(q).release(); }
    std::size_t held(std::size_t q) const { return queues_.at(q).outstanding(); }

    // A reserved block has landed in queue `q`, over `in_channel` or entering here (nullopt).
    void arrive(const NocBlock& b, Cycle now, std::size_t q, std::optional<NocDim> in_channel) {
        Resident r{b, now, q, in_channel, std::nullopt};
        const bool on_column = in_channel && topo_->channel(*in_channel).column;
        if (b.dest == NocBlock::Dest::Engine && b.dst_hub == config_.hub)
            r.next = topo_->fold_from(b.dst_port, config_.hub);
        else if (b.dst_hub != config_.hub)
            r.next = topo_->next_hop(config_.hub, b.dst_hub, on_column);
        waiting_.push_back(r);
    }

    // The slots a block must leave free in the next queue when it leaves on `channel`: one
    // when it enters that ring (the bubble), none when it continues along it.
    std::size_t keep_free(const Resident& r, NocDim channel) const {
        if (!config_.bubble) return 0;
        const bool continues =
            r.in_channel && topo_->channel(*r.in_channel).same_ring(topo_->channel(channel));
        return continues ? 0 : 1;
    }

    // ---- the fold-link side, driven by the port ----
    // Indices into waiting(), oldest first, of the blocks leaving on channel `c`.
    std::vector<std::size_t> waiting_for(NocDim c) const {
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < waiting_.size(); ++i)
            if (waiting_[i].next && *waiting_[i].next == c) idx.push_back(i);
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            return older(waiting_[a].block, waiting_[b].block);
        });
        return idx;
    }
    const std::vector<Resident>& waiting() const { return waiting_; }
    // Take a waiting block out to start its departure; its slot stays held until release().
    Resident take(std::size_t i, Cycle now) {
        const Resident r = waiting_.at(i);
        Cycle& m = is_ring_queue(r.queue) ? stats_.max_residence : stats_.max_entry_wait;
        m = std::max(m, now - r.arrived);
        waiting_.erase(waiting_.begin() + static_cast<std::ptrdiff_t>(i));
        return r;
    }

    // ---- the local side: a BlockMover pushes a block out of the L3 into the entry queue ----
    bool push_from_l3(const NocBlock& b, Cycle now) {
        if (local_in_ || !reserve(local_queue())) return false;
        local_in_ = NocTransfer{b, now, now + config_.block_cycles, 0};
        ++stats_.accepted_from_l3;
        return true;
    }

    // ---- phases (NocFabric orders them across hubs and ports) ----
    void complete(Cycle now) {
        for (std::size_t i = 0; i < ring_out_.size(); ++i) {
            auto& t = out_flight_[i];
            if (!t || t->end > now) continue;
            auto& next = (*hubs_)[topo_->channel(ring_out_[i]).to_hub];
            next.arrive(t->block, now, next.ring_queue(ring_out_[i]), ring_out_[i]);
            release(t->src_queue);
            t.reset();
        }
        if (local_out_ && local_out_->end <= now) {
            delivered_.push_back({local_out_->block, now});
            ++stats_.delivered;
            release(local_out_->src_queue);
            local_out_.reset();
        }
        if (local_in_ && local_in_->end <= now) {
            arrive(local_in_->block, now, local_queue(), std::nullopt);
            local_in_.reset();
        }
    }

    void arbitrate(Cycle now) {
        // Ring outputs: per output, the oldest block that the next hub has room for.
        for (std::size_t i = 0; i < ring_out_.size(); ++i) {
            if (out_flight_[i]) continue;
            const NocDim c = ring_out_[i];
            auto& next = (*hubs_)[topo_->channel(c).to_hub];
            const std::size_t nq = next.ring_queue(c);
            for (std::size_t j : waiting_for(c)) {
                if (!next.reserve(nq, keep_free(waiting_[j], c))) continue;
                const Resident r = take(j, now);
                out_flight_[i] = NocTransfer{r.block, now, now + config_.block_cycles, r.queue};
                ++stats_.forwarded;
                break;
            }
        }
        // The L3 accepts every block delivered to it (its credit is the executor's, step 4b).
        if (!local_out_) {
            std::optional<std::size_t> best;
            for (std::size_t i = 0; i < waiting_.size(); ++i)
                if (!waiting_[i].next && (!best || older(waiting_[i].block, waiting_[*best].block)))
                    best = i;
            if (best) {
                const Resident r = take(*best, now);
                local_out_ = NocTransfer{r.block, now, now + config_.block_cycles, r.queue};
            }
        }
        watchdog(now);
    }

    // ---- IProcess ----
    std::vector<TimingEvent> tick(Cycle now) override {
        complete(now);
        arbitrate(now);
        return {};
    }
    bool is_idle() const override {
        if (local_in_ || local_out_) return false;
        for (const auto& t : out_flight_)
            if (t) return false;
        return true;
    }
    bool has_pending_work() const override { return !waiting_.empty(); }
    std::uint32_t id() const override { return config_.hub; }
    std::string name() const override { return "noc/hub[" + std::to_string(config_.hub) + "]"; }
    void reset() override {
        waiting_.clear();
        delivered_.clear();
        for (auto& t : out_flight_) t.reset();
        local_in_.reset();
        local_out_.reset();
        for (auto& q : queues_) q.reset();
        stats_ = {};
    }

    const Config& config() const { return config_; }
    const Stats& stats() const { return stats_; }
    struct Delivery { NocBlock block; Cycle cycle; };
    const std::vector<Delivery>& delivered() const { return delivered_; }

    // Age order: born first, then id. Stateless (§3.3): nothing but the blocks decides it.
    static bool older(const NocBlock& a, const NocBlock& b) {
        return a.born != b.born ? a.born < b.born : a.id < b.id;
    }

private:
    Config config_;
    const NocTopology* topo_;
    std::vector<NocHubProcess>* hubs_;
    std::vector<NocDim> in_channels_;                   // ring queue i <- in_channels_[i]
    std::vector<NocDim> entry_ports_;                   // ports whose fold link ends here
    std::vector<CreditPool> queues_;                    // ring queues, local entry, port entries
    std::vector<NocDim> ring_out_;                      // non-fold out-channels, channel order
    std::vector<std::optional<NocTransfer>> out_flight_;
    std::optional<NocTransfer> local_in_, local_out_;
    std::vector<Resident> waiting_;
    std::vector<Delivery> delivered_;
    Stats stats_;

    // TF-HUB-2: no block waits in a RING queue longer than the bound. Entry queues are not
    // watched: the bubble rule keeps the rings moving, not every entering block admitted, so an
    // entry wait is starvation to measure (max_entry_wait), not a wedge.
    void watchdog(Cycle now) {
        if (!config_.watchdog_cycles || stats_.watchdog_fired) return;
        for (const auto& r : waiting_)
            if (is_ring_queue(r.queue) && now - r.arrived > config_.watchdog_cycles) {
                stats_.watchdog_fired = true;
                stats_.watchdog_cycle = now;
                stats_.watchdog_block = r.block.id;
                return;
            }
    }
};

} // namespace sw::kpu::timing
