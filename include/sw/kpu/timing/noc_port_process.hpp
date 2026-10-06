// ============================================================================
// include/sw/kpu/timing/noc_port_process.hpp
// A fold-end NoC port controller (docs/plans/noc-port-arbitration.md §2, §3.2-3.4, step 4).
//
// The port sits on the fold link between hub_a and hub_b and owns both directed channels
// across it. Three kinds of traffic cross them:
//   ring-through  a block on its way around the loop, hub to hub;
//   injection     a DMA engine's block entering the machine, from its input queue into a hub;
//   ejection      a block a BlockMover pushed out of an L3, leaving the ring into its DMA
//                 engine's output queue (the engine writes it to DRAM later).
//
// TWO BUSES, EACH BINARY (§3.2, TF-PORT-1). Full duplex is an injection bus and an ejection
// bus, each carrying at most one block at a time. An injection also holds the fold channel
// into its hub; an ejection holds the fold channel out of the hub it leaves.
//
// THE ARBITER IS GREEDY AND STATELESS (§3.3). Ring traffic goes first: a block crossing the
// fold link takes its channel before any injection is considered, and every hub forwards its
// ring traffic before any port injects (NocFabric::tick). Then the oldest head across the
// input queues is injected, if its channel is free and its hub has a buffer. Age is a property
// of the block (when it entered the NoC), so there is no round-robin pointer and no history.
// Ring-first can starve injection; nothing bounds the wait, so TF-PORT-3 measures it.
//
// EJECTION NEVER STALLS THE NoC (§3.4, TF-PORT-2) when the output queue is at least the
// derived depth and the engine can write as fast as the bus ejects. An output-queue slot is
// reserved when the ejection starts and released when the engine's DRAM write completes, so a
// slot is held for one block time plus the write latency. Every cycle an ejection waits on a
// full output queue is counted and attributed: to RATE when the engine cannot retire blocks
// as fast as the bus delivers them (no depth fixes that), to DEPTH when the queue is below the
// derived depth, and otherwise left unattributed, which the derivation says cannot happen.
//
// The rate condition is read PER ENGINE: an output queue belongs to one engine, and the whole
// ejection stream can be bound for it, so the derived depth is sufficient exactly when that
// engine retires a block at least once per block time (dma_write_interval <= block_cycles).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/timing/credit_pool.hpp>
#include <sw/kpu/timing/noc_hub_process.hpp>
#include <sw/kpu/timing/noc_topology.hpp>
#include <sw/kpu/timing/process_interface.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing {

// §3.4: depth >= ceil(L_dma_write * r_eject / S) + 1, in blocks.
inline std::size_t derived_output_queue_blocks(Cycle dma_write_latency,
                                               double eject_bytes_per_cycle, double block_bytes) {
    if (!(eject_bytes_per_cycle > 0) || !(block_bytes > 0))
        throw std::invalid_argument("derived_output_queue_blocks: rate and block size must be "
                                    "positive");
    return static_cast<std::size_t>(
               std::ceil(static_cast<double>(dma_write_latency) * eject_bytes_per_cycle /
                         block_bytes)) + 1;
}

class NocPortProcess : public IProcess {
public:
    struct Config {
        NocDim port = 0;
        NocDim engines = 1;                     // DMA engines attached: one queue pair each
        std::size_t input_queue_blocks = 2;     // per engine
        std::size_t output_queue_blocks = 1;    // per engine, resolved (never 0 here)
        std::size_t derived_output_queue_blocks = 1;
        Cycle block_cycles = 1;                 // one block across the fold link or a bus
        Cycle dma_write_latency = 0;            // an engine's DRAM write of one block
        Cycle dma_write_interval = 1;           // cycles between an engine's write issues
    };

    enum class Kind : std::uint8_t { RingThrough, Injection, Ejection };

    // One crossing of the fold link, in start order: what tests read the arbiter's choices from.
    struct Crossing {
        Kind kind;
        std::uint64_t block = 0;
        NocDim engine = 0;                      // injection/ejection only
        Cycle start = 0;
    };

    struct Stats {
        std::uint64_t injected = 0, ring_through = 0, ejected = 0, written = 0;
        Cycle injection_busy_cycles = 0, ejection_busy_cycles = 0;
        Cycle max_input_wait = 0;               // TF-PORT-3
        Cycle eject_stall_cycles = 0;           // TF-PORT-2: an ejection waited on a full queue
        Cycle eject_stalls_rate = 0, eject_stalls_depth = 0, eject_stalls_unattributed = 0;
    };

    NocPortProcess(const Config& config, const NocTopology& topo,
                   std::vector<NocHubProcess>& hubs)
        : config_(config), topo_(&topo), hubs_(&hubs) {
        if (config.engines == 0)
            throw std::invalid_argument(name() + ": a port controller needs an attached engine");
        if (config.input_queue_blocks == 0 || config.output_queue_blocks == 0)
            throw std::invalid_argument(name() + ": queue depths must be at least 1");
        const auto& p = topo.port(config.port);
        hub_a_ = p.hub_a;
        hub_b_ = p.hub_b;
        ch_ab_ = topo.fold_from(config.port, hub_a_);
        ch_ba_ = topo.fold_from(config.port, hub_b_);
        engines_.reserve(config.engines);
        for (NocDim e = 0; e < config.engines; ++e)
            engines_.push_back(Engine{CreditPool(config.output_queue_blocks), {}, {}, {}, 0});
    }

    // ---- the engine side: a DMA engine pushes a block into its input queue, with credit ----
    bool offer(NocDim engine, const NocBlock& b) {
        auto& q = engines_.at(engine).in_q;
        if (q.size() >= config_.input_queue_blocks) return false;
        q.push_back(b);
        return true;
    }
    std::size_t input_queue_size(NocDim engine) const { return engines_.at(engine).in_q.size(); }
    std::size_t output_queue_held(NocDim engine) const {
        return engines_.at(engine).out_credits.outstanding();
    }

    // ---- phases ----
    void complete(Cycle now) {
        for (auto* slot : {&fold_ab_, &fold_ba_}) {
            auto& f = *slot;
            if (!f || f->t.end > now) continue;
            switch (f->kind) {
                case Kind::RingThrough: {
                    auto& to = (*hubs_)[f->to];
                    to.arrive(f->t.block, now, to.ring_queue(f->channel), f->channel);
                    (*hubs_)[f->from].release(f->t.src_queue);
                    break;
                }
                case Kind::Injection: {
                    auto& to = (*hubs_)[f->to];
                    to.arrive(f->t.block, now, to.port_queue(config_.port), std::nullopt);
                    injection_bus_.reset();
                    break;
                }
                case Kind::Ejection:
                    (*hubs_)[f->from].release(f->t.src_queue);
                    engines_[f->t.block.dst_engine].staged.push_back(f->t.block);
                    ejection_bus_.reset();
                    break;
            }
            f.reset();
        }
        // An engine issues a staged block's DRAM write, then retires the writes that end now,
        // so a write of latency L holds its output-queue slot exactly L cycles past arrival.
        for (auto& eng : engines_) {
            if (!eng.staged.empty() && eng.next_issue <= now) {
                eng.writing.push_back({eng.staged.front(), now, now + config_.dma_write_latency});
                eng.staged.pop_front();
                eng.next_issue = now + config_.dma_write_interval;
            }
            while (!eng.writing.empty() && eng.writing.front().end <= now) {
                written_.push_back({eng.writing.front().block, now});
                eng.writing.pop_front();
                eng.out_credits.release();
                ++stats_.written;
            }
        }
    }

    // Ring traffic across the fold link: ring-through and ejection, oldest first per channel.
    void arbitrate_ring(Cycle now) {
        bool stalled = false;
        for (const bool ab : {true, false}) {
            auto& slot = ab ? fold_ab_ : fold_ba_;
            if (slot) continue;
            const NocDim from = ab ? hub_a_ : hub_b_, to = ab ? hub_b_ : hub_a_;
            const NocDim ch = ab ? ch_ab_ : ch_ba_;
            auto& src = (*hubs_)[from];
            auto& dst = (*hubs_)[to];
            for (std::size_t i : src.waiting_for(ch)) {
                const NocBlock& b = src.waiting()[i].block;
                if (ejects_here(b, from)) {
                    if (ejection_bus_) continue;
                    if (!engines_.at(b.dst_engine).out_credits.acquire()) {
                        stalled = true;
                        continue;
                    }
                    const auto r = src.take(i, now);
                    slot = Fold{{r.block, now, now + config_.block_cycles, r.queue},
                                Kind::Ejection, from, to, ch};
                    ejection_bus_ = slot->t;
                    log_.push_back({Kind::Ejection, r.block.id, r.block.dst_engine, now});
                    ++stats_.ejected;
                    break;
                }
                if (!dst.reserve(dst.ring_queue(ch), src.keep_free(src.waiting()[i], ch))) continue;
                const auto r = src.take(i, now);
                slot = Fold{{r.block, now, now + config_.block_cycles, r.queue},
                            Kind::RingThrough, from, to, ch};
                log_.push_back({Kind::RingThrough, r.block.id, 0, now});
                ++stats_.ring_through;
                break;
            }
        }
        if (stalled) {
            // Every engine of a port has the same write rate and queue depth, so the
            // attribution does not depend on which engine's queue was full.
            ++stats_.eject_stall_cycles;
            if (config_.dma_write_interval > config_.block_cycles) ++stats_.eject_stalls_rate;
            else if (config_.output_queue_blocks < config_.derived_output_queue_blocks)
                ++stats_.eject_stalls_depth;
            else ++stats_.eject_stalls_unattributed;
        }
    }

    // Injection: the oldest input-queue head whose fold channel is free and whose hub has a
    // slot in this port's entry queue. The block enters a ring from there, under the bubble.
    void arbitrate_inject(Cycle now) {
        if (!injection_bus_) {
            std::vector<NocDim> order;
            for (NocDim e = 0; e < engines_.size(); ++e)
                if (!engines_[e].in_q.empty()) order.push_back(e);
            std::sort(order.begin(), order.end(), [&](NocDim x, NocDim y) {
                return NocHubProcess::older(engines_[x].in_q.front(), engines_[y].in_q.front());
            });
            for (NocDim e : order) {
                const NocBlock& b = engines_[e].in_q.front();
                const NocDim to = topo_->injection_hub(config_.port, b.dst_hub);
                const bool ab = to == hub_b_;
                auto& slot = ab ? fold_ab_ : fold_ba_;
                auto& hub = (*hubs_)[to];
                if (slot || !hub.reserve(hub.port_queue(config_.port))) continue;
                const NocBlock in = b;
                engines_[e].in_q.pop_front();
                slot = Fold{{in, now, now + config_.block_cycles, 0}, Kind::Injection,
                            ab ? hub_a_ : hub_b_, to, ab ? ch_ab_ : ch_ba_};
                injection_bus_ = slot->t;
                log_.push_back({Kind::Injection, in.id, e, now});
                stats_.max_input_wait = std::max(stats_.max_input_wait, now - in.born);
                ++stats_.injected;
                break;
            }
        }
        if (injection_bus_) ++stats_.injection_busy_cycles;
        if (ejection_bus_) ++stats_.ejection_busy_cycles;
    }

    // ---- IProcess ----
    std::vector<TimingEvent> tick(Cycle now) override {
        complete(now);
        arbitrate_ring(now);
        arbitrate_inject(now);
        return {};
    }
    bool is_idle() const override {
        if (fold_ab_ || fold_ba_) return false;
        for (const auto& e : engines_)
            if (!e.staged.empty() || !e.writing.empty()) return false;
        return true;
    }
    bool has_pending_work() const override {
        for (const auto& e : engines_)
            if (!e.in_q.empty()) return true;
        return false;
    }
    std::uint32_t id() const override { return config_.port; }
    std::string name() const override { return "noc/port[" + std::to_string(config_.port) + "]"; }
    void reset() override {
        fold_ab_.reset();
        fold_ba_.reset();
        injection_bus_.reset();
        ejection_bus_.reset();
        for (auto& e : engines_) {
            e.in_q.clear();
            e.staged.clear();
            e.writing.clear();
            e.out_credits.reset();
            e.next_issue = 0;
        }
        log_.clear();
        written_.clear();
        stats_ = {};
    }

    const Config& config() const { return config_; }
    const Stats& stats() const { return stats_; }
    const std::vector<Crossing>& crossings() const { return log_; }
    // TF-PORT-1: what each bus holds right now (at most one block, by construction).
    bool injection_bus_busy() const { return injection_bus_.has_value(); }
    bool ejection_bus_busy() const { return ejection_bus_.has_value(); }
    const std::vector<NocHubProcess::Delivery>& written() const { return written_; }

private:
    struct Fold {
        NocTransfer t;
        Kind kind;
        NocDim from, to;
        NocDim channel;                         // the fold channel it holds
    };
    struct Write {
        NocBlock block;
        Cycle start = 0, end = 0;
    };
    struct Engine {
        CreditPool out_credits;                 // output-queue slots, ejection to write done
        std::deque<NocBlock> in_q;              // waiting to be injected
        std::deque<NocBlock> staged;            // ejected, waiting for the engine's write unit
        std::deque<Write> writing;              // DRAM writes in flight, in issue order
        Cycle next_issue = 0;
    };

    Config config_;
    const NocTopology* topo_;
    std::vector<NocHubProcess>* hubs_;
    NocDim hub_a_ = 0, hub_b_ = 0, ch_ab_ = 0, ch_ba_ = 0;
    std::optional<Fold> fold_ab_, fold_ba_;
    std::optional<NocTransfer> injection_bus_, ejection_bus_;
    std::vector<Engine> engines_;
    std::vector<Crossing> log_;
    std::vector<NocHubProcess::Delivery> written_;
    Stats stats_;

    bool ejects_here(const NocBlock& b, NocDim from) const {
        return b.dest == NocBlock::Dest::Engine && b.dst_port == config_.port && b.dst_hub == from;
    }
};

} // namespace sw::kpu::timing
