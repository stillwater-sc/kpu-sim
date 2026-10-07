// ============================================================================
// include/sw/kpu/timing/noc_fabric.hpp
// The CSP NoC: one NocHubProcess per L3 tile and one NocPortProcess per fold-end port, built
// from the ArrayLayout and the deployment's `noc` section (docs/plans/noc-port-arbitration.md
// step 4a).
//
// This is the fabric on its own. Its entry points are the pushes the machine has (§1.2):
//   inject    a DMA engine pushes a block into its port's input queue, bound for an L3 tile;
//   eject     a BlockMover pushes a block out of an L3 tile into its hub, bound for a DMA
//             engine's output queue on a port;
//   transfer  a BlockMover pushes a block from one L3 tile to another (the L3 -> L3 hop).
// Wiring it into ConcurrentTimingExecutor is step 4b: that needs a destination L3 tile per
// block, which the executor's pooled L3 does not have yet, and the DMA burst window
// (docs/plans/dram-bank-model.md step 3) on the engine side. Here the L3 accepts every block
// the hub delivers; the L3 credit stays the executor's.
//
// TICK ORDER. A cycle's completions happen before anything else in it, including the pushes
// a caller makes between ticks, so tick() ends by completing the transfers of the NEXT cycle:
//   1. every port moves its ring traffic across its fold link (ring-through, ejection);
//   2. every hub forwards its ring traffic and delivers to its L3;
//   3. every port injects;
//   4. the clock advances, and every hub and port completes the transfers that end at the new
//      cycle (blocks land, slots free, DMA writes retire).
// Step 1 before step 3 is ring first (§3.3): an injection gets a fold channel only if no ring
// block took it this cycle. Injected blocks land in an entry queue, not a ring queue, and enter
// a ring from there under the bubble rule (NocHubProcess), so injection never fills a ring.
// No result depends on which hub or port is visited first within a step, except who wins a
// contested slot, and that order (hub, then port index) is fixed.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/timing/noc_hub_process.hpp>
#include <sw/kpu/timing/noc_port_process.hpp>
#include <sw/kpu/timing/noc_topology.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing {

class NocFabric {
public:
    struct Config {
        std::size_t hub_buffer_blocks = 8;      // over the hub's 4 ring inputs, evenly
        std::size_t entry_queue_blocks = 2;     // per entry source (L3, each port ending here)
        std::size_t input_queue_blocks = 2;
        std::size_t output_queue_blocks = 0;    // 0 = derived (§3.4)
        Cycle block_cycles = 1;                 // one block across a link or a bus
        Cycle dma_write_latency = 0;            // an engine's DRAM write of one block
        Cycle dma_write_interval = 0;           // 0 = one block time
        std::vector<NocDim> engines_per_port;   // empty = one engine on every port; 0 allowed
        Cycle watchdog_cycles = 0;              // TF-HUB-2 bound; 0 = no watchdog
        bool bubble = true;                     // §3.5; off only to show what it prevents

        // From a device's declared `noc`. A block is `block_bytes`; it crosses a link at
        // movers.noc_bytes_per_cycle and an engine issues one block's DRAM write every
        // block_bytes / dma.bytes_per_cycle cycles. nullopt with the reason in `why`.
        static std::optional<Config> from(const sw::kpu::program::platform::DeviceSpecification& d,
                                          double block_bytes, Cycle dma_write_latency,
                                          std::string* why = nullptr) {
            auto fail = [&](std::string s) -> std::optional<Config> {
                if (why) *why = std::move(s);
                return std::nullopt;
            };
            if (!d.noc) return fail("the device declares no noc section");
            if (std::string p = sw::kpu::program::platform::noc_problem(d); !p.empty())
                return fail(p);
            if (!(block_bytes > 0)) return fail("block_bytes must be positive");
            if (!(d.movers.noc_bytes_per_cycle > 0) || !(d.dma.bytes_per_cycle > 0))
                return fail("movers.noc_bytes_per_cycle and dma.bytes_per_cycle must be positive");
            Config c;
            c.hub_buffer_blocks = d.noc->hub_buffer_blocks;
            c.input_queue_blocks = d.noc->port.input_queue_blocks;
            c.output_queue_blocks = d.noc->port.output_queue_blocks;
            c.block_cycles = static_cast<Cycle>(std::ceil(block_bytes / d.movers.noc_bytes_per_cycle));
            c.dma_write_latency = dma_write_latency;
            c.dma_write_interval = static_cast<Cycle>(std::ceil(block_bytes / d.dma.bytes_per_cycle));
            return c;
        }
    };

    NocFabric(const sw::kpu::program::platform::ArrayLayout& layout, const Config& config)
        : config_(config), topo_(layout) {
        if (config.block_cycles == 0)
            throw std::invalid_argument("NocFabric: block_cycles must be at least 1");
        // Below two per input the bubble rule can never admit a block; the spec refuses that
        // (noc_problem), but the fabric builds it so a test can show the wedge.
        if (config.hub_buffer_blocks == 0 ||
            config.hub_buffer_blocks % sw::kpu::program::platform::kNocHubInputs != 0)
            throw std::invalid_argument("NocFabric: hub_buffer_blocks (" +
                                        std::to_string(config.hub_buffer_blocks) +
                                        ") must be a positive multiple of the hub's 4 inputs");
        if (!config.engines_per_port.empty() &&
            config.engines_per_port.size() != topo_.port_count())
            throw std::invalid_argument("NocFabric: engines_per_port names " +
                                        std::to_string(config.engines_per_port.size()) +
                                        " ports; the layout has " +
                                        std::to_string(topo_.port_count()));
        // The ejection bus moves one block per block time: r_eject / S = 1 / block_cycles.
        derived_out_ = derived_output_queue_blocks(config.dma_write_latency, 1.0,
                                                   static_cast<double>(config.block_cycles));
        out_depth_ = config.output_queue_blocks ? config.output_queue_blocks : derived_out_;

        hubs_.reserve(topo_.hub_count());
        for (NocDim h = 0; h < topo_.hub_count(); ++h)
            hubs_.emplace_back(
                NocHubProcess::Config{h,
                                      config.hub_buffer_blocks /
                                          sw::kpu::program::platform::kNocHubInputs,
                                      config.entry_queue_blocks, config.block_cycles,
                                      config.watchdog_cycles, config.bubble},
                topo_, hubs_);
        ports_.reserve(topo_.port_count());
        for (NocDim k = 0; k < topo_.port_count(); ++k) {
            NocPortProcess::Config pc;
            pc.port = k;
            pc.engines = config.engines_per_port.empty() ? 1 : config.engines_per_port[k];
            pc.input_queue_blocks = config.input_queue_blocks;
            pc.output_queue_blocks = out_depth_;
            pc.derived_output_queue_blocks = derived_out_;
            pc.block_cycles = config.block_cycles;
            pc.dma_write_latency = config.dma_write_latency;
            pc.dma_write_interval = config.dma_write_interval ? config.dma_write_interval
                                                              : config.block_cycles;
            ports_.emplace_back(pc, topo_, hubs_);
        }
    }
    // The processes point at each other through hubs_, so the fabric stays where it was built.
    NocFabric(const NocFabric&) = delete;
    NocFabric& operator=(const NocFabric&) = delete;

    // A DMA engine on `port` pushes a block bound for hub `dst_hub`'s L3. False when the
    // engine's input queue has no credit; the engine holds the block and tries again.
    std::optional<std::uint64_t> inject(NocDim port, NocDim engine, NocDim dst_hub,
                                        std::uint64_t tag = 0) {
        NocBlock b = make(tag);
        b.dest = NocBlock::Dest::L3;
        b.dst_hub = dst_hub;
        if (dst_hub >= topo_.hub_count())
            throw std::out_of_range("NocFabric::inject: no hub " + std::to_string(dst_hub));
        if (!ports_.at(port).offer(engine, b)) return std::nullopt;
        ++next_id_;
        return b.id;
    }

    // A BlockMover on hub `hub`'s L3 pushes a block out to `engine` on `port`. False when the
    // hub has no buffer or its local input is busy.
    std::optional<std::uint64_t> eject(NocDim hub, NocDim port, NocDim engine,
                                       std::uint64_t tag = 0) {
        if (engine >= ports_.at(port).config().engines)
            throw std::out_of_range("NocFabric::eject: port " + std::to_string(port) +
                                    " has no engine " + std::to_string(engine));
        NocBlock b = make(tag);
        b.dest = NocBlock::Dest::Engine;
        b.dst_port = port;
        b.dst_engine = engine;
        b.dst_hub = topo_.exit_hub(port, hub);
        if (!hubs_.at(hub).push_from_l3(b, now_)) return std::nullopt;
        ++next_id_;
        return b.id;
    }

    // A BlockMover on hub `src_hub`'s L3 pushes a block to hub `dst_hub`'s L3 (the L3 -> L3
    // hop). False when the source hub has no buffer or its local input is busy.
    std::optional<std::uint64_t> transfer(NocDim src_hub, NocDim dst_hub, std::uint64_t tag = 0) {
        if (dst_hub >= topo_.hub_count())
            throw std::out_of_range("NocFabric::transfer: no hub " + std::to_string(dst_hub));
        NocBlock b = make(tag);
        b.dest = NocBlock::Dest::L3;
        b.dst_hub = dst_hub;
        if (!hubs_.at(src_hub).push_from_l3(b, now_)) return std::nullopt;
        ++next_id_;
        return b.id;
    }

    void tick() {
        for (auto& p : ports_) p.arbitrate_ring(now_);
        for (auto& h : hubs_) h.arbitrate(now_);
        for (auto& p : ports_) p.arbitrate_inject(now_);
        ++now_;
        for (auto& h : hubs_) h.complete(now_);
        for (auto& p : ports_) p.complete(now_);
    }

    // Nothing queued, buffered, in flight or being written.
    bool quiescent() const {
        for (const auto& h : hubs_)
            if (!h.is_complete()) return false;
        for (const auto& p : ports_)
            if (!p.is_complete()) return false;
        return true;
    }

    // Tick until quiescent or `max_cycles` more cycles have passed; true when quiescent.
    bool run_until_quiescent(Cycle max_cycles) {
        for (Cycle i = 0; i < max_cycles && !quiescent(); ++i) tick();
        return quiescent();
    }

    bool watchdog_fired() const {
        for (const auto& h : hubs_)
            if (h.stats().watchdog_fired) return true;
        return false;
    }

    Cycle now() const { return now_; }
    const Config& config() const { return config_; }
    const NocTopology& topology() const { return topo_; }
    const std::vector<NocHubProcess>& hubs() const { return hubs_; }
    const std::vector<NocPortProcess>& ports() const { return ports_; }
    std::size_t derived_output_queue_depth() const { return derived_out_; }
    std::size_t output_queue_depth() const { return out_depth_; }

private:
    Config config_;
    NocTopology topo_;
    std::vector<NocHubProcess> hubs_;
    std::vector<NocPortProcess> ports_;
    std::size_t derived_out_ = 1, out_depth_ = 1;
    Cycle now_ = 0;
    std::uint64_t next_id_ = 0;

    NocBlock make(std::uint64_t tag) const {
        NocBlock b;
        b.id = next_id_;
        b.tag = tag;
        b.born = now_;
        return b;
    }
};

} // namespace sw::kpu::timing
