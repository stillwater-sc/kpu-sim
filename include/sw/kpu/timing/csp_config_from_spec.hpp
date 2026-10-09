// ============================================================================
// include/sw/kpu/timing/csp_config_from_spec.hpp
// A deployment's device, as the CSP cycle-accurate executor's Config
// (docs/plans/noc-port-arbitration.md §4.1, step 4b.1).
//
// Until now every ConcurrentTimingExecutor::Config was built by hand, so nothing tied a CSP
// run to the device a deployment declares. This is the one place that mapping lives. It maps
// what the executor has a counterpart for, and names what it does not map rather than
// inventing a value:
//
//   memory.controllers       num_memory_controllers (1 when undeclared)
//   memory.dram              dram: the hosted cycle-accurate LPDDR5 controller
//   dma.engines              num_dma_engines, numbered per controller like the floorplan:
//                            engine i = mc * (engines / controllers) + e
//   l3.capacity_tiles        l3_buffer_count (tile-sized L3 buffers, the credit pool)
//   movers.block_movers      num_block_movers
//   the array layout         l3_tiles (one credit pool and Tag CAM each) and the L3 tile of
//                            every BlockMover (its site); without a layout, one pooled L3
//   movers.streamers         split over the row and column streamer pools, at least one each
//   noc.port.output_queue_blocks   dma_store_buffer_blocks, when declared (0 = derived: noted)
//   dma.window               dma_window: bursts each engine keeps in flight (needs memory.dram)
//
// NOT MAPPED, listed in `unmapped`: the spec's rates (dma.bytes_per_cycle and the movers'
// bytes per cycle) against the executor's GB/s and latency fields, macs_per_cycle against its
// compute latency, and l2.banks_per_tile, which is an L-T2 bank count and not a count of
// tile-sized L2 buffers. The executor keeps its defaults for those.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/floorplan.hpp>
#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/dram_bridge.hpp>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::timing {

struct CspDeviceConfig {
    ConcurrentTimingExecutor::Config config;
    std::vector<std::string> unmapped;      // spec fields the executor kept its defaults for
};

// The executor Config for device `d`, or nullopt with the reason in `why`.
inline std::optional<CspDeviceConfig> csp_config_from(
        const sw::kpu::program::platform::DeviceSpecification& d, std::string* why = nullptr) {
    auto fail = [&](std::string s) -> std::optional<CspDeviceConfig> {
        if (why) *why = "device \"" + d.name + "\": " + std::move(s);
        return std::nullopt;
    };
    CspDeviceConfig out;
    auto& c = out.config;

    const std::size_t mcs = d.memory.controllers ? *d.memory.controllers : 1;
    if (mcs == 0) return fail("memory.controllers is zero");
    if (d.dma.engines == 0) return fail("dma.engines is zero; the executor needs an engine");
    if (d.dma.engines % mcs != 0)
        return fail("dma.engines (" + std::to_string(d.dma.engines) + ") does not split evenly over " +
                    std::to_string(mcs) + " memory controllers");
    c.num_memory_controllers = mcs;
    c.num_dma_engines = d.dma.engines;
    const std::size_t per = d.dma.engines / mcs;
    c.dma_engine_controller.resize(d.dma.engines);
    for (std::size_t i = 0; i < d.dma.engines; ++i) c.dma_engine_controller[i] = i / per;

    if (d.memory.dram) {
        try {
            c.dram = DramHosting::of(d);
        } catch (const std::exception& e) {
            return fail(std::string("memory.dram cannot be hosted: ") + e.what());
        }
    }

    if (d.dma.window) {
        if (!c.dram) return fail("dma.window needs memory.dram: bursts are the DRAM's");
        c.dma_window = *d.dma.window;
    }

    if (d.l3.capacity_tiles == 0)
        return fail("l3.capacity_tiles is not declared; the CSP L3 credit pool needs a size");
    c.l3_buffer_count = d.l3.capacity_tiles;

    // The store buffer IS the NoC port's per-engine output queue (plan §1.2): where a
    // BlockMover's ejection lands until the engine writes it to DRAM.
    if (d.noc && d.noc->port.output_queue_blocks > 0)
        c.dma_store_buffer_blocks = d.noc->port.output_queue_blocks;
    else if (d.noc)
        out.unmapped.push_back("noc.port.output_queue_blocks = 0 (derived from the DMA write "
                               "latency, plan §3.4): the store buffer keeps its default of " +
                               std::to_string(c.dma_store_buffer_blocks) + " until the NoC is wired");

    c.num_block_movers = d.movers.block_movers;

    // L3 placement (step 4b.3): the layout's L3 tiles, and each mover on the tile it sits on.
    std::string layout_why;
    if (const auto L = sw::kpu::program::platform::ArrayLayout::of(d, &layout_why)) {
        const auto& sites = L->block_movers();
        if (sites.size() != d.movers.block_movers)
            return fail("movers.block_movers (" + std::to_string(d.movers.block_movers) +
                        ") is not the layout's " + std::to_string(sites.size()) +
                        " BlockMovers (one per L3 edge that abuts a compute tile)");
        if (d.l3.capacity_tiles < L->l3_count())
            return fail("l3.capacity_tiles (" + std::to_string(d.l3.capacity_tiles) +
                        ") cannot give each of " + std::to_string(L->l3_count()) +
                        " L3 tiles a buffer");
        c.l3_tiles = L->l3_count();
        c.block_mover_l3_tile.clear();
        for (const auto& s : sites) c.block_mover_l3_tile.push_back(s.l3);
    } else {
        out.unmapped.push_back("no array layout (" + layout_why + "): one pooled L3, every "
                               "BlockMover on it");
    }
    const std::size_t s = d.movers.streamers;
    c.num_row_streamers = s > 1 ? (s + 1) / 2 : 1;
    c.num_col_streamers = s > 1 ? s / 2 : 1;
    if (s == 1)
        out.unmapped.push_back("movers.streamers = 1: the executor has a row and a column "
                               "streamer pool and needs one in each, so it runs 2");

    out.unmapped.push_back("dma.bytes_per_cycle and movers.*_bytes_per_cycle: the executor's "
                           "rates are GB/s and latency fields; its defaults are kept");
    // The compute fabric (docs/plans/system-schedule-debugger.md §3.2): one compute at a time
    // per compute tile, its latency from the MAC rate.
    if (d.compute_tiles == 0) return fail("compute_tiles is zero; the executor needs a compute tile");
    if (!(d.macs_per_cycle > 0.0)) return fail("macs_per_cycle must be positive");
    c.num_compute_tiles = d.compute_tiles;
    c.macs_per_cycle = d.macs_per_cycle;
    if (d.l2.banks_per_tile)
        out.unmapped.push_back("l2.banks_per_tile: an L-T2 bank count, not tile-sized L2 "
                               "buffers; l2_bank_count keeps its default");
    return out;
}


// The NoC wiring for device `d` (step 4b.4; opt-in: set Config::noc to it), or nullopt with
// the reason in `why`. Tile-granular until DRAM step 3 (Q11): one NoC block is one tile of
// `block_bytes`. The fabric's port "write" is the hand-off into the engine's store buffer
// (latency 0, one per cycle), and its output queue is that buffer's depth: the MC stays the
// only DRAM writer, and because a BlockMover reserves the buffer slot before the block leaves
// L3, an ejection can never find the output queue full.
inline std::optional<ConcurrentTimingExecutor::Config::NocWiring> csp_noc_wiring(
        const sw::kpu::program::platform::DeviceSpecification& d, double block_bytes,
        std::size_t store_buffer_blocks, std::string* why = nullptr) {
    using sw::kpu::program::platform::ArrayLayout;
    using sw::kpu::program::platform::DeploymentSpec;
    auto fail = [&](std::string s) -> std::optional<ConcurrentTimingExecutor::Config::NocWiring> {
        if (why) *why = "device \"" + d.name + "\": " + std::move(s);
        return std::nullopt;
    };
    std::string reason;
    const auto L = ArrayLayout::of(d, &reason);
    if (!L) return fail("no array layout: " + reason);
    if (!L->has_noc()) return fail(L->noc_reason());
    auto fabric = NocFabric::Config::from(d, block_bytes, 0, &reason);
    if (!fabric) return fail(reason);
    fabric->dma_write_latency = 0;
    fabric->dma_write_interval = 1;
    fabric->output_queue_blocks = store_buffer_blocks;

    // A public entry point: it checks what csp_config_from checks, rather than trusting it ran.
    const std::size_t mcs = d.memory.controllers ? *d.memory.controllers : 1;
    if (mcs == 0 || d.dma.engines == 0 || d.dma.engines % mcs != 0)
        return fail("dma.engines (" + std::to_string(d.dma.engines) + ") does not split evenly "
                    "over " + std::to_string(mcs) + " memory controllers");
    const std::size_t per = d.dma.engines / mcs;

    DeploymentSpec one;
    one.devices = {d};
    std::vector<sw::kpu::program::platform::DmaPortAttachment> attach;
    try {
        attach = sw::kpu::program::platform::dma_port_attachment(one);
    } catch (const sw::kpu::program::platform::FloorplanError& e) {
        return fail(std::string("no engine-to-port attachment: ") + e.what());
    }
    ConcurrentTimingExecutor::Config::NocWiring w{*L, *fabric, {}};
    w.engine_port.assign(d.dma.engines, {0, 0});
    std::vector<bool> placed(d.dma.engines, false);
    std::map<NocDim, NocDim> on_port;
    for (const auto& a : attach) {
        if (static_cast<std::size_t>(a.mc) >= mcs || static_cast<std::size_t>(a.engine) >= per)
            return fail("the floorplan attaches mc[" + std::to_string(a.mc) + "]/dma[" +
                        std::to_string(a.engine) + "], outside the declared " + std::to_string(mcs) +
                        " controllers of " + std::to_string(per) + " engines");
        const std::size_t i = static_cast<std::size_t>(a.mc) * per + a.engine;
        if (placed[i])
            return fail("the floorplan attaches mc[" + std::to_string(a.mc) + "]/dma[" +
                        std::to_string(a.engine) + "] to two ports");
        w.engine_port[i] = {a.port, on_port[a.port]++};
        placed[i] = true;
    }
    for (std::size_t i = 0; i < placed.size(); ++i)
        if (!placed[i])
            return fail("DMA engine " + std::to_string(i) + " is attached to no NoC port, so its "
                        "loads could not enter the NoC");
    w.fabric.engines_per_port.assign(L->ports().size(), 0);
    for (const auto& [port, n] : on_port) w.fabric.engines_per_port.at(port) = n;
    return w;
}

} // namespace sw::kpu::timing
