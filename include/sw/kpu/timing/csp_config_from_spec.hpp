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
//   movers.streamers         split over the row and column streamer pools, at least one each
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

#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/dram_bridge.hpp>

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

    if (d.l3.capacity_tiles == 0)
        return fail("l3.capacity_tiles is not declared; the CSP L3 credit pool needs a size");
    c.l3_buffer_count = d.l3.capacity_tiles;

    c.num_block_movers = d.movers.block_movers;
    const std::size_t s = d.movers.streamers;
    c.num_row_streamers = s > 1 ? (s + 1) / 2 : 1;
    c.num_col_streamers = s > 1 ? s / 2 : 1;
    if (s == 1)
        out.unmapped.push_back("movers.streamers = 1: the executor has a row and a column "
                               "streamer pool and needs one in each, so it runs 2");

    out.unmapped.push_back("dma.bytes_per_cycle and movers.*_bytes_per_cycle: the executor's "
                           "rates are GB/s and latency fields; its defaults are kept");
    out.unmapped.push_back("macs_per_cycle: the executor's compute timing is compute_latency; "
                           "its default is kept");
    if (d.l2.banks_per_tile)
        out.unmapped.push_back("l2.banks_per_tile: an L-T2 bank count, not tile-sized L2 "
                               "buffers; l2_bank_count keeps its default");
    return out;
}

} // namespace sw::kpu::timing
