// ============================================================================
// include/sw/kpu/program/characterize/characterization.hpp
// Structural + first-order modeled metrics for an L0 TileProgram on a device,
// plus CSV / JSON / Chrome-trace emitters for the DoE harness.
//
// Structural metrics (op counts, MAC/byte volume, arithmetic intensity, footprint)
// are exact from the tile program. Performance/energy are modeled (device_model.hpp)
// pending the L1 timing layer — the harness sweeps these across the experiment grid.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/characterize/device_model.hpp>
#include <sw/kpu/program/characterize/tile_dag.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::program::characterize {

struct Metrics {
    // structural (exact)
    std::size_t ops = 0, feeds = 0, drains = 0, computes = 0;
    double total_macs = 0.0;
    double total_move_bytes = 0.0;
    double arithmetic_intensity = 0.0;   // flops / byte moved (2*MAC / bytes)
    std::size_t distinct_tiles = 0;      // total tiles across all operands
    std::size_t peak_live_tiles = 0;     // max concurrently-live tiles (footprint proxy)

    // modeled performance (cycles)
    double critical_path_cycles = 0.0;   // makespan at unlimited compute tiles
    double makespan_cycles = 0.0;        // list-scheduled on the device
    double lower_bound_cycles = 0.0;     // max(critical path, work/resources)
    double compute_util = 0.0;
    double movement_util = 0.0;
    bool compute_bound = false;          // compute saturates before movement

    // modeled energy (pJ)
    double energy_compute_pj = 0.0;
    double energy_move_pj = 0.0;
    double energy_leak_pj = 0.0;
    double energy_total_pj = 0.0;

    // feasibility
    bool feasible = true;                // peak_live_tiles fits L3 (if bounded)

    // dataflow (populated only when an L1 StreamProgram is supplied)
    std::string dataflow;                // space-time mapping name
    std::string stationary;              // stationary operand ("" = none/hex)
    int c_bubble = 0;                    // C-drain bubble (0 = dense)
    std::string network;                 // required fabric ("Mesh2D" / "Hexagonal(+overlay)")
};

// Peak number of simultaneously-live tiles over program order (footprint proxy):
// a tile is live from its first to its last touch.
inline std::size_t peak_live_tiles(const TileProgram& prog) {
    const auto& ops = prog.ops();
    std::map<std::string, std::pair<std::size_t, std::size_t>> span;
    auto key = [](const TileCoord& c) {
        return c.operand + "#" + std::to_string(c.ti) + "#" + std::to_string(c.tj);
    };
    auto touch = [&](const TileCoord& c, std::size_t i) {
        auto it = span.find(key(c));
        if (it == span.end()) span[key(c)] = {i, i};
        else it->second.second = i;
    };
    for (std::size_t i = 0; i < ops.size(); ++i) {
        for (const auto& c : ops[i].inputs)  touch(c, i);
        for (const auto& c : ops[i].outputs) touch(c, i);
    }
    std::vector<int> delta(ops.size() + 1, 0);
    for (const auto& [k, s] : span) { (void)k; ++delta[s.first]; --delta[s.second + 1]; }
    std::size_t live = 0, peak = 0;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        live += static_cast<std::size_t>(delta[i]);
        peak = std::max(peak, live);
    }
    return peak;
}

inline std::size_t distinct_tiles(const TileProgram& prog) {
    std::size_t n = 0;
    for (const auto& name : prog.operand_order()) {
        const auto& op = prog.operand(name);
        n += static_cast<std::size_t>(op.n_tile_rows()) * op.n_tile_cols();
    }
    return n;
}

// Compute all metrics for a program on a device (does not run the functional
// reference — the harness owns correctness because it knows the algorithm/oracle).
// Supplying an L1 StreamProgram makes the schedule systolic + dataflow-sensitive and
// populates the dataflow metrics.
inline Metrics characterize_program(const TileProgram& prog, const DeviceDescriptor& dev,
                                    const stream::StreamProgram* l1 = nullptr) {
    Metrics m;
    m.ops = prog.ops().size();
    m.feeds  = prog.count(TileOpKind::Feed);
    m.drains = prog.count(TileOpKind::Drain);
    m.computes = prog.count(TileOpKind::MatMulAccum) + prog.count(TileOpKind::LuDiagFactor) +
                 prog.count(TileOpKind::TrsmLowerLeft) + prog.count(TileOpKind::TrsmUpperRight);

    TileDag dag(prog, dev, l1);
    auto sch = dag.list_schedule();

    if (l1) {
        m.dataflow = l1->map.name;
        m.stationary = l1->map.stationary_operand();   // from the projection (C is held then drained)
        if (const auto* c = l1->signature("C")) m.c_bubble = c->bubble();
        m.network = stream::to_string(l1->network.required);
        if (l1->network.needs_overlay_on_mesh) m.network += "+overlay";
    }

    m.total_macs = dag.total_macs();
    m.total_move_bytes = dag.total_move_bytes();
    m.arithmetic_intensity = m.total_move_bytes > 0 ? (2.0 * m.total_macs) / m.total_move_bytes : 0.0;
    m.distinct_tiles = distinct_tiles(prog);
    m.peak_live_tiles = peak_live_tiles(prog);

    m.critical_path_cycles = dag.critical_path_cycles();
    m.makespan_cycles = sch.makespan;
    m.lower_bound_cycles = sch.lower_bound;
    m.compute_util = sch.compute_util;
    m.movement_util = sch.movement_util;
    const double C = std::max<Dim>(dev.compute_tiles, 1);
    const double M = std::max<Dim>(dev.move_lanes, 1);
    m.compute_bound = (dag.compute_work_cycles() / C) >= (dag.movement_work_cycles() / M);

    m.energy_compute_pj = m.total_macs * dev.pj_per_mac;
    m.energy_move_pj = m.total_move_bytes * dev.pj_per_byte;
    m.energy_leak_pj = dev.static_pj_per_tile_per_cyc *
                       (double(dev.compute_tiles) + double(dev.move_lanes)) * m.makespan_cycles;
    m.energy_total_pj = m.energy_compute_pj + m.energy_move_pj + m.energy_leak_pj;

    m.feasible = (dev.l3_tiles == 0) || (m.peak_live_tiles <= dev.l3_tiles);
    return m;
}

// ---- CSV -------------------------------------------------------------------
// The harness prepends factor columns (algo,size,tile,compute_tiles,topology).
inline std::string metrics_csv_header() {
    return "ops,feeds,drains,computes,macs,move_bytes,arith_intensity,"
           "distinct_tiles,peak_live_tiles,critical_path,makespan,lower_bound,"
           "compute_util,movement_util,compute_bound,"
           "energy_compute_pj,energy_move_pj,energy_leak_pj,energy_total_pj,feasible";
}
inline std::string metrics_csv_row(const Metrics& m) {
    std::ostringstream s;
    s << m.ops << ',' << m.feeds << ',' << m.drains << ',' << m.computes << ','
      << m.total_macs << ',' << m.total_move_bytes << ',' << m.arithmetic_intensity << ','
      << m.distinct_tiles << ',' << m.peak_live_tiles << ','
      << m.critical_path_cycles << ',' << m.makespan_cycles << ',' << m.lower_bound_cycles << ','
      << m.compute_util << ',' << m.movement_util << ',' << (m.compute_bound ? 1 : 0) << ','
      << m.energy_compute_pj << ',' << m.energy_move_pj << ',' << m.energy_leak_pj << ','
      << m.energy_total_pj << ',' << (m.feasible ? 1 : 0);
    return s.str();
}

// ---- Chrome trace (chrome://tracing / Perfetto) ----------------------------
// Emit the list-scheduled tile sequence as duration events, one lane per resource,
// so the tile program's organization and ordering are directly observable. Time
// unit = modeled cycles.
inline void write_chrome_trace(const TileProgram& prog, const DeviceDescriptor& dev,
                               const std::string& path) {
    TileDag dag(prog, dev);
    dag.list_schedule();
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[trace] failed to open " << path << " for writing\n";
        return;
    }
    f << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n";
    bool first = true;
    auto ev = [&](const std::string& s) {
        if (!first) f << ",\n";
        f << s; first = false;
    };
    // lane metadata: compute workers on pid 1, movement lanes on pid 2
    for (Dim w = 0; w < std::max<Dim>(dev.compute_tiles, 1); ++w)
        ev("{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":" + std::to_string(w) +
           ",\"args\":{\"name\":\"CF" + std::to_string(w) + "\"}}");
    for (Dim w = 0; w < std::max<Dim>(dev.move_lanes, 1); ++w)
        ev("{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":2,\"tid\":" + std::to_string(w) +
           ",\"args\":{\"name\":\"MOVE" + std::to_string(w) + "\"}}");
    for (const auto& n : dag.nodes()) {
        const TileOp& op = prog.ops()[n.op_index];
        const std::string tile = op.outputs.empty()
            ? (op.inputs.empty() ? std::string() : op.inputs[0].to_string())
            : op.outputs[0].to_string();
        const int pid = n.work.is_compute ? 1 : 2;
        std::ostringstream s;
        s << "{\"ph\":\"X\",\"pid\":" << pid << ",\"tid\":" << n.worker
          << ",\"ts\":" << n.start << ",\"dur\":" << (n.duration > 0 ? n.duration : 1)
          << ",\"name\":\"" << to_string(op.kind) << ' ' << tile << "\"}";
        ev(s.str());
    }
    f << "\n]}\n";
}

} // namespace sw::kpu::program::characterize
