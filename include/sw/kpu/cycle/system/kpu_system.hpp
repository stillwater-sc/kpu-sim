#pragma once
// ============================================================================
// include/sw/kpu/cycle/system/kpu_system.hpp
// The join (#548; docs/plans/cycle-based-kpu-prototype.md §6, step 3c): the memory subsystem and
// the compute subsystem on one simulator, DRAM -> L3 -> L2 -> L1 -> array -> Combiner -> L2 -> L3
// -> DRAM, run from one CSP program. What kpu-run --system runs.
//
// THE PROGRAM (plan_system). One L3 program from the walker's actions, in the program's order:
//   Load, Store, Release  the DMA engine's and the L3 tile's (Phase 1)
//   Inherit               a Fill: the tile is resident from the start
//   Move                  a Read: the BlockMover's Move of the slot's tile into L2 (a consumer)
//   Writeback             a Write: the BlockMover's Writeback of a result tile (a producer)
// with the DRAM layout and L-B's DRAM image of plan_l3 (Phase 1), and the BlockMover's, the L2's and
// the Streamer's programs of plan_compute (Phase 2), its L3 the memory subsystem's: slots sized by
// plan_l3, reused after their Releases (the Loads run in time), each Writeback under its L3 slot's
// credit. Nothing is hypothesized: C reaches DRAM from the array.
//
// THE MACHINE (SystemMachine). One LPDDR5 channel (3.2 GHz), the DMA engine and the L3 tile on a
// direct link (Phase 1's single-L3 devices), and the compute subsystem as build_compute wires it
// (compute_subsystem.hpp), the BlockMover on the L3 tile's other side (l3_tile.hpp).
//
// WHAT A RUN CHECKS (run_system):
//   data        DRAM ends with every stored tile as L-B leaves it, bit for bit (int32 C exact)
//   invariants  every cycle: the L3 slots' credits three ways (the engine, the BlockMover, the
//               tile), the L2 slots' three ways (the BlockMover, the Streamer, the L2), and each
//               component's and link's own checks; a fault stops the run as a violation
//   the laws    on every component's and link's counters (check_laws), and the PEs' clocks by cause
//   deadlock    the detector's report, if no progress is made
//
// THE SUMMARY (format_system_summary): the run, then the memory subsystem's lines and the compute
// subsystem's (format_memory_lines, format_compute_lines), then every component's and link's row.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/compute/compute_subsystem.hpp>
#include <sw/kpu/cycle/compute/csp_compute_program.hpp>
#include <sw/kpu/cycle/memory/csp_l3_program.hpp>
#include <sw/kpu/cycle/memory/dma_engine.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/memory/l3_tile.hpp>
#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/memory/memory_subsystem.hpp>
#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/trace.hpp>
#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace sw::kpu::cycle {

struct SystemPlan {
    CspL3Plan memory;          // the DRAM layout, DRAM before the run and L-B's DRAM after (plan_l3)
    L3Program l3;              // the joined L3 program (its Fills replaced by the BlockMover's Writes)
    CspComputePlan compute;    // the BlockMover's, the L2's and the Streamer's programs
    std::size_t loads = 0, stores = 0, releases = 0, fills = 0, reads = 0, writes = 0;
};

// `inputs`: every operand as DRAM holds it before the run; `result`: as L-B leaves it after.
// `bus_bytes`: the BlockMover's beat, the L3 tile's (64).
inline SystemPlan plan_system(const sw::kpu::program::csp::CspProgram& p, const sw::kpu::program::TileProgram& inputs,
                              const sw::kpu::program::TileProgram& result, std::uint32_t bus_bytes = kL3BeatBytes) {
    using Kind = sw::kpu::program::csp::Action::Kind;
    if (bus_bytes != kL3BeatBytes)
        throw std::invalid_argument("plan_system: the BlockMover's beat is the L3 tile's, " + std::to_string(kL3BeatBytes) +
                                    " bytes, not " + std::to_string(bus_bytes));
    SystemPlan sp;
    sp.memory = plan_l3(p, inputs, result);
    CspComputeOptions o;
    o.row_align = bus_bytes;
    o.preloaded = false;
    o.l3_slot_bytes = sp.memory.slot_bytes;
    o.l3_credits = true;
    sp.compute = plan_compute(p, inputs, result, o);

    std::map<std::tuple<std::string, sw::kpu::program::Dim, sw::kpu::program::Dim>, const CspL3Plan::Tile*> dram;
    for (const auto& t : sp.memory.inputs) dram[{t.tile.operand, t.tile.ti, t.tile.tj}] = &t;
    auto tile_of = [&](std::size_t i) -> const CspL3Plan::Tile& {
        const auto& a = p.actions[i];
        const auto it = dram.find({a.tile.operand, a.tile.ti, a.tile.tj});
        if (it == dram.end())
            throw std::invalid_argument("plan_system: action " + std::to_string(i) + " moves " + a.tile.to_string() +
                                        ", which has no DRAM tile");
        return *it->second;
    };
    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        const auto& a = p.actions[i];
        const bool l3 = a.kind == Kind::Load || a.kind == Kind::Store || a.kind == Kind::Release ||
                        a.kind == Kind::Inherit || a.kind == Kind::Move || a.kind == Kind::Writeback;
        if (!l3) continue;
        if (!a.place) throw std::invalid_argument("plan_system: action " + std::to_string(i) + " has no L3 address");
        const std::uint32_t s = a.place->slot;
        switch (a.kind) {
            case Kind::Load: {
                const auto& t = tile_of(i);
                sp.l3.push_back(l3_load(s, t.address, t.bytes));
                ++sp.loads;
                break;
            }
            case Kind::Store: {
                const auto& t = tile_of(i);
                sp.l3.push_back(l3_store(s, t.address, t.bytes));
                ++sp.stores;
                break;
            }
            case Kind::Release:
                sp.l3.push_back(l3_release(s));
                ++sp.releases;
                break;
            case Kind::Inherit:
                sp.l3.push_back(l3_fill(s, tile_of(i).data));
                ++sp.fills;
                break;
            case Kind::Move:
                sp.l3.push_back(l3_read(s, tile_of(i).bytes));
                ++sp.reads;
                break;
            case Kind::Writeback:
                sp.l3.push_back(l3_write(s, tile_of(i).bytes));
                ++sp.writes;
                break;
            default: break;
        }
    }
    if (sp.reads != sp.compute.moves || sp.writes != sp.compute.writebacks)
        throw std::logic_error("plan_system: " + std::to_string(sp.reads) + " L3 Reads for " +
                               std::to_string(sp.compute.moves) + " Moves, " + std::to_string(sp.writes) +
                               " Writes for " + std::to_string(sp.compute.writebacks) + " Writebacks");
    return sp;
}

struct SystemOptions {
    double kpu_ghz = 1.0, l2_ghz = 0.25;
    std::size_t window = 16;                 // the DMA engine's reads (and writes) in flight
    std::size_t store_buffer_beats = 4;
    std::uint32_t l2_banks = 4, read_latency = 4;
    std::uint32_t read_delay = 40;           // the Streamer's D
    std::uint64_t deadlock_window = 10'000;
    Picoseconds limit = 1'000'000'000'000;   // 1 s
    Tracer* tracer = nullptr;
    std::string counters;                    // write_counters_json's path ("" = none)
};

// The joined machine.
struct SystemMachine {
    Simulator sim;
    Domain& dram_d;
    Domain& kpu;
    Domain& l2d;
    Lpddr5Channel& channel;
    DmaEngine& dma;
    L3Tile& l3;
    ComputeParts parts;
    BlockMover& bm;
    L2& l2;
    Streamer& st;
    PEArray& array;
    Combiner& combiner;

    SystemMachine(const SystemPlan& plan, const PEArray::Config& shape, const SystemOptions& o)
        : dram_d(traced(sim, o.tracer).add_domain(Domain::from_ghz("dram", 3.2))),
          kpu(sim.add_domain(Domain::from_ghz("kpu", o.kpu_ghz))),
          l2d(sim.add_domain(Domain::from_ghz("l2", o.l2_ghz))),
          channel(sim.add<Lpddr5Channel>("lpddr5", dram_d, lpddr5_one_channel())),
          dma(sim.add<DmaEngine>("dma", kpu,
                                 DmaEngine::Config{plan.memory.slots, plan.memory.slot_bytes, o.window,
                                                   o.store_buffer_beats, o.window},
                                 plan.l3)),
          l3(sim.add<L3Tile>("l3", kpu, L3Tile::Config{plan.memory.slots, plan.memory.slot_bytes, plan.memory.banks},
                             plan.l3)),
          parts(build_compute(sim, kpu, l2d, l3, plan.compute.l2, plan.compute.commands, compute_options(plan, shape, o))),
          bm(*parts.bm),
          l2(*parts.l2),
          st(*parts.st),
          array(*parts.array),
          combiner(*parts.combiner) {
        auto& req = sim.crossing<DramRequest>("dma -> lpddr5", dma, channel, 4);
        auto& rsp = sim.crossing<DramResponse>("lpddr5 -> dma", channel, dma, 4);
        auto& loads = sim.link<L3Beat>("dma -> l3", dma, l3, 2);
        auto& ejects = sim.link<L3Beat>("l3 -> dma", l3, dma, 2);
        auto& credits = sim.link<SlotCredit>("l3 credits", l3, dma, plan.memory.slots);
        auto& bm_credits = sim.link<SlotCredit>("l3 credits to bm", l3, bm, plan.memory.slots);
        channel.connect(req, rsp);
        dma.connect(req, rsp, loads, ejects, credits);
        l3.connect(loads, ejects, credits);
        l3.connect_blockmover(*parts.l3_reads, *parts.l3_data, *parts.l3_writes, bm_credits);
        std::vector<bool> held(plan.memory.slots);
        for (std::uint32_t s = 0; s < plan.memory.slots; ++s) held[s] = blockmover_holds_at_start(plan.l3, s);
        bm.connect_l3_credits(bm_credits, held);
        sim.add_invariant("l3 slot credits", [this] { check_slot_credits(dma, bm, l3); });
        sim.set_deadlock_window(o.deadlock_window);
    }

private:
    static Simulator& traced(Simulator& s, Tracer* t) {
        if (t) s.attach(*t);
        return s;
    }
    static ComputeOptions compute_options(const SystemPlan& plan, const PEArray::Config& shape, const SystemOptions& o) {
        ComputeOptions c;
        c.l2 = L2::Config{plan.compute.l2_slots, plan.compute.rows_per_slot, o.l2_banks, plan.compute.row_bytes,
                          o.read_latency, 2, 1};
        c.lanes = shape.rows;
        c.cols = shape.cols;
        c.log = false;
        c.read_delay = o.read_delay;
        c.bus_bytes = kL3BeatBytes;
        c.kpu_ghz = o.kpu_ghz;
        c.l2_ghz = o.l2_ghz;
        return c;
    }
};

struct SystemRun {
    MemoryRun memory;     // its run, data and laws; the DMA's, the channel's and the L3 tile's counts
    ComputeRun compute;   // the array's, the L2 banks', the Streamer's and the Combiner's
    [[nodiscard]] bool ok() const { return memory.ok(); }
};

inline SystemRun run_system(const SystemPlan& plan, const program::platform::DeviceSpecification& device,
                            const SystemOptions& o) {
    SystemRun out;
    MemoryRun& m = out.memory;
    ComputeRun& c = out.compute;
    const PEArray::Config shape = PEArray::Config::from(device);
    if (device.noc)
        throw std::invalid_argument("run_system: " + device.name +
                                    " has a NoC; the join runs one L3 tile on a direct link (Phase 1's single-L3 devices)");
    if (plan.compute.max_h > shape.rows || plan.compute.max_w > shape.cols)
        throw std::invalid_argument("run_system: the program's tiles are " + std::to_string(plan.compute.max_h) + " x " +
                                    std::to_string(plan.compute.max_w) + ", past " + device.name + "'s " +
                                    std::to_string(shape.rows) + " x " + std::to_string(shape.cols) + " array");
    SystemMachine sys(plan, shape, o);
    for (const auto& t : plan.memory.inputs) sys.channel.preload(t.address, t.data);
    try {
        m.finished = sys.sim.run_until_idle(o.limit);
    } catch (const Deadlock& e) {
        m.deadlock = std::string(e.what()) + "\n" + e.dump();
    } catch (const InvariantViolation& e) {
        m.violation = std::string(e.what()) + "\n" + e.dump();
    }
    m.end = sys.sim.now();
    if (m.finished)
        for (const auto& t : plan.memory.outputs) {
            if (sys.channel.peek(t.address, t.bytes) != t.data) m.data_errors.push_back(t.tile.to_string() + " in DRAM is not L-B's");
            else ++m.tiles_checked;
        }
    m.rows = report(sys.sim);
    if (!o.counters.empty()) write_counters_json(sys.sim, o.counters);
    m.law_errors = check_laws(m.rows, m.finished);
    m.dma = sys.dma.dma();
    m.dram = sys.channel.dram();
    m.l3_max_live = sys.l3.counts().max_occupancy;
    m.l3_mean_live = sys.l3.counts().mean_occupancy();

    c.finished = m.finished;
    c.end = m.end;
    c.kpu_cycles = sys.kpu.cycle();
    c.l2_cycles = sys.l2d.cycle();
    c.rows = m.rows;
    c.rows_pe = shape.rows;
    c.cols_pe = shape.cols;
    for (std::uint32_t i = 0; i < shape.rows; ++i)
        for (std::uint32_t j = 0; j < shape.cols; ++j) c.pes.push_back(sys.array.pe_cycles(i, j));
    c.pe_total = sys.array.pe_cycles_total();
    for (const ReportRow& r : m.rows)
        if (!r.link && r.name == sys.array.name()) c.array_cycles = r.c.cycles;
    for (std::size_t k = 0; k < c.pes.size(); ++k)
        if (c.pes[k].total() != c.array_cycles) {
            m.law_errors.push_back("PE(" + std::to_string(k / shape.cols) + "," + std::to_string(k % shape.cols) +
                                   ")'s clocks by cause add to " + std::to_string(c.pes[k].total()) + ", not T " +
                                   std::to_string(c.array_cycles));
            break;
        }
    c.banks = sys.l2.banks();
    c.read_ports = 2;
    c.write_ports = 1;
    for (std::uint32_t j = 0; j < shape.cols; ++j) c.fifo_max.push_back(sys.combiner.fifo_max(j));
    c.max_round_trip = sys.st.max_round_trip();
    c.commands = plan.compute.commands.size();
    c.results = sys.combiner.records().size();
    return out;
}

inline std::string format_system_summary(const SystemPlan& plan, const SystemRun& r) {
    const MemoryRun& m = r.memory;
    std::string s;
    char b[512];
    std::snprintf(b, sizeof b,
                  "system: %zu Loads, %zu Stores, %zu Releases, %zu Fills; %zu Reads and %zu Writes by the BlockMover; "
                  "%zu commands (%zu result tiles) on a %u x %u array; %u L3 slots of %u bytes, %u L2 slots\n",
                  plan.loads, plan.stores, plan.releases, plan.fills, plan.reads, plan.writes, plan.compute.calls,
                  plan.compute.drains, r.compute.rows_pe, r.compute.cols_pe, plan.memory.slots, plan.memory.slot_bytes,
                  plan.compute.l2_slots);
    s += b;
    if (m.deadlock) s += "DEADLOCK: " + *m.deadlock + "\n";
    if (m.violation) s += "INVARIANT VIOLATED: " + *m.violation + "\n";
    if (!m.finished && !m.deadlock && !m.violation) s += "the run did not finish within its limit\n";
    std::snprintf(b, sizeof b,
                  "  run          %.3f us, %llu KPU clocks; data: %zu of %zu stored tiles in DRAM bit for bit, L-B's; laws: %s\n",
                  double(m.end) / 1e6, (unsigned long long)r.compute.kpu_cycles, m.tiles_checked, plan.memory.outputs.size(),
                  m.law_errors.empty() ? "hold on every row" : "VIOLATED");
    s += b;
    for (const auto& e : m.data_errors) s += "  DATA: " + e + "\n";
    for (const auto& e : m.law_errors) s += "  LAW: " + e + "\n";
    s += format_memory_lines(m, plan.memory.slots);
    s += format_compute_lines(r.compute);
    s += format_report(m.rows);
    return s;
}

}  // namespace sw::kpu::cycle
