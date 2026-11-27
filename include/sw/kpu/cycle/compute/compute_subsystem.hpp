#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/compute_subsystem.hpp
// The compute subsystem on the cycle core, run on a CSP program's compute traffic (#547;
// docs/plans/cycle-based-kpu-prototype.md §5.4, step 2f): what kpu-run --compute-subsystem runs.
//
// THE MACHINE (ComputeMachine): L3 -> BlockMover -> L2 -> Streamer -> L1 lanes -> PE array -> north
// FIFOs -> Combiner -> L2 -> BlockMover -> L3. The BlockMover, the Streamer, the array and the
// Combiner are in the KPU's domain (1 GHz), the L2 in its own (250 MHz), a crossing each way between
// them; the L3 is the stand-in, loaded with the program's A and B (l3_standin.hpp). The array is the
// deployment's R x C. Its programs are a CspComputePlan (csp_compute_program.hpp).
//
// WHAT A RUN CHECKS:
//   data        every written-back tile in L3 is L-B's, int32 exact
//   invariants  every cycle: the three-way slot credits (check_l2_slot_credits), each component's
//               and link's own checks (the L1 credit, the L2's slot states and bank ports, the
//               Combiner's FIFOs); a north collision, a late row or a slot fault stops the run as
//               a violation, with the state dump
//   the laws    on every component's and link's counters (check_laws, report.hpp); and the PEs'
//               clocks by cause add up to R x C x the array's clocks
//   deadlock    the detector's report, if no progress is made
//
// THE COUNTERS SUMMARY (format_compute_summary): array utilization (useful MACs over R x C x T) and
// the PE-clocks by cause (useful, padded, draining, waiting on L1); each L2 bank's port utilization
// and its reads in flight, mean and peak (#495's question, measured); the Streamer's and the
// BlockMover's busy clocks and stalls by cause; the Combiner's tiles in flight and its FIFOs' peaks.
//
// THE TRACE (a Tracer in the options): a track per component and link, each domain's clock, and
// lanes sampled every KPU clock: each L1 lane buffer's block ("cmd n" while it streams), each PE
// row's west PE ("cmd n" while it multiplies), each north FIFO's occupancy and each L2 bank's reads
// in flight (counter tracks), and each L2 slot's tile ("filling T", then T, "T read" while read).
//
// THE HEATMAP (write_heatmap_json, write_heatmap_svg): each PE's clocks by cause; the SVG shades each
// PE by its useful fraction.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/combiner.hpp>
#include <sw/kpu/cycle/compute/csp_compute_program.hpp>
#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/compute/l3_standin.hpp>
#include <sw/kpu/cycle/compute/pe_array.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/trace.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::cycle {

// The whole path, wired (step 2d's rig, #545, made the library's: kpu-run and the tests build the
// same machine).
struct ComputeMachine {
    struct Options {
        L2::Config l2{6, 16, 4, 16, 4, 2, 1};   // 16-byte rows: K up to 16 int8 a command, 4 int32 a result row
        std::uint32_t lanes = 4;                 // R: the array's rows
        std::uint32_t cols = 0;                  // C: its columns (0: R)
        std::uint32_t fifo_depth = 0;            // the north FIFOs' (0: C)
        bool log = true;                         // the array's MAC and launch log
        std::uint32_t read_delay = 32;
        std::int32_t drain_adjust = 0;
        std::uint32_t bus_bytes = 16;
        double kpu_ghz = 1.0;
        double l2_ghz = 0.25;
        std::size_t streamer_link = 32;
        std::size_t result_link = 16;           // the Combiner -> L2 crossing's entries
        Tracer* tracer = nullptr;
    };

    std::uint32_t cols;
    Simulator sim;
    Domain& kpu;
    Domain& l2d;
    L3Standin& l3;
    BlockMover& bm;
    L2& l2;
    Streamer& st;
    PEArray& array;
    Combiner& combiner;

    ComputeMachine(const L2Program& program, const StreamerProgram& commands, const Options& o)
        : cols(o.cols ? o.cols : o.lanes),
          kpu(traced(sim, o.tracer).add_domain(Domain::from_ghz("kpu", o.kpu_ghz))),
          l2d(sim.add_domain(Domain::from_ghz("l2", o.l2_ghz))),
          l3(sim.add<L3Standin>("l3", kpu, o.bus_bytes)),
          bm(sim.add<BlockMover>("bm", kpu,
                                 BlockMover::Config{o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, o.bus_bytes, 2, 8},
                                 program)),
          l2(sim.add<L2>("l2", l2d, o.l2, program)),
          st(sim.add<Streamer>("st", kpu,
                               Streamer::Config{o.lanes, cols, o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, 1,
                                                o.read_delay, 0, o.drain_adjust},
                               commands, program)),
          array(sim.add<PEArray>("array", kpu, PEArray::Config{o.lanes, cols, 1, o.log})),
          combiner(sim.add<Combiner>("combiner", kpu,
                                     Combiner::Config{cols, o.fifo_depth ? o.fifo_depth : cols, o.l2.row_bytes},
                                     commands)) {
        check_feeds(commands, program);
        auto& l3_reads = sim.link<L3RowRead>("bm->l3 reads", bm, l3, 2);
        auto& l3_data = sim.link<Beat>("l3->bm", l3, bm, 2);
        auto& l3_writes = sim.link<Beat>("bm->l3 writes", bm, l3, 2);
        auto& l2_writes = sim.crossing<L2RowWrite>("bm->l2 writes", bm, l2, 4);
        auto& l2_reads = sim.crossing<L2RowRead>("bm->l2 reads", bm, l2, 4);
        auto& l2_data = sim.crossing<L2RowData>("l2->bm", l2, bm, 4);
        auto& credits = sim.crossing<SlotCredit>("l2 credits", l2, bm, o.l2.slots, 2, 2);
        auto& st_reads = sim.crossing<L2RowRead>("st->l2 reads", st, l2, o.streamer_link);
        auto& st_data = sim.crossing<L2RowData>("l2->st", l2, st, o.streamer_link);
        auto& ready = sim.crossing<SlotReady>("l2 tile-ready", l2, st, 4);
        auto& st_credits = sim.crossing<SlotCredit>("l2 result credits", l2, st, o.l2.slots, 2, 2);
        auto& results = sim.crossing<L2RowWrite>("combiner->l2", combiner, l2, o.result_link);
        std::vector<Link<L1Element>*> a, b;
        std::vector<Link<ArrayResult>*> c;
        for (std::uint32_t m = 0; m < o.lanes; ++m)
            a.push_back(&sim.link<L1Element>("A lane " + std::to_string(m), st, array, 2));
        for (std::uint32_t m = 0; m < cols; ++m) {
            b.push_back(&sim.link<L1Element>("B lane " + std::to_string(m), st, array, 2));
            c.push_back(&sim.link<ArrayResult>("north " + std::to_string(m), array, combiner, 2));
        }
        l3.connect(l3_reads, l3_data, l3_writes);
        bm.connect(l3_reads, l3_data, l3_writes, l2_writes, l2_reads, l2_data, credits);
        l2.connect(l2_writes, l2_reads, l2_data, credits);
        l2.connect_streamer(st_reads, st_data, ready, &st_credits);
        l2.connect_combiner(results);
        st.connect(st_reads, st_data, ready, a, b, &st_credits);
        array.connect(a, b, c);
        combiner.connect(c, results);
        sim.add_invariant("l2 slot credits", [this] { check_l2_slot_credits(bm, st, l2); });
    }

private:
    static Simulator& traced(Simulator& s, Tracer* t) {
        if (t) s.attach(*t);
        return s;
    }
};

struct ComputeRunOptions {
    double kpu_ghz = 1.0, l2_ghz = 0.25;
    std::uint32_t banks = 4;                 // the L2's, each with 2 read ports and 1 write port
    std::uint32_t read_latency = 4;          // L2 clocks
    std::uint32_t read_delay = 40;           // the Streamer's D, in KPU clocks
    std::uint32_t bus_bytes = 64;            // the BlockMover's beat (step 2a's default); the L2's row is whole beats of it
    std::uint64_t deadlock_window = 10'000;  // edges without progress
    Picoseconds limit = 1'000'000'000'000;   // 1 s of simulated time
    Tracer* tracer = nullptr;
    std::string counters;                    // where to write the counters (write_counters_json; "" = none)
};

struct ComputeRun {
    bool finished = false;
    std::optional<std::string> deadlock;     // the detector's report: its message and the state dump
    std::optional<std::string> violation;    // an invariant that failed, with the state dump
    Picoseconds end = 0;
    Cycle kpu_cycles = 0, l2_cycles = 0, array_cycles = 0;
    std::vector<std::string> data_errors, law_errors;
    std::size_t tiles_checked = 0;
    std::vector<ReportRow> rows;
    std::uint32_t rows_pe = 0, cols_pe = 0;              // R, C
    std::vector<PEArray::PeCycles> pes;                  // row-major, R x C
    PEArray::PeCycles pe_total;
    std::vector<L2::BankStats> banks;
    std::uint32_t read_ports = 0, write_ports = 0;
    std::vector<std::size_t> fifo_max;                   // each north FIFO's peak
    Cycle max_round_trip = 0;
    std::size_t commands = 0, results = 0;
    [[nodiscard]] bool ok() const {
        return finished && !deadlock && !violation && data_errors.empty() && law_errors.empty();
    }
    [[nodiscard]] double utilization() const {
        const double all = double(rows_pe) * cols_pe * double(array_cycles);
        return all > 0 ? double(pe_total.useful) / all : 0.0;
    }
};

namespace compute_detail {

// A lane of spans laid down from a value sampled each clock: a span per run of one label.
struct SampledLane {
    std::uint32_t lane = 0;
    std::string label;   // "" = nothing drawn
    Cycle from = 0;
    void at(Tracer& t, const Domain& d, Cycle now, const std::string& l) {
        if (l == label) return;
        if (!label.empty()) t.span(lane, label, d.edge(from), d.edge(now));
        label = l;
        from = now;
    }
};

inline std::string slot_label(const L2& l2, std::uint32_t s) {
    using State = L2::State;
    switch (l2.state(s)) {
        case State::Free: return "";
        case State::Owned: return "filling " + l2.tile(s).to_string();
        case State::Ready: return l2.tile(s).to_string();
        case State::Reading: return l2.tile(s).to_string() + " read";
    }
    return "";
}

}  // namespace compute_detail

inline ComputeRun run_compute_subsystem(const CspComputePlan& plan, const program::platform::DeviceSpecification& device,
                                        const ComputeRunOptions& o) {
    using namespace compute_detail;
    ComputeRun out;
    const PEArray::Config shape = PEArray::Config::from(device);
    if (plan.max_h > shape.rows || plan.max_w > shape.cols)
        throw std::invalid_argument("run_compute_subsystem: the program's tiles are " + std::to_string(plan.max_h) + " x " +
                                    std::to_string(plan.max_w) + ", past " + device.name + "'s " +
                                    std::to_string(shape.rows) + " x " + std::to_string(shape.cols) + " array");
    if (o.bus_bytes == 0) throw std::invalid_argument("run_compute_subsystem: the BlockMover's beat must be positive");
    if (plan.row_bytes % o.bus_bytes != 0)
        throw std::invalid_argument("run_compute_subsystem: the L2's " + std::to_string(plan.row_bytes) +
                                    "-byte row is not whole " + std::to_string(o.bus_bytes) + "-byte beats");
    ComputeMachine::Options mo;
    mo.l2 = L2::Config{plan.l2_slots, plan.rows_per_slot, o.banks, plan.row_bytes, o.read_latency, 2, 1};
    mo.lanes = shape.rows;
    mo.cols = shape.cols;
    mo.log = false;
    mo.read_delay = o.read_delay;
    mo.bus_bytes = o.bus_bytes;
    mo.kpu_ghz = o.kpu_ghz;
    mo.l2_ghz = o.l2_ghz;
    mo.tracer = o.tracer;
    ComputeMachine m(plan.l2, plan.commands, mo);
    m.sim.set_deadlock_window(o.deadlock_window);
    for (const auto& t : plan.inputs) m.l3.load(t.address, t.bytes);

    // The lanes, sampled every KPU clock.
    std::vector<SampledLane> a_lanes, b_lanes, pe_rows, slots;
    std::vector<std::uint32_t> fifo_tracks, bank_tracks;
    Cycle sampled = ~Cycle{0};
    if (o.tracer) {
        Tracer& t = *o.tracer;
        for (std::uint32_t i = 0; i < shape.rows; ++i) a_lanes.push_back({t.add_lane("L1 A lane " + std::to_string(i), m.kpu.name()), "", 0});
        for (std::uint32_t j = 0; j < shape.cols; ++j) b_lanes.push_back({t.add_lane("L1 B lane " + std::to_string(j), m.kpu.name()), "", 0});
        for (std::uint32_t i = 0; i < shape.rows; ++i) pe_rows.push_back({t.add_lane("PE row " + std::to_string(i), m.kpu.name()), "", 0});
        for (std::uint32_t j = 0; j < shape.cols; ++j) fifo_tracks.push_back(t.add_lane("north FIFO " + std::to_string(j), m.kpu.name()));
        for (std::uint32_t b = 0; b < o.banks; ++b) bank_tracks.push_back(t.add_lane("L2 bank " + std::to_string(b) + " reads in flight", m.l2d.name()));
        for (std::uint32_t s = 0; s < plan.l2_slots; ++s) slots.push_back({t.add_lane("L2 slot " + std::to_string(s), m.kpu.name()), "", 0});
        m.sim.add_invariant("trace lanes", [&] {
            const Cycle now = m.kpu.cycle();
            if (now == sampled) return;
            sampled = now;
            const Picoseconds at = m.kpu.edge(now);
            for (std::uint32_t i = 0; i < shape.rows; ++i)
                a_lanes[i].at(t, m.kpu, now, m.st.a_lanes().left(i) ? "cmd " + std::to_string(m.st.a_lanes().command(i)) : "");
            for (std::uint32_t j = 0; j < shape.cols; ++j)
                b_lanes[j].at(t, m.kpu, now, m.st.b_lanes().left(j) ? "cmd " + std::to_string(m.st.b_lanes().command(j)) : "");
            for (std::uint32_t i = 0; i < shape.rows; ++i) {
                const auto c = m.array.west_command(i);
                pe_rows[i].at(t, m.kpu, now, c ? "cmd " + std::to_string(*c) : "");
            }
            for (std::uint32_t j = 0; j < shape.cols; ++j) t.occupancy(fifo_tracks[j], at, m.combiner.fifo(j));
            for (std::uint32_t b = 0; b < o.banks; ++b) t.occupancy(bank_tracks[b], at, m.l2.in_flight(b));
            for (std::uint32_t s = 0; s < plan.l2_slots; ++s) slots[s].at(t, m.kpu, now, slot_label(m.l2, s));
        });
    }

    try {
        out.finished = m.sim.run_until_idle(o.limit);
    } catch (const Deadlock& e) {
        out.deadlock = std::string(e.what()) + "\n" + e.dump();
    } catch (const InvariantViolation& e) {
        out.violation = std::string(e.what()) + "\n" + e.dump();
    }
    out.end = m.sim.now();
    out.kpu_cycles = m.kpu.cycle();
    out.l2_cycles = m.l2d.cycle();
    if (o.tracer) {
        const Cycle last = m.kpu.cycle();
        for (auto* v : {&a_lanes, &b_lanes, &pe_rows, &slots})
            for (SampledLane& l : *v) l.at(*o.tracer, m.kpu, last, "");
    }

    if (out.finished)
        for (const auto& t : plan.outputs) {
            if (m.l3.peek(t.address, t.bytes.size()) != t.bytes)
                out.data_errors.push_back(t.tile.to_string() + " in L3 slot " + std::to_string(t.l3_slot) + " is not L-B's");
            else ++out.tiles_checked;
        }
    out.rows = report(m.sim);
    if (!o.counters.empty()) write_counters_json(m.sim, o.counters);
    out.law_errors = check_laws(out.rows, out.finished);

    out.rows_pe = shape.rows;
    out.cols_pe = shape.cols;
    for (std::uint32_t i = 0; i < shape.rows; ++i)
        for (std::uint32_t j = 0; j < shape.cols; ++j) out.pes.push_back(m.array.pe_cycles(i, j));
    out.pe_total = m.array.pe_cycles_total();
    for (const ReportRow& r : out.rows)
        if (!r.link && r.name == m.array.name()) out.array_cycles = r.c.cycles;
    for (std::size_t k = 0; k < out.pes.size(); ++k)
        if (out.pes[k].total() != out.array_cycles) {
            out.law_errors.push_back("PE(" + std::to_string(k / shape.cols) + "," + std::to_string(k % shape.cols) +
                                     ")'s clocks by cause add to " + std::to_string(out.pes[k].total()) + ", not T " +
                                     std::to_string(out.array_cycles));
            break;
        }
    out.banks = m.l2.banks();
    out.read_ports = mo.l2.read_ports;
    out.write_ports = mo.l2.write_ports;
    for (std::uint32_t j = 0; j < shape.cols; ++j) out.fifo_max.push_back(m.combiner.fifo_max(j));
    out.max_round_trip = m.st.max_round_trip();
    out.commands = plan.commands.size();
    out.results = m.combiner.records().size();
    return out;
}

// The summary kpu-run prints: what ran, the data and the laws, the array, the L2's banks, the
// Streamer, the BlockMover and the Combiner, then every component's and link's row.
inline std::string format_compute_summary(const CspComputePlan& plan, const ComputeRun& r) {
    std::string s;
    char b[512];
    auto row = [&](const std::string& name) -> const ReportRow* {
        for (const auto& x : r.rows)
            if (!x.link && x.name == name) return &x;
        return nullptr;
    };
    auto stalls = [](const Counters& c) {
        std::string t;
        for (std::size_t k = 0; k < kStallCauses; ++k)
            if (c.stalls[k]) t += std::string(t.empty() ? "" : ", ") + to_string(static_cast<Stall>(k)) + " " + std::to_string(c.stalls[k]);
        return t.empty() ? std::string("none") : t;
    };
    std::snprintf(b, sizeof b,
                  "compute subsystem: %zu Moves, %zu Feeds, %zu commands (%zu result tiles), %zu Writebacks, %zu L2 "
                  "Releases; %u L2 slots of %u rows x %u bytes; a %u x %u array\n",
                  plan.moves, plan.feeds, plan.calls, plan.drains, plan.writebacks, plan.l2_releases, plan.l2_slots,
                  plan.rows_per_slot, plan.row_bytes, r.rows_pe, r.cols_pe);
    s += b;
    if (r.deadlock) s += "DEADLOCK: " + *r.deadlock + "\n";
    if (r.violation) s += "INVARIANT VIOLATED: " + *r.violation + "\n";
    if (!r.finished && !r.deadlock && !r.violation) s += "the run did not finish within its limit\n";
    std::snprintf(b, sizeof b, "  run          %.3f us, %llu KPU clocks; data: %zu of %zu result tiles exact; laws: %s\n",
                  double(r.end) / 1e6, (unsigned long long)r.kpu_cycles, r.tiles_checked, plan.outputs.size(),
                  r.law_errors.empty() ? "hold on every row" : "VIOLATED");
    s += b;
    for (const auto& e : r.data_errors) s += "  DATA: " + e + "\n";
    for (const auto& e : r.law_errors) s += "  LAW: " + e + "\n";
    const double pe_clocks = double(r.rows_pe) * r.cols_pe * double(r.array_cycles);
    auto pct = [&](std::uint64_t n) { return pe_clocks > 0 ? 100.0 * double(n) / pe_clocks : 0.0; };
    std::snprintf(b, sizeof b,
                  "  array        utilization %.2f%% (useful MACs of R x C x %llu clocks); PE-clocks: useful %.2f%%, "
                  "padded %.2f%%, draining %.2f%%, waiting on L1 %.2f%%\n",
                  100.0 * r.utilization(), (unsigned long long)r.array_cycles, pct(r.pe_total.useful),
                  pct(r.pe_total.padded), pct(r.pe_total.draining), pct(r.pe_total.waiting));
    s += b;
    for (std::size_t k = 0; k < r.banks.size(); ++k) {
        const L2::BankStats& x = r.banks[k];
        const double T = double(r.l2_cycles);
        std::snprintf(b, sizeof b,
                      "  L2 bank %-4zu reads %llu (ports %.1f%%), writes %llu (port %.1f%%); reads in flight: mean %.2f, "
                      "at most %llu; port waits %llu\n",
                      k, (unsigned long long)x.reads, T > 0 ? 100.0 * double(x.reads) / (T * r.read_ports) : 0.0,
                      (unsigned long long)x.writes, T > 0 ? 100.0 * double(x.writes) / (T * r.write_ports) : 0.0,
                      T > 0 ? double(x.in_flight_integral) / T : 0.0, (unsigned long long)x.max_in_flight,
                      (unsigned long long)x.stalls);
        s += b;
    }
    if (const ReportRow* d = row("st")) {
        std::snprintf(b, sizeof b,
                      "  Streamer     %zu commands, busy %llu of %llu clocks (U %.3f); round trip at most %llu clocks; "
                      "stalls: %s\n",
                      r.commands, (unsigned long long)d->c.busy, (unsigned long long)d->c.cycles, d->c.utilization(),
                      (unsigned long long)r.max_round_trip, stalls(d->c).c_str());
        s += b;
    }
    if (const ReportRow* d = row("bm")) {
        std::snprintf(b, sizeof b, "  BlockMover   busy %llu of %llu clocks (U %.3f); stalls: %s\n",
                      (unsigned long long)d->c.busy, (unsigned long long)d->c.cycles, d->c.utilization(),
                      stalls(d->c).c_str());
        s += b;
    }
    if (const ReportRow* d = row("combiner")) {
        std::size_t fmax = 0;
        for (const std::size_t f : r.fifo_max) fmax = std::max(fmax, f);
        std::snprintf(b, sizeof b,
                      "  Combiner     %zu result tiles; tiles in flight: mean %.3f, at most %llu; a tile's mean residence "
                      "%.1f clocks; the deepest north FIFO held %zu\n",
                      r.results, d->c.mean_occupancy(), (unsigned long long)d->c.max_occupancy, d->c.mean_residence(),
                      fmax);
        s += b;
    }
    s += format_report(r.rows);
    return s;
}

// The heatmap's data: each PE's clocks by cause, row-major.
inline void write_heatmap_json(const ComputeRun& r, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_heatmap_json: cannot write " + path);
    out << "{\"format\":\"kpu-array-heatmap\",\"version\":1,\"rows\":" << r.rows_pe << ",\"cols\":" << r.cols_pe
        << ",\"cycles\":" << r.array_cycles << ",\"utilization\":" << r.utilization();
    for (const char* field : {"useful", "padded", "draining", "waiting"}) {
        out << ",\"" << field << "\":[";
        for (std::uint32_t i = 0; i < r.rows_pe; ++i) {
            out << (i ? "," : "") << "[";
            for (std::uint32_t j = 0; j < r.cols_pe; ++j) {
                const PEArray::PeCycles& c = r.pes[std::size_t(i) * r.cols_pe + j];
                const std::uint64_t v = field[0] == 'u' ? c.useful : field[0] == 'p' ? c.padded
                                      : field[0] == 'd' ? c.draining : c.waiting;
                out << (j ? "," : "") << v;
            }
            out << "]";
        }
        out << "]";
    }
    out << "}\n";
    out.close();   // a failure flushing the last bytes is the write's failure too
    if (!out) throw std::runtime_error("write_heatmap_json: writing " + path + " failed");
}

// The heatmap: a cell per PE, shaded by its useful fraction (white 0, dark blue 1), row 0 at the top,
// column 0 at the left (A enters from the west, B from the north).
inline void write_heatmap_svg(const ComputeRun& r, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_heatmap_svg: cannot write " + path);
    const unsigned cell = std::max(4u, std::min(24u, 768u / std::max(1u, std::max(r.rows_pe, r.cols_pe))));
    const unsigned w = r.cols_pe * cell, h = r.rows_pe * cell, top = 28;
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\"" << h + top
        << "\" font-family=\"sans-serif\" font-size=\"12\">\n<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";
    char line[256];
    std::snprintf(line, sizeof line, "<text x=\"2\" y=\"16\">array utilization %.2f%%: %u x %u PEs, %llu clocks</text>\n",
                  100.0 * r.utilization(), r.rows_pe, r.cols_pe, (unsigned long long)r.array_cycles);
    out << line;
    for (std::uint32_t i = 0; i < r.rows_pe; ++i)
        for (std::uint32_t j = 0; j < r.cols_pe; ++j) {
            const PEArray::PeCycles& c = r.pes[std::size_t(i) * r.cols_pe + j];
            const double u = r.array_cycles ? double(c.useful) / double(r.array_cycles) : 0.0;
            const int R = int(255 - u * (255 - 8)), G = int(255 - u * (255 - 48)), B = int(255 - u * (255 - 107));
            std::snprintf(line, sizeof line,
                          "<rect x=\"%u\" y=\"%u\" width=\"%u\" height=\"%u\" fill=\"rgb(%d,%d,%d)\"><title>PE(%u,%u): "
                          "useful %llu, padded %llu, draining %llu, waiting %llu</title></rect>\n",
                          j * cell, top + i * cell, cell, cell, R, G, B, i, j, (unsigned long long)c.useful,
                          (unsigned long long)c.padded, (unsigned long long)c.draining, (unsigned long long)c.waiting);
            out << line;
        }
    out << "</svg>\n";
    out.close();
    if (!out) throw std::runtime_error("write_heatmap_svg: writing " + path + " failed");
}

}  // namespace sw::kpu::cycle
