// ============================================================================
// src/program/tile_flow_record.cpp
// The tile-flow record and its .tflow bundle (#286 step 2). See the header for what v0 can
// say at L-T1 and what it says it cannot.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/record/tile_flow_record.hpp>
#include <sw/kpu/program/record/columnar.hpp>
#include <sw/kpu/program/record/tile_flow_lod.hpp>

#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/tile_transaction_executor.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace sw::kpu::program::record {

namespace {
using json = nlohmann::ordered_json;

// The largest cycle an f64 time column holds exactly (every integer up to 2^53 is exact).
using columnar::kMaxExactCycle;

std::string pooled_name(const std::string& dev, const char* kind) {
    return dev + "/" + kind + "[*]";
}

// The tile an op MOVES: a Feed's input, a Drain's output (or input when it names none).
const TileCoord* moved_tile(const TileOp& op) {
    if (op.kind == TileOpKind::Drain) {
        if (!op.inputs.empty()) return &op.inputs[0];
        if (!op.outputs.empty()) return &op.outputs[0];
        return nullptr;
    }
    if (!op.inputs.empty()) return &op.inputs[0];
    if (!op.outputs.empty()) return &op.outputs[0];
    return nullptr;
}

} // namespace

std::string Tile::key() const { return tile_key(TileCoord{operand, ti, tj}); }

std::uint32_t TileFlowRecord::station(const std::string& name) const {
    for (std::size_t i = 0; i < stations.size(); ++i)
        if (stations[i].name == name) return static_cast<std::uint32_t>(i);
    throw RecordError("record: no station named \"" + name + "\"");
}

// ============================================================================
// Building
// ============================================================================
TileFlowRecord build_record(const TileProgram& prog, const driver::RunOutcome& outcome,
                            const platform::DeploymentSpec& spec, const Placement& placement,
                            Dim device, std::uint64_t foreign_slots) {
    if (!outcome.has_timing)
        throw RecordError(std::string("record: level ") + driver::short_name(outcome.level) +
                          " models no time, so it has no intervals to record; run a level "
                          "that models resources (L-T1)");
    if (device >= spec.device_count())
        throw RecordError("record: device " + std::to_string(device) + " of " +
                          std::to_string(spec.device_count()));
    const auto& ops = prog.ops();
    if (outcome.timeline.size() != ops.size())
        throw RecordError("record: the outcome's timeline has " +
                          std::to_string(outcome.timeline.size()) + " ops, the program " +
                          std::to_string(ops.size()) + "; they are not the same run");

    const platform::DeviceSpecification& d = spec.device(device);
    const auto dev = spec.device_view(device);
    TileFlowRecord rec;
    rec.level = driver::short_name(outcome.level);
    rec.device = d.name;
    rec.device_label = dev.label();
    rec.deployment_digest = platform::deployment_digest(spec);
    rec.makespan = outcome.makespan;
    if (rec.makespan > kMaxExactCycle)
        throw RecordError("record: makespan " + std::to_string(rec.makespan) +
                          " exceeds 2^53 cycles, which the f64 time columns cannot hold exactly");

    // ---- stations: DRAM, pooled L3, unmodelled L2/L1/DMA buffers, one per compute tile
    rec.stations.push_back({d.name + "/dram", "dram", 0, false, true});
    rec.stations.push_back({pooled_name(d.name, "l3"), "l3", dev.l3_tiles, true, true});
    rec.stations.push_back({pooled_name(d.name, "l2"), "l2", 0, true, false});
    rec.stations.push_back({pooled_name(d.name, "l1"), "l1", 0, true, false});
    // Where an ejected block waits for its DMA engine to write it to DRAM (version 3).
    rec.stations.push_back({pooled_name(d.name, "dmabuf"), "dmabuf", 0, true, false});
    rec.unmodelled = {"l2", "l1", "dmabuf"};
    const Dim n_cf = std::max<Dim>(placement.compute_tiles(), 1);
    for (Dim c = 0; c < n_cf; ++c)
        rec.stations.push_back({d.name + "/cf[" + std::to_string(c) + "]", "cf", 1, false, true});
    const std::uint32_t DRAM = 0, L3 = 1, L2 = 2, L1 = 3, DMABUF = 4, CF0 = 5;
    if (outcome.stats)
        for (const auto& [m, lanes] : outcome.stats->mover_lanes)
            rec.movers.push_back({to_string(m), lanes});
    rec.element_bytes = d.element_bytes;
    for (const std::string& name : prog.operand_order()) {
        const TensorOperand& o = prog.operand(name);
        rec.operands.push_back({o.name, o.rows, o.cols, o.tile_rows, o.tile_cols});
    }

    // ---- tiles and ops, in first-appearance order (deterministic)
    std::map<std::string, std::uint32_t> tile_of;
    auto tile_id = [&](const TileCoord& c) {
        const std::string k = tile_key(c);
        const auto it = tile_of.find(k);
        if (it != tile_of.end()) return it->second;
        const auto id = static_cast<std::uint32_t>(rec.tiles.size());
        rec.tiles.push_back({c.operand, c.ti, c.tj});
        tile_of.emplace(k, id);
        return id;
    };
    for (const TileOp& op : ops) {
        Op o;
        o.kind = static_cast<std::uint8_t>(op.kind);
        std::map<std::uint32_t, std::size_t> at;          // tile -> position in o.tiles
        auto touch = [&](const TileCoord& c, bool writes) {
            const std::uint32_t id = tile_id(c);
            const auto it = at.find(id);
            if (it != at.end()) { if (writes) o.written[it->second] = 1; return; }
            at.emplace(id, o.tiles.size());
            o.tiles.push_back(id);
            o.written.push_back(writes ? 1 : 0);
        };
        for (const TileCoord& c : op.inputs) touch(c, false);
        for (const TileCoord& c : op.outputs) touch(c, true);
        rec.ops.push_back(std::move(o));
    }

    // ---- L3 residency, from the executor's series
    for (const L3Residency& r : outcome.l3_residency) {
        const auto it = tile_of.find(r.tile);
        if (it == tile_of.end())
            throw RecordError("record: residency names tile \"" + r.tile + "\", which the program never touches");
        Residency out{it->second, L3, r.start, r.finish, 0};
        if (r.seeded) out.flags |= Residency::Seeded;
        if (r.held) out.flags |= Residency::Held;
        rec.residency.push_back(out);
    }

    // ---- transits, one per hop, and computes, one per compute op
    auto endpoints = [&](Hop h) -> std::pair<std::uint32_t, std::uint32_t> {
        switch (h) {
            case Hop::DmaDramToL3:      return {DRAM, L3};
            case Hop::BlockMoverL3ToL2: return {L3, L2};
            case Hop::StreamerL2ToL1:   return {L2, L1};
            case Hop::StreamerL1ToL2:   return {L1, L2};
            case Hop::BlockMoverL2ToL3: return {L2, L3};
            case Hop::BlockMoverL3ToDmaBuffer: return {L3, DMABUF};
            case Hop::DmaBufferToDram:  return {DMABUF, DRAM};
            case Hop::BlockMoverL3ToL3: return {L3, L3};
        }
        return {L3, L3};
    };
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const TileOpRecord& t = outcome.timeline[i];
        if (t.resource == ResourceKind::ComputeTile && ops[i].kind != TileOpKind::Feed &&
            ops[i].kind != TileOpKind::Drain) {
            if (t.resource_id >= n_cf)
                throw RecordError("record: op " + std::to_string(i) + " ran on compute tile " +
                                  std::to_string(t.resource_id) + " of a " + std::to_string(n_cf) +
                                  "-tile placement");
            rec.computes.push_back({static_cast<std::uint32_t>(i), CF0 + t.resource_id, t.start, t.finish});
            continue;
        }
        const TileCoord* moved = moved_tile(ops[i]);
        if (!moved) continue;
        for (const HopRecord& h : t.hops) {
            const auto [src, dst] = endpoints(h.hop);
            rec.transits.push_back({tile_id(*moved), static_cast<std::uint32_t>(i),
                                    static_cast<std::uint8_t>(h.hop),
                                    static_cast<std::uint8_t>(mover_of(h.hop)), h.lane, h.start,
                                    h.finish, src, dst});
        }
    }
    std::sort(rec.transits.begin(), rec.transits.end(), [](const Transit& a, const Transit& b) {
        return a.t0 != b.t0 ? a.t0 < b.t0 : a.op != b.op ? a.op < b.op : a.hop < b.hop;
    });
    std::sort(rec.computes.begin(), rec.computes.end(), [](const Compute& a, const Compute& b) {
        return a.t0 != b.t0 ? a.t0 < b.t0 : a.op < b.op;
    });

    // The caller's slots this program cannot name (TileExecutionRequest::foreign_held_slots):
    // held for the whole run and counted in the executor's peak, so the record carries them
    // or its occupancy series would be low by exactly that much.
    rec.foreign_slots = foreign_slots;
    return rec;
}

// ============================================================================
// A CSP program's run at L-T1 (version 4)
// ============================================================================
TileFlowRecord build_csp_record(const csp::lang::Program& ast, const driver::CspLevelOutcome& outcome,
                                const TileProgram& operands, const platform::DeploymentSpec& spec, Dim device) {
    if (outcome.level != driver::ExecutionLevel::BlockSequential || outcome.skipped)
        throw RecordError(std::string("record: a CSP program's record is built from its L-T1 run; this is ") +
                          driver::short_name(outcome.level) + (outcome.skipped ? " (skipped)" : ""));
    if (device >= spec.device_count())
        throw RecordError("record: device " + std::to_string(device) + " of " + std::to_string(spec.device_count()));
    const platform::DeviceSpecification& d = spec.device(device);
    const auto dev = spec.device_view(device);
    TileFlowRecord rec;
    rec.op_space = "csp-actions";
    rec.level = driver::short_name(outcome.level);
    rec.device = d.name;
    rec.device_label = dev.label();
    rec.deployment_digest = platform::deployment_digest(spec);
    rec.makespan = outcome.makespan;
    if (rec.makespan > kMaxExactCycle)
        throw RecordError("record: makespan " + std::to_string(rec.makespan) +
                          " exceeds 2^53 cycles, which the f64 time columns cannot hold exactly");

    // ---- stations: as version 3, with L3's capacity the PROGRAM's (its credits)
    rec.stations.push_back({d.name + "/dram", "dram", 0, false, true});
    rec.stations.push_back({pooled_name(d.name, "l3"), "l3", static_cast<std::uint64_t>(ast.l3), true, true});
    rec.stations.push_back({pooled_name(d.name, "l2"), "l2", 0, true, false});
    rec.stations.push_back({pooled_name(d.name, "l1"), "l1", 0, true, false});
    rec.stations.push_back({pooled_name(d.name, "dmabuf"), "dmabuf", 0, true, false});
    rec.unmodelled = {"l2", "l1", "dmabuf"};
    const std::size_t n_cf = std::max<std::size_t>(outcome.compute_tiles, 1);
    for (std::size_t c = 0; c < n_cf; ++c)
        rec.stations.push_back({d.name + "/cf[" + std::to_string(c) + "]", "cf", 1, false, true});
    const std::uint32_t DRAM = 0, L3 = 1, L2 = 2, L1 = 3, DMABUF = 4, CF0 = 5;
    for (const auto& [proc, name] : {std::pair{"dma", "dma"}, std::pair{"bm", "block-mover"},
                                     std::pair{"str", "streamer"}})
        rec.movers.push_back({name, static_cast<Dim>(outcome.lanes.count(proc) ? outcome.lanes.at(proc) : 0)});
    rec.element_bytes = d.element_bytes;
    for (const std::string& name : operands.operand_order()) {
        const TensorOperand& o = operands.operand(name);
        rec.operands.push_back({o.name, o.rows, o.cols, o.tile_rows, o.tile_cols});
    }

    // ---- tiles and ops: one op per action, in program order (the program's stream)
    std::map<std::string, std::uint32_t> tile_of;
    auto tile_id = [&](const TileCoord& c) {
        const std::string k = tile_key(c);
        const auto it = tile_of.find(k);
        if (it != tile_of.end()) return it->second;
        const auto id = static_cast<std::uint32_t>(rec.tiles.size());
        rec.tiles.push_back({c.operand, c.ti, c.tj});
        tile_of.emplace(k, id);
        return id;
    };
    csp::lang::ActionStream stream(ast);
    while (auto e = stream.next()) {
        const csp::Action& a = e->action;
        Op o;
        o.kind = static_cast<std::uint8_t>(a.kind);
        std::map<std::uint32_t, std::size_t> at;
        auto touch = [&](const TileCoord& c, bool writes) {
            const std::uint32_t id = tile_id(c);
            const auto it = at.find(id);
            if (it != at.end()) { if (writes) o.written[it->second] = 1; return; }
            at.emplace(id, o.tiles.size());
            o.tiles.push_back(id);
            o.written.push_back(writes ? 1 : 0);
        };
        if (a.kind == csp::Action::Kind::Call && e->op) {
            for (const TileCoord& c : e->op->inputs) touch(c, false);
            for (const TileCoord& c : e->op->outputs) touch(c, true);
        } else {
            // A Load or a Writeback fills its L3 slot: it WRITES the tile there (TF9).
            touch(a.tile, a.kind == csp::Action::Kind::Load || a.kind == csp::Action::Kind::Writeback);
        }
        rec.ops.push_back(std::move(o));
    }

    // ---- residency: the program's slots, credit to Release
    for (const auto& sl : outcome.slots)
        rec.residency.push_back({tile_id(sl.tile), L3, sl.t0, sl.t1, 0});

    // ---- transits (each movement leg) and computes (each Call)
    using K = csp::Action::Kind;
    using P = csp::TransactionalInterpreter::Proc;
    for (const auto& r : outcome.records) {
        if (r.kind == K::Release) continue;
        const auto op = static_cast<std::uint32_t>(r.action);
        if (r.kind == K::Call) {
            if (r.lane >= n_cf)
                throw RecordError("record: action " + std::to_string(r.action) + " ran on compute tile " +
                                  std::to_string(r.lane) + " of " + std::to_string(n_cf));
            rec.computes.push_back({op, CF0 + static_cast<std::uint32_t>(r.lane), r.start, r.finish});
            continue;
        }
        Hop h = Hop::DmaDramToL3;
        std::uint32_t src = DRAM, dst = L3;
        switch (r.kind) {
            case K::Load:      h = Hop::DmaDramToL3;      src = DRAM; dst = L3; break;
            case K::Move:      h = Hop::BlockMoverL3ToL2; src = L3;   dst = L2; break;
            case K::Feed:      h = Hop::StreamerL2ToL1;   src = L2;   dst = L1; break;
            case K::Drain:     h = Hop::StreamerL1ToL2;   src = L1;   dst = L2; break;
            case K::Writeback: h = Hop::BlockMoverL2ToL3; src = L2;   dst = L3; break;
            case K::Store:
                if (r.proc == P::Bm) { h = Hop::BlockMoverL3ToDmaBuffer; src = L3; dst = DMABUF; }
                else                 { h = Hop::DmaBufferToDram;         src = DMABUF; dst = DRAM; }
                break;
            default: break;
        }
        rec.transits.push_back({tile_id(r.tile), op, static_cast<std::uint8_t>(h),
                                static_cast<std::uint8_t>(mover_of(h)), static_cast<std::uint32_t>(r.lane), r.start,
                                r.finish, src, dst});
    }
    std::sort(rec.transits.begin(), rec.transits.end(), [](const Transit& a, const Transit& b) {
        return a.t0 != b.t0 ? a.t0 < b.t0 : a.op != b.op ? a.op < b.op : a.hop < b.hop;
    });
    std::sort(rec.computes.begin(), rec.computes.end(), [](const Compute& a, const Compute& b) {
        return a.t0 != b.t0 ? a.t0 < b.t0 : a.op < b.op;
    });
    return rec;
}

// ============================================================================
// Occupancy
// ============================================================================
std::uint64_t l3_occupancy_at(const TileFlowRecord& rec, Cycle t) {
    std::uint64_t n = rec.foreign_slots;
    for (const Residency& r : rec.residency)
        if (rec.stations[r.station].kind == "l3" && r.t0 <= t && t < r.t1) ++n;
    return n;
}

std::uint64_t peak_l3_occupancy(const TileFlowRecord& rec) {
    // Half-open intervals: a slot is held from t0 up to, not including, t1. Ends sort before
    // starts at the same cycle, which is the executor's own order -- completions return their
    // credits before the next ops fire.
    std::vector<std::pair<Cycle, int>> ev;
    for (const Residency& r : rec.residency)
        if (rec.stations[r.station].kind == "l3" && r.t1 > r.t0) {
            ev.emplace_back(r.t0, +1);
            ev.emplace_back(r.t1, -1);
        }
    std::sort(ev.begin(), ev.end());
    std::int64_t cur = 0, best = 0;
    for (const auto& [t, d] : ev) {
        cur += d;
        best = std::max(best, cur);
    }
    return static_cast<std::uint64_t>(best) + rec.foreign_slots;
}

// ============================================================================
// The bundle
// ============================================================================
using namespace columnar;

void write_tflow(const TileFlowRecord& rec, const std::string& dir) {
    if (!little_endian()) throw RecordError("record: the .tflow writer assumes a little-endian host");
    // EVERY time, not just the makespan: a record built by hand, or edited, can carry an
    // endpoint past 2^53, and the f64 columns would round it silently.
    auto exact = [](Cycle c, const char* what) {
        if (c > kMaxExactCycle)
            throw RecordError(std::string("record: ") + what + " " + std::to_string(c) +
                              " exceeds 2^53 cycles, which an f64 time column cannot hold exactly");
    };
    exact(rec.makespan, "makespan");
    for (const Residency& r : rec.residency) { exact(r.t0, "residency t0"); exact(r.t1, "residency t1"); }
    for (const Transit& x : rec.transits) { exact(x.t0, "transit t0"); exact(x.t1, "transit t1"); }
    for (const Compute& c : rec.computes) { exact(c.t0, "compute t0"); exact(c.t1, "compute t1"); }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) throw RecordError("record: cannot create " + dir + ": " + ec.message());

    json m = json::object();
    m["format"] = "kpu-tflow";
    // 2: op_tiles gained `written` (#286 step 3). A version-1 bundle cannot answer "was this
    // slot filled", so it is refused with a message, not read as "nothing writes".
    // 3: every hop is a push (docs/plans/noc-port-arbitration.md step 3). The writeback leg
    // "DMA reads L3" became a BlockMover ejecting into a DMA buffer, then the DMA writing DRAM,
    // with a dmabuf station between them. A version-2 bundle's hop 5 means the old leg, so it
    // is refused rather than read as an ejection.
    // 4: a CSP program's record -- the columns are version 3's, an op is one of the program's
    // ACTIONS (kpu-run-csp-programs step 4b), and `ops` says so.
    m["version"] = rec.op_space == "csp-actions" ? 4 : 3;
    if (rec.op_space == "csp-actions") m["ops"] = "csp-actions";
    m["level"] = rec.level;
    m["device"] = rec.device;
    m["device_label"] = rec.device_label;
    m["deployment_digest"] = rec.deployment_digest;
    m["makespan"] = rec.makespan;
    m["foreign_slots"] = rec.foreign_slots;
    m["unmodelled"] = rec.unmodelled;
    json st = json::array();
    for (const Station& s : rec.stations)
        st.push_back(json{{"name", s.name}, {"kind", s.kind}, {"capacity", s.capacity},
                          {"pooled", s.pooled}, {"modelled", s.modelled}});
    m["stations"] = st;
    json mv = json::array();
    for (const MoverPool& p : rec.movers) mv.push_back(json{{"name", p.name}, {"lanes", p.lanes}});
    m["movers"] = mv;
    m["element_bytes"] = rec.element_bytes;
    json opd = json::array();
    for (const OperandShape& o : rec.operands)
        opd.push_back(json{{"name", o.name}, {"rows", o.rows}, {"cols", o.cols},
                           {"tile_rows", o.tile_rows}, {"tile_cols", o.tile_cols}});
    m["operands"] = opd;
    json tl = json::array();
    for (const Tile& t : rec.tiles) tl.push_back(json::array({t.operand, t.ti, t.tj}));
    m["tiles"] = tl;

    json tables = json::object();
    {
        Table t{"residency", "residency.bin", rec.residency.size(), {}};
        std::vector<std::uint32_t> tile, station;
        std::vector<Cycle> t0, t1;
        std::vector<std::uint8_t> flags;
        for (const Residency& r : rec.residency) {
            tile.push_back(r.tile); station.push_back(r.station);
            t0.push_back(r.t0); t1.push_back(r.t1); flags.push_back(r.flags);
        }
        t.put("tile", "u32", tile); t.put("station", "u32", station);
        t.put("t0", "f64", times(t0)); t.put("t1", "f64", times(t1)); t.put("flags", "u8", flags);
        tables["residency"] = write_table(t, dir);
    }
    {
        Table t{"transit", "transit.bin", rec.transits.size(), {}};
        std::vector<std::uint32_t> tile, op, lane, src, dst;
        std::vector<std::uint8_t> hop, mover;
        std::vector<Cycle> t0, t1;
        for (const Transit& x : rec.transits) {
            tile.push_back(x.tile); op.push_back(x.op); hop.push_back(x.hop); mover.push_back(x.mover);
            lane.push_back(x.lane); t0.push_back(x.t0); t1.push_back(x.t1); src.push_back(x.src); dst.push_back(x.dst);
        }
        t.put("tile", "u32", tile); t.put("op", "u32", op); t.put("hop", "u8", hop);
        t.put("mover", "u8", mover); t.put("lane", "u32", lane); t.put("t0", "f64", times(t0));
        t.put("t1", "f64", times(t1)); t.put("src", "u32", src); t.put("dst", "u32", dst);
        tables["transit"] = write_table(t, dir);
    }
    {
        Table t{"compute", "compute.bin", rec.computes.size(), {}};
        std::vector<std::uint32_t> op, station;
        std::vector<Cycle> t0, t1;
        for (const Compute& c : rec.computes) {
            op.push_back(c.op); station.push_back(c.station); t0.push_back(c.t0); t1.push_back(c.t1);
        }
        t.put("op", "u32", op); t.put("station", "u32", station);
        t.put("t0", "f64", times(t0)); t.put("t1", "f64", times(t1));
        tables["compute"] = write_table(t, dir);
    }
    {
        Table t{"op_tiles", "op_tiles.bin", rec.ops.size(), {}};
        std::vector<std::uint8_t> kind, written;
        std::vector<std::uint32_t> offset{0}, tile;
        for (const Op& o : rec.ops) {
            kind.push_back(o.kind);
            tile.insert(tile.end(), o.tiles.begin(), o.tiles.end());
            written.insert(written.end(), o.written.begin(), o.written.end());
            offset.push_back(static_cast<std::uint32_t>(tile.size()));
        }
        t.put("kind", "u8", kind); t.put("offset", "u32", offset); t.put("tile", "u32", tile);
        t.put("written", "u8", written);
        tables["op_tiles"] = write_table(t, dir);
    }
    m["tables"] = tables;

    std::ofstream out(dir + "/manifest.json", std::ios::binary);
    out << m.dump(1) << "\n";
    if (!out) throw RecordError("record: cannot write " + dir + "/manifest.json");

    // The pyramid is derived, so it is written beside the record rather than inside it: a
    // consumer that wants only the record never reads it, and the checker verifies it.
    write_lod(build_lod(rec), dir);
}


TileFlowRecord read_tflow(const std::string& dir) {
    json m;
    try {
        m = json::parse(slurp(dir + "/manifest.json"));
    } catch (const nlohmann::json::parse_error& e) {
        throw RecordError(std::string("record: manifest is not valid JSON: ") + e.what());
    }
    if (m.value("format", "") != "kpu-tflow") throw RecordError("record: not a kpu-tflow bundle");
    if (m.value("version", 0) == 1)
        throw RecordError("record: a version-1 bundle has no op_tiles.written column; re-record "
                          "it with this build's kpu-run --tflow");
    if (m.value("version", 0) == 2)
        throw RecordError("record: a version-2 bundle records writeback as a DMA read (hop 5 = "
                          "dma:l3->dram); re-record it with this build's kpu-run --tflow");
    const int version = m.value("version", 0);
    if (version != 3 && version != 4)
        throw RecordError("record: version " + std::to_string(version) + " is not one this build reads (3, 4)");
    TileFlowRecord rec;
    if (version == 4) {
        if (m.value("ops", "") != "csp-actions")
            throw RecordError("record: a version-4 bundle must say what its ops are (\"ops\": \"csp-actions\")");
        rec.op_space = "csp-actions";
    }
    rec.level = m.at("level").get<std::string>();
    rec.device = m.at("device").get<std::string>();
    rec.device_label = m.at("device_label").get<std::string>();
    rec.deployment_digest = m.at("deployment_digest").get<std::string>();
    rec.makespan = m.at("makespan").get<Cycle>();
    rec.foreign_slots = m.at("foreign_slots").get<std::uint64_t>();
    rec.unmodelled = m.at("unmodelled").get<std::vector<std::string>>();
    for (const json& s : m.at("stations"))
        rec.stations.push_back({s.at("name").get<std::string>(), s.at("kind").get<std::string>(),
                                s.at("capacity").get<std::uint64_t>(), s.at("pooled").get<bool>(),
                                s.at("modelled").get<bool>()});
    rec.element_bytes = m.value("element_bytes", Dim{0});
    if (m.contains("operands"))
        for (const json& o : m.at("operands"))
            rec.operands.push_back({o.at("name").get<std::string>(), o.at("rows").get<Dim>(),
                                    o.at("cols").get<Dim>(), o.at("tile_rows").get<Dim>(),
                                    o.at("tile_cols").get<Dim>()});
    if (m.contains("movers"))
        for (const json& p : m.at("movers"))
            rec.movers.push_back({p.at("name").get<std::string>(), p.at("lanes").get<Dim>()});
    for (const json& t : m.at("tiles"))
        rec.tiles.push_back({t.at(0).get<std::string>(), t.at(1).get<Dim>(), t.at(2).get<Dim>()});

    const json& tables = m.at("tables");
    auto blob_of = [&](const char* name) { return slurp(dir + "/" + tables.at(name).at("file").get<std::string>()); };
    // A time column read back must hold cycles: finite, non-negative, whole, and no larger
    // than the writer could have written exactly. Casting anything else would be undefined
    // behaviour or a silently different run.
    auto cycles = [](const std::vector<double>& v) {
        std::vector<Cycle> out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            const double t = v[i];
            if (!std::isfinite(t) || t < 0.0 || std::floor(t) != t ||
                t > static_cast<double>(kMaxExactCycle))
                throw RecordError("record: a time column holds " + std::to_string(t) +
                                  ", which is not a cycle count");
            out[i] = static_cast<Cycle>(t);
        }
        return out;
    };
    {
        const json& t = tables.at("residency");
        const std::string b = blob_of("residency");
        const std::size_t n = t.at("rows").get<std::size_t>();
        const auto tile = read_col<std::uint32_t>(t, b, "tile", "u32", n);
        const auto station = read_col<std::uint32_t>(t, b, "station", "u32", n);
        const auto t0 = cycles(read_col<double>(t, b, "t0", "f64", n));
        const auto t1 = cycles(read_col<double>(t, b, "t1", "f64", n));
        const auto flags = read_col<std::uint8_t>(t, b, "flags", "u8", n);
        for (std::size_t i = 0; i < n; ++i) rec.residency.push_back({tile[i], station[i], t0[i], t1[i], flags[i]});
    }
    {
        const json& t = tables.at("transit");
        const std::string b = blob_of("transit");
        const std::size_t n = t.at("rows").get<std::size_t>();
        const auto tile = read_col<std::uint32_t>(t, b, "tile", "u32", n);
        const auto op = read_col<std::uint32_t>(t, b, "op", "u32", n);
        const auto hop = read_col<std::uint8_t>(t, b, "hop", "u8", n);
        const auto mover = read_col<std::uint8_t>(t, b, "mover", "u8", n);
        const auto lane = read_col<std::uint32_t>(t, b, "lane", "u32", n);
        const auto t0 = cycles(read_col<double>(t, b, "t0", "f64", n));
        const auto t1 = cycles(read_col<double>(t, b, "t1", "f64", n));
        const auto src = read_col<std::uint32_t>(t, b, "src", "u32", n);
        const auto dst = read_col<std::uint32_t>(t, b, "dst", "u32", n);
        for (std::size_t i = 0; i < n; ++i)
            rec.transits.push_back({tile[i], op[i], hop[i], mover[i], lane[i], t0[i], t1[i], src[i], dst[i]});
    }
    {
        const json& t = tables.at("compute");
        const std::string b = blob_of("compute");
        const std::size_t n = t.at("rows").get<std::size_t>();
        const auto op = read_col<std::uint32_t>(t, b, "op", "u32", n);
        const auto station = read_col<std::uint32_t>(t, b, "station", "u32", n);
        const auto t0 = cycles(read_col<double>(t, b, "t0", "f64", n));
        const auto t1 = cycles(read_col<double>(t, b, "t1", "f64", n));
        for (std::size_t i = 0; i < n; ++i) rec.computes.push_back({op[i], station[i], t0[i], t1[i]});
    }
    {
        const json& t = tables.at("op_tiles");
        const std::string b = blob_of("op_tiles");
        const std::size_t n = t.at("rows").get<std::size_t>();
        const auto kind = read_col<std::uint8_t>(t, b, "kind", "u8", n);
        const auto offset = read_col<std::uint32_t>(t, b, "offset", "u32", n + 1);
        if (offset[0] != 0) throw RecordError("record: op_tiles offsets must start at 0");
        for (std::size_t i = 0; i < n; ++i)
            if (offset[i] > offset[i + 1])
                throw RecordError("record: op_tiles offset " + std::to_string(i + 1) +
                                  " decreases; the op tile table is corrupt");
        const auto tile = read_col<std::uint32_t>(t, b, "tile", "u32", n ? offset[n] : 0);
        const auto written = read_col<std::uint8_t>(t, b, "written", "u8", n ? offset[n] : 0);
        for (std::size_t i = 0; i < n; ++i)
            rec.ops.push_back({kind[i],
                               std::vector<std::uint32_t>(tile.begin() + offset[i], tile.begin() + offset[i + 1]),
                               std::vector<std::uint8_t>(written.begin() + offset[i], written.begin() + offset[i + 1])});
    }
    // Every index points into its table, so a consumer (the occupancy sweep, the viewer) can
    // index without checking -- the checks are here, once, at the trust boundary.
    auto in = [](std::uint32_t i, std::size_t n, const char* what) {
        if (i >= n)
            throw RecordError(std::string("record: a ") + what + " index " + std::to_string(i) +
                              " is out of range (" + std::to_string(n) + ")");
    };
    for (const Residency& r : rec.residency) { in(r.tile, rec.tiles.size(), "tile"); in(r.station, rec.stations.size(), "station"); }
    for (const Transit& x : rec.transits) {
        in(x.tile, rec.tiles.size(), "tile"); in(x.op, rec.ops.size(), "op");
        in(x.src, rec.stations.size(), "station"); in(x.dst, rec.stations.size(), "station");
    }
    for (const Compute& c : rec.computes) { in(c.op, rec.ops.size(), "op"); in(c.station, rec.stations.size(), "station"); }
    for (const Op& o : rec.ops) for (std::uint32_t t : o.tiles) in(t, rec.tiles.size(), "tile");
    return rec;
}

} // namespace sw::kpu::program::record
