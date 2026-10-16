// ============================================================================
// src/program/domain_flow_record.cpp
// The .dflow bundle: writer and reader (docs/plans/domain-flow-debugger.md §3.4).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/record/domain_flow_record.hpp>
#include <sw/kpu/program/record/columnar.hpp>

#include <filesystem>

namespace sw::kpu::program::record {

using namespace columnar;
using DFR = DomainFlowRecord;

namespace {

const char* kChanNames[csp::kChannels] = {"dram", "l3", "l2", "cf"};
const char* kHeld[csp::kChannels] = {"held_dram", "held_l3", "held_l2", "held_cf"};
const char* kPeak[csp::kChannels] = {"peak_dram", "peak_l3", "peak_l2", "peak_cf"};

template <class T, class R, class F> std::vector<T> col(const std::vector<R>& rows, F get) {
    std::vector<T> v;
    v.reserve(rows.size());
    for (const R& r : rows) v.push_back(static_cast<T>(get(r)));
    return v;
}

}  // namespace

void write_dflow(const DFR& rec, const std::string& dir) {
    if (!little_endian()) throw RecordError("record: the .dflow writer assumes a little-endian host");
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) throw RecordError("record: cannot create " + dir + ": " + ec.message());

    json m = json::object();
    m["format"] = "kpu-dflow";
    m["version"] = kDflowVersion;
    m["level"] = "L-B";
    m["run_id"] = rec.run_id;
    m["program"] = json{{"name", rec.program_name}, {"text", rec.program_text}};
    m["machine"] = json{{"kind", "flat"}, {"l3", rec.l3}, {"banks", rec.banks}, {"place", rec.place}};
    json chans = json::array();
    for (std::size_t c = 0; c < csp::kChannels; ++c) {
        // Only L3's capacity is declared by a level-1 program; the others are unbounded at L-B.
        chans.push_back(json{{"name", kChanNames[c]},
                             {"capacity", c == static_cast<std::size_t>(csp::Chan::L3) ? json(rec.l3) : json(nullptr)}});
    }
    m["channels"] = chans;
    m["processes"] = json::array({"dma", "bm", "str", "cf"});
    json kinds = json::array();
    for (int k = 0; k <= static_cast<int>(csp::Action::Kind::Retain); ++k)
        kinds.push_back(csp::to_string(static_cast<csp::Action::Kind>(k)));
    m["kinds"] = kinds;
    json ops = json::array();
    for (const auto& o : rec.operands)
        ops.push_back(json{{"name", o.name}, {"rows", o.rows}, {"cols", o.cols}, {"tile_rows", o.tile_rows},
                           {"tile_cols", o.tile_cols}});
    m["operands"] = ops;
    m["vars"] = rec.vars;
    m["totals"] = json{{"actions", rec.actions}, {"loads", rec.loads}, {"stores", rec.stores}};
    m["step_limit"] = rec.step_limit;
    m["cut"] = rec.cut;
    m["refusal"] = rec.refusal ? json{{"action", rec.refusal->action}, {"what", rec.refusal->what}} : json(nullptr);
    m["not_modelled"] = json::array({"time: L-B has no time; the axis is the step (action)",
                                     "L2 and fabric capacity: unbounded at L-B (a level-1 program declares only L3)",
                                     "tile values between the recorded actions: a snapshot is taken only where an action leaves a selected tile"});
    {
        json v = json::object();
        v["format"] = "float32";                    // tile_check's ElementFormat: the operands' format
        v["selected"] = rec.values_selected;
        v["cap_bytes"] = rec.values_cap;
        v["cut_at"] = rec.values_cut_at ? json(*rec.values_cut_at) : json(nullptr);
        v["finals"] = rec.finals_note.empty() ? json(true) : json(rec.finals_note);
        m["values"] = v;
    }

    json tables = json::object();
    {
        Table t{"steps", "steps.bin", rec.steps.size(), {}};
        t.put("action", "u64", col<std::uint64_t>(rec.steps, [](const DFR::Step& s) { return s.action; }));
        t.put("residency", "u64", col<std::uint64_t>(rec.steps, [](const DFR::Step& s) { return s.residency; }));
        t.put("where", "u64", col<std::uint64_t>(rec.steps, [](const DFR::Step& s) { return s.where; }));
        t.put("tile", "u32", col<std::uint32_t>(rec.steps, [](const DFR::Step& s) { return s.tile; }));
        for (std::size_t c = 0; c < csp::kChannels; ++c)
            t.put(kHeld[c], "u32", col<std::uint32_t>(rec.steps, [c](const DFR::Step& s) { return s.held[c]; }));
        t.put("l3_tile", "u32", col<std::uint32_t>(rec.steps, [](const DFR::Step& s) { return s.l3_tile; }));
        t.put("slot", "u32", col<std::uint32_t>(rec.steps, [](const DFR::Step& s) { return s.slot; }));
        t.put("bank", "u32", col<std::uint32_t>(rec.steps, [](const DFR::Step& s) { return s.bank; }));
        t.put("kind", "u8", col<std::uint8_t>(rec.steps, [](const DFR::Step& s) { return s.kind; }));
        t.put("process", "u8", col<std::uint8_t>(rec.steps, [](const DFR::Step& s) { return s.process; }));
        t.put("from", "u8", col<std::uint8_t>(rec.steps, [](const DFR::Step& s) { return s.from; }));
        t.put("to", "u8", col<std::uint8_t>(rec.steps, [](const DFR::Step& s) { return s.to; }));
        t.put("flags", "u8", col<std::uint8_t>(rec.steps, [](const DFR::Step& s) { return s.flags; }));
        tables["steps"] = write_table(t, dir);
    }
    {
        Table t{"tiles", "tiles.bin", rec.tiles.size(), {}};
        t.put("operand", "u32", col<std::uint32_t>(rec.tiles, [](const DFR::Tile& x) { return x.operand; }));
        t.put("ti", "u32", col<std::uint32_t>(rec.tiles, [](const DFR::Tile& x) { return x.ti; }));
        t.put("tj", "u32", col<std::uint32_t>(rec.tiles, [](const DFR::Tile& x) { return x.tj; }));
        tables["tiles"] = write_table(t, dir);
    }
    {
        Table t{"residency", "residency.bin", rec.residencies.size(), {}};
        t.put("open", "u64", col<std::uint64_t>(rec.residencies, [](const DFR::Residency& r) { return r.open; }));
        t.put("close", "u64", col<std::uint64_t>(rec.residencies, [](const DFR::Residency& r) { return r.close; }));
        t.put("tile", "u32", col<std::uint32_t>(rec.residencies, [](const DFR::Residency& r) { return r.tile; }));
        t.put("l3_tile", "u32", col<std::uint32_t>(rec.residencies, [](const DFR::Residency& r) { return r.l3_tile; }));
        t.put("slot", "u32", col<std::uint32_t>(rec.residencies, [](const DFR::Residency& r) { return r.slot; }));
        t.put("bank", "u32", col<std::uint32_t>(rec.residencies, [](const DFR::Residency& r) { return r.bank; }));
        t.put("chan", "u8", col<std::uint8_t>(rec.residencies, [](const DFR::Residency& r) { return r.chan; }));
        t.put("flags", "u8", col<std::uint8_t>(rec.residencies, [](const DFR::Residency& r) { return r.flags; }));
        tables["residency"] = write_table(t, dir);
    }
    {
        Table t{"where", "where.bin", rec.wheres.size(), {}};
        t.put("line", "u32", col<std::uint32_t>(rec.wheres, [](const DFR::Where& w) { return w.line; }));
        t.put("loop_first", "u32", col<std::uint32_t>(rec.wheres, [](const DFR::Where& w) { return w.loop_first; }));
        t.put("loop_count", "u16", col<std::uint16_t>(rec.wheres, [](const DFR::Where& w) { return w.loop_count; }));
        tables["where"] = write_table(t, dir);
    }
    {
        Table t{"loops", "loops.bin", rec.loops.size(), {}};
        t.put("value", "i64", col<std::int64_t>(rec.loops, [](const DFR::Loop& l) { return l.value; }));
        t.put("hi", "i64", col<std::int64_t>(rec.loops, [](const DFR::Loop& l) { return l.hi; }));
        t.put("var", "u32", col<std::uint32_t>(rec.loops, [](const DFR::Loop& l) { return l.var; }));
        t.put("line", "u32", col<std::uint32_t>(rec.loops, [](const DFR::Loop& l) { return l.line; }));
        tables["loops"] = write_table(t, dir);
    }
    {
        Table t{"iteration", "iteration.bin", rec.iterations.size(), {}};
        using It = DFR::Iteration;
        t.put("value", "i64", col<std::int64_t>(rec.iterations, [](const It& x) { return x.value; }));
        t.put("first", "u64", col<std::uint64_t>(rec.iterations, [](const It& x) { return x.first; }));
        t.put("last", "u64", col<std::uint64_t>(rec.iterations, [](const It& x) { return x.last; }));
        t.put("actions", "u64", col<std::uint64_t>(rec.iterations, [](const It& x) { return x.actions; }));
        t.put("loads", "u64", col<std::uint64_t>(rec.iterations, [](const It& x) { return x.loads; }));
        t.put("stores", "u64", col<std::uint64_t>(rec.iterations, [](const It& x) { return x.stores; }));
        t.put("line", "u32", col<std::uint32_t>(rec.iterations, [](const It& x) { return x.line; }));
        for (std::size_t c = 0; c < csp::kChannels; ++c)
            t.put(kPeak[c], "u32", col<std::uint32_t>(rec.iterations, [c](const It& x) { return x.peak[c]; }));
        t.put("depth", "u16", col<std::uint16_t>(rec.iterations, [](const It& x) { return x.depth; }));
        tables["iteration"] = write_table(t, dir);
    }
    {
        Table t{"snapshots", "snapshots.bin", rec.snapshots.size(), {}};
        using Sn = DFR::Snapshot;
        t.put("step", "u64", col<std::uint64_t>(rec.snapshots, [](const Sn& x) { return x.step; }));
        t.put("first", "u64", col<std::uint64_t>(rec.snapshots, [](const Sn& x) { return x.first; }));
        t.put("tile", "u32", col<std::uint32_t>(rec.snapshots, [](const Sn& x) { return x.tile; }));
        t.put("height", "u32", col<std::uint32_t>(rec.snapshots, [](const Sn& x) { return x.rows; }));
        t.put("width", "u32", col<std::uint32_t>(rec.snapshots, [](const Sn& x) { return x.cols; }));
        t.put("chan", "u8", col<std::uint8_t>(rec.snapshots, [](const Sn& x) { return x.chan; }));
        tables["snapshots"] = write_table(t, dir);
    }
    {
        static_assert(sizeof(float) == 4, "f32 elements");
        Table t{"elements", "elements.bin", rec.elements.size(), {}};
        t.put("v", "f32", rec.elements);
        tables["elements"] = write_table(t, dir);
    }
    {
        Table t{"lt1_legs", "lt1_legs.bin", rec.legs.size(), {}};
        using Lg = DFR::Leg;
        t.put("action", "u64", col<std::uint64_t>(rec.legs, [](const Lg& x) { return x.action; }));
        t.put("start", "u64", col<std::uint64_t>(rec.legs, [](const Lg& x) { return x.start; }));
        t.put("finish", "u64", col<std::uint64_t>(rec.legs, [](const Lg& x) { return x.finish; }));
        t.put("slot_wait", "u64", col<std::uint64_t>(rec.legs, [](const Lg& x) { return x.slot_wait; }));
        t.put("lane", "u32", col<std::uint32_t>(rec.legs, [](const Lg& x) { return x.lane; }));
        t.put("kind", "u8", col<std::uint8_t>(rec.legs, [](const Lg& x) { return x.kind; }));
        t.put("proc", "u8", col<std::uint8_t>(rec.legs, [](const Lg& x) { return x.proc; }));
        t.put("leg", "u8", col<std::uint8_t>(rec.legs, [](const Lg& x) { return x.leg; }));
        tables["lt1_legs"] = write_table(t, dir);
    }
    {
        Table t{"lt1_tiles", "lt1_tiles.bin", rec.proximity.size(), {}};
        using Px = DFR::Proximity;
        t.put("max_abs", "f64", col<double>(rec.proximity, [](const Px& x) { return x.max_abs; }));
        t.put("max_rel", "f64", col<double>(rec.proximity, [](const Px& x) { return x.max_rel; }));
        t.put("tile", "u32", col<std::uint32_t>(rec.proximity, [](const Px& x) { return x.tile; }));
        t.put("pass", "u8", col<std::uint8_t>(rec.proximity, [](const Px& x) { return x.pass ? 1 : 0; }));
        tables["lt1_tiles"] = write_table(t, dir);
    }
    {
        Table t{"lt1_chains", "lt1_chains.bin", rec.chains.size(), {}};
        using Ch = DFR::Chain;
        t.put("first_action", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.first_action; }));
        t.put("last_action", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.last_action; }));
        t.put("drain_action", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.drain_action; }));
        t.put("in0", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.in0; }));
        t.put("front", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.front; }));
        t.put("in1", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.in1; }));
        t.put("last", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.last; }));
        t.put("out1", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.out1; }));
        t.put("link_delay", "u64", col<std::uint64_t>(rec.chains, [](const Ch& x) { return x.link_delay; }));
        t.put("tile", "u32", col<std::uint32_t>(rec.chains, [](const Ch& x) { return x.tile; }));
        t.put("lane", "u32", col<std::uint32_t>(rec.chains, [](const Ch& x) { return x.lane; }));
        t.put("used_rows", "u32", col<std::uint32_t>(rec.chains, [](const Ch& x) { return x.rows; }));   // not "rows":
        // the checker keys a table's row count by that name
        t.put("used_cols", "u32", col<std::uint32_t>(rec.chains, [](const Ch& x) { return x.cols; }));
        t.put("calls", "u32", col<std::uint32_t>(rec.chains, [](const Ch& x) { return x.calls; }));
        tables["lt1_chains"] = write_table(t, dir);
    }
    {
        Table t{"lt1_calls", "lt1_calls.bin", rec.calls.size(), {}};
        using Ca = DFR::Call;
        t.put("action", "u64", col<std::uint64_t>(rec.calls, [](const Ca& x) { return x.action; }));
        t.put("chain", "u32", col<std::uint32_t>(rec.calls, [](const Ca& x) { return x.chain; }));
        t.put("pos", "u32", col<std::uint32_t>(rec.calls, [](const Ca& x) { return x.pos; }));
        tables["lt1_calls"] = write_table(t, dir);
    }
    if (rec.lt1) {
        json lanes = json::object();
        for (const auto& [p, n] : rec.lt1_lanes) lanes[p] = n;
        m["lt1"] = json{{"level", "L-T1"}, {"run_id", rec.lt1_run_id}, {"makespan", rec.lt1_makespan},
                        {"lanes", lanes}, {"slot_bound", rec.lt1_slot_bound},
                        {"slot_wait_cycles", rec.lt1_slot_wait_cycles}, {"uncalibrated", true},
                        {"bound", json{{"atol", rec.lt1_atol}, {"rtol", rec.lt1_rtol}}}, {"pass", rec.lt1_pass}};
        json one_leg = json::object();
        for (const auto& [why, n] : rec.one_leg) one_leg[why] = n;
        m["lt1"]["one_leg"] = one_leg;
        // The fabric pipeline: null when the deployment declares no PE array (every call one leg).
        m["lt1"]["fabric"] = rec.fabric
            ? json{{"pe_rows", rec.pe_rows}, {"pe_cols", rec.pe_cols}, {"mapping", "output-stationary"},
                   {"drain_edge", "north"}, {"chains", rec.chains.size()}, {"link_delay_cycles", rec.link_delay_cycles}}
            : json(nullptr);
    } else {
        m["lt1"] = nullptr;
    }
    m["tables"] = tables;

    std::ofstream out(dir + "/manifest.json", std::ios::binary);
    out << m.dump(1) << "\n";
    if (!out) throw RecordError("record: cannot write " + dir + "/manifest.json");
}

DFR read_dflow(const std::string& dir) {
    if (!little_endian()) throw RecordError("record: the .dflow reader assumes a little-endian host");
    json m;
    try {
        m = json::parse(slurp(dir + "/manifest.json"));
    } catch (const json::exception& e) {
        throw RecordError("record: " + dir + "/manifest.json: " + e.what());
    }
    try {
        if (m.at("format").get<std::string>() != "kpu-dflow") throw RecordError("record: " + dir + " is not a .dflow bundle");
        const auto version = m.at("version").get<std::uint32_t>();
        if (version < 1 || version > kDflowVersion)
            throw RecordError("record: .dflow version " + m.at("version").dump() + " (this reader reads 1 to " +
                              std::to_string(kDflowVersion) + ")");
        DFR rec;
        rec.run_id = m.at("run_id").get<std::string>();
        rec.program_name = m.at("program").at("name").get<std::string>();
        rec.program_text = m.at("program").at("text").get<std::string>();
        rec.l3 = m.at("machine").at("l3").get<std::uint32_t>();
        rec.banks = m.at("machine").at("banks").get<std::uint32_t>();
        rec.place = m.at("machine").at("place").get<std::string>();
        for (const json& o : m.at("operands"))
            rec.operands.push_back({o.at("name").get<std::string>(), o.at("rows").get<std::uint64_t>(),
                                    o.at("cols").get<std::uint64_t>(), o.at("tile_rows").get<std::uint64_t>(),
                                    o.at("tile_cols").get<std::uint64_t>()});
        rec.vars = m.at("vars").get<std::vector<std::string>>();
        rec.actions = m.at("totals").at("actions").get<std::uint64_t>();
        rec.loads = m.at("totals").at("loads").get<std::uint64_t>();
        rec.stores = m.at("totals").at("stores").get<std::uint64_t>();
        rec.step_limit = m.at("step_limit").get<std::uint64_t>();
        rec.cut = m.at("cut").get<bool>();
        if (!m.at("refusal").is_null())
            rec.refusal = DFR::Refusal{m.at("refusal").at("action").get<std::uint64_t>(),
                                       m.at("refusal").at("what").get<std::string>()};
        const json& tables = m.at("tables");
        auto table = [&](const char* name, std::size_t& rows) -> std::pair<const json&, std::string> {
            const json& t = tables.at(name);
            rows = t.at("rows").get<std::size_t>();
            return {t, slurp(dir + "/" + t.at("file").get<std::string>())};
        };
        std::size_t n = 0;
        {
            auto [t, blob] = table("steps", n);
            auto action = read_col<std::uint64_t>(t, blob, "action", "u64", n);
            auto residency = read_col<std::uint64_t>(t, blob, "residency", "u64", n);
            auto where = read_col<std::uint64_t>(t, blob, "where", "u64", n);
            auto tile = read_col<std::uint32_t>(t, blob, "tile", "u32", n);
            std::array<std::vector<std::uint32_t>, csp::kChannels> held;
            for (std::size_t c = 0; c < csp::kChannels; ++c) held[c] = read_col<std::uint32_t>(t, blob, kHeld[c], "u32", n);
            auto l3_tile = read_col<std::uint32_t>(t, blob, "l3_tile", "u32", n);
            auto slot = read_col<std::uint32_t>(t, blob, "slot", "u32", n);
            auto bank = read_col<std::uint32_t>(t, blob, "bank", "u32", n);
            auto kind = read_col<std::uint8_t>(t, blob, "kind", "u8", n);
            auto process = read_col<std::uint8_t>(t, blob, "process", "u8", n);
            auto from = read_col<std::uint8_t>(t, blob, "from", "u8", n);
            auto to = read_col<std::uint8_t>(t, blob, "to", "u8", n);
            auto flags = read_col<std::uint8_t>(t, blob, "flags", "u8", n);
            rec.steps.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                DFR::Step& s = rec.steps[i];
                s.action = action[i];
                s.residency = residency[i];
                s.where = where[i];
                s.tile = tile[i];
                for (std::size_t c = 0; c < csp::kChannels; ++c) s.held[c] = held[c][i];
                s.l3_tile = l3_tile[i];
                s.slot = slot[i];
                s.bank = bank[i];
                s.kind = kind[i];
                s.process = process[i];
                s.from = from[i];
                s.to = to[i];
                s.flags = flags[i];
            }
        }
        {
            auto [t, blob] = table("tiles", n);
            auto operand = read_col<std::uint32_t>(t, blob, "operand", "u32", n);
            auto ti = read_col<std::uint32_t>(t, blob, "ti", "u32", n);
            auto tj = read_col<std::uint32_t>(t, blob, "tj", "u32", n);
            for (std::size_t i = 0; i < n; ++i) rec.tiles.push_back({operand[i], ti[i], tj[i]});
        }
        {
            auto [t, blob] = table("residency", n);
            auto open = read_col<std::uint64_t>(t, blob, "open", "u64", n);
            auto close = read_col<std::uint64_t>(t, blob, "close", "u64", n);
            auto tile = read_col<std::uint32_t>(t, blob, "tile", "u32", n);
            auto l3_tile = read_col<std::uint32_t>(t, blob, "l3_tile", "u32", n);
            auto slot = read_col<std::uint32_t>(t, blob, "slot", "u32", n);
            auto bank = read_col<std::uint32_t>(t, blob, "bank", "u32", n);
            auto chan = read_col<std::uint8_t>(t, blob, "chan", "u8", n);
            auto flags = read_col<std::uint8_t>(t, blob, "flags", "u8", n);
            for (std::size_t i = 0; i < n; ++i)
                rec.residencies.push_back({tile[i], chan[i], open[i], close[i], l3_tile[i], slot[i], bank[i], flags[i]});
        }
        {
            auto [t, blob] = table("where", n);
            auto line = read_col<std::uint32_t>(t, blob, "line", "u32", n);
            auto first = read_col<std::uint32_t>(t, blob, "loop_first", "u32", n);
            auto count = read_col<std::uint16_t>(t, blob, "loop_count", "u16", n);
            for (std::size_t i = 0; i < n; ++i) rec.wheres.push_back({line[i], first[i], count[i]});
        }
        {
            auto [t, blob] = table("loops", n);
            auto value = read_col<std::int64_t>(t, blob, "value", "i64", n);
            auto hi = read_col<std::int64_t>(t, blob, "hi", "i64", n);
            auto var = read_col<std::uint32_t>(t, blob, "var", "u32", n);
            auto line = read_col<std::uint32_t>(t, blob, "line", "u32", n);
            for (std::size_t i = 0; i < n; ++i) rec.loops.push_back({var[i], value[i], hi[i], line[i]});
        }
        {
            auto [t, blob] = table("iteration", n);
            auto value = read_col<std::int64_t>(t, blob, "value", "i64", n);
            auto first = read_col<std::uint64_t>(t, blob, "first", "u64", n);
            auto last = read_col<std::uint64_t>(t, blob, "last", "u64", n);
            auto actions = read_col<std::uint64_t>(t, blob, "actions", "u64", n);
            auto loads = read_col<std::uint64_t>(t, blob, "loads", "u64", n);
            auto stores = read_col<std::uint64_t>(t, blob, "stores", "u64", n);
            auto line = read_col<std::uint32_t>(t, blob, "line", "u32", n);
            std::array<std::vector<std::uint32_t>, csp::kChannels> peak;
            for (std::size_t c = 0; c < csp::kChannels; ++c) peak[c] = read_col<std::uint32_t>(t, blob, kPeak[c], "u32", n);
            auto depth = read_col<std::uint16_t>(t, blob, "depth", "u16", n);
            for (std::size_t i = 0; i < n; ++i) {
                DFR::Iteration it;
                it.depth = depth[i];
                it.line = line[i];
                it.value = value[i];
                it.first = first[i];
                it.last = last[i];
                it.actions = actions[i];
                it.loads = loads[i];
                it.stores = stores[i];
                for (std::size_t c = 0; c < csp::kChannels; ++c) it.peak[c] = peak[c][i];
                rec.iterations.push_back(it);
            }
        }
        if (version >= 2) {
            const json& v = m.at("values");
            rec.values_selected = v.at("selected").get<std::string>();
            rec.values_cap = v.at("cap_bytes").get<std::uint64_t>();
            if (!v.at("cut_at").is_null()) rec.values_cut_at = v.at("cut_at").get<std::uint64_t>();
            if (!v.at("finals").is_boolean()) rec.finals_note = v.at("finals").get<std::string>();
            {
                auto [t, blob] = table("snapshots", n);
                auto step = read_col<std::uint64_t>(t, blob, "step", "u64", n);
                auto first = read_col<std::uint64_t>(t, blob, "first", "u64", n);
                auto tile = read_col<std::uint32_t>(t, blob, "tile", "u32", n);
                auto rows = read_col<std::uint32_t>(t, blob, "height", "u32", n);
                auto cols = read_col<std::uint32_t>(t, blob, "width", "u32", n);
                auto chan = read_col<std::uint8_t>(t, blob, "chan", "u8", n);
                for (std::size_t i = 0; i < n; ++i) rec.snapshots.push_back({step[i], first[i], tile[i], rows[i], cols[i], chan[i]});
            }
            {
                auto [t, blob] = table("elements", n);
                rec.elements = read_col<float>(t, blob, "v", "f32", n);
            }
        }
        if (version >= 3 && !m.at("lt1").is_null()) {
            const json& o = m.at("lt1");
            rec.lt1 = true;
            rec.lt1_run_id = o.at("run_id").get<std::string>();
            rec.lt1_makespan = o.at("makespan").get<std::uint64_t>();
            rec.lt1_slot_bound = o.at("slot_bound").get<std::uint64_t>();
            rec.lt1_slot_wait_cycles = o.at("slot_wait_cycles").get<std::uint64_t>();
            for (const auto& [p, n] : o.at("lanes").items()) rec.lt1_lanes.emplace_back(p, n.get<std::uint32_t>());
            rec.lt1_atol = o.at("bound").at("atol").get<double>();
            rec.lt1_rtol = o.at("bound").at("rtol").get<double>();
            rec.lt1_pass = o.at("pass").get<bool>();
        }
        if (version >= 3) {
            {
                auto [t, blob] = table("lt1_legs", n);
                auto action = read_col<std::uint64_t>(t, blob, "action", "u64", n);
                auto start = read_col<std::uint64_t>(t, blob, "start", "u64", n);
                auto finish = read_col<std::uint64_t>(t, blob, "finish", "u64", n);
                auto wait = read_col<std::uint64_t>(t, blob, "slot_wait", "u64", n);
                auto lane = read_col<std::uint32_t>(t, blob, "lane", "u32", n);
                auto kind = read_col<std::uint8_t>(t, blob, "kind", "u8", n);
                auto proc = read_col<std::uint8_t>(t, blob, "proc", "u8", n);
                auto leg = read_col<std::uint8_t>(t, blob, "leg", "u8", n);
                for (std::size_t i = 0; i < n; ++i)
                    rec.legs.push_back({action[i], start[i], finish[i], wait[i], lane[i], kind[i], proc[i], leg[i]});
            }
            {
                auto [t, blob] = table("lt1_tiles", n);
                auto max_abs = read_col<double>(t, blob, "max_abs", "f64", n);
                auto max_rel = read_col<double>(t, blob, "max_rel", "f64", n);
                auto tile = read_col<std::uint32_t>(t, blob, "tile", "u32", n);
                auto pass = read_col<std::uint8_t>(t, blob, "pass", "u8", n);
                for (std::size_t i = 0; i < n; ++i) rec.proximity.push_back({tile[i], max_abs[i], max_rel[i], pass[i] != 0});
            }
        }
        if (version >= 4) {
            if (rec.lt1) {
                const json& o = m.at("lt1");
                for (const auto& [why, k] : o.at("one_leg").items()) rec.one_leg.emplace_back(why, k.get<std::uint64_t>());
                if (!o.at("fabric").is_null()) {
                    const json& f = o.at("fabric");
                    rec.fabric = true;
                    rec.pe_rows = f.at("pe_rows").get<std::uint32_t>();
                    rec.pe_cols = f.at("pe_cols").get<std::uint32_t>();
                    rec.link_delay_cycles = f.at("link_delay_cycles").get<std::uint64_t>();
                }
            }
            {
                auto [t, blob] = table("lt1_chains", n);
                auto first = read_col<std::uint64_t>(t, blob, "first_action", "u64", n);
                auto lastA = read_col<std::uint64_t>(t, blob, "last_action", "u64", n);
                auto drain = read_col<std::uint64_t>(t, blob, "drain_action", "u64", n);
                auto in0 = read_col<std::uint64_t>(t, blob, "in0", "u64", n);
                auto front = read_col<std::uint64_t>(t, blob, "front", "u64", n);
                auto in1 = read_col<std::uint64_t>(t, blob, "in1", "u64", n);
                auto last = read_col<std::uint64_t>(t, blob, "last", "u64", n);
                auto out1 = read_col<std::uint64_t>(t, blob, "out1", "u64", n);
                auto delay = read_col<std::uint64_t>(t, blob, "link_delay", "u64", n);
                auto tile = read_col<std::uint32_t>(t, blob, "tile", "u32", n);
                auto lane = read_col<std::uint32_t>(t, blob, "lane", "u32", n);
                auto rows = read_col<std::uint32_t>(t, blob, "used_rows", "u32", n);
                auto cols = read_col<std::uint32_t>(t, blob, "used_cols", "u32", n);
                auto calls = read_col<std::uint32_t>(t, blob, "calls", "u32", n);
                for (std::size_t i = 0; i < n; ++i)
                    rec.chains.push_back({tile[i], lane[i], rows[i], cols[i], calls[i], first[i], lastA[i], drain[i],
                                          in0[i], front[i], in1[i], last[i], out1[i], delay[i]});
            }
            {
                auto [t, blob] = table("lt1_calls", n);
                auto action = read_col<std::uint64_t>(t, blob, "action", "u64", n);
                auto chain = read_col<std::uint32_t>(t, blob, "chain", "u32", n);
                auto pos = read_col<std::uint32_t>(t, blob, "pos", "u32", n);
                for (std::size_t i = 0; i < n; ++i) rec.calls.push_back({action[i], chain[i], pos[i]});
            }
        }
        return rec;
    } catch (const json::exception& e) {
        throw RecordError("record: " + dir + "/manifest.json: " + e.what());
    }
}

} // namespace sw::kpu::program::record
