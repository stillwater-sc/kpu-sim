// ============================================================================
// src/program/memory_flow_record.cpp
// The .mflow bundle: writer, reader, and its pyramid (docs/plans/memory-side-debugger.md §3.3).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/record/memory_flow_record.hpp>
#include <sw/kpu/program/record/columnar.hpp>
#include <sw/kpu/program/record/tile_flow_lod.hpp>

#include <cmath>
#include <filesystem>
#include <map>
#include <utility>

namespace sw::kpu::program::record {

using namespace columnar;
using columnar::kMaxExactCycle;
using MFR = MemoryFlowRecord;

const char* to_string(MFR::CommandKind k) {
    switch (k) {
        case MFR::CommandKind::Activate: return "ACT";
        case MFR::CommandKind::Read: return "RD";
        case MFR::CommandKind::Write: return "WR";
        case MFR::CommandKind::Precharge: return "PRE";
        case MFR::CommandKind::Refresh: return "REF";
    }
    return "?";
}
const char* to_string(MFR::Outcome o) {
    switch (o) {
        case MFR::Outcome::Unknown: return "unknown";
        case MFR::Outcome::Hit: return "hit";
        case MFR::Outcome::Empty: return "empty";
        case MFR::Outcome::Conflict: return "conflict";
    }
    return "?";
}
const char* to_string(MFR::PortKind k) {
    switch (k) {
        case MFR::PortKind::Offer: return "offer";
        case MFR::PortKind::Accept: return "accept";
        case MFR::PortKind::Refuse: return "refuse";
        case MFR::PortKind::Land: return "land";
        case MFR::PortKind::EjectArrive: return "eject_arrive";
        case MFR::PortKind::EjectWait: return "eject_wait";
        case MFR::PortKind::EjectDeliver: return "eject_deliver";
    }
    return "?";
}

std::uint32_t MFR::station(const std::string& name) const {
    for (std::size_t i = 0; i < stations.size(); ++i)
        if (stations[i].name == name) return static_cast<std::uint32_t>(i);
    return kNone;
}

namespace {

std::string bank_name(const std::string& dev, unsigned mc, unsigned ch, unsigned bg, unsigned ba) {
    return dev + "/mc[" + std::to_string(mc) + "]/ch[" + std::to_string(ch) + "]/bg[" +
           std::to_string(bg) + "]/ba[" + std::to_string(ba) + "]";
}
std::string bus_name(const std::string& dev, unsigned mc, unsigned ch) {
    return dev + "/mc[" + std::to_string(mc) + "]/ch[" + std::to_string(ch) + "]/bus";
}

// The pyramid over the stations (header comment of memory_flow_record.hpp).
Lod build_mflow_lod(const MFR& rec) {
    std::vector<LodRow> rows;
    std::map<std::string, std::size_t> at;
    for (const auto& s : rec.stations) {
        at[s.name] = rows.size();
        rows.push_back({s.name, s.kind, s.capacity, true});
    }
    std::vector<std::vector<std::pair<Cycle, Cycle>>> iv(rows.size());
    auto add = [&](const std::string& name, Cycle a, Cycle b) {
        auto it = at.find(name);
        if (it != at.end()) iv[it->second].emplace_back(a, b);
    };
    for (const auto& c : rec.commands)
        add(bank_name(rec.device, c.mc, c.channel, c.bank_group, c.bank), c.issue, c.end);
    for (const auto& b : rec.bursts) {
        if (b.data_end > b.data_start) add(bus_name(rec.device, b.mc, b.channel), b.data_start, b.data_end);
        add(rec.device + "/dma[" + std::to_string(b.engine) + "]", b.submitted, b.done);
    }
    for (const auto& r : rec.requests)
        if (!r.is_load) add(rec.device + "/dmabuf[" + std::to_string(r.engine) + "]", r.credit, r.retired);
    std::map<std::uint32_t, Cycle> accepted;
    for (const auto& p : rec.ports) {
        if (p.kind == MFR::PortKind::Accept) accepted[p.request] = p.t;
        if (p.kind == MFR::PortKind::Land) {
            auto it = accepted.find(p.request);
            if (it != accepted.end())
                add(rec.device + "/noc/port[" + std::to_string(p.port) + "]", it->second, p.t);
        }
    }
    return build_lod_rows(std::move(rows), iv, std::vector<std::uint32_t>(at.size(), 0), rec.makespan);
}

template <class T, class F> std::vector<T> column(const std::vector<F>& rows, T (*get)(const F&)) {
    std::vector<T> v;
    v.reserve(rows.size());
    for (const auto& r : rows) v.push_back(get(r));
    return v;
}

} // namespace

void write_mflow(const MFR& rec, const std::string& dir) {
    if (!little_endian()) throw RecordError("record: the .mflow writer assumes a little-endian host");
    auto exact = [](Cycle c, const char* what) {
        if (c > kMaxExactCycle)
            throw RecordError(std::string("record: ") + what + " " + std::to_string(c) +
                              " exceeds 2^53 cycles, which an f64 time column cannot hold exactly");
    };
    exact(rec.makespan, "makespan");
    for (const auto& b : rec.bursts) { exact(b.done, "burst done"); exact(b.data_end, "burst data end"); }
    for (const auto& c : rec.commands) exact(c.end, "command end");
    for (const auto& r : rec.requests) exact(r.retired, "request retired");
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) throw RecordError("record: cannot create " + dir + ": " + ec.message());

    json m = json::object();
    m["format"] = "kpu-mflow";
    m["version"] = kMflowVersion;
    m["device"] = rec.device;
    m["makespan"] = rec.makespan;
    m["window"] = rec.window;
    m["timing_note"] = rec.timing_note;
    m["ceiling_bytes_per_cycle"] = rec.ceiling_bytes_per_cycle;
    m["burst_bytes"] = rec.burst_bytes;
    m["arbitration"] = rec.arbitration;
    m["grant_quantum"] = rec.grant_quantum;
    json params = json::object();
    for (const auto& [name, ticks] : rec.dram_timing) params[name] = ticks;
    m["dram_timing"] = json{{"unit", "controller clock tick"}, {"ticks_per_cycle", rec.ticks_per_cycle},
                            {"params", params}};
    json st = json::array();
    for (const auto& s : rec.stations) st.push_back(json{{"name", s.name}, {"kind", s.kind}, {"capacity", s.capacity}});
    m["stations"] = st;

    json tables = json::object();
    {
        Table t{"bursts", "bursts.bin", rec.bursts.size(), {}};
        const auto& v = rec.bursts;
        t.put("engine", "u32", column<std::uint32_t>(v, +[](const MFR::Burst& b) { return b.engine; }));
        t.put("request", "u32", column<std::uint32_t>(v, +[](const MFR::Burst& b) { return b.request; }));
        t.put("row", "u32", column<std::uint32_t>(v, +[](const MFR::Burst& b) { return b.row; }));
        t.put("col", "u32", column<std::uint32_t>(v, +[](const MFR::Burst& b) { return b.col; }));
        t.put("mc", "u8", column<std::uint8_t>(v, +[](const MFR::Burst& b) { return b.mc; }));
        t.put("channel", "u8", column<std::uint8_t>(v, +[](const MFR::Burst& b) { return b.channel; }));
        t.put("rank", "u8", column<std::uint8_t>(v, +[](const MFR::Burst& b) { return b.rank; }));
        t.put("bank_group", "u8", column<std::uint8_t>(v, +[](const MFR::Burst& b) { return b.bank_group; }));
        t.put("bank", "u8", column<std::uint8_t>(v, +[](const MFR::Burst& b) { return b.bank; }));
        t.put("is_load", "u8", column<std::uint8_t>(v, +[](const MFR::Burst& b) -> std::uint8_t { return b.is_load; }));
        t.put("outcome", "u8", column<std::uint8_t>(v, +[](const MFR::Burst& b) { return static_cast<std::uint8_t>(b.outcome); }));
        t.put("t_posted", "f64", column<double>(v, +[](const MFR::Burst& b) { return static_cast<double>(b.posted); }));
        t.put("t_submit", "f64", column<double>(v, +[](const MFR::Burst& b) { return static_cast<double>(b.submitted); }));
        t.put("t_cmd", "f64", column<double>(v, +[](const MFR::Burst& b) { return static_cast<double>(b.first_command); }));
        t.put("t_data0", "f64", column<double>(v, +[](const MFR::Burst& b) { return static_cast<double>(b.data_start); }));
        t.put("t_data1", "f64", column<double>(v, +[](const MFR::Burst& b) { return static_cast<double>(b.data_end); }));
        t.put("t_done", "f64", column<double>(v, +[](const MFR::Burst& b) { return static_cast<double>(b.done); }));
        tables["bursts"] = write_table(t, dir);
    }
    {
        Table t{"commands", "commands.bin", rec.commands.size(), {}};
        const auto& v = rec.commands;
        t.put("row", "u32", column<std::uint32_t>(v, +[](const MFR::Command& c) { return c.row; }));
        t.put("burst", "u32", column<std::uint32_t>(v, +[](const MFR::Command& c) { return c.burst; }));
        t.put("mc", "u8", column<std::uint8_t>(v, +[](const MFR::Command& c) { return c.mc; }));
        t.put("channel", "u8", column<std::uint8_t>(v, +[](const MFR::Command& c) { return c.channel; }));
        t.put("bank_group", "u8", column<std::uint8_t>(v, +[](const MFR::Command& c) { return c.bank_group; }));
        t.put("bank", "u8", column<std::uint8_t>(v, +[](const MFR::Command& c) { return c.bank; }));
        t.put("kind", "u8", column<std::uint8_t>(v, +[](const MFR::Command& c) { return static_cast<std::uint8_t>(c.kind); }));
        t.put("t_issue", "f64", column<double>(v, +[](const MFR::Command& c) { return static_cast<double>(c.issue); }));
        t.put("t_end", "f64", column<double>(v, +[](const MFR::Command& c) { return static_cast<double>(c.end); }));
        t.put("t_data0", "f64", column<double>(v, +[](const MFR::Command& c) { return static_cast<double>(c.data_start); }));
        t.put("t_data1", "f64", column<double>(v, +[](const MFR::Command& c) { return static_cast<double>(c.data_end); }));
        t.put("k_issue", "u64", column<std::uint64_t>(v, +[](const MFR::Command& c) { return c.tick; }));
        t.put("k_end", "u64", column<std::uint64_t>(v, +[](const MFR::Command& c) { return c.tick_end; }));
        t.put("k_data0", "u64", column<std::uint64_t>(v, +[](const MFR::Command& c) { return c.tick_data_start; }));
        t.put("k_data1", "u64", column<std::uint64_t>(v, +[](const MFR::Command& c) { return c.tick_data_end; }));
        tables["commands"] = write_table(t, dir);
    }
    {
        Table t{"requests", "requests.bin", rec.requests.size(), {}};
        const auto& v = rec.requests;
        t.put("address", "u64", column<std::uint64_t>(v, +[](const MFR::Request& r) { return r.address; }));
        t.put("engine", "u32", column<std::uint32_t>(v, +[](const MFR::Request& r) { return r.engine; }));
        t.put("port", "u32", column<std::uint32_t>(v, +[](const MFR::Request& r) { return r.port; }));
        t.put("bytes", "u32", column<std::uint32_t>(v, +[](const MFR::Request& r) { return r.bytes; }));
        t.put("is_load", "u8", column<std::uint8_t>(v, +[](const MFR::Request& r) -> std::uint8_t { return r.is_load; }));
        t.put("t_offered", "f64", column<double>(v, +[](const MFR::Request& r) { return static_cast<double>(r.offered); }));
        t.put("t_credit", "f64", column<double>(v, +[](const MFR::Request& r) { return static_cast<double>(r.credit); }));
        t.put("t_first", "f64", column<double>(v, +[](const MFR::Request& r) { return static_cast<double>(r.first_burst); }));
        t.put("t_last", "f64", column<double>(v, +[](const MFR::Request& r) { return static_cast<double>(r.last_burst); }));
        t.put("t_retired", "f64", column<double>(v, +[](const MFR::Request& r) { return static_cast<double>(r.retired); }));
        tables["requests"] = write_table(t, dir);
    }
    {
        Table t{"buffers", "buffers.bin", rec.buffers.size(), {}};
        const auto& v = rec.buffers;
        t.put("engine", "u32", column<std::uint32_t>(v, +[](const MFR::BufferSample& b) { return b.engine; }));
        t.put("held", "u32", column<std::uint32_t>(v, +[](const MFR::BufferSample& b) { return b.held; }));
        t.put("staged", "u32", column<std::uint32_t>(v, +[](const MFR::BufferSample& b) { return b.staged; }));
        t.put("t", "f64", column<double>(v, +[](const MFR::BufferSample& b) { return static_cast<double>(b.t); }));
        tables["buffers"] = write_table(t, dir);
    }
    {
        Table t{"ports", "ports.bin", rec.ports.size(), {}};
        const auto& v = rec.ports;
        t.put("port", "u32", column<std::uint32_t>(v, +[](const MFR::PortEvent& p) { return p.port; }));
        t.put("engine", "u32", column<std::uint32_t>(v, +[](const MFR::PortEvent& p) { return p.engine; }));
        t.put("request", "u32", column<std::uint32_t>(v, +[](const MFR::PortEvent& p) { return p.request; }));
        t.put("kind", "u8", column<std::uint8_t>(v, +[](const MFR::PortEvent& p) { return static_cast<std::uint8_t>(p.kind); }));
        t.put("t", "f64", column<double>(v, +[](const MFR::PortEvent& p) { return static_cast<double>(p.t); }));
        tables["ports"] = write_table(t, dir);
    }
    m["tables"] = tables;

    std::ofstream out(dir + "/manifest.json");
    out << m.dump(1) << "\n";
    if (!out) throw RecordError("record: cannot write " + dir + "/manifest.json");
    write_lod(build_mflow_lod(rec), dir);
}

MFR read_mflow(const std::string& dir) {
    json m;
    try {
        m = json::parse(slurp(dir + "/manifest.json"));
    } catch (const json::exception& e) {
        throw RecordError("record: " + dir + "/manifest.json is not JSON: " + e.what());
    }
    if (m.value("format", "") != "kpu-mflow")
        throw RecordError("record: " + dir + " is not a .mflow bundle (format '" + m.value("format", "") + "')");
    if (m.value("version", 0u) != kMflowVersion)
        throw RecordError("record: .mflow version " + std::to_string(m.value("version", 0u)) +
                          "; this reader reads version " + std::to_string(kMflowVersion));
    MFR rec;
    rec.device = m.at("device").get<std::string>();
    rec.makespan = m.at("makespan").get<Cycle>();
    rec.window = m.at("window").get<std::uint32_t>();
    rec.timing_note = m.value("timing_note", "");
    rec.ceiling_bytes_per_cycle = m.value("ceiling_bytes_per_cycle", 0.0);
    rec.burst_bytes = m.value("burst_bytes", 0u);
    rec.arbitration = m.value("arbitration", "round_robin");
    rec.grant_quantum = m.value("grant_quantum", 1u);
    if (m.contains("dram_timing")) {
        const json& dt = m.at("dram_timing");
        rec.ticks_per_cycle = dt.value("ticks_per_cycle", 0.0);
        for (auto it = dt.at("params").begin(); it != dt.at("params").end(); ++it)
            rec.dram_timing.emplace_back(it.key(), it.value().get<std::uint32_t>());
    }
    for (const auto& s : m.at("stations"))
        rec.stations.push_back({s.at("name").get<std::string>(), s.at("kind").get<std::string>(),
                                s.at("capacity").get<std::uint64_t>()});
    const json& tables = m.at("tables");
    auto load = [&](const char* name, std::string& blob) -> const json& {
        const json& t = tables.at(name);
        blob = slurp(dir + "/" + t.at("file").get<std::string>());
        return t;
    };
    // A time column is untrusted input: it must hold a whole, non-negative cycle count that an
    // f64 holds exactly. Casting anything else to Cycle would be undefined or would truncate.
    auto t64 = [](const std::vector<double>& v, std::size_t i) {
        const double t = v[i];
        if (!std::isfinite(t) || t < 0.0 || std::floor(t) != t || t > static_cast<double>(kMaxExactCycle))
            throw RecordError("record: a time column holds " + std::to_string(t) +
                              ", which is not a cycle count");
        return static_cast<Cycle>(t);
    };
    {
        std::string b;
        const json& t = load("bursts", b);
        const std::size_t n = t.at("rows").get<std::size_t>();
        auto engine = read_col<std::uint32_t>(t, b, "engine", "u32", n);
        auto request = read_col<std::uint32_t>(t, b, "request", "u32", n);
        auto row = read_col<std::uint32_t>(t, b, "row", "u32", n);
        auto col = read_col<std::uint32_t>(t, b, "col", "u32", n);
        auto mc = read_col<std::uint8_t>(t, b, "mc", "u8", n);
        auto ch = read_col<std::uint8_t>(t, b, "channel", "u8", n);
        auto rk = read_col<std::uint8_t>(t, b, "rank", "u8", n);
        auto bg = read_col<std::uint8_t>(t, b, "bank_group", "u8", n);
        auto ba = read_col<std::uint8_t>(t, b, "bank", "u8", n);
        auto ld = read_col<std::uint8_t>(t, b, "is_load", "u8", n);
        auto oc = read_col<std::uint8_t>(t, b, "outcome", "u8", n);
        auto tp = read_col<double>(t, b, "t_posted", "f64", n);
        auto ts = read_col<double>(t, b, "t_submit", "f64", n);
        auto tc = read_col<double>(t, b, "t_cmd", "f64", n);
        auto d0 = read_col<double>(t, b, "t_data0", "f64", n);
        auto d1 = read_col<double>(t, b, "t_data1", "f64", n);
        auto td = read_col<double>(t, b, "t_done", "f64", n);
        for (std::size_t i = 0; i < n; ++i) {
            if (oc[i] > static_cast<std::uint8_t>(MFR::Outcome::Conflict))
                throw RecordError("record: burst " + std::to_string(i) + " has an unknown outcome");
            rec.bursts.push_back({engine[i], request[i], mc[i], ch[i], rk[i], bg[i], ba[i], row[i], col[i],
                                  ld[i] != 0, static_cast<MFR::Outcome>(oc[i]), t64(ts, i), t64(tc, i),
                                  t64(d0, i), t64(d1, i), t64(td, i), t64(tp, i)});
        }
    }
    {
        std::string b;
        const json& t = load("commands", b);
        const std::size_t n = t.at("rows").get<std::size_t>();
        auto row = read_col<std::uint32_t>(t, b, "row", "u32", n);
        auto burst = read_col<std::uint32_t>(t, b, "burst", "u32", n);
        auto mc = read_col<std::uint8_t>(t, b, "mc", "u8", n);
        auto ch = read_col<std::uint8_t>(t, b, "channel", "u8", n);
        auto bg = read_col<std::uint8_t>(t, b, "bank_group", "u8", n);
        auto ba = read_col<std::uint8_t>(t, b, "bank", "u8", n);
        auto kind = read_col<std::uint8_t>(t, b, "kind", "u8", n);
        auto ti = read_col<double>(t, b, "t_issue", "f64", n);
        auto te = read_col<double>(t, b, "t_end", "f64", n);
        auto d0 = read_col<double>(t, b, "t_data0", "f64", n);
        auto d1 = read_col<double>(t, b, "t_data1", "f64", n);
        auto ki = read_col<std::uint64_t>(t, b, "k_issue", "u64", n);
        auto ke = read_col<std::uint64_t>(t, b, "k_end", "u64", n);
        auto k0 = read_col<std::uint64_t>(t, b, "k_data0", "u64", n);
        auto k1 = read_col<std::uint64_t>(t, b, "k_data1", "u64", n);
        for (std::size_t i = 0; i < n; ++i) {
            if (kind[i] > static_cast<std::uint8_t>(MFR::CommandKind::Refresh))
                throw RecordError("record: command " + std::to_string(i) + " has an unknown kind");
            if (burst[i] != kNone && burst[i] >= rec.bursts.size())
                throw RecordError("record: command " + std::to_string(i) + " names burst " +
                                  std::to_string(burst[i]) + ", past the bursts table");
            rec.commands.push_back({mc[i], ch[i], bg[i], ba[i], row[i], static_cast<MFR::CommandKind>(kind[i]),
                                    burst[i], t64(ti, i), t64(te, i), t64(d0, i), t64(d1, i),
                                    ki[i], ke[i], k0[i], k1[i]});
        }
    }
    {
        std::string b;
        const json& t = load("requests", b);
        const std::size_t n = t.at("rows").get<std::size_t>();
        auto addr = read_col<std::uint64_t>(t, b, "address", "u64", n);
        auto engine = read_col<std::uint32_t>(t, b, "engine", "u32", n);
        auto port = read_col<std::uint32_t>(t, b, "port", "u32", n);
        auto bytes = read_col<std::uint32_t>(t, b, "bytes", "u32", n);
        auto ld = read_col<std::uint8_t>(t, b, "is_load", "u8", n);
        auto to = read_col<double>(t, b, "t_offered", "f64", n);
        auto tc = read_col<double>(t, b, "t_credit", "f64", n);
        auto tf = read_col<double>(t, b, "t_first", "f64", n);
        auto tl = read_col<double>(t, b, "t_last", "f64", n);
        auto tr = read_col<double>(t, b, "t_retired", "f64", n);
        for (std::size_t i = 0; i < n; ++i)
            rec.requests.push_back({engine[i], port[i], ld[i] != 0, addr[i], bytes[i], t64(to, i),
                                    t64(tc, i), t64(tf, i), t64(tl, i), t64(tr, i)});
    }
    {
        std::string b;
        const json& t = load("buffers", b);
        const std::size_t n = t.at("rows").get<std::size_t>();
        auto engine = read_col<std::uint32_t>(t, b, "engine", "u32", n);
        auto held = read_col<std::uint32_t>(t, b, "held", "u32", n);
        auto staged = read_col<std::uint32_t>(t, b, "staged", "u32", n);
        auto tt = read_col<double>(t, b, "t", "f64", n);
        for (std::size_t i = 0; i < n; ++i) rec.buffers.push_back({engine[i], t64(tt, i), held[i], staged[i]});
    }
    {
        std::string b;
        const json& t = load("ports", b);
        const std::size_t n = t.at("rows").get<std::size_t>();
        auto port = read_col<std::uint32_t>(t, b, "port", "u32", n);
        auto engine = read_col<std::uint32_t>(t, b, "engine", "u32", n);
        auto request = read_col<std::uint32_t>(t, b, "request", "u32", n);
        auto kind = read_col<std::uint8_t>(t, b, "kind", "u8", n);
        auto tt = read_col<double>(t, b, "t", "f64", n);
        for (std::size_t i = 0; i < n; ++i) {
            if (kind[i] > static_cast<std::uint8_t>(MFR::PortKind::EjectDeliver))
                throw RecordError("record: port event " + std::to_string(i) + " has an unknown kind");
            rec.ports.push_back({port[i], engine[i], request[i], static_cast<MFR::PortKind>(kind[i]), t64(tt, i)});
        }
    }
    // Indices into the requests table, checked once every table is in.
    for (std::size_t i = 0; i < rec.bursts.size(); ++i)
        if (rec.bursts[i].request != kNone && rec.bursts[i].request >= rec.requests.size())
            throw RecordError("record: burst " + std::to_string(i) + " names request " +
                              std::to_string(rec.bursts[i].request) + ", past the requests table");
    for (std::size_t i = 0; i < rec.ports.size(); ++i)
        if (rec.ports[i].request != kNone && rec.ports[i].request >= rec.requests.size())
            throw RecordError("record: port event " + std::to_string(i) + " names request " +
                              std::to_string(rec.ports[i].request) + ", past the requests table");
    return rec;
}

} // namespace sw::kpu::program::record
