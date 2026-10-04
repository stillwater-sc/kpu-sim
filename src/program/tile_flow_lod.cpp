// ============================================================================
// src/program/tile_flow_lod.cpp
// The level-of-detail pyramid over a tile-flow record (#286 step 3). See the header for the
// rows, the metrics and why every metric merges exactly.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/record/tile_flow_lod.hpp>

#include <sw/kpu/program/tile_transaction_executor.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace sw::kpu::program::record {

namespace {
using json = nlohmann::ordered_json;
using Interval = std::pair<Cycle, Cycle>;

// One row's metrics at the base level, by an exact sweep: occupancy is a step function of
// time, constant between interval ends, so each constant segment adds level x overlap to every
// bin it crosses and raises that bin's peak to its level. Ends sort before starts at the same
// cycle -- the executor's order, in which completions return credits before ops fire -- and a
// zero-length interval occupies no time, matching peak_l3_occupancy().
void accumulate(const std::vector<Interval>& iv, std::uint32_t constant, Cycle makespan,
                Cycle width, std::uint64_t bins, double* occ, std::uint32_t* peak,
                std::uint32_t* starts) {
    auto bin_of = [&](Cycle t) { return std::min<std::uint64_t>(t / width, bins - 1); };
    for (const Interval& x : iv) ++starts[bin_of(x.first)];

    std::vector<std::pair<Cycle, int>> ev;
    for (const Interval& x : iv)
        if (x.second > x.first) {
            ev.emplace_back(x.first, +1);
            ev.emplace_back(x.second, -1);
        }
    std::sort(ev.begin(), ev.end());

    auto segment = [&](Cycle a, Cycle b, std::int64_t level) {
        if (b <= a || level <= 0) return;
        for (std::uint64_t bi = a / width; bi < bins && bi * width < b; ++bi) {
            const Cycle lo = std::max<Cycle>(a, bi * width), hi = std::min<Cycle>(b, (bi + 1) * width);
            occ[bi] += static_cast<double>(level) * static_cast<double>(hi - lo);
            peak[bi] = std::max<std::uint32_t>(peak[bi], static_cast<std::uint32_t>(level));
        }
    };
    std::int64_t cur = constant;
    Cycle prev = 0;
    for (std::size_t i = 0; i < ev.size();) {
        const Cycle t = ev[i].first;
        segment(prev, std::min(t, makespan), cur);
        while (i < ev.size() && ev[i].first == t) cur += ev[i++].second;
        prev = std::max(prev, std::min(t, makespan));
    }
    segment(prev, makespan, cur);
}

} // namespace

LodLevel merge_level(const LodLevel& c, std::size_t rows) {
    LodLevel p;
    p.k = c.k + 1;
    p.bins = (c.bins + 1) / 2;
    p.occ.assign(rows * p.bins, 0.0);
    p.peak.assign(rows * p.bins, 0);
    p.starts.assign(rows * p.bins, 0);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::uint64_t b = 0; b < c.bins; ++b) {
            const std::size_t from = r * c.bins + b, to = r * p.bins + b / 2;
            p.occ[to] += c.occ[from];
            p.peak[to] = std::max(p.peak[to], c.peak[from]);
            p.starts[to] += c.starts[from];
        }
    return p;
}

Lod build_lod(const TileFlowRecord& rec, std::uint64_t max_base_bins) {
    if (max_base_bins == 0) throw RecordError("lod: max_base_bins must be positive");
    Lod lod;
    for (const Station& s : rec.stations)
        lod.rows.push_back({s.name, s.kind, s.capacity, s.modelled});
    for (const MoverPool& m : rec.movers)
        lod.rows.push_back({"mover:" + m.name, "mover", m.lanes, true});

    unsigned k = 0;
    auto bins_at = [&](unsigned kk) {
        const Cycle w = Cycle{1} << kk;
        return std::max<std::uint64_t>(1, (rec.makespan + w - 1) / w);
    };
    while (bins_at(k) > max_base_bins) ++k;
    LodLevel base;
    base.k = k;
    base.bins = bins_at(k);
    const std::size_t rows = lod.rows.size();
    base.occ.assign(rows * base.bins, 0.0);
    base.peak.assign(rows * base.bins, 0);
    base.starts.assign(rows * base.bins, 0);
    const Cycle width = Cycle{1} << k;

    for (std::size_t r = 0; r < rows; ++r) {
        std::vector<Interval> iv;
        std::uint32_t constant = 0;
        if (r < rec.stations.size()) {
            const std::string& kind = rec.stations[r].kind;
            if (kind == "l3") {
                for (const Residency& x : rec.residency)
                    if (x.station == r) iv.emplace_back(x.t0, x.t1);
                constant = static_cast<std::uint32_t>(rec.foreign_slots);
            } else if (kind == "cf") {
                for (const Compute& c : rec.computes)
                    if (c.station == r) iv.emplace_back(c.t0, c.t1);
            }
        } else {
            const std::string& pool = rec.movers[r - rec.stations.size()].name;
            for (const Transit& t : rec.transits)
                if (to_string(static_cast<Mover>(t.mover)) == pool) iv.emplace_back(t.t0, t.t1);
        }
        accumulate(iv, constant, rec.makespan, width, base.bins, base.occ.data() + r * base.bins,
                   base.peak.data() + r * base.bins, base.starts.data() + r * base.bins);
    }
    lod.levels.push_back(std::move(base));
    while (lod.levels.back().bins > 1) lod.levels.push_back(merge_level(lod.levels.back(), rows));
    return lod;
}

// ---- files -------------------------------------------------------------------
namespace {
template <class T> std::size_t append(std::string& blob, const std::vector<T>& v) {
    while (blob.size() % 8) blob.push_back('\0');
    const std::size_t off = blob.size();
    blob.resize(off + v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(blob.data() + off, v.data(), v.size() * sizeof(T));
    return off;
}

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw RecordError("lod: cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

template <class T> std::vector<T> take(const std::string& blob, std::size_t off, std::size_t n) {
    if (off > blob.size() || n > (blob.size() - off) / sizeof(T))
        throw RecordError("lod: a column runs past the end of lod.bin");
    std::vector<T> v(n);
    if (n) std::memcpy(v.data(), blob.data() + off, n * sizeof(T));
    return v;
}
} // namespace

void write_lod(const Lod& lod, const std::string& dir) {
    std::string blob;
    json m = json::object();
    m["format"] = "kpu-tflow-lod";
    m["version"] = 1;
    json rows = json::array();
    for (const LodRow& r : lod.rows)
        rows.push_back(json{{"name", r.name}, {"kind", r.kind}, {"capacity", r.capacity}, {"modelled", r.modelled}});
    m["rows"] = rows;
    m["metrics"] = json{{"occ", "f64 occupancy x cycles; merges by sum"},
                        {"peak", "u32 max occupancy in the bin; merges by max"},
                        {"starts", "u32 intervals beginning in the bin; merges by sum"}};
    json levels = json::array();
    for (const LodLevel& l : lod.levels) {
        json o = json{{"k", l.k}, {"bins", l.bins}};
        o["occ"] = append(blob, l.occ);
        o["peak"] = append(blob, l.peak);
        o["starts"] = append(blob, l.starts);
        levels.push_back(o);
    }
    m["file"] = "lod.bin";
    m["levels"] = levels;
    std::ofstream bin(dir + "/lod.bin", std::ios::binary);
    bin << blob;
    std::ofstream js(dir + "/lod.json", std::ios::binary);
    js << m.dump(1) << "\n";
    if (!bin || !js) throw RecordError("lod: cannot write " + dir + "/lod.{json,bin}");
}

Lod read_lod(const std::string& dir) {
    json m;
    try {
        m = json::parse(slurp(dir + "/lod.json"));
    } catch (const nlohmann::json::parse_error& e) {
        throw RecordError(std::string("lod: lod.json is not valid JSON: ") + e.what());
    }
    if (m.value("format", "") != "kpu-tflow-lod" || m.value("version", 0) != 1)
        throw RecordError("lod: not a version-1 kpu-tflow-lod file");
    Lod lod;
    for (const json& r : m.at("rows"))
        lod.rows.push_back({r.at("name").get<std::string>(), r.at("kind").get<std::string>(),
                            r.at("capacity").get<std::uint64_t>(), r.at("modelled").get<bool>()});
    const std::string blob = slurp(dir + "/" + m.at("file").get<std::string>());
    for (const json& l : m.at("levels")) {
        LodLevel x;
        x.k = l.at("k").get<unsigned>();
        x.bins = l.at("bins").get<std::uint64_t>();
        // Checked before the multiply: a hostile bin count must be refused, not wrap around
        // into a small allocation that the column reads then overrun.
        if (x.bins == 0) throw RecordError("lod: a level declares zero bins");
        if (!lod.rows.empty() && x.bins > std::numeric_limits<std::size_t>::max() / lod.rows.size())
            throw RecordError("lod: a level declares " + std::to_string(x.bins) +
                              " bins, more than any file could hold");
        const std::size_t n = lod.rows.size() * static_cast<std::size_t>(x.bins);
        x.occ = take<double>(blob, l.at("occ").get<std::size_t>(), n);
        x.peak = take<std::uint32_t>(blob, l.at("peak").get<std::size_t>(), n);
        x.starts = take<std::uint32_t>(blob, l.at("starts").get<std::size_t>(), n);
        lod.levels.push_back(std::move(x));
    }
    return lod;
}

} // namespace sw::kpu::program::record
