#pragma once
// ============================================================================
// include/sw/kpu/cycle/report.hpp
// The counters summary (docs/plans/cycle-based-kpu-prototype.md §3.3): one row per component and
// link, with its counts and the operational laws derived from them (counters.hpp).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/counters.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::cycle {

struct ReportRow {
    std::string name, domain;
    bool link = false;
    Counters c;
};

inline std::vector<ReportRow> report(const Simulator& sim) {
    std::vector<ReportRow> rows;
    sim.for_each_counted([&](const std::string& name, const std::string& domain, bool link, const Counters& c) {
        rows.push_back({name, domain, link, c});
    });
    return rows;
}

// The laws on the counters, row by row; an empty list when every one holds. Flow balance: a
// component holds what it accepted and has not completed; a link, whose holding is counted in its
// sender's cycles (a crossing's receiver may pop after the sender's last), is empty at the end of
// a run that `finished`.
inline std::vector<std::string> check_laws(const std::vector<ReportRow>& rows, bool finished) {
    std::vector<std::string> bad;
    auto close = [](double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(b)); };
    for (const ReportRow& r : rows) {
        const Counters& c = r.c;
        const std::string who = (r.link ? "link " : "component ") + r.name;
        if (!r.link && c.busy + c.stalled() + c.idle != c.cycles)
            bad.push_back(who + ": busy " + std::to_string(c.busy) + " + stalled " + std::to_string(c.stalled()) +
                          " + idle " + std::to_string(c.idle) + " != T " + std::to_string(c.cycles));
        if (c.arrivals < c.completions || (!r.link && c.arrivals - c.completions != c.occupancy) ||
            (r.link && finished && c.arrivals != c.completions))
            bad.push_back(who + ": " + std::to_string(c.arrivals) + " arrivals - " + std::to_string(c.completions) +
                          " completions, " + std::to_string(c.occupancy) + " held");
        if (c.completions > 0) {
            if (!close(c.utilization(), c.throughput() * c.service_time()))
                bad.push_back(who + ": U != X S");
            if (!close(c.mean_occupancy(), c.throughput() * c.mean_residence()))
                bad.push_back(who + ": N != X R");
        }
        if (c.utilization() > 1.0 + 1e-12) bad.push_back(who + ": U > 1");
    }
    return bad;
}

// A fixed-width table: T, A, C, busy, idle, the stalls by cause, U = B/T, X = C/T, S = B/C,
// N = Q/T and R = Q/C, and the peak occupancy.
inline std::string format_report(const std::vector<ReportRow>& rows) {
    std::string out;
    char line[512];
    std::snprintf(line, sizeof line, "%-24s %-8s %-4s %10s %8s %8s %10s %10s %8s %8s %8s %8s %8s %6s  %s\n", "name",
                  "domain", "kind", "T", "A", "C", "busy", "idle", "U", "X", "S", "N", "R", "max", "stalls");
    out += line;
    for (const auto& r : rows) {
        const Counters& c = r.c;
        std::string stalls;
        for (std::size_t i = 0; i < kStallCauses; ++i)
            if (c.stalls[i]) {
                if (!stalls.empty()) stalls += ", ";
                stalls += std::string(to_string(static_cast<Stall>(i))) + " " + std::to_string(c.stalls[i]);
            }
        std::snprintf(line, sizeof line,
                      "%-24s %-8s %-4s %10llu %8llu %8llu %10llu %10llu %8.4f %8.4f %8.3f %8.3f %8.3f %6llu  %s\n",
                      r.name.c_str(), r.domain.c_str(), r.link ? "link" : "comp",
                      static_cast<unsigned long long>(c.cycles), static_cast<unsigned long long>(c.arrivals),
                      static_cast<unsigned long long>(c.completions), static_cast<unsigned long long>(c.busy),
                      static_cast<unsigned long long>(c.idle), c.utilization(), c.throughput(), c.service_time(),
                      c.mean_occupancy(), c.mean_residence(), static_cast<unsigned long long>(c.max_occupancy),
                      stalls.empty() ? "-" : stalls.c_str());
        out += line;
    }
    return out;
}

// The counters file (§3.3): every domain (name, period, cycles) and every row's counts, as JSON.
inline void write_counters_json(const Simulator& sim, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cycle::write_counters_json: cannot write " + path);
    auto q = [](const std::string& s) { return "\"" + json_escape(s) + "\""; };
    out << "{\"domains\":[";
    bool first = true;
    sim.for_each_domain([&](const std::string& name, double period_ps, Cycle cycles) {
        out << (first ? "" : ",") << "{\"name\":" << q(name) << ",\"period_ps\":" << period_ps
            << ",\"cycles\":" << cycles << "}";
        first = false;
    });
    out << "],\"rows\":[";
    first = true;
    for (const auto& r : report(sim)) {
        const Counters& c = r.c;
        out << (first ? "\n" : ",\n") << "{\"name\":" << q(r.name) << ",\"domain\":" << q(r.domain)
            << ",\"kind\":\"" << (r.link ? "link" : "component") << "\",\"T\":" << c.cycles
            << ",\"A\":" << c.arrivals << ",\"C\":" << c.completions << ",\"busy\":" << c.busy
            << ",\"idle\":" << c.idle << ",\"Q\":" << c.occupancy_integral << ",\"max\":" << c.max_occupancy
            << ",\"stalls\":{";
        for (std::size_t i = 0; i < kStallCauses; ++i)
            out << (i ? "," : "") << q(to_string(static_cast<Stall>(i))) << ":" << c.stalls[i];
        out << "}}";
        first = false;
    }
    out << "\n]}\n";
    out.close();   // a failure flushing the last bytes is the write's failure too
    if (!out) throw std::runtime_error("cycle::write_counters_json: writing " + path + " failed");
}

}  // namespace sw::kpu::cycle
