// ============================================================================
// tools/csp-coverage/kpu_csp_coverage.cpp
// kpu-csp-coverage -- the coverage report's measured part (#617; docs/plans/distributed-csp-programs.md
// §3.5): every pattern class's generator, on every DMM mesh fixture, measured
// (csp/distributed_coverage.hpp), with each class's verdict by the report's rules.
//
//   kpu-csp-coverage --deploy-dir DIR [--check DOC | --write DOC]
//
// Prints the block that docs/analysis/distributed-csp-coverage.md carries between its markers.
// --check DOC exits 1 if DOC's block is not what the programs measure now (CI keeps the report
// measured); --write DOC replaces it.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/distributed_coverage.hpp>
#include <sw/kpu/program/csp/gen/distributed_append.hpp>
#include <sw/kpu/program/csp/gen/distributed_broadcast.hpp>
#include <sw/kpu/program/csp/gen/distributed_chained_gemm.hpp>
#include <sw/kpu/program/csp/gen/distributed_elementwise.hpp>
#include <sw/kpu/program/csp/gen/distributed_gather.hpp>
#include <sw/kpu/program/csp/gen/distributed_gemm.hpp>
#include <sw/kpu/program/csp/gen/distributed_reduction.hpp>
#include <sw/kpu/program/csp/gen/distributed_transpose.hpp>
#include <sw/kpu/program/csp/gen/distributed_window.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;

namespace {

constexpr const char* kBegin = "<!-- coverage:begin (kpu-csp-coverage writes this block; do not edit) -->";
constexpr const char* kEnd = "<!-- coverage:end -->";

struct Pattern {
    const char* id;
    const char* name;
    std::function<DistributedProgram(const platform::DeviceSpecification&, Dim side)> make;
};

// Each class at the sizes its generator's tests run, per mesh side.
std::vector<Pattern> patterns() {
    using namespace gen;
    return {
        {"P1", "GEMM", [](const auto& d, Dim s) { return generate_distributed_gemm({32 * s, 32 * s, s > 2 ? 48u : 32u, 16}, d); }},
        {"P2", "sliding window", [](const auto& d, Dim s) { return generate_distributed_box3x3({8 * s, 8 * s}, d); }},
        {"P3", "row reduction", [](const auto& d, Dim s) { return generate_distributed_rowsum({s > 2 ? 64u : 32u, s > 2 ? 128u : 32u, 16}, d); }},
        {"P4", "gather/scatter", [](const auto& d, Dim s) { return generate_distributed_gather({8 * s * s, 8, 13, 8 * s * s, 8}, d); }},
        {"P5", "broadcast", [](const auto& d, Dim s) { return generate_distributed_broadcast({s > 2 ? 64u : 32u, 16}, d); }},
        {"P6", "two-operand aligned", [](const auto& d, Dim s) { return generate_distributed_add({16 * s, 16 * s, 16}, d); }},
        {"P7", "layout transform", [](const auto& d, Dim s) { return generate_distributed_transpose({8 * s, 8 * s}, d); }},
        {"P8", "chained-GEMM fusion", [](const auto& d, Dim s) { return generate_distributed_chained_gemm({8 * s, 16, 8 * s, 16, 8}, d); }},
        {"P9", "append", [](const auto& d, Dim) { return generate_distributed_append({2, 3, 3, 8}, d); }},
    };
}

std::string kib(std::uint64_t b) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(1);
    o << b / 1024.0;
    return o.str();
}

// The report's rules, on the largest mesh's measures.
std::string verdict(const DistributedCoverage& m) {
    if (!m.checks) return "**needs the allocation changed:** the checks fail (" + m.failure + ")";
    if (m.movers > 8 || m.sends > 4) return "**needs the allocation changed:** " + std::to_string(m.movers) + " movers on a node";
    std::vector<std::string> cost;
    if (m.transit) cost.push_back(std::to_string(m.transit) + " of its " + std::to_string(m.pushes) + " mesh pushes only forward");
    if (2 * m.l3_slots > m.l3_capacity)
        cost.push_back(std::to_string(m.l3_slots) + " of a node's " + std::to_string(m.l3_capacity) + " L3 slots");
    if (cost.empty()) return "**fits**";
    std::string s = "**fits at a cost:** ";
    for (std::size_t i = 0; i < cost.size(); ++i) s += (i ? "; " : "") + cost[i];
    return s;
}

std::string report(const std::string& dir) {
    const std::vector<std::pair<const char*, Dim>> meshes = {{"kpu_dmm_1x1.json", 1}, {"kpu_dmm_2x2.json", 2}, {"kpu_dmm_4x4.json", 4}};
    std::ostringstream t, v;
    t << "| Class | Mesh | Movers (sends) | L3 slots | L2 slots | Credits per flow | Hops | Pushes (transit) | L-B rounds | D0-D5 | "
         "DMA KiB | Busiest port KiB | Busiest ring segment KiB, 4 / 8 / 16 / 32 PHYs |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    v << "| Class | Verdict on the 4x4 mesh |\n|---|---|\n";
    for (const Pattern& pt : patterns()) {
        DistributedCoverage last;
        for (const auto& [file, side] : meshes) {
            const platform::DeviceSpecification d = platform::read_spec_file(dir + "/" + file).device(0);
            DistributedCoverage m;
            try {
                m = measure_distributed(pt.make(d, side), d);
            } catch (const std::exception& e) {
                m.failure = e.what();
            }
            t << "| " << pt.id << " " << pt.name << " | " << side << "x" << side << " | " << m.movers << " (" << m.sends << ") | "
              << m.l3_slots << " / " << m.l3_capacity << " | " << m.l2_slots << " / " << m.l2_capacity << " | ";
            if (m.credits_max) t << m.credits_min << (m.credits_min != m.credits_max ? "-" + std::to_string(m.credits_max) : "");
            else t << "-";
            t << " | " << m.hops << " | " << m.pushes << " (" << m.transit << ") | " << m.rounds << " | " << (m.checks ? "pass" : "FAIL")
              << " | " << kib(m.dma_bytes) << " | " << kib(m.port_bytes) << " | ";
            bool first = true;
            for (const auto& [n, b] : m.ring) {
                t << (first ? "" : " / ") << kib(b);
                first = false;
            }
            t << " |\n";
            last = m;
        }
        v << "| " << pt.id << " " << pt.name << " | " << verdict(last) << " |\n";
    }
    return std::string(kBegin) + "\n\n" + v.str() + "\n" + t.str() + "\n" + kEnd;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    std::string r = s.str();
    std::string out;
    for (char c : r)
        if (c != '\r') out += c;                              // a Windows checkout's line ends
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string dir, check, write;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--deploy-dir" || a == "--check" || a == "--write") && i + 1 < argc) {
            (a == "--deploy-dir" ? dir : a == "--check" ? check : write) = argv[++i];
        } else {
            std::cerr << "usage: kpu-csp-coverage --deploy-dir DIR [--check DOC | --write DOC]\n";
            return 2;
        }
    }
    if (dir.empty()) {
        std::cerr << "usage: kpu-csp-coverage --deploy-dir DIR [--check DOC | --write DOC]\n";
        return 2;
    }
    std::string block;
    try {
        block = report(dir);
    } catch (const std::exception& e) {
        std::cerr << "kpu-csp-coverage: " << e.what() << "\n";
        return 2;
    }
    const std::string doc = check.empty() ? write : check;
    if (doc.empty()) {
        std::cout << block << "\n";
        return 0;
    }
    const std::string text = read_file(doc);
    const auto b = text.find(kBegin), e = text.find(kEnd);
    if (b == std::string::npos || e == std::string::npos || e < b) {
        std::cerr << "kpu-csp-coverage: " << doc << " has no coverage block\n";
        return 2;
    }
    const std::string now = text.substr(b, e + std::string(kEnd).size() - b);
    if (!check.empty()) {
        if (now == block) return 0;
        std::cerr << "kpu-csp-coverage: " << doc << "'s block is not what the programs measure; run kpu-csp-coverage --write\n";
        return 1;
    }
    std::ofstream out(doc, std::ios::binary);
    out << text.substr(0, b) << block << text.substr(e + std::string(kEnd).size());
    return 0;
}
