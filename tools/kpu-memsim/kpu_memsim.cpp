// ============================================================================
// tools/kpu-memsim/kpu_memsim.cpp
// kpu-memsim -- run the memory side of a deployment on its own and record it
// (docs/plans/memory-side-debugger.md §3.6, step 3).
//
//   kpu-memsim --deploy <spec.json> [--device N] --scenario <scenario.json> --out <dir.mflow>
//
// The scenario names the request models on each NoC port and the port stubs' behaviour; the
// tool runs MemorySideHarness, writes the .mflow bundle, and prints a summary: makespan,
// bandwidth against the DRAM ceiling, bursts by page outcome, commands, and refusals.
//
// SCENARIO (JSON; unknown keys are refused; any number may be a "0x..." string):
//   window, store_buffer_blocks, l3_slots, max_cycles      (all optional)
//   ports:   { infinite, block_cycles, input_queue_blocks, consume_latency, eject_interval }
//   streams: [ { port: <n> | "first", issue_interval, model: {...} } ]
//   model:   { kind: stream | strided | random | matrix_tiles | replay_matmul, load: bool,
//              base, count, bytes, stride, region, seed,                 (stream/strided/random)
//              rows, cols, element_bytes, tile_rows, tile_cols, pitch,   (matrix_tiles)
//              M, N, K, tile }                                           (replay_matmul)
//
// Exit codes: 0 = run complete and recorded; 1 = the run did not finish within max_cycles;
// 2 = bad arguments, spec or scenario.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/memory_flow_record.hpp>
#include <sw/kpu/timing/memory_side_harness.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

using json = nlohmann::json;
using namespace sw::kpu::timing;
using namespace sw::kpu::timing::memside;
namespace rec = sw::kpu::program::record;

namespace {

struct UsageError : std::runtime_error { using std::runtime_error::runtime_error; };

void reject_unknown(const json& o, const std::set<std::string>& keys, const std::string& where) {
    if (!o.is_object()) throw UsageError(where + " must be an object");
    for (auto it = o.begin(); it != o.end(); ++it)
        if (!keys.count(it.key())) throw UsageError(where + ": unknown key '" + it.key() + "'");
}

// A non-negative whole number no larger than `max` (the destination field's range): a JSON
// integer, or a decimal or "0x" string. Anything else is refused rather than wrapped.
std::uint64_t num(const json& o, const char* key, std::uint64_t dflt, const std::string& where,
                  std::uint64_t max = std::numeric_limits<std::uint64_t>::max()) {
    if (!o.contains(key)) return dflt;
    const json& v = o.at(key);
    std::uint64_t x = 0;
    if (v.is_number_unsigned()) {
        x = v.get<std::uint64_t>();
    } else if (v.is_number_integer()) {
        if (v.get<std::int64_t>() < 0) throw UsageError(where + "." + key + " must not be negative");
        x = static_cast<std::uint64_t>(v.get<std::int64_t>());
    } else if (v.is_string()) {
        // stoull accepts leading space, '+' and '-' (wrapping a negative); a number here must
        // start with a digit.
        const std::string s = v.get<std::string>();
        if (s.empty() || !std::isdigit(static_cast<unsigned char>(s[0])))
            throw UsageError(where + "." + key + " is not a non-negative number: " + v.dump());
        try {
            std::size_t used = 0;
            x = std::stoull(s, &used, 0);
            if (used != s.size()) throw std::invalid_argument(s);
        } catch (const std::exception&) {
            throw UsageError(where + "." + key + " is not a number: " + v.dump());
        }
    } else {
        throw UsageError(where + "." + key + " must be a number");
    }
    if (x > max)
        throw UsageError(where + "." + key + " (" + std::to_string(x) + ") is larger than " + std::to_string(max));
    return x;
}
constexpr std::uint64_t kU32 = std::numeric_limits<std::uint32_t>::max();

RequestModel model_of(const json& m, const std::string& where) {
    reject_unknown(m, {"kind", "load", "base", "count", "bytes", "stride", "region", "seed", "rows", "cols",
                       "element_bytes", "tile_rows", "tile_cols", "pitch", "M", "N", "K", "tile"}, where);
    const std::string kind = m.value("kind", "stream");
    const bool load = m.value("load", true);
    if (kind == "replay_matmul") {
        schedule::MatMulScheduleGenerator::Config g;
        g.M = static_cast<Size>(num(m, "M", 128, where, kU32));
        g.N = static_cast<Size>(num(m, "N", g.M, where, kU32));
        g.K = static_cast<Size>(num(m, "K", g.M, where, kU32));
        g.Ti = g.Tj = g.Tk = static_cast<Size>(num(m, "tile", 32, where, kU32));
        const auto s = schedule::MatMulScheduleGenerator(g).generate();
        if (!s.valid) throw UsageError(where + ": the matmul schedule is not valid");
        return RequestModel::replay_of(s, load);
    }
    RequestModel r;
    r.is_load = load;
    r.base = num(m, "base", 0, where);
    r.count = static_cast<std::uint32_t>(num(m, "count", 0, where, kU32));
    r.bytes = static_cast<std::uint32_t>(num(m, "bytes", 4096, where, kU32));
    r.stride = num(m, "stride", 0, where);
    r.region = num(m, "region", 0, where);
    r.seed = num(m, "seed", 1, where);
    r.rows = static_cast<std::uint32_t>(num(m, "rows", 0, where, kU32));
    r.cols = static_cast<std::uint32_t>(num(m, "cols", 0, where, kU32));
    r.element_bytes = static_cast<std::uint32_t>(num(m, "element_bytes", 4, where, kU32));
    r.tile_rows = static_cast<std::uint32_t>(num(m, "tile_rows", 0, where, kU32));
    r.tile_cols = static_cast<std::uint32_t>(num(m, "tile_cols", 0, where, kU32));
    r.pitch = num(m, "pitch", 0, where);
    if (kind == "stream") r.kind = RequestModel::Kind::Stream;
    else if (kind == "strided") r.kind = RequestModel::Kind::Strided;
    else if (kind == "random") {
        r.kind = RequestModel::Kind::Random;
        if (r.bytes == 0) throw UsageError(where + ".bytes must be greater than zero for a random model");
    }
    else if (kind == "matrix_tiles") r.kind = RequestModel::Kind::MatrixTiles;
    else throw UsageError(where + ".kind '" + kind + "' is not one of stream, strided, random, matrix_tiles, "
                                  "replay_matmul");
    return r;
}

MemorySideHarness::Config scenario_of(const json& s, const sw::kpu::program::platform::DeviceSpecification& d) {
    reject_unknown(s, {"window", "store_buffer_blocks", "l3_slots", "max_cycles", "ports", "streams"}, "scenario");
    MemorySideHarness::Config c;
    c.window = num(s, "window", 0, "scenario");
    c.store_buffer_blocks = num(s, "store_buffer_blocks", 0, "scenario");
    c.l3_slots = num(s, "l3_slots", 0, "scenario");
    c.max_cycles = num(s, "max_cycles", c.max_cycles, "scenario");
    if (s.contains("ports")) {
        const json& p = s.at("ports");
        reject_unknown(p, {"infinite", "block_cycles", "input_queue_blocks", "consume_latency", "eject_interval"},
                       "scenario.ports");
        c.ports.infinite = p.value("infinite", false);
        c.ports.block_cycles = num(p, "block_cycles", 0, "scenario.ports");
        c.ports.input_queue_blocks = num(p, "input_queue_blocks", 0, "scenario.ports");
        c.ports.consume_latency = num(p, "consume_latency", 0, "scenario.ports");
        c.ports.eject_interval = num(p, "eject_interval", 1, "scenario.ports");
    }
    // "first" = the lowest port any engine is attached to.
    unsigned first = ~0u;
    {
        MemorySideHarness probe(d, c);
        for (std::size_t e = 0; e < probe.engines(); ++e) first = std::min(first, probe.port_of(e));
    }
    if (!s.contains("streams") || !s.at("streams").is_array() || s.at("streams").empty())
        throw UsageError("scenario.streams must list at least one stream");
    std::size_t i = 0;
    for (const json& st : s.at("streams")) {
        const std::string where = "scenario.streams[" + std::to_string(i++) + "]";
        reject_unknown(st, {"port", "issue_interval", "model"}, where);
        MemorySideHarness::Stream x;
        if (!st.contains("port") || (st.at("port").is_string() && st.at("port").get<std::string>() == "first"))
            x.port = first;
        else
            x.port = static_cast<unsigned>(num(st, "port", 0, where, std::numeric_limits<unsigned>::max()));
        x.issue_interval = num(st, "issue_interval", 0, where);
        if (!st.contains("model")) throw UsageError(where + ".model is required");
        x.model = model_of(st.at("model"), where + ".model");
        c.streams.push_back(std::move(x));
    }
    return c;
}

void usage() {
    std::cerr << "usage: kpu-memsim --deploy <spec.json> [--device N] --scenario <scenario.json> --out <dir>\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string deploy, scenario, out;
    unsigned device = 0;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw UsageError(a + " needs a value");
                return argv[++i];
            };
            if (a == "--deploy") deploy = value();
            else if (a == "--scenario") scenario = value();
            else if (a == "--out") out = value();
            else if (a == "--device") {
                const std::string v = value();
                if (v.empty() || v.size() > 9 || v.find_first_not_of("0123456789") != std::string::npos)
                    throw UsageError("--device must be a device index, not '" + v + "'");
                device = static_cast<unsigned>(std::stoul(v));
            }
            else if (a == "-h" || a == "--help") { usage(); return 0; }
            else throw UsageError("unknown argument '" + a + "'");
        }
        if (deploy.empty() || scenario.empty() || out.empty()) throw UsageError("--deploy, --scenario and --out are required");

        const auto spec = sw::kpu::program::platform::read_spec_file(deploy);
        if (std::string why = spec.validate(); !why.empty()) throw UsageError("deployment: " + why);
        if (device >= spec.device_count()) throw UsageError("--device " + std::to_string(device) + ": the deployment has " +
                                                            std::to_string(spec.device_count()));
        const auto& d = spec.device(device);
        std::ifstream in(scenario);
        if (!in) throw UsageError("cannot read " + scenario);
        json s;
        try {
            s = json::parse(in);        // the whole file: trailing text after the scenario is refused
        } catch (const json::exception& e) {
            throw UsageError(scenario + " is not JSON: " + e.what());
        }
        const auto config = scenario_of(s, d);

        MemorySideHarness h(d, config);
        const bool finished = h.run();
        const auto record = to_record(h, d.name);
        rec::write_mflow(record, out);

        // Summary.
        const auto& m = *d.memory.dram;
        const double ceiling = static_cast<double>(m.channels) * (m.channel_width_bits / 8.0) *
                               m.data_rate_mtps * 1e6 / 1e9 *
                               static_cast<double>(d.memory.controllers.value_or(1));
        const double bpc = h.now() ? static_cast<double>(h.bytes_moved()) / static_cast<double>(h.now()) : 0.0;
        std::size_t outcome[4] = {0, 0, 0, 0}, refusals = 0, waits = 0;
        for (const auto& b : record.bursts) ++outcome[static_cast<int>(b.outcome)];
        for (const auto& p : record.ports) {
            refusals += p.kind == rec::MemoryFlowRecord::PortKind::Refuse;
            waits += p.kind == rec::MemoryFlowRecord::PortKind::EjectWait;
        }
        std::printf("kpu-memsim %s: %zu requests, %llu bytes in %llu cycles: %.1f B/cycle (%.0f%% of the "
                    "%.1f B/cycle DRAM ceiling)\n",
                    d.name.c_str(), record.requests.size(), static_cast<unsigned long long>(h.bytes_moved()),
                    static_cast<unsigned long long>(h.now()), bpc, ceiling > 0 ? 100.0 * bpc / ceiling : 0.0, ceiling);
        std::printf("  window %u; bursts %zu: hit %zu, empty %zu, conflict %zu; commands %zu; refusals %zu; "
                    "ejection waits %zu\n",
                    record.window, record.bursts.size(), outcome[1], outcome[2], outcome[3], record.commands.size(),
                    refusals, waits);
        std::printf("  recorded to %s\n", out.c_str());
        if (!finished) {
            std::fprintf(stderr, "kpu-memsim: the run did not finish within %llu cycles\n",
                         static_cast<unsigned long long>(config.max_cycles));
            return 1;
        }
        return 0;
    } catch (const UsageError& e) {
        std::fprintf(stderr, "kpu-memsim: %s\n", e.what());
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kpu-memsim: %s\n", e.what());
        return 2;
    }
}
