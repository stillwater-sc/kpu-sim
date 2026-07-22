// ============================================================================
// examples/characterize/tile_characterize.cpp
// Design-of-experiments harness for characterizing L0 TilePrograms.
//
// Sweeps (algorithm x size x tile-shape x compute-tiles x topology), for each cell:
//   1. builds the tile program (so you can SEE the tile sequence: --disasm/--trace),
//   2. runs the L0 functional reference and VALIDATES it against an oracle,
//   3. characterizes structural + first-order modeled performance/energy metrics,
// and emits a table (+ CSV/JSON). The compute-tiles sweep is the domain-flow analogue
// of CUDA's occupancy-vs-resources question: how does achievable concurrency (and
// thus makespan/energy) scale with the hardware you give the program?
//
//   tile_characterize --algo lu --sizes 64,128,256 --tiles 16,32
//       --compute-tiles 1,4,16,64 --topology single,checkerboard
//       --csv out.csv --trace first.json --disasm
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/derive/lu_tile_program.hpp>
#include <sw/kpu/program/characterize/characterization.hpp>
#include <sw/kpu/program/stream/derive/matmul_streams.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::characterize;

namespace {

std::vector<std::uint32_t> parse_ints(const std::string& csv) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stoul(tok));
    return out;
}
std::vector<std::string> parse_strs(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(tok);
    return out;
}

std::string arg(const std::vector<std::string>& a, const std::string& k, const std::string& def) {
    for (std::size_t i = 0; i + 1 < a.size(); ++i) if (a[i] == k) return a[i + 1];
    return def;
}
bool has_flag(const std::vector<std::string>& a, const std::string& k) {
    for (const auto& s : a) if (s == k) return true;
    return false;
}

// ---- build + validate each algorithm ---------------------------------------
float validate_matmul(TileProgram& prog, Dim M, Dim N, Dim K) {
    auto& A = prog.operand("A"); auto& B = prog.operand("B");
    for (std::size_t i = 0; i < A.values.size(); ++i) A.values[i] = float((i * 7 + 1) % 13) - 6.0f;
    for (std::size_t i = 0; i < B.values.size(); ++i) B.values[i] = float((i * 5 + 2) % 11) - 5.0f;
    TileProgramReference ref; ref.run(prog);
    // naive oracle
    const auto& Av = A.values; const auto& Bv = B.values;
    float max_err = 0.0f;
    const auto& C = prog.operand("C").values;
    for (Dim i = 0; i < M; ++i)
        for (Dim j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (Dim k = 0; k < K; ++k)
                acc += Av[std::size_t(i) * K + k] * Bv[std::size_t(k) * N + j];
            max_err = std::max(max_err, std::fabs(acc - C[std::size_t(i) * N + j]));
        }
    return max_err;
}

float validate_lu(TileProgram& prog, Dim N) {
    auto& A = prog.operand("A");
    std::vector<float> A0(std::size_t(N) * N);
    for (Dim i = 0; i < N; ++i)
        for (Dim j = 0; j < N; ++j)
            A0[std::size_t(i) * N + j] =
                (i == j) ? 4.0f + float((i * 3) % 5)
                         : 1.0f / (1.0f + std::fabs(float(int(i) - int(j))));
    if (N > 1) A0[1 * std::size_t(N) + 0] = 7.0f;   // force a within-tile swap
    A.values = A0;

    TileProgramReference ref;
    auto sum = ref.run(prog);
    const auto& F = prog.operand("A").values;
    auto idx = [N](Dim i, Dim j) { return std::size_t(i) * N + j; };
    float max_err = 0.0f;
    for (Dim i = 0; i < N; ++i)
        for (Dim j = 0; j < N; ++j) {
            float lu = 0.0f;
            for (Dim k = 0; k < N; ++k) {
                float l = (i > k) ? F[idx(i, k)] : (i == k ? 1.0f : 0.0f);
                float u = (k <= j) ? F[idx(k, j)] : 0.0f;
                lu += l * u;
            }
            float pa = A0[idx(sum.permutation[i], j)];
            max_err = std::max(max_err, std::fabs(lu - pa));
        }
    return max_err;
}

DeviceDescriptor make_device(const std::string& topo, Dim cf,
                             double macs_per_cycle, double bytes_per_cycle,
                             double pj_mac, double pj_byte, Dim l3_tiles) {
    DeviceDescriptor d = DeviceDescriptor::single();
    if (topo == "news") d = DeviceDescriptor::news();
    else if (topo == "checkerboard") d = DeviceDescriptor::checkerboard(cf);
    d.compute_tiles = cf;
    if (topo == "single") d.move_lanes = 1;
    else if (topo == "news") d.move_lanes = 4;
    else d.move_lanes = cf;
    d.fabric_macs_per_cycle = macs_per_cycle;
    d.bytes_per_cycle = bytes_per_cycle;
    d.pj_per_mac = pj_mac;
    d.pj_per_byte = pj_byte;
    d.l3_tiles = l3_tiles;
    return d;
}

bool known_dataflow(const std::string& n) {
    return n == "output-stationary" || n == "os" ||
           n == "weight-stationary" || n == "ws" ||
           n == "a-stationary"      || n == "as" ||
           n == "fully-streaming"   || n == "hex";
}

// dataflow name (or alias) -> space-time mapping (caller must pre-validate via known_dataflow)
stream::SpaceTimeMap map_for(const std::string& name) {
    if (name == "weight-stationary" || name == "ws") return stream::SpaceTimeMap::b_stationary();
    if (name == "a-stationary"      || name == "as") return stream::SpaceTimeMap::a_stationary();
    if (name == "fully-streaming"   || name == "hex") return stream::SpaceTimeMap::fully_streaming();
    return stream::SpaceTimeMap::output_stationary();   // "output-stationary" / "os"
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    if (has_flag(a, "--help") || has_flag(a, "-h")) {
        std::cout <<
            "tile_characterize — DoE harness for L0 tile programs\n"
            "  --algo matmul|lu           (default matmul)\n"
            "  --sizes N[,N...]           square problem size(s) (default 128)\n"
            "  --tiles T[,T...]           tile dimension(s)      (default 32)\n"
            "  --compute-tiles C[,C...]   CF tile count(s)       (default 1,4,16)\n"
            "  --topology single|news|checkerboard[,...] (default single)\n"
            "  --dataflow output-stationary|weight-stationary|a-stationary|fully-streaming[,...]\n"
            "             (matmul only; aliases os/ws/as/hex; systolic L1 timing + network)\n"
            "  --macs-per-cycle F  --bytes-per-cycle F  --pj-per-mac F  --pj-per-byte F\n"
            "  --l3-tiles N               L3 capacity in tiles for feasibility (0=unbounded)\n"
            "  --csv FILE   --json FILE   --trace FILE (first cell)  --disasm  --no-validate\n";
        return 0;
    }

    const std::string algo = arg(a, "--algo", "matmul");
    const auto sizes = parse_ints(arg(a, "--sizes", "128"));
    const auto tiles = parse_ints(arg(a, "--tiles", "32"));
    const auto cfs   = parse_ints(arg(a, "--compute-tiles", "1,4,16"));
    const auto topos = parse_strs(arg(a, "--topology", "single"));
    const auto dataflows = parse_strs(arg(a, "--dataflow", "output-stationary"));  // matmul only
    const double macs_pc  = std::stod(arg(a, "--macs-per-cycle", "256"));
    const double bytes_pc = std::stod(arg(a, "--bytes-per-cycle", "64"));
    const double pj_mac   = std::stod(arg(a, "--pj-per-mac", "1.0"));
    const double pj_byte  = std::stod(arg(a, "--pj-per-byte", "20.0"));
    const Dim l3_tiles    = std::stoul(arg(a, "--l3-tiles", "0"));
    const bool validate = !has_flag(a, "--no-validate");
    const std::string csv_path = arg(a, "--csv", "");
    const std::string json_path = arg(a, "--json", "");
    const std::string trace_path = arg(a, "--trace", "");
    const bool disasm = has_flag(a, "--disasm");

    // dataflow sweep applies to matmul (L1 systolic timing); LU has no stream deriver.
    if (algo != "lu")
        for (const auto& df : dataflows)
            if (!known_dataflow(df)) {
                std::cerr << "error: unknown --dataflow '" << df << "' "
                             "(expected output-stationary|weight-stationary|a-stationary|"
                             "fully-streaming, or os/ws/as/hex)\n";
                return 2;
            }
    const std::vector<std::string> dfs = (algo == "lu")
        ? std::vector<std::string>{"-"} : dataflows;

    std::ostringstream csv, json;
    csv << "algo,size,tile,compute_tiles,topology,dataflow,stationary,c_bubble,network,func_max_err,"
        << metrics_csv_header() << "\n";
    json << "[\n";

    std::cout << "algo   size tile  CF topology    dataflow          stat bub network            "
                 "makespan  cmp_util  bound  energy_pJ    err\n";
    std::cout << "------------------------------------------------------------------------------------"
                 "-------------------------------------\n";

    bool did_trace = false, did_disasm = false, first_json = true;
    for (const auto& topo : topos)
        for (Dim size : sizes)
            for (Dim tile : tiles) {
                if (tile > size) continue;
                for (Dim cf : cfs) {
                    TileProgram prog =
                        (algo == "lu") ? derive_lu_tile_program(size, tile)
                                       : derive_matmul_tile_program(size, size, size, tile, tile, tile);
                    float err = -1.0f;
                    if (validate)
                        err = (algo == "lu") ? validate_lu(prog, size)
                                             : validate_matmul(prog, size, size, size);

                    DeviceDescriptor dev = make_device(topo, cf, macs_pc, bytes_pc, pj_mac, pj_byte, l3_tiles);

                    for (const auto& df : dfs) {
                        stream::StreamProgram l1;
                        const stream::StreamProgram* l1p = nullptr;
                        if (algo != "lu") { l1 = stream::derive_matmul_streams(prog, map_for(df)); l1p = &l1; }
                        Metrics m = characterize_program(prog, dev, l1p);

                        if (disasm && !did_disasm) { std::cout << "\n" << prog.disassemble() << "\n"; did_disasm = true; }
                        if (!trace_path.empty() && !did_trace) {
                            write_chrome_trace(prog, dev, trace_path, l1p);
                            std::cout << "[trace] wrote " << trace_path << " (chrome://tracing)\n";
                            did_trace = true;
                        }

                        char line[320];
                        std::snprintf(line, sizeof(line),
                            "%-6s %4u %4u %3u %-10s %-17s %-4s %3d %-18s %9.1f %8.2f %5s %10.3g %8.1g\n",
                            algo.c_str(), size, tile, cf, topo.c_str(), df.c_str(),
                            m.stationary.empty() ? "-" : m.stationary.c_str(), m.c_bubble,
                            m.network.empty() ? "-" : m.network.c_str(),
                            m.makespan_cycles, m.compute_util, m.compute_bound ? "cmp" : "mov",
                            m.energy_total_pj, err);
                        std::cout << line;

                        csv << algo << ',' << size << ',' << tile << ',' << cf << ',' << topo << ','
                            << df << ',' << (m.stationary.empty() ? "-" : m.stationary) << ','
                            << m.c_bubble << ',' << (m.network.empty() ? "-" : m.network) << ','
                            << err << ',' << metrics_csv_row(m) << "\n";
                        if (!first_json) json << ",\n";
                        first_json = false;
                        json << "  {\"algo\":\"" << algo << "\",\"size\":" << size << ",\"tile\":" << tile
                             << ",\"compute_tiles\":" << cf << ",\"topology\":\"" << topo
                             << "\",\"dataflow\":\"" << df << "\",\"stationary\":\"" << m.stationary
                             << "\",\"c_bubble\":" << m.c_bubble << ",\"network\":\"" << m.network
                             << "\",\"func_max_err\":" << err
                             << ",\"macs\":" << m.total_macs << ",\"arith_intensity\":" << m.arithmetic_intensity
                             << ",\"critical_path\":" << m.critical_path_cycles
                             << ",\"makespan\":" << m.makespan_cycles
                             << ",\"lower_bound\":" << m.lower_bound_cycles
                             << ",\"compute_util\":" << m.compute_util
                             << ",\"movement_util\":" << m.movement_util
                             << ",\"compute_bound\":" << (m.compute_bound ? "true" : "false")
                             << ",\"peak_live_tiles\":" << m.peak_live_tiles
                             << ",\"energy_total_pj\":" << m.energy_total_pj
                             << ",\"feasible\":" << (m.feasible ? "true" : "false") << "}";
                    }
                }
            }
    json << "\n]\n";

    if (!csv_path.empty()) { std::ofstream(csv_path) << csv.str(); std::cout << "[csv] wrote " << csv_path << "\n"; }
    if (!json_path.empty()) { std::ofstream(json_path) << json.str(); std::cout << "[json] wrote " << json_path << "\n"; }
    return 0;
}
