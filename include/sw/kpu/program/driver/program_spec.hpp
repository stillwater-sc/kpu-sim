// ============================================================================
// include/sw/kpu/program/driver/program_spec.hpp
// Argument -> program and argument -> device, shared by every tool.
//
// Design note §D1: the driver and the characterization harness must SHARE this
// mapping rather than each parsing its own --sizes/--tiles and building its own
// DeviceDescriptor. Two copies drift, and then a bug reproduces in one tool and
// not the other — which is the expensive kind.
//
// No derivation logic lives here: it calls derive_*_tile_program and the
// DeviceDescriptor presets. This header is where the NAMES agree.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/characterize/device_model.hpp>
#include <sw/kpu/program/derive/lu_tile_program.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/stream/derive/matmul_streams.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <cmath>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::driver {

// ---- argument helpers ------------------------------------------------------
// THROWS rather than wrapping. std::stoul turns "-2" into an enormous count without
// complaining, and an uncaught throw on "abc" is SIGABRT -- which a CI job cannot tell
// apart from a crash in the model. A sweep list is command-line text, so it gets the same
// treatment as every other flag here.
inline std::vector<std::uint32_t> parse_ints(const std::string& csv) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        if (tok[0] == '-')
            throw std::invalid_argument("'" + tok + "' is not a non-negative integer");
        // ONLY THE stoul CALL IS INSIDE THE try. With the checks in there too, the catch for
        // std::invalid_argument caught this function's OWN diagnostics and relabelled them:
        // "12abc" reported "is not an integer" instead of "has trailing characters", and an
        // out-of-range value reported the same. The exit code was right and the message sent
        // the reader to the wrong problem, which is the more expensive half.
        std::size_t consumed = 0;
        unsigned long v = 0;
        try {
            v = std::stoul(tok, &consumed);
        } catch (const std::out_of_range&) {
            throw std::invalid_argument("'" + tok + "' is out of range");
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument("'" + tok + "' is not an integer");
        }
        if (consumed != tok.size())
            throw std::invalid_argument("'" + tok + "' has trailing characters");
        if (v > 0xFFFFFFFFul)
            throw std::invalid_argument("'" + tok + "' is out of range");
        out.push_back(static_cast<std::uint32_t>(v));
    }
    return out;
}

inline std::vector<std::string> parse_strs(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(tok);
    return out;
}

inline std::string arg(const std::vector<std::string>& a, const std::string& k,
                       const std::string& def) {
    for (std::size_t i = 0; i + 1 < a.size(); ++i) if (a[i] == k) return a[i + 1];
    return def;
}

inline bool has_flag(const std::vector<std::string>& a, const std::string& k) {
    for (const auto& s : a) if (s == k) return true;
    return false;
}

// Is the option PRESENT at all? `arg()` cannot say: it returns the fallback both when an
// option is absent and when it is the last token with no value, so a terminal `--timeline`
// looks exactly like no `--timeline`. For a flag that must carry a value, those are
// different errors and only one of them is silent.
inline bool arg_present(const std::vector<std::string>& a, const std::string& k) {
    for (const auto& s : a) if (s == k) return true;
    return false;
}

// A required value: present, non-empty, and not another option. Without the last check a
// trailing `--timeline --step` would write a trace to a file called "--step".
inline bool arg_required(const std::vector<std::string>& a, const std::string& k,
                         std::string& out, std::string& error) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != k) continue;
        if (i + 1 >= a.size()) { error = k + ": missing value"; return false; }
        if (!a[i + 1].empty() && a[i + 1][0] == '-') {
            error = k + ": missing value (next token is '" + a[i + 1] + "')";
            return false;
        }
        out = a[i + 1];
        return true;
    }
    return true;                 // absent: leave `out` at its default
}

// A CHECKED unsigned parse, because std::stoul is the wrong tool for a CLI: it throws on
// "abc", it happily accepts "12abc", and it wraps "-1" to ULONG_MAX without complaint --
// so `--compute-tiles -1` becomes an enormous count rather than an error. An uncaught
// throw also aborts with SIGABRT, which a CI job cannot tell apart from a crash in the
// model. Returns false and fills `error` instead.
//
// It also catches `--size --tile 16`, where arg() hands back the NEXT FLAG as the value.
inline bool parse_dim(const std::vector<std::string>& a, const std::string& key,
                      std::uint32_t fallback, std::uint32_t& out, std::string& error) {
    std::string raw = std::to_string(fallback);
    if (!arg_required(a, key, raw, error)) return false;
    if (raw.empty()) { error = key + ": empty value"; return false; }
    if (raw[0] == '-') {
        error = key + ": '" + raw + "' is not a non-negative integer";
        return false;
    }
    try {
        std::size_t consumed = 0;
        const unsigned long v = std::stoul(raw, &consumed);
        if (consumed != raw.size()) {
            error = key + ": '" + raw + "' has trailing characters";
            return false;
        }
        if (v > 0xFFFFFFFFul) { error = key + ": '" + raw + "' is out of range"; return false; }
        out = static_cast<std::uint32_t>(v);
        return true;
    } catch (const std::exception&) {
        error = key + ": '" + raw + "' is not an integer";
        return false;
    }
}

// A CHECKED double, for the same reasons as parse_dim and with one extra: std::stod
// PARSES "inf", and an infinite bandwidth is not a fast machine -- it is a makespan of 0 or
// a NaN reported as a result. Shared by both tools, because two copies of "what
// --macs-per-cycle accepts" drift and then a bug reproduces in one tool and not the other
// (§D1).
//
// `require_positive` distinguishes a RATE from a COEFFICIENT: a bandwidth of zero is not a
// machine, but zero energy per MAC is a legitimate modelling choice ("ignore compute
// energy").
inline bool parse_double(const std::vector<std::string>& a, const std::string& key,
                         double fallback, bool require_positive, double& out,
                         std::string& error) {
    if (!arg_present(a, key)) { out = fallback; return true; }
    std::string raw;
    if (!arg_required(a, key, raw, error)) return false;   // terminal, or a flag as value
    if (raw.empty()) { error = key + ": empty value"; return false; }
    try {
        std::size_t consumed = 0;
        const double v = std::stod(raw, &consumed);
        if (consumed != raw.size()) {
            error = key + ": '" + raw + "' has trailing characters";
            return false;
        }
        if (!std::isfinite(v) || (require_positive ? !(v > 0.0) : !(v >= 0.0))) {
            error = key + ": '" + raw + "' must be finite and " +
                    (require_positive ? "positive" : "non-negative");
            return false;
        }
        out = v;
        return true;
    } catch (const std::exception&) {
        error = key + ": '" + raw + "' is not a number";
        return false;
    }
}

// ---- what program to run ---------------------------------------------------
struct ProgramSpec {
    std::string algo = "matmul";     // "matmul" | "lu"
    Dim size = 64;                   // square: M = N = K = size, or N for LU
    Dim tile = 16;

    std::string label() const {
        return algo + "/" + std::to_string(size) + "^3/t" + std::to_string(tile);
    }
};

inline bool known_algo(const std::string& a) { return a == "matmul" || a == "lu"; }

inline TileProgram derive(const ProgramSpec& s) {
    if (s.algo == "lu") return derive_lu_tile_program(s.size, s.tile);
    if (s.algo == "matmul")
        return derive_matmul_tile_program(s.size, s.size, s.size, s.tile, s.tile, s.tile);
    throw std::invalid_argument("unknown --algo '" + s.algo + "' (matmul | lu)");
}

// Deterministic, non-trivial operand values. Identical for every level, which is the
// precondition for comparing their results at all: a run whose inputs differ tells you
// nothing about the models.
//
// EVERY VALUE HERE IS EXACTLY REPRESENTABLE, and that is a requirement rather than a
// coincidence. This project builds Release with `-march=native`, so the compiler may
// contract `a - b * c` into an FMA -- a single rounding instead of two -- and the result
// then depends on the HOST CPU. If the fill itself needs rounding, the input values differ
// between machines, and no checked-in golden file can be reproduced elsewhere.
//
// That is not hypothetical: the LU fill used `* 0.1f`, which left 3129 of its off-diagonal
// values inexact, and CI failed on the LU corpus INPUT -- not its output. With a power of
// two the product is exact, so single and double rounding agree because neither rounds.
inline void fill(TileProgram& p, const ProgramSpec& s) {
    if (s.algo == "lu") {
        auto& A = p.operand("A");
        for (Dim i = 0; i < s.size; ++i)
            for (Dim j = 0; j < s.size; ++j)
                // Diagonally dominant, for a stable factorisation, and every term a
                // multiple of 1/8.
                A.at(i, j) = (i == j) ? 4.0f + float((i * 3) % 5)
                                      : 0.5f - float((i * 7 + j * 3) % 9) * 0.125f;
        return;
    }
    auto& A = p.operand("A");
    auto& B = p.operand("B");
    for (std::size_t i = 0; i < A.values.size(); ++i)
        A.values[i] = float((i * 7 + 1) % 13) - 6.0f + 0.25f * float(i % 3);
    for (std::size_t i = 0; i < B.values.size(); ++i)
        B.values[i] = float((i * 5 + 2) % 11) - 5.0f - 0.125f * float(i % 5);
}

// ---- values for a program nobody derived --------------------------------------
// A loaded L0 program has no ProgramSpec behind it, so fill() -- which knows "A" and "B" by
// name -- cannot be used. These two work from the OP LIST instead.

// The operands the program READS. An operand that appears in some op's `inputs` needs a
// value; one that appears only in `outputs` is produced, and pre-filling it would make a
// value comparison compare the fill rather than the model.
//
// The rule is "read AT ALL", not "read before it is written", and the difference is not
// cosmetic. Tile LU factors A IN PLACE and its FIRST op (LuDiagFactor) declares A[k,k] as an
// OUTPUT, so a read-before-written rule marks A as produced and fills NOTHING -- leaving the
// factorisation to run on a zero matrix and report success. An in-place operand is both read
// and written, and it still needs an input.
inline std::vector<std::string> program_inputs(const TileProgram& p) {
    std::vector<std::string> order;
    std::set<std::string> seen;
    for (const TileOp& op : p.ops())
        for (const TileCoord& c : op.inputs)
            if (seen.insert(c.operand).second) order.push_back(c.operand);
    return order;
}

// Deterministic values for every operand the program reads, derived from the OPERAND NAME
// and the element position -- so two operands never get the same pattern, and the same file
// fills identically on every machine and at every level.
//
// EVERY VALUE IS EXACTLY REPRESENTABLE, for the reason fill() states at length: Release
// builds with `-march=native`, so a value that needed rounding would differ between hosts
// and no run of a checked-in file could be reproduced elsewhere.
//
// NO NUMERICAL-STABILITY PROMISE IS MADE, and that is worth saying rather than hoping. A
// square operand gets a dominant diagonal, which keeps a factorisation well behaved in
// practice, but nothing here can guarantee it: a program whose inputs need structure (a
// specific conditioning, a symmetry, a sparsity pattern) should CARRY ITS VALUES rather than
// have them invented. That is what `VALUES inline` is for.
inline void fill_inputs(TileProgram& p) {
    for (const std::string& name : program_inputs(p)) {
        TensorOperand& t = p.operand(name);
        std::uint32_t h = 2166136261u;                      // FNV-1a over the operand name
        for (char ch : name) {
            h ^= static_cast<std::uint32_t>(static_cast<unsigned char>(ch));
            h *= 16777619u;
        }
        const bool square = (t.rows == t.cols);
        for (Dim r = 0; r < t.rows; ++r)
            for (Dim c = 0; c < t.cols; ++c) {
                // MIXED, not a linear combination of r and c. `h + 131*r + 17*c` looks
                // adequate and is not: 17*c vanishes mod 17, so the integer part of a value
                // was CONSTANT ALONG EACH ROW and only the eighths varied -- every row spanned
                // a range of 1.0 with eight distinct values. Near-degenerate inputs weaken
                // exactly what this fill is for: a level that transposed an index, or read a
                // neighbouring element, would still produce a nearly identical answer.
                std::uint32_t k = h;
                k ^= r * 2654435761u; k *= 2246822519u;
                k ^= c * 3266489917u; k *= 668265263u;
                k ^= k >> 15;
                // Integers in [-8, 8] plus a multiple of 1/8: no rounding anywhere.
                float v = float(k % 17) - 8.0f + 0.125f * float((k >> 8) % 8);
                // A dominant diagonal for a square operand, which is what keeps an in-place
                // factorisation from pivoting on noise. It is a nudge, not a guarantee (see
                // above): off-diagonal row sums grow with the operand and this term does not.
                if (square && r == c) v += 32.0f;
                t.at(r, c) = v;
            }
    }
}

// The operand a run's result lands in, which is what a comparison reads.
inline const char* result_operand(const ProgramSpec& s) {
    return s.algo == "lu" ? "A" : "C";      // LU factors in place
}

// ---- the L1 stream program (optional) ---------------------------------------
// The dataflow name -> space-time mapping, shared with tile_characterize for the same
// reason as everything else here: two spellings of "output-stationary" drift.
// Accepts the CLI aliases AND each map's OWN name, so map_for(m.name).name == m.name for
// every preset. That round-trip is load-bearing: a serialized program records the map's own
// name, and a reader passing it back through map_for() must get the same map. Without the
// canonical names here, map_for("weight(B)-stationary") fell through to the default and
// returned OUTPUT-stationary -- silently the wrong dataflow, which is exactly the failure
// the serializer refuses unknown names to avoid.
inline bool known_dataflow(const std::string& n) {
    return n == "output-stationary" || n == "os" ||
           n == "weight-stationary" || n == "ws" || n == "weight(B)-stationary" ||
           n == "a-stationary"      || n == "as" || n == "A-stationary" ||
           n == "fully-streaming"   || n == "hex" || n == "fully-streaming(hex)";
}

inline stream::SpaceTimeMap map_for(const std::string& name) {
    if (name == "weight-stationary" || name == "ws" || name == "weight(B)-stationary")
        return stream::SpaceTimeMap::b_stationary();
    if (name == "a-stationary" || name == "as" || name == "A-stationary")
        return stream::SpaceTimeMap::a_stationary();
    if (name == "fully-streaming" || name == "hex" || name == "fully-streaming(hex)")
        return stream::SpaceTimeMap::fully_streaming();
    return stream::SpaceTimeMap::output_stationary();   // "output-stationary" / "os"
}

// ---- what device to run it on ----------------------------------------------
// Movement is per CSP process (design note §6): DMA (DRAM<->L3), BlockMover
// (L3<->L2) and Streamer (L2<->L1) each own their lanes. There is no aggregate
// movement pool and no collapsed hop.
struct DeviceSpec {
    std::string topology = "single";        // single | news | checkerboard
    Dim compute_tiles = 1;
    double macs_per_cycle = 256.0;
    Dim l3_tiles = 0;                       // 0 = unbounded

    Dim dma_engines = 1;
    double dma_bytes_per_cycle = 64.0;
    Dim block_movers = 1;
    double bm_bytes_per_cycle = 128.0;
    Dim streamers = 1;
    double str_bytes_per_cycle = 256.0;
    Dim noc_links = 0;                      // 0 = topology declares no L3<->L3 path
    double noc_bytes_per_cycle = 128.0;

    // Analytical-harness coefficients; the executor does not use these.
    double bytes_per_cycle = 64.0;
    double pj_per_mac = 1.0;
    double pj_per_byte = 20.0;
};

// One spelling of the topology names, in the layer that owns the machine description.
// Two copies drifted apart once already, which is the whole argument of this header.
inline bool known_topology(const std::string& t) { return platform::known_topology_name(t); }

// ---- the CLI builds a DEPLOYMENT, not a descriptor --------------------------
// The flags are unchanged; what they produce is now the one machine description
// (platform/deployment_spec.hpp), and the DeviceDescriptor the executors schedule on is
// a VIEW of it. Building the descriptor directly here is what made the CLI a fifth
// description of a device, and a fifth description is a fifth thing to disagree.
//
// DeviceSpec carries only what a flag can say. The §3.3 resource fields (L3 banks, L2
// banks per compute tile, L1 vectors, DMA burst) are left ABSENT rather than defaulted,
// because absent means "not declared" and a default is not a declaration -- see the
// spec header. `--deploy` (increment 4) is how those get set.
inline platform::DeploymentSpec make_deployment(const DeviceSpec& s) {
    if (!known_topology(s.topology))
        throw std::invalid_argument("unknown --topology '" + s.topology + "'");
    platform::DeviceSpecification d;
    d.name = "dev0";
    d.topology = s.topology;
    d.compute_tiles = s.compute_tiles;
    d.macs_per_cycle = s.macs_per_cycle;
    // `--l3-tiles` is a CAPACITY in tiles, which is what the credit model bounds -- not
    // the number of L3 modules. The spec keeps those apart on purpose.
    d.l3.capacity_tiles = s.l3_tiles;
    d.dma.engines = s.dma_engines;
    d.dma.bytes_per_cycle = s.dma_bytes_per_cycle;
    d.movers.block_movers = s.block_movers;
    d.movers.bm_bytes_per_cycle = s.bm_bytes_per_cycle;
    d.movers.streamers = s.streamers;
    d.movers.str_bytes_per_cycle = s.str_bytes_per_cycle;
    d.movers.noc_links = s.noc_links;
    d.movers.noc_bytes_per_cycle = s.noc_bytes_per_cycle;
    d.analytical.bytes_per_cycle = s.bytes_per_cycle;
    d.analytical.pj_per_mac = s.pj_per_mac;
    d.analytical.pj_per_byte = s.pj_per_byte;

    platform::DeploymentSpec spec;
    spec.devices = {d};
    // The flags can still describe an impossible machine (`--compute-tiles 0`), and the
    // spec is where that is caught -- once, rather than once per tool.
    const std::string bad = spec.validate();
    if (!bad.empty()) throw std::invalid_argument(bad);
    return spec;
}

// Kept as the one-line projection, because every existing caller wants the descriptor
// and should not have to learn about deployments to get one.
inline characterize::DeviceDescriptor make_device(const DeviceSpec& s) {
    return make_deployment(s).device_view();
}

} // namespace sw::kpu::program::driver
