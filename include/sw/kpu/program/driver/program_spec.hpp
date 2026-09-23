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
#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/stream/derive/matmul_streams.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::driver {

// ---- argument helpers ------------------------------------------------------
inline std::vector<std::uint32_t> parse_ints(const std::string& csv) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stoul(tok));
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

// Deterministic, non-trivial operand values. Identical for every level, which is
// the precondition for comparing their results at all: a run whose inputs differ
// tells you nothing about the models.
inline void fill(TileProgram& p, const ProgramSpec& s) {
    if (s.algo == "lu") {
        auto& A = p.operand("A");
        for (Dim i = 0; i < s.size; ++i)
            for (Dim j = 0; j < s.size; ++j)
                A.at(i, j) = (i == j) ? 4.0f + float((i * 3) % 5)
                                      : 0.5f - float((i * 7 + j * 3) % 9) * 0.1f;
        return;
    }
    auto& A = p.operand("A");
    auto& B = p.operand("B");
    for (std::size_t i = 0; i < A.values.size(); ++i)
        A.values[i] = float((i * 7 + 1) % 13) - 6.0f + 0.25f * float(i % 3);
    for (std::size_t i = 0; i < B.values.size(); ++i)
        B.values[i] = float((i * 5 + 2) % 11) - 5.0f - 0.125f * float(i % 5);
}

// The operand a run's result lands in, which is what a comparison reads.
inline const char* result_operand(const ProgramSpec& s) {
    return s.algo == "lu" ? "A" : "C";      // LU factors in place
}

// ---- the L1 stream program (optional) ---------------------------------------
// The dataflow name -> space-time mapping, shared with tile_characterize for the same
// reason as everything else here: two spellings of "output-stationary" drift.
inline bool known_dataflow(const std::string& n) {
    return n == "output-stationary" || n == "os" ||
           n == "weight-stationary" || n == "ws" ||
           n == "a-stationary"      || n == "as" ||
           n == "fully-streaming"   || n == "hex";
}

inline stream::SpaceTimeMap map_for(const std::string& name) {
    if (name == "weight-stationary" || name == "ws") return stream::SpaceTimeMap::b_stationary();
    if (name == "a-stationary"      || name == "as") return stream::SpaceTimeMap::a_stationary();
    if (name == "fully-streaming"   || name == "hex") return stream::SpaceTimeMap::fully_streaming();
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

inline bool known_topology(const std::string& t) {
    return t == "single" || t == "news" || t == "checkerboard";
}

inline characterize::DeviceDescriptor make_device(const DeviceSpec& s) {
    using characterize::DeviceDescriptor;
    DeviceDescriptor d = DeviceDescriptor::single();
    if (s.topology == "news") d = DeviceDescriptor::news();
    else if (s.topology == "checkerboard") d = DeviceDescriptor::checkerboard(s.compute_tiles);
    else if (!known_topology(s.topology))
        throw std::invalid_argument("unknown --topology '" + s.topology + "'");

    d.compute_tiles = s.compute_tiles;
    // Aggregate lanes, for the analytical harness only (see DeviceDescriptor).
    if (s.topology == "single") d.move_lanes = 1;
    else if (s.topology == "news") d.move_lanes = 4;
    else d.move_lanes = s.compute_tiles;

    d.fabric_macs_per_cycle = s.macs_per_cycle;
    d.bytes_per_cycle = s.bytes_per_cycle;
    d.pj_per_mac = s.pj_per_mac;
    d.pj_per_byte = s.pj_per_byte;
    d.l3_tiles = s.l3_tiles;

    d.dma_engines = s.dma_engines;
    d.dma_bytes_per_cycle = s.dma_bytes_per_cycle;
    d.block_movers = s.block_movers;
    d.bm_bytes_per_cycle = s.bm_bytes_per_cycle;
    d.streamers = s.streamers;
    d.str_bytes_per_cycle = s.str_bytes_per_cycle;
    d.noc_links = s.noc_links;
    d.noc_bytes_per_cycle = s.noc_bytes_per_cycle;
    return d;
}

} // namespace sw::kpu::program::driver
