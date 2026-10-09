// ============================================================================
// include/sw/kpu/program/csp/lang/emit.hpp
// L0 -> .csp: a derived program, written out in the language (docs/plans/csp-language.md §2).
//
// The emitter is the language's compiler-side producer: it walks L0 in order and writes the
// residency decisions as statements -- `resident` when a call needs a tile that is not in L3,
// `release` after its last use, and, when L3 is full, the resident tile used furthest ahead
// (Belady) stored if written and released. Unlike the action-level lowering (csp/lower.hpp),
// it decides at CALL granularity: a call's operands are resident together, as the language
// requires. The text it returns compiles (compile.hpp) and can be read, edited and rerun.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/lang/print.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

class EmitError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {

inline std::vector<TileCoord> unique_tiles(std::vector<TileCoord> v) {
    std::vector<TileCoord> out;
    for (const auto& t : v)
        if (std::none_of(out.begin(), out.end(), [&](const TileCoord& x) { return x.to_string() == t.to_string(); }))
            out.push_back(t);
    return out;
}

}  // namespace detail

// Write `l0` as a level-1 .csp program for an L3 of `l3_capacity` tiles.
inline std::string emit(const TileProgram& l0, std::size_t l3_capacity, const std::string& name = "") {
    if (l3_capacity == 0) throw EmitError("csp emit: the L3 capacity must be positive");
    const auto& ops = l0.ops();
    bool explicit_style = false;
    for (const auto& op : ops)
        if (op.kind == TileOpKind::Feed || op.kind == TileOpKind::Drain) explicit_style = true;

    // Which compute ops need each tile in L3 (an accumulator's result does not: it lives in the
    // fabric until its Drain).
    auto operands = [&](const TileOp& op) {
        std::vector<TileCoord> v = op.inputs;
        if (!explicit_style) v.insert(v.end(), op.outputs.begin(), op.outputs.end());
        return detail::unique_tiles(v);
    };
    std::map<std::string, std::vector<std::size_t>> uses;
    std::map<std::string, TileCoord> coord;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (ops[i].kind == TileOpKind::Feed || ops[i].kind == TileOpKind::Drain) continue;
        for (const auto& t : operands(ops[i])) {
            uses[t.to_string()].push_back(i);
            coord.emplace(t.to_string(), t);
        }
    }
    auto next_use = [&](const std::string& k, std::size_t after) {
        const auto& u = uses.at(k);
        auto it = std::upper_bound(u.begin(), u.end(), after);
        return it == u.end() ? std::numeric_limits<std::size_t>::max() : *it;
    };

    std::ostringstream o;
    o << "csp " << kVersion << "\n";
    o << "program " << (name.empty() ? std::string("emitted") : name)   // an identifier; L0 names hold spaces
      << " machine flat(l3 = " << l3_capacity << ") {\n";
    std::set<std::string> written, read;
    for (const auto& op : ops) {
        for (const auto& t : op.outputs) written.insert(t.operand);
        for (const auto& t : op.inputs) read.insert(t.operand);
    }
    for (const auto& n : l0.operand_order()) {
        const TensorOperand& t = l0.operand(n);
        const bool in = read.count(n) != 0, out = written.count(n) != 0;
        o << "  tensor " << n << "[" << t.rows << "," << t.cols << "] tile " << t.tile_rows << "x" << t.tile_cols << " "
          << (in && out ? "inout" : out ? "out" : "in") << ";\n";
    }

    std::map<std::string, bool> resident;                    // tile -> written since loaded
    std::vector<std::string> open;                           // open accumulators
    auto indent = [&]() { return std::string(2 + 2 * open.size(), ' '); };
    auto leave = [&](const std::string& k) {
        if (resident.at(k)) o << indent() << "store " << tile_text(coord.at(k)) << ";\n";
        o << indent() << "release " << tile_text(coord.at(k)) << ";\n";
        resident.erase(k);
    };
    auto make_room = [&](std::size_t at, std::size_t need, const std::set<std::string>& keep) {
        while (resident.size() + need > l3_capacity) {
            std::string victim;
            std::size_t furthest = 0;
            bool found = false;
            for (const auto& [k, w] : resident) {
                if (keep.count(k)) continue;
                const std::size_t n = next_use(k, at);
                if (!found || n > furthest) { victim = k; furthest = n; found = true; }
            }
            if (!found)
                throw EmitError("csp emit: L0 op " + std::to_string(at) + " needs " + std::to_string(need + resident.size()) +
                                " tiles in L3 at once; the L3 holds " + std::to_string(l3_capacity));
            leave(victim);
        }
    };

    for (std::size_t i = 0; i < ops.size(); ++i) {
        const TileOp& op = ops[i];
        if (op.kind == TileOpKind::Feed) continue;           // a call's deliveries are implied
        if (op.kind == TileOpKind::Drain) {
            const std::string k = op.outputs.at(0).to_string();
            if (open.empty() || open.back() != k)
                throw EmitError("csp emit: L0 op " + std::to_string(i) + " drains " + k + ", not the open accumulator");
            open.pop_back();
            o << indent() << "}\n";
            make_room(i, 1, {});                             // the writeback's moment in L3
            o << indent() << "store " << tile_text(op.outputs.at(0)) << ";\n";
            continue;
        }
        // A compute op: its operands resident together, then the call.
        const auto need = operands(op);
        std::set<std::string> keep;
        for (const auto& t : need) keep.insert(t.to_string());
        std::size_t missing = 0;
        for (const auto& t : need) missing += resident.count(t.to_string()) ? 0 : 1;
        make_room(i, missing, keep);
        for (const auto& t : need)
            if (!resident.count(t.to_string())) {
                o << indent() << "resident " << tile_text(t) << ";\n";
                resident[t.to_string()] = false;
            }
        const TileCoord& y = op.outputs.at(0);
        std::string fn, args;
        switch (op.kind) {
            case TileOpKind::MatMulAccum:
                fn = "gemm";
                args = tile_text(op.inputs.at(0)) + ", " + tile_text(op.inputs.at(1));
                break;
            case TileOpKind::LuDiagFactor: fn = "getrf"; args = tile_text(y); break;
            case TileOpKind::PivotApply:   fn = "laswp"; args = tile_text(y); break;
            case TileOpKind::TrsmLowerLeft:  fn = "trsm_ll"; args = tile_text(op.inputs.at(0)) + ", " + tile_text(y); break;
            case TileOpKind::TrsmUpperRight: fn = "trsm_ur"; args = tile_text(op.inputs.at(0)) + ", " + tile_text(y); break;
            default: throw EmitError(std::string("csp emit: L0 op kind ") + to_string(op.kind));
        }
        if (explicit_style && (open.empty() || open.back() != y.to_string())) {
            o << indent() << "acc " << tile_text(y) << " in fabric {\n";
            open.push_back(y.to_string());
        }
        o << indent() << "call " << fn << "(" << args << ") " << (op.kind == TileOpKind::MatMulAccum ? "+->" : "->")
          << " " << tile_text(y);
        if (op.pivot_slot >= 0) o << " pivot " << op.pivot_slot;
        if (op.kind == TileOpKind::MatMulAccum && op.alpha != 1.0f) o << " alpha " << alpha_text(op.alpha);
        o << ";\n";
        if (!explicit_style) resident[y.to_string()] = true;
        for (const auto& t : need)
            if (next_use(t.to_string(), i) == std::numeric_limits<std::size_t>::max()) leave(t.to_string());
    }
    if (!open.empty()) throw EmitError("csp emit: the accumulator " + open.back() + " is never drained");
    std::vector<std::string> left;
    for (const auto& [k, w] : resident) left.push_back(k);
    for (const auto& k : left) leave(k);
    o << "}\n";
    return o.str();
}

}  // namespace sw::kpu::program::csp::lang
