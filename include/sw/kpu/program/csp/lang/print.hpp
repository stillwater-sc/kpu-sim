// ============================================================================
// include/sw/kpu/program/csp/lang/print.hpp
// The CSP program IR -> .csp text (docs/plans/csp-language.md §2).
//
// Prints a program the language compiler produced (compile.hpp) back as source, statement for
// statement: compile(print(p)) reproduces p's actions exactly. The output is flat -- loops are
// not recovered -- but it is the program, readable and editable.
//
// An IR whose actions do not form statements (the action-level lowering of csp/lower.hpp can
// release a tile between one operand's feed and the next operand's load, which a call cannot
// express: its operands are co-resident) is refused by name; that IR's text form comes from
// the L0 emitter (emit.hpp), which decides residency at call granularity.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

class PrintError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline std::string tile_text(const TileCoord& t) {
    return t.operand + "[" + std::to_string(t.ti) + ", " + std::to_string(t.tj) + "]";
}

inline std::string print(const CspProgram& p) {
    using K = Action::Kind;
    // How each operand is used, for its declaration.
    std::set<std::string> loaded, stored;
    for (const Action& a : p.actions) {
        if (a.kind == K::Load) loaded.insert(a.tile.operand);
        if (a.kind == K::Store) stored.insert(a.tile.operand);
    }
    std::ostringstream o;
    o << "csp " << kVersion << "\n";
    o << "program " << (p.name.empty() ? std::string("p") : p.name) << " machine flat(l3 = "
      << p.channel(Chan::L3).capacity << ") {\n";
    for (const auto& name : p.source.operand_order()) {
        const TensorOperand& t = p.source.operand(name);
        const bool in = loaded.count(name) != 0, out = stored.count(name) != 0;
        o << "  tensor " << name << "[" << t.rows << "," << t.cols << "] tile " << t.tile_rows << "x" << t.tile_cols
          << " " << (in && out ? "inout" : out ? "out" : "in") << ";\n";
    }
    std::vector<std::string> open;                       // accumulators, innermost last
    auto indent = [&]() { return std::string(2 + 2 * open.size(), ' '); };
    auto expect = [&](std::size_t i, K k, const TileCoord& t) {
        if (i >= p.actions.size() || p.actions[i].kind != k || p.actions[i].tile.to_string() != t.to_string())
            throw PrintError("csp print: action " + std::to_string(i) + " is not the " + to_string(k) + " of " +
                             t.to_string() + " a statement needs; this IR does not form statements");
    };
    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        const Action& a = p.actions[i];
        switch (a.kind) {
            case K::Load: o << indent() << "resident " << tile_text(a.tile) << ";\n"; break;
            case K::Release: o << indent() << "release " << tile_text(a.tile) << ";\n"; break;
            case K::Store: o << indent() << "store " << tile_text(a.tile) << ";\n"; break;
            case K::Move:
            case K::Feed:
                break;                                   // a call's deliveries: implied
            case K::Call: {
                const TileOp& op = p.source.ops().at(a.l0_op);
                const TileCoord& y = op.outputs.at(0);
                if (a.accumulate && std::find(open.begin(), open.end(), y.to_string()) != open.end() &&
                    open.back() != y.to_string())
                    throw PrintError("csp print: accumulators " + open.back() + " and " + y.to_string() +
                                     " overlap; acc blocks nest, they do not interleave");
                if (a.accumulate && (open.empty() || open.back() != y.to_string())) {
                    o << indent() << "acc " << tile_text(y) << " in fabric {\n";
                    open.push_back(y.to_string());
                }
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
                    default: throw PrintError("csp print: a call of " + std::string(to_string(op.kind)));
                }
                o << indent() << "call " << fn << "(" << args << ") "
                  << (op.kind == TileOpKind::MatMulAccum ? "+->" : "->") << " " << tile_text(y);
                if (op.pivot_slot >= 0) o << " pivot " << op.pivot_slot;
                if (op.kind == TileOpKind::MatMulAccum && op.alpha != 1.0f) {
                    std::ostringstream al;
                    al << op.alpha;
                    o << " alpha " << al.str();
                }
                o << ";\n";
                if (!a.accumulate) {                     // in place: its result goes back to L3
                    expect(i + 1, K::Drain, y);
                    expect(i + 2, K::Writeback, y);
                    i += 2;
                }
                break;
            }
            case K::Drain: {
                // The end of an accumulator: drained, written back, stored, released.
                if (open.empty() || open.back() != a.tile.to_string())
                    throw PrintError("csp print: action " + std::to_string(i) + " drains " + a.tile.to_string() +
                                     ", which is not the innermost open accumulator");
                open.pop_back();
                o << indent() << "}\n";
                expect(i + 1, K::Writeback, a.tile);
                expect(i + 2, K::Store, a.tile);
                expect(i + 3, K::Release, a.tile);
                o << indent() << "store " << tile_text(a.tile) << ";\n";
                i += 3;
                break;
            }
            case K::Writeback:
                throw PrintError("csp print: action " + std::to_string(i) + " is a writeback no statement explains");
        }
    }
    if (!open.empty()) throw PrintError("csp print: the accumulator " + open.back() + " is never drained");
    o << "}\n";
    return o.str();
}

}  // namespace sw::kpu::program::csp::lang
