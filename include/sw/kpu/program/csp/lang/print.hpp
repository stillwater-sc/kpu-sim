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
#include <iomanip>
#include <limits>
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

// An alpha as text that reads back to the same float (the lexer takes exponents).
inline std::string alpha_text(float a) {
    std::ostringstream o;
    o << std::setprecision(std::numeric_limits<float>::max_digits10) << a;
    return o.str();
}

inline std::string tile_text(const TileCoord& t) {
    return t.operand + "[" + std::to_string(t.ti) + ", " + std::to_string(t.tj) + "]";
}

// An operand of one tile column prints as a vector: b[j] (the language's `vector` decl).
inline bool is_vector(const TensorOperand& t) { return t.cols == 1 && t.tile_cols == 1; }

// A result's `via` list, from its Drain's and Writeback's stages ("" when it has none).
inline std::string via_text(const std::vector<csp::Stage>& drain, const std::vector<csp::Stage>& writeback) {
    std::string out;
    for (const auto* v : {&drain, &writeback})
        for (const csp::Stage& st : *v) {
            out += out.empty() ? " via " : ", ";
            out += to_string(st.op);
            if (st.op == VeOp::Add) out += "(" + st.arg.operand + "[" + std::to_string(st.arg.ti) + "])";
            out += std::string(" @ ") + to_string(st.place);
        }
    return out;
}

inline std::string print(const CspProgram& p) {
    using K = Action::Kind;
    // How each operand is used, for its declaration.
    std::set<std::string> loaded, stored;
    for (const Action& a : p.actions) {
        if (a.kind == K::Load || a.kind == K::Inherit) loaded.insert(a.tile.operand);   // arrives with a value
        if (a.kind == K::Store || a.kind == K::Retain) stored.insert(a.tile.operand);   // leaves with one
    }
    std::ostringstream o;
    o << "csp " << kVersion << "\n";
    o << "program " << (p.name.empty() ? std::string("p") : p.name) << " machine flat(l3 = "
      << p.channel(Chan::L3).capacity << ") {\n";
    for (const auto& name : p.source.operand_order()) {
        const TensorOperand& t = p.source.operand(name);
        const bool in = loaded.count(name) != 0, out = stored.count(name) != 0;
        const char* io = in && out ? "inout" : out ? "out" : "in";
        if (is_vector(t))
            o << "  vector " << name << "[" << t.rows << "] tile " << t.tile_rows << " " << io << ";\n";
        else
            o << "  tensor " << name << "[" << t.rows << "," << t.cols << "] tile " << t.tile_rows << "x" << t.tile_cols
              << " " << io << ";\n";
    }
    std::vector<std::string> open;                       // accumulators, innermost last
    // The last accumulating call into each tile before that tile's Drain: its acc block may
    // close after it. Blocks open lazily and close innermost-first once their last call is out,
    // so sequential, later-stored and interleaved (register-blocked) accumulators all print as
    // properly nested blocks.
    std::vector<bool> last_call(p.actions.size(), false);
    {
        std::set<std::string> seen;                      // tiles with a later call before their Drain
        for (std::size_t i = p.actions.size(); i-- > 0;) {
            const Action& a = p.actions[i];
            const std::string k = a.tile.to_string();
            if (a.kind == K::Drain) seen.erase(k);
            if (a.kind == K::Call && a.accumulate && !seen.count(k)) {
                last_call[i] = true;
                seen.insert(k);
            }
        }
    }
    std::set<std::string> done;                          // open accumulators past their last call
    auto close_finished = [&]() {
        while (!open.empty() && done.count(open.back())) {
            done.erase(open.back());
            open.pop_back();
            o << std::string(2 + 2 * open.size(), ' ') << "}\n";
        }
    };
    auto indent = [&]() { return std::string(2 + 2 * open.size(), ' '); };
    auto text = [&](const TileCoord& t) {
        return is_vector(p.source.operand(t.operand)) ? t.operand + "[" + std::to_string(t.ti) + "]" : tile_text(t);
    };
    auto expect = [&](std::size_t i, K k, const TileCoord& t) {
        if (i >= p.actions.size() || p.actions[i].kind != k || p.actions[i].tile.to_string() != t.to_string())
            throw PrintError("csp print: action " + std::to_string(i) + " is not the " + to_string(k) + " of " +
                             t.to_string() + " a statement needs; this IR does not form statements");
    };
    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        const Action& a = p.actions[i];
        switch (a.kind) {
            case K::Load: o << indent() << "resident " << text(a.tile) << ";\n"; break;
            case K::Release: o << indent() << "release " << text(a.tile) << ";\n"; break;
            case K::Inherit: o << indent() << "inherit " << text(a.tile) << ";\n"; break;
            case K::Retain: o << indent() << "retain " << text(a.tile) << ";\n"; break;
            case K::Store: o << indent() << "store " << text(a.tile) << ";\n"; break;
            case K::Move:
            case K::Feed:
                break;                                   // a call's deliveries: implied
            case K::Call: {
                const TileOp& op = p.source.ops().at(a.l0_op);
                const TileCoord& y = op.outputs.at(0);
                if (a.accumulate && std::find(open.begin(), open.end(), y.to_string()) == open.end()) {
                    o << indent() << "acc " << text(y) << " in fabric {\n";
                    open.push_back(y.to_string());
                }
                std::string fn, args;
                switch (op.kind) {
                    case TileOpKind::MatMulAccum:
                        fn = "gemm";
                        args = text(op.inputs.at(0)) + ", " + text(op.inputs.at(1));
                        break;
                    case TileOpKind::LuDiagFactor: fn = "getrf"; args = text(y); break;
                    case TileOpKind::PivotApply:   fn = "laswp"; args = text(y); break;
                    case TileOpKind::TrsmLowerLeft:  fn = "trsm_ll"; args = text(op.inputs.at(0)) + ", " + text(y); break;
                    case TileOpKind::TrsmUpperRight: fn = "trsm_ur"; args = text(op.inputs.at(0)) + ", " + text(y); break;
                    case TileOpKind::BiasAdd:        fn = "add"; args = text(y) + ", " + text(op.inputs.at(0)); break;
                    case TileOpKind::Activation:     fn = to_string(op.act); args = text(y); break;
                    default: throw PrintError("csp print: a call of " + std::string(to_string(op.kind)));
                }
                o << indent() << "call " << fn << "(" << args << ") "
                  << (op.kind == TileOpKind::MatMulAccum ? "+->" : "->") << " " << text(y);
                if (op.pivot_slot >= 0) o << " pivot " << op.pivot_slot;
                if (op.kind == TileOpKind::MatMulAccum && op.alpha != 1.0f) o << " alpha " << alpha_text(op.alpha);
                if (!a.accumulate) {                     // in place: its result's context
                    expect(i + 1, K::Drain, y);
                    expect(i + 2, K::Writeback, y);
                    o << via_text(p.actions[i + 1].context, p.actions[i + 2].context);
                }
                o << ";\n";
                if (a.accumulate && last_call[i]) {
                    done.insert(y.to_string());
                    close_finished();
                }
                if (!a.accumulate) {                     // in place: its result goes back to L3
                    expect(i + 1, K::Drain, y);
                    expect(i + 2, K::Writeback, y);
                    i += 2;
                }
                break;
            }
            case K::Drain: {
                // An accumulator's store: drained, written back, stored, released. Its block
                // closed after its last call.
                if (std::find(open.begin(), open.end(), a.tile.to_string()) != open.end())
                    throw PrintError("csp print: action " + std::to_string(i) + " drains " + a.tile.to_string() +
                                     " while its accumulator is open");
                expect(i + 1, K::Writeback, a.tile);
                // ...or retained: drained and written back into a slot it keeps.
                if (i + 2 < p.actions.size() && p.actions[i + 2].kind == K::Retain) {
                    expect(i + 2, K::Retain, a.tile);
                    o << indent() << "retain " << text(a.tile) << via_text(a.context, p.actions[i + 1].context) << ";\n";
                    i += 2;
                    break;
                }
                expect(i + 2, K::Store, a.tile);
                expect(i + 3, K::Release, a.tile);
                o << indent() << "store " << text(a.tile) << via_text(a.context, p.actions[i + 1].context) << ";\n";
                i += 3;
                break;
            }
            case K::Writeback:
                throw PrintError("csp print: action " + std::to_string(i) + " is a writeback no statement explains");
        }
    }
    if (!open.empty()) throw PrintError("csp print: the accumulator " + open.back() + " never closes");
    o << "}\n";
    return o.str();
}

}  // namespace sw::kpu::program::csp::lang
