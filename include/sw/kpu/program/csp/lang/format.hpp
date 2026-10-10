// ============================================================================
// include/sw/kpu/program/csp/lang/format.hpp
// The canonical text of a CSP program (docs/plans/kpu-run-csp-programs.md step 4a).
//
// format(parse(s)) prints the program's STRUCTURE -- its loops, residency, calls, contexts --
// in one fixed layout: two-space indentation, one statement per line, no comments, index
// expressions fully parenthesized only where precedence needs it. Two spellings of one program
// (whitespace, comments, redundant parentheses) format identically, so the canonical text's
// digest is the program's identity (the platform's program digest).
//
// Constant subexpressions are folded. It is a fixed point: format(parse(format(p))) == format(p). (print.hpp prints the IR -- the
// flat trace -- and loses the loops; this prints the program.)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/csp/lang/print.hpp>

#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

namespace detail {

// Precedence: '*' binds tighter than '+' and '-', which associate left.
inline int prec(const Expr& e) {
    if (e.kind != Expr::Kind::Bin) return 3;
    return e.op == '*' ? 2 : 1;
}

// A subexpression with no loop variable is its value: `(1 + 2)` and `3` are one spelling.
inline std::optional<long long> constant(const Expr& e) {
    switch (e.kind) {
        case Expr::Kind::Int: return e.value;
        case Expr::Kind::Var: return std::nullopt;
        case Expr::Kind::Bin: {
            const auto l = constant(e.args.at(0)), r = constant(e.args.at(1));
            if (!l || !r) return std::nullopt;
            return e.op == '+' ? *l + *r : e.op == '-' ? *l - *r : *l * *r;
        }
    }
    return std::nullopt;
}

inline std::string expr_text(const Expr& e) {
    if (const auto c = constant(e)) {
        // A negative constant has no spelling of its own (the language has no unary minus).
        return *c >= 0 ? std::to_string(*c) : "0 - " + std::to_string(-*c);
    }
    switch (e.kind) {
        case Expr::Kind::Int: return std::to_string(e.value);
        case Expr::Kind::Var: return e.name;
        case Expr::Kind::Bin: {
            const Expr& l = e.args.at(0);
            const Expr& r = e.args.at(1);
            const int p = prec(e);
            // Left: parenthesize a looser operand. Right: also an equal one, since the operators
            // associate left (a - (b - c) is not a - b - c).
            std::string ls = expr_text(l), rs = expr_text(r);
            const bool l_bin = l.kind == Expr::Kind::Bin && !constant(l);
            const bool r_bin = (r.kind == Expr::Kind::Bin && !constant(r)) || (constant(r) && *constant(r) < 0);
            if (l_bin && prec(l) < p) ls = "(" + ls + ")";
            if (r_bin && (r.kind != Expr::Kind::Bin || prec(r) <= p)) rs = "(" + rs + ")";
            return ls + " " + e.op + " " + rs;
        }
    }
    return "?";
}

inline std::string ref_text(const TileRef& r) {
    std::string s = r.name + "[";
    for (std::size_t i = 0; i < r.index.size(); ++i) {
        if (i) s += ", ";
        s += r.index[i].all ? std::string(":") : expr_text(r.index[i].expr);
    }
    return s + "]";
}

inline std::string refs_text(const std::vector<TileRef>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + ref_text(v[i]);
    return s;
}

inline std::string context_text(const std::vector<Stage>& via) {
    std::string s;
    for (std::size_t i = 0; i < via.size(); ++i) {
        s += i ? ", " : " via ";
        s += via[i].op;
        if (!via[i].args.empty()) s += "(" + refs_text(via[i].args) + ")";
        s += " @ " + via[i].place;
    }
    return s;
}

inline void stmts(std::ostringstream& o, const std::vector<Stmt>& body, int depth);

inline void stmt(std::ostringstream& o, const Stmt& s, int depth) {
    const std::string in(static_cast<std::size_t>(2 * depth), ' ');
    switch (s.kind) {
        case Stmt::Kind::For:
            o << in << "for " << s.var << " in " << expr_text(s.lo) << ".." << expr_text(s.hi) << " {\n";
            stmts(o, s.body, depth + 1);
            o << in << "}\n";
            return;
        case Stmt::Kind::Resident: o << in << "resident " << refs_text(s.tiles) << ";\n"; return;
        case Stmt::Kind::Release:  o << in << "release " << refs_text(s.tiles) << ";\n"; return;
        case Stmt::Kind::Acc:
            o << in << "acc " << ref_text(s.out) << " in fabric {\n";
            stmts(o, s.body, depth + 1);
            o << in << "}\n";
            return;
        case Stmt::Kind::Call:
            o << in << "call " << s.fn << "(" << refs_text(s.tiles) << ") " << (s.accumulate ? "+->" : "->") << " "
              << ref_text(s.out);
            if (s.pivot) o << " pivot " << expr_text(*s.pivot);
            if (s.alpha) o << " alpha " << alpha_text(static_cast<float>(*s.alpha));
            o << context_text(s.context) << ";\n";
            return;
        case Stmt::Kind::Store:
            o << in << "store " << ref_text(s.out) << context_text(s.context) << ";\n";
            return;
        case Stmt::Kind::Distribute:
            o << in << "distribute " << s.tiles.at(0).name << " over grid " << s.map << ";\n";
            return;
        case Stmt::Kind::Broadcast:
            o << in << "broadcast " << refs_text(s.tiles) << " along " << s.along << ";\n";
            return;
    }
}

inline void stmts(std::ostringstream& o, const std::vector<Stmt>& body, int depth) {
    for (const Stmt& s : body) stmt(o, s, depth);
}

}  // namespace detail

// The program's canonical text.
inline std::string format(const Program& p) {
    std::ostringstream o;
    o << "csp " << p.version << "\n";
    o << "program " << p.name << " machine " << p.machine << "(";
    if (p.machine == "grid") o << p.grid_rows << "x" << p.grid_cols << ", ";
    o << "l3 = " << p.l3 << ") {\n";
    for (const Decl& d : p.decls) {
        if (d.is_vector)
            o << "  vector " << d.name << "[" << d.rows << "] tile " << d.tile_rows << " " << d.io << ";\n";
        else
            o << "  tensor " << d.name << "[" << d.rows << "," << d.cols << "] tile " << d.tile_rows << "x"
              << d.tile_cols << " " << d.io << ";\n";
    }
    detail::stmts(o, p.body, 1);
    o << "}\n";
    return o.str();
}

}  // namespace sw::kpu::program::csp::lang
