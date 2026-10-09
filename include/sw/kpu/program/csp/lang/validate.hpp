// ============================================================================
// include/sw/kpu/program/csp/lang/validate.hpp
// Symbolic validation of a CSP program (docs/plans/csp-language.md step 1c; ADR 0004 §4).
//
// The walker (walk.hpp) checks a program by executing it, which for a 1M x 1M matmul is ~10^13
// statements. This checks the same rules over the program's STRUCTURE, without unrolling:
//
//   * index expressions are affine in the loop variables; their ranges come from interval
//     arithmetic over the loop bounds, so "A[i+1, k] may be outside 0..N" is caught statically;
//   * a tile reference is a FAMILY (an operand and, per dimension, ':' or an affine form); the
//     residency state is a set of families, and residency questions are answered on families:
//       - "already resident"  : a new family that may overlap a resident one (an affine
//                               difference whose interval contains 0) is refused;
//       - "operand resident"  : a call's operand must be COVERED by a resident family (':' or an
//                               identical affine form, per dimension);
//       - "release / store"   : must mirror a resident family exactly;
//   * every loop body is checked ONCE, for a representative iteration (the language has no
//     conditionals, and tile counts of a family do not depend on the iteration), and must be
//     RESIDENCY-BALANCED: an iteration releases what it makes resident and stores what it
//     accumulates, so iterations cannot collide with each other. (Cross-iteration residency --
//     prefetching the next iteration's tiles -- is a later step; such a program is refused by
//     name and still runs through the walker.)
//   * capacity: the live tile count is exact per statement (a family's count is a product of
//     ':' extents), and the peak over the program is reported.
//
// Totals (loads, calls, stores) are exact for rectangular loop nests (constant bounds) and
// absent otherwise.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

struct Validation {
    std::size_t peak_l3 = 0;                  // the most L3 slots the program holds at once
    std::optional<std::uint64_t> loads, calls, stores;   // exact for rectangular loop nests
};

namespace detail {

// sum(coeff[v] * v) + constant
struct Affine {
    std::map<std::string, long long> coeff;
    long long constant = 0;
    bool operator==(const Affine& o) const { return coeff == o.coeff && constant == o.constant; }
    bool is_constant() const { return coeff.empty(); }
};

inline Affine operator+(Affine a, const Affine& b) {
    for (const auto& [v, c] : b.coeff) {
        a.coeff[v] += c;
        if (a.coeff[v] == 0) a.coeff.erase(v);
    }
    a.constant += b.constant;
    return a;
}
inline Affine scale(Affine a, long long s) {
    if (s == 0) return Affine{};
    for (auto& [v, c] : a.coeff) c *= s;
    a.constant *= s;
    return a;
}

struct Interval { long long lo = 0, hi = 0; };   // inclusive

class Symbolic {
public:
    explicit Symbolic(const Program& ast) : ast_(ast) {
        if (ast_.machine != "flat")
            throw CompileError(0, "machine " + ast_.machine + ": level 2 (a distributed machine) is not built yet; "
                                  "this step compiles level-1 programs (machine flat)");
        if (ast_.l3 <= 0) throw CompileError(0, "machine flat: the L3 capacity must be positive");
        cap_ = static_cast<std::size_t>(ast_.l3);
        operands_ = declared_operands(ast_);
    }

    Validation run() {
        for (const Stmt& s : ast_.body) stmt(s, 1, true);
        for (const Entry& e : state_)
            throw CompileError(0, text(e.family) + " is still resident at the end of the program; release it");
        for (const Acc& a : accs_) throw CompileError(0, "the accumulator " + text(a.tile) + " is never stored");
        Validation v;
        v.peak_l3 = peak_;
        if (exact_) {
            v.loads = loads_;
            v.calls = calls_;
            v.stores = stores_;
        }
        return v;
    }

private:
    struct Dimension { bool all = false; Affine a; };
    struct Family { std::string operand; std::vector<Dimension> dims; int line = 0; };
    struct Entry { Family family; bool dirty = false; };
    struct Acc { Family tile; bool closed = false; bool fed = false; std::size_t depth = 0; };
    struct Var { Interval range; bool definite; };

    const Program& ast_;
    std::size_t cap_ = 0;
    TileProgram operands_;
    std::map<std::string, Var> vars_;
    std::vector<std::string> loop_order_;           // enclosing loop variables, outermost first
    std::vector<Entry> state_;
    std::vector<Acc> accs_;
    std::size_t live_ = 0, peak_ = 0;
    bool exact_ = true;
    std::uint64_t loads_ = 0, calls_ = 0, stores_ = 0;

    // ---- affine forms and ranges ----
    Affine affine(const Expr& e, int line) const {
        switch (e.kind) {
            case Expr::Kind::Int: { Affine a; a.constant = e.value; return a; }
            case Expr::Kind::Var: {
                if (!vars_.count(e.name)) throw CompileError(line, "'" + e.name + "' is not a loop variable in scope");
                Affine a;
                a.coeff[e.name] = 1;
                return a;
            }
            case Expr::Kind::Bin: {
                const Affine l = affine(e.args.at(0), line), r = affine(e.args.at(1), line);
                if (e.op == '+') return l + r;
                if (e.op == '-') return l + scale(r, -1);
                if (l.is_constant()) return scale(r, l.constant);
                if (r.is_constant()) return scale(l, r.constant);
                throw CompileError(line, "index expressions must be affine in the loop variables (a product of two "
                                         "variables is not)");
            }
        }
        return {};
    }
    Interval range(const Affine& a) const {
        Interval r{a.constant, a.constant};
        for (const auto& [v, c] : a.coeff) {
            const Interval x = vars_.at(v).range;
            if (c >= 0) { r.lo += c * x.lo; r.hi += c * x.hi; }
            else { r.lo += c * x.hi; r.hi += c * x.lo; }
        }
        return r;
    }
    static std::string text(const Affine& a) {
        std::string s;
        for (const auto& [v, c] : a.coeff) {
            if (!s.empty()) s += c < 0 ? " - " : " + ";
            else if (c < 0) s += "-";
            const long long m = c < 0 ? -c : c;
            if (m != 1) s += std::to_string(m) + "*";
            s += v;
        }
        if (a.constant != 0 || s.empty()) {
            if (!s.empty()) s += a.constant < 0 ? " - " : " + ";
            else if (a.constant < 0) s += "-";
            s += std::to_string(a.constant < 0 ? -a.constant : a.constant);
        }
        return s;
    }
    static std::string text(const Family& f) {
        std::string s = f.operand + "[";
        for (std::size_t i = 0; i < f.dims.size(); ++i) {
            if (i) s += ", ";
            s += f.dims[i].all ? std::string(":") : text(f.dims[i].a);
        }
        return s + "]";
    }

    Family family(const TileRef& r) const {
        if (!operands_.has_operand(r.name)) throw CompileError(r.line, "'" + r.name + "' is not a declared operand");
        const TensorOperand& op = operands_.operand(r.name);
        if (r.index.size() != 2) throw CompileError(r.line, r.name + " is a tensor: index it [row, col]");
        Family f;
        f.operand = r.name;
        f.line = r.line;
        const Dim n[2] = {op.n_tile_rows(), op.n_tile_cols()};
        for (std::size_t i = 0; i < 2; ++i) {
            Dimension d;
            d.all = r.index[i].all;
            if (!d.all) {
                d.a = affine(r.index[i].expr, r.line);
                const Interval x = range(d.a);
                if (x.lo < 0 || x.hi >= static_cast<long long>(n[i]))
                    throw CompileError(r.line, r.name + ": tile index " + text(d.a) + " may be outside 0.." +
                                               std::to_string(n[i]) + " (it ranges over " + std::to_string(x.lo) +
                                               ".." + std::to_string(x.hi) + ")");
            }
            f.dims.push_back(d);
        }
        return f;
    }
    Family single(const TileRef& r) const {
        for (const Index& ix : r.index)
            if (ix.all) throw CompileError(r.line, r.name + ": a single tile is required here, not a slice");
        return family(r);
    }
    std::uint64_t count(const Family& f) const {
        const TensorOperand& op = operands_.operand(f.operand);
        std::uint64_t n = 1;
        n *= f.dims[0].all ? op.n_tile_rows() : 1;
        n *= f.dims[1].all ? op.n_tile_cols() : 1;
        return n;
    }
    static bool same(const Family& a, const Family& b) {
        if (a.operand != b.operand) return false;
        for (std::size_t i = 0; i < a.dims.size(); ++i) {
            if (a.dims[i].all != b.dims[i].all) return false;
            if (!a.dims[i].all && !(a.dims[i].a == b.dims[i].a)) return false;
        }
        return true;
    }
    bool may_overlap(const Family& a, const Family& b) const {
        if (a.operand != b.operand) return false;
        for (std::size_t i = 0; i < a.dims.size(); ++i) {
            if (a.dims[i].all || b.dims[i].all) continue;
            const Interval d = range(a.dims[i].a + scale(b.dims[i].a, -1));
            if (d.lo > 0 || d.hi < 0) return false;          // provably distinct along this dimension
        }
        return true;
    }
    // Is single tile `t` provably inside family `f`?
    static bool covers(const Family& f, const Family& t) {
        if (f.operand != t.operand) return false;
        for (std::size_t i = 0; i < f.dims.size(); ++i)
            if (!f.dims[i].all && !(f.dims[i].a == t.dims[i].a)) return false;
        return true;
    }

    std::uint64_t multiplier_ = 1;

    void hold(std::uint64_t n, int line, const std::string& why) {
        if (live_ + n > cap_)
            throw CompileError(line, why + " needs " + std::to_string(n) + " L3 slot" + (n == 1 ? "" : "s") + ", and " +
                                     std::to_string(cap_ - std::min(live_, cap_)) + " of " + std::to_string(cap_) +
                                     " are free");
        live_ += n;
        peak_ = std::max(peak_, live_);
    }

    // ---- statements ----
    void stmt(const Stmt& s, std::size_t depth, bool definite) {
        if (!s.context.empty())
            throw CompileError(s.line, "tile contexts ('via') arrive with the linear operator (csp-language plan step 2)");
        switch (s.kind) {
            case Stmt::Kind::For: loop(s, depth, definite); break;
            case Stmt::Kind::Resident:
                for (const TileRef& r : s.tiles) {
                    const Family f = family(r);
                    for (const Entry& e : state_)
                        if (may_overlap(f, e.family))
                            throw CompileError(s.line, same(f, e.family) || (count(f) == 1 && covers(e.family, f))
                                                           ? text(f) + " is already resident"
                                                           : text(f) + " may already be resident (it can overlap " +
                                                                 text(e.family) + ")");
                    for (const Acc& a : accs_)
                        if (may_overlap(f, a.tile))
                            throw CompileError(s.line, text(f) + " can be an accumulator in the fabric, not a DRAM tile to load");
                    hold(count(f), s.line, "resident " + text(f));
                    state_.push_back(Entry{f, false});
                    loads_ += count(f) * multiplier_;
                }
                break;
            case Stmt::Kind::Release:
                for (const TileRef& r : s.tiles) {
                    const Family f = family(r);
                    auto it = std::find_if(state_.begin(), state_.end(), [&](const Entry& e) { return same(e.family, f); });
                    if (it == state_.end()) {
                        const bool partial = std::any_of(state_.begin(), state_.end(),
                                                         [&](const Entry& e) { return may_overlap(e.family, f); });
                        throw CompileError(s.line, partial ? "release " + text(f) + " must mirror a resident statement "
                                                                 "(symbolic validation releases what was made resident, as written)"
                                                           : "release " + text(f) + ": it is not resident");
                    }
                    if (it->dirty)
                        throw CompileError(s.line, "release " + text(f) + ": a call wrote it and it is not stored; store it first");
                    live_ -= count(f);
                    state_.erase(it);
                }
                break;
            case Stmt::Kind::Acc: {
                const Family y = single(s.out);
                for (const Entry& e : state_)
                    if (may_overlap(y, e.family))
                        throw CompileError(s.line, "acc " + text(y) + ": it can be resident in L3; an accumulator lives in the fabric");
                for (const Acc& a : accs_)
                    if (may_overlap(y, a.tile)) throw CompileError(s.line, "acc " + text(y) + ": it may already have an accumulator");
                accs_.push_back(Acc{y, false, false, depth});
                for (const Stmt& b : s.body) stmt(b, depth, definite);
                Acc& a = *std::find_if(accs_.begin(), accs_.end(), [&](const Acc& x) { return same(x.tile, y); });
                if (!a.fed)
                    throw CompileError(s.line, "acc " + text(y) + " receives no call; an accumulator needs at least one "
                                               "'+->' into it");
                a.closed = true;
                break;
            }
            case Stmt::Kind::Call: call(s, depth, definite); break;
            case Stmt::Kind::Store: store(s); break;
            case Stmt::Kind::Distribute:
            case Stmt::Kind::Broadcast:
                throw CompileError(s.line, "distribute and broadcast are level 2 (a distributed machine), not built yet");
        }
    }

    void loop(const Stmt& s, std::size_t depth, bool definite) {
        if (vars_.count(s.var)) throw CompileError(s.line, "loop variable '" + s.var + "' shadows an outer one");
        const Affine lo = affine(s.lo, s.line), hi = affine(s.hi, s.line);
        const Interval lor = range(lo), hir = range(hi);
        if (hir.hi - 1 < lor.lo) return;                         // never runs
        const Interval trips = range(hi + scale(lo, -1));
        const bool rectangular = lo.is_constant() && hi.is_constant();
        if (!rectangular) exact_ = false;
        vars_[s.var] = Var{Interval{lor.lo, hir.hi - 1}, trips.lo >= 1};
        loop_order_.push_back(s.var);
        const std::uint64_t saved_mult = multiplier_;
        if (rectangular) multiplier_ *= static_cast<std::uint64_t>(hi.constant - lo.constant);

        const std::vector<Entry> before = state_;
        const std::size_t accs_before = accs_.size();
        for (const Stmt& b : s.body) stmt(b, depth + 1, definite && trips.lo >= 1);

        // Residency-balanced: an iteration leaves exactly the families it found.
        for (const Entry& e : state_)
            if (std::none_of(before.begin(), before.end(), [&](const Entry& x) { return same(x.family, e.family); }))
                throw CompileError(s.line, "the loop over " + s.var + " leaves " + text(e.family) +
                                           " resident: an iteration must release what it makes resident (cross-iteration "
                                           "residency is a later step)");
        for (const Entry& e : before)
            if (std::none_of(state_.begin(), state_.end(), [&](const Entry& x) { return same(x.family, e.family); }))
                throw CompileError(s.line, "the loop over " + s.var + " releases " + text(e.family) +
                                           ", which was made resident outside it");
        if (accs_.size() > accs_before)
            throw CompileError(s.line, "the loop over " + s.var + " leaves the accumulator " + text(accs_.back().tile) +
                                       " unstored: an iteration must store what it accumulates");
        multiplier_ = saved_mult;
        loop_order_.pop_back();
        vars_.erase(s.var);
    }

    void call(const Stmt& s, std::size_t depth, bool definite) {
        std::vector<Family> args;
        for (const TileRef& r : s.tiles) args.push_back(single(r));
        const Family y = single(s.out);
        // The same function rules as the walker.
        if (s.fn == "gemm") {
            if (args.size() != 2) throw CompileError(s.line, "gemm takes two operands: gemm(a, b) +-> y");
            if (!s.accumulate) throw CompileError(s.line, "gemm accumulates: write '+->'");
        } else if (s.fn == "getrf" || s.fn == "laswp") {
            if (args.size() != 1 || !same(args[0], y))
                throw CompileError(s.line, s.fn + " works in place: " + s.fn + "(x) -> x pivot p");
            if (!s.pivot) throw CompileError(s.line, s.fn + " needs its pivot slot: pivot p");
            (void)affine(*s.pivot, s.line);
        } else if (s.fn == "trsm_ll" || s.fn == "trsm_ur") {
            if (args.size() != 2 || !same(args[1], y))
                throw CompileError(s.line, s.fn + " works in place on its second operand: " + s.fn + "(d, x) -> x");
        } else {
            throw CompileError(s.line, "unknown tile function '" + s.fn + "' (gemm, getrf, laswp, trsm_ll, trsm_ur)");
        }
        if (s.fn != "gemm" && s.accumulate) throw CompileError(s.line, s.fn + " writes its result: '->', not '+->'");
        if (s.fn != "getrf" && s.fn != "laswp" && s.pivot) throw CompileError(s.line, s.fn + " takes no pivot");

        auto resident = [&](const Family& t) {
            if (std::none_of(state_.begin(), state_.end(), [&](const Entry& e) { return covers(e.family, t); }))
                throw CompileError(s.line, "call " + s.fn + " reads " + text(t) + ", which is not provably resident; "
                                           "make it resident first");
        };
        auto acc = std::find_if(accs_.begin(), accs_.end(), [&](const Acc& a) { return same(a.tile, y); });
        if (acc != accs_.end() && !acc->closed) {
            for (const Family& a : args) resident(a);
            // The accumulator is fed if this call runs in every iteration of the loops inside it.
            if (definite || depth <= acc->depth) acc->fed = true;
            else {
                bool all_definite = true;
                for (std::size_t i = acc->depth - 1; i < loop_order_.size(); ++i)   // the loops inside the acc
                    all_definite = all_definite && vars_.at(loop_order_[i]).definite;
                if (all_definite) acc->fed = true;
            }
            calls_ += multiplier_;
            return;
        }
        if (acc != accs_.end())
            throw CompileError(s.line, "call " + s.fn + " writes " + text(y) + ", whose accumulator is closed; store it first");
        for (const Acc& a : accs_)
            if (may_overlap(a.tile, y))
                throw CompileError(s.line, "call " + s.fn + " writes " + text(y) + ", which can be the accumulator " + text(a.tile));
        auto owner = std::find_if(state_.begin(), state_.end(), [&](const Entry& e) { return covers(e.family, y); });
        if (owner == state_.end())
            throw CompileError(s.line, "call " + s.fn + " writes " + text(y) + ", which is neither provably resident nor "
                                       "an open accumulator");
        for (const Family& a : args) resident(a);
        owner->dirty = true;
        calls_ += multiplier_;
    }

    void store(const Stmt& s) {
        const Family y = family(s.out);
        auto acc = std::find_if(accs_.begin(), accs_.end(), [&](const Acc& a) { return same(a.tile, y); });
        if (acc != accs_.end()) {
            if (!acc->closed)
                throw CompileError(s.line, "store " + text(y) + " inside its own accumulator; store it after the acc block");
            hold(1, s.line, "store " + text(y) + "'s writeback");      // the writeback's moment in L3
            live_ -= 1;
            accs_.erase(acc);
            stores_ += multiplier_;
            return;
        }
        auto it = std::find_if(state_.begin(), state_.end(), [&](const Entry& e) { return same(e.family, y); });
        if (it == state_.end())
            throw CompileError(s.line, "store " + text(y) + " must name an accumulator or mirror a resident statement");
        if (!it->dirty) throw CompileError(s.line, "store " + text(y) + ": nothing has written it since it was loaded");
        it->dirty = false;
        stores_ += count(y) * multiplier_;
    }
};

}  // namespace detail

// Validate a program over its structure, without executing it. Throws CompileError, by line.
inline Validation validate(const Program& ast) { return detail::Symbolic(ast).run(); }
inline Validation validate(const std::string& source) { return validate(parse(source)); }

}  // namespace sw::kpu::program::csp::lang
