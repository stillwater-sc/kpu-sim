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
//   * residency across operators (kpu-run-csp-programs step 4d): `inherit` opens the program, at
//     its top, and holds its slots from the first statement; `retain` ends a residency (or
//     writes an accumulator back into a slot) without a Release, and the slot is held to the end.
//     A `retain` inside loops must name a distinct tile in every iteration -- each enclosing loop
//     variable indexes one dimension of it, alone -- and is charged for every iteration at once
//     (the loops' bounds must be constant), which is exact at the end and conservative before.
//   * a late store (`store C[i, j] late;`) is checked where it is written, as any store; the
//     walker emits its Drain, Writeback, Store and Release after the next accumulator's calls, at
//     the end of the program, or before a `resident` or `acc` that names the tile again. Its
//     writeback needs a slot THERE, so when the program has a late store, every acc block's open
//     and close and the program's end must leave one slot free (a `resident` needs a slot of its
//     own anyway, and the late store's is free again before the Load). A late store
//     names an accumulator, and carries no epilogue (an add's vector would have to be resident
//     where it is emitted, which the structure does not say).
//   * a stage block (`stage T in l2 { }`) copies tiles of resident families into L2: T must be
//     covered by a resident family, may not overlap a family staged around it, and fits the
//     machine clause's `l2 = N` with the families staged around it. Inside the block the staged
//     tiles stay resident in L3 and unwritten: a release, a retain or a call that may touch them
//     is refused.
//   * an L2 accumulator (`acc T in l2 { }`; #429, #430) holds an L2 slot per tile of T, from the
//     block's open to the tile's store, counted against `l2 = N` with the stage blocks. A call
//     `+->` into a tile T covers is a merge; T is fed when such a call runs in every iteration of
//     the loops inside the block (the walker checks each tile is fed). Its store names T as the
//     acc does, is not late, and carries bm.egress stages only.
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
#include <set>
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

struct Validation {
    std::size_t peak_l3 = 0;                  // the most L3 slots the program holds at once (with a late
                                              // store: counting its writeback at every place it can land)
    std::size_t peak_l2 = 0;                  // the most L2 slots its stage blocks hold at once
    std::uint64_t inherited = 0;              // slots held from the start (inherit)
    std::uint64_t retained = 0;               // slots held at the end (retain)
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
    Symbolic(const Program& ast, const Target* target) : ast_(ast), target_(target) {
        if (ast_.machine != "flat")
            throw CompileError(0, "machine " + ast_.machine + ": level 2 (a distributed machine) is not built yet; "
                                  "this step compiles level-1 programs (machine flat)");
        if (ast_.l3 <= 0) throw CompileError(0, "machine flat: the L3 capacity must be positive");
        cap_ = static_cast<std::size_t>(ast_.l3);
        operands_ = declared_operands(ast_);
        for (const Decl& d : ast_.decls) if (d.is_vector) vectors_.insert(d.name);
        has_late_ = any_late(ast_.body);
    }

    Validation run() {
        for (const Stmt& s : ast_.body) {
            if (s.kind != Stmt::Kind::Inherit) started_ = true;
            else if (started_)
                throw CompileError(s.line, "inherit comes first: the tiles arrive resident before the program runs, "
                                           "so `inherit` statements open the program, outside any loop");
            stmt(s, 1, true);
        }
        for (const Entry& e : state_)
            throw CompileError(0, text(e.family) + " is still resident at the end of the program; release it");
        if (has_late_) late_room(0, "the end of the program");
        for (const Acc& a : accs_) throw CompileError(0, "the accumulator " + text(a.tile) + " is never stored");
        Validation v;
        v.peak_l3 = peak_;
        v.peak_l2 = peak_l2_;
        v.inherited = inherited_;
        v.retained = retained_count_;
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
    struct Entry { Family family; bool dirty = false; bool inherited = false; };
    struct Box { std::string operand; std::vector<std::pair<bool, Interval>> dims; };   // (all, range)
    // finals: an L2 accumulator's tiles whose final merge (a call with `via`) has run. A family
    // over a loop variable is widened to the variable's whole dimension when its loop ends.
    // formatted: a final merge converted its tile into a block format, after which nothing may run.
    // requantized: the tiles whose final merge carried the scale (#427), widened over a loop's
    // variable when the loop ends, as finals are; a requantized result's store must be covered.
    struct Acc { Family tile; bool closed = false; bool fed = false; std::size_t depth = 0; bool l2 = false;
                 std::vector<Family> finals = {}; bool formatted = false; std::vector<Family> requantized = {};
                 std::optional<Family> init = {};          // `in l2 init X`: X until each tile's first merge
                 std::vector<Family> merged = {};          // the tiles a call has merged into, as written
                 bool reserved = false; };                 // a fabric accumulator: its results' L2 slots taken
    struct Var { Interval range; bool definite; };

    const Program& ast_;
    const Target* target_ = nullptr;
    std::set<std::string> vectors_;
    std::size_t cap_ = 0;
    TileProgram operands_;
    std::map<std::string, Var> vars_;
    std::vector<std::string> loop_order_;           // enclosing loop variables, outermost first
    std::vector<Entry> state_;
    std::vector<Acc> accs_;
    std::size_t live_ = 0, peak_ = 0;
    bool exact_ = true;
    bool started_ = false;                          // a top-level statement other than inherit ran
    std::size_t nonrect_ = 0;                       // enclosing loops with non-constant bounds
    std::vector<Box> retained_;                     // every tile a retain names, as a range per dimension
    std::uint64_t inherited_ = 0, retained_count_ = 0;
    std::uint64_t loads_ = 0, calls_ = 0, stores_ = 0;
    bool has_late_ = false;                         // the program has a `store ... late`
    std::vector<Family> staged_;                    // families the enclosing stage blocks hold
    std::size_t l2_live_ = 0, peak_l2_ = 0;
    std::size_t l2_results_ = 0;            // the open fabric accumulators' result slots
    // The L2 slots held at a check: the stages', the partials', the open fabric results', and the
    // late results still pending. The walker keeps a late result's slot until it emits the store:
    // at the close of an accumulator that opened after it, before a statement that names the
    // tile again, or at the program's end. This pass releases them at the same points
    // (release_late). It sees a loop body once, not once per iteration, so a body with a late
    // store starts with that store's result pending from the previous iteration (loop()):
    // exact in the steady state, conservative in a loop's first iteration.
    struct Late { std::size_t seq = 0, n = 0; Family tile; bool carried = false; };
    std::vector<Late> late_pending_;
    std::size_t late_seq_ = 0;
    std::size_t l2_held() const {
        std::size_t n = l2_live_ + l2_results_;
        for (const Late& l : late_pending_) n += l.n;
        return n;
    }
    void hold_late(std::size_t n, const Family& tile, bool carried = false) {
        late_pending_.push_back(Late{late_seq_++, n, tile, carried});
    }
    // An accumulator's close: the late stores written before it opened (seq < due) are emitted.
    void release_late_before(std::size_t due) {
        late_pending_.erase(std::remove_if(late_pending_.begin(), late_pending_.end(),
                                           [&](const Late& l) { return l.seq < due; }),
                            late_pending_.end());
    }
    // A statement naming `f`: the walker emits the late stores up to the one that is it. It
    // matches the exact tile, so this matches the same family in the same iteration, never a
    // result carried from the previous iteration (whose indices differ).
    void release_late_naming(const Family& f) {
        for (std::size_t i = 0; i < late_pending_.size(); ++i)
            if (!late_pending_[i].carried && same(late_pending_[i].tile, f)) {
                late_pending_.erase(late_pending_.begin(), late_pending_.begin() + static_cast<std::ptrdiff_t>(i) + 1);
                return;
            }
    }

    void not_staged(const Family& f, int line, const std::string& what) const {
        for (const Family& st : staged_)
            if (may_overlap(f, st))
                throw CompileError(line, what + " " + text(f) + ", which can be staged in L2 (" + text(st) +
                                         "): a staged tile stays resident and unwritten until its stage block ends");
    }
    // `acc T in l2 { ... }`: an L2 slot per tile of T, held to the tile's store.
    void acc_l2(const Stmt& s, std::size_t depth, bool definite) {
        if (ast_.l2 <= 0)
            throw CompileError(s.line, "acc ... in l2: the machine clause declares no L2 (write `l2 = N`)");
        if (target_ && !target_->combiner)
            throw CompileError(s.line, "acc ... in l2: the Combiner merges its partials, and the target has no Combiner "
                                       "(the deployment's `combiner` block)");
        if (s.init && target_ && !target_->combiner->count(VeOp::Add))
            throw CompileError(s.line, "acc ... in l2 init: the Combiner adds the init to the first merge's partial, and "
                                       "the target's Combiner does not run add (the deployment's `combiner` ops)");
        const Family y = family(s.out);
        for (const Entry& e : state_)
            if (may_overlap(y, e.family))
                throw CompileError(s.line, "acc " + text(y) + " in l2: it can be resident in L3; an L2 accumulator's "
                                           "partial starts in its L2 slot, so release it first");
        for (const Acc& a : accs_)
            if (may_overlap(y, a.tile)) throw CompileError(s.line, "acc " + text(y) + ": it may already have an accumulator");
        not_retained(y, s.line);
        const std::size_t n = count(y);
        const std::size_t cap = static_cast<std::size_t>(ast_.l2);
        if (l2_held() + n > cap)
            throw CompileError(s.line, "acc " + text(y) + " in l2 needs " + std::to_string(n) + " L2 slot" +
                                       (n == 1 ? "" : "s") + " for its partials, and " +
                                       std::to_string(cap - std::min(l2_held(), cap)) + " of " + std::to_string(ast_.l2) +
                                       " are free");
        if (s.init) init_l2(s, y);
        l2_live_ += n;
        peak_l2_ = std::max(peak_l2_, l2_held());
        release_late_naming(y);
        const std::size_t due = late_seq_;
        accs_.push_back(Acc{y, false, false, depth, true});
        if (s.init) {                                   // its residency is checked again at the first merge
            Family x = family(*s.init);
            x.line = s.line;
            accs_.back().init = x;
        }
        for (const Stmt& b : s.body) stmt(b, depth, definite);
        Acc& a = *std::find_if(accs_.begin(), accs_.end(), [&](const Acc& x) { return same(x.tile, y); });
        if (!a.fed)
            throw CompileError(s.line, "acc " + text(y) + " in l2 receives no call; each of its tiles needs at least "
                                       "one '+->' into it");
        a.closed = true;
        release_late_before(due);
    }

    void stage(const Stmt& s, std::size_t depth, bool definite) {
        if (ast_.l2 <= 0)
            throw CompileError(s.line, "stage ... in l2: the machine clause declares no L2 (write `l2 = N`)");
        std::vector<Family> mine;
        std::size_t n = 0;
        for (const TileRef& r : s.tiles) {
            const Family f = family(r);
            const bool covered = std::any_of(state_.begin(), state_.end(), [&](const Entry& e) {
                if (e.family.operand != f.operand) return false;
                for (std::size_t i = 0; i < f.dims.size(); ++i)
                    if (!e.family.dims[i].all && (f.dims[i].all || !(e.family.dims[i].a == f.dims[i].a))) return false;
                return true;
            });
            if (!covered)
                throw CompileError(s.line, "stage " + text(f) + ": it is not provably resident; a stage copies resident "
                                           "tiles from L3");
            for (const Family& st : staged_)
                if (may_overlap(f, st)) throw CompileError(s.line, "stage " + text(f) + ": it may already be staged (" + text(st) + ")");
            for (const Family& st : mine)
                if (may_overlap(f, st)) throw CompileError(s.line, "stage " + text(f) + ": it may already be staged (" + text(st) + ")");
            mine.push_back(f);
            n += count(f);
        }
        const std::size_t cap = static_cast<std::size_t>(ast_.l2);
        if (l2_held() + n > cap)
            throw CompileError(s.line, "stage needs " + std::to_string(n) + " L2 slot" + (n == 1 ? "" : "s") + ", and " +
                                       std::to_string(cap - std::min(l2_held(), cap)) + " of " + std::to_string(ast_.l2) +
                                       " are free");
        l2_live_ += n;
        peak_l2_ = std::max(peak_l2_, l2_held());
        for (const Family& f : mine) staged_.push_back(f);
        for (const Stmt& b : s.body) stmt(b, depth, definite);
        staged_.resize(staged_.size() - mine.size());
        l2_live_ -= n;
    }

    static bool any_late(const std::vector<Stmt>& body) {
        for (const Stmt& s : body)
            if ((s.kind == Stmt::Kind::Store && s.late) || any_late(s.body)) return true;
        return false;
    }
    // A late store's writeback takes a slot where the walker emits it: an acc block's open or
    // close, or the program's end. The structure does not say which of those has a store pending,
    // so each counts the slot, in the capacity check and in the peak (conservative, never low).
    void late_room(int line, const std::string& where) {
        if (live_ + 1 > cap_)
            throw CompileError(line, "a late store is emitted at " + where + " and its writeback needs an L3 slot, "
                                     "and all " + std::to_string(cap_) + " are held there");
        peak_ = std::max(peak_, live_ + 1);
    }

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
        const bool vec = vectors_.count(r.name) != 0;   // one dimension: its tiles down the column
        if (vec && r.index.size() != 1) throw CompileError(r.line, r.name + " is a vector: index it [j]");
        if (!vec && r.index.size() != 2) throw CompileError(r.line, r.name + " is a tensor: index it [row, col]");
        Family f;
        f.operand = r.name;
        f.line = r.line;
        const Dim n[2] = {op.n_tile_rows(), op.n_tile_cols()};
        for (std::size_t i = 0; i < r.index.size(); ++i) {
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
        if (f.dims.size() > 1) n *= f.dims[1].all ? op.n_tile_cols() : 1;
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
    // Is family `t` provably inside family `f`? A slice of t is inside only a slice of f, so a
    // resident X[0, 0] does not hold X[0, :] (covers() takes t as a single tile).
    static bool contains(const Family& f, const Family& t) {
        if (f.operand != t.operand) return false;
        for (std::size_t i = 0; i < f.dims.size(); ++i) {
            if (f.dims[i].all) continue;
            if (t.dims[i].all || !(f.dims[i].a == t.dims[i].a)) return false;
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

    // ---- residency across operators ----
    Box box(const Family& f) const {
        Box b;
        b.operand = f.operand;
        for (const Dimension& d : f.dims) b.dims.push_back({d.all, d.all ? Interval{} : range(d.a)});
        return b;
    }
    void not_retained(const Family& f, int line) const {
        for (const Box& b : retained_) {
            if (b.operand != f.operand) continue;
            bool overlap = true;
            for (std::size_t i = 0; i < f.dims.size() && overlap; ++i) {
                if (b.dims[i].first || f.dims[i].all) continue;
                const Interval x = range(f.dims[i].a);
                if (x.hi < b.dims[i].second.lo || x.lo > b.dims[i].second.hi) overlap = false;
            }
            if (overlap)
                throw CompileError(line, text(f) + " may be retained: a retained slot is held for the operator after "
                                         "this one, and this program cannot name the tile again");
        }
    }
    // Every enclosing loop variable indexes one dimension of the retained family, alone: then each
    // iteration retains different tiles, and the iterations' total is the family's count times
    // the trip counts.
    void distinct_per_iteration(const Family& f, int line) const {
        if (nonrect_ > 0)
            throw CompileError(line, "retain " + text(f) + " inside a loop with non-constant bounds: its slots are "
                                     "counted for every iteration, which needs constant bounds");
        for (const std::string& v : loop_order_) {
            std::size_t in = 0;
            bool alone = true;
            for (const Dimension& d : f.dims)
                if (!d.all && d.a.coeff.count(v)) {
                    ++in;
                    alone = alone && d.a.coeff.size() == 1;
                }
            if (in != 1 || !alone)
                throw CompileError(line, "retain " + text(f) + " inside the loop over " + v + ": every iteration must "
                                         "retain different tiles, so " + v + " must index one dimension of it, alone");
        }
    }
    void retain(const Stmt& s, const TileRef& r) {
        const Family f = family(r);
        if (scaled(f.operand))
            throw CompileError(s.line, "retain " + text(f) + ": it is requantized on its way to memory; retaining one "
                                       "is not built yet, so store it");
        not_staged(f, s.line, "retain");
        distinct_per_iteration(f, s.line);
        const std::uint64_t n = count(f), total = n * multiplier_;
        auto acc = std::find_if(accs_.begin(), accs_.end(), [&](const Acc& a) { return same(a.tile, f); });
        if (acc != accs_.end()) {
            if (!acc->closed)
                throw CompileError(s.line, "retain " + text(f) + " inside its own accumulator; retain it after the acc block");
            if (acc->l2)
                throw CompileError(s.line, "retain " + text(f) + ": " + text(f) + " is an L2 accumulator; retaining one "
                                           "is not built yet, so store it");
            context(s, f);
            hold(total, s.line, "retain " + text(f) + "'s writeback");
            if (acc->reserved) l2_results_ -= count(acc->tile);       // its results' L2 slots, freed
            accs_.erase(acc);
        } else {
            auto it = std::find_if(state_.begin(), state_.end(), [&](const Entry& e) { return same(e.family, f); });
            if (it == state_.end())
                throw CompileError(s.line, "retain " + text(f) + ": it is neither resident nor an accumulator (a retain "
                                           "mirrors a resident statement, as written)");
            if (!s.context.empty())
                throw CompileError(s.line, "retain " + text(f) + " via ...: a resident tile stays where it is, so nothing "
                                           "moves it to run the stages on; place them on the call that writes it");
            hold(total - n, s.line, "retain " + text(f) + " in every iteration");   // its own slots are held
            state_.erase(it);
        }
        retained_.push_back(box(f));
        retained_count_ += total;
    }

    // ---- statements ----
    // A result's `via` list (context.hpp), and each add's vector provably resident and tiled as
    // the result's columns.
    // A requantized result (`acc` declared, #427): its accumulator values need a scale on every
    // path into storage (`required`). Its context may also hold a bias before the scale, in the
    // accumulator's type, and relu after it (integer_context_refusal). Any other tile's context
    // has no scale, and computes in fp32.
    bool context(const Stmt& s, const Family& y, bool required = false) {
        const auto resolved = resolve_result_context(s.context, target_, [](int line, const std::string& m) {
            throw CompileError(line, m);
        });
        const bool has =
            std::any_of(resolved.begin(), resolved.end(), [](const ResolvedStage& r) { return r.op == VeOp::Scale; });
        // The type a bias is added in: the accumulator's, before a requantized result's scale.
        std::string bias_type = numeric::kFp32;
        if (scaled(y.operand)) {
            std::vector<VeOp> ops;
            for (const ResolvedStage& r : resolved) ops.push_back(r.op);
            if (const auto why = integer_context_refusal(ops, text(y), type_of(y.operand), acc_of(y.operand), required))
                throw CompileError(s.line, "via ...: " + *why);
            bias_type = acc_of(y.operand);
        } else {
            if (has)
                throw CompileError(s.line, "via ... scale: scale requantizes an accumulator into its tensor's type, and " +
                                               y.operand + " declares no acc");
            if (!s.context.empty() && type_of(y.operand) != numeric::kFp32)
                throw CompileError(s.line, "via ...: a tile context computes in fp32; " + y.operand + " is " +
                                               type_of(y.operand));
        }
        // Each stage's vector operands: an add's bias (in bias_type), a per-channel scale's
        // multipliers and shifts (int32: requantize's parameters, #427). Each is a vector as wide as
        // the result's tile, provably resident.
        struct Read {
            const TileRef* ref;
            std::string type, what;
        };
        for (const ResolvedStage& r : resolved) {
            std::vector<Read> reads;
            if (r.op == VeOp::Add && r.arg) reads.push_back({&*r.arg, bias_type, "add"});
            if (r.op == VeOp::Scale && r.arg) {
                reads.push_back({&*r.arg, "int32", "scale"});
                reads.push_back({&*r.arg2, "int32", "scale"});
            }
            for (const Read& rd : reads) {
                const TileRef& ref = *rd.ref;
                if (!vectors_.count(ref.name))
                    throw CompileError(ref.line, rd.what + "(" + ref.name + "[..]): " + rd.what + " takes a vector operand");
                const Family b = single(ref);
                check_width(b.operand, y.operand, ref.line);
                if (type_of(b.operand) != rd.type)
                    throw CompileError(ref.line, rd.what + "(" + text(b) + "): " +
                                                     (rd.what == "scale"
                                                          ? std::string("a per-channel scale's multipliers and shifts are int32; ")
                                                          : rd.type == numeric::kFp32
                                                                ? std::string("a tile context computes in fp32; ")
                                                                : "a bias before " + y.operand + "'s scale is added to its " +
                                                                      rd.type + " accumulator values; ") +
                                                     b.operand + " is " + type_of(b.operand));
                if (std::none_of(state_.begin(), state_.end(), [&](const Entry& e) { return covers(e.family, b); }))
                    throw CompileError(ref.line, rd.what + "(" + text(b) + ") @ " + to_string(r.place) + " reads " +
                                                     text(b) + ", which is not provably resident; make it resident first");
            }
        }
        return has;                                     // a non-scaled result's has none (refused above)
    }
    // `acc y in fabric init x` (docs/plans/accumulator-init-stream.md), as the walker checks it: x
    // provably resident and tiled as y (a vector: as y's columns). `init y` on its own accumulator
    // mirrors a resident statement of y, whose residency the fabric takes over.
    void init(const Stmt& s, const Family& y) {
        const Family x = single(*s.init);
        const std::string what = "acc " + text(y) + " init " + text(x);
        if (type_of(x.operand) != acc_of(y.operand))
            throw CompileError(s.line, what + ": " + x.operand + " is " + type_of(x.operand) + " and " + y.operand +
                                       (scaled(y.operand) ? " accumulates in " : " is ") + acc_of(y.operand) +
                                       "; an init is in the accumulator's number system");
        const TensorOperand& xo = operands_.operand(x.operand);
        const TensorOperand& yo = operands_.operand(y.operand);
        if (vectors_.count(x.operand)) {
            if (yo.cols < 2)
                throw CompileError(s.line, what + ": a vector init is a bias row, broadcast down " + y.operand +
                                           "'s rows; " + y.operand + " has one column, so give it a tensor init");
            check_width(x.operand, y.operand, s.line);
            if (y.dims.size() == 2 && !(x.dims[0].a == y.dims[1].a))
                throw CompileError(s.line, what + ": a bias row is indexed by the accumulator's column, " + x.operand +
                                           "[" + text(y.dims[1].a) + "]");
        } else if (xo.rows != yo.rows || xo.cols != yo.cols || xo.tile_rows != yo.tile_rows ||
                   xo.tile_cols != yo.tile_cols) {
            throw CompileError(s.line, what + ": a tensor init is tiled as the accumulator (" + y.operand + " is " +
                                       std::to_string(yo.rows) + "x" + std::to_string(yo.cols) + " in " +
                                       std::to_string(yo.tile_rows) + "x" + std::to_string(yo.tile_cols) + " tiles)");
        } else if (y.dims.size() == 2 && (!(x.dims[0].a == y.dims[0].a) || !(x.dims[1].a == y.dims[1].a))) {
            throw CompileError(s.line, what + ": a tensor init starts the tile in the same place, " + x.operand + "[" +
                                       text(y.dims[0].a) + ", " + text(y.dims[1].a) + "]");
        }
        if (same(x, y)) {
            const auto d = std::find_if(ast_.decls.begin(), ast_.decls.end(), [&](const Decl& e) { return e.name == y.operand; });
            if (d != ast_.decls.end() && d->io == "out")
                throw CompileError(s.line, what + ": " + y.operand + " is declared out, so it has no value to accumulate "
                                           "onto; declare it inout");
            auto it = std::find_if(state_.begin(), state_.end(), [&](const Entry& e) { return same(e.family, y); });
            if (it == state_.end())
                throw CompileError(s.line, what + " reads " + text(y) + ", which is not provably resident (it mirrors "
                                           "a resident statement of " + text(y) + ", as written)");
            not_staged(y, s.line, what + ": the accumulator takes over the residency of");
            live_ -= count(y);
            state_.erase(it);
            return;
        }
        if (std::none_of(state_.begin(), state_.end(), [&](const Entry& e) { return covers(e.family, x); }))
            throw CompileError(s.line, what + " reads " + text(x) + ", which is not provably resident; make it resident "
                                       "first");
    }

    // `acc Y in l2 init X` (#427), as the walker checks it: X in the accumulator's number system,
    // tiled as Y (a vector: as Y's columns, indexed by its column), and provably resident at the
    // acc statement. The Combiner adds it on each tile's first merge.
    void init_l2(const Stmt& s, const Family& y) {
        const Family x = family(*s.init);
        const std::string what = "acc " + text(y) + " in l2 init " + text(x);
        if (x.operand == y.operand)
            throw CompileError(s.line, what + ": accumulating onto " + y.operand + " itself, through the Combiner, is not "
                                       "built yet; init a fabric accumulator with it (`acc .. in fabric init`)");
        if (type_of(x.operand) != acc_of(y.operand))
            throw CompileError(s.line, what + ": " + x.operand + " is " + type_of(x.operand) + " and " + y.operand +
                                       " accumulates in " + acc_of(y.operand) + "; an init is in the accumulator's number system");
        const TensorOperand& xo = operands_.operand(x.operand);
        const TensorOperand& yo = operands_.operand(y.operand);
        if (vectors_.count(x.operand)) {
            if (yo.cols < 2)
                throw CompileError(s.line, what + ": a vector init is a bias row, broadcast down " + y.operand +
                                           "'s rows; " + y.operand + " has one column, so give it a tensor init");
            check_width(x.operand, y.operand, s.line);
            // A bias row is indexed by each tile's column: X's one dimension is Y's second.
            if (y.dims.size() == 2 && !same_dim(x.dims[0], y.dims[1]))
                throw CompileError(s.line, what + ": a bias row is indexed by the accumulator's column, " + x.operand +
                                           "[" + dim_text(y.dims[1]) + "]");
        } else if (xo.rows != yo.rows || xo.cols != yo.cols || xo.tile_rows != yo.tile_rows ||
                   xo.tile_cols != yo.tile_cols) {
            throw CompileError(s.line, what + ": a tensor init is tiled as the accumulator (" + y.operand + " is " +
                                       std::to_string(yo.rows) + "x" + std::to_string(yo.cols) + " in " +
                                       std::to_string(yo.tile_rows) + "x" + std::to_string(yo.tile_cols) + " tiles)");
        } else {
            if (x.dims.size() != y.dims.size())
                throw CompileError(s.line, what + ": a tensor init needs a tensor accumulator, and " + y.operand +
                                           " is a vector");
            // Each tile's init is X's tile in the same place: X names the same tiles as Y.
            for (std::size_t i = 0; i < y.dims.size() && i < x.dims.size(); ++i)
                if (!same_dim(x.dims[i], y.dims[i]))
                    throw CompileError(s.line, what + ": a tensor init starts each tile in the same place, " + x.operand +
                                               "[" + dim_text(y.dims[0]) + ", " + dim_text(y.dims[1]) + "]");
        }
        if (std::none_of(state_.begin(), state_.end(), [&](const Entry& e) { return contains(e.family, x); }))
            throw CompileError(s.line, what + " reads " + text(x) + ", which is not provably resident; make it resident "
                                       "first");
    }
    static bool same_dim(const Dimension& a, const Dimension& b) { return a.all == b.all && (a.all || a.a == b.a); }
    std::string dim_text(const Dimension& d) const { return d.all ? std::string(":") : text(d.a); }

    void check_width(const std::string& b, const std::string& y, int line) const {
        const TensorOperand& bo = operands_.operand(b);
        const TensorOperand& yo = operands_.operand(y);
        if (bo.rows != yo.cols || bo.tile_rows != yo.tile_cols)
            throw CompileError(line, "add: the vector " + b + " must be tiled as " + y + "'s columns (" +
                                     std::to_string(yo.cols) + " tile " + std::to_string(yo.tile_cols) + ")");
    }

    void stmt(const Stmt& s, std::size_t depth, bool definite) {
        switch (s.kind) {
            case Stmt::Kind::For: loop(s, depth, definite); break;
            case Stmt::Kind::Resident:
                for (const TileRef& r : s.tiles) {
                    const Family f = family(r);
                    if (scaled(f.operand))
                        throw CompileError(s.line, "resident " + text(f) + ": " + f.operand + " is a requantized result "
                                                   "(acc " + acc_of(f.operand) + "), computed and stored, never made "
                                                   "resident");
                    for (const Entry& e : state_)
                        if (may_overlap(f, e.family))
                            throw CompileError(s.line, same(f, e.family) || (count(f) == 1 && covers(e.family, f))
                                                           ? text(f) + " is already resident"
                                                           : text(f) + " may already be resident (it can overlap " +
                                                                 text(e.family) + ")");
                    for (const Acc& a : accs_)
                        if (may_overlap(f, a.tile))
                            throw CompileError(s.line, text(f) + " can be an accumulator in the fabric, not a DRAM tile to load");
                    not_retained(f, s.line);
                    release_late_naming(f);
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
                    not_staged(f, s.line, "release");
                    live_ -= count(f);
                    state_.erase(it);
                }
                break;
            case Stmt::Kind::Acc: {
                if (s.in_l2) {
                    acc_l2(s, depth, definite);
                    break;
                }
                if (s.dataflow == "ws" || s.dataflow == "is")       // #395, as the walker refuses it
                    throw CompileError(s.line, "acc ... in fabric dataflow " + s.dataflow + ": a " +
                                               (s.dataflow == "ws" ? "weight" : "input") + "-stationary reduction's "
                                               "partial sums leave the array after every call; merge them in L2 "
                                               "(`acc ... in l2 dataflow " + s.dataflow + "`)");
                const Family y = single(s.out);
                if (s.init) init(s, y);
                for (const Entry& e : state_)
                    if (may_overlap(y, e.family))
                        throw CompileError(s.line, "acc " + text(y) + ": it can be resident in L3; an accumulator lives in the fabric");
                for (const Acc& a : accs_)
                    if (may_overlap(y, a.tile)) throw CompileError(s.line, "acc " + text(y) + ": it may already have an accumulator");
                not_retained(y, s.line);
                if (has_late_) late_room(s.line, "the open of acc " + text(y));
                release_late_naming(y);
                // An unstaged init is copied into L2 as the block opens, before any call.
                if (s.init && unstaged({single(*s.init)}))
                    need_l2(s.line, "acc " + text(y) + " init " + text(single(*s.init)));
                const std::size_t due = late_seq_;      // late stores written before it opened
                accs_.push_back(Acc{y, false, false, depth});
                for (const Stmt& b : s.body) stmt(b, depth, definite);
                Acc& a = *std::find_if(accs_.begin(), accs_.end(), [&](const Acc& x) { return same(x.tile, y); });
                if (!a.fed)
                    throw CompileError(s.line, "acc " + text(y) + " receives no call; an accumulator needs at least one "
                                               "'+->' into it");
                a.closed = true;
                if (has_late_) late_room(s.line, "the close of acc " + text(y));
                release_late_before(due);
                break;
            }
            case Stmt::Kind::Inherit:
                if (depth != 1 || started_)
                    throw CompileError(s.line, "inherit comes first: the tiles arrive resident before the program runs, "
                                               "so `inherit` statements open the program, outside any loop");
                for (const TileRef& r : s.tiles) {
                    const Family f = family(r);
                    const auto d = std::find_if(ast_.decls.begin(), ast_.decls.end(),
                                                [&](const Decl& x) { return x.name == r.name; });
                    if (d != ast_.decls.end() && d->io == "out")
                        throw CompileError(s.line, "inherit " + r.name + "[..]: " + r.name + " is declared out; an "
                                                   "inherited tile arrives with a value (declare it in or inout)");
                    for (const Entry& e : state_)
                        if (may_overlap(f, e.family)) throw CompileError(s.line, text(f) + " is already resident");
                    hold(count(f), s.line, "inherit " + text(f));
                    state_.push_back(Entry{f, false, true});
                    inherited_ += count(f);
                }
                break;
            case Stmt::Kind::Retain:
                for (const TileRef& r : s.tiles) retain(s, r);
                break;
            case Stmt::Kind::Stage: stage(s, depth, definite); break;
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
        if (!rectangular) ++nonrect_;
        vars_[s.var] = Var{Interval{lor.lo, hir.hi - 1}, trips.lo >= 1};
        loop_order_.push_back(s.var);
        const std::uint64_t saved_mult = multiplier_;
        if (rectangular) multiplier_ *= static_cast<std::uint64_t>(hi.constant - lo.constant);

        const std::vector<Entry> before = state_;
        const std::vector<Acc> accs_found = accs_;
        const std::size_t accs_before = accs_.size();
        std::vector<std::size_t> finals_before, merged_before;
        for (const Acc& a : accs_) finals_before.push_back(a.finals.size());
        for (const Acc& a : accs_) merged_before.push_back(a.merged.size());
        // The previous iteration's late stores, still pending as this one starts (l2_held).
        if (trips.hi >= 2)
            for (const Stmt& b : s.body)
                if (b.kind == Stmt::Kind::Store && b.late) {
                    try {
                        const Family f = family(b.out);
                        hold_late(static_cast<std::size_t>(count(f)), f, /*carried=*/true);
                    } catch (const CompileError&) {}        // the store's own check reports it
                }
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
        // A body checked once must not consume what it found: the next iteration would find it
        // gone. An accumulator that existed before the loop must still exist after it, and a tile
        // written before the loop must still be dirty (a store in the body would store it again,
        // unwritten). Clean-to-dirty is fine: that is how a loop writes in place.
        for (const Acc& a : accs_found)
            if (std::none_of(accs_.begin(), accs_.end(), [&](const Acc& x) { return same(x.tile, a.tile); }))
                throw CompileError(s.line, "the loop over " + s.var + " stores the accumulator " + text(a.tile) +
                                           ", which it did not open: a second iteration would store it again");
        for (const Entry& e : before) {
            if (!e.dirty) continue;
            auto it = std::find_if(state_.begin(), state_.end(), [&](const Entry& x) { return same(x.family, e.family); });
            if (it != state_.end() && !it->dirty)
                throw CompileError(s.line, "the loop over " + s.var + " stores " + text(e.family) +
                                           ", written before it: a second iteration would store an unwritten tile");
        }
        if (accs_.size() > accs_before)
            throw CompileError(s.line, "the loop over " + s.var + " leaves the accumulator " + text(accs_.back().tile) +
                                       " unstored: an iteration must store what it accumulates");
        // A final merge in the body runs once per iteration: its tile must change with the loop's
        // variable, or a second iteration merges into a tile already finalized.
        for (std::size_t n = 0; n < finals_before.size() && n < accs_.size(); ++n)
            for (std::size_t f = finals_before[n]; f < accs_[n].finals.size(); ++f) {
                const Family& fin = accs_[n].finals[f];
                const bool varies = std::any_of(fin.dims.begin(), fin.dims.end(), [&](const Dimension& d) {
                    auto c = d.a.coeff.find(s.var);
                    return !d.all && c != d.a.coeff.end() && c->second != 0;
                });
                if (!varies && trips.hi > 1)
                    throw CompileError(fin.line, "call gemm into " + text(fin) + " via ...: the final merge runs in every "
                                                 "iteration of the loop over " + s.var + ", so its second iteration is "
                                                 "after the tile's final merge");
            }
        // A final merge inside the loop finalized the tiles its variable ranges over: past the loop
        // the family stands for all of them (conservative over a loop that does not cover the
        // dimension).
        for (Acc& a : accs_)
            for (auto* fams : {&a.finals, &a.requantized})
                for (Family& f : *fams)
                    for (Dimension& d : f.dims) {
                        auto c = d.a.coeff.find(s.var);
                        if (!d.all && c != d.a.coeff.end() && c->second != 0) d.all = true;
                    }
        // The tiles merged in the body (#446's review). A body that may run zero times guarantees
        // no merge after the loop, so its merges are forgotten. A merge over the loop's variable
        // stands, past the loop, for the tiles the loop provably covered: the whole dimension when
        // the loop runs over exactly its tiles, from 0; otherwise it is forgotten. Forgetting is
        // the conservative side: a later call then checks its init again.
        for (std::size_t n = 0; n < merged_before.size() && n < accs_.size(); ++n)
            if (trips.lo < 1) accs_[n].merged.resize(merged_before[n]);
        for (Acc& a : accs_) {
            std::vector<Family> kept;
            for (Family f : a.merged) {
                bool keep = true;
                const TensorOperand& op = operands_.operand(f.operand);
                const long long tiles[2] = {static_cast<long long>(op.n_tile_rows()), static_cast<long long>(op.n_tile_cols())};
                for (std::size_t i = 0; i < f.dims.size(); ++i) {
                    Dimension& d = f.dims[i];
                    auto c = d.a.coeff.find(s.var);
                    if (d.all || c == d.a.coeff.end() || c->second == 0) continue;
                    const bool whole = d.a.coeff.size() == 1 && c->second == 1 && d.a.constant == 0 && rectangular &&
                                       lo.constant == 0 && i < 2 && hi.constant == tiles[i];
                    if (whole) d.all = true;
                    else keep = false;
                }
                if (keep) kept.push_back(f);
            }
            a.merged = std::move(kept);
        }
        multiplier_ = saved_mult;
        if (!rectangular) --nonrect_;
        loop_order_.pop_back();
        vars_.erase(s.var);
    }

    // An operand's number system (numeric.hpp), as declared.
    std::string type_of(const std::string& operand) const {
        for (const Decl& d : ast_.decls)
            if (d.name == operand) return d.type;
        return numeric::kFp32;
    }
    // The number system it is computed in (`acc`, #427), and whether that is not its type.
    std::string acc_of(const std::string& operand) const {
        for (const Decl& d : ast_.decls)
            if (d.name == operand) return d.acc.empty() ? d.type : d.acc;
        return numeric::kFp32;
    }
    bool scaled(const std::string& operand) const { return acc_of(operand) != type_of(operand); }
    // The number systems a call computes in (docs/plans/typed-value-path.md): a gemm's inputs
    // share one, and its (input, accumulator) pair has a registered MAC; every other tile
    // function computes in fp32.
    void types(const Stmt& s, const std::vector<Family>& args, const Family& y) const {
        if (s.fn == "gemm") {
            const std::string in = type_of(args[0].operand), in2 = type_of(args[1].operand), acc = acc_of(y.operand);
            if (in != in2)
                throw CompileError(s.line, "gemm(" + args[0].operand + ", " + args[1].operand + "): " + args[0].operand +
                                               " is " + in + " and " + args[1].operand + " is " + in2 +
                                               "; a gemm's inputs share a number system");
            if (!numeric::mac(in, acc))
                throw CompileError(s.line, "gemm: no MAC for " + in + " -> " + acc + " (" + y.operand + " is " + acc +
                                               "; registered: " + numeric::mac_names() + ")");
            if (in != numeric::kFp32 && s.alpha && *s.alpha != 1.0)
                throw CompileError(s.line, "gemm: a " + in + " gemm computes alpha 1");
            return;
        }
        for (const Family& f : args)
            if (type_of(f.operand) != numeric::kFp32)
                throw CompileError(s.line, s.fn + " computes in fp32; " + f.operand + " is " + type_of(f.operand));
        if (type_of(y.operand) != numeric::kFp32)
            throw CompileError(s.line, s.fn + " computes in fp32; " + y.operand + " is " + type_of(y.operand));
    }

    // Every call copies its operands into L2 slots the program places, one at a time (each Move,
    // Feed, free), and a writing call's result takes one more (docs/plans/slot-credits-at-lca.md).
    // So a call needs a free slot beyond what is live: the stages', the partials' and the open
    // fabric accumulators' results. As the walker allocates, so the two agree.
    // `slot`: whether the call needs a slot beyond what is held -- a copy of an unstaged operand,
    // or a writing call's result. A call whose operands are all staged, into an open accumulator,
    // needs none.
    void need_l2(int line, const std::string& what, bool slot = true) {
        if (ast_.l2 <= 0)
            throw CompileError(line, what + " copies its operands into L2 slots the program places, and the machine "
                                     "clause declares no L2 (write `l2 = N`)");
        const std::size_t live = l2_held() + (slot ? 1 : 0);
        if (live > static_cast<std::size_t>(ast_.l2))
            throw CompileError(line, what + " needs an L2 slot for a copy, and all " + std::to_string(ast_.l2) +
                                     " are held (" + std::to_string(l2_live_) + " by stages and partials, " +
                                     std::to_string(l2_held() - l2_live_) + " by results)");
        peak_l2_ = std::max(peak_l2_, live);
    }
    bool unstaged(const std::vector<Family>& args) const {
        return std::any_of(args.begin(), args.end(), [&](const Family& a) {
            return std::none_of(staged_.begin(), staged_.end(), [&](const Family& st) { return contains(st, a); });
        });
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
        } else if (s.fn == "add") {
            if (args.size() != 2 || !same(args[0], y))
                throw CompileError(s.line, "add works in place on its first operand: add(y, b) -> y");
            if (!vectors_.count(args[1].operand)) throw CompileError(s.line, "add(y, b): b must be a vector operand");
            check_width(args[1].operand, y.operand, s.line);
        } else if (ActivationFn fn; parse_activation(s.fn, fn)) {
            if (args.size() != 1 || !same(args[0], y))
                throw CompileError(s.line, s.fn + " works in place: " + s.fn + "(y) -> y");
        } else {
            throw CompileError(s.line, "unknown tile function '" + s.fn +
                                       "' (gemm, getrf, laswp, trsm_ll, trsm_ur, add, relu, gelu, silu, atan)");
        }
        if (s.fn != "gemm" && s.accumulate) throw CompileError(s.line, s.fn + " writes its result: '->', not '+->'");
        types(s, args, y);
        if (s.fn != "getrf" && s.fn != "laswp" && s.pivot) throw CompileError(s.line, s.fn + " takes no pivot");

        auto resident = [&](const Family& t) {
            if (std::none_of(state_.begin(), state_.end(), [&](const Entry& e) { return covers(e.family, t); }))
                throw CompileError(s.line, "call " + s.fn + " reads " + text(t) + ", which is not provably resident; "
                                           "make it resident first");
        };
        auto acc = std::find_if(accs_.begin(), accs_.end(), [&](const Acc& a) {
            return a.l2 ? covers(a.tile, y) : same(a.tile, y);
        });
        if (acc != accs_.end() && acc->l2 && !acc->closed) {
            for (const Family& f : acc->finals)
                if (may_overlap(f, y))
                    throw CompileError(s.line, "call gemm into " + text(y) + ", which can be after its final merge (" +
                                               text(f) + ", csp line " + std::to_string(f.line) + "): the call that "
                                               "carries the epilogue is the tile's last");
            if (s.fn != "gemm")
                throw CompileError(s.line, "call " + s.fn + " on the L2 accumulator " + text(y) + ": an L2 accumulator "
                                           "takes gemm '+->', whose result the Combiner merges into its slot");
            need_l2(s.line, "call gemm +-> " + text(y), unstaged(args));   // its unstaged operands' copies
            // `via`: the final merge's epilogue (#427), on the Combiner, after the merge. The walker
            // checks that no call into the tile follows it.
            for (const ResolvedStage& r : resolve_result_context(s.context, target_, [](int line, const std::string& m) {
                     throw CompileError(line, m);
                 })) {
                if (is_block_format(r.op)) acc->formatted = true;
                if (r.place != Place::Combiner)
                    throw CompileError(s.line, "call gemm +-> " + text(y) + " via ... @ " + to_string(r.place) +
                                               ": the epilogue on an L2 accumulator's final merge runs on the Combiner, "
                                               "after the merge" +
                                               std::string(r.place == Place::BmEgress ? "; a bm.egress stage goes on the store"
                                                                                      : "; a fabric stage would act on one partial"));
            }
            if (context(s, y, !s.context.empty())) {
                Family f = y;
                f.line = s.line;
                acc->requantized.push_back(f);
            }
            if (!s.context.empty()) {
                Family f = y;
                f.line = s.line;
                acc->finals.push_back(f);
            }
            for (const Family& a : args)
                if (std::none_of(state_.begin(), state_.end(), [&](const Entry& e) { return covers(e.family, a); }))
                    throw CompileError(s.line, "call " + s.fn + " reads " + text(a) + ", which is not provably resident; "
                                               "make it resident first");
            // A tile's first merge adds the init, so the init is still resident there (#446's review). A
            // call is the first merge of the tiles no earlier call has merged into, as written: each
            // call statement whose tiles are not covered by one before it checks it. It reads only
            // its own tiles' init, as the walker does: X's tile in the same place for a tensor init,
            // and X at the tile's column for a bias row (init_l2).
            const bool first = std::none_of(acc->merged.begin(), acc->merged.end(),
                                            [&](const Family& m) { return contains(m, y); });
            if (acc->init && first) {
                Family need = *acc->init;
                if (need.dims.size() == y.dims.size()) need.dims = y.dims;
                else if (need.dims.size() == 1 && y.dims.size() == 2) need.dims = {y.dims[1]};
                if (std::none_of(state_.begin(), state_.end(), [&](const Entry& e) { return contains(e.family, need); }))
                    throw CompileError(s.line, "call gemm +-> " + text(y) + ": its first merge adds the init " +
                                               text(need) + " (from " + text(*acc->init) + ", csp line " +
                                               std::to_string(acc->init->line) + "), which is not provably resident; "
                                               "make it resident until then");
            }
            if (first) acc->merged.push_back(y);
            if (definite || depth <= acc->depth) acc->fed = true;
            calls_ += multiplier_;
            return;
        }
        if (acc != accs_.end() && !acc->closed) {
            if (s.fn != "gemm")
                throw CompileError(s.line, "call " + s.fn + " on the open accumulator " + text(y) + ": an accumulator "
                                           "takes gemm '+->'; its epilogue goes on its store: store " + text(y) +
                                           " via " + s.fn + " @ fabric");
            if (!s.context.empty())
                throw CompileError(s.line, "an accumulating call's result stays in the fabric; place its epilogue on "
                                           "its store");
            for (const Family& a : args) resident(a);
            // Its first call takes its results' L2 slots, held to its store.
            if (!acc->reserved) {
                l2_results_ += count(acc->tile);
                acc->reserved = true;
            }
            need_l2(s.line, "call " + s.fn + " +-> " + text(y), unstaged(args));
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
        not_staged(y, s.line, "call " + s.fn + " writes");
        need_l2(s.line, "call " + s.fn + " -> " + text(y));        // its operands' copies; then its result's slot
        context(s, y);
        owner->dirty = true;
        calls_ += multiplier_;
    }

    void store(const Stmt& s) {
        const Family y = family(s.out);
        auto acc = std::find_if(accs_.begin(), accs_.end(), [&](const Acc& a) { return same(a.tile, y); });
        if (acc != accs_.end() && acc->l2) {
            if (!acc->closed)
                throw CompileError(s.line, "store " + text(y) + " inside its own accumulator; store it after the acc block");
            if (s.late)
                throw CompileError(s.line, "store " + text(y) + " late: " + text(y) + " is an L2 accumulator, whose "
                                           "partials are already out of the fabric; store it where it is written");
            if (acc->formatted && !s.context.empty())
                throw CompileError(s.line, "store " + text(y) + " via ...: its final merge converted it into a block "
                                           "format, its last operation; no stage may follow it");
            // covers() takes its second family to be a single tile (a ':' there reads as index 0),
            // so a stored family is checked dimension by dimension.
            auto contains = [](const Family& f, const Family& t) {
                for (std::size_t i = 0; i < f.dims.size(); ++i)
                    if (!f.dims[i].all && (t.dims[i].all || !(f.dims[i].a == t.dims[i].a))) return false;
                return f.operand == t.operand;
            };
            // Or the requantized tiles cover it together, one index at a time along its one ':'
            // dimension (C[0, 0] and C[0, 1] cover C[0, :]), with its other dimensions as written.
            auto covered_together = [&](const Family& t) {
                std::size_t slice = t.dims.size();
                for (std::size_t i = 0; i < t.dims.size(); ++i)
                    if (t.dims[i].all) {
                        if (slice != t.dims.size()) return false;   // more than one ':'
                        slice = i;
                    }
                if (slice == t.dims.size()) return false;
                const TensorOperand& o = operands_.operand(t.operand);
                const long long n = static_cast<long long>(slice == 0 ? o.n_tile_rows() : o.n_tile_cols());
                std::set<long long> seen;
                for (const Family& f : acc->requantized) {
                    if (f.operand != t.operand || f.dims[slice].all || !f.dims[slice].a.is_constant()) continue;
                    bool rest = true;
                    for (std::size_t i = 0; i < t.dims.size(); ++i)
                        if (i != slice && !f.dims[i].all && !(f.dims[i].a == t.dims[i].a)) rest = false;
                    if (rest) seen.insert(f.dims[slice].a.constant);
                }
                for (long long k = 0; k < n; ++k)
                    if (!seen.count(k)) return false;
                return true;
            };
            if (scaled(y.operand) && !covered_together(y) &&
                std::none_of(acc->requantized.begin(), acc->requantized.end(),
                             [&](const Family& f) { return contains(f, y); }))
                throw CompileError(s.line, "store " + text(y) + ": it holds " + acc_of(y.operand) + " accumulator values, "
                                           "and not every tile of it was requantized; each tile's final merge needs `via "
                                           "scale(m, shift) @ combiner`");
            if (scaled(y.operand) && !s.context.empty())
                throw CompileError(s.line, "store " + text(y) + " via ...: it was requantized on its final merge; its "
                                           "store takes no stage");
            for (const ResolvedStage& r : resolve_result_context(s.context, target_, [](int line, const std::string& m) {
                     throw CompileError(line, m);
                 }))
                if (r.place != Place::BmEgress)
                    throw CompileError(s.line, "store " + text(y) + " via ... @ " + to_string(r.place) + ": " + text(y) +
                                               " is an L2 accumulator, whose partials the Combiner merged as they "
                                               "drained; put the epilogue on its final merge, or @ bm.egress here");
            context(s, y);
            hold(1, s.line, "store " + text(y) + "'s writeback");      // the writeback's moment in L3, a tile at a time
            live_ -= 1;
            l2_live_ -= count(y);
            accs_.erase(acc);
            stores_ += count(y) * multiplier_;
            return;
        }
        if (acc != accs_.end()) {
            if (!acc->closed)
                throw CompileError(s.line, "store " + text(y) + " inside its own accumulator; store it after the acc block");
            if (s.late && !s.context.empty())
                throw CompileError(s.line, "store " + text(y) + " late via ...: a late store carries no epilogue (it is "
                                           "emitted after the next accumulator, where its vectors' residency is not known)");
            if (scaled(y.operand) && s.late)
                throw CompileError(s.line, "store " + text(y) + " late: it is requantized on its way to memory, and a "
                                           "late store carries no stage; store it with `via scale(m, shift) @ combiner`");
            context(s, y, true);
            hold(1, s.line, "store " + text(y) + "'s writeback");      // the writeback's moment in L3
            live_ -= 1;
            if (acc->reserved) {
                l2_results_ -= count(acc->tile);                       // its results' L2 slots: freed now,
                if (s.late) hold_late(count(acc->tile), acc->tile);   // or, late, held until the walker emits it
            }
            accs_.erase(acc);
            stores_ += multiplier_;
            return;
        }
        auto it = std::find_if(state_.begin(), state_.end(), [&](const Entry& e) { return same(e.family, y); });
        if (s.late)
            throw CompileError(s.line, "store " + text(y) + " late: late defers an accumulator's store; a resident tile "
                                       "is stored where it is written");
        if (it == state_.end())
            throw CompileError(s.line, "store " + text(y) + " must name an accumulator or mirror a resident statement");
        if (!it->dirty && !it->inherited)
            throw CompileError(s.line, "store " + text(y) + ": nothing has written it since it was loaded");
        if (!s.context.empty())
            throw CompileError(s.line, "store " + text(y) + " via ...: a resident tile's store is a DMA write, and the DMA "
                                       "has no vector unit; place the stages on the call that writes it");
        it->dirty = false;
        it->inherited = false;
        stores_ += count(y) * multiplier_;
    }
};

}  // namespace detail

// Validate a program over its structure, without executing it. Throws CompileError, by line.
// With a target, a context's stages are checked against the machine's sites.
inline Validation validate(const Program& ast, const Target* target = nullptr) {
    return detail::Symbolic(ast, target).run();
}
inline Validation validate(const std::string& source, const Target* target = nullptr) {
    return validate(parse(source), target);
}

}  // namespace sw::kpu::program::csp::lang
