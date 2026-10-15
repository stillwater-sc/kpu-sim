// ============================================================================
// include/sw/kpu/program/csp/lang/walk.hpp
// Executing a CSP program's structure (docs/plans/csp-language.md step 1c; ADR 0004 §4).
//
// The Walker runs the AST concretely and INCREMENTALLY: step() executes one leaf statement
// (resident, release, call, store) or one block boundary, and reports the actions it implies
// to a sink. Its own state is bounded by the program, not the problem: the residencies open
// (at most the L3 capacity), the accumulators open, and one frame per enclosing block.
//
// Two sinks:
//   TraceSink   collects every action into a CspProgram (the trace form -- the step-1 compile;
//               small programs, debugging, the record)
//   ActionStream pulls actions one at a time: an interpreter takes the next action when it is
//               ready for it, and nothing is unrolled ahead of it. A 1M x 1M matmul streams.
//
// The checks are the step-1 compiler's, by line (residency explicit only, capacity, operands
// resident, accumulators fed and stored, nothing left resident). The symbolic validator
// (validate.hpp) proves them over the loop structure before anything runs; here they guard
// every program the walker executes.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/csp/lang/place.hpp>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::csp::lang {

class CompileError : public std::runtime_error {
public:
    CompileError(int line, const std::string& what)
        : std::runtime_error(line > 0 ? "csp line " + std::to_string(line) + ": " + what : "csp: " + what),
          line_(line) {}
    int line() const { return line_; }
private:
    int line_;
};

// What the walker reports to its sink.
// Where in the program an action was emitted (docs/plans/domain-flow-debugger.md §3.3): the
// statement's source line and the index of every enclosing loop, outermost first. The walker that
// drives L-B reports it as it emits, so nothing re-walks the program to find where action N came
// from. An `acc` block is not a loop and does not appear.
struct Where {
    int line = 0;                                   // the statement that emitted the action
    struct Loop {
        std::string var;
        long long value = 0, hi = 0;                // this iteration, and the loop's bound (exclusive)
        int line = 0;                               // the `for` statement
        bool operator==(const Loop&) const = default;
    };
    std::vector<Loop> loops;                        // outermost first
    bool operator==(const Where&) const = default;
};

// "line 9  j=1/3 i=2/3": the line, then each loop's index of its bound.
inline std::string to_string(const Where& w) {
    std::string out = "line " + std::to_string(w.line);
    for (std::size_t i = 0; i < w.loops.size(); ++i) {
        out += i ? " " : "  ";
        out += w.loops[i].var + "=" + std::to_string(w.loops[i].value) + "/" + std::to_string(w.loops[i].hi);
    }
    return out;
}

struct Emission {
    Action action;
    std::optional<TileOp> op;       // a Call's tile function (the trace form keeps it in its L0)
    // The L0 ops the walker recorded with this action -- feeds, the call, a drain, a context's
    // stages -- in order. L-B appends them as it executes: its trace (behavioral.hpp).
    std::vector<TileOp> l0;
    bool opens = false;             // the action opens its residency
    bool loaded = false;            //   ... by a Load (else by a result's Writeback)
    bool pinned = false;            //   ... in a slot the program named (`@ slots(...)`)
    Where where;                    // the statement and loop iteration that emitted it
};

// The L3 the program's machine clause declares: `machine flat(l3 = N, banks = B)`.
inline L3Geometry l3_geometry(const Program& ast) {
    return {static_cast<std::uint32_t>(ast.l3), static_cast<std::uint32_t>(ast.banks)};
}

// The operands a program declares (no values: an interpreter supplies them).
inline TileProgram declared_operands(const Program& ast) {
    TileProgram t(ast.name);
    for (const Decl& d : ast.decls) {
        if (t.has_operand(d.name)) throw CompileError(d.line, "operand " + d.name + " is declared twice");
        if (d.rows <= 0 || d.cols <= 0 || d.tile_rows <= 0 || d.tile_cols <= 0)
            throw CompileError(d.line, "operand " + d.name + ": sizes and tile sizes must be positive");
        TensorOperand op;                     // shape only: a 1M x 1M operand must not allocate
        op.name = d.name;
        op.rows = static_cast<Dim>(d.rows);
        op.cols = static_cast<Dim>(d.cols);
        op.tile_rows = static_cast<Dim>(d.tile_rows);
        op.tile_cols = static_cast<Dim>(d.tile_cols);
        t.add_operand(std::move(op));
    }
    return t;
}

template <class Sink>
class Walker {
public:
    // `target`: the machine whose sites a context's stages must run on; null = unchecked.
    Walker(const Program& ast, Sink& sink, const Target* target = nullptr) : ast_(ast), sink_(sink), target_(target) {
        if (ast_.machine != "flat")
            throw CompileError(0, "machine " + ast_.machine + ": level 2 (a distributed machine) is not built yet; "
                                  "this step compiles level-1 programs (machine flat)");
        if (ast_.l3 <= 0) throw CompileError(0, "machine flat: the L3 capacity must be positive");
        cap_ = static_cast<std::size_t>(ast_.l3);
        const auto policy = parse_place_policy(ast_.place);
        if (!policy) throw CompileError(0, "place = " + ast_.place + ": fifo, lowest or bank-spread");
        if (ast_.banks <= 0) throw CompileError(0, "machine flat: banks must be positive");
        alloc_.emplace(static_cast<std::uint32_t>(cap_), static_cast<std::uint32_t>(ast_.banks), *policy);
        operands_ = declared_operands(ast_);
        for (const Decl& d : ast_.decls) if (d.is_vector) vectors_.insert(d.name);
        frames_.push_back(Frame{&ast_.body, 0, Frame::Kind::Program, {}, 0, 0, {}});
    }

    const TileProgram& operands() const { return operands_; }
    std::size_t capacity() const { return cap_; }
    bool done() const { return frames_.empty(); }

    // Execute one leaf statement or one block boundary. False when the program has ended.
    bool step() {
        if (frames_.empty()) return false;
        Frame& f = frames_.back();
        if (f.pc < f.body->size()) {
            const Stmt& s = (*f.body)[f.pc++];
            enter(s);
            return true;
        }
        switch (f.kind) {
            case Frame::Kind::Loop:
                if (++f.value < f.hi) {
                    f.pc = 0;
                    env_[f.var] = f.value;
                    return true;
                }
                env_.erase(f.var);
                frames_.pop_back();
                return true;
            case Frame::Kind::Acc: {
                const std::string k = f.acc;
                const int line = f.line;
                frames_.pop_back();
                if (acc_calls_[k] == 0)
                    throw CompileError(line, "acc " + k + " receives no call; an accumulator needs at least one "
                                             "'+->' into it");
                acc_[k] = AccState::Closed;
                return true;
            }
            case Frame::Kind::Program:
                frames_.pop_back();
                for (const auto& [k, r] : resident_)
                    throw CompileError(0, k + " is still resident at the end of the program; release it");
                for (const auto& [k, state] : acc_)
                    if (state != AccState::Stored) throw CompileError(0, "the accumulator " + k + " is never stored");
                return false;
        }
        return false;
    }

    void run() { while (step()) {} }

private:
    enum class AccState { Open, Closed, Stored };
    struct Frame {
        enum class Kind { Program, Loop, Acc };
        const std::vector<Stmt>* body;
        std::size_t pc;
        Kind kind;
        std::string var;
        long long value, hi;
        std::string acc;
        int line = 0;
    };
    struct Live { std::size_t id; bool dirty; bool inherited = false; };

    const Program& ast_;
    Sink& sink_;
    const Target* target_ = nullptr;
    std::set<std::string> vectors_;                        // operands declared `vector`
    std::size_t cap_ = 0;
    TileProgram operands_;
    std::vector<Frame> frames_;
    std::map<std::string, long long> env_;
    std::map<std::string, Live> resident_;                 // open residencies (<= capacity)
    std::set<std::string> retained_;                       // residencies Retain ended: their slots stay held
    bool started_ = false;                                 // a statement other than `inherit` has run
    std::map<std::string, AccState> acc_;                  // accumulators not yet stored
    std::map<std::string, std::size_t> acc_calls_;
    std::size_t next_residency_ = 0;
    // Placement (ADR 0004 §5): every open residency's L3 address, and who holds each live slot.
    std::optional<L3Allocator> alloc_;
    std::map<std::size_t, L3Address> places_;              // residency -> its address (open ones)
    std::map<std::uint32_t, std::string> slot_owner_;      // live slot -> the tile in it

    // ---- expressions and tiles ----
    long long eval(const Expr& e, int line) const {
        switch (e.kind) {
            case Expr::Kind::Int: return e.value;
            case Expr::Kind::Var: {
                auto it = env_.find(e.name);
                if (it == env_.end()) throw CompileError(line, "'" + e.name + "' is not a loop variable in scope");
                return it->second;
            }
            case Expr::Kind::Bin: {
                const long long a = eval(e.args.at(0), line), b = eval(e.args.at(1), line);
                return e.op == '+' ? a + b : e.op == '-' ? a - b : a * b;
            }
        }
        return 0;
    }
    std::vector<TileCoord> expand(const TileRef& r) const {
        if (!operands_.has_operand(r.name)) throw CompileError(r.line, "'" + r.name + "' is not a declared operand");
        const TensorOperand& op = operands_.operand(r.name);
        const bool vec = vectors_.count(r.name) != 0;   // a vector's tiles run down its one column
        if (vec && r.index.size() != 1) throw CompileError(r.line, r.name + " is a vector: index it [j]");
        if (!vec && r.index.size() != 2) throw CompileError(r.line, r.name + " is a tensor: index it [row, col]");
        auto range = [&](const Index& ix, Dim n) {
            std::vector<Dim> v;
            if (ix.all) {
                for (Dim i = 0; i < n; ++i) v.push_back(i);
                return v;
            }
            const long long x = eval(ix.expr, r.line);
            if (x < 0 || x >= static_cast<long long>(n))
                throw CompileError(r.line, r.name + ": tile index " + std::to_string(x) + " is outside 0.." +
                                           std::to_string(n));
            v.push_back(static_cast<Dim>(x));
            return v;
        };
        std::vector<TileCoord> out;
        if (vec) {
            for (Dim ti : range(r.index[0], op.n_tile_rows())) out.push_back(TileCoord{r.name, ti, 0});
            return out;
        }
        for (Dim ti : range(r.index[0], op.n_tile_rows()))
            for (Dim tj : range(r.index[1], op.n_tile_cols())) out.push_back(TileCoord{r.name, ti, tj});
        return out;
    }
    TileCoord one(const TileRef& r) const {
        for (const Index& ix : r.index)
            if (ix.all) throw CompileError(r.line, r.name + ": a single tile is required here, not a slice");
        return expand(r).front();
    }

    // ---- emission ----
    static std::size_t process_of(Action::Kind k) {
        switch (k) {
            case Action::Kind::Load: case Action::Kind::Store:     return 0;   // dma
            case Action::Kind::Move: case Action::Kind::Writeback: return 1;   // bm
            case Action::Kind::Feed: case Action::Kind::Drain:     return 2;   // str
            case Action::Kind::Call:                               return 3;   // cf
            default:                                               return kNone;
        }
    }
    void emit(Action::Kind kind, const TileCoord& t, std::size_t residency = kNone, std::size_t l0 = kNone,
              const TileOp* op = nullptr, bool opens = false, bool loaded = false, bool accumulate = false,
              std::vector<csp::Stage> context = {}) {
        Emission e;
        e.action.kind = kind;
        e.action.tile = t;
        e.action.process = process_of(kind);
        e.action.l0_op = l0;
        e.action.residency = residency;
        e.action.accumulate = accumulate;
        e.action.context = std::move(context);
        if (op) e.op = *op;
        e.opens = opens;
        e.loaded = loaded;
        e.where.line = line_;
        for (const Frame& f : frames_)
            if (f.kind == Frame::Kind::Loop) e.where.loops.push_back({f.var, f.value, f.hi, f.line});
        if (residency != kNone) {
            const auto pl = places_.find(residency);
            if (pl == places_.end()) throw std::logic_error("csp walk: residency " + std::to_string(residency) + " has no place");
            e.action.place = pl->second;
            e.pinned = opens && pinned_.count(residency) != 0;
            // The trace records the L3 transaction with its address (opset 1.2), and the action
            // implements it.
            TileOp l3;
            l3.kind = l3_kind(kind);
            l3.outputs = {t};
            l3.l3 = pl->second;
            e.action.l0_op = sink_.l0(l3);
        }
        sink_.emit(e);
        // A residency's slot is free once its Release is out; a Retain keeps it past the program.
        if (residency != kNone && (kind == Action::Kind::Release || kind == Action::Kind::Retain)) {
            const L3Address a = places_.at(residency);
            if (kind == Action::Kind::Release) {
                alloc_->give(a);
                slot_owner_.erase(a.slot);
            }
            places_.erase(residency);
            pinned_.erase(residency);
        }
    }
    std::set<std::size_t> pinned_;                          // residencies the program placed itself
    int line_ = 0;                                          // the statement being executed

    // Place residency `id` for tile `t`: by the policy, or in `slot` when the program names one.
    void place(std::size_t id, const TileCoord& t, std::optional<std::uint32_t> slot = std::nullopt, int line = 0) {
        L3Address a;
        if (slot) {
            if (const auto why = alloc_->take_at(*slot, a)) {
                const auto owner = slot_owner_.find(*slot);
                throw CompileError(line, "resident " + t.to_string() + " @ slot " + std::to_string(*slot) + ": " + *why +
                                         (owner != slot_owner_.end() ? " (" + owner->second + ")" : std::string()));
            }
            pinned_.insert(id);
        } else {
            a = alloc_->take();
        }
        places_[id] = a;
        slot_owner_[a.slot] = t.to_string();
    }
    void need_slot(int line, const std::string& why) const {
        if (resident_.size() + retained_.size() >= cap_)
            throw CompileError(line, why + " needs an L3 slot, and all " + std::to_string(cap_) + " are held" +
                                     (retained_.empty() ? std::string()
                                                        : " (" + std::to_string(retained_.size()) + " retained)"));
    }
    void not_retained(int line, const std::string& k) const {
        if (retained_.count(k))
            throw CompileError(line, k + " is retained: its slot is held for the operator after this one, and this "
                                     "program cannot name it again");
    }

    // ---- statements ----
    // A result's `via` list, resolved and checked against the target, as the stages of its Drain
    // (fabric, str.drain) and of its Writeback (bm.egress). An add's vector must be resident in
    // L3 -- the stage reads it there -- and as long as the tile is wide.
    std::pair<std::vector<csp::Stage>, std::vector<csp::Stage>> result_context(const Stmt& s, const TileCoord& y) {
        std::pair<std::vector<csp::Stage>, std::vector<csp::Stage>> out;
        const auto resolved = resolve_result_context(s.context, target_, [](int line, const std::string& m) {
            throw CompileError(line, m);
        });
        for (const ResolvedStage& r : resolved) {
            csp::Stage st;
            st.op = r.op;
            st.place = r.place;
            if (r.arg) {
                if (!vectors_.count(r.arg->name))
                    throw CompileError(r.arg->line, "add(" + r.arg->name + "[..]): add takes a vector operand");
                st.arg = one(*r.arg);
                const std::string k = st.arg.to_string();
                if (!resident_.count(k))
                    throw CompileError(r.arg->line, "add(" + k + ") @ " + to_string(r.place) + " reads " + k +
                                                    ", which is not resident; make it resident first");
                check_width(st.arg, y, r.arg->line);
            }
            (r.place == Place::BmEgress ? out.second : out.first).push_back(st);
        }
        return out;
    }
    // A context's arithmetic, recorded as L0 ops where it applies (after the result is computed,
    // before it leaves the fabric): the trace's L0 is then a complete record of what the
    // program computes, and TileProgramReference over it is the oracle for a fused program as
    // much as for an unfused one. The stages themselves ride the moves.
    void record_stages(const TileCoord& y, const std::vector<csp::Stage>& drain, const std::vector<csp::Stage>& wb) {
        for (const auto* v : {&drain, &wb})
            for (const csp::Stage& st : *v) {
                TileOp op;
                op.label = std::string("context @ ") + to_string(st.place);
                if (st.op == VeOp::Add) {
                    op.kind = TileOpKind::BiasAdd;
                    op.inputs = {st.arg};
                } else {
                    op.kind = TileOpKind::Activation;
                    op.act = activation_of(st.op);
                }
                op.outputs = {y};
                (void)sink_.l0(op);
            }
    }

    // A bias is tiled as the columns it adds to (the symbolic validator's rule, so the two agree).
    void check_width(const TileCoord& b, const TileCoord& y, int line) const {
        const TensorOperand& bo = operands_.operand(b.operand);
        const TensorOperand& yo = operands_.operand(y.operand);
        if (bo.rows != yo.cols || bo.tile_rows != yo.tile_cols)
            throw CompileError(line, "add: the vector " + b.operand + " must be tiled as " + y.operand +
                                     "'s columns (" + std::to_string(yo.cols) + " tile " + std::to_string(yo.tile_cols) +
                                     ")");
    }

    void enter(const Stmt& s) {
        if (s.kind != Stmt::Kind::Inherit) started_ = true;
        line_ = s.line;                                     // every action a statement emits is from it
        switch (s.kind) {
            case Stmt::Kind::For: {
                if (env_.count(s.var)) throw CompileError(s.line, "loop variable '" + s.var + "' shadows an outer one");
                const long long lo = eval(s.lo, s.line), hi = eval(s.hi, s.line);
                if (lo < hi) {
                    frames_.push_back(Frame{&s.body, 0, Frame::Kind::Loop, s.var, lo, hi, {}, s.line});
                    env_[s.var] = lo;
                }
                break;
            }
            case Stmt::Kind::Acc: {
                const TileCoord y = one(s.out);
                const std::string k = y.to_string();
                if (resident_.count(k)) throw CompileError(s.line, "acc " + k + ": it is resident in L3; an accumulator lives in the fabric");
                not_retained(s.line, k);
                if (acc_.count(k)) throw CompileError(s.line, "acc " + k + ": it already has an accumulator");
                acc_[k] = AccState::Open;
                acc_calls_[k] = 0;
                frames_.push_back(Frame{&s.body, 0, Frame::Kind::Acc, {}, 0, 0, k, s.line});
                break;
            }
            case Stmt::Kind::Resident: {
                std::vector<TileCoord> ts;
                for (const TileRef& r : s.tiles)
                    for (const TileCoord& t : expand(r)) ts.push_back(t);
                // `@ slots(lo..hi)`: the program's own placement, one slot per tile, in order.
                long long next_slot = 0;
                if (s.slot_lo) {
                    const long long lo = eval(*s.slot_lo, s.line), hi = eval(*s.slot_hi, s.line);
                    if (hi - lo != static_cast<long long>(ts.size()) || lo < 0)
                        throw CompileError(s.line, "resident @ slots(" + std::to_string(lo) + ".." + std::to_string(hi) +
                                                   "): " + std::to_string(hi - lo) + " slots for " +
                                                   std::to_string(ts.size()) + " tiles; name one slot per tile");
                    next_slot = lo;
                }
                for (const TileCoord& t : ts) {
                    const std::string k = t.to_string();
                    if (resident_.count(k)) throw CompileError(s.line, k + " is already resident");
                    if (acc_.count(k)) throw CompileError(s.line, k + " is an accumulator in the fabric, not a DRAM tile to load");
                    not_retained(s.line, k);
                    need_slot(s.line, "resident " + k);
                    const std::size_t id = next_residency_++;
                    resident_[k] = Live{id, false};
                    if (s.slot_lo) place(id, t, static_cast<std::uint32_t>(next_slot++), s.line);
                    else place(id, t);
                    emit(Action::Kind::Load, t, id, kNone, nullptr, true, true);
                }
                break;
            }
            case Stmt::Kind::Release:
                for (const TileRef& r : s.tiles)
                    for (const TileCoord& t : expand(r)) {
                        const std::string k = t.to_string();
                        auto it = resident_.find(k);
                        if (it == resident_.end()) throw CompileError(s.line, "release " + k + ": it is not resident");
                        if (it->second.dirty)
                            throw CompileError(s.line, "release " + k + ": a call wrote it and it is not stored; store it first");
                        emit(Action::Kind::Release, t, it->second.id);
                        resident_.erase(it);
                    }
                break;
            case Stmt::Kind::Inherit: {
                // The tiles the operator before this one retained: resident from the start, with no
                // Load. So they come first, at the top of the program, before anything else runs.
                if (started_ || frames_.size() != 1)
                    throw CompileError(s.line, "inherit comes first: the tiles arrive resident before the program "
                                               "runs, so `inherit` statements open the program, outside any loop");
                for (const TileRef& r : s.tiles) {
                    const auto d = std::find_if(ast_.decls.begin(), ast_.decls.end(),
                                                [&](const Decl& x) { return x.name == r.name; });
                    if (d != ast_.decls.end() && d->io == "out")
                        throw CompileError(s.line, "inherit " + r.name + "[..]: " + r.name + " is declared out; an "
                                                   "inherited tile arrives with a value (declare it in or inout)");
                    for (const TileCoord& t : expand(r)) {
                        const std::string k = t.to_string();
                        if (resident_.count(k)) throw CompileError(s.line, k + " is already resident");
                        need_slot(s.line, "inherit " + k);
                        const std::size_t id = next_residency_++;
                        resident_[k] = Live{id, false, true};
                        place(id, t);
                        emit(Action::Kind::Inherit, t, id, kNone, nullptr, true, false);
                    }
                }
                break;
            }
            case Stmt::Kind::Retain:
                for (const TileRef& r : s.tiles)
                    for (const TileCoord& t : expand(r)) retain(s, t);
                break;
            case Stmt::Kind::Call: call(s); break;
            case Stmt::Kind::Store:
                for (const TileCoord& y : expand(s.out)) store(s, y);
                break;
            case Stmt::Kind::Distribute:
            case Stmt::Kind::Broadcast:
                throw CompileError(s.line, "distribute and broadcast are level 2 (a distributed machine), not built yet");
        }
    }

    // Out of L3 and into the fabric. An accumulating call's operand carries its L0 Feed, recorded
    // after the Move's L3 transaction, so the trace moves the tile out of its slot before it feeds.
    void deliver(const TileCoord& t, int line, const std::string& fn, const TileOp* feed = nullptr) {
        const std::string k = t.to_string();
        auto it = resident_.find(k);
        if (it == resident_.end())
            throw CompileError(line, "call " + fn + " reads " + k + ", which is not resident; make it resident first");
        emit(Action::Kind::Move, t, it->second.id);
        emit(Action::Kind::Feed, t, kNone, feed ? sink_.l0(*feed) : kNone);
    }

    static TileOpKind l3_kind(Action::Kind k) {
        switch (k) {
            case Action::Kind::Load:      return TileOpKind::L3Load;
            case Action::Kind::Store:     return TileOpKind::L3Store;
            case Action::Kind::Move:      return TileOpKind::L3Move;
            case Action::Kind::Writeback: return TileOpKind::L3Writeback;
            case Action::Kind::Release:   return TileOpKind::L3Release;
            case Action::Kind::Inherit:   return TileOpKind::L3Inherit;
            case Action::Kind::Retain:    return TileOpKind::L3Retain;
            default: break;
        }
        throw std::logic_error(std::string("csp walk: ") + to_string(k) + " is not an L3 transaction");
    }

    void call(const Stmt& s) {
        std::vector<TileCoord> args;
        for (const TileRef& r : s.tiles) args.push_back(one(r));
        const TileCoord y = one(s.out);
        const std::string yk = y.to_string();
        auto same = [](const TileCoord& a, const TileCoord& b) { return a.operand == b.operand && a.ti == b.ti && a.tj == b.tj; };

        TileOp op;
        op.label = "line " + std::to_string(s.line);
        if (s.fn == "gemm") {
            if (args.size() != 2) throw CompileError(s.line, "gemm takes two operands: gemm(a, b) +-> y");
            if (!s.accumulate) throw CompileError(s.line, "gemm accumulates: write '+->'");
            op.kind = TileOpKind::MatMulAccum;
            op.inputs = {args[0], args[1]};
            op.outputs = {y};
            op.alpha = static_cast<float>(s.alpha.value_or(1.0));
        } else if (s.fn == "getrf" || s.fn == "laswp") {
            if (args.size() != 1 || !same(args[0], y))
                throw CompileError(s.line, s.fn + " works in place: " + s.fn + "(x) -> x pivot p");
            if (!s.pivot) throw CompileError(s.line, s.fn + " needs its pivot slot: pivot p");
            op.kind = s.fn == "getrf" ? TileOpKind::LuDiagFactor : TileOpKind::PivotApply;
            op.outputs = {y};
            op.pivot_slot = static_cast<int>(eval(*s.pivot, s.line));
        } else if (s.fn == "trsm_ll" || s.fn == "trsm_ur") {
            if (args.size() != 2 || !same(args[1], y))
                throw CompileError(s.line, s.fn + " works in place on its second operand: " + s.fn + "(d, x) -> x");
            op.kind = s.fn == "trsm_ll" ? TileOpKind::TrsmLowerLeft : TileOpKind::TrsmUpperRight;
            op.inputs = {args[0]};
            op.outputs = {y};
        } else if (s.fn == "add") {
            if (args.size() != 2 || !same(args[0], y))
                throw CompileError(s.line, "add works in place on its first operand: add(y, b) -> y");
            if (!vectors_.count(args[1].operand)) throw CompileError(s.line, "add(y, b): b must be a vector operand");
            check_width(args[1], y, s.line);
            op.kind = TileOpKind::BiasAdd;
            op.inputs = {args[1]};
            op.outputs = {y};
        } else if (ActivationFn fn; parse_activation(s.fn, fn)) {
            if (args.size() != 1 || !same(args[0], y))
                throw CompileError(s.line, s.fn + " works in place: " + s.fn + "(y) -> y");
            op.kind = TileOpKind::Activation;
            op.act = fn;
            op.outputs = {y};
        } else {
            throw CompileError(s.line, "unknown tile function '" + s.fn +
                                       "' (gemm, getrf, laswp, trsm_ll, trsm_ur, add, relu, gelu, silu, atan)");
        }
        if (s.fn != "gemm" && s.accumulate) throw CompileError(s.line, s.fn + " writes its result: '->', not '+->'");
        if (s.fn != "getrf" && s.fn != "laswp" && s.pivot) throw CompileError(s.line, s.fn + " takes no pivot");

        auto acc = acc_.find(yk);
        if (acc != acc_.end() && acc->second == AccState::Open) {
            if (s.fn != "gemm")
                throw CompileError(s.line, "call " + s.fn + " on the open accumulator " + yk + ": an accumulator takes "
                                           "gemm '+->'; its epilogue goes on its store: store " + yk + " via " + s.fn +
                                           " @ fabric");
            if (!s.context.empty())
                throw CompileError(s.line, "an accumulating call's result stays in the fabric; place its epilogue on "
                                           "its store");
            // An output-stationary chain: the operands are fed; the result accumulates in the fabric.
            for (std::size_t i = 0; i < args.size(); ++i) {
                TileOp feed;
                feed.kind = TileOpKind::Feed;
                feed.port_kind = PortKind::Input;
                feed.port = i == 0 ? "West" : "North";
                feed.inputs = {args[i]};
                deliver(args[i], s.line, s.fn, &feed);
            }
            emit(Action::Kind::Call, y, kNone, sink_.l0(op), &op, false, false, true);
            ++acc_calls_[yk];
            return;
        }
        if (acc != acc_.end())
            throw CompileError(s.line, "call " + s.fn + " writes " + yk + ", whose accumulator is closed; store it first");
        auto res = resident_.find(yk);
        if (res == resident_.end())
            throw CompileError(s.line, "call " + s.fn + " writes " + yk + ", which is neither resident nor an open "
                                       "accumulator");
        std::vector<TileCoord> operands = args;
        if (std::none_of(args.begin(), args.end(), [&](const TileCoord& a) { return same(a, y); })) operands.push_back(y);
        for (const TileCoord& a : operands) deliver(a, s.line, s.fn);
        auto [drain_ctx, wb_ctx] = result_context(s, y);
        emit(Action::Kind::Call, y, kNone, sink_.l0(op), &op);
        record_stages(y, drain_ctx, wb_ctx);
        emit(Action::Kind::Drain, y, kNone, kNone, nullptr, false, false, false, std::move(drain_ctx));
        emit(Action::Kind::Writeback, y, resident_.at(yk).id, kNone, nullptr, false, false, false, std::move(wb_ctx));
        resident_.at(yk).dirty = true;
    }

    void store(const Stmt& s, const TileCoord& y) {
        const std::string k = y.to_string();
        auto acc = acc_.find(k);
        if (acc != acc_.end() && acc->second == AccState::Open)
            throw CompileError(s.line, "store " + k + " inside its own accumulator; store it after the acc block");
        if (acc != acc_.end()) {
            // Out of the fabric: drained, written back into an L3 slot for the moment, stored, freed.
            need_slot(s.line, "store " + k + "'s writeback");
            auto [drain_ctx, wb_ctx] = result_context(s, y);
            TileOp drain;
            drain.kind = TileOpKind::Drain;
            drain.port_kind = PortKind::Output;
            drain.port = "South";
            drain.outputs = {y};
            record_stages(y, drain_ctx, wb_ctx);
            const std::size_t d = sink_.l0(drain);
            const std::size_t id = next_residency_++;
            place(id, y);
            emit(Action::Kind::Drain, y, kNone, d, nullptr, false, false, false, std::move(drain_ctx));
            emit(Action::Kind::Writeback, y, id, d, nullptr, true, false, false, std::move(wb_ctx));
            emit(Action::Kind::Store, y, id, d);
            emit(Action::Kind::Release, y, id, d);
            acc_.erase(acc);
            acc_calls_.erase(k);
            return;
        }
        auto it = resident_.find(k);
        if (it == resident_.end()) throw CompileError(s.line, "store " + k + ": it is neither resident nor an accumulator");
        // An inherited tile may hold a result the operator before this one did not store: storing it
        // is how a later operator keeps the orchestrator's promise that a retained tile is stored.
        if (!it->second.dirty && !it->second.inherited)
            throw CompileError(s.line, "store " + k + ": nothing has written it since it was loaded");
        if (!s.context.empty())
            throw CompileError(s.line, "store " + k + " via ...: a resident tile's store is a DMA write, and the DMA has "
                                       "no vector unit; place the stages on the call that writes it");
        emit(Action::Kind::Store, y, it->second.id);
        it->second.dirty = false;
        it->second.inherited = false;
    }

    // `retain y`: the residency ends without a Release -- its slot and the tile outlive the
    // program. A resident tile (written or not) keeps its slot; an accumulator is drained and
    // written back into a slot it keeps, with its epilogue on the way, and never stored.
    void retain(const Stmt& s, const TileCoord& y) {
        const std::string k = y.to_string();
        auto acc = acc_.find(k);
        if (acc != acc_.end()) {
            if (acc->second == AccState::Open)
                throw CompileError(s.line, "retain " + k + " inside its own accumulator; retain it after the acc block");
            need_slot(s.line, "retain " + k + "'s writeback");
            auto [drain_ctx, wb_ctx] = result_context(s, y);
            TileOp drain;
            drain.kind = TileOpKind::Drain;
            drain.port_kind = PortKind::Output;
            drain.port = "South";
            drain.outputs = {y};
            record_stages(y, drain_ctx, wb_ctx);
            const std::size_t d = sink_.l0(drain);
            const std::size_t id = next_residency_++;
            place(id, y);
            emit(Action::Kind::Drain, y, kNone, d, nullptr, false, false, false, std::move(drain_ctx));
            emit(Action::Kind::Writeback, y, id, d, nullptr, true, false, false, std::move(wb_ctx));
            emit(Action::Kind::Retain, y, id, d);
            acc_.erase(acc);
            acc_calls_.erase(k);
            retained_.insert(k);
            return;
        }
        auto it = resident_.find(k);
        if (it == resident_.end()) throw CompileError(s.line, "retain " + k + ": it is neither resident nor an accumulator");
        if (!s.context.empty())
            throw CompileError(s.line, "retain " + k + " via ...: a resident tile stays where it is, so nothing moves it "
                                       "to run the stages on; place them on the call that writes it");
        emit(Action::Kind::Retain, y, it->second.id);
        resident_.erase(it);
        retained_.insert(k);
    }
};

// ---- the trace sink: every action, into a CspProgram ----------------------------------------
class TraceSink {
public:
    explicit TraceSink(CspProgram& p) : p_(p) {}
    std::size_t l0(const TileOp& op) {
        p_.source.push(op);
        return p_.source.ops().size() - 1;
    }
    void emit(const Emission& e) {
        const Action& a = e.action;
        p_.actions.push_back(a);
        const std::size_t i = p_.actions.size() - 1;
        if (a.process != kNone) p_.processes[a.process].actions.push_back(i);
        if (a.residency == kNone) return;
        if (e.opens) {
            if (a.residency != p_.residencies.size())
                throw std::logic_error("csp trace: residencies must open in order");
            Residency r;
            r.tile = a.tile;
            r.open = i;
            r.loaded = e.loaded;
            r.inherited = a.kind == Action::Kind::Inherit;
            r.dirty = !e.loaded && !r.inherited;
            if (a.place) r.place = *a.place;
            r.pinned = e.pinned;
            p_.residencies.push_back(r);
            return;
        }
        Residency& r = p_.residencies.at(a.residency);
        switch (a.kind) {
            case Action::Kind::Move:      ++r.consumers; break;
            case Action::Kind::Store:     ++r.consumers; r.dirty = false; break;
            case Action::Kind::Writeback: r.dirty = true; break;
            case Action::Kind::Release:   r.release = i; break;
            case Action::Kind::Retain:    r.retained = true; break;
            default: break;
        }
    }
private:
    CspProgram& p_;
};

// ---- the stream: actions pulled one at a time ------------------------------------------------
//
// An interpreter calls next() when it is ready for the next action. The walker runs only as far
// as that action needs; the buffer holds the actions of one statement at most.
class ActionStream {
public:
    // The stream owns its program: the walker's frames point into it.
    // With a target, a context's stages are checked against the machine's sites.
    explicit ActionStream(Program ast, std::optional<Target> target = std::nullopt)
        : ast_(std::move(ast)), target_(std::move(target)), sink_(buffer_),
          walker_(ast_, sink_, target_ ? &*target_ : nullptr) {}
    ActionStream(const ActionStream&) = delete;
    ActionStream& operator=(const ActionStream&) = delete;

    const TileProgram& operands() const { return walker_.operands(); }
    std::size_t capacity() const { return walker_.capacity(); }

    // The next action, or nullopt when the program has ended.
    std::optional<Emission> next() {
        while (buffer_.empty()) {
            if (!walker_.step() && buffer_.empty()) {
                if (!sink_.pending_.empty())
                    throw std::logic_error("csp stream: L0 ops recorded after the program's last action");
                return std::nullopt;
            }
        }
        Emission e = std::move(buffer_.front());
        buffer_.pop_front();
        ++emitted_;
        return e;
    }
    std::size_t emitted() const { return emitted_; }
    std::size_t buffered() const { return buffer_.size(); }   // bounded by one statement's actions

private:
    struct BufferSink {
        explicit BufferSink(std::deque<Emission>& b) : b_(b) {}
        // The stream carries the ops themselves, on the next action the walker emits.
        std::size_t l0(const TileOp& op) {
            pending_.push_back(op);
            return kNone;
        }
        void emit(const Emission& e) {
            b_.push_back(e);
            b_.back().l0 = std::move(pending_);
            pending_.clear();
        }
        std::deque<Emission>& b_;
        std::vector<TileOp> pending_;
    };
    Program ast_;
    std::optional<Target> target_;
    std::deque<Emission> buffer_;
    BufferSink sink_;
    Walker<BufferSink> walker_;
    std::size_t emitted_ = 0;
};

}  // namespace sw::kpu::program::csp::lang
