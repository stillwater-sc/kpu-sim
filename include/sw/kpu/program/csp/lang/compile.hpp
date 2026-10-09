// ============================================================================
// include/sw/kpu/program/csp/lang/compile.hpp
// The CSP language -> the CSP program IR (docs/plans/csp-language.md §3.3).
//
// One walk validates and lowers. Loops are unrolled (their bounds are integers); every
// statement becomes the actions of the four level-1 processes; and everything the program
// claims about residency is checked statically, by line, before anything runs:
//
//   resident X      Load X (dma)                       X not already resident; L3 not full
//   release X       Release X (its credit returns)     X resident; not written and unstored
//   call f(..) -> y each operand: Move (bm), Feed (str); then Call (cf); an in-place result:
//                   Drain (str), Writeback (bm)        every operand resident in L3
//   acc y in fabric { ... }   y accumulates in the fabric (from zero) and holds no L3 slot
//   store y         from an accumulator: Drain, Writeback, Store, Release (a slot for the
//                   writeback's moment); a resident, written tile: Store
//
// Residency is explicit only (decision Q2): a call on a tile the program did not make resident
// is an error, never an implicit load. The program ends with nothing resident and every
// accumulator stored.
//
// Functions (tile functions -- the inner loops, collapsed):
//   gemm(a, b) +-> y [alpha s]   y += s . a . b      (in an accumulator, or in place in L3)
//   getrf(x) -> x pivot p        factor x in place, recording pivots in slot p
//   laswp(x) -> x pivot p        apply slot p's row swaps to x
//   trsm_ll(d, x) -> x           x := unit-lower(d)^-1 . x
//   trsm_ur(d, x) -> x           x := x . upper(d)^-1
//
// Level 1 only in this step: a grid machine, `distribute` / `broadcast`, `vector` operands and
// tile contexts (`via`) are refused by name (plan steps 2 and level 2).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
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

namespace detail {

class Compiler {
public:
    explicit Compiler(const Program& ast) : ast_(ast) {}

    CspProgram run() {
        if (ast_.machine != "flat")
            throw CompileError(0, "machine " + ast_.machine + ": level 2 (a distributed machine) is not built yet; "
                                  "this step compiles level-1 programs (machine flat)");
        if (ast_.l3 <= 0) throw CompileError(0, "machine flat: the L3 capacity must be positive");
        cap_ = static_cast<std::size_t>(ast_.l3);
        p_.name = ast_.name;
        p_.source = TileProgram(ast_.name);
        for (const Decl& d : ast_.decls) {
            if (d.is_vector)
                throw CompileError(d.line, "vector " + d.name + ": vector operands arrive with the linear operator "
                                           "(csp-language plan step 2)");
            if (p_.source.has_operand(d.name)) throw CompileError(d.line, "operand " + d.name + " is declared twice");
            if (d.rows <= 0 || d.cols <= 0 || d.tile_rows <= 0 || d.tile_cols <= 0)
                throw CompileError(d.line, "operand " + d.name + ": sizes and tile sizes must be positive");
            p_.source.add_operand(TensorOperand(d.name, static_cast<Dim>(d.rows), static_cast<Dim>(d.cols),
                                                static_cast<Dim>(d.tile_rows), static_cast<Dim>(d.tile_cols)));
        }
        p_.processes = {{ProcessKind::Dma, "dma", {}}, {ProcessKind::BlockMover, "bm", {}},
                        {ProcessKind::Streamer, "str", {}}, {ProcessKind::Compute, "cf", {}}};
        p_.channels = {{Chan::Dram, "dram", 0}, {Chan::L3, "l3", cap_}, {Chan::L2, "l2", 0}, {Chan::Cf, "cf", 0}};
        std::map<std::string, long long> env;
        for (const Stmt& s : ast_.body) stmt(s, env);
        for (const auto& [k, r] : resident_)
            throw CompileError(0, k + " is still resident at the end of the program; release it");
        for (const auto& [k, state] : acc_)
            if (state != AccState::Stored)
                throw CompileError(0, "the accumulator " + k + " is never stored");
        return std::move(p_);
    }

private:
    enum class AccState { Open, Closed, Stored };
    const Program& ast_;
    std::size_t cap_ = 0;
    CspProgram p_;
    std::map<std::string, std::size_t> resident_;          // tile -> open residency
    std::map<std::string, AccState> acc_;                  // tile -> its accumulator's state
    std::map<std::string, std::size_t> acc_calls_;         // tile -> calls into its open accumulator

    // ---- expressions and tiles ----
    long long eval(const Expr& e, const std::map<std::string, long long>& env, int line) const {
        switch (e.kind) {
            case Expr::Kind::Int: return e.value;
            case Expr::Kind::Var: {
                auto it = env.find(e.name);
                if (it == env.end()) throw CompileError(line, "'" + e.name + "' is not a loop variable in scope");
                return it->second;
            }
            case Expr::Kind::Bin: {
                const long long a = eval(e.args.at(0), env, line), b = eval(e.args.at(1), env, line);
                return e.op == '+' ? a + b : e.op == '-' ? a - b : a * b;
            }
        }
        return 0;
    }

    // Every tile a reference names (':' expands along its dimension), in row-major order.
    std::vector<TileCoord> expand(const TileRef& r, const std::map<std::string, long long>& env) const {
        if (!p_.source.has_operand(r.name)) throw CompileError(r.line, "'" + r.name + "' is not a declared operand");
        const TensorOperand& op = p_.source.operand(r.name);
        if (r.index.size() != 2) throw CompileError(r.line, r.name + " is a tensor: index it [row, col]");
        auto range = [&](const Index& ix, Dim n) {
            std::vector<Dim> v;
            if (ix.all) {
                for (Dim i = 0; i < n; ++i) v.push_back(i);
                return v;
            }
            const long long x = eval(ix.expr, env, r.line);
            if (x < 0 || x >= static_cast<long long>(n))
                throw CompileError(r.line, r.name + ": tile index " + std::to_string(x) + " is outside 0.." +
                                           std::to_string(n));
            v.push_back(static_cast<Dim>(x));
            return v;
        };
        std::vector<TileCoord> out;
        for (Dim ti : range(r.index[0], op.n_tile_rows()))
            for (Dim tj : range(r.index[1], op.n_tile_cols())) out.push_back(TileCoord{r.name, ti, tj});
        return out;
    }
    TileCoord one(const TileRef& r, const std::map<std::string, long long>& env) const {
        const auto v = expand(r, env);
        if (v.size() != 1) throw CompileError(r.line, r.name + ": a single tile is required here, not a slice");
        return v.front();
    }

    // ---- actions ----
    std::size_t process(ProcessKind k) const {
        for (std::size_t i = 0; i < p_.processes.size(); ++i) if (p_.processes[i].kind == k) return i;
        return kNone;
    }
    std::size_t emit(Action::Kind kind, const TileCoord& t, ProcessKind who, std::size_t l0 = kNone,
                     std::size_t residency = kNone) {
        Action a;
        a.kind = kind;
        a.tile = t;
        a.process = kind == Action::Kind::Release ? kNone : process(who);
        a.l0_op = l0;
        a.residency = residency;
        p_.actions.push_back(a);
        const std::size_t i = p_.actions.size() - 1;
        if (a.process != kNone) p_.processes[a.process].actions.push_back(i);
        return i;
    }
    std::size_t open(const TileCoord& t, std::size_t action, bool loaded, bool dirty) {
        Residency r;
        r.tile = t;
        r.open = action;
        r.loaded = loaded;
        r.dirty = dirty;
        p_.residencies.push_back(r);
        const std::size_t id = p_.residencies.size() - 1;
        p_.actions[action].residency = id;
        resident_[t.to_string()] = id;
        return id;
    }
    void need_slot(int line, const std::string& why) const {
        if (resident_.size() >= cap_)
            throw CompileError(line, why + " needs an L3 slot, and all " + std::to_string(cap_) + " are held");
    }
    std::size_t l0(TileOp op) {
        p_.source.push(std::move(op));
        return p_.source.ops().size() - 1;
    }

    // ---- statements ----
    void stmt(const Stmt& s, std::map<std::string, long long>& env) {
        if (!s.context.empty())
            throw CompileError(s.line, "tile contexts ('via') arrive with the linear operator (csp-language plan step 2)");
        switch (s.kind) {
            case Stmt::Kind::For: {
                if (env.count(s.var)) throw CompileError(s.line, "loop variable '" + s.var + "' shadows an outer one");
                const long long lo = eval(s.lo, env, s.line), hi = eval(s.hi, env, s.line);
                for (long long v = lo; v < hi; ++v) {
                    env[s.var] = v;
                    for (const Stmt& b : s.body) stmt(b, env);
                }
                env.erase(s.var);
                break;
            }
            case Stmt::Kind::Resident:
                for (const TileRef& r : s.tiles)
                    for (const TileCoord& t : expand(r, env)) {
                        const std::string k = t.to_string();
                        if (resident_.count(k)) throw CompileError(s.line, k + " is already resident");
                        if (acc_.count(k) && acc_.at(k) != AccState::Stored)
                            throw CompileError(s.line, k + " is an accumulator in the fabric, not a DRAM tile to load");
                        need_slot(s.line, "resident " + k);
                        open(t, emit(Action::Kind::Load, t, ProcessKind::Dma), true, false);
                    }
                break;
            case Stmt::Kind::Release:
                for (const TileRef& r : s.tiles)
                    for (const TileCoord& t : expand(r, env)) {
                        const std::string k = t.to_string();
                        auto it = resident_.find(k);
                        if (it == resident_.end()) throw CompileError(s.line, "release " + k + ": it is not resident");
                        if (p_.residencies[it->second].dirty)
                            throw CompileError(s.line, "release " + k + ": a call wrote it and it is not stored; "
                                                       "store it first");
                        p_.residencies[it->second].release = emit(Action::Kind::Release, t, ProcessKind::Dma, kNone, it->second);
                        resident_.erase(it);
                    }
                break;
            case Stmt::Kind::Acc: {
                const TileCoord y = one(s.out, env);
                const std::string k = y.to_string();
                if (resident_.count(k)) throw CompileError(s.line, "acc " + k + ": it is resident in L3; an accumulator lives in the fabric");
                if (acc_.count(k) && acc_.at(k) != AccState::Stored)
                    throw CompileError(s.line, "acc " + k + ": it already has an accumulator");
                acc_[k] = AccState::Open;
                acc_calls_[k] = 0;
                for (const Stmt& b : s.body) stmt(b, env);
                if (acc_calls_[k] == 0)
                    throw CompileError(s.line, "acc " + k + " receives no call; an accumulator needs at least one "
                                               "'+->' into it");
                acc_[k] = AccState::Closed;
                break;
            }
            case Stmt::Kind::Call: call(s, env); break;
            case Stmt::Kind::Store: store(s, env); break;
            case Stmt::Kind::Distribute:
            case Stmt::Kind::Broadcast:
                throw CompileError(s.line, "distribute and broadcast are level 2 (a distributed machine), not built yet");
        }
    }

    void deliver(const TileCoord& t, int line, const std::string& fn) {
        const std::string k = t.to_string();
        auto it = resident_.find(k);
        if (it == resident_.end())
            throw CompileError(line, "call " + fn + " reads " + k + ", which is not resident; make it resident first");
        emit(Action::Kind::Move, t, ProcessKind::BlockMover, kNone, it->second);
        ++p_.residencies[it->second].consumers;
        emit(Action::Kind::Feed, t, ProcessKind::Streamer);
    }

    void call(const Stmt& s, const std::map<std::string, long long>& env) {
        std::vector<TileCoord> args;
        for (const TileRef& r : s.tiles) args.push_back(one(r, env));
        const TileCoord y = one(s.out, env);
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
            op.pivot_slot = static_cast<int>(eval(*s.pivot, env, s.line));
        } else if (s.fn == "trsm_ll" || s.fn == "trsm_ur") {
            if (args.size() != 2 || !same(args[1], y))
                throw CompileError(s.line, s.fn + " works in place on its second operand: " + s.fn + "(d, x) -> x");
            op.kind = s.fn == "trsm_ll" ? TileOpKind::TrsmLowerLeft : TileOpKind::TrsmUpperRight;
            op.inputs = {args[0]};
            op.outputs = {y};
        } else {
            throw CompileError(s.line, "unknown tile function '" + s.fn + "' (gemm, getrf, laswp, trsm_ll, trsm_ur)");
        }
        if (s.fn != "gemm" && s.accumulate) throw CompileError(s.line, s.fn + " writes its result: '->', not '+->'");
        if (s.fn != "getrf" && s.fn != "laswp" && s.pivot) throw CompileError(s.line, s.fn + " takes no pivot");

        const bool in_acc = acc_.count(yk) && acc_.at(yk) == AccState::Open;
        if (in_acc) {
            // An output-stationary chain: the operands are fed; the result accumulates in the fabric.
            for (const TileCoord& a : args) {
                TileOp feed;
                feed.kind = TileOpKind::Feed;
                feed.port_kind = PortKind::Input;
                feed.port = &a == &args[0] ? "West" : "North";
                feed.inputs = {a};
                l0(feed);
                deliver(a, s.line, s.fn);
            }
            const std::size_t c = emit(Action::Kind::Call, y, ProcessKind::Compute, l0(op));
            p_.actions[c].accumulate = true;
            ++acc_calls_[yk];
            return;
        }
        if (acc_.count(yk) && acc_.at(yk) != AccState::Stored)
            throw CompileError(s.line, "call " + s.fn + " writes " + yk + ", whose accumulator is closed; store it first");
        // In place: the result is resident in L3; it travels to the fabric and back, dirty.
        if (!resident_.count(yk))
            throw CompileError(s.line, "call " + s.fn + " writes " + yk + ", which is neither resident nor an open "
                                       "accumulator");
        std::vector<TileCoord> operands = args;
        if (std::none_of(args.begin(), args.end(), [&](const TileCoord& a) { return same(a, y); })) operands.push_back(y);
        for (const TileCoord& a : operands) deliver(a, s.line, s.fn);
        emit(Action::Kind::Call, y, ProcessKind::Compute, l0(op));
        emit(Action::Kind::Drain, y, ProcessKind::Streamer);
        const std::size_t r = resident_.at(yk);
        emit(Action::Kind::Writeback, y, ProcessKind::BlockMover, kNone, r);
        p_.residencies[r].dirty = true;
    }

    void store(const Stmt& s, const std::map<std::string, long long>& env) {
        for (const TileCoord& y : expand(s.out, env)) store_one(s, y);
    }

    void store_one(const Stmt& s, const TileCoord& y) {
        const std::string k = y.to_string();
        auto acc = acc_.find(k);
        if (acc != acc_.end() && acc->second == AccState::Open)
            throw CompileError(s.line, "store " + k + " inside its own accumulator; store it after the acc block");
        if (acc != acc_.end() && acc->second == AccState::Closed) {
            // Out of the fabric: drained, written back into an L3 slot for the moment, stored, freed.
            need_slot(s.line, "store " + k + "'s writeback");
            TileOp drain;
            drain.kind = TileOpKind::Drain;
            drain.port_kind = PortKind::Output;
            drain.port = "South";
            drain.outputs = {y};
            const std::size_t d = l0(drain);
            emit(Action::Kind::Drain, y, ProcessKind::Streamer, d);
            const std::size_t r = open(y, emit(Action::Kind::Writeback, y, ProcessKind::BlockMover, d), false, true);
            emit(Action::Kind::Store, y, ProcessKind::Dma, d, r);
            ++p_.residencies[r].consumers;
            p_.residencies[r].dirty = false;
            p_.residencies[r].release = emit(Action::Kind::Release, y, ProcessKind::Dma, d, r);
            resident_.erase(k);
            acc->second = AccState::Stored;
            return;
        }
        auto it = resident_.find(k);
        if (it == resident_.end()) throw CompileError(s.line, "store " + k + ": it is neither resident nor an accumulator");
        if (!p_.residencies[it->second].dirty) throw CompileError(s.line, "store " + k + ": nothing has written it since it was loaded");
        emit(Action::Kind::Store, y, ProcessKind::Dma, kNone, it->second);
        ++p_.residencies[it->second].consumers;
        p_.residencies[it->second].dirty = false;
    }
};

}  // namespace detail

// Compile a parsed program to the CSP program IR. Throws CompileError, naming the line.
inline CspProgram compile(const Program& ast) { return detail::Compiler(ast).run(); }

// Parse and compile .csp source.
inline CspProgram compile(const std::string& source) { return compile(parse(source)); }

}  // namespace sw::kpu::program::csp::lang
