#pragma once
// ============================================================================
// include/sw/kpu/program/csp/distributed_behavioral.hpp
// L-B for the level-2 program (#606; docs/plans/distributed-csp-programs.md §3.3, §4 step 3): every
// process runs its own actions, each atomic, through channels bounded by their credits, and the
// tiles carry real values. It checks two of the distributed program's rules:
//   D5 liveness  a program that cannot finish is reported, with every stuck process, the action it
//                is at (and its line), and what that action waits for
//   D6 values    compare_values(): the program's result against a reference, bit for bit
//
// THE INTERLEAVING. Round robin over the processes, at most one action each per round, so no
// process starves (a fair interleaving). An action that cannot run yet waits; it is not an error:
// in a dataflow machine a mover waits for its tile to arrive and for its destination's credit.
//   load, push, feed, drain, call (to)   wait for the destination slot to be free: its credit
//   push, feed, drain, call              wait for every source slot to hold a tile (a call: any)
//   push to a DMA engine                 waits for a store-buffer credit
//   store                                waits for the tile at the front of its channel's buffer
//   call (acc)                           waits for its result slot to be live and expecting merges
//   any read of a result                 waits until every merge it expects has landed
// When a round runs nothing and some process has actions left, the program has deadlocked (D5).
//
// RESIDENCIES. A write that fills a slot opens a residency with the reads it states. Each read (a
// push, feed or drain from the slot, or a call's operand) counts one; the last frees the slot and
// so returns its credit. `reads 0` holds the tile: any number of reads, never freed. A program that
// ends with a residency still expecting reads is refused, naming the slot: its credit would never
// return, and the program said it would.
//
// VALUES. DRAM starts with the operands' values (distributed_inputs() makes deterministic ones). A
// load copies a tile out of DRAM, a store writes it back, and result() is DRAM at the end. Calls
// run the shared L0 kernels (tile_kernels.hpp) in the operands' number systems, as level 1's L-B
// does: `call gemm(a, b) to C[i,j]` is C := A.B (a fresh accumulator), `acc` is C := C + A.B. Other
// tile functions arrive with the generators that need them (#608-#616); an unknown one is refused.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/distributed.hpp>
#include <sw/kpu/program/tile_kernels.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::csp {

// D5: the program cannot finish. `stuck` names each process with actions left, and why it waits.
class DistributedDeadlock : public BehavioralError {
public:
    DistributedDeadlock(const std::string& what, std::vector<std::string> stuck)
        : BehavioralError(what), stuck_(std::move(stuck)) {}
    const std::vector<std::string>& stuck() const { return stuck_; }
private:
    std::vector<std::string> stuck_;
};

// Deterministic values for the program's operands: every `in` and `inout` operand's elements are
// small integers (exact in every number system the kernels take), varied by position and `seed`;
// `out` operands start at zero.
inline TileProgram distributed_inputs(const DistributedProgram& p, std::uint64_t seed = 1) {
    TileProgram t(p.name);
    for (const Operand2& o : p.operands) {
        TensorOperand op(o.name, o.rows, o.cols, o.tile_rows, o.tile_cols);
        op.type = o.type;
        if (o.io != "out")
            for (Dim r = 0; r < o.rows; ++r)
                for (Dim c = 0; c < o.cols; ++c)
                    op.set_value(r, c, static_cast<double>(static_cast<long long>((r * 7 + c * 3 + seed * 5) % 9) - 4));
        else if (!op.fp32())
            for (Dim r = 0; r < o.rows; ++r)
                for (Dim c = 0; c < o.cols; ++c) op.set_value(r, c, 0.0);
        t.add_operand(op);
    }
    return t;
}

// D6: empty when every operand `got` and `want` share has the same words, element for element;
// otherwise the first difference. Bit equality is the bar: a distributed program repeats its
// level-1 form's computation (CLAUDE.md: bit equality where the same computation is repeated).
inline std::string compare_values(const TileProgram& got, const TileProgram& want) {
    for (const auto& name : want.operand_order()) {
        if (!got.has_operand(name)) return "operand " + name + " is missing from the result";
        const TensorOperand& g = got.operand(name);
        const TensorOperand& w = want.operand(name);
        if (g.rows != w.rows || g.cols != w.cols) return "operand " + name + " has another shape";
        for (Dim r = 0; r < w.rows; ++r)
            for (Dim c = 0; c < w.cols; ++c)
                if (std::memcmp(&g.values[static_cast<std::size_t>(r) * g.cols + c],
                                &w.values[static_cast<std::size_t>(r) * w.cols + c], sizeof(float)) != 0)
                    return name + "(" + std::to_string(r) + ", " + std::to_string(c) + ") is " +
                           std::to_string(g.value(r, c)) + ", want " + std::to_string(w.value(r, c));
    }
    return {};
}

class DistributedBehavioral {
public:
    struct Summary {
        std::size_t actions = 0, calls = 0, rounds = 0;
        std::size_t peak_live = 0;          // slots holding a residency at once, as executed
    };

    // Run `p` from `values` (an operand per declared operand). On return, result() holds DRAM:
    // what the program stored. Throws DistributedDeadlock (D5), or BehavioralError for a program
    // the run cannot execute (an unknown tile function, a residency left expecting reads).
    Summary run(const DistributedProgram& p, const TileProgram& values) {
        if (const std::string bad = validate_structure(p); !bad.empty())
            throw BehavioralError("dcsp " + p.name + ": " + bad);
        p_ = &p;
        dram_ = values;
        for (const Operand2& o : p.operands)
            if (!dram_.has_operand(o.name)) throw BehavioralError("dcsp " + p.name + ": no values for operand " + o.name);
        slots_.clear();
        buffers_.assign(p.channels.size(), {});
        pc_.assign(p.processes.size(), 0);
        sum_ = Summary{};
        for (;;) {
            bool ran = false, left = false;
            ++sum_.rounds;
            for (std::size_t i = 0; i < p.processes.size(); ++i) {
                const Process2& pr = p.processes[i];
                if (pc_[i] == pr.actions.size()) continue;
                left = true;
                const Action2& a = p.actions.at(pr.actions[pc_[i]]);
                if (wait_reason(a).empty()) {
                    execute(a, where(pr, a));
                    ++pc_[i];
                    ++sum_.actions;
                    ran = true;
                }
            }
            if (!left) break;
            if (!ran) deadlock();
        }
        for (const auto& [at, s] : slots_) {
            if (s.merges_left > 0)
                throw BehavioralError("dcsp " + p.name + ": the program ends with " + s.tile + " in " + at.to_string() +
                                      " still expecting " + std::to_string(s.merges_left) + " merge(s)");
            if (s.reads_left > 0)
                throw BehavioralError("dcsp " + p.name + ": the program ends with " + s.tile + " in " + at.to_string() +
                                      " still expecting " + std::to_string(s.reads_left) +
                                      " read(s), so its credit never returns");
        }
        return sum_;
    }

    const TileProgram& result() const { return dram_; }

    // What a slot holds at the end: its tile's name and words, or nullopt when it is free.
    std::optional<std::pair<std::string, std::vector<float>>> slot(const SlotRef& at) const {
        const auto it = slots_.find(at);
        if (it == slots_.end()) return std::nullopt;
        return std::make_pair(it->second.tile, it->second.value);
    }

private:
    struct Live {
        std::string tile;
        TileCoord coord;
        std::vector<float> value;
        std::uint32_t reads_left = 0;       // 0 with `held`: any number of reads
        bool held = false;
        std::uint32_t merges_left = 0;      // a result: the merges still to land before any read
    };
    struct Packet {
        TileCoord coord;
        std::vector<float> value;
    };

    const DistributedProgram* p_ = nullptr;
    TileProgram dram_;
    std::map<SlotRef, Live> slots_;
    std::vector<std::deque<Packet>> buffers_;   // per channel into a DMA engine: its store buffer
    std::vector<std::size_t> pc_;
    Summary sum_;

    static std::string where(const Process2& pr, const Action2& a) {
        return pr.id.to_string() + (a.line ? ", line " + std::to_string(a.line) : std::string());
    }

    // Empty when `a` can run now; otherwise what it waits for.
    std::string wait_reason(const Action2& a) const {
        const auto holds = [&](const SlotRef& s, const TileCoord* t) -> std::string {
            const auto it = slots_.find(s);
            if (it == slots_.end()) return "its source " + s.to_string() + " holds nothing yet";
            if (t && it->second.tile != t->to_string())
                return "its source " + s.to_string() + " holds " + it->second.tile + ", not " + t->to_string();
            if (it->second.merges_left)
                return "its source " + s.to_string() + " still expects " + std::to_string(it->second.merges_left) + " merge(s)";
            return {};
        };
        if (a.from)
            if (std::string w = holds(*a.from, &a.tile); !w.empty()) return w;
        for (const SlotRef& s : a.args)
            if (std::string w = holds(s, nullptr); !w.empty()) return w;
        if (a.kind == Action2::Kind::Call && a.accumulate) {
            const auto it = slots_.find(*a.to);
            if (it == slots_.end()) return "its result " + a.to->to_string() + " is not live yet";
            if (it->second.merges_left == 0) return "its result " + a.to->to_string() + " expects no more merges";
            return {};
        }
        if (a.to) {
            const auto it = slots_.find(*a.to);
            if (it != slots_.end())
                return "its destination " + a.to->to_string() + " still holds " + it->second.tile +
                       (it->second.held ? " (held)" : " (" + std::to_string(it->second.reads_left) + " read(s) left)");
            return {};
        }
        const Channel2& c = p_->channels.at(a.channel);
        if (a.kind == Action2::Kind::Push) {                       // to a DMA engine's store buffer
            if (buffers_.at(a.channel).size() >= c.credits)
                return "channel " + c.name + "'s store buffer is full (" + std::to_string(c.credits) + ")";
            return {};
        }
        // A store: the tile at the front of its channel's buffer.
        const auto& q = buffers_.at(a.channel);
        if (q.empty()) return "channel " + c.name + " has delivered nothing yet";
        if (q.front().coord.to_string() != a.tile.to_string())
            return "channel " + c.name + " delivers " + q.front().coord.to_string() + " first, not " + a.tile.to_string();
        return {};
    }

    void read(const SlotRef& s, const std::string& w) {
        auto it = slots_.find(s);
        if (it == slots_.end())                                  // run() checks a call's reads first; a guard
            throw BehavioralError(w + ": reads " + s.to_string() + ", which no longer holds a tile");
        if (it->second.held) return;
        if (--it->second.reads_left == 0) slots_.erase(it);     // the last read frees it: its credit returns
    }

    void fill(const SlotRef& s, const TileCoord& t, std::vector<float> v, std::uint32_t reads) {
        Live l;
        l.tile = t.to_string();
        l.coord = t;
        l.value = std::move(v);
        l.reads_left = reads;
        l.held = reads == 0;
        slots_[s] = std::move(l);
        sum_.peak_live = std::max(sum_.peak_live, slots_.size());
    }

    void execute(const Action2& a, const std::string& w) {
        switch (a.kind) {
            case Action2::Kind::Load: {
                const TensorOperand& op = dram_.operand(a.tile.operand);
                fill(*a.to, a.tile, BehavioralInterpreter::extract_tile(op, a.tile.ti, a.tile.tj), a.reads);
                return;
            }
            case Action2::Kind::Push:
            case Action2::Kind::Feed:
            case Action2::Kind::Drain: {
                std::vector<float> v = slots_.at(*a.from).value;
                read(*a.from, w);
                if (a.to) fill(*a.to, a.tile, std::move(v), a.reads);
                else buffers_.at(a.channel).push_back(Packet{a.tile, std::move(v)});
                return;
            }
            case Action2::Kind::Store: {
                Packet pk = std::move(buffers_.at(a.channel).front());
                buffers_.at(a.channel).pop_front();                 // its credit returns
                BehavioralInterpreter::insert_tile(dram_.operand(pk.coord.operand), pk.coord.ti, pk.coord.tj, pk.value);
                return;
            }
            case Action2::Kind::Call:
                call(a, w);
                return;
        }
    }

    // A tile as its own one-tile operand, so a call's inputs and result never share storage: an
    // input and the result may name one tile coordinate, and two inputs may hold two versions of it.
    TensorOperand scratch(const std::string& name, const TileCoord& t, const std::vector<float>* v) const {
        const TensorOperand& src = dram_.operand(t.operand);
        const Dim rows = src.row_end(t.ti) - src.row_begin(t.ti), cols = src.col_end(t.tj) - src.col_begin(t.tj);
        TensorOperand o(name, rows, cols, rows, cols);
        o.type = src.type;
        o.acc = src.acc;
        if (v) o.values = *v;
        else if (!o.fp32())
            for (Dim r = 0; r < rows; ++r)
                for (Dim c = 0; c < cols; ++c) o.set_value(r, c, 0.0);
        return o;
    }

    void call(const Action2& a, const std::string& w) {
        if (a.op != "gemm") throw BehavioralError(w + ": tile function '" + a.op + "' is not one L-B runs at level 2 yet (gemm)");
        if (a.args.size() != 2) throw BehavioralError(w + ": gemm takes two operands");
        // A call reads each operand slot once per mention: the residency must have the reads.
        std::map<SlotRef, std::uint32_t> demand;
        for (const SlotRef& s : a.args) ++demand[s];
        for (const auto& [s, n] : demand) {
            const Live& l = slots_.at(s);
            if (!l.held && l.reads_left < n)
                throw BehavioralError(w + ": the call reads " + s.to_string() + " " + std::to_string(n) + " times, but its residency (" +
                                      l.tile + ") has " + std::to_string(l.reads_left) + " read(s) left");
        }
        const Live& x = slots_.at(a.args[0]);
        const Live& y = slots_.at(a.args[1]);
        const TileCoord out = a.accumulate ? slots_.at(*a.to).coord : a.tile;
        TileProgram k("call");
        k.add_operand(scratch("x", x.coord, &x.value));
        k.add_operand(scratch("y", y.coord, &y.value));
        k.add_operand(scratch("z", out, a.accumulate ? &slots_.at(*a.to).value : nullptr));   // fresh: zero
        TileOp op;
        op.kind = TileOpKind::MatMulAccum;
        op.inputs = {TileCoord{"x", 0, 0}, TileCoord{"y", 0, 0}};
        op.outputs = {TileCoord{"z", 0, 0}};
        TileKernelState st;
        try {
            apply(k, op, st);
        } catch (const std::exception& e) {
            throw BehavioralError(w + ": " + e.what());
        }
        std::vector<float> v = k.operand("z").values;
        for (const SlotRef& s : a.args) read(s, w);
        if (a.accumulate) {
            Live& r = slots_.at(*a.to);
            r.value = std::move(v);
            --r.merges_left;
        } else {
            fill(*a.to, out, std::move(v), a.reads);
            slots_.at(*a.to).merges_left = a.merges;
        }
        ++sum_.calls;
    }

    [[noreturn]] void deadlock() const {
        std::vector<std::string> stuck;
        std::string msg = "dcsp " + p_->name + ": deadlock (D5), no process can run:";
        for (std::size_t i = 0; i < p_->processes.size(); ++i) {
            const Process2& pr = p_->processes[i];
            if (pc_[i] == pr.actions.size()) continue;
            const Action2& a = p_->actions.at(pr.actions[pc_[i]]);
            stuck.push_back(where(pr, a) + ": " + wait_reason(a));
            msg += "\n  " + stuck.back();
        }
        throw DistributedDeadlock(msg, stuck);
    }
};

} // namespace sw::kpu::program::csp
