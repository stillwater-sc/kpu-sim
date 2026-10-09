// ============================================================================
// include/sw/kpu/program/csp/behavioral.hpp
// L-B over the CSP program (docs/plans/csp-program-tile-sequencing.md §4 step 1): every action
// atomic, in program order, moving real tile values between the channels.
//
// It checks the program as much as it runs it. A tile an action reads must be where the program
// says it is: a Move of a tile not resident in L3, a Feed of one not in L2, a Call on an operand
// the fabric does not hold, or any read of a released tile, is a lowering bug, reported by name.
// Calls run the shared L0 kernels (tile_kernels.hpp), so a correct program reproduces the L0
// reference bit for bit.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/tile_kernels.hpp>

#include <array>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::csp {

class BehavioralError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class BehavioralInterpreter {
public:
    struct Summary {
        std::size_t actions = 0, calls = 0;
        std::size_t peak_l3 = 0;            // tiles held in L3 at once, as executed
    };

    // Run `p` from its source operands' values. On return, result() holds the operands as DRAM
    // has them: what the program stored.
    Summary run(const CspProgram& p) {
        begin(p.source);
        for (const Action& a : p.actions)
            step(a, a.kind == Action::Kind::Call ? &p.source.ops().at(a.l0_op) : nullptr);
        return finish();
    }

    // The same, one action at a time (a program's ActionStream: lang/walk.hpp). `operands`
    // carries the values; a Call's tile function travels with its action.
    void begin(const TileProgram& operands) {
        work_ = operands;
        state_ = TileKernelState{};
        for (auto& s : store_) s.clear();
        sum_ = Summary{};
        // DRAM starts with every operand's tiles.
        for (const auto& name : work_.operand_order()) {
            const TensorOperand& op = work_.operand(name);
            for (Dim ti = 0; ti < op.n_tile_rows(); ++ti)
                for (Dim tj = 0; tj < op.n_tile_cols(); ++tj)
                    at(Chan::Dram)[TileCoord{name, ti, tj}.to_string()] = extract(op, ti, tj);
        }
    }

    void step(const Action& a, const TileOp* call_op) {
        const std::string k = a.tile.to_string();
        const std::size_t i = sum_.actions;
        auto where = [&](std::size_t idx) {
            return "csp action " + std::to_string(idx) + " " + to_string(a.kind) + " " + k;
        };
        switch (a.kind) {
            case Action::Kind::Load:
            case Action::Kind::Store:
            case Action::Kind::Move: {
                std::vector<float> v = take(from_chan(a.kind), k, where(i), /*keep=*/true);
                stages(v, a, where(i));         // a Move's copy, transformed on its way (bm.ingress)
                at(to_chan(a.kind))[k] = std::move(v);
                break;
            }
            case Action::Kind::Writeback:
            case Action::Kind::Feed:
            case Action::Kind::Drain: {
                std::vector<float> v = take(from_chan(a.kind), k, where(i), /*keep=*/false);
                stages(v, a, where(i));         // a result's epilogue, where the program placed it
                at(to_chan(a.kind))[k] = std::move(v);
                break;
            }
            case Action::Kind::Release:
                if (!at(Chan::L3).erase(k)) throw BehavioralError(where(i) + ": nothing resident to release");
                break;
            case Action::Kind::Call:
                if (!call_op) throw BehavioralError(where(i) + ": a call without its tile function");
                call(*call_op, a.accumulate, where(i));
                ++sum_.calls;
                break;
        }
        sum_.peak_l3 = std::max(sum_.peak_l3, at(Chan::L3).size());
        ++sum_.actions;
    }

    Summary finish() {
        for (Chan c : {Chan::L3, Chan::L2, Chan::Cf})
            if (!at(c).empty())
                throw BehavioralError("csp program ends with " + std::to_string(at(c).size()) + " tiles left in " +
                                      to_string(c) + " (first: " + at(c).begin()->first + ")");
        // DRAM's tiles back into whole operands.
        for (const auto& name : work_.operand_order()) {
            TensorOperand& op = work_.operand(name);
            for (Dim ti = 0; ti < op.n_tile_rows(); ++ti)
                for (Dim tj = 0; tj < op.n_tile_cols(); ++tj)
                    insert(op, ti, tj, at(Chan::Dram).at(TileCoord{name, ti, tj}.to_string()));
        }
        return sum_;
    }

    // The operands as DRAM holds them after run().
    const TileProgram& result() const { return work_; }

private:
    TileProgram work_;
    TileKernelState state_;
    Summary sum_;
    std::array<std::map<std::string, std::vector<float>>, kChannels> store_;

    std::map<std::string, std::vector<float>>& at(Chan c) { return store_[static_cast<std::size_t>(c)]; }

    std::vector<float> take(Chan from, const std::string& k, const std::string& where, bool keep) {
        auto& s = at(from);
        auto it = s.find(k);
        if (it == s.end())
            throw BehavioralError(where + ": the tile is not in " + std::string(to_string(from)));
        std::vector<float> v = it->second;
        if (!keep) s.erase(it);
        return v;
    }

    // The action's tile context, in order. Each stage computes with the L0 epilogue's own
    // element functions, so a fused epilogue is bit-identical to the unfused one. An add's
    // vector comes from L3, where the program made it resident.
    void stages(std::vector<float>& v, const Action& a, const std::string& where) {
        if (a.context.empty()) return;
        const TensorOperand& t = work_.operand(a.tile.operand);
        const std::size_t cols = t.col_end(a.tile.tj) - t.col_begin(a.tile.tj);
        for (const Stage& st : a.context) {
            if (st.op == VeOp::Add) {
                const std::string b = st.arg.to_string();
                auto it = at(Chan::L3).find(b);
                if (it == at(Chan::L3).end())
                    throw BehavioralError(where + ": add(" + b + ") @ " + to_string(st.place) + ": " + b +
                                          " is not resident in L3");
                bias_tile(v, cols, it->second);
            } else {
                activate_tile(v, activation_of(st.op));
            }
        }
    }

    static std::vector<float> extract(const TensorOperand& op, Dim ti, Dim tj) {
        std::vector<float> v;
        for (Dim r = op.row_begin(ti); r < op.row_end(ti); ++r)
            for (Dim c = op.col_begin(tj); c < op.col_end(tj); ++c) v.push_back(op.at(r, c));
        return v;
    }
    static void insert(TensorOperand& op, Dim ti, Dim tj, const std::vector<float>& v) {
        std::size_t n = 0;
        for (Dim r = op.row_begin(ti); r < op.row_end(ti); ++r)
            for (Dim c = op.col_begin(tj); c < op.col_end(tj); ++c) op.at(r, c) = v.at(n++);
    }

    // A tile function on what the fabric holds: its operands into the working operands, the
    // shared L0 kernel, its outputs back into the fabric. A fed input is consumed by the call.
    // In an explicit (accumulating) program the fabric's accumulator for a result starts at zero
    // and stays in the fabric until its Drain.
    void call(const TileOp& op, bool accumulates, const std::string& where) {
        auto& cf = at(Chan::Cf);
        for (const auto& t : op.inputs) {
            const std::string k = t.to_string();
            if (!cf.count(k)) throw BehavioralError(where + ": operand " + k + " is not in the fabric");
            insert(work_.operand(t.operand), t.ti, t.tj, cf.at(k));
        }
        for (const auto& t : op.outputs) {
            const std::string k = t.to_string();
            auto it = cf.find(k);
            if (it == cf.end()) {
                if (!accumulates) throw BehavioralError(where + ": output " + k + " is not in the fabric");
                const TensorOperand& o = work_.operand(t.operand);
                insert(work_.operand(t.operand), t.ti, t.tj,
                       std::vector<float>(static_cast<std::size_t>(o.row_end(t.ti) - o.row_begin(t.ti)) *
                                          (o.col_end(t.tj) - o.col_begin(t.tj)), 0.0f));
            } else {
                insert(work_.operand(t.operand), t.ti, t.tj, it->second);
            }
        }
        apply(work_, op, state_);
        for (const auto& t : op.outputs) cf[t.to_string()] = extract(work_.operand(t.operand), t.ti, t.tj);
        for (const auto& t : op.inputs) {
            bool is_output = false;
            for (const auto& o : op.outputs)
                if (o.operand == t.operand && o.ti == t.ti && o.tj == t.tj) is_output = true;
            if (!is_output) cf.erase(t.to_string());
        }
    }
};

}  // namespace sw::kpu::program::csp
