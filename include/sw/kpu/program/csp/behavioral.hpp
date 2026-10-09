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
        work_ = p.source;
        state_ = TileKernelState{};
        for (auto& s : store_) s.clear();
        Summary sum;
        // DRAM starts with every operand's tiles.
        for (const auto& name : work_.operand_order()) {
            const TensorOperand& op = work_.operand(name);
            for (Dim ti = 0; ti < op.n_tile_rows(); ++ti)
                for (Dim tj = 0; tj < op.n_tile_cols(); ++tj)
                    at(Chan::Dram)[TileCoord{name, ti, tj}.to_string()] = extract(op, ti, tj);
        }
        const bool accumulates = has_drains(p.source);
        for (std::size_t i = 0; i < p.actions.size(); ++i) {
            const Action& a = p.actions[i];
            const std::string k = a.tile.to_string();
            auto where = [&](std::size_t idx) {
                return "csp action " + std::to_string(idx) + " " + to_string(a.kind) + " " + k;
            };
            switch (a.kind) {
                case Action::Kind::Load:
                case Action::Kind::Store:
                case Action::Kind::Move:
                    at(to_chan(a.kind))[k] = take(from_chan(a.kind), k, where(i), /*keep=*/true);
                    break;
                case Action::Kind::Writeback:
                case Action::Kind::Feed:
                case Action::Kind::Drain:
                    at(to_chan(a.kind))[k] = take(from_chan(a.kind), k, where(i), /*keep=*/false);
                    break;
                case Action::Kind::Release:
                    if (!at(Chan::L3).erase(k)) throw BehavioralError(where(i) + ": nothing resident to release");
                    break;
                case Action::Kind::Call:
                    call(p.source.ops().at(a.l0_op), accumulates, where(i));
                    ++sum.calls;
                    break;
            }
            sum.peak_l3 = std::max(sum.peak_l3, at(Chan::L3).size());
            ++sum.actions;
        }
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
        return sum;
    }

    // The operands as DRAM holds them after run().
    const TileProgram& result() const { return work_; }

private:
    TileProgram work_;
    TileKernelState state_;
    std::array<std::map<std::string, std::vector<float>>, kChannels> store_;

    std::map<std::string, std::vector<float>>& at(Chan c) { return store_[static_cast<std::size_t>(c)]; }

    static bool has_drains(const TileProgram& p) {
        for (const auto& op : p.ops()) if (op.kind == TileOpKind::Drain) return true;
        return false;
    }

    std::vector<float> take(Chan from, const std::string& k, const std::string& where, bool keep) {
        auto& s = at(from);
        auto it = s.find(k);
        if (it == s.end())
            throw BehavioralError(where + ": the tile is not in " + std::string(to_string(from)));
        std::vector<float> v = it->second;
        if (!keep) s.erase(it);
        return v;
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
