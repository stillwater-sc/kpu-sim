// ============================================================================
// include/sw/kpu/program/csp/behavioral.hpp
// L-B over the CSP program (docs/plans/csp-program-tile-sequencing.md §4 step 1): every action
// atomic, in program order, moving real tile values between the channels.
//
// It checks the program as much as it runs it. A tile an action reads must be where the program
// says it is: a Move of a tile not resident in L3, a Feed of one not in L2, a Call on an operand
// the fabric does not hold, or any read of a released tile, is a program bug, reported by name.
// Calls run the shared L0 kernels (tile_kernels.hpp), so a correct program reproduces the L0
// reference bit for bit.
//
// L3 IS ADDRESSED STORAGE (ADR 0004 §5; docs/plans/l3-placement-at-lb.md §3.3). Every action on
// a residency carries the L3 address the walker placed it at, and L-B executes that address: a
// Load, Inherit or opening Writeback writes into a free slot and tags it with the tile; a Move or
// Store reads the slot its tile was placed in; a Release frees exactly that slot; a Retain holds
// it past the program. A write into a live slot, a read of a slot tagged with another tile, and a
// release of a free slot are refused, naming the slot and both tiles (#279's class of defect).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/tile_kernels.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::csp {

class BehavioralError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline L3Geometry l3_geometry(const CspProgram& p) {
    return {static_cast<std::uint32_t>(p.channels.at(static_cast<std::size_t>(Chan::L3)).capacity), p.l3_banks};
}

class BehavioralInterpreter {
public:
    struct Summary {
        std::size_t actions = 0, calls = 0;
        std::size_t peak_l3 = 0;            // L3 slots tagged at once, as executed
    };

    // Run `p` from its source operands' values. On return, result() holds the operands as DRAM
    // has them: what the program stored.
    Summary run(const CspProgram& p) {
        begin(p.source, false, l3_geometry(p));
        for (const Action& a : p.actions)
            step(a, a.kind == Action::Kind::Call ? &p.source.ops().at(a.l0_op) : nullptr);
        return finish();
    }

    // The same, one action at a time (a program's ActionStream: lang/walk.hpp). `operands`
    // carries the values; a Call's tile function travels with its action.
    //
    // With `record`, L-B keeps the trace of what it executes: the operands with their input
    // values, and every L0 op an action carries (step's `l0`), in the order it ran them. That is
    // the program's L0 (ADR 0004 §4): L-B is the oracle, and the one place a CSP program becomes
    // an L0. Off by default: a large program streams, and its trace would not fit.
    void begin(const TileProgram& operands, bool record = false, L3Geometry l3 = {}) {
        recording_ = record;
        geometry_ = l3;
        l3_.clear();
        l3_index_.clear();
        trace_ = TileProgram(operands.name());
        if (record)
            for (const auto& name : operands.operand_order()) trace_.add_operand(operands.operand(name));
        work_ = operands;
        state_ = TileKernelState{};
        for (auto& s : store_) s.clear();
        retained_.clear();
        sum_ = Summary{};
        // DRAM starts with every operand's tiles.
        for (const auto& name : work_.operand_order()) {
            const TensorOperand& op = work_.operand(name);
            for (Dim ti = 0; ti < op.n_tile_rows(); ++ti)
                for (Dim tj = 0; tj < op.n_tile_cols(); ++tj)
                    at(Chan::Dram)[TileCoord{name, ti, tj}.to_string()] = extract(op, ti, tj);
        }
    }

    void step(const Action& a, const TileOp* call_op, const std::vector<TileOp>* l0 = nullptr) {
        if (recording_ && l0)
            for (const TileOp& op : *l0) trace_.push(op);
        const std::string k = a.tile.to_string();
        const std::size_t i = sum_.actions;
        auto where = [&](std::size_t idx) {
            return "csp action " + std::to_string(idx) + " " + to_string(a.kind) + " " + k;
        };
        switch (a.kind) {
            case Action::Kind::Load: {
                const L3Address& s = address(a, where(i));
                std::vector<float> v = take(Chan::Dram, k, where(i), /*keep=*/true);
                write_l3(s, a, k, std::move(v), where(i));
                break;
            }
            case Action::Kind::Store:
            case Action::Kind::Move: {
                std::vector<float> v = read_l3(address(a, where(i)), a, k, where(i)).values;
                stages(v, a, where(i));         // a Move's copy, transformed on its way (bm.ingress)
                at(to_chan(a.kind))[k] = std::move(v);
                break;
            }
            case Action::Kind::Writeback: {
                const L3Address& s = address(a, where(i));
                std::vector<float> v = take(Chan::L2, k, where(i), /*keep=*/false);
                stages(v, a, where(i));         // a result's epilogue, where the program placed it
                write_l3(s, a, k, std::move(v), where(i));
                break;
            }
            case Action::Kind::Feed:
            case Action::Kind::Drain: {
                std::vector<float> v = take(from_chan(a.kind), k, where(i), /*keep=*/false);
                stages(v, a, where(i));         // a result's epilogue, where the program placed it
                at(to_chan(a.kind))[k] = std::move(v);
                break;
            }
            case Action::Kind::Release: {
                const L3Address& s = address(a, where(i));
                read_l3(s, a, k, where(i), "nothing resident to release");
                if (retained_.count(k)) throw BehavioralError(where(i) + ": " + k + " is retained; its slot is not freed");
                l3_index_.erase(k);
                l3_.erase(key_of(s));
                break;
            }
            case Action::Kind::Inherit: {
                // Resident from the start, with the value the operator before this one left: here,
                // the operand's value (an orchestrator puts the retained tile's value there).
                const L3Address& s = address(a, where(i));
                write_l3(s, a, k, take(Chan::Dram, k, where(i), /*keep=*/true), where(i));
                break;
            }
            case Action::Kind::Retain:
                read_l3(address(a, where(i)), a, k, where(i), "nothing resident to retain");
                retained_[k] = a.tile;
                break;
            case Action::Kind::Call:
                if (!call_op) throw BehavioralError(where(i) + ": a call without its tile function");
                call(*call_op, a.accumulate, where(i));
                ++sum_.calls;
                break;
        }
        sum_.peak_l3 = std::max(sum_.peak_l3, l3_.size());
        ++sum_.actions;
    }

    Summary finish() {
        // L3 may hold only what the program retained: those slots outlive it.
        for (const auto& [k, addr] : l3_index_)
            if (!retained_.count(k))
                throw BehavioralError("csp program ends with " + k + " still resident at " + addr.to_string() +
                                      " (neither released nor retained)");
        for (Chan c : {Chan::L2, Chan::Cf})
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
        // A retained tile's value is the one in its slot, which DRAM may not have (it need not be
        // stored): the operands are what the program leaves, in DRAM or retained in L3.
        for (const auto& [k, t] : retained_) insert(work_.operand(t.operand), t.ti, t.tj, slot_of(k).values);
        return sum_;
    }

    // The operands as DRAM holds them after run().
    const TileProgram& result() const { return work_; }

    // The trace begin(..., true) recorded: the operands on their inputs, and the ops L-B ran.
    const TileProgram& trace() const { return trace_; }

    // The tiles the program retained, by key, with the values their slots hold.
    std::map<std::string, std::vector<float>> retained() const {
        std::map<std::string, std::vector<float>> out;
        for (const auto& [k, t] : retained_) out[k] = slot_of(k).values;
        return out;
    }

    // Tiles a channel holds now (stepping reads station occupancy from this). L3's are its
    // tagged slots.
    std::size_t held(Chan c) const {
        return c == Chan::L3 ? l3_.size() : store_[static_cast<std::size_t>(c)].size();
    }

    // The tiles L3 holds now, by address (after finish(): the retained ones). The replay's
    // TileProgramReference::l3_contents() must agree.
    std::map<std::string, std::string> l3_contents() const {
        std::map<std::string, std::string> out;
        for (const auto& [k, addr] : l3_index_) out[addr.to_string()] = k;
        return out;
    }

    // The L3 address a tile is resident at now; none when it is not resident.
    std::optional<L3Address> resident_at(const TileCoord& t) const {
        auto it = l3_index_.find(t.to_string());
        if (it == l3_index_.end()) return std::nullopt;
        return it->second;
    }

private:
    TileProgram work_;
    TileProgram trace_;
    bool recording_ = false;
    TileKernelState state_;
    Summary sum_;
    std::array<std::map<std::string, std::vector<float>>, kChannels> store_;     // DRAM, L2, the fabric
    std::map<std::string, TileCoord> retained_;      // key -> tile

    // The addressed L3: a tagged slot holds one tile, of one residency. Untagged slots are free
    // and not stored, so the state is bounded by the slots in use.
    struct L3Slot {
        TileCoord tag;
        std::size_t residency = kNone;
        std::vector<float> values;
    };
    L3Geometry geometry_;
    std::map<std::pair<std::uint32_t, std::uint32_t>, L3Slot> l3_;     // (module, slot) -> its tile
    std::map<std::string, L3Address> l3_index_;                         // resident tile -> its slot

    static std::pair<std::uint32_t, std::uint32_t> key_of(const L3Address& s) { return {s.l3_tile, s.slot}; }

    const L3Slot& slot_of(const std::string& k) const { return l3_.at(key_of(l3_index_.at(k))); }

    // The address an action on a residency carries, checked against the L3's geometry.
    const L3Address& address(const Action& a, const std::string& where) const {
        if (!a.place) throw BehavioralError(where + ": the action carries no L3 address");
        const L3Address& s = *a.place;
        if (geometry_.slots && s.slot >= geometry_.slots)
            throw BehavioralError(where + ": " + s.to_string() + " is outside the L3's " +
                                  std::to_string(geometry_.slots) + " slots");
        if (geometry_.banks && s.bank != s.slot % geometry_.banks)
            throw BehavioralError(where + ": " + s.to_string() + ": slot " + std::to_string(s.slot) + " is in bank " +
                                  std::to_string(s.slot % geometry_.banks) + " of " + std::to_string(geometry_.banks));
        return s;
    }

    // Write tile `k` into slot `s`: a free slot opens the residency; the slot of the same
    // residency is rewritten in place (a result written back into its resident tile).
    void write_l3(const L3Address& s, const Action& a, const std::string& k, std::vector<float> v,
                  const std::string& where) {
        auto it = l3_.find(key_of(s));
        if (it == l3_.end()) {
            auto other = l3_index_.find(k);
            if (other != l3_index_.end())
                throw BehavioralError(where + ": " + s.to_string() + ": " + k + " is already resident at " +
                                      other->second.to_string());
            l3_[key_of(s)] = L3Slot{a.tile, a.residency, std::move(v)};
            l3_index_[k] = s;
            return;
        }
        const std::string held = it->second.tag.to_string();
        if (held != k || it->second.residency != a.residency)
            throw BehavioralError(where + ": " + s.to_string() + " holds " + held + ", which is live; " + k +
                                  " cannot be written there");
        same_address(s, k, where);
        it->second.values = std::move(v);
    }

    // The slot `s`, which must hold tile `k` of this action's residency.
    const L3Slot& read_l3(const L3Address& s, const Action& a, const std::string& k, const std::string& where,
                          const char* when_free = "the slot is free") const {
        auto it = l3_.find(key_of(s));
        if (it == l3_.end()) {
            auto other = l3_index_.find(k);
            throw BehavioralError(where + ": " + s.to_string() + ": " + when_free + " (" + k +
                                  (other != l3_index_.end() ? " is at " + other->second.to_string()
                                                            : std::string(" is not resident")) + ")");
        }
        const std::string held = it->second.tag.to_string();
        if (held != k || it->second.residency != a.residency)
            throw BehavioralError(where + ": " + s.to_string() + " holds " + held + ", not " + k);
        same_address(s, k, where);
        return it->second;
    }

    // The whole address, bank included, is the one the residency opened with: a slot is keyed by
    // module and slot, so a wrong bank would otherwise pass where no geometry checks it.
    void same_address(const L3Address& s, const std::string& k, const std::string& where) const {
        const L3Address& opened = l3_index_.at(k);
        if (!(opened == s))
            throw BehavioralError(where + ": " + s.to_string() + " is not " + k + "'s address, " + opened.to_string());
    }

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
                if (!l3_index_.count(b))
                    throw BehavioralError(where + ": add(" + b + ") @ " + to_string(st.place) + ": " + b +
                                          " is not resident in L3");
                bias_tile(v, cols, slot_of(b).values);
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
