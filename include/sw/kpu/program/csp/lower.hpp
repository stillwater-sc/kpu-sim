// ============================================================================
// include/sw/kpu/program/csp/lower.hpp
// L0 -> the level-1 CSP program (docs/plans/csp-program-tile-sequencing.md §3.2).
//
// The lowering walks L0 in order on the flat machine (one L3, one fabric) and DECIDES what the
// generators left to a run-time tag match: when each tile comes into L3, how long it stays, and
// what is reloaded because capacity forced it out. A tile stays resident until its last use;
// when L3 is full, the resident tile whose next use is furthest away is evicted (Belady) --
// the fewest reloads possible for this order. Choosing a better ORDER is the schedule's, and a
// later plan; it lowers through this same function.
//
// Two L0 styles exist today, and both are tile-function programs:
//   EXPLICIT (matmul: derive_matmul_tile_program): L0 feeds A and B through ports, the fabric
//     accumulates C from zero (its accumulators are cleared), and L0 drains C. Each Feed becomes
//     Load (if not resident) + Move + Feed; each Drain becomes Drain + Writeback, and C is
//     stored when it has no further use.
//   IMPLICIT (LU: derive_lu_tile_program): L0 has no Feed or Drain; every kernel works in place
//     on its operand tiles. Each op's operands are brought to the fabric (Load if not resident,
//     Move, Feed) and its outputs come back (Drain, Writeback), leaving the L3 copy dirty.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::csp {

struct LoweringOptions {
    // L3 capacity in tiles, from the device spec (a machine parameter); 0 = unbounded.
    std::size_t l3_capacity_tiles = 0;
};

class LoweringError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {

inline bool explicit_style(const TileProgram& p) {
    for (const TileOp& op : p.ops())
        if (op.kind == TileOpKind::Feed || op.kind == TileOpKind::Drain) return true;
    return false;
}

inline std::vector<TileCoord> operands_of(const TileOp& op) {
    std::vector<TileCoord> out;
    auto add = [&](const TileCoord& t) {
        for (const auto& x : out) if (x.operand == t.operand && x.ti == t.ti && x.tj == t.tj) return;
        out.push_back(t);
    };
    for (const auto& t : op.inputs) add(t);
    for (const auto& t : op.outputs) add(t);
    return out;
}

class Lowerer {
public:
    Lowerer(const TileProgram& l0, const LoweringOptions& opt) : l0_(l0), cap_(opt.l3_capacity_tiles) {
        explicit_ = explicit_style(l0);
        // Where each tile needs L3: an explicit Feed's tile; an implicit op's operands; a
        // Drain's tile (it comes back to L3 to be stored, or reused).
        for (std::size_t i = 0; i < l0.ops().size(); ++i) {
            const TileOp& op = l0.ops()[i];
            if (op.kind == TileOpKind::Feed) for (const auto& t : op.inputs) uses_[key(t)].push_back(i);
            else if (op.kind == TileOpKind::Drain) for (const auto& t : op.outputs) uses_[key(t)].push_back(i);
            else if (!explicit_) for (const auto& t : operands_of(op)) uses_[key(t)].push_back(i);
        }
    }

    CspProgram run() {
        p_.name = l0_.name();
        p_.source = l0_;
        p_.processes = {{ProcessKind::Dma, "dma", {}}, {ProcessKind::BlockMover, "bm", {}},
                        {ProcessKind::Streamer, "str", {}}, {ProcessKind::Compute, "cf", {}}};
        p_.channels = {{Chan::Dram, "dram", 0}, {Chan::L3, "l3", cap_}, {Chan::L2, "l2", 0}, {Chan::Cf, "cf", 0}};
        for (std::size_t i = 0; i < l0_.ops().size(); ++i) {
            const TileOp& op = l0_.ops()[i];
            switch (op.kind) {
                case TileOpKind::Feed:
                    for (const auto& t : op.inputs) deliver(t, i, {key(t)});
                    for (const auto& t : op.inputs) retire_if_dead(t, i);
                    break;
                case TileOpKind::Drain:
                    for (const auto& t : op.outputs) {
                        emit(Action::Kind::Drain, t, ProcessKind::Streamer, i);
                        writeback(t, i, {key(t)});
                        retire_if_dead(t, i);
                    }
                    break;
                default: {
                    if (explicit_) {
                        emit(Action::Kind::Call, op.outputs.empty() ? TileCoord{} : op.outputs.front(),
                             ProcessKind::Compute, i);
                        break;
                    }
                    // Implicit: the whole working set must be in L3 at once.
                    const auto operands = operands_of(op);
                    std::set<std::string> keep;
                    for (const auto& t : operands) keep.insert(key(t));
                    for (const auto& t : operands) deliver(t, i, keep);
                    emit(Action::Kind::Call, op.outputs.empty() ? operands.front() : op.outputs.front(),
                         ProcessKind::Compute, i);
                    for (const auto& t : op.outputs) {
                        emit(Action::Kind::Drain, t, ProcessKind::Streamer, i);
                        writeback(t, i, keep);
                    }
                    for (const auto& t : operands) retire_if_dead(t, i);
                    break;
                }
            }
        }
        // What is still resident and dirty goes home.
        std::vector<std::string> left;
        for (const auto& [k, r] : resident_) left.push_back(k);
        for (const auto& k : left) {
            if (p_.residencies[resident_.at(k)].dirty) store(k, kNone);
            release(k, kNone);
        }
        return std::move(p_);
    }

private:
    const TileProgram& l0_;
    std::size_t cap_;
    bool explicit_ = true;
    CspProgram p_;
    std::map<std::string, std::vector<std::size_t>> uses_;   // tile -> L0 ops needing it in L3
    std::map<std::string, std::size_t> resident_;            // tile -> open residency
    std::map<std::string, TileCoord> coord_;

    std::string key(const TileCoord& t) {
        std::string k = t.to_string();
        coord_.emplace(k, t);
        return k;
    }

    std::size_t next_use(const std::string& k, std::size_t after) const {
        auto it = uses_.find(k);
        if (it == uses_.end()) return kNone;
        auto u = std::upper_bound(it->second.begin(), it->second.end(), after);
        return u == it->second.end() ? kNone : *u;
    }

    std::size_t process(ProcessKind k) const {
        for (std::size_t i = 0; i < p_.processes.size(); ++i) if (p_.processes[i].kind == k) return i;
        return kNone;
    }

    std::size_t emit(Action::Kind kind, const TileCoord& t, ProcessKind who, std::size_t l0_op,
                     std::size_t residency = kNone) {
        Action a;
        a.kind = kind;
        a.tile = t;
        a.process = kind == Action::Kind::Release ? kNone : process(who);
        a.l0_op = l0_op;
        a.residency = residency;
        p_.actions.push_back(a);
        const std::size_t idx = p_.actions.size() - 1;
        if (a.process != kNone) p_.processes[a.process].actions.push_back(idx);
        return idx;
    }

    // Free a slot if L3 is full: the resident tile (outside `keep`) used furthest ahead, or not
    // at all, goes -- stored first if it is dirty.
    void make_room(std::size_t at, const std::set<std::string>& keep) {
        if (cap_ == 0) return;
        while (resident_.size() >= cap_) {
            std::string victim;
            std::size_t furthest = 0;
            bool found = false;
            for (const auto& [k, r] : resident_) {
                if (keep.count(k)) continue;
                const std::size_t n = next_use(k, at);
                if (!found || n > furthest) { victim = k; furthest = n; found = true; }
            }
            if (!found)
                throw LoweringError("csp lowering: L0 op " + std::to_string(at) + " needs " +
                                    std::to_string(keep.size()) + " tiles in L3 at once; the L3 holds " +
                                    std::to_string(cap_));
            if (p_.residencies[resident_.at(victim)].dirty) store(victim, at);
            release(victim, at);
        }
    }

    std::size_t open(const std::string& k, std::size_t action, bool loaded, bool dirty) {
        Residency r;
        r.tile = coord_.at(k);
        r.open = action;
        r.loaded = loaded;
        r.dirty = dirty;
        p_.residencies.push_back(r);
        const std::size_t id = p_.residencies.size() - 1;
        p_.actions[action].residency = id;
        resident_[k] = id;
        return id;
    }

    void store(const std::string& k, std::size_t l0_op) {
        const std::size_t r = resident_.at(k);
        emit(Action::Kind::Store, coord_.at(k), ProcessKind::Dma, l0_op, r);
        ++p_.residencies[r].consumers;
        p_.residencies[r].dirty = false;
    }

    void release(const std::string& k, std::size_t l0_op) {
        const std::size_t r = resident_.at(k);
        p_.residencies[r].release = emit(Action::Kind::Release, coord_.at(k), ProcessKind::Dma, l0_op, r);
        resident_.erase(k);
    }

    // The tile to the fabric: into L3 if it is not there, then out of L3 to L2, then fed.
    void deliver(const TileCoord& t, std::size_t at, const std::set<std::string>& keep) {
        const std::string k = key(t);
        if (!resident_.count(k)) {
            make_room(at, keep);
            open(k, emit(Action::Kind::Load, t, ProcessKind::Dma, at), true, false);
        }
        const std::size_t r = resident_.at(k);
        emit(Action::Kind::Move, t, ProcessKind::BlockMover, at, r);
        ++p_.residencies[r].consumers;
        emit(Action::Kind::Feed, t, ProcessKind::Streamer, at);
    }

    // A result back into L3: into its existing slot (dirtying it), or a new one.
    void writeback(const TileCoord& t, std::size_t at, const std::set<std::string>& keep) {
        const std::string k = key(t);
        if (resident_.count(k)) {
            emit(Action::Kind::Writeback, t, ProcessKind::BlockMover, at, resident_.at(k));
            p_.residencies[resident_.at(k)].dirty = true;
            return;
        }
        make_room(at, keep);
        open(k, emit(Action::Kind::Writeback, t, ProcessKind::BlockMover, at), false, true);
    }

    // After its last use a tile leaves L3: stored first if dirty, then released.
    void retire_if_dead(const TileCoord& t, std::size_t at) {
        const std::string k = key(t);
        if (!resident_.count(k) || next_use(k, at) != kNone) return;
        if (p_.residencies[resident_.at(k)].dirty) store(k, at);
        release(k, at);
    }
};

}  // namespace detail

// Lower `l0` to the level-1 CSP program on a flat machine with `opt.l3_capacity_tiles` slots.
// Throws LoweringError when one op's working set does not fit the L3.
inline CspProgram lower(const TileProgram& l0, const LoweringOptions& opt = {}) {
    return detail::Lowerer(l0, opt).run();
}

}  // namespace sw::kpu::program::csp
