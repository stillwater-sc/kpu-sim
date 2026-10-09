// ============================================================================
// include/sw/kpu/program/csp/lang/context.hpp
// Tile contexts in the CSP language (docs/plans/csp-language.md §3.4, step 2): a `via` list of
// vector stages, each at the place it runs, and the machine that has to run it.
//
//   store Y[i, j] via add(b[j]) @ bm.egress, relu @ bm.egress;
//
// A result leaves the fabric through three places, in this order: `fabric` (on the
// accumulator), `str.drain` (the streamer's L1 -> L2 drain), `bm.egress` (the BlockMover's
// L2 -> L3 writeback). Its stages must be listed in that order: the order the tile meets them
// is the order they apply. `bm.ingress` (L3 -> L2) is an operand's way in, and an operand
// context is not built yet.
//
// Where a stage may run is the MACHINE's (decision Q5): a Target lists, per site, the
// operations its vector unit has. atan needs a transcendental unit, so `atan @ bm.egress` on a
// machine whose BlockMover unit lacks it is refused by name, before anything runs. Without a
// target, placement is not checked: the values do not depend on it.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

// What a machine runs where. nullopt: the site has no vector unit.
struct Target {
    // The fabric's epilogue is what its matmul builds today: a bias and a ReLU on the
    // accumulator (execute_matmul). A fabric with more is a spec field when one exists.
    std::optional<std::set<VeOp>> fabric = std::set<VeOp>{VeOp::Add, VeOp::Relu};
    std::optional<std::set<VeOp>> str;      // the streamers' (str.drain)
    std::optional<std::set<VeOp>> bm;       // the BlockMovers' (bm.egress, bm.ingress)
};

// A device's sites, from its deployment spec (movers.vector).
inline Target target_from(const platform::DeviceSpecification& d) {
    auto ops = [](const std::optional<platform::DeviceSpecification::Movers::VectorUnit>& u)
        -> std::optional<std::set<VeOp>> {
        if (!u) return std::nullopt;
        std::set<VeOp> s;
        for (const std::string& name : u->ops) {
            VeOp v;
            if (!parse_veop(name, v)) throw std::invalid_argument("movers.vector: '" + name + "' is not a vector operation");
            s.insert(v);
        }
        return s;
    };
    Target t;
    t.bm = ops(d.movers.bm_vector);
    t.str = ops(d.movers.str_vector);
    return t;
}

inline const char* site_of(Place p) {
    switch (p) {
        case Place::Fabric:    return "compute fabric";
        case Place::StrDrain:  return "streamer";
        case Place::BmEgress:
        case Place::BmIngress: return "BlockMover";
    }
    return "?";
}

// One resolved stage: its op and place, and add's vector argument as written.
struct ResolvedStage {
    VeOp op = VeOp::Relu;
    Place place = Place::Fabric;
    std::optional<TileRef> arg;
};

// Resolve the `via` list of a result (a store's, or an in-place call's): known operations,
// known places on a result's path in path order, and -- given a target -- each operation where
// the machine runs it. Throws a CompileError-compatible message through `fail`.
template <class Fail>
std::vector<ResolvedStage> resolve_result_context(const std::vector<Stage>& via, const Target* target, Fail fail) {
    std::vector<ResolvedStage> out;
    int last_rank = -1;
    Place last = Place::Fabric;
    for (const Stage& s : via) {
        ResolvedStage r;
        if (!parse_veop(s.op, r.op))
            fail(s.line, "'" + s.op + "' is not a vector operation (add, relu, gelu, silu, atan)");
        if (!parse_place(s.place, r.place))
            fail(s.line, "'" + s.place + "' is not a place (fabric, str.drain, bm.egress, bm.ingress)");
        if (r.place == Place::BmIngress)
            fail(s.line, s.op + " @ bm.ingress: bm.ingress is an operand's way into the fabric (L3 -> L2); a result "
                         "leaves through fabric, str.drain and bm.egress");
        if (r.op == VeOp::Add) {
            if (s.args.size() != 1) fail(s.line, "add takes one vector tile: add(b[j])");
            r.arg = s.args.front();
        } else if (!s.args.empty()) {
            fail(s.line, s.op + " takes no operand");
        }
        const int rank = static_cast<int>(r.place);   // Fabric < StrDrain < BmEgress: the result's path
        if (rank < last_rank)
            fail(s.line, s.op + " @ " + s.place + " is listed after a stage @ " + to_string(last) +
                         ", which the tile reaches later; list the stages in the order the tile passes their places");
        last_rank = rank;
        last = r.place;
        if (target) {
            const auto& site = r.place == Place::Fabric ? target->fabric
                               : r.place == Place::StrDrain ? target->str : target->bm;
            if (!site)
                fail(s.line, s.op + " @ " + s.place + ": the target's " + site_of(r.place) + " has no vector unit");
            if (!site->count(r.op)) {
                std::string has;
                for (VeOp v : *site) has += (has.empty() ? "" : ", ") + std::string(to_string(v));
                fail(s.line, s.op + " @ " + s.place + ": the target's " + site_of(r.place) + " vector unit runs " +
                             (has.empty() ? std::string("nothing") : has) + ", not " + s.op +
                             "; place it where the machine has it");
            }
        }
        out.push_back(r);
    }
    return out;
}

}  // namespace sw::kpu::program::csp::lang
