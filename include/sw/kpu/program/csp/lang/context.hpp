// ============================================================================
// include/sw/kpu/program/csp/lang/context.hpp
// Tile contexts in the CSP language (docs/plans/csp-language.md §3.4, step 2): a `via` list of
// vector stages, each at the place it runs, and the machine that has to run it.
//
//   store Y[i, j] via add(b[j]) @ bm.egress, relu @ bm.egress;
//
// A result leaves the fabric through three places, in this order: `fabric` (on the
// accumulator), `combiner` (the Combiner, L1 -> L2), `bm.egress` (the BlockMover's
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
    std::optional<std::set<VeOp>> combiner; // the Combiner's (combiner)
    std::optional<std::set<VeOp>> bm;       // the BlockMovers' (bm.egress, bm.ingress)
};

// A device's sites, from its deployment spec (combiner, movers.vector.bm).
inline Target target_from(const platform::DeviceSpecification& d) {
    auto ops = [](const std::optional<platform::DeviceSpecification::Movers::VectorUnit>& u)
        -> std::optional<std::set<VeOp>> {
        if (!u) return std::nullopt;
        std::set<VeOp> s;
        for (const std::string& name : u->ops) {
            VeOp v;
            if (!parse_veop(name, v)) throw std::invalid_argument("vector unit: '" + name + "' is not a vector operation");
            s.insert(v);
        }
        return s;
    };
    Target t;
    t.bm = ops(d.movers.bm_vector);
    t.combiner = ops(d.combiner);
    return t;
}

inline const char* site_of(Place p) {
    switch (p) {
        case Place::Fabric:    return "compute fabric";
        case Place::Combiner:  return "Combiner";
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
    std::int64_t m = 1;                 // scale
    int shift = 0;
};

// Resolve the `via` list of a result (a store's, or an in-place call's): known operations,
// known places on a result's path in path order, and -- given a target -- each operation where
// the machine runs it. Throws a CompileError-compatible message through `fail`.
template <class Fail>
std::vector<ResolvedStage> resolve_result_context(const std::vector<Stage>& via, const Target* target, Fail fail) {
    std::vector<ResolvedStage> out;
    int last_rank = -1;
    Place last = Place::Fabric;
    const Stage* format = nullptr;                      // a block-format conversion seen: it is last
    const Stage* after_scale = nullptr;                 // an activation or format seen: scale comes before it
    bool scaled = false;
    for (const Stage& s : via) {
        ResolvedStage r;
        if (!parse_veop(s.op, r.op))
            fail(s.line, "'" + s.op + "' is not a vector operation (add, relu, gelu, silu, atan, mxint8, mxfp8)");
        if (format)
            fail(s.line, s.op + " @ " + s.place + " follows the block-format conversion " + format->op +
                         ": the conversion is a result's last operation (bias, scale, activation, format)");
        if (!parse_place(s.place, r.place))
            fail(s.line, "'" + s.place + "' is not a place (fabric, combiner, bm.egress, bm.ingress)");
        if (r.place == Place::BmIngress)
            fail(s.line, s.op + " @ bm.ingress: bm.ingress is an operand's way into the fabric (L3 -> L2); a result "
                         "leaves through fabric, combiner and bm.egress");
        if (is_block_format(r.op)) {
            format = &s;
            if (r.place == Place::Fabric)
                fail(s.line, s.op + " @ fabric: a block-format conversion runs where the result leaves for memory, "
                             "on the Combiner or at bm.egress");
        }
        // The Combiner's order: bias, scale, activation, format.
        if (r.op == VeOp::Scale) {
            if (s.nums.size() != 2) fail(s.line, "scale takes a multiplier and a shift: scale(m, shift)");
            if (s.nums[0] <= -(1LL << 31) || s.nums[0] >= (1LL << 31))
                fail(s.line, "scale: the multiplier " + std::to_string(s.nums[0]) + " is outside (-2^31, 2^31)");
            if (s.nums[1] < 0 || s.nums[1] > 62) fail(s.line, "scale: the shift " + std::to_string(s.nums[1]) + " is outside 0..62");
            if (r.place != Place::Combiner)
                fail(s.line, "scale @ " + s.place + ": the requantization runs on the Combiner, as the result enters L2");
            if (after_scale)
                fail(s.line, "scale follows " + after_scale->op + ": the order is bias, scale, activation, format");
            if (scaled) fail(s.line, "scale: the result is already requantized");
            r.m = s.nums[0];
            r.shift = static_cast<int>(s.nums[1]);
            scaled = true;
        } else if (r.op == VeOp::Add) {
            if (scaled) fail(s.line, "add follows scale: the order is bias, scale, activation, format");
        } else {
            after_scale = after_scale ? after_scale : &s;
        }
        if (r.op == VeOp::Add) {
            if (s.args.size() != 1) fail(s.line, "add takes one vector tile: add(b[j])");
            r.arg = s.args.front();
        } else if (!s.args.empty()) {
            fail(s.line, s.op + " takes no operand");
        }
        const int rank = static_cast<int>(r.place);   // Fabric < Combiner < BmEgress: the result's path
        if (rank < last_rank)
            fail(s.line, s.op + " @ " + s.place + " is listed after a stage @ " + to_string(last) +
                         ", which the tile reaches later; list the stages in the order the tile passes their places");
        last_rank = rank;
        last = r.place;
        // A bias comes before the activation or format it feeds (after the places are checked,
        // whose order is the tile's path).
        if (r.op == VeOp::Add && after_scale)
            fail(s.line, "add follows " + after_scale->op + ": the order is bias, scale, activation, format");
        if (target) {
            const auto& site = r.place == Place::Fabric ? target->fabric
                               : r.place == Place::Combiner ? target->combiner : target->bm;
            // The Combiner is the vector unit; the fabric and the BlockMover each have one.
            const bool cmb = r.place == Place::Combiner;
            if (!site)
                fail(s.line, s.op + " @ " + s.place + (cmb ? std::string(": the target has no Combiner")
                                                           : ": the target's " + std::string(site_of(r.place)) +
                                                                 " has no vector unit"));
            if (!site->count(r.op)) {
                std::string has;
                for (VeOp v : *site) has += (has.empty() ? "" : ", ") + std::string(to_string(v));
                fail(s.line, s.op + " @ " + s.place + ": the target's " + site_of(r.place) + (cmb ? " runs " : " vector unit runs ") +
                             (has.empty() ? std::string("nothing") : has) + ", not " + s.op +
                             "; place it where the machine has it");
            }
        }
        out.push_back(r);
    }
    return out;
}

// What a requantized result's context may hold (#427). The result's accumulator values (`acc`,
// int32) become its type's (int8) at the scale, so each stage computes in the type the values
// have where it runs, in the Combiner's order:
//   - an add before the scale: a bias, in the accumulator's type, adding as the MAC does;
//   - the scale, on the Combiner;
//   - relu after the scale, on the integer result.
// gelu, silu and atan need real values, and a block format converts real values: both are
// refused. The order itself is resolve_result_context's. Returns why the context is refused, or
// nullopt. `required`: the context is a path into storage, which must carry the scale.
inline std::optional<std::string> integer_context_refusal(const std::vector<VeOp>& ops, const std::string& tile,
                                                          const std::string& type, const std::string& acc,
                                                          bool required) {
    bool scale = false;
    for (VeOp op : ops) {
        if (op == VeOp::Scale) scale = true;
        else if (is_block_format(op))
            return std::string(to_string(op)) + ": a block format converts real values, and " + tile + " is " + type +
                   " after its scale";
        else if (op != VeOp::Add && op != VeOp::Relu)
            return std::string(to_string(op)) + " on " + tile + "'s " + type + " values: an integer result's "
                   "activation is relu (" + to_string(op) + " needs real values)";
    }
    if (required && !scale)
        return tile + " holds " + acc + " accumulator values; its context needs scale(m, shift) @ combiner, which "
               "makes them " + type;
    return std::nullopt;
}

}  // namespace sw::kpu::program::csp::lang
