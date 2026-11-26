#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/l1_lanes.hpp
// The L1 lane buffers on the cycle core (#543; docs/plans/cycle-based-kpu-prototype.md §5.1, step
// 2b; docs/plans/systolic-tile-at-lca.md §3.3): one buffer per array lane, each holding one block of
// up to a row's elements, streaming it one element per clock with the `first` and `last` markers.
//
// THE CREDIT. Each lane has one, and the credit is `left <= 1`: the buffer is empty, or its last
// element leaves this clock. A block loaded in that clock streams on from the next, with no gap.
// The credit is a token, not a formula: a load spends it, and it returns at the end of the clock in
// which the lane's `left` falls to 1 or below. Every clock, each lane holds its credit exactly when
// left <= 1 (check()): the credit is conserved.
//
// A FAULT, NEVER A STALL. A load into a lane that does not hold its credit raises an L1Fault naming
// the lane and the command: the RTL's A_/B_LOAD_WITHOUT_CREDIT, as #387's SlotFault is the slots'.
//
// Within a clock the owner calls stream() for every lane, then load() for the blocks landing this
// clock, then settle(): a lane's last element leaves before the next block loads.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/time.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

// A lane's protocol broken: a defect in a level (or a schedule the hardware cannot keep), never a
// property of the program.
class L1Fault : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// One element on a lane into the array.
struct L1Element {
    std::uint64_t id = 0;
    std::uint32_t lane = 0;
    std::uint32_t command = 0;          // the Streamer command whose block it is
    std::uint32_t k = 0;                // its place in the block
    std::uint32_t bits = 0;             // the element, from its row's bytes, little-endian
    bool first = false, last = false;   // the reduction's first and last element
};

class L1LaneBuffers {
public:
    struct LaneStats {
        std::uint64_t loads = 0;
        std::uint64_t elements = 0;
        std::uint64_t seamless = 0;   // loads in the clock the lane's previous block's last element left
    };

    // `who` names the buffers in faults ("A" or "B" lanes); `capacity`: the elements a block holds.
    L1LaneBuffers(std::string who, std::uint32_t lanes, std::uint32_t capacity)
        : who_(std::move(who)), capacity_(capacity), lanes_(lanes) {
        if (lanes == 0 || capacity == 0)
            throw std::invalid_argument("cycle::L1LaneBuffers " + who_ + ": lanes and the capacity must be positive");
    }

    // The element lane `lane` streams this clock, if any: the next of its block's.
    std::optional<L1Element> stream(Cycle now, std::uint32_t lane) {
        Lane& l = at(lane);
        if (l.left == 0 || now < l.start) return std::nullopt;
        const std::uint32_t k = static_cast<std::uint32_t>(l.block.size()) - l.left;
        L1Element e{++id_, lane, l.command, k, l.block[k], l.first && k == 0, l.last && l.left == 1};
        --l.left;
        ++l.stats.elements;
        if (l.left == 0) l.emptied = now;
        return e;
    }

    // Loads `block` into `lane`; it streams from the next clock. `first` and `last` mark the block's
    // first and last elements as the reduction's. Raises L1Fault if the lane does not hold its credit.
    void load(Cycle now, std::uint32_t lane, std::vector<std::uint32_t> block, std::uint32_t command, bool first,
              bool last) {
        Lane& l = at(lane);
        if (!l.credit)
            throw L1Fault("cycle::L1LaneBuffers " + who_ + " at cycle " + std::to_string(now) + ": a load into lane " +
                          std::to_string(lane) + " for command " + std::to_string(command) +
                          " without its credit (its block from command " + std::to_string(l.command) + " has " +
                          std::to_string(l.left) + " more element" + (l.left == 1 ? "" : "s") + " to stream)");
        if (block.empty() || block.size() > capacity_)
            throw std::logic_error("cycle::L1LaneBuffers " + who_ + ": a block of " + std::to_string(block.size()) +
                                   " elements into lane " + std::to_string(lane) + ", which holds 1 to " +
                                   std::to_string(capacity_));
        if (l.left != 0)
            throw std::logic_error("cycle::L1LaneBuffers " + who_ + " at cycle " + std::to_string(now) + ": lane " +
                                   std::to_string(lane) + " holds its credit with " + std::to_string(l.left) +
                                   " elements still to stream");
        l.credit = false;
        ++l.stats.loads;
        if (l.emptied == now && l.stats.elements > 0) ++l.stats.seamless;
        l.left = static_cast<std::uint32_t>(block.size());
        l.block = std::move(block);
        l.command = command;
        l.first = first;
        l.last = last;
        l.start = now + 1;
    }

    // End of the clock: a lane whose `left` is at most 1 has its credit back.
    void settle() {
        for (Lane& l : lanes_)
            if (!l.credit && l.left <= 1) l.credit = true;
    }

    // Every clock: each lane holds its credit exactly when left <= 1.
    void check(Cycle now) const {
        for (std::size_t i = 0; i < lanes_.size(); ++i) {
            const Lane& l = lanes_[i];
            if (l.credit != (l.left <= 1) || l.left > capacity_)
                throw std::logic_error("cycle::L1LaneBuffers " + who_ + " at cycle " + std::to_string(now) + ": lane " +
                                       std::to_string(i) + " has " + std::to_string(l.left) + " elements left and " +
                                       (l.credit ? "holds" : "does not hold") + " its credit");
        }
    }

    [[nodiscard]] std::uint32_t lanes() const { return static_cast<std::uint32_t>(lanes_.size()); }
    [[nodiscard]] std::uint32_t capacity() const { return capacity_; }
    [[nodiscard]] bool credit(std::uint32_t lane) const { return lanes_.at(lane).credit; }
    [[nodiscard]] std::uint32_t left(std::uint32_t lane) const { return lanes_.at(lane).left; }
    [[nodiscard]] const LaneStats& stats(std::uint32_t lane) const { return lanes_.at(lane).stats; }
    [[nodiscard]] bool empty() const {
        for (const Lane& l : lanes_)
            if (l.left != 0) return false;
        return true;
    }

private:
    struct Lane {
        std::vector<std::uint32_t> block;
        std::uint32_t left = 0;
        std::uint32_t command = 0;
        bool first = false, last = false;
        bool credit = true;
        Cycle start = 0;                         // the clock its block starts streaming
        Cycle emptied = ~Cycle{0};               // the clock its last element left
        LaneStats stats;
    };
    Lane& at(std::uint32_t lane) {
        if (lane >= lanes_.size())
            throw std::out_of_range("cycle::L1LaneBuffers " + who_ + ": no lane " + std::to_string(lane));
        return lanes_[lane];
    }

    std::string who_;
    std::uint32_t capacity_;
    std::vector<Lane> lanes_;
    std::uint64_t id_ = 0;
};

}  // namespace sw::kpu::cycle
