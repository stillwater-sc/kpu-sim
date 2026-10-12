// ============================================================================
// include/sw/kpu/program/csp/lang/place.hpp
// L3 placement (docs/plans/l3-placement-at-lb.md; ADR 0004 §5): where every residency lives.
//
// A residency's L3 address is chosen ONCE, by the walker, as a compiler allocates registers, and
// carried on every action that touches it. Every level executes that address; none allocates a
// slot of its own.
//
//   WRONG    each level picks a slot (a per-DMA-engine counter, "any free credit")
//            -> placements no one decided, and no two levels comparable on it
//   CORRECT  L3Allocator::take() as the walker opens a residency; give() as it releases one
//
// The policy is the program's, a schedule choice written in its machine clause
// (`machine flat(l3 = 17, place = fifo)`), never a deployment field:
//   fifo         the slot freed longest ago: a reuse waits on the oldest release   (default)
//   lowest       the lowest free slot: compact, and reuses the slot just freed
//   bank-spread  a free slot in the bank holding the fewest live tiles (lowest such slot)
// A program may also place a residency itself (`resident X @ slots(a..b)`): take_at().
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

enum class PlacePolicy : std::uint8_t { Fifo, Lowest, BankSpread };

inline const char* to_string(PlacePolicy p) {
    switch (p) {
        case PlacePolicy::Fifo:       return "fifo";
        case PlacePolicy::Lowest:     return "lowest";
        case PlacePolicy::BankSpread: return "bank-spread";
    }
    return "?";
}

inline std::optional<PlacePolicy> parse_place_policy(const std::string& s) {
    if (s == "fifo") return PlacePolicy::Fifo;
    if (s == "lowest") return PlacePolicy::Lowest;
    if (s == "bank-spread") return PlacePolicy::BankSpread;
    return std::nullopt;
}

class L3Allocator {
public:
    // One L3 module of `slots` tile buffers in `banks` interleaved banks (level 1).
    L3Allocator(std::uint32_t slots, std::uint32_t banks, PlacePolicy policy)
        : slots_(slots), banks_(banks ? banks : 1), policy_(policy), live_(slots, false),
          bank_live_(banks_, 0) {
        for (std::uint32_t s = 0; s < slots_; ++s) free_.push_back(s);   // fifo: never-used slots in order
    }

    std::uint32_t slots() const { return slots_; }
    std::uint32_t banks() const { return banks_; }
    PlacePolicy policy() const { return policy_; }
    std::size_t held() const { return held_; }
    bool live(std::uint32_t slot) const { return slot < slots_ && live_[slot]; }

    L3Address address(std::uint32_t slot) const { return L3Address{0, slot, slot % banks_}; }

    // A slot by the policy. Throws when every slot is live: the walker checks capacity first, so
    // this is a defect, never a program's error.
    L3Address take() {
        if (free_.empty()) throw std::logic_error("L3Allocator: every slot is live");
        std::uint32_t slot = 0;
        switch (policy_) {
            case PlacePolicy::Fifo: slot = free_.front(); break;
            case PlacePolicy::Lowest: slot = *std::min_element(free_.begin(), free_.end()); break;
            case PlacePolicy::BankSpread: {
                bool found = false;
                for (std::uint32_t s : free_) {
                    if (!found || bank_live_[s % banks_] < bank_live_[slot % banks_] ||
                        (bank_live_[s % banks_] == bank_live_[slot % banks_] && s < slot)) {
                        slot = s;
                        found = true;
                    }
                }
                break;
            }
        }
        return claim(slot);
    }

    // The program's own placement: `slot`, which must be in range and free. Returns the reason it
    // cannot be, or nullopt and the address in `out`.
    std::optional<std::string> take_at(std::uint32_t slot, L3Address& out) {
        if (slot >= slots_) return "slot " + std::to_string(slot) + " is outside the L3 of " + std::to_string(slots_);
        if (live_[slot]) return "slot " + std::to_string(slot) + " holds a live tile";
        out = claim(slot);
        return std::nullopt;
    }

    // A Release: the slot is free again, and joins the back of the fifo order.
    void give(const L3Address& a) {
        if (a.slot >= slots_ || !live_[a.slot])
            throw std::logic_error("L3Allocator: releasing slot " + std::to_string(a.slot) + ", which is not live");
        live_[a.slot] = false;
        --bank_live_[a.slot % banks_];
        --held_;
        free_.push_back(a.slot);
    }

private:
    std::uint32_t slots_, banks_;
    PlacePolicy policy_;
    std::vector<bool> live_;
    std::vector<std::uint32_t> bank_live_;
    std::deque<std::uint32_t> free_;         // free slots, oldest-freed first
    std::size_t held_ = 0;

    L3Address claim(std::uint32_t slot) {
        free_.erase(std::find(free_.begin(), free_.end(), slot));
        live_[slot] = true;
        ++bank_live_[slot % banks_];
        ++held_;
        return address(slot);
    }
};

}  // namespace sw::kpu::program::csp::lang
