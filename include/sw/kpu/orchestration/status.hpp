// ============================================================================
// include/sw/kpu/orchestration/status.hpp
// What an orchestrator may SEE (#305 §6.4) — metadata about placement, never payload.
//
// A deciding orchestrator has to look at the machine, and §3 denies it a data path. The
// line between those two is the one #282 increment 3 already drew for the naming map:
//
//   "does this resource exist?"     answerable from the deployment
//   "what tile is resident where?"  answerable from placement state
//   "what is IN it?"                NOT ANSWERABLE, by design
//
// Everything this type exposes is in the first two rows. There is no accessor that returns
// a tensor element, and adding one would make the orchestrator a backdoor — #284 owns the
// backdoor precisely so it stays flagged in provenance.
//
// WHY THIS IS A SEPARATE TYPE rather than methods on the platform: the platform can see
// everything, and the orchestrator must not. A narrow view is how that restriction is
// stated in the type system instead of in a comment nobody is bound by.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/orchestration/descriptor.hpp>
#include <sw/kpu/program/platform/resource_map.hpp>

#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace sw::kpu::orchestration {

using program::platform::DeploymentSpec;
using program::platform::ResourceMap;

class StatusView {
public:
    StatusView(const DeploymentSpec& spec, const std::set<std::string>& resident_keys,
               std::size_t l3_capacity)
        : map_(spec), resident_(resident_keys), l3_capacity_(l3_capacity) {}

    // ---- inventory: what exists -------------------------------------------
    const std::vector<program::platform::ResourceName>& inventory() const {
        return map_.enumerate();
    }
    std::size_t resource_count() const { return map_.size(); }
    bool resource_exists(const program::platform::ResourceName& n) const {
        return map_.exists(n);
    }

    // ---- occupancy: what is resident where --------------------------------
    bool is_resident(const TileRef& t) const { return resident_.count(t.key()) != 0; }
    std::size_t resident_count() const { return resident_.size(); }

    // ---- credits: what can still be placed --------------------------------
    // 0 capacity means UNBOUNDED, matching DeviceSpecification::l3.capacity_tiles, so the
    // credit count is "as many as you like". Reading 0 as "no credits" would make every
    // default deployment look full -- the same conflation DeviceDescriptor::l3_tiles warns
    // about, and the one that would turn a working allocator into an immediate refusal.
    bool unbounded() const { return l3_capacity_ == 0; }
    std::size_t l3_capacity() const { return l3_capacity_; }
    std::size_t credits_available() const {
        if (unbounded()) return static_cast<std::size_t>(-1);
        return l3_capacity_ > resident_.size() ? l3_capacity_ - resident_.size() : 0;
    }
    // Can `count` further tiles be made resident? The question an allocator actually asks.
    bool can_place(std::size_t count) const {
        return unbounded() || resident_.size() + count <= l3_capacity_;
    }

    // There is deliberately no accessor for tile CONTENTS. See the header.

private:
    ResourceMap map_;
    std::set<std::string> resident_;
    std::size_t l3_capacity_ = 0;
};

} // namespace sw::kpu::orchestration
