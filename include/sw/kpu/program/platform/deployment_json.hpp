// ============================================================================
// include/sw/kpu/program/platform/deployment_json.hpp
// The JSON edge of a DeploymentSpec (ADR 0002 §3.5).
//
// DECLARATION-ONLY ON PURPOSE. The `program/` tree is header-only and standard-library
// only, and pulling <nlohmann/json.hpp> into it would hand that dependency to every
// consumer of a TileProgram. The definitions live in src/program/deployment_json.cpp,
// following tools/dfg/common/dfg_json.hpp, so the struct stays usable from anywhere
// while the parsing stays in one translation unit.
//
// CANONICAL FORM, AND SUGAR. to_json() writes exactly one spelling: a "devices" array,
// two-space indent, optional fields OMITTED when absent — which is what lets "declared"
// survive a round trip, since a default value is not a declaration. Two inputs are
// accepted and NORMALIZED to it:
//
//   * a FLAT object with no "devices" key is one device. That is the ADR's own example,
//     and refusing to load the spelling the authority document shows would make the
//     documentation a trap.
//   * "dma": { "burst": N } is read as burst_bytes, for the same reason: the ADR writes
//     "burst". Canonical output uses burst_bytes, because the unit belongs in the name.
//
// So the round-trip claim is precise: CANONICAL bytes round-trip byte-exactly, and a
// non-canonical spec normalizes on the first pass and is stable after it. Both are
// tested; claiming the stronger thing would be false the moment someone writes `64`
// where a double belongs.
//
// AN UNKNOWN KEY IS REFUSED, not ignored. The vocabulary is small and fixed, so an
// unrecognised key is a typo — and a silently ignored "compute_tile" means the machine
// is not the one the file describes while the run reports success.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu::program::platform {

// One exception type with a readable message, and deliberately no cause enum: nothing
// branches on the kind of spec error yet, and inventing causes nobody reads is worse
// than adding them when a caller needs to tell two apart.
class SpecError : public std::runtime_error {
public:
    explicit SpecError(const std::string& what) : std::runtime_error(what) {}
};

// Canonical JSON, ending in a newline so a spec file is a well-formed text file.
std::string to_json(const DeploymentSpec& spec);

// Parse and VALIDATE. A spec that parses but describes an impossible machine (no
// compute tiles, a zero bandwidth) is refused here rather than dividing by zero deep
// inside an executor, where the message would name an executor internal instead of the
// field that was wrong.
DeploymentSpec from_json(const std::string& text);

// Read a file. Refuses an unreadable path with the path in the message.
DeploymentSpec read_spec_file(const std::string& path);

// Over the CANONICAL bytes, so two specs that differ only in spelling have the same
// digest — which is the property a cache key needs. See digest.hpp on what this is and
// is not evidence of.
std::string deployment_digest(const DeploymentSpec& spec);

} // namespace sw::kpu::program::platform
