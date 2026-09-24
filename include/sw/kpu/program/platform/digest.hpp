// ============================================================================
// include/sw/kpu/program/platform/digest.hpp
// A short, stable digest over canonical bytes.
//
// WHAT THIS IS FOR: labelling a run in its provenance, and looking one up in a
// cache. It is NOT the identity claim. ADR 0002 §3.5 requires that two runs with
// the same (program, initial_state, deployment, level) agree, and asserting that
// through a 64-bit hash would be asserting the absence of a collision. The tests
// compare the canonical BYTES of all four inputs; the digest rides along in the
// provenance, where a collision is a cosmetic annoyance rather than a wrong answer.
//
// FNV-1a rather than a cryptographic hash, deliberately: nothing here defends
// against an adversary choosing inputs, and a dependency-free function keeps this
// header usable from the std-only program layer.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace sw::kpu::program::platform {

inline std::uint64_t fnv1a64(std::string_view bytes) {
    std::uint64_t h = 1469598103934665603ull;             // offset basis
    for (unsigned char c : bytes) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ull;                            // prime
    }
    return h;
}

// 16 lowercase hex digits, fixed width so two digests are comparable by eye and sort
// stably in a report.
inline std::string digest_of(std::string_view bytes) {
    const std::uint64_t h = fnv1a64(bytes);
    static const char* kHex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) out[static_cast<std::size_t>(i)] = kHex[(h >> ((15 - i) * 4)) & 0xF];
    return out;
}

} // namespace sw::kpu::program::platform
