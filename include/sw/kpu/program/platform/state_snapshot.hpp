// ============================================================================
// include/sw/kpu/program/platform/state_snapshot.hpp
// An immutable capture of the state a run starts from (ADR 0002 §3.5).
//
// WHY THIS EXISTS. The ADR's central claim is that a run is a pure function of
// (program, initial_state, deployment, level). Three of those four were already
// arguments; this is the fourth. Without it a run reads whatever the previous run
// left behind, which is neither reproducible nor cacheable — and the claim is simply
// false rather than approximately true.
//
// WHAT IT CAN HONESTLY COVER TODAY, which is the part most likely to be built wrong,
// because the ADR describes the end state and the end state does not exist yet:
//
//   The only mutable state a run touches is the TileProgram's OPERAND VALUES. L-T1
//   tracks tile identity, credits and residency — never payload — and L-B computes
//   straight into the operands. There is no L3 tile with contents, no L2 bank and no
//   L1 vector: §3.3's resource vocabulary is #283's work.
//
// So a v1 snapshot is operand values and nothing else. That is complete for the state
// that exists and incomplete against §3.5, and BOTH have to be true in the code or the
// next person reads a digest and believes something false.
//
// HENCE THE COVERAGE TAG, AND HENCE IT IS INSIDE THE DIGEST. An L-T2 snapshot will
// cover operands AND resource residency. Without the tag in the digest, two snapshots
// covering DIFFERENT STATE CLASSES would digest the same whenever their operands
// matched — so a cache would serve an operands-only result to a run that also staged
// L3 contents. The tag makes that collision impossible rather than unlikely.
//
// NOT A FILE FORMAT. The canonical bytes are an in-process representation: float bits
// are copied verbatim, which is exact and not portable. L0 is the portable format
// (#265), and a snapshot that needs to cross a machine belongs there. Nothing is lost
// by saying so: this project builds Release with -march=native, so a result is already
// host-specific and a cross-machine cache would be unsound for a different reason.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/digest.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::platform {

// What classes of state a snapshot covers. Add a value here and the digest changes for
// every snapshot taken at the new coverage, which is the intended consequence.
enum class StateCoverage {
    Operands,               // v1: operand values only
    // OperandsAndResidency -- #283, when resources hold state
};

inline const char* to_string(StateCoverage c) {
    switch (c) {
        case StateCoverage::Operands: return "operands";
    }
    return "?";
}

// One loaded program's state. The handle index is carried so a snapshot cannot be
// restored into a platform holding a different set of programs -- see restore().
struct ProgramState {
    std::size_t program = 0;
    std::string program_name;
    // Operand name -> values, in the program's DECLARATION order, so the canonical
    // bytes do not depend on a map's iteration order.
    std::vector<std::pair<std::string, std::vector<float>>> operands;
};

class StateSnapshot {
public:
    StateSnapshot() = default;
    explicit StateSnapshot(StateCoverage coverage) : coverage_(coverage) {}

    StateCoverage coverage() const { return coverage_; }
    const std::vector<ProgramState>& programs() const { return programs_; }
    std::vector<ProgramState>& programs() { return programs_; }
    bool empty() const { return programs_.empty(); }

    // Exactly what the digest covers, and what a test compares when it needs to assert
    // that two runs really had the same input rather than the same hash.
    //
    // THE COVERAGE TAG COMES FIRST, so it can never be mistaken for operand data: a
    // snapshot at a future coverage cannot produce these bytes by coincidence.
    std::string canonical_bytes() const {
        std::string out = "coverage=";
        out += to_string(coverage_);
        out += "\n";
        for (const ProgramState& p : programs_) {
            out += "program " + std::to_string(p.program) + " \"" + p.program_name + "\"\n";
            for (const auto& [name, values] : p.operands) {
                out += "  " + name + " " + std::to_string(values.size()) + " ";
                // Float BITS, verbatim: a decimal rendering would have to choose a
                // precision, and any choice short of exact makes two different states
                // digest the same. -0.0f and 0.0f are different states.
                const std::size_t bytes = values.size() * sizeof(float);
                const std::size_t at = out.size();
                out.resize(at + bytes);
                if (bytes) std::memcpy(out.data() + at, values.data(), bytes);
                out += "\n";
            }
        }
        return out;
    }

    std::string digest() const { return digest_of(canonical_bytes()); }

private:
    StateCoverage coverage_ = StateCoverage::Operands;
    std::vector<ProgramState> programs_;
};

// Capture one program's operands. Free functions rather than members, so StateSnapshot
// stays a value with no opinion about where state comes from -- which is what lets #283
// add resource residency without changing it.
inline ProgramState capture(const TileProgram& prog, std::size_t handle) {
    ProgramState st;
    st.program = handle;
    st.program_name = prog.name();
    for (const std::string& key : prog.operand_order())
        st.operands.emplace_back(key, prog.operand(key).values);
    return st;
}

// CHECKING IS SEPARATE FROM WRITING, and that separation is the whole point. An earlier
// version validated and assigned in one loop, so a snapshot whose SECOND program did not
// match left the FIRST one already overwritten -- the platform then held a mix of old and
// new state, which is exactly the "state nobody described" that restore() claims to refuse.
// A partial restore is worse than a refused one, because the run proceeds and reports
// success.
//
// So: check() answers "is this snapshot about this program?" without touching it, and
// assign() cannot fail. restore() checks EVERY program before it writes ANY.
//
// STRICT about shape on purpose: an operand the snapshot does not carry, or a size that
// does not match, means the snapshot and the program are not about the same thing.
// Returns the problem, or empty when there is none.
inline std::string check(const ProgramState& st, const TileProgram& prog) {
    if (st.operands.size() != prog.operand_order().size())
        return "snapshot: program \"" + prog.name() + "\" has " +
               std::to_string(prog.operand_order().size()) + " operands, the snapshot carries " +
               std::to_string(st.operands.size());
    for (const auto& [name, values] : st.operands) {
        if (!prog.has_operand(name))
            return "snapshot: no operand \"" + name + "\" in program \"" + prog.name() + "\"";
        if (prog.operand(name).values.size() != values.size())
            return "snapshot: operand \"" + name + "\" holds " +
                   std::to_string(prog.operand(name).values.size()) +
                   " values, the snapshot carries " + std::to_string(values.size());
    }
    return {};
}

// Write one program's operands back. Throws only via check(), so a caller that has already
// checked cannot be left half-applied.
inline void apply(const ProgramState& st, TileProgram& prog) {
    const std::string bad = check(st, prog);
    if (!bad.empty()) throw std::invalid_argument(bad);
    for (const auto& [name, values] : st.operands) prog.operand(name).values = values;
}

} // namespace sw::kpu::program::platform
