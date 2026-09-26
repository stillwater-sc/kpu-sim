// ============================================================================
// include/sw/kpu/program/platform/virtual_platform.hpp
// The object that owns a deployment (ADR 0002 §3.2, §3.5) — #282 increment 2.
//
// A RUN IS A PURE FUNCTION OF (program, initial_state, deployment, level). run_at()
// took three of those four, so the claim was false in a specific, checkable way: a run
// read whatever the previous one left in the program's operands. This takes the fourth
// as an ARGUMENT and restores it FIRST, which is what makes the identity answerable at
// the call site:
//
//   const StateSnapshot initial = platform.snapshot();   // capture the situation
//   auto r = platform.run(handle, level, initial);       // restore it, then execute
//
// Passing it rather than reading whatever the platform happens to hold is the whole
// point. A reviewer sees which state a run started from without reconstructing the
// history of writes before it, and a cache key cannot silently disagree with the state
// that was actually restored.
//
// WHAT IT DELIBERATELY IS NOT, YET:
//
//   * `IInterpreter` (§4). run_at() is already the level->interpreter seam and it is
//     tested; turning it into a virtual interface before the L-T2 implementation exists
//     would be designing an interface from one example. #283 is when a second real
//     implementation appears and the shape becomes knowable.
//   * `CspProgram` (§3.2, #281). The program that exists is the L0 TileProgram. The
//     HANDLE is what makes that temporary: when #281 lands, the type behind the handle
//     changes without touching a caller. Sketching the future type today would produce
//     a name with nothing behind it.
//   * `Backdoor` (§3.4, #284). The snapshot is the surface it will write through, and
//     the coverage tag exists partly for it.
// `step()` (§3.2) is here as of increment 5, reusing driver::make_stepper so #286 and L-T2
// get ONE stepping seam rather than two.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/step_cursor.hpp>
// The JSON header is declaration-only, so this stays a std-only include graph; the
// definition of deployment_digest() lives in kpu_program, which kpu_simulator links.
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/state_snapshot.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <deque>
#include <memory>
#include <set>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::platform {

// ExecutionLevel lives in driver/ because run_at() does, and moving it would churn every
// caller for no behavioural gain. ADR §4 puts levels with the platform, so this alias is
// where that move will land when something else already requires touching those files.
using driver::ExecutionLevel;
using driver::RunOutcome;

// ----------------------------------------------------------------------------
// A handle, not an index
// ----------------------------------------------------------------------------
// Strongly typed and not default-usable: a bare std::size_t would let an uninitialised
// or arithmetic-derived value index a program, and the failure would be a run of the
// WRONG PROGRAM reporting success. Comparable so a caller can key a map on it.
class ProgramHandle {
public:
    ProgramHandle() = default;
    bool valid() const { return valid_; }
    std::size_t index() const { return index_; }
    bool operator==(const ProgramHandle& o) const {
        return valid_ == o.valid_ && index_ == o.index_;
    }
    // VALIDITY FIRST. An unset handle and the first loaded one differ under ==, and
    // comparing only the index made them equivalent under < -- so a std::set or map keyed on
    // handles would silently keep one of the two.
    bool operator<(const ProgramHandle& o) const {
        if (valid_ != o.valid_) return valid_ < o.valid_;
        return index_ < o.index_;
    }

private:
    friend class VirtualPlatform;
    explicit ProgramHandle(std::size_t i) : index_(i), valid_(true) {}
    std::size_t index_ = 0;
    bool valid_ = false;
};

// ----------------------------------------------------------------------------
// The four inputs, named
// ----------------------------------------------------------------------------
// ADR 0002 §3.5 names FOUR inputs; run() actually takes SIX, and the identity has to say so
// or it is not an identity. `placement` changes which compute tile an op lands on, and so the
// schedule; the L1 stream annotation changes per-op timing. Two runs differing only in those
// would have compared EQUAL -- and the whole point of the identity is that they cannot.
//
// The stream annotation is identified by a DIGEST OF ITS CONTENT, not by its map's name. The
// name was the first answer, justified by "a StreamProgram is a pure function of (program,
// map)" -- which is true of every StreamProgram this repo DERIVES, and was an ASSUMPTION
// ABOUT THE CALLER stated in a comment. run() takes a pointer: a caller can hand it a
// StreamProgram whose wavefront timing or element stride has been changed, run_at will produce
// a different makespan, and the name would have recorded the two runs as identical.
//
// An assumption a type cannot enforce does not belong in an identity, so the content is
// digested instead. The name is kept beside it as a LABEL and is deliberately not compared:
// two annotations with identical content ARE the same annotation whatever they are called, and
// comparing both would be two sources of truth for one input.
struct RunIdentity {
    std::string program_digest;      // the program's STRUCTURE (see below)
    std::string snapshot_digest;     // the coverage tag + the state it covers
    std::string deployment_digest;   // the canonical spec bytes
    std::string placement;           // the whole assignment, not its label
    std::string residency;           // tiles seeded resident; empty when the run starts cold
    std::string stream_digest;        // the annotation's CONTENT; empty when there is none
    std::string dataflow;             // the map's name -- a LABEL, not compared (see above)
    ExecutionLevel level{};

    bool operator==(const RunIdentity& o) const {
        return program_digest == o.program_digest && snapshot_digest == o.snapshot_digest &&
               deployment_digest == o.deployment_digest && placement == o.placement &&
               residency == o.residency && stream_digest == o.stream_digest &&
               level == o.level;
    }

    // EVERY COMPARED COMPONENT IS RENDERED. str() printed the map's NAME and not the digest,
    // so two annotations with the same name and different wavefronts -- which operator==
    // correctly distinguishes -- rendered identically in the provenance. A provenance line
    // that cannot tell two runs apart is not provenance; the name rides along in brackets
    // because it is what a reader recognises.
    std::string str() const {
        std::string out = std::string(driver::short_name(level)) + " prog:" + program_digest +
                          " state:" + snapshot_digest + " deploy:" + deployment_digest +
                          " place:" + digest_of(placement);
        if (!residency.empty()) out += " resident:" + digest_of(residency);
        if (!stream_digest.empty())
            out += " flow:" + stream_digest + (dataflow.empty() ? "" : "(" + dataflow + ")");
        return out;
    }
};

struct PlatformRunResult {
    RunOutcome outcome{};
    RunIdentity identity{};
    // What the deployment declared and this level does not model, so a caller never has
    // to ask the deployment and the level separately to find out.
    std::vector<std::string> unmodelled;
};

// ----------------------------------------------------------------------------
// The platform
// ----------------------------------------------------------------------------
class VirtualPlatform {
public:
    explicit VirtualPlatform(DeploymentSpec spec) : spec_(std::move(spec)) {
        // Validated at construction, so an impossible machine cannot be half-deployed.
        const std::string bad = spec_.validate();
        if (!bad.empty()) throw std::invalid_argument("deployment: " + bad);
        deployment_digest_ = deployment_digest(spec_);
    }

    const DeploymentSpec& deployment() const { return spec_; }
    const std::string& deployment_digest_value() const { return deployment_digest_; }

    // ---- programs -----------------------------------------------------------
    // The platform OWNS the program, because the program is where the state lives today.
    // Handing back a handle rather than a reference is what makes that ownership real: a
    // caller that kept its own copy and expected run() to fill it would be holding the
    // state the platform is supposed to be holding.
    ProgramHandle load_program(TileProgram prog) {
        programs_.push_back(std::move(prog));
        return ProgramHandle(programs_.size() - 1);
    }

    std::size_t program_count() const { return programs_.size(); }

    const TileProgram& program(ProgramHandle h) const { return programs_.at(checked(h)); }
    TileProgram& program(ProgramHandle h) { return programs_.at(checked(h)); }

    // ---- state --------------------------------------------------------------
    // PLATFORM-WIDE, as §3.2 has it: one snapshot is the whole situation. Today that is
    // every loaded program's operands; #283 adds resource residency to the same object
    // under a new coverage.
    StateSnapshot snapshot() const {
        StateSnapshot snap(StateCoverage::Operands);
        for (std::size_t i = 0; i < programs_.size(); ++i)
            snap.programs().push_back(capture(programs_[i], i));
        return snap;
    }

    // REFUSES A SNAPSHOT THAT IS NOT ABOUT THIS PLATFORM. A snapshot taken with one
    // program loaded, restored with two, would leave the second holding whatever it
    // happened to hold — silently, and the run would then not be the one the snapshot
    // describes. An empty snapshot is accepted as "nothing staged" only when nothing is
    // loaded, for the same reason.
    void restore(const StateSnapshot& snap) {
        // EXHAUSTIVE, so the guard is at COMPILE time. A StateSnapshot is never
        // deserialized -- it only ever comes from this platform in this process -- so an
        // unknown coverage cannot arrive at runtime, and a runtime check for one would be
        // unreachable code pretending to be a safeguard. Adding a coverage in #283 breaks
        // this switch under -Wall instead, at the site that has to learn to restore it.
        switch (snap.coverage()) {
            case StateCoverage::Operands:
                break;
        }
        if (snap.programs().size() != programs_.size())
            throw std::invalid_argument(
                "snapshot: covers " + std::to_string(snap.programs().size()) +
                " program(s), the platform holds " + std::to_string(programs_.size()) +
                ": restoring it would leave the rest holding state nobody described");
        // EVERY ENTRY IS CHECKED BEFORE ANY IS WRITTEN. Checking and writing in one pass
        // left the platform holding a MIX of old and new state when a later program did not
        // match -- the earlier ones were already overwritten, and the comment above then
        // described a refusal that had not happened. A partial restore is worse than a
        // refused one: the run proceeds and reports success.
        std::vector<bool> seen(programs_.size(), false);
        for (const ProgramState& st : snap.programs()) {
            if (st.program >= programs_.size())
                throw std::invalid_argument("snapshot: program handle " +
                                            std::to_string(st.program) + " is not loaded");
            // The count check alone is not enough: {0, 0} on a two-program platform passes
            // it and leaves program 1 holding whatever it held.
            if (seen[st.program])
                throw std::invalid_argument("snapshot: program handle " +
                                            std::to_string(st.program) +
                                            " appears twice, so one program would be left "
                                            "unrestored");
            seen[st.program] = true;
            const std::string bad = check(st, programs_[st.program]);
            if (!bad.empty()) throw std::invalid_argument(bad);
        }
        for (const ProgramState& st : snap.programs())
            apply(st, programs_[st.program]);          // checked above; cannot throw now
    }

    // ---- execute ------------------------------------------------------------
    // RESTORES `initial` FIRST. That single line is what makes "a run that reads state a
    // previous run left behind" impossible by construction: the only way to observe a
    // previous run's output is to pass a snapshot taken after it, which is visible at the
    // call site.
    //
    // THE RESTORE IS PLATFORM-WIDE, AND THAT HAS A USAGE CONSEQUENCE. A run resets EVERY
    // loaded program, not just the one it executes -- it has to, or the snapshot's digest
    // would claim state the run did not actually restore, and the identity would be a
    // promise the platform does not keep. So a caller comparing several runs must CAPTURE
    // EACH RESULT as it is produced:
    //
    //     for (...) { auto r = platform.run(h[i], level[i], initial);
    //                 results.push_back(platform.program(h[i])); }   // <- not afterwards
    //
    // Reading platform.program(h[0]) after the loop reads the INPUT that the last run's
    // restore put back, not the answer it computed. test_virtual_platform got this wrong
    // first and the assertion caught it, which is the cheapest place to learn it.
    // `initially_resident` is a RUN INPUT, not a hint: a seeded tile's chain skips the DMA
    // leg, so two runs differing only in it produce different makespans and different
    // transfer counts. It therefore belongs in the identity as well as the signature --
    // which is where the "four inputs were really six" lesson lands for the seventh.
    PlatformRunResult run(ProgramHandle h, ExecutionLevel level,
                          const StateSnapshot& initial, const Placement& placement,
                          const stream::StreamProgram* streams = nullptr,
                          const std::set<std::string>& initially_resident = {}) {
        const std::size_t i = checked(h);
        require_single_device();
        restore(initial);

        PlatformRunResult result;
        result.identity.program_digest = program_digest(programs_[i]);
        result.identity.snapshot_digest = initial.digest();
        result.identity.deployment_digest = deployment_digest_;
        result.identity.placement = placement.canonical_bytes();
        result.identity.stream_digest =
            streams ? digest_of(streams->canonical_bytes()) : std::string();
        result.identity.dataflow = streams ? streams->map.name : std::string();
        result.identity.level = level;
        result.unmodelled = driver::unmodelled_fields(level, spec_);
        result.identity.residency = residency_key(initially_resident);
        result.outcome = driver::run_at(level, programs_[i], spec_.device_view(), placement,
                                       streams, 0, initially_resident);
        return result;
    }

    // The common case: one device, its own compute tiles, no stream annotation.
    PlatformRunResult run(ProgramHandle h, ExecutionLevel level,
                          const StateSnapshot& initial) {
        return run(h, level, initial,
                   Placement::single(spec_.device_view().compute_tiles), nullptr);
    }

    // ---- step ---------------------------------------------------------------
    // ONE TRANSACTION PER CALL, at the granularity the level defines (§3.2: "the unit of
    // stepping IS the level").
    //
    // THE TWO LEVELS STEP BY DIFFERENT MECHANISMS, and the cursor says which rather than
    // papering over it:
    //
    //   L-B   EXECUTES. Each step applies the next op for real, so the platform's state
    //         advances as you step and a caller can read values forming.
    //   L-T1  REPLAYS. The executor's schedule depends on the WHOLE program -- credits,
    //         residency and lane contention are decided across it -- so there is no
    //         meaningful "half a schedule". step_begin() therefore RUNS THE PROGRAM TO
    //         COMPLETION first and the cursor walks the timeline that run produced.
    //
    // That difference is not an implementation detail a caller can ignore: at L-T1 the
    // state is already final before the first step() call. `executes()` on the cursor is
    // what a caller must consult before believing that stepping advanced anything, and
    // hiding it behind a uniform interface would make "step until the value appears" a
    // loop that never terminates at one level and works at the other.
    class Cursor {
    public:
        driver::Stepper& stepper() const { return *stepper_; }
        // True when a step APPLIES work (L-B); false when it walks a finished run (L-T1).
        bool executes() const { return !stepper_->models_time(); }
        ExecutionLevel level() const { return level_; }
        ProgramHandle program() const { return handle_; }
        // Present only for a replay, because only a replay has a run behind it.
        const std::optional<PlatformRunResult>& run_result() const { return run_; }

        bool step() { return stepper_->step(); }
        const driver::StepEvent& current() const { return stepper_->current(); }
        std::size_t position() const { return stepper_->position(); }
        std::size_t size() const { return stepper_->size(); }

    private:
        friend class VirtualPlatform;
        std::unique_ptr<driver::Stepper> stepper_;
        ExecutionLevel level_{};
        ProgramHandle handle_;
        std::optional<PlatformRunResult> run_;
    };

    // Restores `initial` FIRST, exactly as run() does -- stepping is execution, so it has
    // the same identity requirement, and a cursor begun from ambient state would be a
    // walk through a run nobody can reproduce.
    Cursor step_begin(ProgramHandle h, ExecutionLevel level, const StateSnapshot& initial,
                      const Placement& placement,
                      const stream::StreamProgram* streams = nullptr) {
        const std::size_t i = checked(h);
        // THE SAME GUARD AS run(), because stepping IS execution. Without it, stepping at L-B
        // would succeed on a deployment that running at L-B refuses -- an inconsistency a
        // caller would have no way to make sense of, and the kind that gets discovered by
        // someone building on the seam rather than by whoever wrote it.
        require_single_device();
        Cursor cur;
        cur.level_ = level;
        cur.handle_ = h;
        if (level == ExecutionLevel::BlockSequential) {
            // The whole run happens here; see the note above on why there is no half a
            // schedule. run() restores the snapshot, so this cursor is reproducible.
            cur.run_ = run(h, level, initial, placement, streams);
            cur.stepper_ = driver::make_stepper(level, programs_[i], cur.run_->outcome.timeline);
        } else {
            restore(initial);
            cur.stepper_ = driver::make_stepper(level, programs_[i], {});
        }
        return cur;
    }

    Cursor step_begin(ProgramHandle h, ExecutionLevel level, const StateSnapshot& initial) {
        return step_begin(h, level, initial,
                          Placement::single(spec_.device_view().compute_tiles), nullptr);
    }

    // A stable rendering of the seeded set. std::set iterates in order, so the key does not
    // depend on the order a caller inserted in -- two orchestrators that chose the same tiles
    // by different routes made the same decision and must share an identity.
    static std::string residency_key(const std::set<std::string>& keys) {
        std::string out;
        for (const std::string& k : keys) out += k + ";";
        return out;
    }

    // THE PROGRAM'S STRUCTURE, NOT ITS VALUES. Values are the `initial_state` input and
    // already have their own digest; folding them in here would make two of the four
    // inputs cover the same bytes, and "identical inputs" would stop meaning four
    // independent things. write_l0's default WriteOptions omit values, and those bytes
    // are byte-stable and tested (#265 increment 3), which is why that check was worth
    // its strictness.
    static std::string program_digest(const TileProgram& prog) {
        return digest_of(serialize::to_string(prog));
    }

private:
    // MULTI-DEVICE EXECUTION IS NOT IMPLEMENTED, and silently using device 0 would be the
    // worst possible version of that: run_at() receives device_view() (device 0) and
    // unmodelled_fields() inspects device 0, so every other device would be ignored WITHOUT
    // BEING REPORTED -- a deployment described and a machine run that are not the same
    // machine.
    //
    // The naming map (increment 3) is multi-device on purpose, because #284 needs every
    // resource addressable. Naming a resource and executing on it are different capabilities,
    // and this is the one that does not exist yet. Stated once so run() and step_begin()
    // cannot disagree about it.
    void require_single_device() const {
        if (spec_.device_count() > 1)
            throw std::invalid_argument(
                "platform: this deployment has " + std::to_string(spec_.device_count()) +
                " devices, and multi-device execution is not implemented -- run_at() would "
                "schedule device 0 and ignore the rest without saying so");
    }

    std::size_t checked(ProgramHandle h) const {
        if (!h.valid())
            throw std::invalid_argument("platform: an unset ProgramHandle names no program");
        if (h.index() >= programs_.size())
            throw std::invalid_argument("platform: ProgramHandle " +
                                        std::to_string(h.index()) + " is not loaded");
        return h.index();
    }

    DeploymentSpec spec_;
    std::string deployment_digest_;
    // A DEQUE, NOT A VECTOR, and this is load-bearing rather than a preference. An L-B
    // Cursor's BehavioralStepper holds a `TileProgram&` into this container, so a later
    // load_program() on a vector could REALLOCATE and every subsequent step() would write
    // through a dangling reference. It is not a theoretical hazard: the test that loads a
    // second program mid-walk SIGSEGVs with a vector here.
    //
    // std::deque::push_back never invalidates references to existing elements, and every use
    // above is size(), operator[] or at(), all of which behave identically. Loading while
    // stepping is an ordinary thing to do, and #286 builds its event record on this cursor.
    std::deque<TileProgram> programs_;
};

} // namespace sw::kpu::program::platform
