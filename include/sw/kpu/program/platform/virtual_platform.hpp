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

#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/driver/csp_step.hpp>
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
// ADR 0002 §3.5 names FOUR inputs; an L0 run takes FIVE, and the identity has to say so or it
// is not an identity. `placement` changes which compute tile an op lands on, so two runs
// differing only in it must not compare equal. (The L0 L-T1 executor also took an L1 stream
// annotation and a residency decision; both left with it in kpu-run-csp-programs step 4e. A
// CSP program states its own residency, in `inherit` and `retain`, and its digest covers
// them.)
// A CSP program on the platform (docs/plans/kpu-run-csp-programs.md step 4a): the program is
// the input that configures the data path, and every level runs it.
class CspProgramHandle {
public:
    CspProgramHandle() = default;
    bool valid() const { return valid_; }
    std::size_t index() const { return index_; }
    bool operator==(const CspProgramHandle& o) const { return valid_ == o.valid_ && index_ == o.index_; }

private:
    friend class VirtualPlatform;
    explicit CspProgramHandle(std::size_t i) : index_(i), valid_(true) {}
    std::size_t index_ = 0;
    bool valid_ = false;
};

struct RunIdentity {
    std::string program_digest;      // the program's STRUCTURE (see below)
    std::string snapshot_digest;     // the coverage tag + the state it covers
    std::string deployment_digest;   // the canonical spec bytes
    std::string placement;           // the whole assignment, not its label
    std::string schedule;             // a CSP run's schedule options (L-CA's window); empty for L0
    ExecutionLevel level{};

    bool operator==(const RunIdentity& o) const {
        return program_digest == o.program_digest && snapshot_digest == o.snapshot_digest &&
               deployment_digest == o.deployment_digest && placement == o.placement &&
               schedule == o.schedule && level == o.level;
    }

    // EVERY COMPARED COMPONENT IS RENDERED: a provenance line that cannot tell two runs apart
    // is not provenance.
    std::string str() const {
        std::string out = std::string(driver::short_name(level)) + " prog:" + program_digest +
                          " state:" + snapshot_digest + " deploy:" + deployment_digest +
                          " place:" + digest_of(placement);
        if (!schedule.empty()) out += " schedule:" + schedule;
        return out;
    }
};

// A CSP program's run at one level: its outcome (values, timing) and the identity of the run.
struct CspPlatformRun {
    driver::CspLevelOutcome outcome;
    RunIdentity identity;
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

    // ---- CSP programs (kpu-run-csp-programs step 4a) ---------------------------------------
    // A CSP program and its inputs. Validated at load against device 0's sites, so a program
    // the machine cannot run is refused here, by line. A CSP run is a PURE FUNCTION of
    // (program, inputs, deployment, level, schedule): it reads the inputs it was loaded with
    // and returns its values; it does not leave state on the platform. (Cross-run residency
    // is orchestration's, plan step 4d.)
    // `check_placement` false validates without the machine's sites: values do not depend on
    // where a stage runs, and a caller with no real machine (kpu-run without --deploy) runs the
    // program at the levels that do not time it.
    CspProgramHandle load_csp(csp::lang::Program ast, TileProgram inputs, bool check_placement = true) {
        require_single_device();
        const csp::lang::Target target = csp::lang::target_from(spec_.device(0));
        (void)csp::lang::validate(ast, check_placement ? &target : nullptr);
        CspEntry e;
        e.text = csp::lang::format(ast);
        e.digest = digest_of(e.text);
        serialize::WriteOptions values;
        values.include_values = true;
        e.inputs_digest = digest_of(serialize::to_string(inputs, values));
        e.ast = std::move(ast);
        e.inputs = std::move(inputs);
        csp_.push_back(std::move(e));
        return CspProgramHandle(csp_.size() - 1);
    }

    std::size_t csp_program_count() const { return csp_.size(); }
    const csp::lang::Program& csp_program(CspProgramHandle h) const { return csp_.at(checked(h)).ast; }
    const TileProgram& csp_inputs(CspProgramHandle h) const { return csp_.at(checked(h)).inputs; }
    // The canonical text: the program's identity (two spellings of one program are one).
    const std::string& csp_text(CspProgramHandle h) const { return csp_.at(checked(h)).text; }

    // Why `level` cannot run the program on this platform's machine, or nullopt.
    std::optional<std::string> csp_supported(CspProgramHandle h, ExecutionLevel level) const {
        return driver::csp_level_supported(level, csp_.at(checked(h)).ast, &spec_.device(0));
    }

    // Run the program at `level`. `window` is L-CA's driver window, a schedule option.
    CspPlatformRun run_csp(CspProgramHandle h, ExecutionLevel level, std::size_t window = 256) const {
        require_single_device();
        const CspEntry& e = csp_.at(checked(h));
        CspPlatformRun r;
        r.outcome = driver::csp_run_level(level, e.ast, &spec_.device(0), e.inputs, window);
        r.identity.program_digest = e.digest;
        r.identity.snapshot_digest = e.inputs_digest;
        r.identity.deployment_digest = deployment_digest_;
        r.identity.placement = "single";
        r.identity.level = level;
        if (level == ExecutionLevel::CycleAccurate) r.identity.schedule = "window=" + std::to_string(window);
        return r;
    }

    // The oracle for the program: its trace's L0 on its inputs, within `trace_limit` actions.
    // `observer` watches the run action by action (the domain-flow record is written this way).
    driver::CspReference csp_reference(CspProgramHandle h, std::size_t trace_limit = 1'000'000,
                                       driver::CspObserver* observer = nullptr) const {
        const CspEntry& e = csp_.at(checked(h));
        return driver::csp_reference(e.ast, e.inputs, trace_limit, observer);
    }

    // Stepping a CSP program (kpu-run-csp-programs step 4c): one action per step at L-B,
    // applied; at L-T1 the program runs first and the cursor replays its records in start
    // order, as an L0 program's L-T1 cursor replays its timeline. L-CA steps cycles (#283).
    class CspCursor {
    public:
        driver::CspStepper& stepper() const { return *stepper_; }
        bool executes() const { return stepper_->executes(); }
        ExecutionLevel level() const { return level_; }
        // Present only for a replay, because only a replay has a run behind it.
        const std::optional<CspPlatformRun>& run_result() const { return run_; }

        bool step() { return stepper_->step(); }
        const driver::CspStep& current() const { return stepper_->current(); }
        std::size_t position() const { return stepper_->position(); }
        std::size_t size() const { return stepper_->size(); }
        bool size_known() const { return stepper_->size_known(); }

    private:
        friend class VirtualPlatform;
        std::unique_ptr<driver::CspStepper> stepper_;
        ExecutionLevel level_{};
        std::optional<CspPlatformRun> run_;
    };

    CspCursor step_begin(CspProgramHandle h, ExecutionLevel level) const {
        require_single_device();
        const CspEntry& e = csp_.at(checked(h));
        CspCursor cur;
        cur.level_ = level;
        switch (level) {
            case ExecutionLevel::Behavioral:
                cur.stepper_ = std::make_unique<driver::CspBehavioralStepper>(e.ast, e.inputs);
                return cur;
            case ExecutionLevel::BlockSequential: {
                cur.run_ = run_csp(h, level);
                if (cur.run_->outcome.skipped)
                    throw std::invalid_argument("platform: " + *cur.run_->outcome.skipped);
                cur.stepper_ = std::make_unique<driver::CspReplayStepper>(cur.run_->outcome.records,
                                                                          cur.run_->outcome.slots);
                return cur;
            }
            case ExecutionLevel::ResourceTransactional:
            case ExecutionLevel::CycleAccurate:
                break;
        }
        throw std::invalid_argument(std::string("platform: stepping a CSP program at ") + driver::to_string(level) +
                                    " is not implemented: L-B steps actions, L-T1 replays records; L-CA's step is "
                                    "a cycle (#283)");
    }

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
    // An L0 program runs at L-B only (kpu-run-csp-programs step 4e): it is the trace, and the
    // oracle the timed levels' CSP runs answer to. Any other level is refused by run_at() with
    // the reason.
    PlatformRunResult run(ProgramHandle h, ExecutionLevel level,
                          const StateSnapshot& initial, const Placement& placement) {
        const std::size_t i = checked(h);
        require_single_device();
        restore(initial);

        PlatformRunResult result;
        result.identity.program_digest = program_digest(programs_[i]);
        result.identity.snapshot_digest = initial.digest();
        result.identity.deployment_digest = deployment_digest_;
        result.identity.placement = placement.canonical_bytes();
        result.identity.level = level;
        result.unmodelled = driver::unmodelled_fields(level, spec_);
        result.outcome = driver::run_at(level, programs_[i], spec_.device_view(), placement);
        return result;
    }

    // The common case: one device, its own compute tiles.
    PlatformRunResult run(ProgramHandle h, ExecutionLevel level,
                          const StateSnapshot& initial) {
        return run(h, level, initial, Placement::single(spec_.device_view().compute_tiles));
    }

    // ---- step ---------------------------------------------------------------
    // ONE TRANSACTION PER CALL, at the granularity the level defines (§3.2: "the unit of
    // stepping IS the level"). An L0 program steps at L-B, where each step applies the next op
    // for real, so the platform's state advances as you step and a caller can read values
    // forming. A CSP program's L-T1 cursor (step_begin on a CspProgramHandle) replays records.
    class Cursor {
    public:
        driver::Stepper& stepper() const { return *stepper_; }
        ExecutionLevel level() const { return level_; }
        ProgramHandle program() const { return handle_; }

        bool step() { return stepper_->step(); }
        const driver::StepEvent& current() const { return stepper_->current(); }
        std::size_t position() const { return stepper_->position(); }
        std::size_t size() const { return stepper_->size(); }

    private:
        friend class VirtualPlatform;
        std::unique_ptr<driver::Stepper> stepper_;
        ExecutionLevel level_{};
        ProgramHandle handle_;
    };

    // Restores `initial` FIRST, exactly as run() does -- stepping is execution, so it has
    // the same identity requirement, and a cursor begun from ambient state would be a
    // walk through a run nobody can reproduce.
    Cursor step_begin(ProgramHandle h, ExecutionLevel level, const StateSnapshot& initial) {
        const std::size_t i = checked(h);
        // THE SAME GUARD AS run(), because stepping IS execution. Without it, stepping at L-B
        // would succeed on a deployment that running at L-B refuses.
        require_single_device();
        Cursor cur;
        cur.level_ = level;
        cur.handle_ = h;
        cur.stepper_ = driver::make_stepper(level, programs_[i]);   // refuses all but L-B
        restore(initial);
        return cur;
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

    std::size_t checked(CspProgramHandle h) const {
        if (!h.valid()) throw std::invalid_argument("platform: an unset CspProgramHandle names no program");
        if (h.index() >= csp_.size())
            throw std::invalid_argument("platform: CspProgramHandle " + std::to_string(h.index()) + " is not loaded");
        return h.index();
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

    struct CspEntry {
        csp::lang::Program ast;
        TileProgram inputs;
        std::string text, digest, inputs_digest;
    };
    std::deque<CspEntry> csp_;
};

} // namespace sw::kpu::program::platform
