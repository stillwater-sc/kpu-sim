// ============================================================================
// tests/program/test_virtual_platform.cpp
// A run is a pure function of (program, initial_state, deployment, level)
// (#282 increment 2). run_at() took three of the four; the platform takes all of them
// and restores the state FIRST.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/virtual_platform.hpp>

#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;
using namespace sw::kpu::program::platform;
using sw::kpu::program::Cycle;

namespace {

bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

bool all_operands_identical(const TileProgram& a, const TileProgram& b) {
    if (a.operand_order() != b.operand_order()) return false;
    for (const std::string& key : a.operand_order())
        if (!bit_identical(a.operand(key).values, b.operand(key).values)) return false;
    return true;
}

ProgramSpec spec_for(const char* algo) {
    ProgramSpec ps;
    ps.algo = algo;
    ps.size = 32;
    ps.tile = 16;
    return ps;
}

TileProgram filled(const char* algo) {
    const ProgramSpec ps = spec_for(algo);
    TileProgram p = derive(ps);
    fill(p, ps);
    return p;
}

VirtualPlatform default_platform() {
    return VirtualPlatform(make_deployment(DeviceSpec{}));
}

} // namespace

TEST_CASE("the same four inputs produce the same result", "[program][platform][run]") {
    // The claim ADR 0002 §3.5 rests on. Asserted by comparing the CANONICAL BYTES of the
    // inputs rather than their digests: asserting it through a 64-bit hash would be
    // asserting the absence of a collision, which is not what the issue asks for.
    for (const char* algo : {"matmul", "lu"}) {
        VirtualPlatform platform = default_platform();
        const ProgramHandle h = platform.load_program(filled(algo));
        const StateSnapshot initial = platform.snapshot();

        const auto first = platform.run(h, ExecutionLevel::BlockSequential, initial);
        const TileProgram after_first = platform.program(h);

        // Same program, same state, same deployment, same level.
        const auto second = platform.run(h, ExecutionLevel::BlockSequential, initial);

        INFO("algo " << algo);
        CHECK(first.identity == second.identity);
        CHECK(first.outcome.makespan == second.outcome.makespan);
        CHECK(all_operands_identical(after_first, platform.program(h)));

        // THE INPUTS REALLY WERE THE SAME INPUTS, BY BYTES AND NOT BY HASH -- which is the
        // claim docs/plans/virtual-platform.md §4 makes and the reason the digest is not the
        // identity. The first version of this compared canonical_bytes().SIZE() of a snapshot
        // taken AFTER the run: operand shapes never change, so it passed unconditionally and
        // proved nothing. It was the one assertion standing behind the headline property.
        platform.restore(initial);
        CHECK(platform.snapshot().canonical_bytes() == initial.canonical_bytes());
        CHECK(initial.canonical_bytes().size() > 64);   // not two empty strings
    }
}

TEST_CASE("a run cannot read what the previous run left behind",
          "[program][platform][run]") {
    // TILE LU IS THE RIGHT WITNESS because it factors A IN PLACE: running it on its own
    // output gives a different answer. That is what makes the assertion above evidence of
    // the RESTORE rather than of an idempotent program -- a matmul that zeroes and
    // re-accumulates C would pass either way.
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("lu"));
    const StateSnapshot initial = platform.snapshot();

    platform.run(h, ExecutionLevel::BlockSequential, initial);
    const TileProgram once = platform.program(h);

    // Passing `initial` again resets the situation, so the answer is the same.
    platform.run(h, ExecutionLevel::BlockSequential, initial);
    CHECK(all_operands_identical(once, platform.program(h)));

    // ...and the program is genuinely not idempotent: starting from the PREVIOUS OUTPUT
    // gives something else. If this failed, the check above would prove nothing.
    const StateSnapshot after = platform.snapshot();
    CHECK(after.digest() != initial.digest());
    platform.run(h, ExecutionLevel::BlockSequential, after);
    CHECK_FALSE(all_operands_identical(once, platform.program(h)));
}

TEST_CASE("the platform holds the state, so a snapshot after a run is the result",
          "[program][platform][run]") {
    // The reason load_program takes the program BY VALUE and hands back a handle: a caller
    // holding its own copy and expecting run() to fill it would be holding the state the
    // platform is supposed to hold.
    VirtualPlatform platform = default_platform();
    TileProgram mine = filled("matmul");
    const ProgramHandle h = platform.load_program(mine);
    const StateSnapshot initial = platform.snapshot();

    platform.run(h, ExecutionLevel::BlockSequential, initial);

    // The caller's copy is untouched; the platform's holds the result.
    bool caller_computed_something = false;
    for (float v : mine.operand("C").values)
        caller_computed_something = caller_computed_something || (v != 0.0f);
    CHECK_FALSE(caller_computed_something);

    bool platform_computed_something = false;
    for (float v : platform.program(h).operand("C").values)
        platform_computed_something = platform_computed_something || (v != 0.0f);
    CHECK(platform_computed_something);

    // And restoring the initial snapshot puts it back, which is what makes a sweep over
    // one platform sound.
    platform.restore(initial);
    CHECK(all_operands_identical(mine, platform.program(h)));
}

TEST_CASE("every level run from one snapshot agrees, through the platform",
          "[program][platform][run]") {
    // The differential test of #285, now expressed as the pure function: same program,
    // same state, same deployment, differing only in LEVEL.
    for (const char* algo : {"matmul", "lu"}) {
        VirtualPlatform platform = default_platform();
        std::vector<ProgramHandle> handles;
        std::vector<ExecutionLevel> levels;
        for (ExecutionLevel l : all_levels()) {
            if (!level_implemented(l)) continue;
            handles.push_back(platform.load_program(filled(algo)));
            levels.push_back(l);
        }
        REQUIRE(handles.size() >= 2);
        // One snapshot covers every loaded program, and each was filled identically -- so
        // the state each level starts from is the same state.
        const StateSnapshot initial = platform.snapshot();

        // EACH RESULT IS CAPTURED IMMEDIATELY, and that is not incidental: restore() is
        // platform-wide, so the NEXT run resets every program -- including the one that
        // just produced an answer. Comparing platform.program(h) after the loop compares
        // one level's output against another level's restored INPUT, which is what the
        // first version of this test did and what the platform is right to do. The purity
        // guarantee has a usage consequence, and this is it.
        std::vector<RunIdentity> ids;
        std::vector<TileProgram> results;
        for (std::size_t i = 0; i < handles.size(); ++i) {
            const auto r = platform.run(handles[i], levels[i], initial);
            ids.push_back(r.identity);
            results.push_back(platform.program(handles[i]));
        }
        INFO("algo " << algo);
        for (std::size_t i = 1; i < results.size(); ++i)
            CHECK(all_operands_identical(results[0], results[i]));
        // The identities differ in exactly one field: the level.
        for (std::size_t i = 1; i < ids.size(); ++i) {
            CHECK(ids[i].program_digest == ids[0].program_digest);
            CHECK(ids[i].snapshot_digest == ids[0].snapshot_digest);
            CHECK(ids[i].deployment_digest == ids[0].deployment_digest);
            CHECK(ids[i].level != ids[0].level);
            CHECK_FALSE(ids[i] == ids[0]);
        }
    }
}

TEST_CASE("the identity moves when any one of the four inputs moves",
          "[program][platform][run]") {
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();
    const auto base = platform.run(h, ExecutionLevel::BlockSequential, initial);

    SECTION("a different level") {
        const auto other = platform.run(h, ExecutionLevel::Behavioral, initial);
        CHECK(other.identity.level != base.identity.level);
        CHECK_FALSE(other.identity == base.identity);
    }
    SECTION("a different state") {
        platform.restore(initial);
        platform.program(h).operand("A").values[0] += 1.0f;
        const StateSnapshot nudged = platform.snapshot();
        CHECK(nudged.digest() != initial.digest());
        const auto other = platform.run(h, ExecutionLevel::BlockSequential, nudged);
        CHECK(other.identity.program_digest == base.identity.program_digest);
        CHECK(other.identity.snapshot_digest != base.identity.snapshot_digest);
    }
    SECTION("a different program") {
        const ProgramHandle other_h = platform.load_program(filled("lu"));
        // The snapshot now has to cover both programs, or restoring it is refused.
        const StateSnapshot both = platform.snapshot();
        const auto other = platform.run(other_h, ExecutionLevel::BlockSequential, both);
        CHECK(other.identity.program_digest != base.identity.program_digest);
    }
    SECTION("a different deployment") {
        DeviceSpec ds;
        ds.compute_tiles = 4;
        VirtualPlatform bigger(make_deployment(ds));
        const ProgramHandle bh = bigger.load_program(filled("matmul"));
        const auto other = bigger.run(bh, ExecutionLevel::BlockSequential, bigger.snapshot());
        CHECK(other.identity.deployment_digest != base.identity.deployment_digest);
        // Same program and same state, though: the machine changed, not the problem.
        CHECK(other.identity.program_digest == base.identity.program_digest);
        CHECK(other.identity.snapshot_digest == base.identity.snapshot_digest);
    }
}

TEST_CASE("the program digest covers structure, and the snapshot covers values",
          "[program][platform][run]") {
    // The four inputs have to be FOUR INDEPENDENT THINGS. Folding values into the program
    // digest would make two of them cover the same bytes, and "identical inputs" would stop
    // meaning what it says.
    const ProgramSpec ps = spec_for("matmul");
    TileProgram a = derive(ps);
    TileProgram b = derive(ps);
    fill(b, ps);
    CHECK(VirtualPlatform::program_digest(a) == VirtualPlatform::program_digest(b));

    VirtualPlatform platform = default_platform();
    const ProgramHandle ha = platform.load_program(a);
    const ProgramHandle hb = platform.load_program(b);
    const StateSnapshot snap = platform.snapshot();
    REQUIRE(snap.programs().size() == 2);
    // ...and the state does tell them apart, which is where the difference belongs.
    CHECK(snap.programs()[0].operands != snap.programs()[1].operands);
    CHECK(platform.program(ha).name() == platform.program(hb).name());
}

TEST_CASE("a snapshot that is not about this platform is refused",
          "[program][platform][state]") {
    // Half-restoring would start a run from a state nobody described while reporting
    // success -- the same shape as every other self-contradiction this codebase refuses.
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot one_program = platform.snapshot();

    platform.load_program(filled("lu"));
    // Taken with one program loaded, restored with two: the second would keep whatever it
    // held, silently.
    CHECK_THROWS_AS(platform.restore(one_program), std::invalid_argument);
    CHECK_THROWS_AS(platform.run(h, ExecutionLevel::Behavioral, one_program),
                    std::invalid_argument);
    // The snapshot the platform gives now is accepted.
    CHECK_NOTHROW(platform.restore(platform.snapshot()));

    // A snapshot from a DIFFERENT platform, whose programs have different shapes, is
    // refused on shape rather than silently truncated.
    VirtualPlatform other = default_platform();
    ProgramSpec big = spec_for("matmul");
    big.size = 64;
    TileProgram wide = derive(big);
    fill(wide, big);
    other.load_program(wide);
    other.load_program(filled("lu"));
    CHECK_THROWS_AS(platform.restore(other.snapshot()), std::invalid_argument);
}

TEST_CASE("an unset or unknown handle names no program", "[program][platform][state]") {
    // A bare std::size_t would let an uninitialised value index a program, and the failure
    // would be a run of the WRONG program reporting success.
    VirtualPlatform platform = default_platform();
    const ProgramHandle unset;
    CHECK_FALSE(unset.valid());
    CHECK_THROWS_AS(platform.program(unset), std::invalid_argument);
    CHECK_THROWS_AS(platform.run(unset, ExecutionLevel::Behavioral, platform.snapshot()),
                    std::invalid_argument);

    const ProgramHandle a = platform.load_program(filled("matmul"));
    const ProgramHandle b = platform.load_program(filled("lu"));
    CHECK(a.valid());
    CHECK_FALSE(a == b);
    CHECK(platform.program(a).name() != platform.program(b).name());
    CHECK(platform.program_count() == 2);

    // A handle from another platform happens to be in range here, which is precisely why
    // a handle is not a promise about WHICH program -- so the test states what it does
    // guarantee: the index is checked, not the provenance.
    VirtualPlatform other = default_platform();
    other.load_program(filled("matmul"));
    CHECK_NOTHROW(platform.program(a));
}

TEST_CASE("an impossible machine cannot be half-deployed", "[program][platform][deploy]") {
    DeploymentSpec bad;
    bad.devices.clear();
    CHECK_THROWS_AS(VirtualPlatform(bad), std::invalid_argument);

    DeploymentSpec zero;
    zero.device(0).compute_tiles = 0;
    CHECK_THROWS_AS(VirtualPlatform(zero), std::invalid_argument);
}

TEST_CASE("the result carries what the level does not model",
          "[program][platform][deploy]") {
    // So a caller never has to ask the deployment and the level separately to find out
    // that a declared field evaporated in the projection.
    DeploymentSpec spec;
    spec.device(0).l3.capacity_tiles = 8;
    spec.device(0).l3.banks = 4;
    VirtualPlatform platform(spec);
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();

    const auto at_t1 = platform.run(h, ExecutionLevel::BlockSequential, initial);
    CHECK(at_t1.unmodelled.size() == 1);                     // l3.banks
    CHECK(at_t1.unmodelled.front().find("l3.banks") != std::string::npos);

    const auto at_b = platform.run(h, ExecutionLevel::Behavioral, initial);
    CHECK(at_b.unmodelled.size() == 2);                      // + the capacity
    CHECK(at_b.identity.deployment_digest == at_t1.identity.deployment_digest);
}

TEST_CASE("the coverage tag is part of what a snapshot digests",
          "[program][platform][state]") {
    // Why it has to be: an L-T2 snapshot will cover operands AND resource residency, and
    // without the tag in the bytes two snapshots covering DIFFERENT STATE CLASSES would
    // digest the same whenever their operands matched. A cache would then serve an
    // operands-only result to a run that also staged L3 contents.
    VirtualPlatform platform = default_platform();
    platform.load_program(filled("matmul"));
    const StateSnapshot snap = platform.snapshot();
    const std::string bytes = snap.canonical_bytes();

    // The tag leads, so it cannot be mistaken for operand data.
    CHECK(bytes.rfind("coverage=operands\n", 0) == 0);
    CHECK(snap.digest() == digest_of(bytes));

    // Re-tag the same state and the digest moves. This stands in for a second coverage
    // value, which does not exist yet and should not be invented here: what is being
    // asserted is that the tag PARTICIPATES, which is the property #283 will depend on.
    const std::string retagged =
        "coverage=operands+residency\n" + bytes.substr(std::strlen("coverage=operands\n"));
    CHECK(digest_of(retagged) != snap.digest());

    // -0.0f and 0.0f are different states, so the bytes must distinguish them -- a decimal
    // rendering at any finite precision would not.
    VirtualPlatform signs = default_platform();
    TileProgram z("z");
    z.add_operand(TensorOperand("A", 1, 2, 1, 2));
    const ProgramHandle zh = signs.load_program(z);
    signs.program(zh).operand("A").values = {0.0f, 0.0f};
    const std::string positive = signs.snapshot().digest();
    signs.program(zh).operand("A").values = {-0.0f, 0.0f};
    CHECK(signs.snapshot().digest() != positive);
}

TEST_CASE("a snapshot refuses to be applied to a program of another shape",
          "[program][platform][state]") {
    // apply() is strict on purpose: a missing operand or a mismatched size means the
    // snapshot and the program are not about the same thing.
    const ProgramSpec ps = spec_for("matmul");
    TileProgram p = derive(ps);
    ProgramState st = capture(p, 0);

    st.operands.pop_back();
    CHECK_THROWS_AS(apply(st, p), std::invalid_argument);

    ProgramState resized = capture(p, 0);
    resized.operands.front().second.pop_back();
    CHECK_THROWS_AS(apply(resized, p), std::invalid_argument);

    ProgramState renamed = capture(p, 0);
    renamed.operands.front().first = "Z";
    CHECK_THROWS_AS(apply(renamed, p), std::invalid_argument);

    CHECK_NOTHROW(apply(capture(p, 0), p));
}

// ----------------------------------------------------------------------------
// Review of #303
// ----------------------------------------------------------------------------
TEST_CASE("a snapshot that does not fit is refused WITHOUT half-restoring",
          "[program][platform][state]") {
    // The failure the header comment claimed was impossible and was not: apply() validated
    // and assigned in one loop, so a snapshot whose SECOND program did not match left the
    // FIRST already overwritten. The platform then held a mix of old and new state, and the
    // run proceeded and reported success -- worse than a refusal, because nothing said so.
    VirtualPlatform platform = default_platform();
    const ProgramHandle a = platform.load_program(filled("matmul"));
    const ProgramHandle b = platform.load_program(filled("lu"));
    const StateSnapshot good = platform.snapshot();

    // Break the SECOND entry only, leaving the first perfectly restorable.
    StateSnapshot broken = good;
    REQUIRE(broken.programs().size() == 2);
    broken.programs()[1].operands.front().second.pop_back();

    // Move both programs away from the snapshot, so a restore would be observable.
    platform.program(a).operand("A").values[0] += 1.0f;
    platform.program(b).operand("A").values[0] += 1.0f;
    const StateSnapshot moved = platform.snapshot();

    CHECK_THROWS_AS(platform.restore(broken), std::invalid_argument);
    // NOTHING was written: the platform still holds exactly what it held before the attempt.
    CHECK(platform.snapshot().canonical_bytes() == moved.canonical_bytes());

    // ...and a good snapshot still restores both, so the check is not simply refusing
    // everything.
    CHECK_NOTHROW(platform.restore(good));
    CHECK(platform.snapshot().canonical_bytes() == good.canonical_bytes());
}

TEST_CASE("a snapshot naming one program twice is refused", "[program][platform][state]") {
    // The COUNT check alone passes {0, 0} on a two-program platform, and program 1 would be
    // left holding whatever it held -- a silent partial restore that the count made look
    // complete.
    VirtualPlatform platform = default_platform();
    platform.load_program(filled("matmul"));
    platform.load_program(filled("lu"));
    StateSnapshot duplicated = platform.snapshot();
    REQUIRE(duplicated.programs().size() == 2);
    duplicated.programs()[1] = duplicated.programs()[0];      // both name program 0

    CHECK_THROWS_AS(platform.restore(duplicated), std::invalid_argument);
    try {
        platform.restore(duplicated);
        FAIL("a duplicated program handle must be refused");
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("twice") != std::string::npos);
    }
}

TEST_CASE("a multi-device deployment cannot be run, and says why",
          "[program][platform][run]") {
    // run_at() receives device_view() -- device 0 -- and unmodelled_fields() inspects device
    // 0, so a two-device deployment would schedule the first machine and ignore the rest
    // WITHOUT REPORTING IT. Naming a resource (increment 3, multi-device on purpose) and
    // executing on it are different capabilities; this is the one that does not exist.
    DeviceSpecification a;
    a.name = "left";
    DeviceSpecification b;
    b.name = "right";
    DeploymentSpec spec;
    spec.devices = {a, b};

    VirtualPlatform platform(spec);                 // constructing is fine: naming works
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();
    CHECK(platform.deployment().device_count() == 2);

    try {
        platform.run(h, ExecutionLevel::BlockSequential, initial);
        FAIL("a multi-device run must be refused");
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        CHECK(what.find("multi-device") != std::string::npos);
        CHECK(what.find("2 devices") != std::string::npos);
    }

    // One device still runs, so the guard is a guard and not a blanket.
    DeploymentSpec one;
    one.devices = {a};
    VirtualPlatform single(one);
    const ProgramHandle sh = single.load_program(filled("matmul"));
    CHECK_NOTHROW(single.run(sh, ExecutionLevel::BlockSequential, single.snapshot()));
}

// ----------------------------------------------------------------------------
// Second review of #303
// ----------------------------------------------------------------------------
TEST_CASE("a snapshot naming one operand twice is refused", "[program][platform][state]") {
    // A COUNT IS NOT A COVER. {A, A} on a two-operand program passes the count check, and
    // apply() then writes A twice and leaves B untouched -- the partial restore the check
    // exists to prevent, one level below where it was first closed.
    const ProgramSpec ps = spec_for("matmul");
    TileProgram p = derive(ps);
    fill(p, ps);
    REQUIRE(p.operand_order().size() == 3);

    ProgramState doubled = capture(p, 0);
    doubled.operands[1] = doubled.operands[0];        // A, A, C -- count still 3
    CHECK_FALSE(check(doubled, p).empty());
    CHECK(check(doubled, p).find("twice") != std::string::npos);
    CHECK_THROWS_AS(apply(doubled, p), std::invalid_argument);

    // ...and through the platform, nothing is written.
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(p);
    StateSnapshot broken = platform.snapshot();
    broken.programs()[0].operands[1] = broken.programs()[0].operands[0];
    platform.program(h).operand("B").values[0] += 1.0f;
    const StateSnapshot moved = platform.snapshot();
    CHECK_THROWS_AS(platform.restore(broken), std::invalid_argument);
    CHECK(platform.snapshot().canonical_bytes() == moved.canonical_bytes());
}

TEST_CASE("the identity covers the placement and the dataflow too",
          "[program][platform][run]") {
    // ADR 0002 §3.5 names four inputs; run() takes SIX. `placement` changes which compute tile
    // an op lands on and so the schedule, and the L1 annotation changes per-op timing -- so
    // two runs differing only in those must not compare equal.
    VirtualPlatform platform(make_deployment([] {
        DeviceSpec ds;
        ds.compute_tiles = 4;
        return ds;
    }()));
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();

    const auto unpinned = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                       Placement::single(4), nullptr);
    // A pinned placement over the SAME compute tiles: label() cannot tell these apart, which
    // is exactly why the identity records the assignment instead of the label.
    std::vector<Dim> assignment(platform.program(h).ops().size(), 0);
    const auto pinned_a = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                       Placement::pinned(assignment, 4), nullptr);
    std::vector<Dim> other = assignment;
    other.back() = 3;
    const auto pinned_b = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                       Placement::pinned(other, 4), nullptr);

    CHECK(Placement::single(4).label() != Placement::pinned(assignment, 4).label());
    CHECK(Placement::pinned(assignment, 4).label() == Placement::pinned(other, 4).label());
    CHECK_FALSE(unpinned.identity == pinned_a.identity);
    CHECK_FALSE(pinned_a.identity == pinned_b.identity);      // the label would have matched
    CHECK(pinned_a.identity.program_digest == pinned_b.identity.program_digest);

    // The dataflow is recorded as the map's NAME, because a StreamProgram is a pure function
    // of (program, map) and both are already in the identity (#265 increment 4's reasoning).
    auto streams = stream::derive_matmul_streams(platform.program(h), map_for("ws"));
    const auto annotated = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                       Placement::single(4), &streams);
    CHECK(annotated.identity.dataflow == map_for("ws").name);
    CHECK(unpinned.identity.dataflow.empty());
    CHECK_FALSE(annotated.identity == unpinned.identity);

    auto other_flow = stream::derive_matmul_streams(platform.program(h), map_for("as"));
    const auto annotated_b = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                         Placement::single(4), &other_flow);
    CHECK_FALSE(annotated.identity == annotated_b.identity);

    // A TAMPERED ANNOTATION MUST NOT SHARE AN IDENTITY. The first version recorded the map's
    // NAME, justified by "a StreamProgram is a pure function of (program, map)" -- true of
    // every one this repo derives, and an assumption about the CALLER stated in a comment.
    // run() takes a pointer, so a caller can change a wavefront's depth or an element stride,
    // get a different makespan, and the name would have called the two runs identical. An
    // assumption a type cannot enforce does not belong in an identity.
    auto tampered = streams;
    REQUIRE_FALSE(tampered.computes.empty());
    tampered.computes.begin()->second.k_depth += 1;
    CHECK(tampered.map.name == streams.map.name);          // the NAME is unchanged...
    const auto tampered_run = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                          Placement::single(4), &tampered);
    CHECK(tampered_run.identity.dataflow == annotated.identity.dataflow);   // ...and so is the label
    CHECK_FALSE(tampered_run.identity == annotated.identity);               // but not the identity
    // ...and the RENDERED provenance distinguishes them too. str() printed the name and not
    // the digest, so two runs operator== correctly told apart rendered identically -- and a
    // provenance line that cannot tell two runs apart is not provenance.
    CHECK(tampered_run.identity.str() != annotated.identity.str());
    // Every compared component appears in the rendering, which is what makes that true.
    CHECK(annotated.identity.str().find(annotated.identity.stream_digest) != std::string::npos);
    CHECK(annotated.identity.str().find(annotated.identity.program_digest) != std::string::npos);
    CHECK(annotated.identity.str().find(annotated.identity.snapshot_digest) != std::string::npos);
    CHECK(annotated.identity.str().find(annotated.identity.deployment_digest) != std::string::npos);
    CHECK(unpinned.identity.str().find("flow:") == std::string::npos);   // none to report

    // Same for a signature field the L1 cost model reads.
    auto strided = streams;
    REQUIRE_FALSE(strided.signatures.empty());
    strided.signatures.begin()->second.element_stride += 1;
    const auto strided_run = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                         Placement::single(4), &strided);
    CHECK_FALSE(strided_run.identity == annotated.identity);

    // A RATE THAT DIFFERS BEYOND SIX DECIMALS MUST STILL DIFFER. std::to_string(double) gives
    // six decimal places, so 1.0 and 1.0000001 rendered identically and two annotations that
    // schedule differently shared an identity. That is the same mistake #265 increment 2 found
    // in the L0 format's `alpha`, in a file written the same day -- which is why the assertion
    // is here rather than left to the reader's trust in max_digits10.
    auto fine = streams;
    REQUIRE_FALSE(fine.signatures.empty());
    fine.signatures.begin()->second.rate = 1.0;
    auto finer = streams;
    finer.signatures.begin()->second.rate = 1.0000001;
    CHECK(fine.canonical_bytes() != finer.canonical_bytes());
    const auto fine_run = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                      Placement::single(4), &fine);
    const auto finer_run = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                       Placement::single(4), &finer);
    CHECK_FALSE(fine_run.identity == finer_run.identity);

    // THE NAME REALLY IS OUTSIDE THE COMPARISON. The rule was stated in a comment while the
    // name sat inside the digested bytes, so renaming a map and changing nothing else gave a
    // different identity for the same run -- the code contradicting its own documentation.
    auto renamed = streams;
    renamed.map.name = "a-different-label";
    CHECK(renamed.canonical_bytes() == streams.canonical_bytes());
    const auto renamed_run = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                         Placement::single(4), &renamed);
    CHECK(renamed_run.identity == annotated.identity);            // same run...
    CHECK(renamed_run.identity.dataflow != annotated.identity.dataflow);   // ...different label

    // ...while an identical annotation, freshly derived, digests the same -- so the check is
    // about CONTENT and not about object identity.
    auto rederived = stream::derive_matmul_streams(platform.program(h), map_for("ws"));
    CHECK(rederived.canonical_bytes() == streams.canonical_bytes());
    const auto rederived_run = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                           Placement::single(4), &rederived);
    CHECK(rederived_run.identity == annotated.identity);

    // And two DIFFERENT maps still differ, so dropping the name from the bytes did not make
    // the four presets indistinguishable -- what a map does is tau and proj, and those remain.
    CHECK(stream::derive_matmul_streams(platform.program(h), map_for("os")).canonical_bytes() !=
          streams.canonical_bytes());

    // Same six inputs, same identity -- the property all of this exists to support.
    const auto again = platform.run(h, ExecutionLevel::BlockSequential, initial,
                                    Placement::single(4), &streams);
    CHECK(again.identity == annotated.identity);
}

TEST_CASE("an unset handle sorts apart from the first loaded one",
          "[program][platform][state]") {
    // Comparing only the index made them equivalent under <, so a std::set or map keyed on
    // handles would silently keep one of the two.
    VirtualPlatform platform = default_platform();
    const ProgramHandle unset;
    const ProgramHandle first = platform.load_program(filled("matmul"));
    CHECK_FALSE(unset == first);
    CHECK((unset < first) != (first < unset));          // strictly ordered, either way round
    std::set<ProgramHandle> keys{unset, first};
    CHECK(keys.size() == 2);
}

// ----------------------------------------------------------------------------
// Closing the class, rather than the instances
// ----------------------------------------------------------------------------
TEST_CASE("every RunIdentity field is either compared or a declared label, and all are rendered",
          "[program][platform][run]") {
    // FOUR REVIEW ROUNDS ON THIS PR FOUND THE SAME SHAPE OF BUG: a field of RunIdentity that
    // was not compared when it should have been (placement, the stream content), or not
    // rendered when it was compared (stream_digest), or compared when the comment said it was
    // a label (the map name). Fixing each instance leaves the class open, and the class is
    // "someone adds a field and forgets one of the two".
    //
    // So this test walks every field. The STRUCTURED BINDING below is the tripwire: its arity
    // must match RunIdentity exactly, so adding a member is a COMPILE ERROR here rather than a
    // silent omission -- the same technique as the exhaustive switch over StateCoverage, and
    // for the same reason. A designated-initializer aggregate would NOT do: a new member would
    // just default-initialise and the test would still build.
    const RunIdentity base{"prog0",  "state0",     "deploy0", "place0",
                           "resid0", "flow0",      "the-label",
                           ExecutionLevel::BlockSequential};
    {
        const auto& [program, state, deployment, placement, residency, stream, dataflow,
                     level] = base;
        (void)program; (void)state; (void)deployment; (void)placement;
        (void)residency; (void)stream; (void)dataflow; (void)level;
    }

    // ---- the COMPARED fields: perturbing any one must change == AND str() ----------------
    auto perturbed = [&](auto&& mutate) {
        RunIdentity o = base;
        mutate(o);
        return o;
    };
    const std::vector<std::pair<const char*, RunIdentity>> compared = {
        {"program_digest",    perturbed([](RunIdentity& o) { o.program_digest += "x"; })},
        {"snapshot_digest",   perturbed([](RunIdentity& o) { o.snapshot_digest += "x"; })},
        {"deployment_digest", perturbed([](RunIdentity& o) { o.deployment_digest += "x"; })},
        {"placement",         perturbed([](RunIdentity& o) { o.placement += "x"; })},
        // Added when #305 increment 2 made seeded residency a run input: a seeded tile's
        // chain skips the DMA leg, so two runs differing only here produce different
        // makespans. The tripwire below is what forced this entry rather than letting the
        // field be compared-but-unrendered, or rendered-but-uncompared.
        {"residency",         perturbed([](RunIdentity& o) { o.residency += "x"; })},
        {"stream_digest",     perturbed([](RunIdentity& o) { o.stream_digest += "x"; })},
        {"level",             perturbed([](RunIdentity& o) { o.level = ExecutionLevel::Behavioral; })},
    };
    for (const auto& [field, other] : compared) {
        INFO("field " << field);
        CHECK_FALSE(other == base);                       // it is part of the identity...
        CHECK(other.str() != base.str());                 // ...and it is visible in provenance
    }
    CHECK(base == base);

    // ---- the DECLARED LABEL: not compared, but still rendered ---------------------------
    // `dataflow` is the map's name. Two annotations with identical content ARE the same
    // annotation whatever they are called, so it must NOT be compared -- and it must still be
    // rendered, because it is what a reader recognises. Both halves are asserted, since the
    // first version of this code got the comparison wrong and a later one got the rendering
    // wrong.
    const RunIdentity relabelled = perturbed([](RunIdentity& o) { o.dataflow = "other-label"; });
    CHECK(relabelled == base);
    CHECK(relabelled.str() != base.str());

    // Every compared string appears verbatim in the rendering, so "rendered" is not satisfied
    // by a digest of a digest.
    const std::string text = base.str();
    for (const std::string& part : {base.program_digest, base.snapshot_digest,
                                    base.deployment_digest, base.stream_digest, base.dataflow}) {
        INFO("part " << part);
        CHECK(text.find(part) != std::string::npos);
    }
    // The placement and the seeded residency are rendered as DIGESTS rather than verbatim,
    // because a pinned assignment is one number per op and a resident set is one key per
    // tile -- either would swamp the line. Stated here so the exceptions are deliberate
    // rather than oversights, and asserted both ways: the digest appears, the raw bytes do
    // not.
    CHECK(text.find(digest_of(base.placement)) != std::string::npos);
    CHECK(text.find("place0") == std::string::npos);
    CHECK(text.find(digest_of(base.residency)) != std::string::npos);
    CHECK(text.find("resid0") == std::string::npos);
}

TEST_CASE("the residency decision is serialized unambiguously", "[program][platform][run]") {
    // A SEPARATOR IS NOT A SERIALIZATION. Nothing constrains a tile key's characters --
    // `TileCoord::operand` takes any string -- so joining a set with ";" made one rendering
    // ambiguous: the single key `A#0#0;B#0#0` and the two-key set {`A#0#0`, `B#0#0`} produced
    // identical bytes. Two runs seeding different tiles then compared EQUAL in the identity
    // while producing different makespans, and an identity that can collide is worse than no
    // identity, because it is trusted. Length prefixes fix it.
    //
    // The seeds here match no tile in the program on purpose: what is under test is the
    // RENDERING of a decision, and identical behaviour is what leaves the identity as the only
    // thing that can tell the two runs apart.
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();
    const Placement pl = Placement::single(platform.deployment().device_view().compute_tiles);

    const auto one_key = platform.run(h, ExecutionLevel::BlockSequential, initial, pl, nullptr,
                                      {"A#0#0;B#0#0"});
    const auto two_keys = platform.run(h, ExecutionLevel::BlockSequential, initial, pl, nullptr,
                                       {"A#0#0", "B#0#0"});
    CHECK_FALSE(one_key.identity == two_keys.identity);
    CHECK(one_key.identity.program_digest == two_keys.identity.program_digest);

    // SEEDED and RETAINED are different claims about the same key, so moving a key between
    // them must change the identity: one says "already there", the other "must still be there
    // afterwards", and they produce different transfer counts.
    const auto seeded = platform.run(h, ExecutionLevel::BlockSequential, initial, pl, nullptr,
                                     {"A#0#0"}, {});
    const auto retained = platform.run(h, ExecutionLevel::BlockSequential, initial, pl, nullptr,
                                       {}, {"A#0#0"});
    CHECK_FALSE(seeded.identity == retained.identity);

    // A cold run that keeps nothing still renders EMPTY, not "i[]r[]" -- the field's own
    // comment says "empty when the run starts cold", and a rendering that is never empty would
    // make every run look as though it had made a residency decision.
    const auto cold = platform.run(h, ExecutionLevel::BlockSequential, initial, pl, nullptr);
    CHECK(cold.identity.residency.empty());
    CHECK(cold.identity.str().find("resident:") == std::string::npos);
}

// ----------------------------------------------------------------------------
// Stepping on the platform (#282 increment 5)
// ----------------------------------------------------------------------------
TEST_CASE("at L-B a step executes, so the state advances as you step",
          "[program][platform][step]") {
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();

    auto cur = platform.step_begin(h, ExecutionLevel::Behavioral, initial);
    CHECK(cur.executes());
    CHECK(cur.level() == ExecutionLevel::Behavioral);
    CHECK(cur.program() == h);
    // There is no run behind an L-B cursor, because nothing has run: the steps ARE the run.
    CHECK_FALSE(cur.run_result().has_value());
    CHECK(cur.size() == platform.program(h).ops().size());

    // Nothing computed yet.
    bool before = false;
    for (float v : platform.program(h).operand("C").values) before = before || (v != 0.0f);
    CHECK_FALSE(before);

    // Step to the end and the result appears -- in the PLATFORM's program, which is what
    // "stepping advances the state" means.
    std::size_t taken = 0;
    while (cur.step()) ++taken;
    CHECK(taken == cur.size());
    CHECK(cur.position() == cur.size());
    bool after = false;
    for (float v : platform.program(h).operand("C").values) after = after || (v != 0.0f);
    CHECK(after);

    // And it is the same answer run() gives, which is the only thing that makes stepping a
    // way to understand a run rather than a separate model.
    VirtualPlatform reference = default_platform();
    const ProgramHandle rh = reference.load_program(filled("matmul"));
    reference.run(rh, ExecutionLevel::Behavioral, reference.snapshot());
    CHECK(all_operands_identical(reference.program(rh), platform.program(h)));
}

TEST_CASE("at L-T1 a step replays a finished run, and the cursor says so",
          "[program][platform][step]") {
    // The executor's schedule depends on the WHOLE program -- credits, residency and lane
    // contention are decided across it -- so there is no meaningful half a schedule.
    // step_begin() runs to completion and the cursor walks the timeline that run produced.
    // A caller that believed stepping advanced the state would be wrong here, which is why
    // executes() exists rather than a uniform pretence.
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();

    auto cur = platform.step_begin(h, ExecutionLevel::BlockSequential, initial);
    CHECK_FALSE(cur.executes());
    REQUIRE(cur.run_result().has_value());
    CHECK(cur.run_result()->outcome.makespan > 0);
    CHECK(cur.run_result()->identity.level == ExecutionLevel::BlockSequential);

    // The state is ALREADY FINAL before the first step.
    bool computed = false;
    for (float v : platform.program(h).operand("C").values) computed = computed || (v != 0.0f);
    CHECK(computed);

    std::size_t taken = 0;
    Cycle last = 0;
    while (cur.step()) {
        // A replay is ordered in time, which is what makes it readable.
        CHECK(cur.current().at >= last);
        last = cur.current().at;
        ++taken;
    }
    CHECK(taken > 0);
    CHECK(taken == cur.size());
    CHECK(last == cur.run_result()->outcome.makespan);
}

TEST_CASE("a cursor starts from the snapshot it was given", "[program][platform][step]") {
    // Stepping is execution, so it carries the same identity requirement as run(): a cursor
    // begun from ambient state would be a walk through a run nobody can reproduce.
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("lu"));
    const StateSnapshot initial = platform.snapshot();

    auto first = platform.step_begin(h, ExecutionLevel::Behavioral, initial);
    std::vector<std::size_t> ops_first;
    while (first.step()) ops_first.push_back(first.current().op_index);
    const TileProgram after_first = platform.program(h);

    // Begun again from the SAME snapshot -- not from the state the first walk left.
    auto second = platform.step_begin(h, ExecutionLevel::Behavioral, initial);
    std::vector<std::size_t> ops_second;
    while (second.step()) ops_second.push_back(second.current().op_index);
    CHECK(ops_first == ops_second);
    CHECK(all_operands_identical(after_first, platform.program(h)));
}

TEST_CASE("a level with no stepper is refused, and so is a multi-device deployment",
          "[program][platform][step]") {
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();
    CHECK_THROWS_AS(platform.step_begin(h, ExecutionLevel::ResourceTransactional, initial),
                    std::invalid_argument);
    CHECK_THROWS_AS(platform.step_begin(h, ExecutionLevel::CycleAccurate, initial),
                    std::invalid_argument);
    CHECK_THROWS_AS(platform.step_begin(ProgramHandle{}, ExecutionLevel::Behavioral, initial),
                    std::invalid_argument);

    // THE SAME GUARD AS run(). Stepping at L-B would otherwise succeed where running at L-B
    // is refused, and an inconsistency like that gets discovered by whoever builds on the
    // seam rather than by whoever wrote it.
    DeviceSpecification a, b;
    a.name = "left";
    b.name = "right";
    DeploymentSpec two;
    two.devices = {a, b};
    VirtualPlatform multi(two);
    const ProgramHandle mh = multi.load_program(filled("matmul"));
    const StateSnapshot ms = multi.snapshot();
    CHECK_THROWS_AS(multi.step_begin(mh, ExecutionLevel::Behavioral, ms),
                    std::invalid_argument);
    CHECK_THROWS_AS(multi.run(mh, ExecutionLevel::Behavioral, ms), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// Review of #304
// ----------------------------------------------------------------------------
TEST_CASE("loading a program while a cursor is live does not invalidate it",
          "[program][platform][step]") {
    // AN L-B STEPPER HOLDS A TileProgram& INTO THE PLATFORM'S CONTAINER. With a
    // std::vector, a later load_program() can reallocate and every subsequent step() writes
    // through a dangling reference. Loading a second program while stepping the first is an
    // ordinary thing to do -- and #286 builds its event record on this cursor, so the seam
    // has to survive it.
    //
    // std::deque::push_back never invalidates references to existing elements, which is what
    // makes this test pass rather than merely usually pass.
    VirtualPlatform platform = default_platform();
    const ProgramHandle h = platform.load_program(filled("matmul"));
    const StateSnapshot initial = platform.snapshot();

    auto cur = platform.step_begin(h, ExecutionLevel::Behavioral, initial);
    REQUIRE(cur.executes());
    CHECK(cur.step());                       // one step, then grow the container

    // Enough appends to force a vector to reallocate several times over.
    for (int i = 0; i < 16; ++i) platform.load_program(filled("lu"));
    CHECK(platform.program_count() == 17);

    while (cur.step()) {}
    CHECK(cur.position() == cur.size());

    // The stepped program holds the right answer, which is what a dangling write would have
    // destroyed.
    VirtualPlatform reference = default_platform();
    const ProgramHandle rh = reference.load_program(filled("matmul"));
    reference.run(rh, ExecutionLevel::Behavioral, reference.snapshot());
    CHECK(all_operands_identical(reference.program(rh), platform.program(h)));
}

TEST_CASE("a sweep list reports the cause it actually found", "[program][driver]") {
    // The inner throws were INSIDE the try, so the catch for std::invalid_argument caught
    // this function's OWN diagnostics and relabelled them: "12abc" reported "is not an
    // integer" rather than "has trailing characters", and an out-of-range value reported the
    // same. The exit code was right and the message sent the reader somewhere else.
    auto why = [](const char* csv) {
        try {
            parse_ints(csv);
        } catch (const std::invalid_argument& e) {
            return std::string(e.what());
        }
        return std::string("<accepted>");
    };
    CHECK(why("12abc").find("trailing characters") != std::string::npos);
    CHECK(why("4294967296").find("out of range") != std::string::npos);
    CHECK(why("99999999999999999999999").find("out of range") != std::string::npos);
    CHECK(why("abc").find("not an integer") != std::string::npos);
    CHECK(why("-2").find("non-negative") != std::string::npos);
    CHECK(why("32,64") == "<accepted>");
    CHECK(parse_ints("32,64") == std::vector<std::uint32_t>{32, 64});
}
