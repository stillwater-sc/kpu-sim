// ============================================================================
// tests/program/test_driver_run_at.cpp
// The run_at seam (#285 §D2): one function maps a level to an interpreter, and
// it refuses a level it cannot run rather than substituting one it can.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/driver/step_cursor.hpp>
#include <sw/kpu/program/driver/timeline_trace.hpp>

#include <cstring>
#include <map>
#include <string>
#include <variant>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;

namespace {

bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

} // namespace

TEST_CASE("every implemented level computes identical values", "[program][driver]") {
    // The whole reason the driver exists: values are level-invariant (ADR 0002 §2), so
    // this is a differential test of the models, not a smoke test of the CLI.
    for (const char* algo : {"matmul", "lu"}) {
        ProgramSpec ps;
        ps.algo = algo;
        ps.size = 48;
        ps.tile = 16;
        DeviceSpec ds;
        const auto device = make_device(ds);

        std::vector<float> authority;
        std::size_t authority_swaps = 0;
        std::vector<Dim> authority_perm;
        for (ExecutionLevel l : all_levels()) {
            if (!level_implemented(l)) continue;
            TileProgram p = derive(ps);
            fill(p, ps);
            const auto out = run_at(l, p, device, Placement::single(device.compute_tiles));
            const auto& vals = p.operand(result_operand(ps)).values;
            if (l == ExecutionLevel::Behavioral) {
                authority = vals;
                authority_swaps = out.summary.row_swaps;
                authority_perm = out.summary.permutation;
                continue;
            }
            CHECK(bit_identical(authority, vals));
            // LU's pivoting is state a value diff alone would not catch.
            CHECK(out.summary.row_swaps == authority_swaps);
            CHECK(out.summary.permutation == authority_perm);
        }
        REQUIRE_FALSE(authority.empty());
    }
}

TEST_CASE("a level with no interpreter is refused, never substituted",
          "[program][driver]") {
    ProgramSpec ps;
    TileProgram p = derive(ps);
    fill(p, ps);
    // Snapshot the result operand, so "nothing ran" is checkable. Asserting only that the
    // vector is non-empty would pass even if run_at had computed a full result into it --
    // the vector is sized by derive(), not by the run.
    const std::vector<float> c_before = p.operand("C").values;
    const DeviceSpec ds;
    const auto device = make_device(ds);

    // The failure mode this guards against is a driver that quietly answers a
    // different question than it was asked.
    for (ExecutionLevel l : {ExecutionLevel::ResourceTransactional,
                             ExecutionLevel::CycleAccurate}) {
        REQUIRE_FALSE(level_implemented(l));
        CHECK_THROWS_AS(run_at(l, p, device, Placement::single(1)), std::invalid_argument);
        // The refusal has to say where to follow it up, or it is just a dead end.
        CHECK(not_implemented_reason(l).find("#283") != std::string::npos);
    }
    // And nothing ran: the operand is bit-for-bit what fill() left there.
    REQUIRE_FALSE(c_before.empty());
    CHECK(bit_identical(p.operand("C").values, c_before));
}

TEST_CASE("L-B reports that it models no timing, rather than a timing of zero",
          "[program][driver]") {
    // makespan 0 would read as "instant". has_timing == false is a different claim,
    // and the distinction matters when a report is compared across levels.
    ProgramSpec ps;
    TileProgram p = derive(ps);
    fill(p, ps);
    const DeviceSpec ds;
    const auto out = run_at(ExecutionLevel::Behavioral, p, make_device(ds),
                            Placement::single(1));
    CHECK_FALSE(out.has_timing);
    CHECK_FALSE(out.stats.has_value());
    CHECK_FALSE(out.provenance.has_value());
    CHECK(out.ops > 0);

    TileProgram q = derive(ps);
    fill(q, ps);
    const auto t1 = run_at(ExecutionLevel::BlockSequential, q, make_device(ds),
                           Placement::single(1));
    CHECK(t1.has_timing);
    CHECK(t1.makespan > 0);
    REQUIRE(t1.stats.has_value());
    REQUIRE(t1.provenance.has_value());
    CHECK_FALSE(t1.provenance->calibrated);      // increment 6 has not happened
}

TEST_CASE("level names round-trip, and an unknown name is rejected",
          "[program][driver]") {
    for (ExecutionLevel l : all_levels()) {
        const auto back = parse_level(to_string(l));
        REQUIRE(back.has_value());
        CHECK(*back == l);
        const auto shortly = parse_level(short_name(l));
        REQUIRE(shortly.has_value());
        CHECK(*shortly == l);
    }
    CHECK_FALSE(parse_level("transactional").has_value());   // ambiguous under ADR 0002
    CHECK_FALSE(parse_level("csp").has_value());             // CSP is not a level at all
    CHECK_FALSE(parse_level("").has_value());
}

TEST_CASE("the shared spec builds the device the executor actually schedules on",
          "[program][driver]") {
    // §D1: one mapping, shared with the characterizer. Movement is per process, so the
    // spec must carry per-process lanes and no aggregate movement pool.
    DeviceSpec ds;
    ds.dma_engines = 2;
    ds.block_movers = 3;
    ds.streamers = 4;
    ds.l3_tiles = 24;
    const auto d = make_device(ds);
    CHECK(d.dma_engines == 2);
    CHECK(d.block_movers == 3);
    CHECK(d.streamers == 4);
    CHECK(d.l3_tiles == 24);
    CHECK(d.label().find("dma2") != std::string::npos);
    CHECK(d.label().find("bm3") != std::string::npos);
    CHECK(d.label().find("str4") != std::string::npos);
}

// ============================================================================
// Increment 2 — the timeline mapping (§D5)
//
// Nothing serialized TileOpRecord or HopRecord before this, so "reuse the trace
// format" meant writing the mapping. These tests pin the property that makes the
// mapping worth having: ONE EVENT PER HOP, so a transfer's legs stay separable.
// ============================================================================

TEST_CASE("the timeline emits one event per hop, not one per op",
          "[program][driver][timeline]") {
    ProgramSpec ps;
    ps.size = 48;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);
    DeviceSpec ds;
    const auto device = make_device(ds);
    const auto out = run_at(ExecutionLevel::BlockSequential, p, device,
                            Placement::single(device.compute_tiles));
    REQUIRE_FALSE(out.timeline.empty());

    const auto entries = to_trace_entries(p, out.timeline, device.element_bytes);

    // Count the legs the run actually recorded, then require exactly that many events
    // plus one for each op that recorded no leg (compute, and anything residency freed).
    std::size_t hops = 0, hopless = 0;
    for (const auto& rec : out.timeline) {
        if (rec.hops.empty()) ++hopless; else hops += rec.hops.size();
    }
    REQUIRE(hops > 0);
    CHECK(entries.size() == hops + hopless);
    // A per-op mapping would have produced far fewer: the whole point is that a
    // transfer's legs remain separable.
    CHECK(entries.size() > out.timeline.size());
}

TEST_CASE("each event lands on the component that performed the leg",
          "[program][driver][timeline]") {
    ProgramSpec ps;
    ps.size = 48;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);
    DeviceSpec ds;
    const auto device = make_device(ds);
    const auto out = run_at(ExecutionLevel::BlockSequential, p, device,
                            Placement::single(device.compute_tiles));
    const auto entries = to_trace_entries(p, out.timeline, device.element_bytes);

    // The hop names its governing process, and the trace vocabulary has one component per
    // process, so this must be a faithful rename rather than a guess.
    CHECK(component_of(Hop::DmaDramToL3) == sw::trace::ComponentType::DMA_ENGINE);
    CHECK(component_of(Hop::DmaL3ToDram) == sw::trace::ComponentType::DMA_ENGINE);
    CHECK(component_of(Hop::BlockMoverL3ToL2) == sw::trace::ComponentType::BLOCK_MOVER);
    CHECK(component_of(Hop::BlockMoverL2ToL3) == sw::trace::ComponentType::BLOCK_MOVER);
    CHECK(component_of(Hop::StreamerL2ToL1) == sw::trace::ComponentType::STREAMER);
    CHECK(component_of(Hop::StreamerL1ToL2) == sw::trace::ComponentType::STREAMER);

    // Every event's interval must be the leg's interval, exactly: a trace whose spans
    // disagree with the run it came from is worse than no trace.
    std::size_t checked = 0;
    std::size_t e = 0;
    for (const auto& rec : out.timeline) {
        if (rec.hops.empty()) { ++e; continue; }
        for (const auto& hr : rec.hops) {
            REQUIRE(e < entries.size());
            CHECK(entries[e].cycle_issue == hr.start);
            CHECK(entries[e].cycle_complete == hr.finish);
            CHECK(entries[e].component_type == component_of(hr.hop));
            CHECK(entries[e].component_id == hr.lane);
            CHECK(entries[e].description.find(to_string(hr.hop)) != std::string::npos);
            ++e;
            ++checked;
        }
    }
    CHECK(checked > 0);
    CHECK(e == entries.size());
}

TEST_CASE("compute events carry MAC counts and movement events carry bytes",
          "[program][driver][timeline]") {
    ProgramSpec ps;
    ps.size = 32;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);
    DeviceSpec ds;
    const auto device = make_device(ds);
    const auto out = run_at(ExecutionLevel::BlockSequential, p, device,
                            Placement::single(device.compute_tiles));
    const auto entries = to_trace_entries(p, out.timeline, device.element_bytes);

    std::size_t with_macs = 0, with_bytes = 0;
    for (const auto& en : entries) {
        if (const auto* c = std::get_if<sw::trace::ComputePayload>(&en.payload)) {
            CHECK(c->num_operations > 0);
            CHECK_FALSE(c->kernel_name.empty());
            ++with_macs;
        } else if (const auto* d = std::get_if<sw::trace::DMAPayload>(&en.payload)) {
            CHECK(d->bytes_transferred > 0);
            ++with_bytes;
        }
    }
    CHECK(with_macs > 0);
    CHECK(with_bytes > 0);
}

TEST_CASE("a stream program changes timing and leaves values alone",
          "[program][driver][timeline]") {
    // --streams derives the L1 stream program, whose drain bubble lands on the Streamer
    // legs. It must move the clock without touching the arithmetic.
    ProgramSpec ps;
    ps.size = 48;
    ps.tile = 16;
    DeviceSpec ds;
    const auto device = make_device(ds);

    auto run = [&](bool use_streams, std::vector<float>& values) {
        TileProgram p = derive(ps);
        fill(p, ps);
        stream::StreamProgram sp;
        if (use_streams) sp = stream::derive_matmul_streams(p, map_for("output-stationary"));
        const auto o = run_at(ExecutionLevel::BlockSequential, p, device,
                              Placement::single(device.compute_tiles),
                              use_streams ? &sp : nullptr);
        values = p.operand(result_operand(ps)).values;
        return o;
    };
    std::vector<float> plain_vals, stream_vals;
    const auto plain = run(false, plain_vals);
    const auto streamed = run(true, stream_vals);

    CHECK(streamed.makespan != plain.makespan);        // the clock moved
    CHECK(bit_identical(plain_vals, stream_vals));     // the values did not
    CHECK(streamed.provenance->l1_timing);
    CHECK_FALSE(plain.provenance->l1_timing);
}

TEST_CASE("the NoC has no trace identity, and the mapping refuses to invent one",
          "[program][driver][timeline]") {
    // Mapping an L3->L3 leg onto BLOCK_MOVER would put NoC lane 0 and BlockMover lane 0 on
    // the same track, misreporting the occupancy of both. Picking a component for a link
    // the trace vocabulary does not model is a decision about that vocabulary, so this
    // refuses instead.
    CHECK_THROWS_AS(component_of(Hop::BlockMoverL3ToL3), std::invalid_argument);
    // The other three are faithful renames and must not throw.
    CHECK_NOTHROW(component_of(Hop::DmaDramToL3));
    CHECK_NOTHROW(component_of(Hop::BlockMoverL3ToL2));
    CHECK_NOTHROW(component_of(Hop::StreamerL2ToL1));

    // The guard is unreachable today: no chain contains an L3->L3 leg, even with NoC links
    // configured, so a run still traces cleanly.
    ProgramSpec ps;
    ps.size = 32;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);
    DeviceSpec ds;
    ds.noc_links = 4;
    const auto device = make_device(ds);
    const auto out = run_at(ExecutionLevel::BlockSequential, p, device,
                            Placement::single(device.compute_tiles));
    for (const auto& rec : out.timeline)
        for (const auto& hr : rec.hops)
            CHECK(hr.hop != Hop::BlockMoverL3ToL3);
    CHECK_NOTHROW(to_trace_entries(p, out.timeline, device.element_bytes));
}

TEST_CASE("an option present without a value is an error, not an absence",
          "[program][driver]") {
    // arg() cannot tell those apart: it returns the fallback both when an option is absent
    // and when it is the last token. For a flag that must carry a value those are
    // different errors, and only one of them is silent.
    std::string out, err;
    const std::vector<std::string> terminal{"--size", "32", "--timeline"};
    CHECK(arg_present(terminal, "--timeline"));
    CHECK_FALSE(arg_required(terminal, "--timeline", out, err));
    CHECK(err.find("missing value") != std::string::npos);

    // A following flag is not a value either, or `--timeline --step` writes a trace to a
    // file called "--step".
    const std::vector<std::string> flag_as_value{"--timeline", "--step"};
    err.clear();
    CHECK_FALSE(arg_required(flag_as_value, "--timeline", out, err));
    CHECK(err.find("--step") != std::string::npos);

    // Absent leaves the default alone and is NOT an error.
    out = "untouched";
    err.clear();
    const std::vector<std::string> absent{"--size", "32"};
    CHECK(arg_required(absent, "--timeline", out, err));
    CHECK(out == "untouched");
    CHECK(err.empty());

    // And a real value is taken.
    const std::vector<std::string> given{"--timeline", "run.json"};
    CHECK(arg_required(given, "--timeline", out, err));
    CHECK(out == "run.json");
}

// ============================================================================
// Increment 3 — single-stepping (§D4)
//
// One step is one transaction AT THAT LEVEL. The two levels differ in kind, not
// just in grain: L-B steps by executing, L-T1 steps by replaying, and the tests
// hold each to what it actually claims.
// ============================================================================

TEST_CASE("behavioral stepping executes, one op at a time", "[program][driver][step]") {
    ProgramSpec ps;
    ps.size = 32;
    ps.tile = 16;

    // Stepping every op must land in the same place as running the level outright. If it
    // did not, the stepper would be showing a different computation than the one it claims
    // to be stepping through.
    TileProgram whole = derive(ps);
    fill(whole, ps);
    const DeviceSpec ds;
    run_at(ExecutionLevel::Behavioral, whole, make_device(ds), Placement::single(1));

    TileProgram stepped = derive(ps);
    fill(stepped, ps);
    BehavioralStepper cur(stepped);
    CHECK(cur.size() == stepped.ops().size());
    CHECK_FALSE(cur.models_time());

    std::size_t steps = 0;
    while (cur.step()) {
        CHECK(cur.current().kind == StepKind::OpApplied);
        CHECK(cur.current().op_index == steps);
        ++steps;
    }
    CHECK(steps == stepped.ops().size());
    CHECK(bit_identical(whole.operand(result_operand(ps)).values,
                        stepped.operand(result_operand(ps)).values));
}

TEST_CASE("behavioral stepping shows values forming", "[program][driver][step]") {
    // The reason real execution beats a replay at L-B: the result is observably
    // incomplete partway through, which is what makes it useful for debugging arithmetic.
    ProgramSpec ps;
    ps.size = 32;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);
    BehavioralStepper cur(p);

    const std::vector<float> before = p.operand("C").values;
    std::size_t taken = 0;
    while (taken < cur.size() / 2 && cur.step()) ++taken;
    const std::vector<float> midway = p.operand("C").values;
    while (cur.step()) {}
    const std::vector<float> after = p.operand("C").values;

    CHECK_FALSE(bit_identical(before, midway));    // something happened
    CHECK_FALSE(bit_identical(midway, after));     // and it was not finished
}

TEST_CASE("block-sequential steps are ordered so a replay reads correctly",
          "[program][driver][step]") {
    ProgramSpec ps;
    ps.size = 48;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);
    const DeviceSpec ds;
    const auto device = make_device(ds);
    const auto out = run_at(ExecutionLevel::BlockSequential, p, device,
                            Placement::single(device.compute_tiles));
    REQUIRE_FALSE(out.timeline.empty());

    BlockSequentialStepper cur(out.timeline);
    CHECK(cur.models_time());

    // Two events per leg (start, end) plus a fire and a completion per op.
    std::size_t legs = 0;
    for (const auto& rec : out.timeline) legs += rec.hops.size();
    CHECK(cur.size() == legs * 2 + out.timeline.size() * 2);

    Cycle last_cycle = 0;
    std::map<std::size_t, std::size_t> open_legs;      // op -> legs currently open
    std::map<std::size_t, bool> fired, completed;
    std::size_t steps = 0;
    while (cur.step()) {
        const StepEvent& e = cur.current();
        CHECK(e.at >= last_cycle);                     // never goes backwards
        last_cycle = e.at;
        switch (e.kind) {
            case StepKind::OpFired:
                CHECK_FALSE(fired[e.op_index]);        // fired exactly once
                fired[e.op_index] = true;
                break;
            case StepKind::HopStarted:
                CHECK(fired[e.op_index]);              // a leg cannot precede the fire
                ++open_legs[e.op_index];
                break;
            case StepKind::HopFinished:
                CHECK(open_legs[e.op_index] > 0);      // and cannot end before it starts
                --open_legs[e.op_index];
                break;
            case StepKind::OpCompleted:
                CHECK(fired[e.op_index]);
                CHECK(open_legs[e.op_index] == 0);     // every leg closed first
                CHECK_FALSE(completed[e.op_index]);
                completed[e.op_index] = true;
                break;
            case StepKind::OpApplied:
                FAIL("L-T1 does not apply: it replays");
                break;
        }
        ++steps;
    }
    CHECK(steps == cur.size());
    for (const auto& rec : out.timeline) {
        CHECK(fired[rec.op_index]);
        CHECK(completed[rec.op_index]);
    }
}

TEST_CASE("lane occupancy is conserved across a replay", "[program][driver][step]") {
    // Every lane taken is given back, so the count returns to zero. A leak here would mean
    // the replay disagrees with the run about which process is busy -- which is the one
    // thing a stepper is for.
    ProgramSpec ps;
    ps.size = 48;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);
    DeviceSpec ds;
    ds.dma_engines = 2;
    ds.block_movers = 2;
    const auto device = make_device(ds);
    const auto out = run_at(ExecutionLevel::BlockSequential, p, device,
                            Placement::single(device.compute_tiles));

    auto lanes_of = [&](Mover m) -> Dim {
        switch (m) {
            case Mover::Dma:        return device.dma_engines;
            case Mover::BlockMover: return device.block_movers;
            case Mover::Streamer:   return device.streamers;
            case Mover::Noc:        return device.noc_links;
        }
        return 0;
    };
    BlockSequentialStepper cur(out.timeline);
    std::size_t peak_dma = 0;
    while (cur.step()) {
        // EVERY process, not just the DMA. An earlier version of this test checked only
        // Mover::Dma and therefore missed a real over-capacity report on the BlockMover:
        // the replay incremented a lane count before decrementing it, because it ordered a
        // lower-indexed op's acquire ahead of a higher-indexed op's release at the same
        // cycle. A conservation test that examines one resource is not a conservation test.
        for (const auto& kv : cur.lanes_busy()) {
            const Dim have = lanes_of(kv.first);
            if (have) CHECK(kv.second <= have);
            if (kv.first == Mover::Dma) peak_dma = std::max(peak_dma, kv.second);
        }
    }
    CHECK(peak_dma > 0);
    for (const auto& kv : cur.lanes_busy()) CHECK(kv.second == 0);
    CHECK(cur.in_flight() == 0);
}

TEST_CASE("a replay never reports more lanes busy than the device has",
          "[program][driver][step]") {
    // The default device has ONE lane per process, which is where the ordering bug showed:
    // a release and an acquire at the same cycle, across two ops. Sweeping the tight
    // configurations is the point -- generous ones hide it.
    for (const char* algo : {"matmul", "lu"}) {
        for (Dim engines : {Dim(1), Dim(2)}) {
            ProgramSpec ps;
            ps.algo = algo;
            ps.size = 48;
            ps.tile = 16;
            DeviceSpec ds;
            ds.dma_engines = engines;
            ds.block_movers = 1;          // deliberately the scarcest
            ds.streamers = 1;
            const auto device = make_device(ds);
            TileProgram p = derive(ps);
            fill(p, ps);
            const auto out = run_at(ExecutionLevel::BlockSequential, p, device,
                                    Placement::single(device.compute_tiles));
            BlockSequentialStepper cur(out.timeline);
            std::size_t peak_bm = 0;
            while (cur.step()) {
                const auto& busy = cur.lanes_busy();
                auto it = busy.find(Mover::BlockMover);
                if (it != busy.end()) {
                    CHECK(it->second <= device.block_movers);
                    peak_bm = std::max(peak_bm, it->second);
                }
                auto dma = busy.find(Mover::Dma);
                if (dma != busy.end()) CHECK(dma->second <= device.dma_engines);
                auto str = busy.find(Mover::Streamer);
                if (str != busy.end()) CHECK(str->second <= device.streamers);
            }
            CHECK(peak_bm > 0);                     // the BlockMover really was used
        }
    }
}

TEST_CASE("a level with no interpreter has no stepper either",
          "[program][driver][step]") {
    ProgramSpec ps;
    TileProgram p = derive(ps);
    fill(p, ps);
    const std::vector<TileOpRecord> empty;
    for (ExecutionLevel l : {ExecutionLevel::ResourceTransactional,
                             ExecutionLevel::CycleAccurate})
        CHECK_THROWS_AS(make_stepper(l, p, empty), std::invalid_argument);
}

TEST_CASE("a zero-work op fires before it completes, even sharing one cycle",
          "[program][driver][step]") {
    // Releases are ordered before acquires within a cycle to mirror the executor, but a
    // zero-work op has no hops and start == finish, so its fire and its completion share a
    // cycle. Ranking every completion ahead of every fire would complete it BEFORE it
    // fired and leave in_flight() stuck at 1.
    //
    // No derived program produces such an op today -- verified: zero_work_ops is 0 for
    // matmul and LU at every size tried -- so this builds the timeline by hand rather than
    // asserting against a run that cannot exercise it.
    std::vector<TileOpRecord> timeline;

    TileOpRecord instant{};             // the zero-work op, sharing one cycle
    instant.op_index = 0;
    instant.kind = TileOpKind::Feed;
    instant.start = 8;
    instant.finish = 8;
    instant.zero_work = true;
    timeline.push_back(instant);

    TileOpRecord moving{};              // a real transfer completing at the same cycle
    moving.op_index = 1;
    moving.kind = TileOpKind::Feed;
    moving.start = 0;
    moving.finish = 8;
    moving.hops.push_back(HopRecord{Hop::DmaDramToL3, 0, 0, 8});
    timeline.push_back(moving);

    BlockSequentialStepper cur(timeline);
    std::map<std::size_t, bool> fired;
    while (cur.step()) {
        const StepEvent& e = cur.current();
        if (e.kind == StepKind::OpFired) fired[e.op_index] = true;
        if (e.kind == StepKind::HopStarted || e.kind == StepKind::OpCompleted)
            CHECK(fired[e.op_index]);   // nothing happens to an op before it fires
    }
    CHECK(fired[0]);
    CHECK(fired[1]);
    CHECK(cur.in_flight() == 0);        // and nothing is left dangling
}
