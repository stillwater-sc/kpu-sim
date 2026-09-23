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
#include <sw/kpu/program/driver/timeline_trace.hpp>

#include <cstring>
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
