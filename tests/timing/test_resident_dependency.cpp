// ============================================================================
// tests/timing/test_resident_dependency.cpp
// Schedule-tier compute-resident dependency mechanism (issue #155, epic E8).
//
// A COMPUTE carrying resident_tiles consumes a tile produced by a PRIOR
// compute directly from the compute fabric - no drain/reload, no DRAM
// round-trip race. This is the capability online softmax / norm apply
// phases need. Verified two ways: the functional path proves the resident
// value reaches the consumer (and therefore that ordering held), the
// timing-only path proves the schedule executes.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {

TileDescriptor scalar_tile(MatrixID matrix, Size ti, Address base) {
    TileDescriptor t;
    t.tile_id = {matrix, ti, 0, 0};
    t.height = 1; t.width = 1; t.element_size = 4; t.size_bytes = 4;
    t.dram_address = base + ti * 0x1000;
    t.matrix_base_address = base;
    return t;
}

// A two-compute schedule: a stats compute produces B = 2*A from a fed A,
// then an apply compute consumes B RESIDENT and produces C = B + 1.
ScheduleResult build_schedule() {
    auto a = scalar_tile(MatrixID::A, 0, 0x100000);
    auto b = scalar_tile(MatrixID::B, 0, 0x200000);
    auto c = scalar_tile(MatrixID::C, 0, 0x300000);

    ScheduleResult s;
    s.operations.push_back(ScheduleOperation::load(a));
    s.operations.push_back(ScheduleOperation::move(a));
    s.operations.push_back(ScheduleOperation::feed(a));
    s.operations.push_back(ScheduleOperation::compute(b, {a.tile_id}));  // stats
    // apply: no fresh feed, B consumed resident
    s.operations.push_back(ScheduleOperation::compute(
        c, /*fed*/ std::vector<TileID>{}, /*resident*/ std::vector<TileID>{b.tile_id}));
    s.operations.push_back(ScheduleOperation::drain(c));
    s.operations.push_back(ScheduleOperation::writeback(c));
    s.operations.push_back(ScheduleOperation::store(c));
    s.metadata.l3_buffer_count = 0;  // exempt from envelope check
    s.metadata.l2_bank_count = 0;
    s.valid = true;
    return s;
}

} // namespace

TEST_CASE("Resident dependency delivers a prior compute's value to the consumer",
          "[timing][executor][resident]") {
    auto schedule = build_schedule();

    ConcurrentTimingExecutor::Config cfg; cfg.max_cycles = 1000000;
    ConcurrentTimingExecutor executor(cfg);
    executor.set_tile_payload({MatrixID::A, 0, 0, 0}, TilePayload{1, 1, {5.0f}});

    ScheduleExecutor sched_exec(executor);
    sched_exec.set_functional_compute_binder(
        [](const ScheduleOperation& op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            // Fed inputs first, then resident inputs (both must be listed in
            // input_tiles; resident_tiles marks which are compute-resident)
            spec.input_tiles = op.dependency_tiles;
            for (const auto& r : op.resident_tiles) spec.input_tiles.push_back(r);
            spec.resident_tiles = op.resident_tiles;
            if (op.tile.tile_id.matrix == MatrixID::B) {
                spec.operation = [](const std::vector<TilePayload>& in) {
                    TilePayload out = in.at(0);            // stats: B = 2*A
                    for (float& v : out.values) v *= 2.0f;
                    return out;
                };
            } else {
                spec.operation = [](const std::vector<TilePayload>& in) {
                    TilePayload out = in.at(0);            // apply: C = B + 1
                    for (float& v : out.values) v += 1.0f;
                    return out;
                };
            }
            return spec;
        });

    auto result = sched_exec.execute(schedule);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);

    // C = 2*A + 1 = 11. If the apply compute had run before the stats
    // compute completed, B's resident payload would have been absent and
    // the functional op would have thrown - so a correct value here proves
    // the resident dependency ordered the chain.
    REQUIRE(executor.tile_payload_at(MemoryLevel::DRAM, {MatrixID::C, 0, 0, 0})
                .values[0] == Catch::Approx(11.0f));
}

TEST_CASE("Resident-dependency schedule executes on the timing-only path",
          "[timing][executor][resident]") {
    auto schedule = build_schedule();
    ConcurrentTimingExecutor::Config cfg; cfg.max_cycles = 1000000;
    ConcurrentTimingExecutor executor(cfg);
    ScheduleExecutor sched_exec(executor);   // no functional binder
    auto result = sched_exec.execute(schedule);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
}
