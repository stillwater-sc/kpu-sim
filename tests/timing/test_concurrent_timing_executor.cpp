// ============================================================================
// tests/timing/test_concurrent_timing_executor.cpp
// Unit tests for ConcurrentTimingExecutor
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/functional_mlp_executor.hpp>
#include <sw/kpu/timing/functional_domain_flow.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::isa;

// ============================================================================
// Helper functions
// ============================================================================

static TileDescriptor make_tile(MatrixID matrix, Size ti, Size tj, Size tk = 0,
                                Size size_bytes = 1024) {
    TileDescriptor desc;
    desc.tile_id = {matrix, ti, tj, tk};
    desc.dram_address = 0x10000 + ti * 0x1000 + tj * 0x100;
    desc.size_bytes = size_bytes;
    return desc;
}

static ConcurrentTimingExecutor::Config default_config() {
    ConcurrentTimingExecutor::Config config;
    config.num_memory_controllers = 1;
    config.mc_request_queue_depth = 32;
    config.mc_bandwidth_gbps = 25.6;
    config.l3_buffer_count = 8;
    config.num_block_movers = 2;
    config.l2_bank_count = 16;
    config.num_row_streamers = 1;
    config.num_col_streamers = 1;
    config.clock_ghz = 1.0;
    config.max_cycles = 10000;
    config.enable_livelock_detection = false;  // Disable for basic tests
    return config;
}

static size_t count_events(const std::vector<TimingEvent>& events, EventType type) {
    size_t count = 0;
    for (const auto& e : events) {
        if (e.type == type) count++;
    }
    return count;
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor construction", "[timing][executor]") {
    auto config = default_config();
    ConcurrentTimingExecutor executor(config);

    REQUIRE(executor.num_memory_controllers() == 1);
    REQUIRE(executor.num_block_movers() == 2);
    REQUIRE(executor.num_row_streamers() == 1);
    REQUIRE(executor.num_col_streamers() == 1);
    REQUIRE(executor.current_cycle() == 0);
    REQUIRE(executor.is_complete());  // No work scheduled
}

TEST_CASE("ConcurrentTimingExecutor configuration", "[timing][executor]") {
    ConcurrentTimingExecutor::Config config;
    config.num_memory_controllers = 2;  // Two memory controllers
    config.num_block_movers = 4;
    config.num_row_streamers = 2;
    config.num_col_streamers = 2;
    config.l3_buffer_count = 32;
    config.l2_bank_count = 64;

    ConcurrentTimingExecutor executor(config);

    REQUIRE(executor.num_memory_controllers() == 2);
    REQUIRE(executor.num_block_movers() == 4);
    REQUIRE(executor.num_row_streamers() == 2);
    REQUIRE(executor.num_col_streamers() == 2);
    REQUIRE(executor.l3_credits().available() == 32);
    REQUIRE(executor.l2_credits().available() == 64);
}

// ============================================================================
// Single Tile Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor single tile load", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);

    REQUIRE_FALSE(executor.is_complete());

    bool completed = executor.run();

    REQUIRE(completed);
    REQUIRE(executor.l3_tag_cam().lookup(tile.tile_id));

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_loaded == 1);
}

TEST_CASE("ConcurrentTimingExecutor single tile pipeline: load → move → feed", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);
    executor.schedule_move(tile);
    executor.schedule_feed(tile);

    bool completed = executor.run();

    REQUIRE(completed);

    // Verify events
    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::DMA_LOAD_COMPLETE) == 1);
    REQUIRE(count_events(events, EventType::TILE_ARRIVED_L3) == 1);
    REQUIRE(count_events(events, EventType::BM_MOVE_COMPLETE) == 1);
    REQUIRE(count_events(events, EventType::TILE_ARRIVED_L2) == 1);
    REQUIRE(count_events(events, EventType::STR_FEED_COMPLETE) == 1);
    REQUIRE(count_events(events, EventType::TILE_FED_TO_COMPUTE) == 1);

    // All credits should be returned
    REQUIRE(executor.l3_credits().available() == executor.config().l3_buffer_count);
    REQUIRE(executor.l2_credits().available() == executor.config().l2_bank_count);
}

// ============================================================================
// Multiple Tile Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor multiple tiles load", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Schedule 4 loads
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_loaded == 4);
}

TEST_CASE("ConcurrentTimingExecutor multiple tiles full pipeline", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Schedule 4 tiles through full pipeline
    std::vector<TileDescriptor> tiles;
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        tiles.push_back(tile);
        executor.schedule_load(tile);
        executor.schedule_move(tile);
        executor.schedule_feed(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::TILE_FED_TO_COMPUTE) == 4);

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_loaded == 4);
    REQUIRE(stats.tiles_moved == 4);
    REQUIRE(stats.tiles_fed == 4);
}

// ============================================================================
// Store and Writeback Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor store after load", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);

    // Load tile to L3
    executor.schedule_load(tile);
    executor.run();
    REQUIRE(executor.l3_tag_cam().lookup(tile.tile_id));

    // Store tile back to DRAM
    executor.schedule_store(tile);
    bool completed = executor.run();

    REQUIRE(completed);
    REQUIRE_FALSE(executor.l3_tag_cam().lookup(tile.tile_id));

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_stored == 1);
}

TEST_CASE("ConcurrentTimingExecutor drain and writeback", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // First, load and feed an A tile to "compute"
    auto a_tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(a_tile);
    executor.schedule_move(a_tile);
    executor.schedule_feed(a_tile);

    // Signal compute complete (depends on A tile being fed)
    auto c_tile = make_tile(MatrixID::C, 0, 0);
    executor.schedule_compute(c_tile, a_tile.tile_id);

    // Result tile drains from compute to L2, then writes back to L3
    executor.schedule_drain(c_tile);
    executor.schedule_writeback(c_tile);
    executor.schedule_store(c_tile);

    bool completed = executor.run();

    REQUIRE(completed);

    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::TILE_DRAINED) == 1);
    REQUIRE(count_events(events, EventType::BM_WRITEBACK_COMPLETE) == 1);
    REQUIRE(count_events(events, EventType::DMA_STORE_COMPLETE) == 1);
}

TEST_CASE("Functional CSP executor produces XOR MLP values under backpressure",
          "[timing][executor][functional][mlp]") {
    const std::vector<float> input = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    const std::vector<float> w1 = {
         1.0f,  1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f, -1.0f
    };
    const std::vector<float> b1 = {-0.5f, -1.5f, 0.5f, 1.5f};
    const std::vector<float> w2 = {2.0f, -6.0f, 0.0f, 0.0f};
    const std::vector<float> b2 = {0.0f};

    auto config = default_config();
    config.l3_buffer_count = 1;
    config.l2_bank_count = 1;
    config.num_block_movers = 1;
    config.compute_latency = 8;
    config.max_cycles = 100000;

    FunctionalMLPExecutor mlp(config);
    mlp.add_layer(2, 4, w1, b1, ConcurrentTimingExecutor::FunctionalActivation::RELU,
                  "hidden");
    mlp.add_layer(4, 1, w2, b2, ConcurrentTimingExecutor::FunctionalActivation::NONE,
                  "output");
    auto output = mlp.forward(input, 4);

    REQUIRE(output.size() == 4);
    REQUIRE(output[0] == Catch::Approx(0.0f));
    REQUIRE(output[1] == Catch::Approx(1.0f));
    REQUIRE(output[2] == Catch::Approx(1.0f));
    REQUIRE(output[3] == Catch::Approx(0.0f));

    REQUIRE(mlp.statistics().layers_completed == 2);
    REQUIRE(mlp.statistics().total_cycles > 0);
    REQUIRE(mlp.statistics().total_stall_cycles > 0);
    REQUIRE(count_events(mlp.events(), EventType::COMPUTE_COMPLETE) == 2);
    REQUIRE(count_events(mlp.events(), EventType::TILE_DRAINED) == 1);
    REQUIRE(count_events(mlp.events(), EventType::DMA_STORE_COMPLETE) == 1);
}

TEST_CASE("Functional payload bytes cross every hierarchy boundary only on CSP completion",
          "[timing][executor][functional][hierarchy]") {
    ConcurrentTimingExecutor executor(default_config());
    auto a = make_tile(MatrixID::A, 3, 0, 0, 4);
    a.height = 1; a.width = 1;
    executor.set_tile_payload(a.tile_id, TilePayload{1, 1, {7.0f}});
    executor.schedule_load(a);
    executor.schedule_move(a);
    executor.schedule_feed(a);
    size_t cursor = 0;
    bool saw_l3 = false, saw_l2 = false, saw_l1 = false, saw_compute = false;
    while (!executor.is_complete()) {
        executor.step();
        for (; cursor < executor.events().size(); ++cursor) {
            const auto& event = executor.events()[cursor];
            if (event.tile_id != a.tile_id) continue;
            if (event.type == EventType::TILE_ARRIVED_L3) {
                saw_l3 = executor.has_tile_payload_at(MemoryLevel::L3, a.tile_id);
            } else if (event.type == EventType::BM_MOVE_COMPLETE) {
                saw_l2 = executor.has_tile_payload_at(MemoryLevel::L2, a.tile_id);
            } else if (event.type == EventType::TILE_FED_TO_COMPUTE) {
                saw_l1 = executor.has_tile_payload_at(MemoryLevel::L1, a.tile_id);
                saw_compute = executor.has_tile_payload_at(MemoryLevel::COMPUTE, a.tile_id);
            }
        }
    }
    REQUIRE(saw_l3); REQUIRE(saw_l2); REQUIRE(saw_l1); REQUIRE(saw_compute);
    REQUIRE(executor.tile_payload_at(MemoryLevel::COMPUTE, a.tile_id).values[0] ==
            Catch::Approx(7.0f));
}

TEST_CASE("tiles_at enumerates per-level occupancy for the tile tracker",
          "[timing][executor][functional][tracker]") {
    // Issue #165: tiles_at(level) is the observer the tile-state tracker
    // needs - it returns the full resident set at a level (not just "is
    // this one tile here"), sorted for a deterministic log.
    ConcurrentTimingExecutor executor(default_config());

    // Two A tiles streaming in; check occupancy migrates DRAM -> L3 -> L2 ->
    // L1/compute as each tile advances.
    auto a0 = make_tile(MatrixID::A, 0, 0, 0, 4); a0.height = 1; a0.width = 1;
    auto a1 = make_tile(MatrixID::A, 1, 0, 0, 4); a1.height = 1; a1.width = 1;
    executor.set_tile_payload(a0.tile_id, TilePayload{1, 1, {2.0f}});
    executor.set_tile_payload(a1.tile_id, TilePayload{1, 1, {3.0f}});

    // Before running, both tiles are staged at DRAM only
    REQUIRE(executor.tiles_at(MemoryLevel::DRAM) ==
            std::vector<TileID>{a0.tile_id, a1.tile_id});   // sorted
    REQUIRE(executor.tiles_at(MemoryLevel::L3).empty());
    REQUIRE(executor.tiles_at(MemoryLevel::L2).empty());

    for (auto* a : {&a0, &a1}) {
        executor.schedule_load(*a);
        executor.schedule_move(*a);
        executor.schedule_feed(*a);
    }

    // Track that at some point each level held a tile, and the L1 arrival
    // cycle is recorded and monotonic
    bool l3_seen = false, l2_seen = false, l1_seen = false;
    Cycle last_l1_arrival = 0;
    while (!executor.is_complete()) {
        executor.step();
        if (!executor.tiles_at(MemoryLevel::L3).empty()) l3_seen = true;
        if (!executor.tiles_at(MemoryLevel::L2).empty()) l2_seen = true;
        for (const auto& id : executor.tiles_at(MemoryLevel::L1)) {
            l1_seen = true;
            const Cycle arr = executor.tile_arrival_cycle_at(MemoryLevel::L1, id);
            REQUIRE(arr >= last_l1_arrival);
            last_l1_arrival = arr;
        }
    }
    REQUIRE(l3_seen); REQUIRE(l2_seen); REQUIRE(l1_seen);

    // Both tiles reached the compute fabric with their values intact
    auto compute_tiles = executor.tiles_at(MemoryLevel::COMPUTE);
    REQUIRE(compute_tiles == std::vector<TileID>{a0.tile_id, a1.tile_id});
    REQUIRE(executor.tile_payload_at(MemoryLevel::COMPUTE, a0.tile_id).values[0] ==
            Catch::Approx(2.0f));

    // A timing-only executor (no payloads) reports empty occupancy
    ConcurrentTimingExecutor timing_only(default_config());
    REQUIRE(timing_only.tiles_at(MemoryLevel::DRAM).empty());
}

TEST_CASE("Functional Domain Flow executes an arbitrary branched DAG",
          "[timing][executor][functional][domain-flow]") {
    FunctionalDomainFlowExecutor runner(default_config());
    auto a = make_tile(MatrixID::A, 7, 0, 0, 4); a.height = 1; a.width = 1;
    auto b = make_tile(MatrixID::B, 7, 0, 0, 4); b.height = 1; b.width = 1;
    auto c = make_tile(MatrixID::C, 7, 0, 0, 4); c.height = 1; c.width = 1;
    auto d = make_tile(MatrixID::C, 8, 0, 0, 4); d.height = 1; d.width = 1;
    runner.set_tile_payload(a.tile_id, TilePayload{1, 1, {2.0f}});
    runner.set_tile_payload(b.tile_id, TilePayload{1, 1, {4.0f}});

    using Program = FunctionalDomainFlowProgram;
    Program program;
    const size_t load_a = program.add({Program::Operation::LOAD, a, {}, {}});
    const size_t load_b = program.add({Program::Operation::LOAD, b, {}, {}});
    const size_t move_a = program.add({Program::Operation::MOVE, a, {}, {load_a}});
    const size_t move_b = program.add({Program::Operation::MOVE, b, {}, {load_b}});
    const size_t feed_a = program.add({Program::Operation::FEED, a, {}, {move_a}});
    const size_t feed_b = program.add({Program::Operation::FEED, b, {}, {move_b}});
    ConcurrentTimingExecutor::MatMulComputeSpec matmul;
    matmul.a_tiles = {a.tile_id}; matmul.b_tiles = {b.tile_id};
    const size_t compute = program.add({Program::Operation::MATMUL, c, matmul,
                                        {feed_a, feed_b}});
    Program::Node custom;
    custom.operation = Program::Operation::COMPUTE;
    custom.tile = d;
    custom.predecessors = {compute};
    custom.compute.input_tiles = {c.tile_id};
    custom.compute.resident_tiles = {c.tile_id};
    custom.compute.operation = [](const std::vector<TilePayload>& inputs) {
        TilePayload result = inputs.at(0);
        for (float& value : result.values) value += 1.0f;
        return result;
    };
    const size_t transformed = program.add(std::move(custom));
    const size_t drain = program.add({Program::Operation::DRAIN, d, {}, {transformed}});
    const size_t writeback = program.add({Program::Operation::WRITEBACK, d, {}, {drain}});
    program.add({Program::Operation::STORE, d, {}, {writeback}});

    REQUIRE(runner.run(program));
    REQUIRE(runner.executor().tile_payload_at(MemoryLevel::DRAM, d.tile_id).values[0] ==
            Catch::Approx(9.0f));
    REQUIRE(count_events(runner.executor().events(), EventType::DMA_LOAD_COMPLETE) == 2);
}

TEST_CASE("Chained accumulator reduction runs on existing resident-dep machinery",
          "[timing][executor][functional][reduction]") {
    // E3-T2 validation (issue #105): a streaming reduction is a chain of
    // functional computes whose target IS the accumulator tile, each
    // (except the first) taking the accumulator as a resident dependency.
    // The #66 per-instance resident accounting enforces the chain order:
    // COMPUTE_k requires k completed computes of the accumulator, so it
    // cannot fire before COMPUTE_{k-1}. The first compute must NOT list the
    // accumulator (required=max(1,0)=1 would deadlock) - init on first touch.
    FunctionalDomainFlowExecutor runner(default_config());

    const std::vector<float> values = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    std::vector<TileDescriptor> x;
    for (Size k = 0; k < values.size(); ++k) {
        auto tile = make_tile(MatrixID::A, k, 0, 0, 4);
        tile.height = 1; tile.width = 1;
        runner.set_tile_payload(tile.tile_id, TilePayload{1, 1, {values[k]}});
        x.push_back(tile);
    }
    auto acc = make_tile(MatrixID::C, 0, 0, 0, 4);
    acc.height = 1; acc.width = 1;

    using Program = FunctionalDomainFlowProgram;
    Program program;
    size_t prev_compute = 0;
    for (size_t k = 0; k < x.size(); ++k) {
        const size_t load = program.add({Program::Operation::LOAD, x[k], {}, {}});
        const size_t move = program.add({Program::Operation::MOVE, x[k], {}, {load}});
        const size_t feed = program.add({Program::Operation::FEED, x[k], {}, {move}});

        Program::Node compute;
        compute.operation = Program::Operation::COMPUTE;
        compute.tile = acc;
        if (k == 0) {
            // Init on first touch: input is X_0 only, no resident accumulator
            compute.predecessors = {feed};
            compute.compute.input_tiles = {x[k].tile_id};
            compute.compute.operation = [](const std::vector<TilePayload>& in) {
                return in.at(0);  // seed the accumulator
            };
        } else {
            // Combine: accumulator (resident) + this tile
            compute.predecessors = {feed, prev_compute};
            compute.compute.input_tiles = {x[k].tile_id, acc.tile_id};
            compute.compute.resident_tiles = {acc.tile_id};
            compute.compute.operation = [](const std::vector<TilePayload>& in) {
                TilePayload out = in.at(0);
                out.values[0] += in.at(1).values[0];  // running sum
                return out;
            };
        }
        prev_compute = program.add(std::move(compute));
    }
    const size_t drain = program.add({Program::Operation::DRAIN, acc, {}, {prev_compute}});
    const size_t wb = program.add({Program::Operation::WRITEBACK, acc, {}, {drain}});
    program.add({Program::Operation::STORE, acc, {}, {wb}});

    REQUIRE(runner.run(program));
    REQUIRE(runner.executor().tile_payload_at(MemoryLevel::DRAM, acc.tile_id).values[0] ==
            Catch::Approx(30.0f));  // 2+4+6+8+10
}

TEST_CASE("Repeated tile feeds cannot satisfy a later functional compute early",
          "[timing][executor][functional][dependency]") {
    auto config = default_config();
    config.compute_latency = 4;
    ConcurrentTimingExecutor executor(config);

    auto a = make_tile(MatrixID::A, 0, 0, 0, 4);
    a.height = 1; a.width = 1;
    auto b = make_tile(MatrixID::B, 0, 0, 0, 4);
    b.height = 1; b.width = 1;
    auto c0 = make_tile(MatrixID::C, 0, 0, 0, 4);
    c0.height = 1; c0.width = 1;
    auto c1 = make_tile(MatrixID::C, 0, 1, 0, 4);
    c1.height = 1; c1.width = 1;

    executor.set_tile_payload(a.tile_id, TilePayload{1, 1, {2.0f}});
    executor.set_tile_payload(b.tile_id, TilePayload{1, 1, {3.0f}});

    ConcurrentTimingExecutor::MatMulComputeSpec compute;
    compute.a_tiles = {a.tile_id};
    compute.b_tiles = {b.tile_id};

    for (auto* c : {&c0, &c1}) {
        executor.schedule_load(a);
        executor.schedule_load(b);
        executor.schedule_move(a);
        executor.schedule_move(b);
        executor.schedule_feed(a);
        executor.schedule_feed(b);
        executor.schedule_matmul_compute(*c, compute);
        executor.schedule_drain(*c);
        executor.schedule_writeback(*c);
        executor.schedule_store(*c);
    }

    REQUIRE(executor.run());
    REQUIRE(executor.tile_payload(c0.tile_id).values[0] == Catch::Approx(6.0f));
    REQUIRE(executor.tile_payload(c1.tile_id).values[0] == Catch::Approx(6.0f));

    Cycle second_b_feed = 0;
    Cycle second_compute_start = 0;
    size_t b_feeds = 0;
    for (const auto& event : executor.events()) {
        if (event.type == EventType::TILE_FED_TO_COMPUTE && event.tile_id == b.tile_id) {
            if (++b_feeds == 2) second_b_feed = event.cycle;
        }
        if (event.type == EventType::COMPUTE_START && event.tile_id == c1.tile_id) {
            second_compute_start = event.cycle;
        }
    }
    REQUIRE(b_feeds == 2);
    REQUIRE(second_compute_start >= second_b_feed);
}

// ============================================================================
// Work Distribution Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor distributes work across memory controllers", "[timing][executor]") {
    auto config = default_config();
    config.num_memory_controllers = 2;
    ConcurrentTimingExecutor executor(config);

    // Schedule 8 loads - should distribute across memory controllers
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_loaded == 8);
}

TEST_CASE("ConcurrentTimingExecutor distributes work across BlockMovers", "[timing][executor]") {
    auto config = default_config();
    config.num_block_movers = 4;
    ConcurrentTimingExecutor executor(config);

    // Pre-load tiles to L3
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
    }
    executor.run();

    // Schedule moves - should distribute across movers
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_move(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_moved == 8);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor reset", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Run some work
    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);
    executor.run();

    REQUIRE(executor.current_cycle() > 0);
    REQUIRE_FALSE(executor.events().empty());

    // Reset
    executor.reset();

    REQUIRE(executor.current_cycle() == 0);
    REQUIRE(executor.events().empty());
    REQUIRE(executor.is_complete());
    REQUIRE(executor.l3_credits().available() == executor.config().l3_buffer_count);
    REQUIRE(executor.l2_credits().available() == executor.config().l2_bank_count);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor statistics", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Full pipeline for 4 tiles
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
        executor.schedule_move(tile);
        executor.schedule_feed(tile);
    }

    executor.run();

    auto stats = executor.get_statistics();

    REQUIRE(stats.total_cycles > 0);
    REQUIRE(stats.tiles_loaded == 4);
    REQUIRE(stats.tiles_moved == 4);
    REQUIRE(stats.tiles_fed == 4);
    REQUIRE(stats.bytes_loaded == 4 * 1024);  // 4 tiles × 1024 bytes
}

TEST_CASE("ConcurrentTimingExecutor utilization", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
        executor.schedule_move(tile);
        executor.schedule_feed(tile);
    }

    executor.run();

    auto stats = executor.get_statistics();

    // Utilization should be between 0 and 1
    REQUIRE(stats.dma_utilization() >= 0.0);
    REQUIRE(stats.dma_utilization() <= 1.0);
    REQUIRE(stats.bm_utilization() >= 0.0);
    REQUIRE(stats.bm_utilization() <= 1.0);
    REQUIRE(stats.str_utilization() >= 0.0);
    REQUIRE(stats.str_utilization() <= 1.0);
}

// ============================================================================
// Step-by-Step Execution Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor step execution", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);

    // Step through manually
    REQUIRE_FALSE(executor.is_complete());

    Cycle cycle = 0;
    while (!executor.is_complete() && cycle < 1000) {
        executor.step();
        cycle++;
    }

    REQUIRE(executor.is_complete());
    REQUIRE(executor.current_cycle() == cycle);
}

// ============================================================================
// Export Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor Chrome trace export", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);
    executor.schedule_move(tile);
    executor.run();

    // Export to temp file
    std::string temp_file = (std::filesystem::temp_directory_path() / "test_trace.json").string();
    executor.export_chrome_trace(temp_file);

    // Verify file exists and has content
    std::ifstream file(temp_file);
    REQUIRE(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    REQUIRE(content.find("traceEvents") != std::string::npos);
    REQUIRE(content.find("DMA_LOAD_START") != std::string::npos);
}

TEST_CASE("ConcurrentTimingExecutor CSV export", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);
    executor.run();

    // Export to temp file
    std::string temp_file = (std::filesystem::temp_directory_path() / "test_trace.csv").string();
    executor.export_csv(temp_file);

    // Verify file exists and has content
    std::ifstream file(temp_file);
    REQUIRE(file.is_open());

    std::string line;
    std::getline(file, line);  // Header
    REQUIRE(line.find("cycle") != std::string::npos);
    REQUIRE(line.find("type") != std::string::npos);
}

// ============================================================================
// Credit Flow Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor limited credits", "[timing][executor]") {
    auto config = default_config();
    config.l3_buffer_count = 2;  // Very limited L3
    config.l2_bank_count = 2;    // Very limited L2
    ConcurrentTimingExecutor executor(config);

    // Schedule more tiles than buffer capacity
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
        executor.schedule_move(tile);
        executor.schedule_feed(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::TILE_FED_TO_COMPUTE) == 8);
}

// ============================================================================
// A and B Matrix Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor A and B matrices", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Load and feed both A (row) and B (column) tiles
    auto a_tile = make_tile(MatrixID::A, 0, 0);
    auto b_tile = make_tile(MatrixID::B, 0, 0);

    executor.schedule_load(a_tile);
    executor.schedule_load(b_tile);
    executor.schedule_move(a_tile);
    executor.schedule_move(b_tile);
    executor.schedule_feed(a_tile);  // Goes to row streamer
    executor.schedule_feed(b_tile);  // Goes to column streamer

    bool completed = executor.run();

    REQUIRE(completed);

    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::TILE_FED_TO_COMPUTE) == 2);
}

// ============================================================================
// Matmul-like Pattern Test
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor matmul pattern", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Simulate a 2x2x2 tiled matmul
    // C[ti][tj] += A[ti][tk] * B[tk][tj]

    constexpr Size Ti = 2, Tj = 2, Tk = 2;

    // Schedule all loads first
    for (Size ti = 0; ti < Ti; ++ti) {
        for (Size tk = 0; tk < Tk; ++tk) {
            auto a_tile = make_tile(MatrixID::A, ti, 0, tk);
            executor.schedule_load(a_tile);
            executor.schedule_move(a_tile);
            executor.schedule_feed(a_tile);
        }
    }

    for (Size tk = 0; tk < Tk; ++tk) {
        for (Size tj = 0; tj < Tj; ++tj) {
            auto b_tile = make_tile(MatrixID::B, 0, tj, tk);
            executor.schedule_load(b_tile);
            executor.schedule_move(b_tile);
            executor.schedule_feed(b_tile);
        }
    }

    // Schedule result drains (with COMPUTE dependency)
    for (Size ti = 0; ti < Ti; ++ti) {
        for (Size tj = 0; tj < Tj; ++tj) {
            auto c_tile = make_tile(MatrixID::C, ti, tj);
            // COMPUTE depends on last B tile for this column: B[0,tj,Tk-1]
            TileID last_b;
            last_b.matrix = MatrixID::B;
            last_b.ti = 0;
            last_b.tj = tj;
            last_b.tk = Tk - 1;
            executor.schedule_compute(c_tile, last_b);
            executor.schedule_drain(c_tile);
            executor.schedule_writeback(c_tile);
            executor.schedule_store(c_tile);
        }
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto stats = executor.get_statistics();
    // tiles_loaded counts TILE_ARRIVED_L3 events, which includes:
    // - DMA loads (A + B tiles)
    // - BlockMover writebacks (C tiles arriving at L3 from L2)
    REQUIRE(stats.tiles_loaded == Ti * Tk + Tk * Tj + Ti * Tj);  // A + B + C writebacks
    REQUIRE(stats.tiles_fed == Ti * Tk + Tk * Tj);     // All A/B tiles fed
    REQUIRE(stats.tiles_drained == Ti * Tj);           // C tiles drained
    REQUIRE(stats.tiles_stored == Ti * Tj);            // C tiles stored
}
