// ============================================================================
// tests/harness/test_pipeline_harness.cpp
// Unit tests for full data movement pipeline harness
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/harness/pipeline_harness.hpp>

using namespace sw::kpu::harness;
using Catch::Approx;

TEST_CASE("Pipeline Harness construction", "[harness][pipeline]") {
    SECTION("Default configuration") {
        PipelineHarnessConfig config;
        DataMovementPipelineHarness harness(config);

        REQUIRE(harness.current_cycle() == 0);
        REQUIRE_FALSE(harness.has_pending());
    }

    SECTION("Custom configuration") {
        PipelineHarnessConfig config;
        config.dma_config.num_dma_engines = 4;
        config.block_mover_config.l2_banks = 32;
        config.streamer_config.systolic_size = 32;

        DataMovementPipelineHarness harness(config);
        REQUIRE(harness.dma_harness().credits_available() > 0);
    }
}

TEST_CASE("Pipeline Harness DRAM initialization", "[harness][pipeline]") {
    PipelineHarnessConfig config;
    DataMovementPipelineHarness harness(config);

    SECTION("Initialize with pattern") {
        harness.init_dram_pattern(0, 64, 64, 4);  // 64x64 FP32 matrix

        auto data = harness.read_dram(0);
        REQUIRE(data.size() == 64 * 64 * 4);
    }

    SECTION("Initialize with raw data") {
        std::vector<float> test_data(16 * 16, 1.0f);
        harness.init_dram(0, test_data.data(), 16, 16, sizeof(float));

        auto data = harness.read_dram(0);
        REQUIRE(data.size() == 16 * 16 * sizeof(float));
    }
}

TEST_CASE("Pipeline Harness simple schedule execution", "[harness][pipeline]") {
    PipelineHarnessConfig config;
    DataMovementPipelineHarness harness(config);

    harness.init_dram_pattern(0, 32, 32, 2);

    SECTION("Single tile DMA load") {
        PipelineSchedule schedule;
        TileID tile{0, 0, 0};
        schedule.add(ScheduleOperation::dma_load(tile, 0));

        harness.load_schedule(schedule);
        bool completed = harness.run();

        REQUIRE(completed);
        REQUIRE(harness.dma_harness().dma_stats().tiles_loaded >= 1);
    }

    SECTION("DMA load then BlockMover") {
        PipelineSchedule schedule;
        TileID tile{0, 0, 0};

        schedule.add(ScheduleOperation::dma_load(tile, 0));

        auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
        bm_op.dependencies.push_back(tile);
        schedule.add(bm_op);

        harness.load_schedule(schedule);
        bool completed = harness.run();

        REQUIRE(completed);
    }
}

TEST_CASE("Pipeline Harness multi-tile schedule", "[harness][pipeline]") {
    PipelineHarnessConfig config;
    config.dma_config.l3_buffer_count = 8;
    config.block_mover_config.l2_banks = 16;
    DataMovementPipelineHarness harness(config);

    harness.init_dram_pattern(0, 64, 64, 2);

    PipelineSchedule schedule;

    // Load 4 tiles through pipeline
    for (uint32_t i = 0; i < 4; ++i) {
        TileID tile{0, i / 2, i % 2};

        // DMA load
        schedule.add(ScheduleOperation::dma_load(tile, i * 512));

        // BlockMover
        auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
        bm_op.dependencies.push_back(tile);
        schedule.add(bm_op);

        // Stream
        auto str_op = ScheduleOperation::stream_rows(tile, UINT32_MAX);
        str_op.dependencies.push_back(tile);
        schedule.add(str_op);
    }

    harness.load_schedule(schedule);
    bool completed = harness.run();

    REQUIRE(completed);
    REQUIRE(harness.current_operation_index() == schedule.size());
}

TEST_CASE("Pipeline Harness journey tracking", "[harness][pipeline]") {
    PipelineHarnessConfig config;
    DataMovementPipelineHarness harness(config);

    harness.init_dram_pattern(0, 32, 32, 2);

    PipelineSchedule schedule;
    TileID tile{0, 0, 0};

    schedule.add(ScheduleOperation::dma_load(tile, 0));
    auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
    bm_op.dependencies.push_back(tile);
    schedule.add(bm_op);
    auto str_op = ScheduleOperation::stream_rows(tile, UINT32_MAX);
    str_op.dependencies.push_back(tile);
    schedule.add(str_op);

    harness.load_schedule(schedule);
    harness.run();

    SECTION("Journey exists") {
        auto journey_opt = harness.get_tile_journey(tile);
        REQUIRE(journey_opt.has_value());
    }

    SECTION("Journey statistics calculated") {
        auto stats = harness.get_journey_stats();
        REQUIRE(stats.total_tiles >= 1);
    }
}

TEST_CASE("Pipeline Harness statistics", "[harness][pipeline]") {
    PipelineHarnessConfig config;
    config.peak_gflops = 1000.0;
    DataMovementPipelineHarness harness(config);

    harness.init_dram_pattern(0, 32, 32, 2);

    PipelineSchedule schedule;
    for (uint32_t i = 0; i < 2; ++i) {
        TileID tile{0, 0, i};
        schedule.add(ScheduleOperation::dma_load(tile, i * 512));
        auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
        bm_op.dependencies.push_back(tile);
        schedule.add(bm_op);
    }

    harness.load_schedule(schedule);
    harness.run();

    auto stats = harness.get_pipeline_stats();

    REQUIRE(stats.dma_stats.tiles_loaded >= 2);
    REQUIRE(stats.block_mover_stats.tiles_moved >= 2);
}

TEST_CASE("Pipeline Harness validation", "[harness][pipeline]") {
    PipelineHarnessConfig config;
    DataMovementPipelineHarness harness(config);

    SECTION("Empty harness validates") {
        auto result = harness.validate();
        REQUIRE(result.passed);
    }

    SECTION("Completed schedule validates") {
        harness.init_dram_pattern(0, 32, 32, 2);

        PipelineSchedule schedule;
        TileID tile{0, 0, 0};
        schedule.add(ScheduleOperation::dma_load(tile, 0));

        harness.load_schedule(schedule);
        harness.run();

        auto result = harness.validate();
        REQUIRE(result.passed);
    }
}

TEST_CASE("Pipeline Harness reset", "[harness][pipeline]") {
    PipelineHarnessConfig config;
    DataMovementPipelineHarness harness(config);

    harness.init_dram_pattern(0, 32, 32, 2);

    PipelineSchedule schedule;
    TileID tile{0, 0, 0};
    schedule.add(ScheduleOperation::dma_load(tile, 0));

    harness.load_schedule(schedule);
    harness.run();

    REQUIRE(harness.current_cycle() > 0);

    harness.reset();

    REQUIRE(harness.current_cycle() == 0);
    REQUIRE(harness.current_operation_index() == 0);
    REQUIRE_FALSE(harness.has_pending());
}
