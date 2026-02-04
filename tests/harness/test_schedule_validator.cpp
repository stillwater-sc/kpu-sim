// ============================================================================
// tests/harness/test_schedule_validator.cpp
// Unit tests for schedule validation infrastructure
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/harness/schedule_validator.hpp>

using namespace sw::kpu::harness;
using Catch::Approx;

TEST_CASE("ScheduleValidationResult construction", "[harness][validator]") {
    SECTION("Default construction passes") {
        ScheduleValidationResult result;
        REQUIRE(result.passed);
        REQUIRE(result.error_count() == 0);
        REQUIRE(result.warning_count() == 0);
    }

    SECTION("Adding errors marks as failed") {
        ScheduleValidationResult result;
        result.add_resource_error("Test error");
        REQUIRE_FALSE(result.passed);
        REQUIRE(result.error_count() == 1);
    }

    SECTION("Adding warnings doesn't mark as failed") {
        ScheduleValidationResult result;
        result.add_efficiency_warning("Test warning");
        REQUIRE(result.passed);
        REQUIRE(result.warning_count() == 1);
    }

    SECTION("Error counts accumulate") {
        ScheduleValidationResult result;
        result.add_resource_error("Resource error");
        result.add_dependency_error("Dependency error");
        result.add_ordering_error("Ordering error");
        result.add_invariant_error("Invariant error");

        REQUIRE(result.error_count() == 4);
        REQUIRE(result.resource_errors.size() == 1);
        REQUIRE(result.dependency_errors.size() == 1);
        REQUIRE(result.ordering_errors.size() == 1);
        REQUIRE(result.invariant_errors.size() == 1);
    }
}

TEST_CASE("ScheduleValidationResult summary", "[harness][validator]") {
    SECTION("Passed summary") {
        ScheduleValidationResult result;
        auto summary = result.summary();
        REQUIRE(summary.find("PASSED") != std::string::npos);
    }

    SECTION("Failed summary") {
        ScheduleValidationResult result;
        result.add_resource_error("Error");
        auto summary = result.summary();
        REQUIRE(summary.find("FAILED") != std::string::npos);
        REQUIRE(summary.find("1 errors") != std::string::npos);
    }
}

TEST_CASE("ScheduleValidator construction", "[harness][validator]") {
    SECTION("Default configuration") {
        ScheduleValidator validator;
        // Just verify it constructs
        REQUIRE(true);
    }

    SECTION("Custom configuration") {
        ScheduleValidator::Config config;
        config.l3_buffer_count = 16;
        config.l2_bank_count = 32;
        config.l1_depth = 8;
        config.strict_ordering = false;

        ScheduleValidator validator(config);
        REQUIRE(true);
    }
}

TEST_CASE("ScheduleValidator resource requirements", "[harness][validator]") {
    ScheduleValidator::Config config;
    config.l3_buffer_count = 4;
    config.l2_bank_count = 8;
    ScheduleValidator validator(config);

    SECTION("Empty schedule validates") {
        PipelineSchedule schedule;
        auto result = validator.check_resource_requirements(schedule);
        REQUIRE(result.passed);
    }

    SECTION("Within resource limits validates") {
        PipelineSchedule schedule;
        for (uint32_t i = 0; i < 4; ++i) {
            TileID tile{0, 0, i};
            schedule.add(ScheduleOperation::dma_load(tile, i * 512));

            auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
            bm_op.dependencies.push_back(tile);
            schedule.add(bm_op);
        }

        auto result = validator.check_resource_requirements(schedule);
        REQUIRE(result.passed);
    }

    SECTION("Exceeding L3 buffers fails") {
        PipelineSchedule schedule;
        // Load 5 tiles without releasing any (exceeds 4 buffer limit)
        for (uint32_t i = 0; i < 5; ++i) {
            TileID tile{0, 0, i};
            schedule.add(ScheduleOperation::dma_load(tile, i * 512));
        }

        auto result = validator.check_resource_requirements(schedule);
        REQUIRE_FALSE(result.passed);
        REQUIRE(result.resource_errors.size() >= 1);
    }
}

TEST_CASE("ScheduleValidator dependency checking", "[harness][validator]") {
    ScheduleValidator validator;

    SECTION("Empty schedule validates") {
        PipelineSchedule schedule;
        auto result = validator.check_dependencies(schedule);
        REQUIRE(result.passed);
    }

    SECTION("Valid dependency chain validates") {
        PipelineSchedule schedule;
        TileID tile{0, 0, 0};

        schedule.add(ScheduleOperation::dma_load(tile, 0));

        auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
        bm_op.dependencies.push_back(tile);
        schedule.add(bm_op);

        auto result = validator.check_dependencies(schedule);
        REQUIRE(result.passed);
    }

    SECTION("Missing dependency fails") {
        PipelineSchedule schedule;
        TileID tile1{0, 0, 0};
        TileID tile2{0, 0, 1};

        // Add operation that depends on tile2, but tile2 is never added
        auto op = ScheduleOperation::dma_load(tile1, 0);
        op.dependencies.push_back(tile2);
        schedule.add(op);

        auto result = validator.check_dependencies(schedule);
        REQUIRE_FALSE(result.passed);
        REQUIRE(result.dependency_errors.size() >= 1);
    }
}

TEST_CASE("ScheduleValidator ordering checking", "[harness][validator]") {
    ScheduleValidator::Config config;
    config.strict_ordering = true;
    ScheduleValidator validator(config);

    SECTION("Correct ordering validates") {
        PipelineSchedule schedule;
        TileID tile{0, 0, 0};

        // DMA load first
        schedule.add(ScheduleOperation::dma_load(tile, 0));
        // Then BlockMover
        schedule.add(ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX));
        // Then Stream
        schedule.add(ScheduleOperation::stream_rows(tile, UINT32_MAX));

        auto result = validator.check_ordering(schedule);
        REQUIRE(result.passed);
    }

    SECTION("BlockMover before DMA fails") {
        PipelineSchedule schedule;
        TileID tile{0, 0, 0};

        // BlockMover first (wrong order)
        schedule.add(ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX));
        // Then DMA load
        schedule.add(ScheduleOperation::dma_load(tile, 0));

        auto result = validator.check_ordering(schedule);
        REQUIRE_FALSE(result.passed);
        REQUIRE(result.ordering_errors.size() >= 1);
    }

    SECTION("Streamer before BlockMover fails") {
        PipelineSchedule schedule;
        TileID tile{0, 0, 0};

        // DMA load first (correct)
        schedule.add(ScheduleOperation::dma_load(tile, 0));
        // Streamer before BlockMover (wrong)
        schedule.add(ScheduleOperation::stream_rows(tile, UINT32_MAX));
        // BlockMover last
        schedule.add(ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX));

        auto result = validator.check_ordering(schedule);
        REQUIRE_FALSE(result.passed);
        REQUIRE(result.ordering_errors.size() >= 1);
    }
}

TEST_CASE("ScheduleValidator full validation", "[harness][validator]") {
    ScheduleValidator::Config config;
    config.l3_buffer_count = 8;
    config.l2_bank_count = 16;
    config.strict_ordering = true;
    ScheduleValidator validator(config);

    SECTION("Valid schedule passes all checks") {
        PipelineSchedule schedule;
        for (uint32_t i = 0; i < 4; ++i) {
            TileID tile{0, 0, i};

            schedule.add(ScheduleOperation::dma_load(tile, i * 512));

            auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
            bm_op.dependencies.push_back(tile);
            schedule.add(bm_op);

            auto str_op = ScheduleOperation::stream_rows(tile, UINT32_MAX);
            str_op.dependencies.push_back(tile);
            schedule.add(str_op);
        }

        auto result = validator.validate_schedule(schedule);
        REQUIRE(result.passed);
    }

    SECTION("Invalid schedule accumulates errors") {
        PipelineSchedule schedule;
        TileID tile{0, 0, 0};
        TileID missing_dep{99, 99, 99};

        // BlockMover before DMA (ordering error)
        auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
        bm_op.dependencies.push_back(missing_dep);  // Missing dependency
        schedule.add(bm_op);

        auto result = validator.validate_schedule(schedule);
        REQUIRE_FALSE(result.passed);
        REQUIRE(result.error_count() >= 1);
    }
}

TEST_CASE("ScheduleValidator credit invariant", "[harness][validator]") {
    ScheduleValidator validator;

    SECTION("Positive credits valid") {
        REQUIRE(validator.check_credit_invariant(HarnessComponent::L3_BUFFER, 5));
    }

    SECTION("Zero credits valid") {
        REQUIRE(validator.check_credit_invariant(HarnessComponent::L3_BUFFER, 0));
    }

    SECTION("Negative credits invalid") {
        REQUIRE_FALSE(validator.check_credit_invariant(HarnessComponent::L3_BUFFER, -1));
    }
}

TEST_CASE("ScheduleValidator ordering invariant", "[harness][validator]") {
    ScheduleValidator validator;
    TileID tile{0, 0, 0};

    SECTION("L3 before L2 valid") {
        std::set<HarnessComponent> visited{HarnessComponent::L3_BUFFER};
        REQUIRE(validator.check_ordering_invariant(tile, HarnessComponent::L2_BANK, visited));
    }

    SECTION("L2 before L3 invalid") {
        std::set<HarnessComponent> visited{HarnessComponent::L2_BANK};
        REQUIRE_FALSE(validator.check_ordering_invariant(tile, HarnessComponent::L3_BUFFER, visited));
    }

    SECTION("Empty visited set valid") {
        std::set<HarnessComponent> visited;
        REQUIRE(validator.check_ordering_invariant(tile, HarnessComponent::L3_BUFFER, visited));
    }
}

TEST_CASE("ScheduleValidator data integrity", "[harness][validator]") {
    ScheduleValidator validator;
    TileID tile{0, 0, 0};

    SECTION("Matching data passes") {
        std::vector<float> expected{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> actual{1.0f, 2.0f, 3.0f, 4.0f};

        REQUIRE(validator.check_data_integrity(
            tile, expected.data(), actual.data(),
            expected.size() * sizeof(float)));
    }

    SECTION("Different data fails") {
        std::vector<float> expected{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> actual{1.0f, 2.0f, 100.0f, 4.0f};

        REQUIRE_FALSE(validator.check_data_integrity(
            tile, expected.data(), actual.data(),
            expected.size() * sizeof(float)));
    }

    SECTION("Within tolerance passes") {
        std::vector<float> expected{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> actual{1.000001f, 2.000001f, 3.000001f, 4.000001f};

        REQUIRE(validator.check_data_integrity(
            tile, expected.data(), actual.data(),
            expected.size() * sizeof(float), 1e-5));
    }
}

TEST_CASE("ScheduleValidator output validation", "[harness][validator]") {
    ScheduleValidator validator;

    SECTION("Matching output validates") {
        std::vector<float> expected{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> actual{1.0f, 2.0f, 3.0f, 4.0f};

        auto result = validator.validate_output(
            actual.data(), expected.data(),
            expected.size() * sizeof(float));
        REQUIRE(result.passed);
    }

    SECTION("Mismatched output fails") {
        std::vector<float> expected{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> actual{1.0f, 200.0f, 3.0f, 4.0f};

        auto result = validator.validate_output(
            actual.data(), expected.data(),
            expected.size() * sizeof(float));
        REQUIRE_FALSE(result.passed);
    }
}

TEST_CASE("ScheduleValidator journey validation", "[harness][validator]") {
    ScheduleValidator validator;

    SECTION("Empty journey list validates") {
        std::vector<TileJourney> journeys;
        auto result = validator.validate_journeys(journeys);
        REQUIRE(result.passed);
    }

    SECTION("Complete journey validates") {
        TileJourney journey;
        journey.tile_id = TileID{0, 0, 0};
        journey.dram_fetch_start = 0;
        journey.l3_arrival = 100;
        journey.l2_arrival = 200;
        journey.l1_arrival = 300;
        journey.compute_start = 350;
        journey.compute_end = 400;

        std::vector<TileJourney> journeys{journey};
        auto result = validator.validate_journeys(journeys);
        REQUIRE(result.passed);
    }

    SECTION("Invalid timing order fails") {
        TileJourney journey;
        journey.tile_id = TileID{0, 0, 0};
        journey.dram_fetch_start = 200;  // After L3 arrival (wrong)
        journey.l3_arrival = 100;

        std::vector<TileJourney> journeys{journey};
        auto result = validator.validate_journeys(journeys);
        REQUIRE_FALSE(result.passed);
    }

    SECTION("L2 before L3 fails") {
        TileJourney journey;
        journey.tile_id = TileID{0, 0, 0};
        journey.l3_arrival = 200;
        journey.l2_arrival = 100;  // Before L3 (wrong)

        std::vector<TileJourney> journeys{journey};
        auto result = validator.validate_journeys(journeys);
        REQUIRE_FALSE(result.passed);
    }
}

