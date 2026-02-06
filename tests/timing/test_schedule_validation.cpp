// ============================================================================
// tests/timing/test_schedule_validation.cpp
// Unit tests for schedule validation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_validator.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using namespace sw::kpu::isa;

// ============================================================================
// Helper Functions
// ============================================================================

TileDescriptor make_test_tile(MatrixID matrix, Size ti = 0, Size tj = 0, Size tk = 0) {
    TileDescriptor tile;
    tile.tile_id.matrix = matrix;
    tile.tile_id.ti = ti;
    tile.tile_id.tj = tj;
    tile.tile_id.tk = tk;
    tile.size_bytes = 1024;
    return tile;
}

ScheduleResult create_valid_schedule() {
    ScheduleResult result;
    result.valid = true;

    // Create a simple valid schedule for one tile of each type
    auto a_tile = make_test_tile(MatrixID::A);
    auto b_tile = make_test_tile(MatrixID::B);
    auto c_tile = make_test_tile(MatrixID::C);

    // A tile: load -> move -> feed
    result.operations.push_back(ScheduleOperation::load(a_tile));
    result.operations.push_back(ScheduleOperation::move(a_tile));
    result.operations.push_back(ScheduleOperation::feed(a_tile));

    // B tile: load -> move -> feed
    result.operations.push_back(ScheduleOperation::load(b_tile));
    result.operations.push_back(ScheduleOperation::move(b_tile, true));
    result.operations.push_back(ScheduleOperation::feed(b_tile));

    // C tile: drain -> writeback -> store
    result.operations.push_back(ScheduleOperation::drain(c_tile));
    result.operations.push_back(ScheduleOperation::writeback(c_tile));
    result.operations.push_back(ScheduleOperation::store(c_tile));

    return result;
}

// ============================================================================
// ValidationIssue Tests
// ============================================================================

TEST_CASE("ValidationIssue to_string", "[timing][validation]") {
    ValidationIssue issue;
    issue.severity = ValidationSeverity::ERROR;
    issue.rule_id = "VAL-001";
    issue.message = "Test error message";
    issue.operation_index = 42;

    std::string str = issue.to_string();

    REQUIRE(str.find("ERROR") != std::string::npos);
    REQUIRE(str.find("VAL-001") != std::string::npos);
    REQUIRE(str.find("Test error message") != std::string::npos);
    REQUIRE(str.find("42") != std::string::npos);
}

// ============================================================================
// ValidationResult Tests
// ============================================================================

TEST_CASE("ValidationResult counting", "[timing][validation]") {
    ValidationResult result;

    result.issues.push_back({ValidationSeverity::ERROR, "E1", "Error 1", 0});
    result.issues.push_back({ValidationSeverity::ERROR, "E2", "Error 2", 0});
    result.issues.push_back({ValidationSeverity::WARNING, "W1", "Warning 1", 0});
    result.issues.push_back({ValidationSeverity::INFO, "I1", "Info 1", 0});

    REQUIRE(result.count_errors() == 2);
    REQUIRE(result.count_warnings() == 1);
}

TEST_CASE("ValidationResult to_string", "[timing][validation]") {
    ValidationResult result;
    result.valid = false;

    result.issues.push_back({ValidationSeverity::ERROR, "VAL-001", "Schedule empty", 0});

    std::string str = result.to_string();

    REQUIRE(str.find("FAILED") != std::string::npos);
    REQUIRE(str.find("VAL-001") != std::string::npos);
}

// ============================================================================
// ScheduleValidator Basic Tests
// ============================================================================

TEST_CASE("ScheduleValidator VAL-001 empty schedule", "[timing][validation]") {
    ScheduleValidator validator;

    ScheduleResult empty_schedule;
    empty_schedule.valid = true;  // Marked valid but no operations

    auto result = validator.validate(empty_schedule);

    REQUIRE_FALSE(result.valid);
    REQUIRE(result.count_errors() >= 1);

    // Should have VAL-001 error
    bool found_val001 = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-001") {
            found_val001 = true;
            REQUIRE(issue.severity == ValidationSeverity::ERROR);
        }
    }
    REQUIRE(found_val001);
}

TEST_CASE("ScheduleValidator valid schedule passes", "[timing][validation]") {
    ScheduleValidator validator;

    auto schedule = create_valid_schedule();
    auto result = validator.validate(schedule);

    REQUIRE(result.valid);
    REQUIRE(result.count_errors() == 0);
}

// ============================================================================
// VAL-002 Tile Completeness Tests
// ============================================================================

TEST_CASE("ScheduleValidator VAL-002 incomplete A tile", "[timing][validation]") {
    ScheduleValidator validator;

    ScheduleResult schedule;
    schedule.valid = true;

    auto a_tile = make_test_tile(MatrixID::A);

    // A tile with LOAD but no MOVE
    schedule.operations.push_back(ScheduleOperation::load(a_tile));
    schedule.operations.push_back(ScheduleOperation::feed(a_tile));  // Skip MOVE

    auto result = validator.validate(schedule);

    // Should have warning about missing MOVE
    bool found_warning = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-002" && issue.message.find("no MOVE") != std::string::npos) {
            found_warning = true;
            REQUIRE(issue.severity == ValidationSeverity::WARNING);
        }
    }
    REQUIRE(found_warning);
}

TEST_CASE("ScheduleValidator VAL-002 incomplete C tile", "[timing][validation]") {
    ScheduleValidator validator;

    ScheduleResult schedule;
    schedule.valid = true;

    auto c_tile = make_test_tile(MatrixID::C);

    // C tile with DRAIN but no WRITEBACK
    schedule.operations.push_back(ScheduleOperation::drain(c_tile));
    schedule.operations.push_back(ScheduleOperation::store(c_tile));  // Skip WRITEBACK

    auto result = validator.validate(schedule);

    // Should have warning about missing WRITEBACK
    bool found_warning = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-002" && issue.message.find("no WRITEBACK") != std::string::npos) {
            found_warning = true;
        }
    }
    REQUIRE(found_warning);
}

// ============================================================================
// VAL-003 Interleaving Tests
// ============================================================================

TEST_CASE("ScheduleValidator VAL-003 non-interleaved warning", "[timing][validation]") {
    ScheduleValidator validator;

    ScheduleResult schedule;
    schedule.valid = true;

    auto a_tile = make_test_tile(MatrixID::A);
    auto b_tile = make_test_tile(MatrixID::B);

    // Non-interleaved: A-A-A-A-B-B-B-B
    for (int i = 0; i < 5; ++i) {
        schedule.operations.push_back(ScheduleOperation::load(a_tile));
    }
    for (int i = 0; i < 5; ++i) {
        schedule.operations.push_back(ScheduleOperation::load(b_tile));
    }

    auto result = validator.validate(schedule);

    // Should have warning about not being interleaved
    bool found_warning = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-003") {
            found_warning = true;
        }
    }
    REQUIRE(found_warning);
}

TEST_CASE("ScheduleValidator VAL-003 require_interleaving config", "[timing][validation]") {
    ScheduleValidator::Config config;
    config.require_interleaving = true;

    ScheduleValidator validator(config);

    ScheduleResult schedule;
    schedule.valid = true;

    auto a_tile = make_test_tile(MatrixID::A);
    auto b_tile = make_test_tile(MatrixID::B);

    // Non-interleaved
    for (int i = 0; i < 5; ++i) {
        schedule.operations.push_back(ScheduleOperation::load(a_tile));
    }
    for (int i = 0; i < 5; ++i) {
        schedule.operations.push_back(ScheduleOperation::load(b_tile));
    }

    auto result = validator.validate(schedule);

    // With require_interleaving, should be an ERROR
    REQUIRE_FALSE(result.valid);
    bool found_error = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-003" && issue.severity == ValidationSeverity::ERROR) {
            found_error = true;
        }
    }
    REQUIRE(found_error);
}

// ============================================================================
// VAL-004 A/B Balance Tests
// ============================================================================

TEST_CASE("ScheduleValidator VAL-004 consecutive threshold", "[timing][validation]") {
    ScheduleValidator::Config config;
    config.max_consecutive_threshold = 3;

    ScheduleValidator validator(config);

    ScheduleResult schedule;
    schedule.valid = true;

    auto a_tile = make_test_tile(MatrixID::A);

    // 5 consecutive A operations exceeds threshold of 3
    for (int i = 0; i < 5; ++i) {
        schedule.operations.push_back(ScheduleOperation::load(a_tile));
    }

    auto result = validator.validate(schedule);

    // Should have warning about consecutive A
    bool found_warning = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-004" && issue.message.find("consecutive A") != std::string::npos) {
            found_warning = true;
        }
    }
    REQUIRE(found_warning);
}

TEST_CASE("ScheduleValidator VAL-004 unbalanced ratio", "[timing][validation]") {
    ScheduleValidator::Config config;
    config.max_ab_ratio = 2.0;  // Max 2:1 ratio

    ScheduleValidator validator(config);

    ScheduleResult schedule;
    schedule.valid = true;

    auto a_tile = make_test_tile(MatrixID::A);
    auto b_tile = make_test_tile(MatrixID::B);

    // 10:1 A:B ratio
    for (int i = 0; i < 10; ++i) {
        schedule.operations.push_back(ScheduleOperation::load(a_tile));
    }
    schedule.operations.push_back(ScheduleOperation::load(b_tile));

    auto result = validator.validate(schedule);

    // Should have warning about unbalanced ratio
    bool found_warning = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-004" && issue.message.find("ratio") != std::string::npos) {
            found_warning = true;
        }
    }
    REQUIRE(found_warning);
}

// ============================================================================
// VAL-006 Operation Order Tests
// ============================================================================

TEST_CASE("ScheduleValidator VAL-006 MOVE before LOAD", "[timing][validation]") {
    ScheduleValidator validator;

    ScheduleResult schedule;
    schedule.valid = true;

    auto a_tile = make_test_tile(MatrixID::A);

    // Wrong order: MOVE before LOAD
    schedule.operations.push_back(ScheduleOperation::move(a_tile));
    schedule.operations.push_back(ScheduleOperation::load(a_tile));
    schedule.operations.push_back(ScheduleOperation::feed(a_tile));

    auto result = validator.validate(schedule);

    // Should have error about wrong order
    bool found_error = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-006" && issue.message.find("MOVE before LOAD") != std::string::npos) {
            found_error = true;
            REQUIRE(issue.severity == ValidationSeverity::ERROR);
        }
    }
    REQUIRE(found_error);
}

TEST_CASE("ScheduleValidator VAL-006 STORE before WRITEBACK", "[timing][validation]") {
    ScheduleValidator validator;

    ScheduleResult schedule;
    schedule.valid = true;

    auto c_tile = make_test_tile(MatrixID::C);

    // Wrong order: STORE before WRITEBACK
    schedule.operations.push_back(ScheduleOperation::drain(c_tile));
    schedule.operations.push_back(ScheduleOperation::store(c_tile));
    schedule.operations.push_back(ScheduleOperation::writeback(c_tile));

    auto result = validator.validate(schedule);

    // Should have error about wrong order
    bool found_error = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-006" && issue.message.find("STORE before WRITEBACK") != std::string::npos) {
            found_error = true;
        }
    }
    REQUIRE(found_error);
}

// ============================================================================
// Livelock Safety Validation Tests
// ============================================================================

TEST_CASE("ScheduleValidator livelock safety with sufficient credits", "[timing][validation][livelock]") {
    ScheduleValidator validator;

    MatMulScheduleGenerator::Config gen_config;
    gen_config.M = 32;
    gen_config.N = 32;
    gen_config.K = 32;
    gen_config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

    MatMulScheduleGenerator gen(gen_config);
    auto schedule = gen.generate();

    // Validate with plenty of credits
    auto result = validator.validate_livelock_safety(schedule, 32, 64);

    REQUIRE(result.valid);
    REQUIRE(result.count_errors() == 0);
}

TEST_CASE("ScheduleValidator livelock safety with limited credits", "[timing][validation][livelock]") {
    ScheduleValidator validator;

    MatMulScheduleGenerator::Config gen_config;
    gen_config.M = 64;
    gen_config.N = 64;
    gen_config.K = 64;
    gen_config.strategy = MatMulScheduleGenerator::Strategy::BLOCKED_AB;  // Livelock-prone

    MatMulScheduleGenerator gen(gen_config);
    auto schedule = gen.generate();

    // Validate with very limited credits
    auto result = validator.validate_livelock_safety(schedule, 6, 12);

    // Should have livelock warning or error
    REQUIRE_FALSE(result.valid);

    bool found_ll_issue = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id.find("VAL-LL") != std::string::npos) {
            found_ll_issue = true;
        }
    }
    REQUIRE(found_ll_issue);
}

TEST_CASE("ScheduleValidator livelock safety VAL-LL3", "[timing][validation][livelock]") {
    ScheduleValidator validator;

    ScheduleResult schedule;
    schedule.valid = true;

    auto a_tile = make_test_tile(MatrixID::A);
    auto b_tile = make_test_tile(MatrixID::B);

    // Non-interleaved schedule
    for (int i = 0; i < 10; ++i) {
        schedule.operations.push_back(ScheduleOperation::load(a_tile));
    }
    for (int i = 0; i < 10; ++i) {
        schedule.operations.push_back(ScheduleOperation::load(b_tile));
    }

    // Very limited credits - should trigger VAL-LL3
    auto result = validator.validate_livelock_safety(schedule, 6, 8);

    bool found_ll3 = false;
    for (const auto& issue : result.issues) {
        if (issue.rule_id == "VAL-LL3") {
            found_ll3 = true;
            REQUIRE(issue.severity == ValidationSeverity::ERROR);
        }
    }
    REQUIRE(found_ll3);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST_CASE("validate_schedule convenience function", "[timing][validation]") {
    auto schedule = create_valid_schedule();
    auto result = validate_schedule(schedule);

    REQUIRE(result.valid);
}

TEST_CASE("validate_livelock_safety convenience function", "[timing][validation][livelock]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

    MatMulScheduleGenerator gen(config);
    auto schedule = gen.generate();

    auto result = validate_livelock_safety(schedule, 16, 32);
    REQUIRE(result.valid);
}

TEST_CASE("is_livelock_safe convenience function", "[timing][validation][livelock]") {
    SECTION("Interleaved schedule is safe") {
        MatMulScheduleGenerator::Config config;
        config.M = 32;
        config.N = 32;
        config.K = 32;
        config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

        MatMulScheduleGenerator gen(config);
        auto schedule = gen.generate();

        REQUIRE(is_livelock_safe(schedule));
    }

    SECTION("Blocked schedule is not safe") {
        MatMulScheduleGenerator::Config config;
        config.M = 64;
        config.N = 64;
        config.K = 64;
        config.strategy = MatMulScheduleGenerator::Strategy::BLOCKED_AB;

        MatMulScheduleGenerator gen(config);
        auto schedule = gen.generate();

        REQUIRE_FALSE(is_livelock_safe(schedule));
    }
}

// ============================================================================
// Integration with MatMul Generator Tests
// ============================================================================

TEST_CASE("Validate MatMul interleaved schedule", "[timing][validation][matmul]") {
    ScheduleValidator validator;

    MatMulScheduleGenerator::Config config;
    config.M = 64;
    config.N = 64;
    config.K = 64;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

    MatMulScheduleGenerator gen(config);
    auto schedule = gen.generate();

    auto result = validator.validate(schedule);

    INFO(result.to_string());
    REQUIRE(result.valid);
    REQUIRE(result.count_errors() == 0);
}

TEST_CASE("Validate MatMul output_stationary schedule", "[timing][validation][matmul]") {
    ScheduleValidator validator;

    MatMulScheduleGenerator::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.strategy = MatMulScheduleGenerator::Strategy::OUTPUT_STATIONARY;

    MatMulScheduleGenerator gen(config);
    auto schedule = gen.generate();

    auto result = validator.validate(schedule);

    INFO(result.to_string());
    // Output stationary may have warnings but should pass validation
    REQUIRE(result.valid);
}

TEST_CASE("Large MatMul with limited credits warning", "[timing][validation][matmul][livelock]") {
    ScheduleValidator validator;

    MatMulScheduleGenerator::Config config;
    config.M = 128;
    config.N = 128;
    config.K = 128;
    config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

    MatMulScheduleGenerator gen(config);
    auto schedule = gen.generate();

    // Check livelock safety with very limited credits
    auto result = validator.validate_livelock_safety(schedule, 8, 16);

    // May have warnings about peak usage
    INFO(result.to_string());

    // But interleaved should still be considered safe
    REQUIRE(is_livelock_safe(schedule));
}
