// tests/calibration/calibration_storage_test.cpp
//
// Unit tests for calibration storage
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/calibration/calibration_storage.hpp>

#include <filesystem>
#include <fstream>

using namespace sw::kpu::calibration;
using Catch::Matchers::WithinRel;

namespace {

// Get the project root directory (assumes test runs from build directory)
std::filesystem::path get_project_root() {
    auto cwd = std::filesystem::current_path();
    // If we're in build/, go up one level
    if (cwd.filename() == "build") {
        return cwd.parent_path();
    }
    return cwd;
}

} // anonymous namespace

TEST_CASE("CalibrationStorage: make_calibration_filename", "[calibration]") {
    SECTION("LPDDR5 6400") {
        auto filename = make_calibration_filename("LPDDR5", 6400);
        REQUIRE(filename == "lpddr5_6400.json");
    }

    SECTION("DDR5 4800") {
        auto filename = make_calibration_filename("DDR5", 4800);
        REQUIRE(filename == "ddr5_4800.json");
    }

    SECTION("HBM3 different speed") {
        auto filename = make_calibration_filename("HBM3", 9600);
        REQUIRE(filename == "hbm3_9600.json");
    }
}

TEST_CASE("CalibrationStorage: load existing calibration file", "[calibration]") {
    auto project_root = get_project_root();
    auto cal_path = project_root / "configs" / "calibration" / "lpddr5_6400.json";

    // Skip if file doesn't exist (CI might not have it)
    if (!std::filesystem::exists(cal_path)) {
        SKIP("Calibration file not found: " << cal_path);
    }

    auto result = load_calibration(cal_path);
    REQUIRE(result.has_value());

    const auto& data = result.value();

    SECTION("Metadata is correct") {
        REQUIRE(data.version == "1.0");
        REQUIRE(data.technology == "LPDDR5");
        REQUIRE(data.speed_grade_mt_s == 6400);
    }

    SECTION("Behavioral calibration is loaded") {
        REQUIRE(data.behavioral.fixed_read_latency_cycles == 36);
        REQUIRE(data.behavioral.fixed_write_latency_cycles == 39);
    }

    SECTION("Transactional calibration is loaded") {
        REQUIRE(data.transactional.mean_read_latency_cycles == 36);
        REQUIRE(data.transactional.mean_write_latency_cycles == 39);
        REQUIRE_THAT(data.transactional.page_hit_factor, WithinRel(0.611, 0.01));
        REQUIRE_THAT(data.transactional.page_conflict_factor, WithinRel(1.389, 0.01));
    }

    SECTION("Validation results are loaded") {
        REQUIRE(data.validation.status == "PASSED");
        REQUIRE(data.validation.max_acceptable_error_pct == 5.0);
    }

    SECTION("is_valid returns correct result") {
        REQUIRE(data.is_valid() == true);
    }
}

TEST_CASE("CalibrationStorage: save and reload calibration", "[calibration]") {
    // Create test data
    CalibrationData data;
    data.technology = "TEST_TECH";
    data.speed_grade_mt_s = 1234;
    data.calibration_date = "2026-01-05";
    data.description = "Test calibration";
    data.source_patterns = {"pattern1", "pattern2"};

    data.cycle_accurate_reference.total_requests = 100;
    data.cycle_accurate_reference.mean_read_latency_cycles = 42.5;

    data.behavioral.fixed_read_latency_cycles = 43;
    data.behavioral.fixed_write_latency_cycles = 45;

    data.transactional.mean_read_latency_cycles = 42;
    data.transactional.page_hit_factor = 0.55;

    data.validation.status = "PASSED";
    data.validation.behavioral_cycle_error_pct = 2.1;

    // Save to temp file
    auto temp_path = std::filesystem::temp_directory_path() / "test_calibration.json";
    REQUIRE(save_calibration(data, temp_path));
    REQUIRE(std::filesystem::exists(temp_path));

    // Reload and verify
    auto loaded = load_calibration(temp_path);
    REQUIRE(loaded.has_value());

    const auto& ld = loaded.value();
    REQUIRE(ld.technology == "TEST_TECH");
    REQUIRE(ld.speed_grade_mt_s == 1234);
    REQUIRE(ld.behavioral.fixed_read_latency_cycles == 43);
    REQUIRE_THAT(ld.transactional.page_hit_factor, WithinRel(0.55, 0.01));
    REQUIRE(ld.validation.status == "PASSED");

    // Clean up
    std::filesystem::remove(temp_path);
}

TEST_CASE("CalibrationStorage: load nonexistent file returns nullopt", "[calibration]") {
    auto result = load_calibration("/nonexistent/path/to/file.json");
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("CalibrationStorage: find_calibration_file", "[calibration]") {
    auto project_root = get_project_root();
    auto cal_dir = project_root / "configs" / "calibration";

    // Skip if directory doesn't exist
    if (!std::filesystem::exists(cal_dir)) {
        SKIP("Calibration directory not found: " << cal_dir);
    }

    SECTION("Find existing file") {
        auto result = find_calibration_file(cal_dir, "LPDDR5", 6400);
        if (std::filesystem::exists(cal_dir / "lpddr5_6400.json")) {
            REQUIRE(result.has_value());
            REQUIRE(result.value().filename() == "lpddr5_6400.json");
        }
    }

    SECTION("Nonexistent technology returns nullopt") {
        auto result = find_calibration_file(cal_dir, "NONEXISTENT", 9999);
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("CalibrationStorage: is_valid checks error thresholds", "[calibration]") {
    CalibrationData data;

    SECTION("PASSED with low errors is valid") {
        data.validation.status = "PASSED";
        data.validation.behavioral_cycle_error_pct = 3.0;
        data.validation.transactional_cycle_error_pct = 2.0;
        data.validation.max_acceptable_error_pct = 5.0;
        REQUIRE(data.is_valid() == true);
    }

    SECTION("PASSED with high behavioral error is invalid") {
        data.validation.status = "PASSED";
        data.validation.behavioral_cycle_error_pct = 6.0;  // Over threshold
        data.validation.transactional_cycle_error_pct = 2.0;
        data.validation.max_acceptable_error_pct = 5.0;
        REQUIRE(data.is_valid() == false);
    }

    SECTION("FAILED status is invalid") {
        data.validation.status = "FAILED";
        data.validation.behavioral_cycle_error_pct = 3.0;
        data.validation.transactional_cycle_error_pct = 2.0;
        data.validation.max_acceptable_error_pct = 5.0;
        REQUIRE(data.is_valid() == false);
    }
}
