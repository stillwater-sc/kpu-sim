// tests/calibration/calibration_extraction_test.cpp
//
// Unit tests for calibration extraction from cycle-accurate model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/calibration/calibration_extraction.hpp>
#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>

#include <sstream>

using namespace sw::kpu::calibration;
using namespace sw::kpu::lpddr5;
using Catch::Matchers::WithinRel;

TEST_CASE("CalibrationExtraction: Statistics helper methods", "[calibration][extraction]") {
    Statistics stats;

    SECTION("avg_read_latency with data") {
        stats.reads = 100;
        stats.read_latency_total = 3600;
        REQUIRE_THAT(stats.avg_read_latency(), WithinRel(36.0, 0.001));
    }

    SECTION("avg_read_latency with zero reads") {
        stats.reads = 0;
        stats.read_latency_total = 0;
        REQUIRE(stats.avg_read_latency() == 0.0);
    }

    SECTION("avg_write_latency with data") {
        stats.writes = 50;
        stats.write_latency_total = 1950;
        REQUIRE_THAT(stats.avg_write_latency(), WithinRel(39.0, 0.001));
    }

    SECTION("page scenario rates") {
        stats.page_hits = 60;
        stats.page_empty = 25;
        stats.page_conflicts = 15;

        REQUIRE_THAT(stats.page_hit_rate(), WithinRel(0.60, 0.001));
        REQUIRE_THAT(stats.page_empty_rate(), WithinRel(0.25, 0.001));
        REQUIRE_THAT(stats.page_conflict_rate(), WithinRel(0.15, 0.001));
    }

    SECTION("page scenario latencies") {
        stats.page_hit_latency_total = 2200;
        stats.page_hit_count = 100;
        stats.page_empty_latency_total = 1500;
        stats.page_empty_count = 40;
        stats.page_conflict_latency_total = 3000;
        stats.page_conflict_count = 60;

        REQUIRE_THAT(stats.avg_page_hit_latency(), WithinRel(22.0, 0.001));
        REQUIRE_THAT(stats.avg_page_empty_latency(), WithinRel(37.5, 0.001));
        REQUIRE_THAT(stats.avg_page_conflict_latency(), WithinRel(50.0, 0.001));
    }

    SECTION("page factors relative to mean") {
        // Setup: 100 requests, mean latency = 36 cycles
        stats.reads = 100;
        stats.writes = 0;
        stats.total_latency = 3600;

        // Page hit: 22 cycles avg -> factor = 22/36 = 0.611
        stats.page_hit_latency_total = 2200;
        stats.page_hit_count = 100;

        // Page empty: 36 cycles avg -> factor = 36/36 = 1.0
        stats.page_empty_latency_total = 0;
        stats.page_empty_count = 0;

        // Page conflict: 50 cycles avg -> factor = 50/36 = 1.389
        stats.page_conflict_latency_total = 0;
        stats.page_conflict_count = 0;

        REQUIRE_THAT(stats.page_hit_factor(), WithinRel(0.611, 0.01));
    }
}

TEST_CASE("CalibrationExtraction: extract_reference", "[calibration][extraction]") {
    Statistics stats;
    stats.reads = 75;
    stats.writes = 25;
    stats.read_latency_total = 2700;  // avg = 36
    stats.write_latency_total = 975;   // avg = 39
    stats.total_latency = 3675;
    stats.page_hits = 50;
    stats.page_empty = 30;
    stats.page_conflicts = 20;

    auto ref = extract_reference(stats, 10000);

    REQUIRE(ref.total_requests == 100);
    REQUIRE(ref.total_cycles == 10000);
    REQUIRE_THAT(ref.mean_read_latency_cycles, WithinRel(36.0, 0.001));
    REQUIRE_THAT(ref.mean_write_latency_cycles, WithinRel(39.0, 0.001));
    REQUIRE_THAT(ref.page_hit_rate, WithinRel(0.50, 0.001));
    REQUIRE_THAT(ref.page_empty_rate, WithinRel(0.30, 0.001));
    REQUIRE_THAT(ref.page_conflict_rate, WithinRel(0.20, 0.001));
}

TEST_CASE("CalibrationExtraction: derive_behavioral", "[calibration][extraction]") {
    Statistics stats;
    stats.reads = 100;
    stats.writes = 100;
    stats.read_latency_total = 3600;   // avg = 36
    stats.write_latency_total = 3900;  // avg = 39

    auto cal = derive_behavioral(stats);

    REQUIRE(cal.fixed_read_latency_cycles == 36);
    REQUIRE(cal.fixed_write_latency_cycles == 39);
}

TEST_CASE("CalibrationExtraction: derive_transactional", "[calibration][extraction]") {
    Statistics stats;
    stats.reads = 100;
    stats.writes = 100;
    stats.read_latency_total = 3600;
    stats.write_latency_total = 3900;
    stats.total_latency = 7500;  // mean = 37.5

    // Page hit: avg 23 cycles, factor = 23/37.5 = 0.613
    stats.page_hit_latency_total = 2300;
    stats.page_hit_count = 100;
    stats.page_hits = 100;

    // Page empty: avg 37.5 cycles, factor = 1.0
    stats.page_empty_latency_total = 1875;
    stats.page_empty_count = 50;
    stats.page_empty = 50;

    // Page conflict: avg 52 cycles, factor = 52/37.5 = 1.387
    stats.page_conflict_latency_total = 2600;
    stats.page_conflict_count = 50;
    stats.page_conflicts = 50;

    auto cal = derive_transactional(stats);

    REQUIRE(cal.mean_read_latency_cycles == 36);
    REQUIRE(cal.mean_write_latency_cycles == 39);
    REQUIRE_THAT(cal.page_hit_factor, WithinRel(0.613, 0.01));
    REQUIRE_THAT(cal.page_empty_factor, WithinRel(1.0, 0.01));
    REQUIRE_THAT(cal.page_conflict_factor, WithinRel(1.387, 0.01));
}

TEST_CASE("CalibrationExtraction: extract_calibration full workflow", "[calibration][extraction]") {
    Statistics stats;
    stats.reads = 500;
    stats.writes = 500;
    stats.read_latency_total = 18000;   // avg = 36
    stats.write_latency_total = 19500;  // avg = 39
    stats.total_latency = 37500;
    stats.page_hits = 400;
    stats.page_empty = 350;
    stats.page_conflicts = 250;
    stats.page_hit_latency_total = 8800;   // avg = 22
    stats.page_hit_count = 400;
    stats.page_empty_latency_total = 14000; // avg = 40
    stats.page_empty_count = 350;
    stats.page_conflict_latency_total = 12500; // avg = 50
    stats.page_conflict_count = 250;

    auto data = extract_calibration(stats, 50000, "LPDDR5", 6400,
                                     {"pattern1.bin", "pattern2.bin"});

    // Check metadata
    REQUIRE(data.technology == "LPDDR5");
    REQUIRE(data.speed_grade_mt_s == 6400);
    REQUIRE(data.source_patterns.size() == 2);
    REQUIRE_FALSE(data.calibration_date.empty());

    // Check reference
    REQUIRE(data.cycle_accurate_reference.total_requests == 1000);
    REQUIRE(data.cycle_accurate_reference.total_cycles == 50000);
    REQUIRE_THAT(data.cycle_accurate_reference.mean_read_latency_cycles, WithinRel(36.0, 0.001));

    // Check behavioral
    REQUIRE(data.behavioral.fixed_read_latency_cycles == 36);
    REQUIRE(data.behavioral.fixed_write_latency_cycles == 39);

    // Check transactional
    REQUIRE(data.transactional.mean_read_latency_cycles == 36);
    REQUIRE(data.transactional.page_hit_factor < 1.0);  // Should be < 1 for page hits
    REQUIRE(data.transactional.page_conflict_factor > 1.0);  // Should be > 1 for conflicts

    // Check validation is not yet done
    REQUIRE(data.validation.status == "NOT_VALIDATED");
}

TEST_CASE("CalibrationExtraction: print_calibration_summary", "[calibration][extraction]") {
    CalibrationData data;
    data.technology = "LPDDR5";
    data.speed_grade_mt_s = 6400;
    data.calibration_date = "2026-01-05";
    data.cycle_accurate_reference.total_requests = 1000;
    data.cycle_accurate_reference.total_cycles = 50000;
    data.cycle_accurate_reference.mean_read_latency_cycles = 36.0;
    data.cycle_accurate_reference.mean_write_latency_cycles = 39.0;
    data.cycle_accurate_reference.page_hit_rate = 0.4;
    data.behavioral.fixed_read_latency_cycles = 36;
    data.behavioral.fixed_write_latency_cycles = 39;
    data.transactional.mean_read_latency_cycles = 36;
    data.transactional.page_hit_factor = 0.6;
    data.transactional.page_conflict_factor = 1.4;
    data.validation.status = "NOT_VALIDATED";

    std::stringstream ss;
    print_calibration_summary(ss, data);
    std::string output = ss.str();

    REQUIRE(output.find("LPDDR5") != std::string::npos);
    REQUIRE(output.find("6400") != std::string::npos);
    REQUIRE(output.find("36") != std::string::npos);
    REQUIRE(output.find("NOT_VALIDATED") != std::string::npos);
}
