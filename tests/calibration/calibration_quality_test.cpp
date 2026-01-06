// tests/calibration/calibration_quality_test.cpp
//
// Unit tests for calibration quality assessment
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/calibration/calibration_quality.hpp>

#include <sstream>

using namespace sw::kpu::calibration;
using Catch::Matchers::WithinRel;

TEST_CASE("CalibrationQuality: sample size assessment", "[calibration][quality]") {
    QualityCriteria criteria;
    criteria.min_total_requests = 100;
    criteria.recommended_requests = 1000;

    SECTION("Zero requests is error") {
        CalibrationData data;
        data.cycle_accurate_reference.total_requests = 0;

        QualityAssessment assessment;
        assess_sample_quality(data, criteria, assessment);

        REQUIRE(assessment.has_errors());
        REQUIRE(assessment.sample_quality_score == 0);
    }

    SECTION("Below minimum is error") {
        CalibrationData data;
        data.cycle_accurate_reference.total_requests = 50;

        QualityAssessment assessment;
        assess_sample_quality(data, criteria, assessment);

        REQUIRE(assessment.has_errors());
        REQUIRE(assessment.sample_quality_score < 50);
    }

    SECTION("At minimum is warning") {
        CalibrationData data;
        data.cycle_accurate_reference.total_requests = 100;

        QualityAssessment assessment;
        assess_sample_quality(data, criteria, assessment);

        REQUIRE(assessment.has_warnings());
        REQUIRE_FALSE(assessment.has_errors());
        REQUIRE(assessment.sample_quality_score >= 50);
    }

    SECTION("At recommended is full score") {
        CalibrationData data;
        data.cycle_accurate_reference.total_requests = 1000;

        QualityAssessment assessment;
        assess_sample_quality(data, criteria, assessment);

        REQUIRE_FALSE(assessment.has_warnings());
        REQUIRE_FALSE(assessment.has_errors());
        REQUIRE(assessment.sample_quality_score == 100);
    }
}

TEST_CASE("CalibrationQuality: coverage assessment", "[calibration][quality]") {
    QualityCriteria criteria;

    SECTION("Balanced coverage gets high score") {
        CalibrationData data;
        data.cycle_accurate_reference.page_hit_rate = 0.40;
        data.cycle_accurate_reference.page_empty_rate = 0.10;
        data.cycle_accurate_reference.page_conflict_rate = 0.50;

        QualityAssessment assessment;
        assess_coverage_quality(data, criteria, assessment);

        REQUIRE_FALSE(assessment.has_errors());
        REQUIRE(assessment.coverage_quality_score >= 80);
    }

    SECTION("Skewed distribution gets warning") {
        CalibrationData data;
        data.cycle_accurate_reference.page_hit_rate = 0.95;
        data.cycle_accurate_reference.page_empty_rate = 0.02;
        data.cycle_accurate_reference.page_conflict_rate = 0.03;

        QualityAssessment assessment;
        assess_coverage_quality(data, criteria, assessment);

        REQUIRE(assessment.has_warnings());
        REQUIRE(assessment.coverage_quality_score < 80);
    }
}

TEST_CASE("CalibrationQuality: latency assessment", "[calibration][quality]") {
    QualityCriteria criteria;
    criteria.min_read_latency = 10;
    criteria.max_read_latency = 10000;

    SECTION("Normal latencies pass") {
        CalibrationData data;
        data.behavioral.fixed_read_latency_cycles = 100;
        data.behavioral.fixed_write_latency_cycles = 110;

        QualityAssessment assessment;
        assess_latency_quality(data, criteria, assessment);

        REQUIRE_FALSE(assessment.has_errors());
        REQUIRE(assessment.latency_quality_score >= 80);
    }

    SECTION("Too low latency is error") {
        CalibrationData data;
        data.behavioral.fixed_read_latency_cycles = 5;
        data.behavioral.fixed_write_latency_cycles = 5;

        QualityAssessment assessment;
        assess_latency_quality(data, criteria, assessment);

        REQUIRE(assessment.has_errors());
    }

    SECTION("Very high latency is warning") {
        CalibrationData data;
        data.behavioral.fixed_read_latency_cycles = 15000;
        data.behavioral.fixed_write_latency_cycles = 15000;

        QualityAssessment assessment;
        assess_latency_quality(data, criteria, assessment);

        REQUIRE(assessment.has_warnings());
    }
}

TEST_CASE("CalibrationQuality: factor assessment", "[calibration][quality]") {
    QualityCriteria criteria;

    SECTION("Valid factors pass") {
        CalibrationData data;
        data.transactional.page_hit_factor = 0.6;
        data.transactional.page_empty_factor = 1.0;
        data.transactional.page_conflict_factor = 1.4;

        QualityAssessment assessment;
        assess_factor_quality(data, criteria, assessment);

        REQUIRE_FALSE(assessment.has_errors());
        REQUIRE(assessment.factor_quality_score >= 90);
    }

    SECTION("Page hit factor >= 1 is error") {
        CalibrationData data;
        data.transactional.page_hit_factor = 1.2;
        data.transactional.page_empty_factor = 1.0;
        data.transactional.page_conflict_factor = 1.4;

        QualityAssessment assessment;
        assess_factor_quality(data, criteria, assessment);

        REQUIRE(assessment.has_errors());
    }

    SECTION("Page conflict factor < 1 is error") {
        CalibrationData data;
        data.transactional.page_hit_factor = 0.6;
        data.transactional.page_empty_factor = 1.0;
        data.transactional.page_conflict_factor = 0.8;

        QualityAssessment assessment;
        assess_factor_quality(data, criteria, assessment);

        REQUIRE(assessment.has_errors());
    }
}

TEST_CASE("CalibrationQuality: full assessment", "[calibration][quality]") {
    SECTION("Good calibration gets high grade") {
        CalibrationData data;
        data.cycle_accurate_reference.total_requests = 1000;
        data.cycle_accurate_reference.total_cycles = 50000;
        data.cycle_accurate_reference.mean_read_latency_cycles = 50;
        data.cycle_accurate_reference.mean_write_latency_cycles = 55;
        data.cycle_accurate_reference.page_hit_rate = 0.40;
        data.cycle_accurate_reference.page_empty_rate = 0.10;
        data.cycle_accurate_reference.page_conflict_rate = 0.50;

        data.behavioral.fixed_read_latency_cycles = 50;
        data.behavioral.fixed_write_latency_cycles = 55;

        data.transactional.mean_read_latency_cycles = 50;
        data.transactional.mean_write_latency_cycles = 55;
        data.transactional.page_hit_factor = 0.6;
        data.transactional.page_empty_factor = 1.0;
        data.transactional.page_conflict_factor = 1.4;

        data.validation.status = "PASSED";

        auto assessment = assess_calibration_quality(data);

        REQUIRE_FALSE(assessment.has_errors());
        REQUIRE((assessment.quality_grade() == "A" || assessment.quality_grade() == "B"));
    }

    SECTION("Poor calibration gets low grade") {
        CalibrationData data;
        data.cycle_accurate_reference.total_requests = 10;  // Too few
        data.cycle_accurate_reference.page_hit_rate = 0.99;  // Skewed
        data.behavioral.fixed_read_latency_cycles = 5;  // Too low
        data.transactional.page_hit_factor = 1.5;  // Invalid

        auto assessment = assess_calibration_quality(data);

        REQUIRE(assessment.has_errors());
        REQUIRE((assessment.quality_grade() == "D" || assessment.quality_grade() == "F"));
    }
}

TEST_CASE("CalibrationQuality: report generation", "[calibration][quality]") {
    CalibrationData data;
    data.cycle_accurate_reference.total_requests = 500;
    data.cycle_accurate_reference.page_hit_rate = 0.40;
    data.cycle_accurate_reference.page_empty_rate = 0.10;
    data.cycle_accurate_reference.page_conflict_rate = 0.50;
    data.behavioral.fixed_read_latency_cycles = 100;
    data.behavioral.fixed_write_latency_cycles = 110;
    data.transactional.page_hit_factor = 0.6;
    data.transactional.page_empty_factor = 1.0;
    data.transactional.page_conflict_factor = 1.4;

    auto assessment = assess_calibration_quality(data);

    std::stringstream ss;
    print_quality_report(ss, assessment, true);
    std::string report = ss.str();

    REQUIRE(report.find("Quality Report") != std::string::npos);
    REQUIRE(report.find("Grade") != std::string::npos);
    REQUIRE(report.find("Sample Size") != std::string::npos);
}
