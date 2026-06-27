// include/sw/kpu/calibration/calibration_quality.hpp
//
// Quality metrics and acceptance criteria for memory controller calibration
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/calibration/calibration_storage.hpp>

#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace sw::kpu::calibration {

// ============================================================================
// Quality Issue Severity Levels
// ============================================================================

enum class Severity {
    INFO,       // Informational, no action needed
    WARNING,    // Potential issue, may affect accuracy
    ERROR       // Critical issue, calibration may be unreliable
};

inline const char* severity_to_string(Severity s) {
    switch (s) {
        case Severity::INFO: return "INFO";
        case Severity::WARNING: return "WARNING";
        case Severity::ERROR: return "ERROR";
    }
    return "UNKNOWN";
}

// ============================================================================
// Quality Issue
// ============================================================================

struct QualityIssue {
    Severity severity;
    std::string category;
    std::string message;
    std::string recommendation;
};

// ============================================================================
// Quality Criteria Configuration
// ============================================================================

struct QualityCriteria {
    // Sample size thresholds
    uint64_t min_total_requests = 100;
    uint64_t recommended_requests = 1000;

    // Page scenario coverage (minimum percentage of each type)
    double min_page_hit_pct = 5.0;
    double min_page_empty_pct = 1.0;
    double min_page_conflict_pct = 5.0;

    // Latency validity ranges (cycles)
    uint32_t min_read_latency = 10;
    uint32_t max_read_latency = 10000;
    uint32_t min_write_latency = 10;
    uint32_t max_write_latency = 10000;

    // Page factor validity ranges
    double min_page_hit_factor = 0.1;
    double max_page_hit_factor = 1.0;
    double min_page_empty_factor = 0.5;
    double max_page_empty_factor = 1.5;
    double min_page_conflict_factor = 1.0;
    double max_page_conflict_factor = 3.0;

    // Read/write latency ratio (write should be >= read typically)
    double min_write_read_ratio = 0.8;
    double max_write_read_ratio = 2.0;

    // Validation error thresholds
    double behavioral_latency_error_threshold = 25.0;
    double behavioral_cycle_error_threshold = 50.0;
    double transactional_latency_error_threshold = 15.0;
    double transactional_cycle_error_threshold = 30.0;
};

// ============================================================================
// Quality Assessment Results
// ============================================================================

struct QualityAssessment {
    std::vector<QualityIssue> issues;

    // Summary scores (0-100, higher is better)
    double sample_quality_score = 0.0;
    double coverage_quality_score = 0.0;
    double latency_quality_score = 0.0;
    double factor_quality_score = 0.0;
    double overall_quality_score = 0.0;

    // Derived status
    bool has_errors() const {
        for (const auto& issue : issues) {
            if (issue.severity == Severity::ERROR) return true;
        }
        return false;
    }

    bool has_warnings() const {
        for (const auto& issue : issues) {
            if (issue.severity == Severity::WARNING) return true;
        }
        return false;
    }

    size_t error_count() const {
        size_t count = 0;
        for (const auto& issue : issues) {
            if (issue.severity == Severity::ERROR) ++count;
        }
        return count;
    }

    size_t warning_count() const {
        size_t count = 0;
        for (const auto& issue : issues) {
            if (issue.severity == Severity::WARNING) ++count;
        }
        return count;
    }

    std::string quality_grade() const {
        if (overall_quality_score >= 90) return "A";
        if (overall_quality_score >= 80) return "B";
        if (overall_quality_score >= 70) return "C";
        if (overall_quality_score >= 60) return "D";
        return "F";
    }
};

// ============================================================================
// Quality Assessment Functions
// ============================================================================

/// Assess sample size quality
inline void assess_sample_quality(
    const CalibrationData& data,
    const QualityCriteria& criteria,
    QualityAssessment& assessment)
{
    const auto& ref = data.cycle_accurate_reference;

    if (ref.total_requests == 0) {
        QualityIssue issue;
        issue.severity = Severity::ERROR;
        issue.category = "sample_size";
        issue.message = "No requests in calibration data";
        issue.recommendation = "Run calibration with at least " + std::to_string(criteria.min_total_requests) + " requests";
        assessment.issues.push_back(issue);
        assessment.sample_quality_score = 0;
        return;
    }

    if (ref.total_requests < criteria.min_total_requests) {
        QualityIssue issue;
        issue.severity = Severity::ERROR;
        issue.category = "sample_size";
        issue.message = "Insufficient sample size: " + std::to_string(ref.total_requests) +
            " requests (minimum: " + std::to_string(criteria.min_total_requests) + ")";
        issue.recommendation = "Increase calibration request count";
        assessment.issues.push_back(issue);
        assessment.sample_quality_score = 30;
    } else if (ref.total_requests < criteria.recommended_requests) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "sample_size";
        issue.message = "Sample size below recommended: " + std::to_string(ref.total_requests) +
            " requests (recommended: " + std::to_string(criteria.recommended_requests) + ")";
        issue.recommendation = "Consider increasing request count for better accuracy";
        assessment.issues.push_back(issue);
        double ratio = static_cast<double>(ref.total_requests) / criteria.recommended_requests;
        assessment.sample_quality_score = 50 + 50 * ratio;
    } else {
        assessment.sample_quality_score = 100;
    }
}

/// Assess page scenario coverage
inline void assess_coverage_quality(
    const CalibrationData& data,
    const QualityCriteria& criteria,
    QualityAssessment& assessment)
{
    const auto& ref = data.cycle_accurate_reference;

    double hit_pct = ref.page_hit_rate * 100.0;
    double empty_pct = ref.page_empty_rate * 100.0;
    double conflict_pct = ref.page_conflict_rate * 100.0;

    [[maybe_unused]] int issues_found = 0;

    if (hit_pct < criteria.min_page_hit_pct) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "coverage";
        issue.message = "Low page hit coverage: " + std::to_string(hit_pct).substr(0, 5) +
            "% (minimum: " + std::to_string(criteria.min_page_hit_pct) + "%)";
        issue.recommendation = "Add more sequential access patterns to calibration workload";
        assessment.issues.push_back(issue);
        ++issues_found;
    }

    if (empty_pct < criteria.min_page_empty_pct) {
        QualityIssue issue;
        issue.severity = Severity::INFO;
        issue.category = "coverage";
        issue.message = "Low page empty coverage: " + std::to_string(empty_pct).substr(0, 5) +
            "% (minimum: " + std::to_string(criteria.min_page_empty_pct) + "%)";
        issue.recommendation = "Add more cold-start patterns to calibration workload";
        assessment.issues.push_back(issue);
        ++issues_found;
    }

    if (conflict_pct < criteria.min_page_conflict_pct) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "coverage";
        issue.message = "Low page conflict coverage: " + std::to_string(conflict_pct).substr(0, 5) +
            "% (minimum: " + std::to_string(criteria.min_page_conflict_pct) + "%)";
        issue.recommendation = "Add more random access patterns to calibration workload";
        assessment.issues.push_back(issue);
        ++issues_found;
    }

    // Score based on balanced coverage
    double balance_score = 100.0;
    if (hit_pct < criteria.min_page_hit_pct) balance_score -= 20;
    if (empty_pct < criteria.min_page_empty_pct) balance_score -= 10;
    if (conflict_pct < criteria.min_page_conflict_pct) balance_score -= 20;

    // Penalize heavily skewed distributions
    if (hit_pct > 90 || conflict_pct > 90) {
        balance_score -= 30;
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "coverage";
        issue.message = "Heavily skewed page scenario distribution";
        issue.recommendation = "Use a more balanced mix of access patterns";
        assessment.issues.push_back(issue);
    }

    assessment.coverage_quality_score = std::max(0.0, balance_score);
}

/// Assess latency value quality
inline void assess_latency_quality(
    const CalibrationData& data,
    const QualityCriteria& criteria,
    QualityAssessment& assessment)
{
    const auto& beh = data.behavioral;
    // Note: cycle_accurate_reference is available for future latency comparisons if needed

    double score = 100.0;

    // Check read latency range
    if (beh.fixed_read_latency_cycles < criteria.min_read_latency) {
        QualityIssue issue;
        issue.severity = Severity::ERROR;
        issue.category = "latency";
        issue.message = "Read latency too low: " + std::to_string(beh.fixed_read_latency_cycles) +
            " cycles (minimum: " + std::to_string(criteria.min_read_latency) + ")";
        issue.recommendation = "Check calibration workload and timing parameters";
        assessment.issues.push_back(issue);
        score -= 40;
    } else if (beh.fixed_read_latency_cycles > criteria.max_read_latency) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "latency";
        issue.message = "Read latency very high: " + std::to_string(beh.fixed_read_latency_cycles) +
            " cycles (may indicate heavy queuing)";
        issue.recommendation = "Consider calibrating under lighter load";
        assessment.issues.push_back(issue);
        score -= 20;
    }

    // Check write latency range
    if (beh.fixed_write_latency_cycles < criteria.min_write_latency) {
        QualityIssue issue;
        issue.severity = Severity::ERROR;
        issue.category = "latency";
        issue.message = "Write latency too low: " + std::to_string(beh.fixed_write_latency_cycles) +
            " cycles (minimum: " + std::to_string(criteria.min_write_latency) + ")";
        issue.recommendation = "Check calibration workload and timing parameters";
        assessment.issues.push_back(issue);
        score -= 40;
    } else if (beh.fixed_write_latency_cycles > criteria.max_write_latency) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "latency";
        issue.message = "Write latency very high: " + std::to_string(beh.fixed_write_latency_cycles) +
            " cycles (may indicate heavy queuing)";
        issue.recommendation = "Consider calibrating under lighter load";
        assessment.issues.push_back(issue);
        score -= 20;
    }

    // Check read/write ratio
    if (beh.fixed_read_latency_cycles > 0) {
        double ratio = static_cast<double>(beh.fixed_write_latency_cycles) / beh.fixed_read_latency_cycles;
        if (ratio < criteria.min_write_read_ratio) {
            QualityIssue issue;
            issue.severity = Severity::INFO;
            issue.category = "latency";
            issue.message = "Write latency unusually lower than read latency (ratio: " +
                std::to_string(ratio).substr(0, 4) + ")";
            issue.recommendation = "This is atypical for most memory technologies";
            assessment.issues.push_back(issue);
            score -= 10;
        } else if (ratio > criteria.max_write_read_ratio) {
            QualityIssue issue;
            issue.severity = Severity::INFO;
            issue.category = "latency";
            issue.message = "Write latency much higher than read latency (ratio: " +
                std::to_string(ratio).substr(0, 4) + ")";
            issue.recommendation = "Consider checking write workload characteristics";
            assessment.issues.push_back(issue);
            score -= 10;
        }
    }

    assessment.latency_quality_score = std::max(0.0, score);
}

/// Assess page factor quality
inline void assess_factor_quality(
    const CalibrationData& data,
    const QualityCriteria& criteria,
    QualityAssessment& assessment)
{
    const auto& txn = data.transactional;
    double score = 100.0;

    // Check page hit factor (should be < 1.0, page hits are faster)
    if (txn.page_hit_factor < criteria.min_page_hit_factor) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "factors";
        issue.message = "Page hit factor too low: " + std::to_string(txn.page_hit_factor).substr(0, 5);
        issue.recommendation = "Page hit factor should typically be between 0.5-0.9";
        assessment.issues.push_back(issue);
        score -= 20;
    } else if (txn.page_hit_factor > criteria.max_page_hit_factor) {
        QualityIssue issue;
        issue.severity = Severity::ERROR;
        issue.category = "factors";
        issue.message = "Page hit factor >= 1.0: " + std::to_string(txn.page_hit_factor).substr(0, 5) +
            " (page hits should be faster than average)";
        issue.recommendation = "Check calibration data - page hits should have lower latency";
        assessment.issues.push_back(issue);
        score -= 40;
    }

    // Check page conflict factor (should be > 1.0, conflicts are slower)
    if (txn.page_conflict_factor < criteria.min_page_conflict_factor) {
        QualityIssue issue;
        issue.severity = Severity::ERROR;
        issue.category = "factors";
        issue.message = "Page conflict factor < 1.0: " + std::to_string(txn.page_conflict_factor).substr(0, 5) +
            " (conflicts should be slower than average)";
        issue.recommendation = "Check calibration data - page conflicts should have higher latency";
        assessment.issues.push_back(issue);
        score -= 40;
    } else if (txn.page_conflict_factor > criteria.max_page_conflict_factor) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "factors";
        issue.message = "Page conflict factor very high: " + std::to_string(txn.page_conflict_factor).substr(0, 5);
        issue.recommendation = "This may indicate unusual workload characteristics";
        assessment.issues.push_back(issue);
        score -= 10;
    }

    // Check page empty factor (should be close to 1.0)
    if (txn.page_empty_factor < criteria.min_page_empty_factor ||
        txn.page_empty_factor > criteria.max_page_empty_factor) {
        QualityIssue issue;
        issue.severity = Severity::INFO;
        issue.category = "factors";
        issue.message = "Page empty factor outside typical range: " +
            std::to_string(txn.page_empty_factor).substr(0, 5);
        issue.recommendation = "Page empty typically has factor near 1.0";
        assessment.issues.push_back(issue);
        score -= 10;
    }

    assessment.factor_quality_score = std::max(0.0, score);
}

/// Assess validation results quality
inline void assess_validation_quality(
    const CalibrationData& data,
    const QualityCriteria& criteria,
    QualityAssessment& assessment)
{
    const auto& val = data.validation;

    if (val.status == "NOT_VALIDATED") {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "validation";
        issue.message = "Calibration has not been validated";
        issue.recommendation = "Run kpu-validate to verify calibration accuracy";
        assessment.issues.push_back(issue);
        return;
    }

    // Check behavioral errors
    if (val.behavioral_latency_error_pct > criteria.behavioral_latency_error_threshold) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "validation";
        issue.message = "Behavioral latency error high: " +
            std::to_string(val.behavioral_latency_error_pct).substr(0, 5) + "%";
        issue.recommendation = "Consider recalibrating with workloads matching validation patterns";
        assessment.issues.push_back(issue);
    }

    if (val.behavioral_cycle_error_pct > criteria.behavioral_cycle_error_threshold) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "validation";
        issue.message = "Behavioral cycle error high: " +
            std::to_string(val.behavioral_cycle_error_pct).substr(0, 5) + "%";
        issue.recommendation = "Behavioral model doesn't model contention - high cycle error is expected";
        assessment.issues.push_back(issue);
    }

    // Check transactional errors
    if (val.transactional_latency_error_pct > criteria.transactional_latency_error_threshold) {
        QualityIssue issue;
        issue.severity = Severity::WARNING;
        issue.category = "validation";
        issue.message = "Transactional latency error high: " +
            std::to_string(val.transactional_latency_error_pct).substr(0, 5) + "%";
        issue.recommendation = "Transactional model may need tuning for this workload";
        assessment.issues.push_back(issue);
    }

    if (val.status == "FAILED") {
        QualityIssue issue;
        issue.severity = Severity::ERROR;
        issue.category = "validation";
        issue.message = "Validation failed - calibration did not meet acceptance criteria";
        issue.recommendation = "Review validation errors and recalibrate if necessary";
        assessment.issues.push_back(issue);
    }
}

/// Perform full quality assessment
inline QualityAssessment assess_calibration_quality(
    const CalibrationData& data,
    const QualityCriteria& criteria = QualityCriteria{})
{
    QualityAssessment assessment;

    assess_sample_quality(data, criteria, assessment);
    assess_coverage_quality(data, criteria, assessment);
    assess_latency_quality(data, criteria, assessment);
    assess_factor_quality(data, criteria, assessment);
    assess_validation_quality(data, criteria, assessment);

    // Calculate overall score (weighted average)
    assessment.overall_quality_score =
        0.20 * assessment.sample_quality_score +
        0.20 * assessment.coverage_quality_score +
        0.30 * assessment.latency_quality_score +
        0.30 * assessment.factor_quality_score;

    // Reduce score for errors and warnings
    assessment.overall_quality_score -= assessment.error_count() * 15;
    assessment.overall_quality_score -= assessment.warning_count() * 5;
    assessment.overall_quality_score = std::max(0.0, assessment.overall_quality_score);

    return assessment;
}

/// Print quality assessment report
inline void print_quality_report(
    std::ostream& os,
    const QualityAssessment& assessment,
    bool verbose = false)
{
    os << "=== Calibration Quality Report ===" << "\n\n";

    os << "Overall Grade: " << assessment.quality_grade()
       << " (" << std::fixed << std::setprecision(1)
       << assessment.overall_quality_score << "/100)\n\n";

    os << "Component Scores:\n";
    os << "  Sample Size:  " << std::setw(5) << assessment.sample_quality_score << "/100\n";
    os << "  Coverage:     " << std::setw(5) << assessment.coverage_quality_score << "/100\n";
    os << "  Latency:      " << std::setw(5) << assessment.latency_quality_score << "/100\n";
    os << "  Factors:      " << std::setw(5) << assessment.factor_quality_score << "/100\n";
    os << "\n";

    if (assessment.issues.empty()) {
        os << "No issues found.\n";
        return;
    }

    os << "Issues (" << assessment.error_count() << " errors, "
       << assessment.warning_count() << " warnings):\n\n";

    for (const auto& issue : assessment.issues) {
        os << "[" << severity_to_string(issue.severity) << "] "
           << issue.category << ": " << issue.message << "\n";
        if (verbose && !issue.recommendation.empty()) {
            os << "  -> " << issue.recommendation << "\n";
        }
    }
}

} // namespace sw::kpu::calibration
