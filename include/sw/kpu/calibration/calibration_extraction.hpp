// include/sw/kpu/calibration/calibration_extraction.hpp
//
// Extraction of calibration parameters from cycle-accurate simulation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/calibration/calibration_storage.hpp>
#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace sw::kpu::calibration {

// ============================================================================
// Calibration Extraction from Cycle-Accurate Model
// ============================================================================

/// Extract calibration reference data from LPDDR5 cycle-accurate statistics
/// @param stats The statistics from a completed CA simulation run
/// @param total_cycles The total simulation cycles
/// @return CycleAccurateReference populated with metrics from the run
inline CycleAccurateReference extract_reference(
    const lpddr5::Statistics& stats,
    uint64_t total_cycles) {

    CycleAccurateReference ref;
    ref.total_requests = stats.reads + stats.writes;
    ref.total_cycles = total_cycles;
    ref.mean_read_latency_cycles = stats.avg_read_latency();
    ref.mean_write_latency_cycles = stats.avg_write_latency();
    ref.page_hit_rate = stats.page_hit_rate();
    ref.page_empty_rate = stats.page_empty_rate();
    ref.page_conflict_rate = stats.page_conflict_rate();

    // TODO: Compute latency standard deviation if needed
    // This would require tracking sum of squared latencies
    ref.latency_std_dev_cycles = 0.0;

    return ref;
}

/// Derive behavioral model parameters from cycle-accurate statistics
/// The behavioral model uses fixed latencies for read and write operations
/// @param stats The statistics from a completed CA simulation run
/// @return BehavioralCalibration with derived fixed latencies
inline BehavioralCalibration derive_behavioral(const lpddr5::Statistics& stats) {
    BehavioralCalibration cal;

    // Use rounded average latencies as fixed values
    cal.fixed_read_latency_cycles = static_cast<uint32_t>(
        stats.avg_read_latency() + 0.5);
    cal.fixed_write_latency_cycles = static_cast<uint32_t>(
        stats.avg_write_latency() + 0.5);

    return cal;
}

/// Derive transactional model parameters from cycle-accurate statistics
/// The transactional model applies factors based on page hit/empty/conflict
/// @param stats The statistics from a completed CA simulation run
/// @return TransactionalCalibration with derived latency factors
inline TransactionalCalibration derive_transactional(const lpddr5::Statistics& stats) {
    TransactionalCalibration cal;

    // Base latencies (rounded averages)
    cal.mean_read_latency_cycles = static_cast<uint32_t>(
        stats.avg_read_latency() + 0.5);
    cal.mean_write_latency_cycles = static_cast<uint32_t>(
        stats.avg_write_latency() + 0.5);

    // Page scenario factors (latency relative to mean) - kept for reference/legacy
    cal.page_hit_factor = stats.page_hit_factor();
    cal.page_empty_factor = stats.page_empty_factor();
    cal.page_conflict_factor = stats.page_conflict_factor();

    // Per-scenario latencies (preferred for transactional model)
    // These are the actual measured latencies from the CA simulation
    cal.page_hit_latency_cycles = static_cast<uint32_t>(
        stats.avg_page_hit_latency() + 0.5);
    cal.page_empty_latency_cycles = static_cast<uint32_t>(
        stats.avg_page_empty_latency() + 0.5);
    cal.page_conflict_latency_cycles = static_cast<uint32_t>(
        stats.avg_page_conflict_latency() + 0.5);

    // TODO: Compute latency std dev if needed
    cal.latency_std_dev_cycles = 0;

    return cal;
}

/// Extract complete calibration data from a cycle-accurate simulation run
/// @param stats The statistics from a completed CA simulation run
/// @param total_cycles The total simulation cycles
/// @param technology Memory technology name (e.g., "LPDDR5")
/// @param speed_grade Speed grade in MT/s (e.g., 6400)
/// @param pattern_names Optional names of patterns used for calibration
/// @return CalibrationData ready to be saved
inline CalibrationData extract_calibration(
    const lpddr5::Statistics& stats,
    uint64_t total_cycles,
    const std::string& technology,
    uint32_t speed_grade,
    const std::vector<std::string>& pattern_names = {}) {

    CalibrationData data;

    // Metadata
    data.technology = technology;
    data.speed_grade_mt_s = speed_grade;
    data.source_patterns = pattern_names;

    // Generate timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d");
    data.calibration_date = ss.str();

    // Extract reference metrics
    data.cycle_accurate_reference = extract_reference(stats, total_cycles);

    // Derive model parameters
    data.behavioral = derive_behavioral(stats);
    data.transactional = derive_transactional(stats);

    // Mark as not yet validated
    data.validation.status = "NOT_VALIDATED";

    // Generate description
    std::stringstream desc;
    desc << "Calibration for " << technology << " " << speed_grade << " MT/s. "
         << "Derived from " << (stats.reads + stats.writes) << " requests "
         << "over " << total_cycles << " cycles.";
    data.description = desc.str();

    return data;
}

/// Print calibration summary to output stream
inline void print_calibration_summary(
    std::ostream& os,
    const CalibrationData& data) {

    os << "=== Calibration Summary ===" << "\n";
    os << "Technology: " << data.technology << " " << data.speed_grade_mt_s << " MT/s\n";
    os << "Date: " << data.calibration_date << "\n";
    os << "\n";

    os << "--- Cycle-Accurate Reference ---\n";
    os << "Total requests: " << data.cycle_accurate_reference.total_requests << "\n";
    os << "Total cycles: " << data.cycle_accurate_reference.total_cycles << "\n";
    os << "Mean read latency: " << data.cycle_accurate_reference.mean_read_latency_cycles << " cycles\n";
    os << "Mean write latency: " << data.cycle_accurate_reference.mean_write_latency_cycles << " cycles\n";
    os << "Page hit rate: " << (data.cycle_accurate_reference.page_hit_rate * 100.0) << "%\n";
    os << "Page empty rate: " << (data.cycle_accurate_reference.page_empty_rate * 100.0) << "%\n";
    os << "Page conflict rate: " << (data.cycle_accurate_reference.page_conflict_rate * 100.0) << "%\n";
    os << "\n";

    os << "--- Behavioral Model Parameters ---\n";
    os << "Fixed read latency: " << data.behavioral.fixed_read_latency_cycles << " cycles\n";
    os << "Fixed write latency: " << data.behavioral.fixed_write_latency_cycles << " cycles\n";
    os << "\n";

    os << "--- Transactional Model Parameters ---\n";
    os << "Mean read latency: " << data.transactional.mean_read_latency_cycles << " cycles\n";
    os << "Mean write latency: " << data.transactional.mean_write_latency_cycles << " cycles\n";
    os << "Page hit factor: " << data.transactional.page_hit_factor << "\n";
    os << "Page empty factor: " << data.transactional.page_empty_factor << "\n";
    os << "Page conflict factor: " << data.transactional.page_conflict_factor << "\n";
    os << "\n";

    os << "--- Validation ---\n";
    os << "Status: " << data.validation.status << "\n";
    if (data.validation.status != "NOT_VALIDATED") {
        os << "Behavioral error: " << data.validation.behavioral_cycle_error_pct << "%\n";
        os << "Transactional error: " << data.validation.transactional_cycle_error_pct << "%\n";
    }
}

} // namespace sw::kpu::calibration
