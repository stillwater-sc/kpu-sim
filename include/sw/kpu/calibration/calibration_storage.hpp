// include/sw/kpu/calibration/calibration_storage.hpp
//
// Calibration parameter storage for multi-fidelity memory controller models
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <filesystem>

namespace sw::kpu::calibration {

// ============================================================================
// Calibration Data Structures
// ============================================================================

/// Reference metrics from cycle-accurate simulation
struct CycleAccurateReference {
    uint64_t total_requests = 0;
    uint64_t total_cycles = 0;
    double mean_read_latency_cycles = 0.0;
    double mean_write_latency_cycles = 0.0;
    double latency_std_dev_cycles = 0.0;
    double page_hit_rate = 0.0;
    double page_empty_rate = 0.0;
    double page_conflict_rate = 0.0;
};

/// Behavioral model calibration parameters
struct BehavioralCalibration {
    uint32_t fixed_read_latency_cycles = 100;
    uint32_t fixed_write_latency_cycles = 100;
};

/// Transactional model calibration parameters
struct TransactionalCalibration {
    // Mean latencies (for backward compatibility and reference)
    uint32_t mean_read_latency_cycles = 80;
    uint32_t mean_write_latency_cycles = 90;
    uint32_t latency_std_dev_cycles = 20;

    // Page scenario factors (relative to mean, for legacy use)
    double page_hit_factor = 0.6;
    double page_empty_factor = 1.0;
    double page_conflict_factor = 1.4;

    // Per-scenario latencies (preferred - direct from CA measurement)
    // These represent the actual observed latencies for each scenario
    uint32_t page_hit_latency_cycles = 50;
    uint32_t page_empty_latency_cycles = 80;
    uint32_t page_conflict_latency_cycles = 120;
};

/// Validation results from cross-validation
struct ValidationResults {
    double behavioral_latency_error_pct = 0.0;
    double behavioral_cycle_error_pct = 0.0;
    double transactional_latency_error_pct = 0.0;
    double transactional_cycle_error_pct = 0.0;
    double max_acceptable_error_pct = 5.0;
    std::string status = "NOT_VALIDATED";
};

/// Complete calibration data for a memory technology/speed combination
struct CalibrationData {
    // Metadata
    std::string version = "1.0";
    std::string technology = "LPDDR5";
    uint32_t speed_grade_mt_s = 6400;
    std::string calibration_date;
    std::string description;
    std::vector<std::string> source_patterns;

    // Reference data from cycle-accurate runs
    CycleAccurateReference cycle_accurate_reference;

    // Calibration parameters for each fidelity level
    BehavioralCalibration behavioral;
    TransactionalCalibration transactional;

    // Validation results
    ValidationResults validation;

    /// Check if calibration meets acceptance criteria
    bool is_valid() const {
        return validation.status == "PASSED" &&
               validation.behavioral_cycle_error_pct <= validation.max_acceptable_error_pct &&
               validation.transactional_cycle_error_pct <= validation.max_acceptable_error_pct;
    }
};

// ============================================================================
// Calibration Storage API
// ============================================================================

/// Load calibration data from a JSON file
/// @param path Path to the calibration JSON file
/// @return CalibrationData if successful, nullopt on error
std::optional<CalibrationData> load_calibration(const std::filesystem::path& path);

/// Save calibration data to a JSON file
/// @param data The calibration data to save
/// @param path Path to write the JSON file
/// @return true on success, false on error
bool save_calibration(const CalibrationData& data, const std::filesystem::path& path);

/// Find calibration file for a technology/speed combination
/// @param config_dir Base directory for calibration configs
/// @param technology Memory technology (e.g., "LPDDR5")
/// @param speed_grade Speed grade in MT/s (e.g., 6400)
/// @return Path to calibration file if found, nullopt otherwise
std::optional<std::filesystem::path> find_calibration_file(
    const std::filesystem::path& config_dir,
    const std::string& technology,
    uint32_t speed_grade);

/// Get the default calibration directory path
/// Looks for configs/calibration relative to current working directory
/// or uses KPUSIM_CALIBRATION_DIR environment variable if set
std::filesystem::path get_calibration_dir();

// ============================================================================
// Calibration Key Generation
// ============================================================================

/// Generate a calibration file name from technology and speed grade
/// @param technology Memory technology (e.g., "LPDDR5")
/// @param speed_grade Speed grade in MT/s (e.g., 6400)
/// @return Filename like "lpddr5_6400.json"
std::string make_calibration_filename(const std::string& technology, uint32_t speed_grade);

} // namespace sw::kpu::calibration
