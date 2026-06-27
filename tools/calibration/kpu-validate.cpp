// ============================================================================
// tools/calibration/kpu-validate.cpp
// Memory controller calibration validation tool
//
// Runs the same workloads on all three fidelity levels (CA, Transactional,
// Behavioral) and compares the results to validate calibration accuracy.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/calibration/calibration_storage.hpp>
#include <sw/kpu/calibration/calibration_extraction.hpp>
#include <sw/kpu/calibration/calibration_quality.hpp>
#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>
#include <sw/kpu/models/behavioral/memory/memory_controller.hpp>
#include <sw/kpu/models/transactional/memory/memory_controller.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <random>
#include <cmath>

namespace fs = std::filesystem;
using namespace sw::kpu::calibration;
using namespace sw::kpu::lpddr5;
using namespace sw::kpu;

// ============================================================================
// Help Text
// ============================================================================

void print_usage() {
    std::cout << R"(
KPU Memory Controller Calibration Validation Tool

Usage: kpu-validate <calibration_file> [options]

Runs the same workloads on all three fidelity levels (Cycle-Accurate,
Transactional, Behavioral) and compares the results to validate calibration.

Arguments:
  <calibration_file>    Path to calibration JSON file

Options:
  --update              Update the calibration file with validation results
  --requests=<count>    Number of requests per workload type (default: 500)
  --seed=<value>        Random seed for workload generation (default: 42)
  --threshold=<pct>     Maximum acceptable error percentage (default: 5.0)
  --quality             Show detailed quality assessment report
  --verbose             Print detailed progress information
  --help                Show this help message

Examples:
  kpu-validate configs/calibration/lpddr5_6400.json
  kpu-validate lpddr5_6400.json --update --verbose
  kpu-validate lpddr5_6400.json --threshold=10.0 --requests=1000
  kpu-validate lpddr5_6400.json --quality
)";
}

// ============================================================================
// Address Generation
// ============================================================================

constexpr uint64_t BANK_BITS = 4;
constexpr uint64_t ROW_BITS = 16;
constexpr uint64_t COL_BITS = 10;
constexpr uint64_t COL_SHIFT = 6;
constexpr uint64_t BANK_SHIFT = COL_SHIFT + COL_BITS;
constexpr uint64_t ROW_SHIFT = BANK_SHIFT + BANK_BITS;

inline uint64_t make_address(uint8_t bank, uint32_t row, uint32_t col) {
    return (static_cast<uint64_t>(row) << ROW_SHIFT) |
           (static_cast<uint64_t>(bank) << BANK_SHIFT) |
           (static_cast<uint64_t>(col) << COL_SHIFT);
}

// ============================================================================
// Validation Workload
// ============================================================================

struct ValidationRequest {
    bool is_read;
    uint64_t address;
};

std::vector<ValidationRequest> generate_validation_workload(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<ValidationRequest> requests;
    requests.reserve(count);

    std::uniform_int_distribution<unsigned int> bank_dist(0, 15);
    std::uniform_int_distribution<uint32_t> row_dist(0, 65535);
    std::uniform_int_distribution<int> pattern_dist(0, 9);

    uint8_t current_bank = static_cast<uint8_t>(bank_dist(rng));
    uint32_t current_row = row_dist(rng);

    for (size_t i = 0; i < count; ++i) {
        int pattern = pattern_dist(rng);

        if (pattern < 5) {
            // 50% page hits
            uint32_t col = (i % 16) * 64;
            bool is_read = (pattern != 0);
            requests.push_back({is_read, make_address(current_bank, current_row, col)});
        } else if (pattern < 7) {
            // 20% page empty
            current_bank = static_cast<uint8_t>(bank_dist(rng));
            current_row = row_dist(rng);
            bool is_read = (pattern != 5);
            requests.push_back({is_read, make_address(current_bank, current_row, 0)});
        } else {
            // 30% page conflicts
            current_row = row_dist(rng);
            bool is_read = (pattern != 7);
            requests.push_back({is_read, make_address(current_bank, current_row, 0)});
        }
    }

    return requests;
}

// ============================================================================
// Validation Results
// ============================================================================

struct FidelityResults {
    std::string name;
    uint64_t total_cycles = 0;
    uint64_t total_requests = 0;
    double avg_latency = 0.0;
    double avg_read_latency = 0.0;
    double avg_write_latency = 0.0;
};

// ============================================================================
// Run Workload on Cycle-Accurate Controller
// ============================================================================

FidelityResults run_ca_workload(
    const std::vector<ValidationRequest>& requests,
    [[maybe_unused]] const CalibrationData& cal,
    bool verbose)
{
    FidelityResults results;
    results.name = "Cycle-Accurate";

    // Create CA controller
    LPDDR5MemoryController::Config config;
    config.num_channels = 1;
    LPDDR5MemoryController mc(config);

    std::vector<uint8_t> write_data(64, 0xAA);

    if (verbose) {
        std::cout << "  Running CA simulation..." << std::flush;
    }

    // Submit all requests
    for (const auto& req : requests) {
        std::optional<uint64_t> id;
        size_t retry_count = 0;

        while (!id.has_value() && retry_count < 1000) {
            if (req.is_read) {
                id = mc.submit_read(req.address, 64);
            } else {
                id = mc.submit_write(req.address, write_data.data(), 64);
            }

            if (!id.has_value()) {
                mc.tick();
                ++retry_count;
            }
        }

        mc.tick();
    }

    // Drain
    size_t drain_cycles = 0;
    while (mc.has_pending() && drain_cycles < 500000) {
        mc.tick();
        ++drain_cycles;
    }

    const auto& stats = mc.lpddr5_stats();
    results.total_cycles = mc.current_cycle();
    results.total_requests = stats.reads + stats.writes;
    results.avg_latency = stats.avg_latency();
    results.avg_read_latency = stats.avg_read_latency();
    results.avg_write_latency = stats.avg_write_latency();

    if (verbose) {
        std::cout << " done (" << results.total_cycles << " cycles)\n";
    }

    return results;
}

// ============================================================================
// Run Workload on Transactional Controller
// ============================================================================

FidelityResults run_transactional_workload(
    const std::vector<ValidationRequest>& requests,
    const CalibrationData& cal,
    bool verbose)
{
    FidelityResults results;
    results.name = "Transactional";

    // Create transactional controller with calibrated parameters
    MemoryControllerConfig config;
    config.fidelity = SimulationFidelity::TRANSACTIONAL;
    config.technology = MemoryTechnology::LPDDR5;
    config.speed_mt_s = cal.speed_grade_mt_s;
    config.num_channels = 1;
    config.queue_depth = 0;  // Unlimited queue for validation

    // Apply calibration - mean latencies (for fallback)
    config.timing.mean_read_latency = cal.transactional.mean_read_latency_cycles;
    config.timing.mean_write_latency = cal.transactional.mean_write_latency_cycles;
    config.timing.latency_variance = cal.transactional.latency_std_dev_cycles;

    // Apply calibration - page scenario factors (for fallback)
    config.timing.page_hit_factor = cal.transactional.page_hit_factor;
    config.timing.page_empty_factor = cal.transactional.page_empty_factor;
    config.timing.page_conflict_factor = cal.transactional.page_conflict_factor;

    // Apply calibration - per-scenario latencies (preferred when available)
    config.timing.page_hit_latency = cal.transactional.page_hit_latency_cycles;
    config.timing.page_empty_latency = cal.transactional.page_empty_latency_cycles;
    config.timing.page_conflict_latency = cal.transactional.page_conflict_latency_cycles;

    TransactionalMemoryController mc(config);
    mc.set_latency_variance(0.0);  // No variance for validation

    std::vector<uint8_t> write_data(64, 0xAA);
    size_t completed = 0;

    if (verbose) {
        std::cout << "  Running Transactional simulation..." << std::flush;
    }

    // Submit all requests with completion tracking
    for (const auto& req : requests) {
        std::optional<uint64_t> id;
        size_t retry_count = 0;

        while (!id.has_value() && retry_count < 1000) {
            if (req.is_read) {
                id = mc.submit_read(req.address, 64, [&completed]() { ++completed; });
            } else {
                id = mc.submit_write(req.address, write_data.data(), 64, [&completed]() { ++completed; });
            }

            if (!id.has_value()) {
                mc.tick();
                ++retry_count;
            }
        }

        mc.tick();
    }

    // Drain until all complete
    size_t drain_cycles = 0;
    while (completed < requests.size() && drain_cycles < 500000) {
        mc.tick();
        ++drain_cycles;
    }

    const auto& stats = mc.stats();
    results.total_cycles = mc.current_cycle();
    results.total_requests = stats.reads + stats.writes;
    results.avg_latency = stats.avg_latency();
    results.avg_read_latency = results.avg_latency;
    results.avg_write_latency = results.avg_latency;

    if (verbose) {
        std::cout << " done (" << results.total_cycles << " cycles)\n";
    }

    return results;
}

// ============================================================================
// Run Workload on Behavioral Controller
// ============================================================================

FidelityResults run_behavioral_workload(
    const std::vector<ValidationRequest>& requests,
    const CalibrationData& cal,
    bool verbose)
{
    FidelityResults results;
    results.name = "Behavioral";

    // Create behavioral controller with calibrated parameters
    MemoryControllerConfig config;
    config.fidelity = SimulationFidelity::BEHAVIORAL;
    config.technology = MemoryTechnology::LPDDR5;
    config.speed_mt_s = cal.speed_grade_mt_s;
    config.num_channels = 1;

    // Apply calibration
    config.timing.fixed_read_latency = cal.behavioral.fixed_read_latency_cycles;
    config.timing.fixed_write_latency = cal.behavioral.fixed_write_latency_cycles;

    BehavioralMemoryController mc(config);

    std::vector<uint8_t> write_data(64, 0xAA);
    size_t completed = 0;

    if (verbose) {
        std::cout << "  Running Behavioral simulation..." << std::flush;
    }

    // Submit all requests with completion tracking
    for (const auto& req : requests) {
        if (req.is_read) {
            mc.submit_read(req.address, 64, [&completed]() { ++completed; });
        } else {
            mc.submit_write(req.address, write_data.data(), 64, [&completed]() { ++completed; });
        }
        mc.tick();
    }

    // Drain until all complete
    size_t drain_cycles = 0;
    while (completed < requests.size() && drain_cycles < 500000) {
        mc.tick();
        ++drain_cycles;
    }

    const auto& stats = mc.stats();
    results.total_cycles = mc.current_cycle();
    results.total_requests = stats.reads + stats.writes;
    results.avg_latency = stats.avg_latency();
    results.avg_read_latency = results.avg_latency;
    results.avg_write_latency = results.avg_latency;

    if (verbose) {
        std::cout << " done (" << results.total_cycles << " cycles)\n";
    }

    return results;
}

// ============================================================================
// Main Validation
// ============================================================================

struct ValidationOptions {
    std::string calibration_file;
    bool update_file = false;
    size_t requests = 500;
    uint32_t seed = 42;
    double threshold = 5.0;
    bool verbose = false;
    bool show_quality = false;
};

int run_validation(const ValidationOptions& opts) {
    // Load calibration
    auto cal_result = load_calibration(opts.calibration_file);
    if (!cal_result.has_value()) {
        std::cerr << "Error: Failed to load calibration from " << opts.calibration_file << "\n";
        return 1;
    }

    CalibrationData cal = cal_result.value();

    std::cout << "=== KPU Calibration Validation ===" << "\n";
    std::cout << "Calibration: " << opts.calibration_file << "\n";
    std::cout << "Technology: " << cal.technology << " " << cal.speed_grade_mt_s << " MT/s\n";
    std::cout << "Requests: " << opts.requests << "\n";
    std::cout << "Seed: " << opts.seed << "\n";
    std::cout << "Threshold: " << opts.threshold << "%\n";
    std::cout << "\n";

    // Generate validation workload
    auto workload = generate_validation_workload(opts.requests, opts.seed);

    if (opts.verbose) {
        std::cout << "Generated " << workload.size() << " requests\n\n";
    }

    // Run on all fidelity levels
    auto ca_results = run_ca_workload(workload, cal, opts.verbose);
    auto txn_results = run_transactional_workload(workload, cal, opts.verbose);
    auto beh_results = run_behavioral_workload(workload, cal, opts.verbose);

    // Compute errors
    double behavioral_latency_error = 0.0;
    double behavioral_cycle_error = 0.0;
    double transactional_latency_error = 0.0;
    double transactional_cycle_error = 0.0;

    if (ca_results.avg_latency > 0) {
        behavioral_latency_error = std::abs(beh_results.avg_latency - ca_results.avg_latency)
                                   / ca_results.avg_latency * 100.0;
        transactional_latency_error = std::abs(txn_results.avg_latency - ca_results.avg_latency)
                                      / ca_results.avg_latency * 100.0;
    }

    if (ca_results.total_cycles > 0) {
        behavioral_cycle_error = std::abs(static_cast<double>(beh_results.total_cycles) -
                                          static_cast<double>(ca_results.total_cycles))
                                 / static_cast<double>(ca_results.total_cycles) * 100.0;
        transactional_cycle_error = std::abs(static_cast<double>(txn_results.total_cycles) -
                                             static_cast<double>(ca_results.total_cycles))
                                    / static_cast<double>(ca_results.total_cycles) * 100.0;
    }

    // Print results
    std::cout << "\n=== Results ===" << "\n";
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\n" << std::left << std::setw(20) << "Metric"
              << std::right << std::setw(15) << "CA (Reference)"
              << std::setw(15) << "Transactional"
              << std::setw(15) << "Behavioral" << "\n";
    std::cout << std::string(65, '-') << "\n";

    std::cout << std::left << std::setw(20) << "Total cycles"
              << std::right << std::setw(15) << ca_results.total_cycles
              << std::setw(15) << txn_results.total_cycles
              << std::setw(15) << beh_results.total_cycles << "\n";

    std::cout << std::left << std::setw(20) << "Avg latency"
              << std::right << std::setw(15) << ca_results.avg_latency
              << std::setw(15) << txn_results.avg_latency
              << std::setw(15) << beh_results.avg_latency << "\n";

    std::cout << std::left << std::setw(20) << "Requests"
              << std::right << std::setw(15) << ca_results.total_requests
              << std::setw(15) << txn_results.total_requests
              << std::setw(15) << beh_results.total_requests << "\n";

    std::cout << "\n=== Error Analysis ===" << "\n";
    std::cout << std::left << std::setw(25) << "Model"
              << std::setw(20) << "Latency Error"
              << std::setw(20) << "Cycle Error"
              << std::setw(15) << "Status" << "\n";
    std::cout << std::string(80, '-') << "\n";

    bool behavioral_pass = behavioral_latency_error <= opts.threshold &&
                           behavioral_cycle_error <= opts.threshold;
    bool transactional_pass = transactional_latency_error <= opts.threshold &&
                              transactional_cycle_error <= opts.threshold;

    std::cout << std::left << std::setw(25) << "Transactional"
              << std::setw(20) << (std::to_string(transactional_latency_error).substr(0, 5) + "%")
              << std::setw(20) << (std::to_string(transactional_cycle_error).substr(0, 5) + "%")
              << std::setw(15) << (transactional_pass ? "PASS" : "FAIL") << "\n";

    std::cout << std::left << std::setw(25) << "Behavioral"
              << std::setw(20) << (std::to_string(behavioral_latency_error).substr(0, 5) + "%")
              << std::setw(20) << (std::to_string(behavioral_cycle_error).substr(0, 5) + "%")
              << std::setw(15) << (behavioral_pass ? "PASS" : "FAIL") << "\n";

    bool overall_pass = behavioral_pass && transactional_pass;
    std::cout << "\n" << "Overall validation: " << (overall_pass ? "PASSED" : "FAILED") << "\n";

    // Update calibration if requested
    if (opts.update_file) {
        cal.validation.behavioral_latency_error_pct = behavioral_latency_error;
        cal.validation.behavioral_cycle_error_pct = behavioral_cycle_error;
        cal.validation.transactional_latency_error_pct = transactional_latency_error;
        cal.validation.transactional_cycle_error_pct = transactional_cycle_error;
        cal.validation.max_acceptable_error_pct = opts.threshold;
        cal.validation.status = overall_pass ? "PASSED" : "FAILED";

        if (save_calibration(cal, opts.calibration_file)) {
            std::cout << "\nCalibration file updated with validation results.\n";
        } else {
            std::cerr << "\nError: Failed to update calibration file.\n";
            return 1;
        }
    }

    // Show quality assessment if requested
    if (opts.show_quality) {
        std::cout << "\n";
        auto assessment = assess_calibration_quality(cal);
        print_quality_report(std::cout, assessment, opts.verbose);
    }

    return overall_pass ? 0 : 1;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ValidationOptions opts;

    if (argc < 2) {
        print_usage();
        return 1;
    }

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        else if (arg == "--update") {
            opts.update_file = true;
        }
        else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        }
        else if (arg == "--quality" || arg == "-q") {
            opts.show_quality = true;
        }
        else if (arg.rfind("--requests=", 0) == 0) {
            opts.requests = std::stoul(arg.substr(11));
        }
        else if (arg.rfind("--seed=", 0) == 0) {
            opts.seed = std::stoul(arg.substr(7));
        }
        else if (arg.rfind("--threshold=", 0) == 0) {
            opts.threshold = std::stod(arg.substr(12));
        }
        else if (arg[0] != '-') {
            opts.calibration_file = arg;
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
    }

    if (opts.calibration_file.empty()) {
        std::cerr << "Error: No calibration file specified.\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    return run_validation(opts);
}
