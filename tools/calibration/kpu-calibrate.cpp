// ============================================================================
// tools/calibration/kpu-calibrate.cpp
// Memory controller calibration tool
//
// Runs cycle-accurate simulation with standard workloads and extracts
// calibration parameters for behavioral and transactional models.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/calibration/calibration_storage.hpp>
#include <sw/kpu/calibration/calibration_extraction.hpp>
#include <sw/kpu/components/lpddr5_memory_controller.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <random>
#include <cstring>

namespace fs = std::filesystem;
using namespace sw::kpu::calibration;
using namespace sw::kpu::lpddr5;

// ============================================================================
// Help Text
// ============================================================================

void print_usage() {
    std::cout << R"(
KPU Memory Controller Calibration Tool

Usage: kpu-calibrate [options]

Runs cycle-accurate LPDDR5 simulation with standard workloads and extracts
calibration parameters for behavioral and transactional memory controller models.

Options:
  --output=<file>       Output calibration file (default: stdout summary only)
  --technology=<tech>   Memory technology (default: LPDDR5)
  --speed=<mt/s>        Speed grade in MT/s (default: 6400)
  --requests=<count>    Number of requests per workload type (default: 1000)
  --seed=<value>        Random seed for workload generation (default: random)
  --verbose             Print detailed progress information
  --help                Show this help message

Workload Mix:
  The calibration runs a balanced mix of access patterns:
  - Page hits: sequential accesses to same row
  - Page empty: accesses to closed banks
  - Page conflicts: accesses to different rows in same bank
  - Mixed read/write: alternating reads and writes
  - Multi-bank: parallel accesses across banks

Examples:
  kpu-calibrate --output=lpddr5_6400.json
  kpu-calibrate --requests=5000 --verbose
  kpu-calibrate --technology=LPDDR5 --speed=8533 --output=lpddr5_8533.json
)";
}

// ============================================================================
// Address Generation
// ============================================================================

constexpr uint64_t BANK_BITS = 4;
constexpr uint64_t ROW_BITS = 16;
constexpr uint64_t COL_BITS = 10;
constexpr uint64_t COL_SHIFT = 6;  // 64-byte cache lines
constexpr uint64_t BANK_SHIFT = COL_SHIFT + COL_BITS;
constexpr uint64_t ROW_SHIFT = BANK_SHIFT + BANK_BITS;

inline uint64_t make_address(uint8_t bank, uint32_t row, uint32_t col) {
    return (static_cast<uint64_t>(row) << ROW_SHIFT) |
           (static_cast<uint64_t>(bank) << BANK_SHIFT) |
           (static_cast<uint64_t>(col) << COL_SHIFT);
}

// ============================================================================
// Workload Generation
// ============================================================================

struct CalibrationWorkload {
    std::string name;
    std::vector<std::pair<RequestType, uint64_t>> requests;  // type, address
};

/// Generate page hit workload (sequential accesses to same row)
CalibrationWorkload make_page_hit_workload(size_t count, std::mt19937& rng) {
    CalibrationWorkload w;
    w.name = "page_hits";

    std::uniform_int_distribution<uint8_t> bank_dist(0, 15);
    std::uniform_int_distribution<uint32_t> row_dist(0, 65535);

    uint8_t bank = bank_dist(rng);
    uint32_t row = row_dist(rng);

    for (size_t i = 0; i < count; ++i) {
        uint32_t col = (i % 16) * 64;  // Stay within same row
        auto type = (i % 4 == 0) ? RequestType::WRITE : RequestType::READ;
        w.requests.push_back({type, make_address(bank, row, col)});
    }

    return w;
}

/// Generate page conflict workload (different rows in same bank)
CalibrationWorkload make_page_conflict_workload(size_t count, std::mt19937& rng) {
    CalibrationWorkload w;
    w.name = "page_conflicts";

    std::uniform_int_distribution<uint8_t> bank_dist(0, 15);

    uint8_t bank = bank_dist(rng);

    for (size_t i = 0; i < count; ++i) {
        uint32_t row = i % 1000;  // Different row each time
        auto type = (i % 4 == 0) ? RequestType::WRITE : RequestType::READ;
        w.requests.push_back({type, make_address(bank, row, 0)});
    }

    return w;
}

/// Generate page empty workload (accesses to idle banks)
CalibrationWorkload make_page_empty_workload(size_t count, std::mt19937& rng) {
    CalibrationWorkload w;
    w.name = "page_empty";

    std::uniform_int_distribution<uint32_t> row_dist(0, 65535);

    for (size_t i = 0; i < count; ++i) {
        uint8_t bank = i % 16;  // Round-robin through banks
        uint32_t row = row_dist(rng);
        auto type = (i % 4 == 0) ? RequestType::WRITE : RequestType::READ;
        w.requests.push_back({type, make_address(bank, row, 0)});
    }

    return w;
}

/// Generate mixed workload (realistic access pattern)
CalibrationWorkload make_mixed_workload(size_t count, std::mt19937& rng) {
    CalibrationWorkload w;
    w.name = "mixed";

    std::uniform_int_distribution<uint8_t> bank_dist(0, 15);
    std::uniform_int_distribution<uint32_t> row_dist(0, 65535);
    std::uniform_int_distribution<int> pattern_dist(0, 9);

    uint8_t current_bank = bank_dist(rng);
    uint32_t current_row = row_dist(rng);

    for (size_t i = 0; i < count; ++i) {
        int pattern = pattern_dist(rng);

        if (pattern < 5) {
            // 50% page hits - stay in same row
            uint32_t col = (i % 16) * 64;
            auto type = (pattern == 0) ? RequestType::WRITE : RequestType::READ;
            w.requests.push_back({type, make_address(current_bank, current_row, col)});
        } else if (pattern < 7) {
            // 20% page empty - new bank
            current_bank = bank_dist(rng);
            current_row = row_dist(rng);
            auto type = (pattern == 5) ? RequestType::WRITE : RequestType::READ;
            w.requests.push_back({type, make_address(current_bank, current_row, 0)});
        } else {
            // 30% page conflicts - different row, same bank
            current_row = row_dist(rng);
            auto type = (pattern == 7) ? RequestType::WRITE : RequestType::READ;
            w.requests.push_back({type, make_address(current_bank, current_row, 0)});
        }
    }

    return w;
}

// ============================================================================
// Calibration Runner
// ============================================================================

struct CalibrationOptions {
    std::string output_file;
    std::string technology = "LPDDR5";
    uint32_t speed_grade = 6400;
    size_t requests_per_workload = 1000;
    uint32_t seed = 0;
    bool verbose = false;
};

bool run_calibration(const CalibrationOptions& opts) {
    // Initialize RNG
    std::mt19937 rng(opts.seed);

    if (opts.verbose) {
        std::cout << "=== KPU Memory Controller Calibration ===" << "\n";
        std::cout << "Technology: " << opts.technology << " " << opts.speed_grade << " MT/s\n";
        std::cout << "Requests per workload: " << opts.requests_per_workload << "\n";
        std::cout << "Random seed: " << opts.seed << "\n";
        std::cout << "\n";
    }

    // Create cycle-accurate memory controller with default LPDDR5 6400 timing
    LPDDR5MemoryController::Config config;
    config.num_channels = 1;
    // TimingParams defaults are already set for LPDDR5 6400

    LPDDR5MemoryController mc(config);

    // Generate workloads
    std::vector<CalibrationWorkload> workloads;
    workloads.push_back(make_page_hit_workload(opts.requests_per_workload, rng));
    workloads.push_back(make_page_conflict_workload(opts.requests_per_workload, rng));
    workloads.push_back(make_page_empty_workload(opts.requests_per_workload, rng));
    workloads.push_back(make_mixed_workload(opts.requests_per_workload * 2, rng));

    // Track pattern names for calibration data
    std::vector<std::string> pattern_names;

    // Run each workload
    for (const auto& workload : workloads) {
        if (opts.verbose) {
            std::cout << "Running workload: " << workload.name
                      << " (" << workload.requests.size() << " requests)..." << std::flush;
        }

        pattern_names.push_back(workload.name);

        // Submit all requests
        size_t submitted = 0;
        std::vector<uint8_t> write_data(64, 0xAA);  // Reusable write buffer

        for (const auto& [type, addr] : workload.requests) {
            // Try to submit, retrying if queue is full
            std::optional<uint64_t> id;
            size_t retry_count = 0;
            constexpr size_t MAX_RETRIES = 1000;

            while (!id.has_value() && retry_count < MAX_RETRIES) {
                if (type == RequestType::READ) {
                    id = mc.submit_read(addr, 64);
                } else {
                    id = mc.submit_write(addr, write_data.data(), 64);
                }

                if (!id.has_value()) {
                    // Queue full - tick to make progress
                    mc.tick();
                    ++retry_count;
                }
            }

            if (id.has_value()) {
                ++submitted;
            }

            // Tick to make progress after submission
            mc.tick();
        }

        // Drain remaining requests
        size_t drain_cycles = 0;
        while (mc.has_pending() && drain_cycles < 500000) {
            mc.tick();
            ++drain_cycles;
        }

        if (opts.verbose) {
            std::cout << " done (" << drain_cycles << " drain cycles)\n";
        }
    }

    // Get final statistics
    const auto& stats = mc.lpddr5_stats();
    uint64_t total_cycles = mc.current_cycle();

    if (opts.verbose) {
        std::cout << "\n";
        std::cout << "=== Raw Statistics ===" << "\n";
        std::cout << "Total requests: " << (stats.reads + stats.writes) << "\n";
        std::cout << "  Reads: " << stats.reads << "\n";
        std::cout << "  Writes: " << stats.writes << "\n";
        std::cout << "Total cycles: " << total_cycles << "\n";
        std::cout << "\n";
        std::cout << "Page scenarios:" << "\n";
        std::cout << "  Hits: " << stats.page_hits << " ("
                  << std::fixed << std::setprecision(1)
                  << (stats.page_hit_rate() * 100) << "%)\n";
        std::cout << "  Empty: " << stats.page_empty << " ("
                  << (stats.page_empty_rate() * 100) << "%)\n";
        std::cout << "  Conflicts: " << stats.page_conflicts << " ("
                  << (stats.page_conflict_rate() * 100) << "%)\n";
        std::cout << "\n";
        std::cout << "Latencies:" << "\n";
        std::cout << "  Avg read: " << std::setprecision(2) << stats.avg_read_latency() << " cycles\n";
        std::cout << "  Avg write: " << stats.avg_write_latency() << " cycles\n";
        std::cout << "  Avg page hit: " << stats.avg_page_hit_latency() << " cycles\n";
        std::cout << "  Avg page empty: " << stats.avg_page_empty_latency() << " cycles\n";
        std::cout << "  Avg page conflict: " << stats.avg_page_conflict_latency() << " cycles\n";
        std::cout << "\n";
    }

    // Extract calibration data
    auto calibration = extract_calibration(
        stats, total_cycles, opts.technology, opts.speed_grade, pattern_names);

    // Print summary
    std::cout << "\n";
    print_calibration_summary(std::cout, calibration);

    // Save to file if requested
    if (!opts.output_file.empty()) {
        if (save_calibration(calibration, opts.output_file)) {
            std::cout << "\nCalibration saved to: " << opts.output_file << "\n";
            return true;
        } else {
            std::cerr << "Error: Failed to save calibration to " << opts.output_file << "\n";
            return false;
        }
    }

    return true;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    CalibrationOptions opts;

    // Generate random seed by default
    std::random_device rd;
    opts.seed = rd();

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        else if (arg.rfind("--output=", 0) == 0) {
            opts.output_file = arg.substr(9);
        }
        else if (arg.rfind("--technology=", 0) == 0) {
            opts.technology = arg.substr(13);
        }
        else if (arg.rfind("--speed=", 0) == 0) {
            opts.speed_grade = std::stoul(arg.substr(8));
        }
        else if (arg.rfind("--requests=", 0) == 0) {
            opts.requests_per_workload = std::stoul(arg.substr(11));
        }
        else if (arg.rfind("--seed=", 0) == 0) {
            opts.seed = std::stoul(arg.substr(7));
        }
        else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
    }

    // Run calibration
    if (!run_calibration(opts)) {
        return 1;
    }

    return 0;
}
