// ============================================================================
// tools/harness/schedule_runner.cpp
// CLI tool for running and analyzing KPU data movement schedules
//
// Usage:
//   schedule-runner [options]
//
// Options:
//   --harness <type>       Component to test (dma, block_mover, streamer, pipeline)
//   --stats <level>        Statistics level (basic, detailed)
//   --trace <format>       Trace format (chrome, csv, json)
//   --trace-output <path>  Trace output file
//   --analyze <type>       Analysis type (bottleneck, bandwidth)
//   --help                 Show help
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/harness/dma_harness.hpp>
#include <sw/kpu/harness/block_mover_harness.hpp>
#include <sw/kpu/harness/streamer_harness.hpp>
#include <sw/kpu/harness/pipeline_harness.hpp>
#include <sw/kpu/harness/schedule_validator.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

using namespace sw::kpu::harness;

// ============================================================================
// Command Line Parsing
// ============================================================================

struct Options {
    std::string harness_type = "pipeline";
    std::string stats_level = "basic";
    std::string trace_format;
    std::string trace_output;
    std::string analysis_type;
    std::string schedule_file;
    bool verbose = false;
    bool help = false;

    // Configuration overrides
    uint32_t num_dma_engines = 1;
    uint32_t l3_buffers = 8;
    uint32_t l2_banks = 16;
    uint32_t num_tiles = 16;

    bool parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                help = true;
                return true;
            } else if (arg == "--harness" && i + 1 < argc) {
                harness_type = argv[++i];
            } else if (arg == "--stats" && i + 1 < argc) {
                stats_level = argv[++i];
            } else if (arg == "--trace" && i + 1 < argc) {
                trace_format = argv[++i];
            } else if (arg == "--trace-output" && i + 1 < argc) {
                trace_output = argv[++i];
            } else if (arg == "--analyze" && i + 1 < argc) {
                analysis_type = argv[++i];
            } else if (arg == "--schedule" && i + 1 < argc) {
                schedule_file = argv[++i];
            } else if (arg == "--verbose" || arg == "-v") {
                verbose = true;
            } else if (arg == "--dma-engines" && i + 1 < argc) {
                num_dma_engines = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--l3-buffers" && i + 1 < argc) {
                l3_buffers = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--l2-banks" && i + 1 < argc) {
                l2_banks = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--tiles" && i + 1 < argc) {
                num_tiles = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg[0] != '-' && schedule_file.empty()) {
                schedule_file = arg;
            } else {
                std::cerr << "Unknown option: " << arg << "\n";
                return false;
            }
        }
        return true;
    }
};

void print_help() {
    std::cout << R"(
KPU Schedule Runner - Test harness CLI

Usage: schedule-runner [options]

Options:
  --harness <type>       Component to test:
                           dma         - DMA engine only
                           block_mover - BlockMover only
                           streamer    - Streamer only
                           pipeline    - Full pipeline (default)

  --stats <level>        Statistics detail level:
                           basic       - Summary only (default)
                           detailed    - Full breakdown

  --trace <format>       Enable tracing:
                           chrome      - Chrome trace format (perfetto)
                           csv         - CSV format
                           json        - JSON format

  --trace-output <path>  Trace output file (default: trace.<format>)

  --analyze <type>       Run analysis:
                           bottleneck  - Identify bottlenecks
                           bandwidth   - Bandwidth utilization

  --schedule <path>      Schedule file to load (optional)

  --dma-engines <n>      Number of DMA engines (default: 1)
  --l3-buffers <n>       Number of L3 buffers (default: 8)
  --l2-banks <n>         Number of L2 banks (default: 16)
  --tiles <n>            Number of test tiles (default: 16)

  --verbose, -v          Verbose output
  --help, -h             Show this help

Examples:
  # Test DMA harness with 4 tiles
  schedule-runner --harness dma --tiles 4 --stats detailed

  # Test full pipeline with tracing
  schedule-runner --harness pipeline --trace chrome --trace-output trace.json

  # Run bottleneck analysis
  schedule-runner --analyze bottleneck --stats detailed
)";
}

// ============================================================================
// Test Generators
// ============================================================================

/// Generate simple tile load test for DMA harness
void run_dma_test(const Options& opts) {
    std::cout << "=== DMA Harness Test ===" << std::endl;

    DMAHarnessConfig config;
    config.num_dma_engines = opts.num_dma_engines;
    config.l3_buffer_count = opts.l3_buffers;
    config.enable_tracing = !opts.trace_format.empty();
    config.enable_stats = true;

    DMAHarness harness(config);

    // Generate tile load requests
    for (uint32_t i = 0; i < opts.num_tiles; ++i) {
        TileID tile{0, i / 4, i % 4};
        uint64_t dram_addr = i * config.tile_size;

        harness.request_tile_load(tile, dram_addr);

        if (opts.verbose) {
            std::cout << "  Requested tile " << tile.to_string()
                      << " from DRAM 0x" << std::hex << dram_addr << std::dec
                      << std::endl;
        }
    }

    // Run simulation
    bool completed = harness.run();

    // Print results
    if (opts.stats_level == "detailed") {
        harness.print_stats();
    } else {
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Tiles loaded:  " << harness.dma_stats().tiles_loaded << std::endl;
        std::cout << "  Bytes moved:   " << harness.dma_stats().bytes_transferred << std::endl;
        std::cout << "  Total cycles:  " << harness.current_cycle() << std::endl;
        std::cout << "  Utilization:   " << (harness.dma_stats().utilization() * 100) << "%" << std::endl;
    }

    // Validation
    auto result = harness.validate();
    if (!result.passed) {
        std::cout << "\nValidation FAILED:" << std::endl;
        result.print();
    } else {
        std::cout << "\nValidation PASSED" << std::endl;
    }

    // Export trace if requested
    if (!opts.trace_format.empty()) {
        std::string path = opts.trace_output.empty() ?
            "dma_trace." + opts.trace_format : opts.trace_output;
        harness.export_trace(path);
        std::cout << "\nTrace exported to: " << path << std::endl;
    }

    std::cout << "\nCompleted: " << (completed ? "yes" : "no") << std::endl;
}

/// Generate BlockMover test
void run_block_mover_test(const Options& opts) {
    std::cout << "=== BlockMover Harness Test ===" << std::endl;

    BlockMoverHarnessConfig config;
    config.l3_buffers = opts.l3_buffers;
    config.l2_banks = opts.l2_banks;
    config.enable_tracing = !opts.trace_format.empty();
    config.enable_stats = true;

    BlockMoverHarness harness(config);

    // Preload L3 with test tiles
    for (uint32_t i = 0; i < std::min(opts.num_tiles, opts.l3_buffers); ++i) {
        TileID tile{0, i / 4, i % 4};
        harness.preload_l3_pattern(i, tile, 16, 16, 2);

        if (opts.verbose) {
            std::cout << "  Preloaded L3 buffer " << i
                      << " with tile " << tile.to_string() << std::endl;
        }
    }

    // Request L3->L2 moves
    for (uint32_t i = 0; i < std::min(opts.num_tiles, opts.l3_buffers); ++i) {
        TileID tile{0, i / 4, i % 4};
        harness.move_l3_to_l2(tile, i);
    }

    // Run simulation
    bool completed = harness.run();

    // Print results
    if (opts.stats_level == "detailed") {
        harness.print_stats();
    } else {
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Tiles moved:   " << harness.block_mover_stats().tiles_moved << std::endl;
        std::cout << "  Bytes moved:   " << harness.block_mover_stats().bytes_moved << std::endl;
        std::cout << "  Total cycles:  " << harness.current_cycle() << std::endl;
    }

    auto result = harness.validate();
    std::cout << "\nValidation: " << result.summary() << std::endl;

    std::cout << "\nCompleted: " << (completed ? "yes" : "no") << std::endl;
}

/// Generate Streamer test
void run_streamer_test(const Options& opts) {
    std::cout << "=== Streamer Harness Test ===" << std::endl;

    StreamerHarnessConfig config;
    config.l2_banks = opts.l2_banks;
    config.enable_tracing = !opts.trace_format.empty();
    config.enable_stats = true;

    StreamerHarness harness(config);

    // Preload L2 with test tiles
    for (uint32_t i = 0; i < std::min(opts.num_tiles, opts.l2_banks); ++i) {
        TileID tile{0, i / 4, i % 4};
        harness.preload_l2_pattern(i, tile, 16, 16, 2);
    }

    // Stream tiles
    for (uint32_t i = 0; i < std::min(opts.num_tiles, opts.l2_banks); ++i) {
        TileID tile{0, i / 4, i % 4};
        if (i % 2 == 0) {
            harness.stream_rows(tile, i);
        } else {
            harness.stream_cols(tile, i);
        }
    }

    // Run simulation
    bool completed = harness.run();

    // Print results
    if (opts.stats_level == "detailed") {
        harness.print_stats();
    } else {
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Rows streamed: " << harness.streamer_stats().rows_streamed << std::endl;
        std::cout << "  Cols streamed: " << harness.streamer_stats().cols_streamed << std::endl;
        std::cout << "  Total cycles:  " << harness.current_cycle() << std::endl;
    }

    auto result = harness.validate();
    std::cout << "\nValidation: " << result.summary() << std::endl;

    std::cout << "\nCompleted: " << (completed ? "yes" : "no") << std::endl;
}

/// Generate full pipeline test
void run_pipeline_test(const Options& opts) {
    std::cout << "=== Pipeline Harness Test ===" << std::endl;

    PipelineHarnessConfig config;
    config.dma_config.num_dma_engines = opts.num_dma_engines;
    config.dma_config.l3_buffer_count = opts.l3_buffers;
    config.block_mover_config.l3_buffers = opts.l3_buffers;
    config.block_mover_config.l2_banks = opts.l2_banks;
    config.streamer_config.l2_banks = opts.l2_banks;
    config.enable_tracing = !opts.trace_format.empty();
    config.enable_stats = true;

    DataMovementPipelineHarness harness(config);

    // Initialize DRAM with test data
    harness.init_dram_pattern(0, 64, 64, 2);  // 64x64 FP16 matrix

    // Build a simple schedule
    PipelineSchedule schedule;

    uint32_t tiles_per_dim = 4;  // 4x4 tiles from 64x64 matrix with 16x16 tiles
    uint32_t total_tiles = std::min(opts.num_tiles, tiles_per_dim * tiles_per_dim);

    for (uint32_t i = 0; i < total_tiles; ++i) {
        TileID tile{0, i / tiles_per_dim, i % tiles_per_dim};

        // DMA load
        schedule.add(ScheduleOperation::dma_load(tile, i * 512));

        // BlockMover L3->L2
        auto bm_op = ScheduleOperation::bm_l3_to_l2(tile, UINT32_MAX, UINT32_MAX);
        bm_op.dependencies.push_back(tile);  // Depends on DMA load
        schedule.add(bm_op);

        // Stream
        auto str_op = ScheduleOperation::stream_rows(tile, UINT32_MAX);
        str_op.dependencies.push_back(tile);  // Depends on BM
        schedule.add(str_op);
    }

    // Validate schedule before execution
    ScheduleValidator validator;
    auto validation = validator.validate_schedule(schedule);
    if (!validation.passed) {
        std::cout << "\nSchedule validation FAILED:" << std::endl;
        validation.print();
        return;
    }

    if (opts.verbose) {
        std::cout << "Schedule validated: " << schedule.size() << " operations" << std::endl;
    }

    // Load and run schedule
    harness.load_schedule(schedule);
    bool completed = harness.run();

    // Print results
    if (opts.stats_level == "detailed") {
        harness.print_stats();
    } else {
        auto stats = harness.get_pipeline_stats();
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Total cycles:     " << stats.total_cycles << std::endl;
        std::cout << "  Pipeline eff:     " << (stats.pipeline_efficiency * 100) << "%" << std::endl;
        std::cout << "  Compute util:     " << (stats.compute_utilization * 100) << "%" << std::endl;
        std::cout << "  Bottleneck:       " << stats.bottleneck_component << std::endl;
    }

    // Run analysis if requested
    if (opts.analysis_type == "bottleneck") {
        auto stats = harness.get_pipeline_stats();
        std::cout << "\n=== Bottleneck Analysis ===" << std::endl;
        std::cout << "  Bottleneck component: " << stats.bottleneck_component << std::endl;
        std::cout << "  Bottleneck util:      " << (stats.bottleneck_utilization * 100) << "%" << std::endl;

        std::cout << "\n  Component Utilization:" << std::endl;
        std::cout << "    DMA:         " << (stats.dma_stats.utilization() * 100) << "%" << std::endl;
        std::cout << "    BlockMover:  " << (stats.block_mover_stats.utilization() * 100) << "%" << std::endl;
        std::cout << "    Streamer:    " << (stats.streamer_stats.utilization() * 100) << "%" << std::endl;
    }

    // Tile journey analysis
    auto journey_stats = harness.get_journey_stats();
    std::cout << "\n=== Tile Journey Statistics ===" << std::endl;
    std::cout << "  Completed tiles: " << journey_stats.completed_tiles << "/" << journey_stats.total_tiles << std::endl;
    std::cout << "  Avg total latency: " << journey_stats.avg_total_latency() << " cycles" << std::endl;
    std::cout << "  Avg data movement: " << journey_stats.avg_data_movement_latency() << " cycles" << std::endl;

    auto result = harness.validate();
    std::cout << "\nValidation: " << result.summary() << std::endl;

    // Export trace
    if (!opts.trace_format.empty()) {
        std::string path = opts.trace_output.empty() ?
            "pipeline_trace." + opts.trace_format : opts.trace_output;
        harness.export_trace(path);
        std::cout << "\nTrace exported to: " << path << std::endl;
    }

    std::cout << "\nCompleted: " << (completed ? "yes" : "no") << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Options opts;

    if (!opts.parse(argc, argv)) {
        return 1;
    }

    if (opts.help) {
        print_help();
        return 0;
    }

    std::cout << "KPU Schedule Runner v0.1.0" << std::endl;
    std::cout << "==========================" << std::endl;

    if (opts.harness_type == "dma") {
        run_dma_test(opts);
    } else if (opts.harness_type == "block_mover") {
        run_block_mover_test(opts);
    } else if (opts.harness_type == "streamer") {
        run_streamer_test(opts);
    } else if (opts.harness_type == "pipeline") {
        run_pipeline_test(opts);
    } else {
        std::cerr << "Unknown harness type: " << opts.harness_type << std::endl;
        return 1;
    }

    return 0;
}
