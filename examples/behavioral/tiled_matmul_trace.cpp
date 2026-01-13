// ============================================================================
// examples/behavioral/tiled_matmul_trace.cpp
// Generate trace for tiled matrix multiplication: D = C + A * B
// ============================================================================
//
// Problem: D[1000,1000] = C[1000,1000] + A[1000,100] * B[100,1000]
//
// Hardware config:
// - 16x16 systolic array
// - 4 L3 tiles (128KB each)
// - 8 L2 banks (64KB each)
// - Double-buffering for A and B tiles
//
// Output: traces/tiled_matmul_trace.json
// View with: tools/visualization/ofg_execution_animation.html
//
// ============================================================================

#include <sw/kpu/behavioral/tiled_matmul_program.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>

using namespace sw::kpu::behavioral;

void print_config(const TiledMatmulConfig& config) {
    std::cout << "=== Problem Configuration ===" << std::endl;
    std::cout << "D[" << config.M << "," << config.N << "] = "
              << "C[" << config.M << "," << config.N << "] + "
              << "A[" << config.M << "," << config.K << "] * "
              << "B[" << config.K << "," << config.N << "]" << std::endl;
    std::cout << std::endl;

    std::cout << "Tile sizes: " << config.tile_m << " x " << config.tile_n
              << " (k=" << config.tile_k << ")" << std::endl;
    std::cout << "Systolic array: " << config.systolic_rows << " x "
              << config.systolic_cols << std::endl;
    std::cout << std::endl;

    std::cout << "Tile decomposition:" << std::endl;
    std::cout << "  M tiles: " << config.m_tiles()
              << " (last partial: " << (config.M % config.tile_m) << ")" << std::endl;
    std::cout << "  N tiles: " << config.n_tiles()
              << " (last partial: " << (config.N % config.tile_n) << ")" << std::endl;
    std::cout << "  K tiles: " << config.k_tiles()
              << " (last partial: " << (config.K % config.tile_k) << ")" << std::endl;
    std::cout << std::endl;

    std::cout << "Total operations:" << std::endl;
    std::cout << "  Output tiles: " << config.total_output_tiles() << std::endl;
    std::cout << "  Matmul ops: " << config.total_matmul_ops() << std::endl;
    std::cout << "  FLOPs: " << config.total_flops() << std::endl;
    std::cout << std::endl;

    std::cout << "Hardware resources:" << std::endl;
    std::cout << "  L3 tiles: " << static_cast<int>(config.num_l3_tiles)
              << " (A buffers: " << static_cast<int>(config.l3_a_buffers[0])
              << "," << static_cast<int>(config.l3_a_buffers[1])
              << ", B buffers: " << static_cast<int>(config.l3_b_buffers[0])
              << "," << static_cast<int>(config.l3_b_buffers[1]) << ")" << std::endl;
    std::cout << "  L2 banks: " << static_cast<int>(config.num_l2_banks) << std::endl;
    std::cout << std::endl;

    std::cout << "Timing parameters:" << std::endl;
    std::cout << "  DMA load: " << config.dma_load_latency << " cycles" << std::endl;
    std::cout << "  DMA store: " << config.dma_store_latency << " cycles" << std::endl;
    std::cout << "  BlockMover push: " << config.bm_push_latency << " cycles" << std::endl;
    std::cout << "  Streamer feed: " << config.str_feed_latency << " cycles" << std::endl;
    std::cout << "  Matmul: " << config.compute_matmul_latency() << " cycles" << std::endl;
    std::cout << "  Drain: " << config.str_drain_latency << " cycles" << std::endl;
    std::cout << std::endl;
}

void print_stats(const TiledMatmulStats& stats, double clock_ghz = 1.0) {
    std::cout << "=== Execution Statistics ===" << std::endl;
    std::cout << "Total cycles: " << stats.total_cycles << std::endl;
    std::cout << "Compute cycles: " << stats.compute_cycles << std::endl;
    std::cout << std::endl;

    std::cout << "DMA operations:" << std::endl;
    std::cout << "  Loads: " << stats.dma_loads << std::endl;
    std::cout << "  Stores: " << stats.dma_stores << std::endl;
    std::cout << "  Bytes transferred: " << stats.dma_bytes << " ("
              << (stats.dma_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << std::endl;

    std::cout << "BlockMover operations:" << std::endl;
    std::cout << "  Pushes: " << stats.bm_pushes << std::endl;
    std::cout << "  Pulls: " << stats.bm_pulls << std::endl;
    std::cout << "  Bytes moved: " << stats.bm_bytes << " ("
              << (stats.bm_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << std::endl;

    std::cout << "Streamer operations:" << std::endl;
    std::cout << "  Feeds: " << stats.str_feeds << std::endl;
    std::cout << "  Drains: " << stats.str_drains << std::endl;
    std::cout << std::endl;

    std::cout << "Compute:" << std::endl;
    std::cout << "  Matmuls: " << stats.matmuls << std::endl;
    std::cout << "  FLOPs: " << stats.flops << std::endl;
    std::cout << std::endl;

    std::cout << "Utilization:" << std::endl;
    std::cout << "  Compute utilization: "
              << std::fixed << std::setprecision(2)
              << (stats.compute_utilization() * 100.0) << "%" << std::endl;
    std::cout << "  Effective TFLOPS @ " << clock_ghz << " GHz: "
              << std::setprecision(3)
              << stats.effective_tflops(clock_ghz) << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Tiled Matrix Multiplication Trace Generator ===" << std::endl;
    std::cout << std::endl;

    // Configure the problem: D[1000,1000] = C[1000,1000] + A[1000,100] * B[100,1000]
    TiledMatmulConfig config;
    config.M = 1000;
    config.N = 1000;
    config.K = 100;

    // Tile sizes matched to 16x16 systolic array
    config.tile_m = 16;
    config.tile_n = 16;
    config.tile_k = 16;
    config.systolic_rows = 16;
    config.systolic_cols = 16;

    // Hardware configuration
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;

    // Timing parameters (typical for on-chip movement)
    config.dma_load_latency = 100;      // ~100 cycles to start DMA
    config.dma_store_latency = 100;
    config.dma_bandwidth = 64;          // 64 bytes/cycle

    config.bm_push_latency = 10;        // L3->L2 is fast
    config.bm_pull_latency = 10;
    config.bm_bandwidth = 32;           // 32 bytes/cycle

    config.str_feed_latency = 8;        // L2->L1 is very fast
    config.str_drain_latency = 16;
    config.str_bandwidth = 16;          // 16 bytes/cycle

    config.accumulate_c = true;

    print_config(config);

    // Optional: reduce problem size if requested
    if (argc > 1) {
        int scale = std::atoi(argv[1]);
        if (scale > 0 && scale < 100) {
            // Scale down for faster testing
            config.M = 16 * scale;
            config.N = 16 * scale;
            config.K = 16 * std::max(1, scale / 10);
            std::cout << "Scaled problem to: D[" << config.M << "," << config.N
                      << "] = A[" << config.M << "," << config.K
                      << "] * B[" << config.K << "," << config.N << "]" << std::endl;
            std::cout << std::endl;
        }
    }

    // Execute with pipelining
    std::cout << "=== Executing Pipelined Tiled Matmul ===" << std::endl;
    TiledMatmulProgram program(config);
    program.execute_pipelined();

    print_stats(program.stats());

    // Create output directory
    std::filesystem::create_directories("traces");

    // Write trace file
    std::string trace_file = "traces/tiled_matmul_trace.json";
    std::cout << "Writing trace to: " << trace_file << std::endl;

    if (program.write_trace_json(trace_file)) {
        std::cout << "Trace file written successfully." << std::endl;
        std::cout << "  Events: " << program.trace().size() << std::endl;
        std::cout << std::endl;
        std::cout << "View with:" << std::endl;
        std::cout << "  tools/visualization/ofg_execution_animation.html" << std::endl;
        std::cout << "  (Load the JSON file using the file input)" << std::endl;
    } else {
        std::cerr << "Failed to write trace file!" << std::endl;
        return 1;
    }

    // Also run non-pipelined for comparison
    std::cout << std::endl << "=== Non-Pipelined Comparison ===" << std::endl;
    TiledMatmulProgram program_seq(config);
    program_seq.execute();

    std::cout << "Sequential cycles: " << program_seq.stats().total_cycles << std::endl;
    std::cout << "Pipelined cycles:  " << program.stats().total_cycles << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2)
              << (static_cast<double>(program_seq.stats().total_cycles) /
                  program.stats().total_cycles)
              << "x" << std::endl;

    return 0;
}
