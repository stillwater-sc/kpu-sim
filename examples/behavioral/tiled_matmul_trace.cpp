// ============================================================================
// examples/behavioral/tiled_matmul_trace.cpp
// Generate trace for tiled matrix multiplication: D = C + A * B
// ============================================================================
//
// EXECUTION MODEL: Credit-based dataflow (NO CACHE!)
// See docs/kpu-execution-model.md for the authoritative reference.
//
// Usage:
//   tiled_matmul_trace [--small|--tiny|--medium|--large] [--order IKJ|KIJ|...]
//
// Example sizes:
//   --tiny:   D[32,32]  = A[32,48]  * B[48,32]   (12 matmuls, ~3K cycles)
//   --small:  D[48,48]  = A[48,48]  * B[48,48]   (27 matmuls, ~8K cycles)
//   --medium: D[128,128]= A[128,64] * B[64,128]  (256 matmuls, ~70K cycles)
//   --large:  D[1000,1000]=A[1000,100]*B[100,1000] (27K matmuls, ~3.6M cycles)
//
// The "tiny" example is designed to show:
// - With IKJ: A tiles have good reuse (each loaded once)
// - With IKJ: B tiles have poor reuse (each loaded m_tiles times)
// - Buffer pressure when DMA is faster than downstream consumption
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
    std::cout << "Loop order: " << to_string(config.loop_order) << std::endl;
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
    std::cout << "  L3 cache capacity: " << config.l3_capacity_tiles << " tiles" << std::endl;
    std::cout << "  L3 buffers: " << static_cast<int>(config.num_l3_tiles)
              << " (A: " << static_cast<int>(config.l3_a_buffers[0])
              << "," << static_cast<int>(config.l3_a_buffers[1])
              << ", B: " << static_cast<int>(config.l3_b_buffers[0])
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
              << (static_cast<double>(stats.dma_bytes) / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << std::endl;

    std::cout << "BlockMover operations:" << std::endl;
    std::cout << "  Pushes: " << stats.bm_pushes << std::endl;
    std::cout << "  Pulls: " << stats.bm_pulls << std::endl;
    std::cout << "  Bytes moved: " << stats.bm_bytes << " ("
              << (static_cast<double>(stats.bm_bytes) / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << std::endl;

    std::cout << "Streamer operations:" << std::endl;
    std::cout << "  Feeds: " << stats.str_feeds << std::endl;
    std::cout << "  Drains: " << stats.str_drains << std::endl;
    std::cout << std::endl;

    std::cout << "Compute:" << std::endl;
    std::cout << "  Matmuls: " << stats.matmuls << std::endl;
    std::cout << "  FLOPs: " << stats.flops << std::endl;
    std::cout << std::endl;

    // Buffer utilization statistics (dataflow model - NO CACHE semantics)
    // See docs/kpu-execution-model.md for the credit-based dataflow model
    std::cout << "Buffer Utilization:" << std::endl;
    std::cout << "  L3 TILE_READY events: " << stats.l3_tile_ready_events << std::endl;
    std::cout << "  L3 BUFFER_AVAILABLE credits: " << stats.l3_buffer_available_events << std::endl;
    std::cout << "  L2 TILE_READY events: " << stats.l2_tile_ready_events << std::endl;
    std::cout << "  L2 BUFFER_AVAILABLE credits: " << stats.l2_buffer_available_events << std::endl;
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
    std::cout << "(Credit-based dataflow model - see docs/kpu-execution-model.md)" << std::endl;
    std::cout << std::endl;

    // Default: large problem
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

    // L3 buffer configuration (NOT cache!)
    config.l3_capacity_tiles = 24;      // For compatibility, not used in dataflow
    config.loop_order = LoopOrder::IKJ; // Best A-tile buffer reuse

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

    // Parse command line arguments
    std::string size_mode = "large";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tiny") {
            // Tiny: D[32,32] = A[32,48] * B[48,32]
            // 2x2 output tiles, 2x3 A tiles, 3x2 B tiles = 12 matmul ops
            // In IKJ: A loaded 6 times (good), B loaded 12 times (2x each - shows reuse issue)
            config.M = 32;
            config.N = 32;
            config.K = 48;
            size_mode = "tiny";
        } else if (arg == "--small") {
            // Small: D[48,48] = A[48,48] * B[48,48]
            // 3x3 output tiles, 3x3 A tiles, 3x3 B tiles = 27 matmul ops
            config.M = 48;
            config.N = 48;
            config.K = 48;
            size_mode = "small";
        } else if (arg == "--medium") {
            // Medium: D[128,128] = A[128,64] * B[64,128]
            // 8x8 output tiles, 8x4 A tiles, 4x8 B tiles = 256 matmul ops
            config.M = 128;
            config.N = 128;
            config.K = 64;
            size_mode = "medium";
        } else if (arg == "--large") {
            // Large: default (1000x1000)
            size_mode = "large";
        } else if (arg == "--order" && i + 1 < argc) {
            std::string order = argv[++i];
            if (order == "IJK") config.loop_order = LoopOrder::IJK;
            else if (order == "JIK") config.loop_order = LoopOrder::JIK;
            else if (order == "IKJ") config.loop_order = LoopOrder::IKJ;
            else if (order == "KIJ") config.loop_order = LoopOrder::KIJ;
            else if (order == "KJI") config.loop_order = LoopOrder::KJI;
            else if (order == "JKI") config.loop_order = LoopOrder::JKI;
            else if (order == "BLOCKED") config.loop_order = LoopOrder::BLOCKED;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: tiled_matmul_trace [OPTIONS]\n\n"
                      << "Size options:\n"
                      << "  --tiny    D[32,32] = A[32,48] * B[48,32]   (12 matmuls, ~3K cycles)\n"
                      << "  --small   D[48,48] = A[48,48] * B[48,48]   (27 matmuls, ~8K cycles)\n"
                      << "  --medium  D[128,128]=A[128,64]*B[64,128]   (256 matmuls, ~70K cycles)\n"
                      << "  --large   D[1000,1000]=A[1000,100]*B[100,1000] (27K matmuls, ~3.6M cycles)\n\n"
                      << "Loop order:\n"
                      << "  --order IKJ   A-tile reuse across j (default)\n"
                      << "  --order KIJ   B-tile reuse across i\n"
                      << "  --order IJK   Sequential accumulation\n"
                      << "  --order JIK   B-column reuse\n"
                      << "  --order KJI   A-column reuse\n"
                      << "  --order JKI   B-column streaming\n\n"
                      << "The --tiny example shows buffer reuse patterns:\n"
                      << "  IKJ: A[i,k] reused for all j, B[k,j] loaded m_tiles times\n"
                      << std::endl;
            return 0;
        }
    }

    std::cout << "Size mode: " << size_mode << std::endl;
    print_config(config);

    // Print reuse analysis for educational purposes
    if (size_mode == "tiny" || size_mode == "small") {
        uint32_t m_tiles = config.m_tiles();
        uint32_t n_tiles = config.n_tiles();
        uint32_t k_tiles = config.k_tiles();

        std::cout << "=== Buffer Reuse Analysis (IKJ loop order) ===" << std::endl;
        std::cout << "A tiles: " << m_tiles << " x " << k_tiles << " = "
                  << (m_tiles * k_tiles) << " unique tiles" << std::endl;
        std::cout << "  Each A[i,k] loaded ONCE, reused for all " << n_tiles << " j values" << std::endl;
        std::cout << "  Total A loads from host: " << (m_tiles * k_tiles) << std::endl;
        std::cout << std::endl;

        std::cout << "B tiles: " << k_tiles << " x " << n_tiles << " = "
                  << (k_tiles * n_tiles) << " unique tiles" << std::endl;
        std::cout << "  Each B[k,j] loaded " << m_tiles << " times (once per i row)" << std::endl;
        std::cout << "  Total B loads from host: " << (k_tiles * n_tiles * m_tiles) << std::endl;
        std::cout << std::endl;
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
                  static_cast<double>(program.stats().total_cycles))
              << "x" << std::endl;

    return 0;
}
