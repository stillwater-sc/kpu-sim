/**
 * @file data_movement_isa_matmul.cpp
 * @brief Matrix multiplication using the Data Movement ISA
 *
 * This example demonstrates the Domain Flow Architecture programming model
 * where the program IS the data movement schedule. Unlike traditional
 * stored-program machines, the KPU compute fabric is reactive - it executes
 * when data tokens arrive. The intelligence is in orchestrating optimal
 * data movement patterns derived from SURE (Space-time Uniform Recurrence
 * Equation) analysis.
 *
 * Key concepts demonstrated:
 * 1. Output-stationary dataflow for matrix multiplication
 * 2. Tiled execution with configurable tile sizes
 * 3. Data Movement ISA instruction generation
 * 4. Program disassembly and analysis
 *
 * Compare this with data_movement_pipeline.cpp which uses direct
 * component API calls - that approach is useful for testing individual
 * components, while this ISA-based approach represents how actual
 * programs execute on the KPU.
 */

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/isa/program_executor.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace sw::kpu::isa;
using sw::kpu::Size;
using sw::kpu::Cycle;

// ============================================================================
// Helper Functions
// ============================================================================

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_program_summary(const DMProgram& program) {
    std::cout << "\nProgram: " << program.name << "\n";
    std::cout << "Dataflow: "
              << (program.dataflow == DMProgram::Dataflow::OUTPUT_STATIONARY ?
                  "Output-Stationary" :
                  program.dataflow == DMProgram::Dataflow::WEIGHT_STATIONARY ?
                  "Weight-Stationary" : "Input-Stationary") << "\n";

    std::cout << "\nMatrix Dimensions:\n";
    std::cout << "  C[" << program.M << "," << program.N << "] = "
              << "A[" << program.M << "," << program.K << "] x "
              << "B[" << program.K << "," << program.N << "]\n";

    std::cout << "\nTiling Configuration:\n";
    std::cout << "  Tile sizes: Ti=" << program.Ti
              << " Tj=" << program.Tj
              << " Tk=" << program.Tk << "\n";

    Size m_tiles = (program.M + program.Ti - 1) / program.Ti;
    Size n_tiles = (program.N + program.Tj - 1) / program.Tj;
    Size k_tiles = (program.K + program.Tk - 1) / program.Tk;
    std::cout << "  Tile counts: " << m_tiles << " x " << n_tiles << " x " << k_tiles
              << " = " << (m_tiles * n_tiles * k_tiles) << " tile iterations\n";
    std::cout << "  Output tiles: " << (m_tiles * n_tiles) << "\n";

    std::cout << "\nInstruction Statistics:\n";
    std::cout << "  Total instructions: " << program.instructions.size() << "\n";
    std::cout << "  DMA operations:     " << program.num_dma_ops() << "\n";
    std::cout << "  BlockMover ops:     " << program.num_bm_ops() << "\n";
    std::cout << "  Streamer ops:       " << program.num_str_ops() << "\n";
    std::cout << "  Sync operations:    " << program.num_sync_ops() << "\n";

    std::cout << "\nTraffic Estimates:\n";
    std::cout << "  External memory: " << std::fixed << std::setprecision(2)
              << (program.estimates.external_mem_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  L3 traffic:      "
              << (program.estimates.l3_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  L2 traffic:      "
              << (program.estimates.l2_bytes / (1024.0 * 1024.0)) << " MB\n";

    // Calculate minimum traffic and reuse
    Size min_bytes = (program.M * program.K + program.K * program.N +
                     program.M * program.N) * 4;  // float32
    double reuse = static_cast<double>(program.estimates.external_mem_bytes) / min_bytes;

    std::cout << "  Minimum external: " << (min_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Reuse factor:     " << reuse << "x\n";

    std::cout << "\nPerformance Metrics:\n";
    Size total_flops = 2ULL * program.M * program.N * program.K;
    std::cout << "  Total FLOPs:           " << (total_flops / 1e9) << " GFLOPs\n";
    std::cout << "  Arithmetic intensity:  " << program.estimates.arithmetic_intensity
              << " FLOPs/byte\n";
}

void print_instruction_trace(const DMProgram& program, size_t max_instructions = 50) {
    std::cout << "\nInstruction Trace (first " << max_instructions << " instructions):\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(5) << "PC" << " | "
              << std::setw(25) << "Operation" << " | "
              << "Details\n";
    std::cout << std::string(80, '-') << "\n";

    size_t count = std::min(max_instructions, program.instructions.size());
    for (size_t i = 0; i < count; ++i) {
        const auto& instr = program.instructions[i];
        std::cout << std::setw(5) << i << " | "
                  << std::setw(25) << std::left << instr.label << std::right
                  << " | ";

        // Print additional details based on opcode - focus on submatrix dimensions
        switch (instr.opcode) {
            case DMOpcode::DMA_LOAD_TILE:
            case DMOpcode::DMA_STORE_TILE: {
                const auto& ops = std::get<DMAOperands>(instr.operands);
                std::cout << ops.size_bytes << " bytes";
                break;
            }
            case DMOpcode::BM_MOVE_TILE:
            case DMOpcode::BM_TRANSPOSE_TILE: {
                const auto& ops = std::get<BlockMoverOperands>(instr.operands);
                std::cout << ops.height << "x" << ops.width << " elements";
                break;
            }
            case DMOpcode::STR_FEED_ROWS:
            case DMOpcode::STR_FEED_COLS:
            case DMOpcode::STR_DRAIN_OUTPUT: {
                const auto& ops = std::get<StreamerOperands>(instr.operands);
                std::cout << ops.height << "x" << ops.width << " elements";
                break;
            }
            case DMOpcode::BARRIER:
                std::cout << "sync all pending ops";
                break;
            case DMOpcode::HALT:
                std::cout << "end program";
                break;
            default:
                break;
        }
        std::cout << "\n";
    }

    if (program.instructions.size() > max_instructions) {
        std::cout << "... (" << (program.instructions.size() - max_instructions)
                  << " more instructions)\n";
    }
    std::cout << std::string(80, '-') << "\n";
}

// ============================================================================
// Example 1: Small Matrix Multiplication
// ============================================================================

void example_small_matmul() {
    print_separator("Example 1: Small MatMul (64x64x64)");

    std::cout << "\nThis example shows a small matrix multiplication that fits\n"
              << "entirely in L2 cache with minimal tiling.\n";

    OutputStationaryProgramBuilder::Config config;
    config.M = 64;
    config.N = 64;
    config.K = 64;
    config.Ti = 32;  // 2x2 output tiles
    config.Tj = 32;
    config.Tk = 32;  // 2 reduction tiles
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;  // float32

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = false;

    OutputStationaryProgramBuilder builder(config);

    auto start = std::chrono::high_resolution_clock::now();
    DMProgram program = builder.build();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "\nProgram generation time: " << duration.count() << " us\n";

    print_program_summary(program);

    // Show tile cache statistics
    std::cout << builder.get_cache_stats();

    print_instruction_trace(program, 30);

    // Validate
    std::string error;
    if (validate_program(program, error)) {
        std::cout << "\nProgram validation: PASSED\n";
    } else {
        std::cout << "\nProgram validation: FAILED - " << error << "\n";
    }
}

// ============================================================================
// Example 2: Large Matrix with Double Buffering
// ============================================================================

void example_large_matmul_double_buffered() {
    print_separator("Example 2: Large MatMul with Double Buffering (512x512x512)");

    std::cout << "\nThis example shows a larger matrix multiplication using\n"
              << "double buffering to overlap data movement with computation.\n"
              << "Double buffering allows loading the next tile while computing\n"
              << "the current one, hiding memory latency.\n";

    OutputStationaryProgramBuilder::Config config;
    config.M = 512;
    config.N = 512;
    config.K = 512;
    config.Ti = 64;
    config.Tj = 64;
    config.Tk = 64;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = true;  // Enable double buffering

    OutputStationaryProgramBuilder builder(config);

    auto start = std::chrono::high_resolution_clock::now();
    DMProgram program = builder.build();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "\nProgram generation time: " << duration.count() << " us\n";

    print_program_summary(program);
}

// ============================================================================
// Example 3: Comparing Tile Size Impact
// ============================================================================

void example_tile_size_comparison() {
    print_separator("Example 3: Tile Size Impact Analysis");

    std::cout << "\nThis example compares how different tile sizes affect\n"
              << "the generated program and traffic estimates.\n"
              << "Larger tiles reduce instruction count but increase memory\n"
              << "pressure, while smaller tiles have more overhead but fit\n"
              << "better in cache.\n";

    const Size M = 256, N = 256, K = 256;

    struct TileExperiment {
        Size Ti, Tj, Tk;
        const char* description;
    };

    std::vector<TileExperiment> experiments = {
        {32, 32, 32, "Small tiles (32x32x32)"},
        {64, 64, 64, "Medium tiles (64x64x64)"},
        {128, 128, 128, "Large tiles (128x128x128)"},
        {64, 64, 128, "Wide K tiles (64x64x128)"},
    };

    std::cout << "\nMatrix: C[" << M << "," << N << "] = A[" << M << "," << K
              << "] x B[" << K << "," << N << "]\n";
    std::cout << "\n" << std::string(90, '-') << "\n";
    std::cout << std::setw(25) << std::left << "Configuration"
              << std::setw(12) << "Instructions"
              << std::setw(15) << "Ext Traffic"
              << std::setw(15) << "Arith Intens"
              << std::setw(12) << "Gen Time\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& exp : experiments) {
        OutputStationaryProgramBuilder::Config config;
        config.M = M;
        config.N = N;
        config.K = K;
        config.Ti = exp.Ti;
        config.Tj = exp.Tj;
        config.Tk = exp.Tk;
        config.L1_Ki = 16;
        config.systolic_size = 16;
        config.element_size = 4;

        config.l3_tile_capacity = 128 * 1024;
        config.l2_bank_capacity = 64 * 1024;
        config.l1_buffer_capacity = 32 * 1024;
        config.num_l3_tiles = 4;
        config.num_l2_banks = 8;
        config.num_l1_buffers = 4;
        config.double_buffer = true;

        OutputStationaryProgramBuilder builder(config);

        auto start = std::chrono::high_resolution_clock::now();
        DMProgram program = builder.build();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << std::setw(25) << std::left << exp.description
                  << std::setw(12) << std::right << program.instructions.size()
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << (program.estimates.external_mem_bytes / (1024.0 * 1024.0)) << " MB"
                  << std::setw(12) << program.estimates.arithmetic_intensity << " F/B"
                  << std::setw(10) << duration.count() << " us\n";
    }
    std::cout << std::string(90, '-') << "\n";
}

// ============================================================================
// Example 4: Concurrent Resource Execution
// ============================================================================

void example_concurrent_execution() {
    print_separator("Example 4: Concurrent Resource Execution");

    std::cout << R"(
The KPU has multiple hardware resources that execute CONCURRENTLY:
  - Multiple DMA engines (one per memory channel)
  - Multiple BlockMovers (L3 -> L2)
  - Multiple Streamers (L2 -> L1)
  - Compute fabric (systolic array)

The previous sequential instruction trace is misleading because it doesn't
show the true parallelism. This example uses the ConcurrentExecutor to
schedule operations onto resources and visualize their occupancy over time.
)";

    // Configure a moderate-sized matmul
    OutputStationaryProgramBuilder::Config config;
    config.M = 128;
    config.N = 128;
    config.K = 128;
    config.Ti = 32;
    config.Tj = 32;
    config.Tk = 32;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = true;

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    std::cout << "\nProgram: " << program.name << "\n";
    std::cout << "Instructions: " << program.instructions.size() << "\n\n";

    // Configure hardware resources using default ResourceConfig
    // Clock domains: DMA @ 250 MHz, BM/STR @ 500 MHz, Compute @ 2 GHz
    // All use 64-byte (512-bit) buses for cache-line aligned transfers
    ResourceConfig hw_config;  // Uses defaults from ResourceConfig

    std::cout << "Hardware Configuration:\n";
    std::cout << "\n  Clock Domains:\n";
    std::cout << "    DMA/L3:     " << hw_config.dma_clock_mhz << " MHz ("
              << (1000.0 / hw_config.dma_clock_mhz) << " ns/cycle)\n";
    std::cout << "    BM/L2:      " << hw_config.block_mover_clock_mhz << " MHz ("
              << (1000.0 / hw_config.block_mover_clock_mhz) << " ns/cycle)\n";
    std::cout << "    STR/L1:     " << hw_config.streamer_clock_mhz << " MHz ("
              << (1000.0 / hw_config.streamer_clock_mhz) << " ns/cycle)\n";
    std::cout << "    Compute:    " << hw_config.compute_clock_mhz << " MHz ("
              << (1000.0 / hw_config.compute_clock_mhz) << " ns/cycle)\n";
    std::cout << "\n  Resources:\n";
    std::cout << "    DMA engines:   " << (int)hw_config.num_memory_channels
              << " @ " << hw_config.dma_bandwidth_gb_s << " GB/s each ("
              << hw_config.dma_bus_width_bytes << "-byte bus)\n";
    std::cout << "    Block movers:  " << (int)hw_config.num_block_movers
              << " @ " << hw_config.block_mover_bandwidth_gb_s << " GB/s each (L3→L2)\n";
    std::cout << "    Streamers:     " << (int)hw_config.num_streamers
              << " @ " << hw_config.streamer_bandwidth_gb_s << " GB/s each (L2→L1)\n";
    std::cout << "\n  Aggregate Bandwidth:\n";
    std::cout << "    External:   " << (hw_config.num_memory_channels * hw_config.dma_bandwidth_gb_s)
              << " GB/s (4 ch × 16 GB/s)\n";
    std::cout << "    L3→L2:      " << (hw_config.num_block_movers * hw_config.block_mover_bandwidth_gb_s)
              << " GB/s (4 BM × 32 GB/s)\n";
    std::cout << "    L2→L1:      " << (hw_config.num_streamers * hw_config.streamer_bandwidth_gb_s)
              << " GB/s (4 STR × 32 GB/s)\n";

    // Execute with concurrent model
    ConcurrentExecutor executor(hw_config);
    Cycle total_cycles = executor.execute(program);

    std::cout << "\nExecution complete in " << total_cycles << " cycles\n";

    // Show utilization stats
    auto stats = executor.get_utilization();
    std::cout << "\nResource Utilization:\n";
    std::cout << "  DMA engines:   " << std::fixed << std::setprecision(1)
              << (stats.dma_utilization * 100) << "%\n";
    std::cout << "  Block movers:  " << (stats.block_mover_utilization * 100) << "%\n";
    std::cout << "  Streamers:     " << (stats.streamer_utilization * 100) << "%\n";

    // Generate timeline visualization
    std::cout << executor.generate_timeline(100);

    // Generate occupancy table
    std::cout << executor.generate_cycle_report();

    // Show cycle-by-cycle view covering first DMA->BM->STR pipeline
    // With LPDDR5X bandwidth: 4096 bytes / 12.8 GB/s = 320 cycles for DMA
    // BM: 4096 / 64 = 64 cycles, STR: 4096 / 128 = 32 cycles
    // Show first 500 cycles to see full pipeline activity
    std::cout << "\nDetailed cycle-by-cycle view (first iteration pipeline):\n";
    std::cout << TimelineFormatter::format_cycle_view(
        executor.get_all_operations(), hw_config, 0, 500);
}

// ============================================================================
// Example 5: Output-Stationary Loop Structure Visualization
// ============================================================================

void example_loop_structure() {
    print_separator("Example 5: Output-Stationary Loop Structure");

    std::cout << R"(
Output-Stationary Dataflow for MatMul C[M,N] = A[M,K] x B[K,N]:

The key insight is that C tiles stay in PE accumulators throughout the
K-reduction loop. This eliminates intermediate C writebacks and maximizes
compute density.

Loop Structure:
    for ti = 0 to M/Ti:             // Output row tiles (outer)
      for tj = 0 to N/Tj:           // Output col tiles
        // C[ti,tj] accumulates in PE registers - NO WRITEBACK
        for tk = 0 to K/Tk:         // Reduction tiles (inner)
          DMA_LOAD A[ti,tk]         // Load A tile from external memory
          DMA_LOAD B[tk,tj]         // Load B tile from external memory
          BM_MOVE A[ti,tk] L3->L2   // Move to L2
          BM_MOVE B[tk,tj] L3->L2
          STR_ROWS A[ti,tk]         // Stream A rows to systolic array
          STR_COLS B[tk,tj]         // Stream B cols to systolic array
          // COMPUTE happens reactively when data arrives at PEs!
        STR_DRAIN C[ti,tj]          // Drain accumulated result
        DMA_STORE C[ti,tj]          // Store to external memory

Reuse Pattern:
  - A[ti,*] is reused across all tj (N/Tj times)
  - B[*,tj] is reused across all ti (M/Ti times)
  - C[ti,tj] accumulates K/Tk partial products before writeback

This is why output-stationary excels when:
  - K is large (many accumulations amortize C writeback)
  - M and N are balanced (good reuse of both A and B)
)";

    // Generate a small program to show the structure
    OutputStationaryProgramBuilder::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 64;  // K > M,N to show accumulation
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;  // 4 reduction tiles
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = false;

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    std::cout << "\nGenerated program for C[32,32] = A[32,64] x B[64,32]:\n";
    std::cout << "Tiles: 2x2 output tiles, 4 reduction tiles\n";
    std::cout << "Shows 4 K-iterations per output tile accumulation.\n\n";

    print_instruction_trace(program, 60);
}

// ============================================================================
// Example 6: Tile Caching Demonstration
// ============================================================================

void example_tile_caching() {
    print_separator("Example 6: Tile Caching Benefits");

    std::cout << R"(
This example demonstrates L3 tile caching, which eliminates redundant DMA
transfers when tiles are reused across loop iterations.

In output-stationary dataflow:
  - A[ti,tk] is reused across all tj iterations (N/Tj times)
  - B[tk,tj] is reused across all ti iterations (M/Ti times)

Without caching, we reload tiles on every access.
With caching, we only load each unique tile once.
)";

    // Common configuration
    OutputStationaryProgramBuilder::Config config;
    config.M = 128;
    config.N = 128;
    config.K = 128;
    config.Ti = 32;
    config.Tj = 32;
    config.Tk = 32;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;
    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = false;

    // Calculate expected values
    Size m_tiles = (config.M + config.Ti - 1) / config.Ti;  // 4
    Size n_tiles = (config.N + config.Tj - 1) / config.Tj;  // 4
    Size k_tiles = (config.K + config.Tk - 1) / config.Tk;  // 4

    Size a_unique_tiles = m_tiles * k_tiles;  // 16 unique A tiles
    Size b_unique_tiles = k_tiles * n_tiles;  // 16 unique B tiles
    Size total_unique = a_unique_tiles + b_unique_tiles;  // 32 unique tiles

    // Without caching: every iteration loads
    Size total_iterations = m_tiles * n_tiles * k_tiles;  // 64
    Size loads_without_cache = total_iterations * 2;  // 128 loads (A + B each iter)

    std::cout << "\nMatrix: C[" << config.M << "," << config.N << "] = "
              << "A[" << config.M << "," << config.K << "] x "
              << "B[" << config.K << "," << config.N << "]\n";
    std::cout << "Tiles: " << m_tiles << "x" << n_tiles << "x" << k_tiles
              << " = " << total_iterations << " iterations\n\n";

    std::cout << "Expected tile counts:\n";
    std::cout << "  Unique A tiles (ti × tk): " << a_unique_tiles << "\n";
    std::cout << "  Unique B tiles (tk × tj): " << b_unique_tiles << "\n";
    std::cout << "  Total unique tiles:       " << total_unique << "\n";
    std::cout << "  Without caching (loads):  " << loads_without_cache << "\n";
    std::cout << "  Potential savings:        " << (loads_without_cache - total_unique)
              << " redundant loads avoided\n\n";

    // Build WITH caching (default)
    std::cout << "--- WITH Tile Caching (default) ---\n";
    config.enable_tile_caching = true;
    OutputStationaryProgramBuilder builder_cached(config);
    DMProgram program_cached = builder_cached.build();

    std::cout << "  DMA operations:    " << program_cached.num_dma_ops() << "\n";
    std::cout << "  External traffic:  " << std::fixed << std::setprecision(2)
              << (program_cached.estimates.external_mem_bytes / 1024.0) << " KB\n";
    std::cout << builder_cached.get_cache_stats();

    // Build WITHOUT caching
    std::cout << "\n--- WITHOUT Tile Caching ---\n";
    config.enable_tile_caching = false;
    OutputStationaryProgramBuilder builder_uncached(config);
    DMProgram program_uncached = builder_uncached.build();

    std::cout << "  DMA operations:    " << program_uncached.num_dma_ops() << "\n";
    std::cout << "  External traffic:  " << std::fixed << std::setprecision(2)
              << (program_uncached.estimates.external_mem_bytes / 1024.0) << " KB\n";

    // Summary comparison
    std::cout << "\n--- Comparison ---\n";
    Size bytes_saved = program_uncached.estimates.external_mem_bytes -
                       program_cached.estimates.external_mem_bytes;
    Size dma_saved = program_uncached.num_dma_ops() - program_cached.num_dma_ops();

    std::cout << "  DMA ops reduced:     " << program_uncached.num_dma_ops()
              << " -> " << program_cached.num_dma_ops()
              << " (" << dma_saved << " fewer, "
              << std::setprecision(1) << (100.0 * dma_saved / program_uncached.num_dma_ops())
              << "% reduction)\n";
    std::cout << "  External traffic:    " << std::setprecision(2)
              << (program_uncached.estimates.external_mem_bytes / 1024.0) << " KB -> "
              << (program_cached.estimates.external_mem_bytes / 1024.0) << " KB ("
              << (bytes_saved / 1024.0) << " KB saved)\n";

    // Calculate minimum traffic
    Size min_bytes = (config.M * config.K + config.K * config.N +
                     config.M * config.N) * config.element_size;
    double reuse_cached = static_cast<double>(program_cached.estimates.external_mem_bytes) / min_bytes;
    double reuse_uncached = static_cast<double>(program_uncached.estimates.external_mem_bytes) / min_bytes;

    std::cout << "  Reuse factor:        " << std::setprecision(2) << reuse_uncached
              << "x -> " << reuse_cached << "x (1.0x is optimal)\n";
    std::cout << "  Arith. intensity:    " << std::setprecision(1)
              << (2.0 * config.M * config.N * config.K / program_uncached.estimates.external_mem_bytes)
              << " -> "
              << (2.0 * config.M * config.N * config.K / program_cached.estimates.external_mem_bytes)
              << " FLOPs/byte\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << R"(
================================================================================
               Data Movement ISA for Domain Flow Architecture
================================================================================

In Domain Flow Architecture, the program IS the data movement schedule.
The compute fabric is reactive - it executes when data tokens arrive.
The intelligence is in orchestrating optimal data movement patterns
derived from SURE (Space-time Uniform Recurrence Equation) analysis.

This example demonstrates:
  1. Building Data Movement programs with OutputStationaryProgramBuilder
  2. Analyzing generated instruction sequences
  3. Understanding traffic estimates and arithmetic intensity
  4. Comparing different tiling strategies
  5. Tile caching for eliminating redundant DMA transfers

Note: This ISA-based approach represents how actual programs execute
on the KPU, unlike direct component API calls which are useful for
testing individual hardware blocks.
================================================================================
)";

    example_small_matmul();
    example_large_matmul_double_buffered();
    example_tile_size_comparison();
    example_concurrent_execution();
    example_loop_structure();
    example_tile_caching();

    print_separator("Summary");
    std::cout << R"(
The Data Movement ISA provides:

1. OPCODES for configuring data movement hardware:
   - DMA_LOAD_TILE / DMA_STORE_TILE: External memory <-> L3
   - BM_MOVE_TILE / BM_TRANSPOSE_TILE: L3 <-> L2 with transforms
   - STR_FEED_ROWS / STR_FEED_COLS: L2 -> L1 systolic feeding
   - STR_DRAIN_OUTPUT: L1 -> L2 result collection
   - BARRIER: Synchronization

2. PROGRAM BUILDER for automatic schedule generation:
   - OutputStationaryProgramBuilder: C stays in PEs
   - (Future) WeightStationaryProgramBuilder: B stays in PEs
   - (Future) InputStationaryProgramBuilder: A stays in PEs

3. PROGRAM EXECUTOR for hardware simulation:
   - Maps ISA instructions to hardware components
   - Cycle-accurate execution tracking
   - Performance metric collection

4. CONCURRENT EXECUTOR for true parallel execution:
   - Multiple DMA engines (one per memory channel)
   - Multiple BlockMovers and Streamers
   - Resource occupancy visualization
   - Timeline/Gantt chart generation

Next steps:
  - Implement weight-stationary and input-stationary builders
  - Add prefetching and advanced double-buffering
  - Connect to DFX compiler for end-to-end flow
)";

    return 0;
}
