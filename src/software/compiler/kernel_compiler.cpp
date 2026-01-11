// Kernel Compiler implementation for KPU simulator
// Provides high-level compilation with automatic tile optimization

#include <sw/compiler/kernel_compiler.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace sw::kpu::compiler {

// ============================================================================
// Helper functions for formatting
// ============================================================================

namespace {

std::string format_bytes_short(Size bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    return std::to_string(bytes) + " B";
}

std::string format_bytes_precise(Size bytes) {
    std::ostringstream ss;
    if (bytes >= 1024 * 1024) {
        ss << std::fixed << std::setprecision(1) << (bytes / (1024.0 * 1024.0)) << " MB";
    } else if (bytes >= 1024) {
        ss << std::fixed << std::setprecision(1) << (bytes / 1024.0) << " KB";
    } else {
        ss << bytes << " B";
    }
    return ss.str();
}

} // anonymous namespace

// ============================================================================
// OperationBreakdown Implementation
// ============================================================================

void OperationBreakdown::compute_bandwidth(double clock_ghz) {
    if (estimated_cycles == 0) return;

    double time_seconds = estimated_cycles / (clock_ghz * 1e9);

    // Compute achieved bandwidth (bytes / time = bytes/second, then to GB/s)
    bandwidth.external_gbps = (external_memory.total_bytes / time_seconds) / 1e9;
    bandwidth.l3_l2_gbps = (l3_l2.total_bytes / time_seconds) / 1e9;
    bandwidth.l2_l1_gbps = (l2_l1.total_bytes / time_seconds) / 1e9;

    // Compute utilization as fraction of peak
    // Peak bandwidth is in bytes/cycle, so peak GB/s = peak_bw * clock_ghz
    double external_peak_gbps = pipeline.external_peak_bw * clock_ghz;
    double l3_l2_peak_gbps = pipeline.l3_l2_peak_bw * clock_ghz;
    double l2_l1_peak_gbps = pipeline.l2_l1_peak_bw * clock_ghz;

    bandwidth.external_utilization = bandwidth.external_gbps / external_peak_gbps;
    bandwidth.l3_l2_utilization = bandwidth.l3_l2_gbps / l3_l2_peak_gbps;
    bandwidth.l2_l1_utilization = bandwidth.l2_l1_gbps / l2_l1_peak_gbps;
}

std::string OperationBreakdown::summary() const {
    std::ostringstream ss;

    ss << "Operation Breakdown:\n";
    ss << std::left;
    ss << "  " << std::setw(22) << "Level"
       << std::setw(10) << "Count"
       << std::setw(12) << "Volume"
       << std::setw(12) << "Avg Size"
       << std::setw(14) << "Latency/Op" << "\n";
    ss << "  " << std::string(68, '-') << "\n";

    // External ↔ L3 (DMA)
    ss << "  " << std::setw(22) << "External <-> L3 (DMA)"
       << std::setw(10) << external_memory.count
       << std::setw(12) << format_bytes_short(external_memory.total_bytes)
       << std::setw(12) << format_bytes_short(external_memory.avg_bytes_per_op)
       << "~" << std::setw(13) << std::to_string(external_memory.avg_latency_cycles) + " cyc" << "\n";

    // L3 ↔ L2 (Block Mover)
    ss << "  " << std::setw(22) << "L3 <-> L2 (BlockMover)"
       << std::setw(10) << l3_l2.count
       << std::setw(12) << format_bytes_short(l3_l2.total_bytes)
       << std::setw(12) << format_bytes_short(l3_l2.avg_bytes_per_op)
       << "~" << std::setw(13) << std::to_string(l3_l2.avg_latency_cycles) + " cyc" << "\n";

    // L2 ↔ L1 (Streamer)
    ss << "  " << std::setw(22) << "L2 <-> L1 (Streamer)"
       << std::setw(10) << l2_l1.count
       << std::setw(12) << format_bytes_short(l2_l1.total_bytes)
       << std::setw(12) << format_bytes_short(l2_l1.avg_bytes_per_op)
       << "~" << std::setw(13) << std::to_string(l2_l1.avg_latency_cycles) + " cyc" << "\n";

    ss << "\n  Pipeline Resources:\n";
    ss << "    DMA Channels:    " << pipeline.dma_channels
       << "  (concurrent transfers)\n";
    ss << "    Block Movers:    " << pipeline.block_movers
       << "  (concurrent transfers)\n";
    ss << "    Streamers:       " << pipeline.streamers
       << "  (concurrent transfers)\n";

    if (estimated_cycles > 0) {
        ss << "\n  Achieved Bandwidth (@ 1 GHz):\n";
        ss << std::fixed << std::setprecision(1);
        ss << "    External Memory: " << std::setw(6) << bandwidth.external_gbps << " GB/s  ("
           << std::setw(3) << static_cast<int>(bandwidth.external_utilization * 100) << "% of "
           << pipeline.external_peak_bw << " GB/s peak)\n";
        ss << "    L3 <-> L2:       " << std::setw(6) << bandwidth.l3_l2_gbps << " GB/s  ("
           << std::setw(3) << static_cast<int>(bandwidth.l3_l2_utilization * 100) << "% of "
           << pipeline.l3_l2_peak_bw << " GB/s peak)\n";
        ss << "    L2 <-> L1:       " << std::setw(6) << bandwidth.l2_l1_gbps << " GB/s  ("
           << std::setw(3) << static_cast<int>(bandwidth.l2_l1_utilization * 100) << "% of "
           << pipeline.l2_l1_peak_bw << " GB/s peak)\n";
    }

    return ss.str();
}

// ============================================================================
// CompilationStats Implementation
// ============================================================================

std::string CompilationStats::summary() const {
    std::ostringstream ss;

    ss << "Compilation Statistics:\n";
    ss << "  Compile Time: " << std::fixed << std::setprecision(1)
       << compile_time_us << " us\n";

    ss << "\n  Tile Configuration:\n";
    ss << "    Auto-optimized: " << (used_auto_tiling ? "yes" : "no") << "\n";
    ss << "    Ti=" << selected_Ti << ", Tj=" << selected_Tj
       << ", Tk=" << selected_Tk << ", L1_Ki=" << selected_L1_Ki << "\n";
    ss << "    Tiles: " << num_m_tiles << " x " << num_n_tiles << " x "
       << num_k_tiles << " = " << total_tiles << " total\n";

    ss << "\n" << operations.summary();

    ss << "\n  Memory Traffic Estimates:\n";
    ss << "    External (DRAM): " << format_bytes_precise(estimated_external_bytes) << "\n";
    ss << "    L3 Cache:        " << format_bytes_precise(estimated_l3_bytes) << "\n";
    ss << "    L2 Cache:        " << format_bytes_precise(estimated_l2_bytes) << "\n";
    ss << "    Arithmetic Intensity: " << std::fixed << std::setprecision(2)
       << estimated_arithmetic_intensity << " FLOPs/byte\n";

    ss << "\n  Dataflow: " << dataflow_strategy_name(dataflow_used) << "\n";

    return ss.str();
}

// ============================================================================
// KernelCompiler Implementation
// ============================================================================

KernelCompiler::KernelCompiler()
    : tile_optimizer_(TileOptimizer::MemoryHierarchy{}) {
}

KernelCompiler::KernelCompiler(const TileOptimizer::MemoryHierarchy& memory)
    : tile_optimizer_(memory) {
}

// ============================================================================
// Main Compilation API
// ============================================================================

Kernel KernelCompiler::compile_matmul(Size M, Size N, Size K,
                                       const CompileOptions& options) {
    auto start_time = std::chrono::high_resolution_clock::now();

    last_succeeded_ = false;
    last_error_.clear();
    last_stats_ = CompilationStats{};

    // Step 1: Determine tile sizes
    TileOptimizer::TileConfig tile_config;

    if (options.is_auto_tiling()) {
        // Use TileOptimizer for automatic tile selection
        tile_config = tile_optimizer_.optimize(M, N, K, options.tile_strategy);
        last_stats_.used_auto_tiling = true;

        if (!tile_config.valid) {
            last_error_ = "Tile optimization failed: " + tile_config.reason;
            return Kernel{};
        }
    } else {
        // Use explicit tile sizes from options
        tile_config.Ti = options.Ti;
        tile_config.Tj = options.Tj;
        tile_config.Tk = options.Tk;
        tile_config.L1_Ki = options.L1_Ki > 0 ? options.L1_Ki : options.Tk;
        tile_config.valid = true;
        last_stats_.used_auto_tiling = false;
    }

    // Store tile sizes in stats
    last_stats_.selected_Ti = tile_config.Ti;
    last_stats_.selected_Tj = tile_config.Tj;
    last_stats_.selected_Tk = tile_config.Tk;
    last_stats_.selected_L1_Ki = tile_config.L1_Ki;

    // Step 2: Build program configuration
    isa::OutputStationaryProgramBuilder::Config prog_config =
        build_program_config(M, N, K, tile_config, options);

    // Step 3: Generate program
    isa::OutputStationaryProgramBuilder builder(prog_config);
    isa::DMProgram program = builder.build();

    // Calculate tile counts first (needed for operation counting)
    auto ceil_div = [](Size a, Size b) { return (a + b - 1) / b; };
    last_stats_.num_m_tiles = ceil_div(M, tile_config.Ti);
    last_stats_.num_n_tiles = ceil_div(N, tile_config.Tj);
    last_stats_.num_k_tiles = ceil_div(K, tile_config.Tk);
    last_stats_.total_tiles = last_stats_.num_m_tiles *
                              last_stats_.num_n_tiles *
                              last_stats_.num_k_tiles;

    // Step 4: Count operations and record stats
    Size elem_size = dtype_size(options.dtype);
    count_operations(program, elem_size, tile_config);

    // Estimate memory traffic
    Size A_bytes = M * K * elem_size;
    Size B_bytes = K * N * elem_size;
    Size C_bytes = M * N * elem_size;

    // With tile caching, A and B are reused
    // A is loaded N/Tj times (once per column strip)
    // B is loaded M/Ti times (once per row strip)
    // With perfect caching, each is loaded once
    if (options.enable_tile_caching) {
        last_stats_.estimated_external_bytes = A_bytes + B_bytes + C_bytes;
    } else {
        last_stats_.estimated_external_bytes =
            A_bytes * last_stats_.num_n_tiles +
            B_bytes * last_stats_.num_m_tiles +
            C_bytes;
    }

    // L3 traffic (moving to L2)
    last_stats_.estimated_l3_bytes = last_stats_.estimated_external_bytes;

    // L2 traffic (streaming to L1)
    last_stats_.estimated_l2_bytes = last_stats_.total_tiles *
        (tile_config.Ti * tile_config.Tk +
         tile_config.Tk * tile_config.Tj +
         tile_config.Ti * tile_config.Tj) * elem_size;

    // Arithmetic intensity
    Size total_flops = 2 * M * N * K;
    last_stats_.estimated_arithmetic_intensity =
        static_cast<double>(total_flops) /
        static_cast<double>(last_stats_.estimated_external_bytes);

    // Dataflow used
    last_stats_.dataflow_used = (options.dataflow == DataflowStrategy::AUTO)
        ? select_dataflow(M, N, K) : options.dataflow;

    // Compile time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    last_stats_.compile_time_us = static_cast<double>(duration.count());

    last_succeeded_ = true;

    // Create and return kernel
    return Kernel(std::move(program), KernelOpType::MATMUL, options.dtype);
}

Kernel KernelCompiler::compile_matmul(Size M, Size N, Size K,
                                       Size Ti, Size Tj, Size Tk) {
    CompileOptions opts = CompileOptions::with_tiles(Ti, Tj, Tk);
    return compile_matmul(M, N, K, opts);
}

Kernel KernelCompiler::compile_matmul(Size M, Size N, Size K,
                                       Size Ti, Size Tj, Size Tk, Size L1_Ki) {
    CompileOptions opts = CompileOptions::with_tiles(Ti, Tj, Tk);
    opts.L1_Ki = L1_Ki;
    return compile_matmul(M, N, K, opts);
}

Kernel KernelCompiler::compile_mlp(Size M, Size N, Size K,
                                    ActivationType activation,
                                    bool has_bias,
                                    DataType dtype,
                                    const CompileOptions& options) {
    auto start_time = std::chrono::high_resolution_clock::now();

    last_succeeded_ = false;
    last_error_.clear();
    last_stats_ = CompilationStats{};

    // Step 1: Determine tile sizes (same as matmul)
    TileOptimizer::TileConfig tile_config;

    if (options.is_auto_tiling()) {
        tile_config = tile_optimizer_.optimize(M, N, K, options.tile_strategy);
        last_stats_.used_auto_tiling = true;

        if (!tile_config.valid) {
            last_error_ = "Tile optimization failed: " + tile_config.reason;
            return Kernel{};
        }
    } else {
        tile_config.Ti = options.Ti;
        tile_config.Tj = options.Tj;
        tile_config.Tk = options.Tk;
        tile_config.L1_Ki = options.L1_Ki > 0 ? options.L1_Ki : options.Tk;
        tile_config.valid = true;
        last_stats_.used_auto_tiling = false;
    }

    // Store tile sizes in stats
    last_stats_.selected_Ti = tile_config.Ti;
    last_stats_.selected_Tj = tile_config.Tj;
    last_stats_.selected_Tk = tile_config.Tk;
    last_stats_.selected_L1_Ki = tile_config.L1_Ki;

    // Step 2: Build program configuration
    CompileOptions opts = options;
    opts.dtype = dtype;
    isa::OutputStationaryProgramBuilder::Config prog_config =
        build_program_config(M, N, K, tile_config, opts);

    // Step 3: Generate program
    // Note: For now, we generate the same program as matmul.
    // The VE configuration is stored in the kernel metadata.
    // Future: Modify OutputStationaryProgramBuilder to emit VE-enabled drain ops.
    isa::OutputStationaryProgramBuilder builder(prog_config);
    isa::DMProgram program = builder.build();

    // Update program name to indicate MLP
    std::ostringstream name_ss;
    name_ss << "mlp_" << M << "x" << N << "x" << K;
    if (has_bias) {
        name_ss << "_bias";
    }
    name_ss << "_" << activation_type_name(activation);
    program.name = name_ss.str();

    // Calculate tile counts first (needed for operation counting)
    auto ceil_div = [](Size a, Size b) { return (a + b - 1) / b; };
    last_stats_.num_m_tiles = ceil_div(M, tile_config.Ti);
    last_stats_.num_n_tiles = ceil_div(N, tile_config.Tj);
    last_stats_.num_k_tiles = ceil_div(K, tile_config.Tk);
    last_stats_.total_tiles = last_stats_.num_m_tiles *
                              last_stats_.num_n_tiles *
                              last_stats_.num_k_tiles;

    // Step 4: Count operations and record stats
    Size elem_size = dtype_size(dtype);
    count_operations(program, elem_size, tile_config);

    // Estimate memory traffic (MLP saves traffic via fusion)
    Size A_bytes = M * K * elem_size;
    Size B_bytes = K * N * elem_size;
    Size C_bytes = M * N * elem_size;
    Size bias_bytes = has_bias ? N * elem_size : 0;

    // With VE fusion, we avoid extra memory passes for bias+activation
    last_stats_.estimated_external_bytes = A_bytes + B_bytes + C_bytes + bias_bytes;
    last_stats_.estimated_l3_bytes = last_stats_.estimated_external_bytes;

    // L2 traffic
    last_stats_.estimated_l2_bytes = last_stats_.total_tiles *
        (tile_config.Ti * tile_config.Tk +
         tile_config.Tk * tile_config.Tj +
         tile_config.Ti * tile_config.Tj) * elem_size;

    // Arithmetic intensity (MLP has slightly higher compute per byte)
    Size total_flops = 2 * M * N * K;  // matmul
    if (has_bias) total_flops += M * N;  // bias add
    if (activation != ActivationType::NONE) total_flops += M * N;  // activation
    last_stats_.estimated_arithmetic_intensity =
        static_cast<double>(total_flops) /
        static_cast<double>(last_stats_.estimated_external_bytes);

    last_stats_.dataflow_used = (options.dataflow == DataflowStrategy::AUTO)
        ? select_dataflow(M, N, K) : options.dataflow;

    // Compile time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    last_stats_.compile_time_us = static_cast<double>(duration.count());

    last_succeeded_ = true;

    // Create MLP kernel using the specialized constructor
    return Kernel(std::move(program), dtype, activation, has_bias);
}

// ============================================================================
// Tile Optimization
// ============================================================================

TileOptimizer::TileConfig KernelCompiler::optimize_tiles(
    Size M, Size N, Size K, TileOptimizer::Strategy strategy) {

    return tile_optimizer_.optimize(M, N, K, strategy);
}

// ============================================================================
// Memory Hierarchy Configuration
// ============================================================================

void KernelCompiler::set_memory_hierarchy(const TileOptimizer::MemoryHierarchy& memory) {
    tile_optimizer_.set_memory_hierarchy(memory);
}

// ============================================================================
// Private Methods
// ============================================================================

isa::OutputStationaryProgramBuilder::Config KernelCompiler::build_program_config(
    Size M, Size N, Size K,
    const TileOptimizer::TileConfig& tiles,
    const CompileOptions& options) {

    isa::OutputStationaryProgramBuilder::Config config;

    // Matrix dimensions
    config.M = M;
    config.N = N;
    config.K = K;

    // Tile sizes from optimization
    config.Ti = tiles.Ti;
    config.Tj = tiles.Tj;
    config.Tk = tiles.Tk;
    config.L1_Ki = tiles.L1_Ki;

    // Hardware configuration
    config.systolic_size = options.systolic_size;
    config.element_size = dtype_size(options.dtype);

    // Memory hierarchy - use options if provided, else defaults
    const auto& mem = tile_optimizer_.memory_hierarchy();

    config.l3_tile_capacity = (options.l3_tile_capacity > 0)
        ? options.l3_tile_capacity : mem.L3_size;

    config.l2_bank_capacity = (options.l2_bank_capacity > 0)
        ? options.l2_bank_capacity : mem.L2_size;

    config.l1_buffer_capacity = (options.l1_buffer_capacity > 0)
        ? options.l1_buffer_capacity : mem.L1_size;

    config.num_l3_tiles = (options.num_l3_tiles > 0)
        ? options.num_l3_tiles : static_cast<uint8_t>(mem.L3_tile_count);

    config.num_l2_banks = (options.num_l2_banks > 0)
        ? options.num_l2_banks : static_cast<uint8_t>(mem.L2_bank_count);

    config.num_l1_buffers = (options.num_l1_buffers > 0)
        ? options.num_l1_buffers : static_cast<uint8_t>(mem.L1_buffer_count);

    // Execution flags
    config.double_buffer = options.double_buffer;
    config.enable_tile_caching = options.enable_tile_caching;

    return config;
}

DataflowStrategy KernelCompiler::select_dataflow(Size M, Size N, Size K) const {
    // Heuristic for dataflow selection:
    // - Output-stationary: balanced M, N, K (general purpose)
    // - Weight-stationary: large M (batch inference), small K*N (weights)
    // - Input-stationary: large N, small M*K

    // For now, always use output-stationary as it's the most general
    // Future: add more sophisticated selection based on problem shape

    // Rough heuristic:
    // If M >> N and M >> K, consider input-stationary
    // If N >> M and N >> K, consider weight-stationary
    // Otherwise, output-stationary

    return DataflowStrategy::OUTPUT_STATIONARY;
}

void KernelCompiler::count_operations(const isa::DMProgram& program,
                                       Size elem_size,
                                       const TileOptimizer::TileConfig& tiles) {
    // Reset operation breakdown
    last_stats_.operations = OperationBreakdown{};
    auto& ops = last_stats_.operations;

    // Count operations and estimate bytes per operation type
    Size dma_tile_bytes = tiles.Ti * tiles.Tk * elem_size;  // Typical A tile
    Size bm_tile_bytes = tiles.Ti * tiles.Tj * elem_size;   // Typical C tile
    Size str_row_bytes = tiles.Tj * elem_size;              // One row to/from systolic

    for (const auto& instr : program.instructions) {
        switch (instr.opcode) {
            case isa::DMOpcode::DMA_LOAD_TILE:
            case isa::DMOpcode::DMA_STORE_TILE:
            case isa::DMOpcode::DMA_PREFETCH_TILE:
                ops.external_memory.count++;
                ops.external_memory.total_bytes += dma_tile_bytes;
                break;
            case isa::DMOpcode::BM_MOVE_TILE:
            case isa::DMOpcode::BM_TRANSPOSE_TILE:
            case isa::DMOpcode::BM_WRITEBACK_TILE:
            case isa::DMOpcode::BM_RESHAPE_TILE:
                ops.l3_l2.count++;
                ops.l3_l2.total_bytes += bm_tile_bytes;
                break;
            case isa::DMOpcode::STR_FEED_ROWS:
            case isa::DMOpcode::STR_FEED_COLS:
            case isa::DMOpcode::STR_DRAIN_OUTPUT:
            case isa::DMOpcode::STR_BROADCAST_ROW:
            case isa::DMOpcode::STR_BROADCAST_COL:
                ops.l2_l1.count++;
                ops.l2_l1.total_bytes += str_row_bytes;
                break;
            default:
                // NOP, BARRIER, WAIT_*, SIGNAL, SET_*, LOOP_*, HALT
                break;
        }
    }

    // Finalize averages
    ops.external_memory.finalize();
    ops.l3_l2.finalize();
    ops.l2_l1.finalize();

    // Estimate latencies based on typical operation characteristics
    // DMA: High latency due to DRAM access (~100-500 cycles for burst, depends on size)
    // Using a simple model: base_latency + bytes / bandwidth
    ops.external_memory.avg_latency_cycles = 100 + ops.external_memory.avg_bytes_per_op / 64;
    ops.l3_l2.avg_latency_cycles = 20 + ops.l3_l2.avg_bytes_per_op / 128;
    ops.l2_l1.avg_latency_cycles = 4 + ops.l2_l1.avg_bytes_per_op / 256;

    // Update legacy fields for backward compatibility
    last_stats_.instruction_count = program.instructions.size();
    last_stats_.dma_ops = ops.external_memory.count;
    last_stats_.block_mover_ops = ops.l3_l2.count;
    last_stats_.streamer_ops = ops.l2_l1.count;
    last_stats_.compute_ops = 0;  // Compute is implicit in streaming
}

} // namespace sw::kpu::compiler
