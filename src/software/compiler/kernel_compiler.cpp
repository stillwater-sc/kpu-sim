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
        ss << std::fixed << std::setprecision(1) << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
    } else if (bytes >= 1024) {
        ss << std::fixed << std::setprecision(1) << (static_cast<double>(bytes) / 1024.0) << " KB";
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

    double time_seconds = static_cast<double>(estimated_cycles) / (clock_ghz * 1e9);

    // Compute achieved bandwidth (bytes / time = bytes/second, then to GB/s)
    bandwidth.external_gbps = (static_cast<double>(external_memory.total_bytes) / time_seconds) / 1e9;
    bandwidth.l3_l2_gbps = (static_cast<double>(l3_l2.total_bytes) / time_seconds) / 1e9;
    bandwidth.l2_l1_gbps = (static_cast<double>(l2_l1.total_bytes) / time_seconds) / 1e9;

    // Compute utilization as fraction of peak
    // Peak bandwidth is in bytes/cycle, so peak GB/s = peak_bw * clock_ghz
    double external_peak_gbps = static_cast<double>(pipeline.external_peak_bw) * clock_ghz;
    double l3_l2_peak_gbps = static_cast<double>(pipeline.l3_l2_peak_bw) * clock_ghz;
    double l2_l1_peak_gbps = static_cast<double>(pipeline.l2_l1_peak_bw) * clock_ghz;

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

DataflowStrategy KernelCompiler::select_dataflow([[maybe_unused]] Size M, [[maybe_unused]] Size N, [[maybe_unused]] Size K) const {
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

// ============================================================================
// Per-op streaming compilers (issue #18/#92)
// ============================================================================

namespace {

isa::DMInstruction ve_marker(isa::DMOpcode opcode, std::string label) {
    isa::DMInstruction instr;
    instr.opcode = opcode;
    instr.operands = std::monostate{};
    instr.label = std::move(label);
    return instr;
}

Size ceil_div_sz(Size a, Size b) { return (a + b - 1) / b; }

} // namespace

void KernelCompiler::append_tile_loop(isa::DMProgram& program, uint8_t loop_id,
                                      Size count,
                                      const std::vector<isa::DMInstruction>& body) {
    constexpr Size kMaxLoopCount = 65535;  // LOOP_BEGIN count is uint16_t
    while (count > 0) {
        Size chunk = count < kMaxLoopCount ? count : kMaxLoopCount;
        program.instructions.push_back(isa::DMInstruction::loop_begin(
            loop_id, static_cast<uint16_t>(chunk)));
        for (const auto& instr : body) {
            program.instructions.push_back(instr);
        }
        program.instructions.push_back(isa::DMInstruction::loop_end(loop_id));
        count -= chunk;
    }
}

isa::DMProgram KernelCompiler::emit_streaming_program(
    const std::string& name,
    const std::vector<StreamingPass>& passes,
    Size tile_elems, Size elem_size,
    uint64_t total_flops, uint64_t external_bytes) {

    isa::DMProgram program;
    program.name = name;
    program.Ti = tile_elems;   // Streaming ops are 1-D tiled: Ti elements/tile
    program.Tj = 1;
    program.Tk = 1;

    // Configuration prologue: tile geometry and base addresses (actual
    // addresses are bound at load time via the memory map)
    program.instructions.push_back(
        isa::DMInstruction::set_tile_dim(tile_elems, 1, 1, elem_size));
    program.instructions.push_back(
        isa::DMInstruction::set_base(isa::MatrixID::A, 0));
    program.instructions.push_back(
        isa::DMInstruction::set_base(isa::MatrixID::C, 0));

    uint8_t loop_id = 0;
    for (const auto& pass : passes) {
        if (pass.input_tiles > 0) {
            std::vector<isa::DMInstruction> body = {
                isa::DMInstruction::dma_load_auto(isa::MatrixID::A, 0),
                isa::DMInstruction::bm_move_auto(isa::MatrixID::A, 0),
                isa::DMInstruction::str_feed_rows_auto(0),
            };
            if (pass.ve_opcode != isa::DMOpcode::NOP) {
                body.push_back(ve_marker(pass.ve_opcode, pass.label));
            }
            append_tile_loop(program, loop_id, pass.input_tiles, body);
            loop_id = static_cast<uint8_t>(loop_id + 1);
        }
        if (pass.output_tiles > 0) {
            std::vector<isa::DMInstruction> body = {
                isa::DMInstruction::str_drain_auto(0),
                isa::DMInstruction::bm_writeback_auto(isa::MatrixID::C, 0),
                isa::DMInstruction::dma_store_auto(isa::MatrixID::C, 0),
            };
            append_tile_loop(program, loop_id, pass.output_tiles, body);
            loop_id = static_cast<uint8_t>(loop_id + 1);
        }
        program.instructions.push_back(isa::DMInstruction::barrier());
    }
    program.instructions.push_back(isa::DMInstruction::halt());

    program.estimates.external_mem_bytes = external_bytes;
    program.estimates.l3_bytes = external_bytes;
    program.estimates.arithmetic_intensity = external_bytes > 0
        ? static_cast<double>(total_flops) / static_cast<double>(external_bytes)
        : 0.0;

    last_stats_ = CompilationStats{};
    last_stats_.instruction_count = program.instructions.size();
    last_stats_.estimated_external_bytes = program.estimates.external_mem_bytes;
    last_stats_.estimated_l3_bytes = program.estimates.l3_bytes;
    last_stats_.estimated_arithmetic_intensity =
        program.estimates.arithmetic_intensity;
    last_succeeded_ = true;
    last_error_.clear();
    return program;
}

Kernel KernelCompiler::compile_softmax(const SoftmaxConfig& config,
                                       const CompileOptions& options) {
    const Size tile_elems = 256;
    const Size elem_size = dtype_size(options.dtype);
    Size data_tiles = ceil_div_sz(config.total_elements(), tile_elems);
    Size stat_tiles = ceil_div_sz(config.num_softmax_ops(), tile_elems);

    // Numerically stable multi-pass softmax (the online single-pass form
    // is the E8 pattern epic)
    std::vector<StreamingPass> passes = {
        {"softmax P1: running row max (VE_REDUCE MAX)",
         isa::DMOpcode::VE_REDUCE, data_tiles, stat_tiles},
        {"softmax P2: exp(x - max) (VE_ELEMENTWISE SUB,EXP)",
         isa::DMOpcode::VE_ELEMENTWISE, data_tiles, data_tiles},
        {"softmax P3: row sum of exp (VE_REDUCE SUM)",
         isa::DMOpcode::VE_REDUCE, data_tiles, stat_tiles},
        {"softmax P4: normalize by sum (VE_ELEMENTWISE DIV)",
         isa::DMOpcode::VE_ELEMENTWISE, data_tiles, data_tiles},
    };

    uint64_t bytes = static_cast<uint64_t>(config.total_elements()) * elem_size;
    auto program = emit_streaming_program("softmax", passes, tile_elems,
                                          elem_size, config.total_flops(),
                                          4 * bytes);
    return Kernel(std::move(program), KernelOpType::SOFTMAX, options.dtype);
}

Kernel KernelCompiler::compile_layernorm(const LayerNormConfig& config,
                                         const CompileOptions& options) {
    const Size tile_elems = 256;
    const Size elem_size = dtype_size(options.dtype);
    Size data_tiles = ceil_div_sz(config.total_elements(), tile_elems);
    Size group_tiles = ceil_div_sz(config.num_groups(), tile_elems);

    std::vector<StreamingPass> passes = {
        {"layernorm P1: per-group mean (VE_REDUCE SUM)",
         isa::DMOpcode::VE_REDUCE, data_tiles, group_tiles},
        {"layernorm P2: per-group variance (VE_ELEMENTWISE SUB,SQ + VE_REDUCE SUM)",
         isa::DMOpcode::VE_REDUCE, data_tiles, group_tiles},
        {"layernorm P3: normalize + affine (VE_ELEMENTWISE)",
         isa::DMOpcode::VE_ELEMENTWISE, data_tiles, data_tiles},
    };

    uint64_t bytes = static_cast<uint64_t>(config.total_elements()) * elem_size;
    auto program = emit_streaming_program("layernorm", passes, tile_elems,
                                          elem_size, config.total_flops(),
                                          3 * bytes);
    return Kernel(std::move(program), KernelOpType::LAYERNORM, options.dtype);
}

Kernel KernelCompiler::compile_rmsnorm(const RMSNormConfig& config,
                                       const CompileOptions& options) {
    const Size tile_elems = 256;
    const Size elem_size = dtype_size(options.dtype);
    Size data_tiles = ceil_div_sz(config.total_elements(), tile_elems);
    Size group_tiles = ceil_div_sz(config.num_groups(), tile_elems);

    // RMSNorm skips mean centering: one reduction pass, one scale pass
    std::vector<StreamingPass> passes = {
        {"rmsnorm P1: per-group mean of squares (VE_REDUCE SUMSQ)",
         isa::DMOpcode::VE_REDUCE, data_tiles, group_tiles},
        {"rmsnorm P2: x * rsqrt(ms + eps) * gamma (VE_ELEMENTWISE)",
         isa::DMOpcode::VE_ELEMENTWISE, data_tiles, data_tiles},
    };

    uint64_t bytes = static_cast<uint64_t>(config.total_elements()) * elem_size;
    auto program = emit_streaming_program("rmsnorm", passes, tile_elems,
                                          elem_size, config.total_flops(),
                                          2 * bytes);
    return Kernel(std::move(program), KernelOpType::RMSNORM, options.dtype);
}

Kernel KernelCompiler::compile_batchnorm(const BatchNormConfig& config,
                                         const CompileOptions& options) {
    const Size tile_elems = 256;
    const Size elem_size = dtype_size(options.dtype);
    Size data_tiles = ceil_div_sz(config.total_elements(), tile_elems);
    // Per-channel parameters: running mean/var (+ gamma/beta when affine)
    Size params_per_channel = config.affine ? 4 : 2;
    Size param_tiles =
        ceil_div_sz(params_per_channel * config.num_features, tile_elems);

    std::vector<StreamingPass> passes = {
        {"batchnorm P0: preload per-channel params (no VE)",
         isa::DMOpcode::NOP, param_tiles, 0},
        {"batchnorm P1: (x - mean) * rsqrt(var + eps) * gamma + beta (VE_ELEMENTWISE)",
         isa::DMOpcode::VE_ELEMENTWISE, data_tiles, data_tiles},
    };

    uint64_t bytes = static_cast<uint64_t>(config.total_elements()) * elem_size;
    auto program = emit_streaming_program("batchnorm", passes, tile_elems,
                                          elem_size, config.total_flops(),
                                          2 * bytes);
    return Kernel(std::move(program), KernelOpType::BATCHNORM, options.dtype);
}

Kernel KernelCompiler::compile_elementwise(const ElementwiseConfig& config,
                                           const CompileOptions& options) {
    const Size tile_elems = 256;
    const Size elem_size = dtype_size(options.dtype);
    Size data_tiles = ceil_div_sz(config.total_elements(), tile_elems);
    Size input_streams = (config.is_unary || config.is_scalar_b) ? 1 : 2;

    std::vector<StreamingPass> passes = {
        {"elementwise: apply op per tile (VE_ELEMENTWISE)",
         isa::DMOpcode::VE_ELEMENTWISE, input_streams * data_tiles, data_tiles},
    };

    uint64_t bytes = static_cast<uint64_t>(config.total_elements()) * elem_size;
    auto program = emit_streaming_program("elementwise", passes, tile_elems,
                                          elem_size, config.total_flops(),
                                          (input_streams + 1) * bytes);
    return Kernel(std::move(program), KernelOpType::ELEMENTWISE, options.dtype);
}

Kernel KernelCompiler::compile_pool2d(const Pool2DConfig& config,
                                      const CompileOptions& options) {
    const Size tile_elems = 256;
    const Size elem_size = dtype_size(options.dtype);
    Size in_tiles = ceil_div_sz(config.input_elements(), tile_elems);
    Size out_tiles = ceil_div_sz(config.output_elements(), tile_elems);

    std::vector<StreamingPass> passes = {
        {"pool2d: windowed reduction (VE_REDUCE MAX/AVG)",
         isa::DMOpcode::VE_REDUCE, in_tiles, out_tiles},
    };

    uint64_t in_bytes = static_cast<uint64_t>(config.input_elements()) * elem_size;
    uint64_t out_bytes = static_cast<uint64_t>(config.output_elements()) * elem_size;
    auto program = emit_streaming_program("pool2d", passes, tile_elems,
                                          elem_size, config.total_flops(),
                                          in_bytes + out_bytes);
    return Kernel(std::move(program), KernelOpType::POOL2D, options.dtype);
}

} // namespace sw::kpu::compiler
