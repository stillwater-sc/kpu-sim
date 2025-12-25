// Kernel Compiler implementation for KPU simulator
// Provides high-level compilation with automatic tile optimization

#include <sw/compiler/kernel_compiler.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace sw::kpu::compiler {

// ============================================================================
// CompilationStats Implementation
// ============================================================================

std::string CompilationStats::summary() const {
    std::ostringstream ss;

    ss << "Compilation Statistics:\n";
    ss << "  Compile Time: " << std::fixed << std::setprecision(1)
       << compile_time_us << " us\n";

    ss << "  Tile Configuration:\n";
    ss << "    Auto-optimized: " << (used_auto_tiling ? "yes" : "no") << "\n";
    ss << "    Ti=" << selected_Ti << ", Tj=" << selected_Tj
       << ", Tk=" << selected_Tk << ", L1_Ki=" << selected_L1_Ki << "\n";
    ss << "    Tiles: " << num_m_tiles << " x " << num_n_tiles << " x "
       << num_k_tiles << " = " << total_tiles << " total\n";

    ss << "  Instruction Breakdown:\n";
    ss << "    Total: " << instruction_count << "\n";
    ss << "    DMA ops: " << dma_ops << "\n";
    ss << "    Block mover ops: " << block_mover_ops << "\n";
    ss << "    Streamer ops: " << streamer_ops << "\n";
    ss << "    Compute ops: " << compute_ops << "\n";

    ss << "  Memory Traffic Estimates:\n";
    ss << "    External (DRAM): " << estimated_external_bytes << " bytes\n";
    ss << "    L3: " << estimated_l3_bytes << " bytes\n";
    ss << "    L2: " << estimated_l2_bytes << " bytes\n";
    ss << "    Arithmetic Intensity: " << std::fixed << std::setprecision(2)
       << estimated_arithmetic_intensity << " FLOPs/byte\n";

    ss << "  Dataflow: " << dataflow_strategy_name(dataflow_used) << "\n";

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

    // Step 4: Count instructions and record stats
    count_instructions(program);

    // Calculate tile counts
    auto ceil_div = [](Size a, Size b) { return (a + b - 1) / b; };
    last_stats_.num_m_tiles = ceil_div(M, tile_config.Ti);
    last_stats_.num_n_tiles = ceil_div(N, tile_config.Tj);
    last_stats_.num_k_tiles = ceil_div(K, tile_config.Tk);
    last_stats_.total_tiles = last_stats_.num_m_tiles *
                              last_stats_.num_n_tiles *
                              last_stats_.num_k_tiles;

    // Estimate memory traffic
    Size elem_size = dtype_size(options.dtype);
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

    // Step 4: Count instructions and record stats
    count_instructions(program);

    // Calculate tile counts
    auto ceil_div = [](Size a, Size b) { return (a + b - 1) / b; };
    last_stats_.num_m_tiles = ceil_div(M, tile_config.Ti);
    last_stats_.num_n_tiles = ceil_div(N, tile_config.Tj);
    last_stats_.num_k_tiles = ceil_div(K, tile_config.Tk);
    last_stats_.total_tiles = last_stats_.num_m_tiles *
                              last_stats_.num_n_tiles *
                              last_stats_.num_k_tiles;

    // Estimate memory traffic (MLP saves traffic via fusion)
    Size elem_size = dtype_size(dtype);
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

void KernelCompiler::count_instructions(const isa::DMProgram& program) {
    last_stats_.instruction_count = program.instructions.size();
    last_stats_.dma_ops = 0;
    last_stats_.block_mover_ops = 0;
    last_stats_.streamer_ops = 0;
    last_stats_.compute_ops = 0;

    for (const auto& instr : program.instructions) {
        switch (instr.opcode) {
            case isa::DMOpcode::DMA_LOAD_TILE:
            case isa::DMOpcode::DMA_STORE_TILE:
            case isa::DMOpcode::DMA_PREFETCH_TILE:
                last_stats_.dma_ops++;
                break;
            case isa::DMOpcode::BM_MOVE_TILE:
            case isa::DMOpcode::BM_TRANSPOSE_TILE:
            case isa::DMOpcode::BM_WRITEBACK_TILE:
            case isa::DMOpcode::BM_RESHAPE_TILE:
                last_stats_.block_mover_ops++;
                break;
            case isa::DMOpcode::STR_FEED_ROWS:
            case isa::DMOpcode::STR_FEED_COLS:
            case isa::DMOpcode::STR_DRAIN_OUTPUT:
            case isa::DMOpcode::STR_BROADCAST_ROW:
            case isa::DMOpcode::STR_BROADCAST_COL:
                last_stats_.streamer_ops++;
                break;
            // Note: No explicit COMPUTE opcode - computation is implicit
            // when streaming data to/from systolic array
            default:
                // NOP, BARRIER, WAIT_*, SIGNAL, SET_*, LOOP_*, HALT
                break;
        }
    }
}

} // namespace sw::kpu::compiler
