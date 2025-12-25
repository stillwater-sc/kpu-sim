#pragma once
// Kernel Compiler for KPU simulator
// High-level interface for compiling kernels with automatic tile optimization

#include <sw/kpu/kernel.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/compiler/tile_optimizer.hpp>

#include <string>
#include <chrono>

namespace sw::kpu::compiler {

/**
 * @brief Dataflow strategy for kernel execution
 *
 * Different dataflow strategies optimize for different scenarios:
 * - OUTPUT_STATIONARY: Keep output tiles in PE registers, best for balanced workloads
 * - WEIGHT_STATIONARY: Keep weight tiles stationary, best for inference with fixed weights
 * - INPUT_STATIONARY: Keep input tiles stationary, best for large output dimensions
 * - AUTO: Let the compiler choose based on problem dimensions
 */
enum class DataflowStrategy : uint8_t {
    OUTPUT_STATIONARY = 0,  ///< C tiles stay in PEs (default, most common)
    WEIGHT_STATIONARY = 1,  ///< B tiles stay in PEs (inference optimized)
    INPUT_STATIONARY = 2,   ///< A tiles stay in PEs (rarely used)
    AUTO = 255              ///< Let compiler choose
};

/**
 * @brief Get string name for dataflow strategy
 */
inline const char* dataflow_strategy_name(DataflowStrategy strategy) {
    switch (strategy) {
        case DataflowStrategy::OUTPUT_STATIONARY: return "output_stationary";
        case DataflowStrategy::WEIGHT_STATIONARY: return "weight_stationary";
        case DataflowStrategy::INPUT_STATIONARY: return "input_stationary";
        case DataflowStrategy::AUTO: return "auto";
        default: return "unknown";
    }
}

/**
 * @brief Compilation options for kernel generation
 *
 * Controls tile sizes, dataflow strategy, and optimization flags.
 * Use defaults() for automatic optimization, or with_tiles() for explicit control.
 */
struct CompileOptions {
    // Dataflow strategy
    DataflowStrategy dataflow = DataflowStrategy::AUTO;

    // Tile sizes (0 = auto-optimize using TileOptimizer)
    Size Ti = 0;       ///< M-dimension tile size
    Size Tj = 0;       ///< N-dimension tile size
    Size Tk = 0;       ///< K-dimension tile size
    Size L1_Ki = 0;    ///< L1 streaming chunk (0 = use Tk or default)

    // Execution flags
    bool double_buffer = true;            ///< Enable double buffering for overlap
    bool enable_tile_caching = true;      ///< Cache tiles in L3 for reuse
    bool generate_prologue = true;        ///< Generate prologue/epilogue instructions

    // Hardware configuration
    Size systolic_size = 16;              ///< Systolic array dimension (16x16)
    DataType dtype = DataType::FLOAT32;   ///< Data type of elements

    // Memory hierarchy (0 = use defaults from TileOptimizer)
    Size l3_tile_capacity = 0;            ///< L3 tile capacity in bytes
    Size l2_bank_capacity = 0;            ///< L2 bank capacity in bytes
    Size l1_buffer_capacity = 0;          ///< L1 buffer capacity in bytes
    uint8_t num_l3_tiles = 0;             ///< Number of L3 tiles
    uint8_t num_l2_banks = 0;             ///< Number of L2 banks
    uint8_t num_l1_buffers = 0;           ///< Number of L1 buffers

    // Tile optimization strategy
    TileOptimizer::Strategy tile_strategy = TileOptimizer::Strategy::ANALYTICAL;

    /**
     * @brief Create default options with auto-optimization
     */
    static CompileOptions defaults() {
        return CompileOptions{};
    }

    /**
     * @brief Create options with explicit tile sizes
     * @param ti M-dimension tile size
     * @param tj N-dimension tile size
     * @param tk K-dimension tile size
     */
    static CompileOptions with_tiles(Size ti, Size tj, Size tk) {
        CompileOptions opts;
        opts.Ti = ti;
        opts.Tj = tj;
        opts.Tk = tk;
        opts.dataflow = DataflowStrategy::OUTPUT_STATIONARY;
        return opts;
    }

    /**
     * @brief Create options for inference (weight-stationary)
     */
    static CompileOptions for_inference() {
        CompileOptions opts;
        opts.dataflow = DataflowStrategy::WEIGHT_STATIONARY;
        return opts;
    }

    /**
     * @brief Check if using automatic tile optimization
     */
    bool is_auto_tiling() const {
        return Ti == 0 || Tj == 0 || Tk == 0;
    }
};

/**
 * @brief Statistics for a single resource type (DMA, BlockMover, Streamer)
 *
 * Captures the granularity of operations in a distributed dataflow machine.
 * Unlike simple instruction counts, this tracks bytes moved and latency.
 */
struct ResourceOperationStats {
    size_t count = 0;              ///< Number of operations issued
    Size total_bytes = 0;          ///< Total bytes moved by this resource
    Size avg_bytes_per_op = 0;     ///< Average bytes per operation
    Cycle avg_latency_cycles = 0;  ///< Average cycles per operation

    /**
     * @brief Finalize statistics after accumulation
     */
    void finalize() {
        if (count > 0) {
            avg_bytes_per_op = total_bytes / count;
        }
    }
};

/**
 * @brief Pipeline resource configuration
 *
 * Describes the concurrency available at each level of the memory hierarchy.
 */
struct PipelineResources {
    size_t dma_channels = 4;    ///< Number of concurrent DMA channels
    size_t block_movers = 8;    ///< Number of concurrent block movers
    size_t streamers = 16;      ///< Number of concurrent streamers

    // Peak bandwidth specifications (bytes/cycle at 1 GHz)
    Size external_peak_bw = 64;  ///< External memory peak (64 GB/s)
    Size l3_l2_peak_bw = 128;    ///< L3↔L2 peak (128 GB/s)
    Size l2_l1_peak_bw = 256;    ///< L2↔L1 peak (256 GB/s)
};

/**
 * @brief Bandwidth estimates and utilization
 */
struct BandwidthEstimates {
    double external_gbps = 0.0;       ///< Achieved external bandwidth (GB/s)
    double l3_l2_gbps = 0.0;          ///< Achieved L3↔L2 bandwidth (GB/s)
    double l2_l1_gbps = 0.0;          ///< Achieved L2↔L1 bandwidth (GB/s)

    double external_utilization = 0.0; ///< Fraction of peak external BW
    double l3_l2_utilization = 0.0;    ///< Fraction of peak L3↔L2 BW
    double l2_l1_utilization = 0.0;    ///< Fraction of peak L2↔L1 BW
};

/**
 * @brief Complete operation breakdown for a compiled kernel
 *
 * Replaces simple "instruction counts" with meaningful operation statistics
 * that capture the granularity and data movement characteristics of a
 * distributed dataflow machine.
 */
struct OperationBreakdown {
    // Per-level operation statistics
    ResourceOperationStats external_memory;  ///< DMA (External ↔ L3)
    ResourceOperationStats l3_l2;            ///< Block Mover (L3 ↔ L2)
    ResourceOperationStats l2_l1;            ///< Streamer (L2 ↔ L1)

    // Pipeline configuration
    PipelineResources pipeline;

    // Bandwidth estimates (computed from stats and cycle estimates)
    BandwidthEstimates bandwidth;

    // Total estimated execution cycles (for bandwidth calculation)
    Cycle estimated_cycles = 0;

    /**
     * @brief Compute bandwidth estimates from operation stats
     * @param clock_ghz Clock frequency in GHz (default 1.0)
     */
    void compute_bandwidth(double clock_ghz = 1.0);

    /**
     * @brief Get formatted summary string
     */
    std::string summary() const;
};

/**
 * @brief Statistics from kernel compilation
 *
 * Provides insights into the compilation process and the generated kernel.
 */
struct CompilationStats {
    // Compilation time
    double compile_time_us = 0.0;         ///< Compilation time in microseconds

    // Tile optimization
    bool used_auto_tiling = false;        ///< Whether auto-optimization was used
    Size selected_Ti = 0;                 ///< Final Ti tile size
    Size selected_Tj = 0;                 ///< Final Tj tile size
    Size selected_Tk = 0;                 ///< Final Tk tile size
    Size selected_L1_Ki = 0;              ///< Final L1 streaming chunk

    // Operation breakdown (replaces instruction_count, dma_ops, etc.)
    OperationBreakdown operations;

    // Legacy fields for backward compatibility (deprecated, use operations instead)
    size_t instruction_count = 0;         ///< @deprecated Use operations.*.count
    size_t dma_ops = 0;                   ///< @deprecated Use operations.external_memory.count
    size_t block_mover_ops = 0;           ///< @deprecated Use operations.l3_l2.count
    size_t streamer_ops = 0;              ///< @deprecated Use operations.l2_l1.count
    size_t compute_ops = 0;               ///< @deprecated Compute is implicit in streaming

    // Memory estimates
    Size estimated_external_bytes = 0;    ///< Estimated external memory traffic
    Size estimated_l3_bytes = 0;          ///< Estimated L3 traffic
    Size estimated_l2_bytes = 0;          ///< Estimated L2 traffic
    double estimated_arithmetic_intensity = 0.0;  ///< FLOPs per byte from DRAM

    // Tile loop counts
    Size num_m_tiles = 0;                 ///< Number of tiles in M dimension
    Size num_n_tiles = 0;                 ///< Number of tiles in N dimension
    Size num_k_tiles = 0;                 ///< Number of tiles in K dimension
    Size total_tiles = 0;                 ///< Total number of tiles

    // Dataflow used
    DataflowStrategy dataflow_used = DataflowStrategy::OUTPUT_STATIONARY;

    /**
     * @brief Get human-readable summary string
     */
    std::string summary() const;
};

/**
 * @brief Kernel Compiler - High-level compilation interface
 *
 * Provides automatic tile optimization and program generation for various
 * kernel types. Uses TileOptimizer internally and generates DMPrograms
 * via OutputStationaryProgramBuilder.
 *
 * Usage:
 *   // Simple compilation with auto-optimization
 *   KernelCompiler compiler;
 *   Kernel kernel = compiler.compile_matmul(1024, 1024, 1024);
 *
 *   // With explicit tile sizes
 *   Kernel kernel = compiler.compile_matmul(1024, 1024, 1024, 64, 64, 128);
 *
 *   // With options
 *   auto opts = CompileOptions::for_inference();
 *   Kernel kernel = compiler.compile_matmul(1024, 1024, 1024, opts);
 *
 *   // Check compilation stats
 *   std::cout << compiler.last_stats().summary();
 */
class KernelCompiler {
public:
    // =========================================
    // Constructors
    // =========================================

    /**
     * @brief Default constructor with default memory hierarchy
     */
    KernelCompiler();

    /**
     * @brief Construct with custom memory hierarchy
     * @param memory Memory hierarchy specification for tile optimization
     */
    explicit KernelCompiler(const TileOptimizer::MemoryHierarchy& memory);

    // =========================================
    // Main Compilation API
    // =========================================

    /**
     * @brief Compile a matrix multiplication kernel with automatic optimization
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param options Compilation options (default: auto-optimize everything)
     * @return Compiled kernel
     *
     * This is the main compilation method. It automatically:
     * 1. Optimizes tile sizes using TileOptimizer
     * 2. Selects appropriate dataflow strategy
     * 3. Generates DMProgram via OutputStationaryProgramBuilder
     * 4. Wraps everything in a Kernel with metadata
     */
    Kernel compile_matmul(Size M, Size N, Size K,
                          const CompileOptions& options = CompileOptions::defaults());

    /**
     * @brief Compile a matrix multiplication kernel with explicit tile sizes
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param Ti M-dimension tile size
     * @param Tj N-dimension tile size
     * @param Tk K-dimension tile size
     * @return Compiled kernel
     *
     * Convenience method for when you know the tile sizes you want.
     */
    Kernel compile_matmul(Size M, Size N, Size K,
                          Size Ti, Size Tj, Size Tk);

    /**
     * @brief Compile a matrix multiplication kernel with explicit tile and L1 sizes
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param Ti M-dimension tile size
     * @param Tj N-dimension tile size
     * @param Tk K-dimension tile size
     * @param L1_Ki L1 streaming chunk size
     * @return Compiled kernel
     */
    Kernel compile_matmul(Size M, Size N, Size K,
                          Size Ti, Size Tj, Size Tk, Size L1_Ki);

    /**
     * @brief Compile a fused MLP kernel (matmul + bias + activation)
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param activation Activation function type
     * @param has_bias Whether to apply bias addition
     * @param dtype Data type (default FLOAT32)
     * @param options Compilation options (default: auto-optimize)
     * @return Compiled kernel
     *
     * Generates C = activation(A @ B + bias) as a fused operation.
     * The Vector Engine applies bias and activation inline during
     * the output drain phase.
     */
    Kernel compile_mlp(Size M, Size N, Size K,
                       ActivationType activation,
                       bool has_bias = true,
                       DataType dtype = DataType::FLOAT32,
                       const CompileOptions& options = CompileOptions::defaults());

    // =========================================
    // Tile Optimization
    // =========================================

    /**
     * @brief Optimize tile sizes for given dimensions
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param strategy Optimization strategy (default: analytical)
     * @return Optimal tile configuration
     *
     * Useful when you want to inspect tile sizes before compilation.
     */
    TileOptimizer::TileConfig optimize_tiles(
        Size M, Size N, Size K,
        TileOptimizer::Strategy strategy = TileOptimizer::Strategy::ANALYTICAL);

    /**
     * @brief Get the underlying tile optimizer
     */
    TileOptimizer& tile_optimizer() { return tile_optimizer_; }
    const TileOptimizer& tile_optimizer() const { return tile_optimizer_; }

    // =========================================
    // Compilation Status
    // =========================================

    /**
     * @brief Get statistics from the last compilation
     */
    const CompilationStats& last_stats() const { return last_stats_; }

    /**
     * @brief Check if the last compilation succeeded
     */
    bool last_succeeded() const { return last_succeeded_; }

    /**
     * @brief Get error message from the last compilation (if failed)
     */
    const std::string& last_error() const { return last_error_; }

    // =========================================
    // Memory Hierarchy Configuration
    // =========================================

    /**
     * @brief Set memory hierarchy for tile optimization
     */
    void set_memory_hierarchy(const TileOptimizer::MemoryHierarchy& memory);

    /**
     * @brief Get current memory hierarchy
     */
    const TileOptimizer::MemoryHierarchy& memory_hierarchy() const {
        return tile_optimizer_.memory_hierarchy();
    }

private:
    TileOptimizer tile_optimizer_;
    CompilationStats last_stats_;
    bool last_succeeded_ = false;
    std::string last_error_;

    /**
     * @brief Build OutputStationaryProgramBuilder::Config from options and tile config
     */
    isa::OutputStationaryProgramBuilder::Config build_program_config(
        Size M, Size N, Size K,
        const TileOptimizer::TileConfig& tiles,
        const CompileOptions& options);

    /**
     * @brief Select dataflow strategy based on problem dimensions
     */
    DataflowStrategy select_dataflow(Size M, Size N, Size K) const;

    /**
     * @brief Count operations and compute statistics
     *
     * Analyzes the program to populate OperationBreakdown with counts,
     * bytes moved, and latency estimates for each resource type.
     */
    void count_operations(const isa::DMProgram& program,
                          Size elem_size,
                          const TileOptimizer::TileConfig& tiles);
};

} // namespace sw::kpu::compiler
