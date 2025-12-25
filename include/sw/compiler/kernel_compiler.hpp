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

    // Instruction breakdown
    size_t instruction_count = 0;         ///< Total instruction count
    size_t dma_ops = 0;                   ///< DMA operations
    size_t block_mover_ops = 0;           ///< Block mover operations
    size_t streamer_ops = 0;              ///< Streamer operations
    size_t compute_ops = 0;               ///< Compute operations

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
     * @brief Count instructions by type for statistics
     */
    void count_instructions(const isa::DMProgram& program);
};

} // namespace sw::kpu::compiler
