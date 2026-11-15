#pragma once
// Combiner for inline bias addition and activation functions
// Processes data during L1→L2 transfer for operator fusion

#include <sw/concepts.hpp>
#include <sw/kpu/models/temporal/compute/sfu.hpp>
#include <vector>
#include <queue>
#include <cstdint>
#include <functional>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

namespace sw::kpu {

/**
 * @brief Combiner Configuration
 */
struct CombinerConfig {
    Size vector_width = 8;              ///< Elements processed per cycle
    Size bias_buffer_size = 4096;       ///< Max bias vector elements
    SFUConfig sfu_config;               ///< SFU configuration
    bool enabled = true;                ///< Enable/disable Combiner
    Size pipeline_depth = 3;            ///< Total pipeline latency (bias + SFU)
};

/**
 * @brief How the Combiner writes a result row into L2 (#429)
 *
 * Overwrite writes the row (an output-stationary result, or a partial's first K block).
 * Accumulate is the merge of partials of the weight- and input-stationary schedules: the row in
 * L2 is read, the new row added to it, and the sum written back (a read-modify-write in L2).
 */
enum class CombinerMerge : uint8_t { Overwrite, Accumulate };

/**
 * @brief A block format the Combiner converts results into (#427)
 *
 * Reserved: the OCP Microscaling formats, MXINT8 first, then MXFP8. Only None is built; an
 * operation that names another is refused.
 */
enum class CombinerBlockFormat : uint8_t { None, MXINT8, MXFP8 };

/**
 * @brief Operation descriptor for Combiner
 *
 * Describes a single bias + activation operation on a tile of data.
 * The Combiner processes data row by row as it flows from L1 to L2.
 */
struct CombinerOperation {
    // Source: L1 scratchpad
    size_t l1_scratchpad_id = 0;
    Address l1_base_addr = 0;

    // Destination: L2 bank
    size_t l2_bank_id = 0;
    Address l2_base_addr = 0;

    // Tile dimensions
    Size height = 0;                    ///< Number of rows (M dimension)
    Size width = 0;                     ///< Number of columns (N dimension)
    Size row_stride = 0;                ///< Stride between rows in bytes
    Size element_size = sizeof(float);  ///< Element size (4 for float32)

    // Bias configuration
    bool bias_enabled = false;          ///< Apply bias addition
    Address bias_addr = 0;              ///< Address of bias vector in L1
    Size bias_stride = 0;               ///< Stride for bias elements

    // Activation configuration
    ActivationType activation = ActivationType::NONE;

    // The merge into L2 (#429): Accumulate reads the L2 row and adds the new one to it. An
    // epilogue (bias, activation) applies to the final merge only, which is #427's, so an
    // accumulating operation takes none.
    CombinerMerge merge = CombinerMerge::Overwrite;

    // Reserved for #427: scaling (requantization) and a block-format conversion. Only the
    // defaults (scale 1, no format) are built; enqueue_operation refuses others.
    float scale = 1.0f;
    CombinerBlockFormat format = CombinerBlockFormat::None;

    // Completion callback
    std::function<void()> completion_callback;
};

/**
 * @brief Combiner Statistics
 */
struct CombinerStats {
    uint64_t elements_processed = 0;    ///< Total elements through Combiner
    uint64_t bias_additions = 0;        ///< Bias operations performed
    uint64_t activations_computed = 0;  ///< Activation functions computed
    uint64_t elements_merged = 0;       ///< Elements accumulated into L2 (read-modify-write)
    uint64_t operations_completed = 0;  ///< Number of Combiner operations completed
    uint64_t cycles_active = 0;         ///< Cycles spent processing
    uint64_t cycles_idle = 0;           ///< Cycles spent idle
};

/**
 * @brief Combiner - Inline Bias + Activation Processor
 *
 * The Combiner processes data inline during L1→L2 transfers,
 * applying bias addition and activation functions without additional
 * memory passes. This achieves operator fusion for MLP layers.
 *
 * Pipeline Architecture:
 *   Cycle 1: Load row from L1 buffer
 *   Cycle 2: Bias addition (vector add with broadcast)
 *   Cycle 3: SFU activation (LUT lookup + interpolation)
 *   Cycle 4: Store to L2 bank
 *
 * Throughput: vector_width elements per cycle (default 8)
 *
 * Integration with Streamer:
 *   - Streamer routes output data through Combiner when enabled
 *   - Combiner appears transparent when disabled (pass-through)
 *   - Zero-copy operation: data flows L1 → Combiner → L2
 *
 * Memory Traffic Savings:
 *   Without Combiner: 5 memory passes (matmul out, bias in, bias out, act in, act out)
 *   With Combiner: 1 memory pass (fused matmul + bias + activation)
 *   Savings: 4× reduction in L2 traffic
 */
class KPU_API Combiner {
public:
    // =========================================
    // Constructors
    // =========================================

    /**
     * @brief Construct with default configuration
     * @param id Unique identifier for this Combiner instance
     */
    explicit Combiner(size_t id = 0);

    /**
     * @brief Construct with specific configuration
     * @param id Unique identifier for this Combiner instance
     * @param config Combiner configuration
     */
    Combiner(size_t id, const CombinerConfig& config);

    ~Combiner() = default;

    // =========================================
    // Configuration
    // =========================================

    /**
     * @brief Update Combiner configuration
     * @param config New configuration
     *
     * Reconfigures the SFU and resizes buffers as needed.
     */
    void configure(const CombinerConfig& config);

    /**
     * @brief Set the activation function
     * @param activation Activation type for SFU
     */
    void set_activation(ActivationType activation);

    /**
     * @brief Preload bias vector into Combiner buffer
     * @param bias_data Pointer to bias values
     * @param count Number of bias elements
     *
     * Bias is broadcast across rows during operation.
     * Must be called before operation if bias is enabled.
     */
    void preload_bias(const float* bias_data, Size count);

    /**
     * @brief Get current configuration
     */
    const CombinerConfig& config() const { return config_; }

    /**
     * @brief Get Combiner instance ID
     */
    size_t id() const { return id_; }

    // =========================================
    // Operation Queue
    // =========================================

    /**
     * @brief Enqueue an operation for processing
     * @param op Operation descriptor
     *
     * Operations are processed in FIFO order.
     */
    void enqueue_operation(const CombinerOperation& op);

    /**
     * @brief Check if operation queue is empty
     */
    bool has_pending_operations() const { return !op_queue_.empty(); }

    /**
     * @brief Get number of pending operations
     */
    size_t pending_operation_count() const { return op_queue_.size(); }

    // =========================================
    // Cycle-Accurate Simulation
    // =========================================

    /**
     * @brief Advance Combiner state by one cycle
     * @param cycle Current simulation cycle
     * @param l1_read Function to read from L1 scratchpad
     * @param l2_write Function to write to L2 bank
     * @return true if an operation completed this cycle
     *
     * The l1_read and l2_write functions abstract the memory interface,
     * allowing the Combiner to be tested independently of the full memory system.
     */
    using L1ReadFunc = std::function<void(size_t scratchpad_id, Address addr, void* data, Size size)>;
    using L2WriteFunc = std::function<void(size_t bank_id, Address addr, const void* data, Size size)>;
    using L2ReadFunc = std::function<void(size_t bank_id, Address addr, void* data, Size size)>;

    bool update(Cycle cycle, L1ReadFunc l1_read, L2WriteFunc l2_write);
    /// The same, with L2 readable: an accumulating operation (CombinerMerge::Accumulate) reads
    /// the row it merges into through `l2_read`.
    bool update(Cycle cycle, L1ReadFunc l1_read, L2WriteFunc l2_write, L2ReadFunc l2_read);

    // =========================================
    // Immediate Processing (Non-Pipelined)
    // =========================================

    /**
     * @brief Process a single row immediately (for testing)
     * @param input Input row data
     * @param output Output row data (pre-allocated)
     * @param width Number of elements in row
     * @param bias_row Row index for bias indexing
     *
     * Applies bias (if enabled) and activation to input row.
     * This bypasses the pipeline for synchronous testing.
     */
    void process_row_immediate(const float* input, float* output, Size width, Size bias_row = 0);

    /**
     * @brief Process entire tile immediately (for testing)
     * @param input Input tile data (row-major)
     * @param output Output tile data (pre-allocated)
     * @param height Number of rows
     * @param width Number of columns
     *
     * Applies bias and activation to entire tile synchronously.
     */
    void process_tile_immediate(const float* input, float* output, Size height, Size width);

    // =========================================
    // State Management
    // =========================================

    /**
     * @brief Check if Combiner is currently processing
     */
    bool is_busy() const { return state_ != State::IDLE; }

    /**
     * @brief Check if Combiner is enabled
     */
    bool is_enabled() const { return config_.enabled; }

    /**
     * @brief Enable the Combiner
     */
    void enable() { config_.enabled = true; }

    /**
     * @brief Disable the Combiner (pass-through mode)
     */
    void disable() { config_.enabled = false; }

    /**
     * @brief Reset Combiner state and clear queues
     */
    void reset();

    // =========================================
    // Statistics
    // =========================================

    /**
     * @brief Get performance statistics
     */
    const CombinerStats& stats() const { return stats_; }

    /**
     * @brief Reset statistics counters
     */
    void reset_stats();

    /**
     * @brief Get the internal SFU (for configuration/testing)
     */
    SFU& sfu() { return sfu_; }
    const SFU& sfu() const { return sfu_; }

    // =========================================
    // Timing
    // =========================================

    /**
     * @brief Get pipeline latency in cycles
     */
    Size get_latency_cycles() const { return config_.pipeline_depth; }

    /**
     * @brief Get throughput in elements per cycle
     */
    Size get_throughput() const { return config_.vector_width; }

    /**
     * @brief Estimate cycles to process a tile
     * @param height Number of rows
     * @param width Number of columns
     * @return Estimated cycle count
     */
    Cycle estimate_cycles(Size height, Size width) const;

private:
    size_t id_;
    CombinerConfig config_;
    SFU sfu_;

    // Bias buffer (preloaded once per layer)
    std::vector<float> bias_buffer_;
    bool bias_loaded_ = false;

    // Operation queue
    std::queue<CombinerOperation> op_queue_;

    // Current operation state
    enum class State {
        IDLE,           ///< No operation in progress
        STARTING,       ///< Setting up operation
        PROCESSING,     ///< Processing rows
        DRAINING,       ///< Draining pipeline
        COMPLETING      ///< Calling completion callback
    };
    State state_ = State::IDLE;

    // Current operation progress
    CombinerOperation current_op_;
    Size current_row_ = 0;
    Cycle op_start_cycle_ = 0;

    // Pipeline buffers (for accurate timing)
    std::vector<float> input_buffer_;   // Current row being processed
    std::vector<float> output_buffer_;  // Processed row ready for output

    // Statistics
    CombinerStats stats_;

    // Internal helpers
    void start_operation();
    void process_row(L1ReadFunc l1_read, L2WriteFunc l2_write, const L2ReadFunc& l2_read);
    std::vector<float> merge_buffer_;   ///< The L2 row an accumulating operation merges into
    void finish_operation();
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
