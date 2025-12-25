#pragma once
// Vector Engine (VE) for inline bias addition and activation functions
// Processes data during L1→L2 transfer for operator fusion

#include <sw/concepts.hpp>
#include <sw/kpu/components/sfu.hpp>
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
 * @brief Vector Engine Configuration
 */
struct VectorEngineConfig {
    Size vector_width = 8;              ///< Elements processed per cycle
    Size bias_buffer_size = 4096;       ///< Max bias vector elements
    SFUConfig sfu_config;               ///< SFU configuration
    bool enabled = true;                ///< Enable/disable VE
    Size pipeline_depth = 3;            ///< Total pipeline latency (bias + SFU)
};

/**
 * @brief Operation descriptor for Vector Engine
 *
 * Describes a single bias + activation operation on a tile of data.
 * The VE processes data row by row as it flows from L1 to L2.
 */
struct VEOperation {
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

    // Completion callback
    std::function<void()> completion_callback;
};

/**
 * @brief Vector Engine Statistics
 */
struct VEStats {
    uint64_t elements_processed = 0;    ///< Total elements through VE
    uint64_t bias_additions = 0;        ///< Bias operations performed
    uint64_t activations_computed = 0;  ///< Activation functions computed
    uint64_t operations_completed = 0;  ///< Number of VE operations completed
    uint64_t cycles_active = 0;         ///< Cycles spent processing
    uint64_t cycles_idle = 0;           ///< Cycles spent idle
};

/**
 * @brief Vector Engine - Inline Bias + Activation Processor
 *
 * The Vector Engine processes data inline during L1→L2 transfers,
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
 *   - Streamer routes output data through VE when enabled
 *   - VE appears transparent when disabled (pass-through)
 *   - Zero-copy operation: data flows L1 → VE → L2
 *
 * Memory Traffic Savings:
 *   Without VE: 5 memory passes (matmul out, bias in, bias out, act in, act out)
 *   With VE: 1 memory pass (fused matmul + bias + activation)
 *   Savings: 4× reduction in L2 traffic
 */
class KPU_API VectorEngine {
public:
    // =========================================
    // Constructors
    // =========================================

    /**
     * @brief Construct with default configuration
     * @param id Unique identifier for this VE instance
     */
    explicit VectorEngine(size_t id = 0);

    /**
     * @brief Construct with specific configuration
     * @param id Unique identifier for this VE instance
     * @param config VE configuration
     */
    VectorEngine(size_t id, const VectorEngineConfig& config);

    ~VectorEngine() = default;

    // =========================================
    // Configuration
    // =========================================

    /**
     * @brief Update VE configuration
     * @param config New configuration
     *
     * Reconfigures the SFU and resizes buffers as needed.
     */
    void configure(const VectorEngineConfig& config);

    /**
     * @brief Set the activation function
     * @param activation Activation type for SFU
     */
    void set_activation(ActivationType activation);

    /**
     * @brief Preload bias vector into VE buffer
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
    const VectorEngineConfig& config() const { return config_; }

    /**
     * @brief Get VE instance ID
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
    void enqueue_operation(const VEOperation& op);

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
     * @brief Advance VE state by one cycle
     * @param cycle Current simulation cycle
     * @param l1_read Function to read from L1 scratchpad
     * @param l2_write Function to write to L2 bank
     * @return true if an operation completed this cycle
     *
     * The l1_read and l2_write functions abstract the memory interface,
     * allowing the VE to be tested independently of the full memory system.
     */
    using L1ReadFunc = std::function<void(size_t scratchpad_id, Address addr, void* data, Size size)>;
    using L2WriteFunc = std::function<void(size_t bank_id, Address addr, const void* data, Size size)>;

    bool update(Cycle cycle, L1ReadFunc l1_read, L2WriteFunc l2_write);

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
     * @brief Check if VE is currently processing
     */
    bool is_busy() const { return state_ != State::IDLE; }

    /**
     * @brief Check if VE is enabled
     */
    bool is_enabled() const { return config_.enabled; }

    /**
     * @brief Enable the Vector Engine
     */
    void enable() { config_.enabled = true; }

    /**
     * @brief Disable the Vector Engine (pass-through mode)
     */
    void disable() { config_.enabled = false; }

    /**
     * @brief Reset VE state and clear queues
     */
    void reset();

    // =========================================
    // Statistics
    // =========================================

    /**
     * @brief Get performance statistics
     */
    const VEStats& stats() const { return stats_; }

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
    VectorEngineConfig config_;
    SFU sfu_;

    // Bias buffer (preloaded once per layer)
    std::vector<float> bias_buffer_;
    bool bias_loaded_ = false;

    // Operation queue
    std::queue<VEOperation> op_queue_;

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
    VEOperation current_op_;
    Size current_row_ = 0;
    Cycle op_start_cycle_ = 0;

    // Pipeline buffers (for accurate timing)
    std::vector<float> input_buffer_;   // Current row being processed
    std::vector<float> output_buffer_;  // Processed row ready for output

    // Statistics
    VEStats stats_;

    // Internal helpers
    void start_operation();
    void process_row(L1ReadFunc l1_read, L2WriteFunc l2_write);
    void finish_operation();
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
