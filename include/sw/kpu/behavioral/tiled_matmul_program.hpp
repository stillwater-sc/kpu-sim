// ============================================================================
// include/sw/kpu/behavioral/tiled_matmul_program.hpp
// Parameterized Tiled Matrix Multiplication Program Generator
// ============================================================================
//
// Generates and executes tiled matrix multiplication:
//   D[M,N] = C[M,N] + A[M,K] * B[K,N]
//
// Features:
// - Parameterized problem dimensions
// - Tile sizes matched to systolic array
// - Credit-based dataflow execution model (NO CACHE semantics!)
// - Programmable loop order for optimal buffer utilization
// - Double-buffering for L3 and L2
// - Timing approximations for behavioral simulation
// - Trace generation with dataflow events for visualization
//
// EXECUTION MODEL: See docs/kpu-execution-model.md
// - Credits flow UPSTREAM (consumer signals buffer availability)
// - Data/tiles flow DOWNSTREAM (producer pushes when credit available)
// - NO cache, NO cache miss - only buffers and credits
//
// ============================================================================

#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <cmath>

namespace sw::kpu::behavioral {

// ============================================================================
// Loop Order Configuration
// ============================================================================

/// Loop ordering strategies for tiled matrix multiplication
enum class LoopOrder : uint8_t {
    IJK,        // for i: for j: for k: → A-row stays in cache
    JIK,        // for j: for i: for k: → B-column stays in cache
    IKJ,        // for i: for k: for j: → A-row reuse, B streamed
    KIJ,        // for k: for i: for j: → B-row stays in cache
    KJI,        // for k: for j: for i: → A-column stays in cache
    JKI,        // for j: for k: for i: → B-column reuse, A streamed
    BLOCKED     // 2-level blocking for better locality
};

inline const char* to_string(LoopOrder order) {
    switch (order) {
        case LoopOrder::IJK: return "IJK";
        case LoopOrder::JIK: return "JIK";
        case LoopOrder::IKJ: return "IKJ";
        case LoopOrder::KIJ: return "KIJ";
        case LoopOrder::KJI: return "KJI";
        case LoopOrder::JKI: return "JKI";
        case LoopOrder::BLOCKED: return "BLOCKED";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for tiled matrix multiplication
struct TiledMatmulConfig {
    // Problem dimensions: D[M,N] = C[M,N] + A[M,K] * B[K,N]
    uint32_t M = 1000;          // Output rows
    uint32_t N = 1000;          // Output columns
    uint32_t K = 100;           // Inner/reduction dimension

    // Tile sizes (default to systolic array size)
    uint32_t tile_m = 16;       // Tile height for A and C
    uint32_t tile_n = 16;       // Tile width for B and C
    uint32_t tile_k = 16;       // Tile depth for A and B

    // Hardware configuration
    uint32_t systolic_rows = 16;
    uint32_t systolic_cols = 16;
    uint8_t num_l3_tiles = 4;   // Total L3 tile buffers (legacy)
    uint8_t num_l2_banks = 8;   // Total L2 bank buffers

    // L3 Cache configuration (new)
    uint32_t l3_capacity_tiles = 24;  // How many tiles fit in L3 cache
    LoopOrder loop_order = LoopOrder::IKJ;  // Default: good A-row reuse

    // For BLOCKED loop order
    uint32_t block_i = 4;       // Block size in i dimension (tiles)
    uint32_t block_j = 4;       // Block size in j dimension (tiles)
    uint32_t block_k = 7;       // Block size in k dimension (tiles)

    // Double-buffering allocation
    // L3: [0,1] for A ping-pong, [2,3] for B ping-pong
    // L2: [0,1] for A, [2,3] for B, [4,5] for C input, [6,7] for D output
    uint8_t l3_a_buffers[2] = {0, 1};
    uint8_t l3_b_buffers[2] = {2, 3};
    uint8_t l2_a_banks[2] = {0, 1};
    uint8_t l2_b_banks[2] = {2, 3};
    uint8_t l2_c_banks[2] = {4, 5};
    uint8_t l2_d_banks[2] = {6, 7};

    // Timing parameters (cycles)
    uint32_t dma_load_latency = 100;    // Host -> L3
    uint32_t dma_store_latency = 100;   // L3 -> Host
    uint32_t bm_push_latency = 10;      // L3 -> L2
    uint32_t bm_pull_latency = 10;      // L2 -> L3
    uint32_t str_feed_latency = 8;      // L2 -> L1
    uint32_t str_drain_latency = 16;    // Accumulator -> L2
    uint32_t matmul_latency = 0;        // 0 = compute from array size

    // Bandwidth (bytes per cycle)
    uint32_t dma_bandwidth = 64;
    uint32_t bm_bandwidth = 32;
    uint32_t str_bandwidth = 16;

    // Element size
    uint32_t element_size = 4;  // sizeof(float)

    // Whether to include C (accumulate) or start from zero
    bool accumulate_c = true;

    // Compute derived values
    uint32_t m_tiles() const { return (M + tile_m - 1) / tile_m; }
    uint32_t n_tiles() const { return (N + tile_n - 1) / tile_n; }
    uint32_t k_tiles() const { return (K + tile_k - 1) / tile_k; }
    uint32_t total_output_tiles() const { return m_tiles() * n_tiles(); }
    uint32_t total_matmul_ops() const { return m_tiles() * n_tiles() * k_tiles(); }

    uint32_t tile_bytes() const { return tile_m * tile_n * element_size; }
    uint32_t a_tile_bytes() const { return tile_m * tile_k * element_size; }
    uint32_t b_tile_bytes() const { return tile_k * tile_n * element_size; }
    uint32_t c_tile_bytes() const { return tile_m * tile_n * element_size; }

    uint32_t compute_matmul_latency() const {
        if (matmul_latency > 0) return matmul_latency;
        // Systolic array latency: fill + compute + drain
        // For 16x16 array processing 16x16 tiles: ~16 + 16 + 16 = 48 cycles typical
        return systolic_rows + tile_k + systolic_cols;
    }

    uint64_t total_flops() const {
        return 2ULL * M * N * K;  // 2 ops per MAC
    }
};

// ============================================================================
// Trace Event Types
// ============================================================================

/// Memory hierarchy levels
enum class TraceLevel : uint8_t {
    HOST = 0,
    DMA = 1,
    L3 = 2,
    BLOCK_MOVER = 3,
    L2 = 4,
    STREAMER = 5,
    L1 = 6,
    COMPUTE = 7
};

inline const char* to_string(TraceLevel level) {
    switch (level) {
        case TraceLevel::HOST: return "HOST";
        case TraceLevel::DMA: return "DMA";
        case TraceLevel::L3: return "L3";
        case TraceLevel::BLOCK_MOVER: return "BLOCK_MOVER";
        case TraceLevel::L2: return "L2";
        case TraceLevel::STREAMER: return "STREAMER";
        case TraceLevel::L1: return "L1";
        case TraceLevel::COMPUTE: return "COMPUTE";
        default: return "UNKNOWN";
    }
}

/// Operand types in trace
enum class TraceOperandType : uint8_t {
    TILE_A = 0,
    TILE_B = 1,
    TILE_C = 2,  // Input C for accumulation
    TILE_D = 3,  // Output D
    BUFFER = 4
};

inline const char* to_string(TraceOperandType type) {
    switch (type) {
        case TraceOperandType::TILE_A: return "A";
        case TraceOperandType::TILE_B: return "B";
        case TraceOperandType::TILE_C: return "C";
        case TraceOperandType::TILE_D: return "D";
        case TraceOperandType::BUFFER: return "BUF";
        default: return "?";
    }
}

/// Operation types in trace (dataflow semantics - NO CACHE TERMINOLOGY)
/// See docs/kpu-execution-model.md for the credit-based dataflow model
enum class TraceOperation : uint8_t {
    // DMA operations
    DMA_PUSH,           // DMA pushes tile to L3 buffer (has credit)
    DMA_PULL,           // DMA pulls tile from L3 to host

    // BlockMover operations
    BM_PUSH,            // BlockMover pushes tile L3 -> L2 (has credit)
    BM_PULL,            // BlockMover pulls tile L2 -> L3

    // Streamer operations
    STR_FEED_A,         // Streamer feeds A tile L2 -> L1 west
    STR_FEED_B,         // Streamer feeds B tile L2 -> L1 north
    STR_DRAIN,          // Streamer drains accumulator -> L2

    // Compute operations
    COMPUTE,            // Systolic array active

    // Dataflow events (tokens)
    TILE_READY,         // Token: tile arrived at buffer (data downstream)
    BUFFER_AVAILABLE,   // Token: buffer has credit (credit upstream)

    // Progress tracking
    TILE_COMPLETE,      // Output tile fully computed
    LOOP_ITER,          // Loop iteration marker

    // Legacy (for compatibility during transition)
    LOAD,               // Deprecated: use DMA_PUSH
    STORE,              // Deprecated: use DMA_PULL
    PUSH_TO_L2,         // Deprecated: use BM_PUSH
    PULL_FROM_L2,       // Deprecated: use BM_PULL
    FEED_WEST,          // Deprecated: use STR_FEED_A
    FEED_NORTH,         // Deprecated: use STR_FEED_B
    DRAIN,              // Deprecated: use STR_DRAIN
    MATMUL              // Deprecated: use COMPUTE
};

inline const char* to_string(TraceOperation op) {
    switch (op) {
        case TraceOperation::DMA_PUSH: return "DMA_PUSH";
        case TraceOperation::DMA_PULL: return "DMA_PULL";
        case TraceOperation::BM_PUSH: return "BM_PUSH";
        case TraceOperation::BM_PULL: return "BM_PULL";
        case TraceOperation::STR_FEED_A: return "STR_FEED_A";
        case TraceOperation::STR_FEED_B: return "STR_FEED_B";
        case TraceOperation::STR_DRAIN: return "STR_DRAIN";
        case TraceOperation::COMPUTE: return "COMPUTE";
        case TraceOperation::TILE_READY: return "TILE_READY";
        case TraceOperation::BUFFER_AVAILABLE: return "BUFFER_AVAILABLE";
        case TraceOperation::TILE_COMPLETE: return "TILE_COMPLETE";
        case TraceOperation::LOOP_ITER: return "LOOP_ITER";
        // Legacy
        case TraceOperation::LOAD: return "LOAD";
        case TraceOperation::STORE: return "STORE";
        case TraceOperation::PUSH_TO_L2: return "PUSH_TO_L2";
        case TraceOperation::PULL_FROM_L2: return "PULL_FROM_L2";
        case TraceOperation::FEED_WEST: return "FEED_WEST";
        case TraceOperation::FEED_NORTH: return "FEED_NORTH";
        case TraceOperation::DRAIN: return "DRAIN";
        case TraceOperation::MATMUL: return "MATMUL";
        default: return "UNKNOWN";
    }
}

/// Trace event for visualization (dataflow semantics)
/// See docs/kpu-execution-model.md for the credit-based dataflow model
struct TraceEvent {
    // Core fields (order preserved for brace-initialization compatibility)
    uint64_t cycle = 0;             // When event occurs
    TraceLevel level = TraceLevel::DMA;           // Which level
    TraceOperation operation = TraceOperation::DMA_PUSH;   // What operation
    TraceOperandType operand = TraceOperandType::TILE_A;   // Which operand
    uint16_t tile_i = 0;            // Tile row index
    uint16_t tile_j = 0;            // Tile column index
    uint16_t tile_k = 0;            // Tile k index (for A, B)
    uint8_t src_buffer = 0;         // Source buffer (for moves)
    uint8_t dst_buffer = 0;         // Destination buffer (for moves)
    uint64_t duration = 0;          // Duration in cycles (0 for instant events)
    std::string name;               // Human-readable name

    // Additional dataflow fields (set explicitly, not via brace init)
    uint8_t buffer_id = 0;          // Which buffer (L3[i], L2[j], etc.)
    uint8_t l3_buffers_occupied = 0;   // How many L3 buffers currently hold tiles
    uint8_t l2_buffers_occupied = 0;   // How many L2 banks currently hold tiles

    // Loop state for visualization. The {} default-member-initializer
    // matters: without it, aggregate brace-init of TraceEvent that
    // omits loop_state triggers -Wmissing-field-initializers under -Werror
    // (~17 sites in this file). With it, the field is value-initialized
    // and the warning is suppressed.
    struct LoopState {
        uint32_t outer = 0;         // Outer loop position (i)
        uint32_t middle = 0;        // Middle loop position (j)
        uint32_t inner = 0;         // Inner loop position (k)
    } loop_state{};

    std::string operand_id() const {
        std::string id = to_string(operand);
        id += "[" + std::to_string(tile_i) + "," + std::to_string(tile_j);
        if (operand == TraceOperandType::TILE_A || operand == TraceOperandType::TILE_B) {
            id += "," + std::to_string(tile_k);
        }
        id += "]";
        return id;
    }

    std::string buffer_str() const {
        std::string loc;
        switch (level) {
            case TraceLevel::L3: loc = "L3"; break;
            case TraceLevel::L2: loc = "L2"; break;
            case TraceLevel::L1: loc = "L1"; break;
            default: loc = "?"; break;
        }
        return loc + "[" + std::to_string(buffer_id) + "]";
    }
};

// ============================================================================
// Execution Statistics
// ============================================================================

struct TiledMatmulStats {
    uint64_t total_cycles = 0;
    uint64_t compute_cycles = 0;    // Actual matmul time
    uint64_t memory_stall_cycles = 0;

    uint32_t dma_loads = 0;
    uint32_t dma_stores = 0;
    uint64_t dma_bytes = 0;

    uint32_t bm_pushes = 0;
    uint32_t bm_pulls = 0;
    uint64_t bm_bytes = 0;

    uint32_t str_feeds = 0;
    uint32_t str_drains = 0;
    uint32_t matmuls = 0;
    uint64_t flops = 0;

    // Buffer utilization statistics (dataflow model)
    uint32_t l3_tile_ready_events = 0;    // How many TILE_READY tokens emitted
    uint32_t l3_buffer_available_events = 0; // How many credits returned
    uint32_t l2_tile_ready_events = 0;
    uint32_t l2_buffer_available_events = 0;

    // Utilization
    double compute_utilization() const {
        return total_cycles > 0 ?
            static_cast<double>(compute_cycles) / static_cast<double>(total_cycles) : 0.0;
    }

    double effective_tflops(double clock_ghz) const {
        return total_cycles > 0 ?
            (static_cast<double>(flops) / static_cast<double>(total_cycles)) * clock_ghz : 0.0;
    }
};

// ============================================================================
// Tiled Matrix Multiplication Program
// ============================================================================

/// Generates and executes tiled matmul with timing model
/// Uses credit-based dataflow execution (see docs/kpu-execution-model.md)
class TiledMatmulProgram {
public:
    explicit TiledMatmulProgram(const TiledMatmulConfig& config)
        : config_(config) {
        reset();
    }

    /// Reset to initial state
    void reset() {
        current_cycle_ = 0;
        trace_.clear();
        stats_ = TiledMatmulStats{};

        // Initialize buffer availability (cycle when buffer becomes available)
        // Buffer occupancy is implicitly tracked: buffer is occupied if l3_available_[i] > current_cycle_
        for (int i = 0; i < 4; ++i) {
            l3_available_[i] = 0;
        }
        for (int i = 0; i < 8; ++i) {
            l2_available_[i] = 0;
        }

        compute_available_ = 0;
        next_a_buffer_ = 0;
        next_b_buffer_ = 0;

        // Reset loop state
        loop_i_ = loop_j_ = loop_k_ = 0;
    }

    /// Execute the full tiled matmul using configured loop order
    void execute() {
        reset();

        // Execute according to configured loop order
        switch (config_.loop_order) {
            case LoopOrder::IJK:
                execute_ijk();
                break;
            case LoopOrder::JIK:
                execute_jik();
                break;
            case LoopOrder::IKJ:
                execute_ikj();
                break;
            case LoopOrder::KIJ:
                execute_kij();
                break;
            case LoopOrder::KJI:
                execute_kji();
                break;
            case LoopOrder::JKI:
                execute_jki();
                break;
            case LoopOrder::BLOCKED:
                execute_blocked();
                break;
        }

        stats_.total_cycles = current_cycle_;
        stats_.flops = config_.total_flops();
    }

    /// Execute with pipelined output tiles (more realistic)
    /// Note: This uses the configured loop order with pipelining
    void execute_pipelined() {
        reset();

        // Pipelined execution uses IKJ order for best A-tile buffer reuse
        // The key insight: for each row i, we stream through k tiles while
        // computing j tiles. A[i,k] stays in L3 buffer for all j iterations.
        execute_ikj_pipelined();

        stats_.total_cycles = current_cycle_;
        stats_.flops = config_.total_flops();
    }

    /// Get execution trace for visualization
    const std::vector<TraceEvent>& trace() const { return trace_; }

    /// Get execution statistics
    const TiledMatmulStats& stats() const { return stats_; }

    /// Get configuration
    const TiledMatmulConfig& config() const { return config_; }

    /// Export trace to JSON file for animation
    bool write_trace_json(const std::string& filename) const;

    /// Generate JSON string
    std::string to_json() const;

private:
    TiledMatmulConfig config_;
    uint64_t current_cycle_ = 0;
    std::vector<TraceEvent> trace_;
    TiledMatmulStats stats_;

    // Resource availability tracking (cycle when buffer becomes available)
    // This implicitly tracks buffer occupancy: buffer is occupied if l3_available_[i] > current_cycle_
    uint64_t l3_available_[4] = {0};
    uint64_t l2_available_[8] = {0};
    uint64_t compute_available_ = 0;

    // Double-buffer indices (0 or 1)
    int next_a_buffer_ = 0;
    int next_b_buffer_ = 0;

    // Current loop state for visualization
    uint32_t loop_i_ = 0;
    uint32_t loop_j_ = 0;
    uint32_t loop_k_ = 0;

    /// Execute a single output tile D[i,j] with all k iterations
    void execute_output_tile(uint32_t i, uint32_t j, uint32_t k_tiles) {
        // Load initial C if accumulating
        if (config_.accumulate_c) {
            load_c_tile(i, j);
        }

        // Process all k tiles
        for (uint32_t k = 0; k < k_tiles; ++k) {
            // DMA: Load A[i,k] and B[k,j]
            uint64_t a_ready = load_a_tile(i, k);
            uint64_t b_ready = load_b_tile(k, j);

            // BlockMover: Push to L2
            uint64_t a_l2_ready = push_to_l2_a(i, k, a_ready);
            uint64_t b_l2_ready = push_to_l2_b(k, j, b_ready);

            // Streamer: Feed to L1
            uint64_t a_l1_ready = feed_west(i, k, a_l2_ready);
            uint64_t b_l1_ready = feed_north(k, j, b_l2_ready);

            // Compute: MATMUL
            uint64_t compute_done = execute_matmul(i, j, k,
                std::max(a_l1_ready, b_l1_ready));

            compute_available_ = compute_done;
        }

        // Drain result from accumulator
        uint64_t d_l2_ready = drain_result(i, j, compute_available_);

        // BlockMover: Pull to L3
        uint64_t d_l3_ready = pull_from_l2(i, j, d_l2_ready);

        // DMA: Store to host
        store_d_tile(i, j, d_l3_ready);
    }

    /// Execute output tile with pipelining (overlapped k iterations)
    void execute_output_tile_pipelined(uint32_t i, uint32_t j, uint32_t k_tiles) {
        // Track when each stage completes for each k
        std::vector<uint64_t> dma_a_done(k_tiles);
        std::vector<uint64_t> dma_b_done(k_tiles);
        std::vector<uint64_t> l2_a_ready(k_tiles);
        std::vector<uint64_t> l2_b_ready(k_tiles);
        std::vector<uint64_t> compute_done(k_tiles);

        // Load initial C if accumulating (before first k)
        uint64_t c_ready = 0;
        if (config_.accumulate_c) {
            c_ready = load_c_tile(i, j);
        }

        // Start DMA for k=0
        dma_a_done[0] = load_a_tile(i, 0);
        dma_b_done[0] = load_b_tile(0, j);

        // Pipeline the k iterations
        for (uint32_t k = 0; k < k_tiles; ++k) {
            // Start DMA for k+1 while processing k
            if (k + 1 < k_tiles) {
                // Use double-buffered L3 slots
                dma_a_done[k + 1] = load_a_tile_prefetch(i, k + 1,
                    std::max(current_cycle_, dma_a_done[k]));
                dma_b_done[k + 1] = load_b_tile_prefetch(k + 1, j,
                    std::max(current_cycle_, dma_b_done[k]));
            }

            // BlockMover: Push k to L2
            l2_a_ready[k] = push_to_l2_a(i, k, dma_a_done[k]);
            l2_b_ready[k] = push_to_l2_b(k, j, dma_b_done[k]);

            // Streamer + Compute for k
            uint64_t a_l1 = feed_west(i, k, l2_a_ready[k]);
            uint64_t b_l1 = feed_north(k, j, l2_b_ready[k]);

            uint64_t ready_time = std::max({a_l1, b_l1, compute_available_});
            if (k == 0 && config_.accumulate_c) {
                ready_time = std::max(ready_time, c_ready);
            }

            compute_done[k] = execute_matmul(i, j, k, ready_time);
            compute_available_ = compute_done[k];
        }

        // Drain and store final result
        uint64_t d_l2_ready = drain_result(i, j, compute_available_);
        uint64_t d_l3_ready = pull_from_l2(i, j, d_l2_ready);
        store_d_tile(i, j, d_l3_ready);
    }

    // ========================================================================
    // DMA Operations
    // ========================================================================

    uint64_t load_a_tile(uint32_t i, uint32_t k) {
        uint8_t buffer = config_.l3_a_buffers[next_a_buffer_];
        next_a_buffer_ = 1 - next_a_buffer_;

        uint64_t start = std::max(current_cycle_, l3_available_[buffer]);
        uint64_t latency = config_.dma_load_latency +
            (config_.a_tile_bytes() / config_.dma_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::DMA, TraceOperation::LOAD,
            TraceOperandType::TILE_A,
            static_cast<uint16_t>(i), 0, static_cast<uint16_t>(k),
            0, buffer, latency,
            "LOAD A[" + std::to_string(i) + "," + std::to_string(k) + "]"
        });

        trace_.push_back({
            end, TraceLevel::L3, TraceOperation::TILE_READY,
            TraceOperandType::TILE_A,
            static_cast<uint16_t>(i), 0, static_cast<uint16_t>(k),
            buffer, buffer, 0, ""
        });

        l3_available_[buffer] = end;
        stats_.dma_loads++;
        stats_.dma_bytes += config_.a_tile_bytes();
        current_cycle_ = std::max(current_cycle_, end);

        return end;
    }

    uint64_t load_a_tile_prefetch(uint32_t i, uint32_t k, uint64_t earliest) {
        uint8_t buffer = config_.l3_a_buffers[next_a_buffer_];
        next_a_buffer_ = 1 - next_a_buffer_;

        uint64_t start = std::max(earliest, l3_available_[buffer]);
        uint64_t latency = config_.dma_load_latency +
            (config_.a_tile_bytes() / config_.dma_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::DMA, TraceOperation::LOAD,
            TraceOperandType::TILE_A,
            static_cast<uint16_t>(i), 0, static_cast<uint16_t>(k),
            0, buffer, latency,
            "LOAD A[" + std::to_string(i) + "," + std::to_string(k) + "] (prefetch)"
        });

        trace_.push_back({
            end, TraceLevel::L3, TraceOperation::TILE_READY,
            TraceOperandType::TILE_A,
            static_cast<uint16_t>(i), 0, static_cast<uint16_t>(k),
            buffer, buffer, 0, ""
        });

        l3_available_[buffer] = end;
        stats_.dma_loads++;
        stats_.dma_bytes += config_.a_tile_bytes();

        return end;
    }

    uint64_t load_b_tile(uint32_t k, uint32_t j) {
        uint8_t buffer = config_.l3_b_buffers[next_b_buffer_];
        next_b_buffer_ = 1 - next_b_buffer_;

        uint64_t start = std::max(current_cycle_, l3_available_[buffer]);
        uint64_t latency = config_.dma_load_latency +
            (config_.b_tile_bytes() / config_.dma_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::DMA, TraceOperation::LOAD,
            TraceOperandType::TILE_B,
            0, static_cast<uint16_t>(j), static_cast<uint16_t>(k),
            0, buffer, latency,
            "LOAD B[" + std::to_string(k) + "," + std::to_string(j) + "]"
        });

        trace_.push_back({
            end, TraceLevel::L3, TraceOperation::TILE_READY,
            TraceOperandType::TILE_B,
            0, static_cast<uint16_t>(j), static_cast<uint16_t>(k),
            buffer, buffer, 0, ""
        });

        l3_available_[buffer] = end;
        stats_.dma_loads++;
        stats_.dma_bytes += config_.b_tile_bytes();
        current_cycle_ = std::max(current_cycle_, end);

        return end;
    }

    uint64_t load_b_tile_prefetch(uint32_t k, uint32_t j, uint64_t earliest) {
        uint8_t buffer = config_.l3_b_buffers[next_b_buffer_];
        next_b_buffer_ = 1 - next_b_buffer_;

        uint64_t start = std::max(earliest, l3_available_[buffer]);
        uint64_t latency = config_.dma_load_latency +
            (config_.b_tile_bytes() / config_.dma_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::DMA, TraceOperation::LOAD,
            TraceOperandType::TILE_B,
            0, static_cast<uint16_t>(j), static_cast<uint16_t>(k),
            0, buffer, latency,
            "LOAD B[" + std::to_string(k) + "," + std::to_string(j) + "] (prefetch)"
        });

        trace_.push_back({
            end, TraceLevel::L3, TraceOperation::TILE_READY,
            TraceOperandType::TILE_B,
            0, static_cast<uint16_t>(j), static_cast<uint16_t>(k),
            buffer, buffer, 0, ""
        });

        l3_available_[buffer] = end;
        stats_.dma_loads++;
        stats_.dma_bytes += config_.b_tile_bytes();

        return end;
    }

    uint64_t load_c_tile(uint32_t i, uint32_t j) {
        uint8_t l3_buffer = 0;  // Use first buffer for C
        uint64_t start = current_cycle_;
        uint64_t latency = config_.dma_load_latency +
            (config_.c_tile_bytes() / config_.dma_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::DMA, TraceOperation::LOAD,
            TraceOperandType::TILE_C,
            static_cast<uint16_t>(i), static_cast<uint16_t>(j), 0,
            0, l3_buffer, latency,
            "LOAD C[" + std::to_string(i) + "," + std::to_string(j) + "]"
        });

        stats_.dma_loads++;
        stats_.dma_bytes += config_.c_tile_bytes();
        current_cycle_ = end;

        return end;
    }

    void store_d_tile(uint32_t i, uint32_t j, uint64_t ready_time) {
        uint64_t start = ready_time;
        uint64_t latency = config_.dma_store_latency +
            (config_.c_tile_bytes() / config_.dma_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::DMA, TraceOperation::STORE,
            TraceOperandType::TILE_D,
            static_cast<uint16_t>(i), static_cast<uint16_t>(j), 0,
            0, 0, latency,
            "STORE D[" + std::to_string(i) + "," + std::to_string(j) + "]"
        });

        stats_.dma_stores++;
        stats_.dma_bytes += config_.c_tile_bytes();
        current_cycle_ = std::max(current_cycle_, end);
    }

    // ========================================================================
    // BlockMover Operations
    // ========================================================================

    /// BlockMover pushes A tile from L3 to L2 (dataflow semantics)
    /// When tile is consumed from L3, returns BUFFER_AVAILABLE credit upstream
    uint64_t push_to_l2_a(uint32_t i, uint32_t k, uint64_t l3_ready) {
        uint8_t l3_buffer = config_.l3_a_buffers[(k + 1) % 2];
        uint8_t l2_bank = config_.l2_a_banks[k % 2];
        uint64_t start = std::max(l3_ready, l2_available_[l2_bank]);
        uint64_t latency = config_.bm_push_latency +
            (config_.a_tile_bytes() / config_.bm_bandwidth);
        uint64_t end = start + latency;

        // BM_PUSH: BlockMover pushes tile L3 -> L2
        TraceEvent push_event;
        push_event.cycle = start;
        push_event.level = TraceLevel::BLOCK_MOVER;
        push_event.operation = TraceOperation::BM_PUSH;
        push_event.operand = TraceOperandType::TILE_A;
        push_event.tile_i = static_cast<uint16_t>(i);
        push_event.tile_k = static_cast<uint16_t>(k);
        push_event.src_buffer = l3_buffer;
        push_event.dst_buffer = l2_bank;
        push_event.duration = latency;
        push_event.name = "BM_PUSH A[" + std::to_string(i) + "," + std::to_string(k) +
                         "] L3[" + std::to_string(l3_buffer) + "] -> L2[" + std::to_string(l2_bank) + "]";
        trace_.push_back(push_event);

        // TILE_READY: Tile arrived at L2 bank
        TraceEvent ready_event;
        ready_event.cycle = end;
        ready_event.level = TraceLevel::L2;
        ready_event.operation = TraceOperation::TILE_READY;
        ready_event.operand = TraceOperandType::TILE_A;
        ready_event.tile_i = static_cast<uint16_t>(i);
        ready_event.tile_k = static_cast<uint16_t>(k);
        ready_event.buffer_id = l2_bank;
        ready_event.name = "TILE_READY A[" + std::to_string(i) + "," + std::to_string(k) +
                          "] @ L2[" + std::to_string(l2_bank) + "]";
        trace_.push_back(ready_event);

        // BUFFER_AVAILABLE: L3 buffer is free (credit returned to DMA)
        TraceEvent credit_event;
        credit_event.cycle = end;
        credit_event.level = TraceLevel::L3;
        credit_event.operation = TraceOperation::BUFFER_AVAILABLE;
        credit_event.operand = TraceOperandType::BUFFER;
        credit_event.buffer_id = l3_buffer;
        credit_event.name = "BUFFER_AVAILABLE L3[" + std::to_string(l3_buffer) + "]";
        trace_.push_back(credit_event);

        l2_available_[l2_bank] = end;
        stats_.bm_pushes++;
        stats_.bm_bytes += config_.a_tile_bytes();

        return end;
    }

    /// BlockMover pushes B tile from L3 to L2 (dataflow semantics)
    uint64_t push_to_l2_b(uint32_t k, uint32_t j, uint64_t l3_ready) {
        uint8_t l3_buffer = config_.l3_b_buffers[(k + 1) % 2];
        uint8_t l2_bank = config_.l2_b_banks[k % 2];
        uint64_t start = std::max(l3_ready, l2_available_[l2_bank]);
        uint64_t latency = config_.bm_push_latency +
            (config_.b_tile_bytes() / config_.bm_bandwidth);
        uint64_t end = start + latency;

        // BM_PUSH: BlockMover pushes tile L3 -> L2
        TraceEvent push_event;
        push_event.cycle = start;
        push_event.level = TraceLevel::BLOCK_MOVER;
        push_event.operation = TraceOperation::BM_PUSH;
        push_event.operand = TraceOperandType::TILE_B;
        push_event.tile_j = static_cast<uint16_t>(j);
        push_event.tile_k = static_cast<uint16_t>(k);
        push_event.src_buffer = l3_buffer;
        push_event.dst_buffer = l2_bank;
        push_event.duration = latency;
        push_event.name = "BM_PUSH B[" + std::to_string(k) + "," + std::to_string(j) +
                         "] L3[" + std::to_string(l3_buffer) + "] -> L2[" + std::to_string(l2_bank) + "]";
        trace_.push_back(push_event);

        // TILE_READY: Tile arrived at L2 bank
        TraceEvent ready_event;
        ready_event.cycle = end;
        ready_event.level = TraceLevel::L2;
        ready_event.operation = TraceOperation::TILE_READY;
        ready_event.operand = TraceOperandType::TILE_B;
        ready_event.tile_j = static_cast<uint16_t>(j);
        ready_event.tile_k = static_cast<uint16_t>(k);
        ready_event.buffer_id = l2_bank;
        ready_event.name = "TILE_READY B[" + std::to_string(k) + "," + std::to_string(j) +
                          "] @ L2[" + std::to_string(l2_bank) + "]";
        trace_.push_back(ready_event);

        // BUFFER_AVAILABLE: L3 buffer is free (credit returned to DMA)
        TraceEvent credit_event;
        credit_event.cycle = end;
        credit_event.level = TraceLevel::L3;
        credit_event.operation = TraceOperation::BUFFER_AVAILABLE;
        credit_event.operand = TraceOperandType::BUFFER;
        credit_event.buffer_id = l3_buffer;
        credit_event.name = "BUFFER_AVAILABLE L3[" + std::to_string(l3_buffer) + "]";
        trace_.push_back(credit_event);

        l2_available_[l2_bank] = end;
        stats_.bm_pushes++;
        stats_.bm_bytes += config_.b_tile_bytes();

        return end;
    }

    uint64_t pull_from_l2(uint32_t i, uint32_t j, uint64_t l2_ready) {
        uint8_t l2_bank = config_.l2_d_banks[0];
        uint8_t l3_buffer = 0;  // Reuse buffer for output
        uint64_t start = l2_ready;
        uint64_t latency = config_.bm_pull_latency +
            (config_.c_tile_bytes() / config_.bm_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::BLOCK_MOVER, TraceOperation::PULL_FROM_L2,
            TraceOperandType::TILE_D,
            static_cast<uint16_t>(i), static_cast<uint16_t>(j), 0,
            l2_bank, l3_buffer, latency,
            "PULL D[" + std::to_string(i) + "," + std::to_string(j) + "] <- L2"
        });

        trace_.push_back({
            end, TraceLevel::L3, TraceOperation::TILE_READY,
            TraceOperandType::TILE_D,
            static_cast<uint16_t>(i), static_cast<uint16_t>(j), 0,
            l3_buffer, l3_buffer, 0, ""
        });

        stats_.bm_pulls++;
        stats_.bm_bytes += config_.c_tile_bytes();

        return end;
    }

    // ========================================================================
    // Streamer Operations
    // ========================================================================

    uint64_t feed_west(uint32_t i, uint32_t k, uint64_t l2_ready) {
        uint64_t start = l2_ready;
        uint64_t latency = config_.str_feed_latency +
            (config_.a_tile_bytes() / config_.str_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::STREAMER, TraceOperation::FEED_WEST,
            TraceOperandType::TILE_A,
            static_cast<uint16_t>(i), 0, static_cast<uint16_t>(k),
            config_.l2_a_banks[k % 2], 0, latency,
            "FEED_WEST A[" + std::to_string(i) + "," + std::to_string(k) + "]"
        });

        stats_.str_feeds++;
        return end;
    }

    uint64_t feed_north(uint32_t k, uint32_t j, uint64_t l2_ready) {
        uint64_t start = l2_ready;
        uint64_t latency = config_.str_feed_latency +
            (config_.b_tile_bytes() / config_.str_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::STREAMER, TraceOperation::FEED_NORTH,
            TraceOperandType::TILE_B,
            0, static_cast<uint16_t>(j), static_cast<uint16_t>(k),
            config_.l2_b_banks[k % 2], 0, latency,
            "FEED_NORTH B[" + std::to_string(k) + "," + std::to_string(j) + "]"
        });

        stats_.str_feeds++;
        return end;
    }

    uint64_t execute_matmul(uint32_t i, uint32_t j, uint32_t k, uint64_t ready_time) {
        uint64_t start = std::max(ready_time, compute_available_);
        uint64_t latency = config_.compute_matmul_latency();
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::COMPUTE, TraceOperation::MATMUL,
            TraceOperandType::TILE_D,
            static_cast<uint16_t>(i), static_cast<uint16_t>(j), static_cast<uint16_t>(k),
            0, 0, latency,
            "MATMUL D[" + std::to_string(i) + "," + std::to_string(j) +
            "] += A[" + std::to_string(i) + "," + std::to_string(k) +
            "] * B[" + std::to_string(k) + "," + std::to_string(j) + "]"
        });

        stats_.matmuls++;
        stats_.compute_cycles += latency;

        return end;
    }

    uint64_t drain_result(uint32_t i, uint32_t j, uint64_t compute_done) {
        uint8_t l2_bank = config_.l2_d_banks[0];
        uint64_t start = compute_done;
        uint64_t latency = config_.str_drain_latency +
            (config_.c_tile_bytes() / config_.str_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::STREAMER, TraceOperation::DRAIN,
            TraceOperandType::TILE_D,
            static_cast<uint16_t>(i), static_cast<uint16_t>(j), 0,
            0, l2_bank, latency,
            "DRAIN D[" + std::to_string(i) + "," + std::to_string(j) + "]"
        });

        trace_.push_back({
            end, TraceLevel::L2, TraceOperation::TILE_READY,
            TraceOperandType::TILE_D,
            static_cast<uint16_t>(i), static_cast<uint16_t>(j), 0,
            l2_bank, l2_bank, 0, ""
        });

        stats_.str_drains++;
        return end;
    }

    // ========================================================================
    // Dataflow-Semantic Tile Loading (Credit-Based)
    // See docs/kpu-execution-model.md for the execution model
    // ========================================================================

    /// Load A tile: DMA waits for L3 buffer credit, then pushes tile
    /// Returns cycle when tile is ready (TILE_READY event)
    uint64_t load_a_tile_dataflow(uint32_t i, uint32_t k) {
        // Select buffer using double-buffering
        uint8_t buffer = config_.l3_a_buffers[next_a_buffer_];
        next_a_buffer_ = 1 - next_a_buffer_;

        // Wait for buffer credit (buffer available)
        uint64_t start = std::max(current_cycle_, l3_available_[buffer]);

        // DMA transfer latency
        uint64_t latency = config_.dma_load_latency +
            (config_.a_tile_bytes() / config_.dma_bandwidth);
        uint64_t end = start + latency;

        // Record DMA_PUSH event (producer pushes to buffer)
        TraceEvent dma_event;
        dma_event.cycle = start;
        dma_event.level = TraceLevel::DMA;
        dma_event.operation = TraceOperation::DMA_PUSH;
        dma_event.operand = TraceOperandType::TILE_A;
        dma_event.tile_i = static_cast<uint16_t>(i);
        dma_event.tile_j = 0;
        dma_event.tile_k = static_cast<uint16_t>(k);
        dma_event.buffer_id = buffer;
        dma_event.dst_buffer = buffer;
        dma_event.duration = latency;
        dma_event.loop_state = {loop_i_, loop_j_, loop_k_};
        dma_event.l3_buffers_occupied = count_l3_occupied();
        dma_event.name = "DMA_PUSH A[" + std::to_string(i) + "," + std::to_string(k) +
                        "] -> L3[" + std::to_string(buffer) + "]";
        trace_.push_back(dma_event);

        // Record TILE_READY event (data token arrives at L3)
        TraceEvent ready_event;
        ready_event.cycle = end;
        ready_event.level = TraceLevel::L3;
        ready_event.operation = TraceOperation::TILE_READY;
        ready_event.operand = TraceOperandType::TILE_A;
        ready_event.tile_i = static_cast<uint16_t>(i);
        ready_event.tile_j = 0;
        ready_event.tile_k = static_cast<uint16_t>(k);
        ready_event.buffer_id = buffer;
        ready_event.loop_state = {loop_i_, loop_j_, loop_k_};
        ready_event.name = "TILE_READY A[" + std::to_string(i) + "," + std::to_string(k) +
                          "] @ L3[" + std::to_string(buffer) + "]";
        trace_.push_back(ready_event);

        // Buffer is occupied until consumed by BlockMover
        l3_available_[buffer] = end;

        stats_.dma_loads++;
        stats_.dma_bytes += config_.a_tile_bytes();

        return end;
    }

    /// Load B tile: DMA waits for L3 buffer credit, then pushes tile
    uint64_t load_b_tile_dataflow(uint32_t k, uint32_t j) {
        // Select buffer using double-buffering
        uint8_t buffer = config_.l3_b_buffers[next_b_buffer_];
        next_b_buffer_ = 1 - next_b_buffer_;

        // Wait for buffer credit (buffer available)
        uint64_t start = std::max(current_cycle_, l3_available_[buffer]);

        // DMA transfer latency
        uint64_t latency = config_.dma_load_latency +
            (config_.b_tile_bytes() / config_.dma_bandwidth);
        uint64_t end = start + latency;

        // Record DMA_PUSH event
        TraceEvent dma_event;
        dma_event.cycle = start;
        dma_event.level = TraceLevel::DMA;
        dma_event.operation = TraceOperation::DMA_PUSH;
        dma_event.operand = TraceOperandType::TILE_B;
        dma_event.tile_i = 0;
        dma_event.tile_j = static_cast<uint16_t>(j);
        dma_event.tile_k = static_cast<uint16_t>(k);
        dma_event.buffer_id = buffer;
        dma_event.dst_buffer = buffer;
        dma_event.duration = latency;
        dma_event.loop_state = {loop_i_, loop_j_, loop_k_};
        dma_event.l3_buffers_occupied = count_l3_occupied();
        dma_event.name = "DMA_PUSH B[" + std::to_string(k) + "," + std::to_string(j) +
                        "] -> L3[" + std::to_string(buffer) + "]";
        trace_.push_back(dma_event);

        // Record TILE_READY event
        TraceEvent ready_event;
        ready_event.cycle = end;
        ready_event.level = TraceLevel::L3;
        ready_event.operation = TraceOperation::TILE_READY;
        ready_event.operand = TraceOperandType::TILE_B;
        ready_event.tile_i = 0;
        ready_event.tile_j = static_cast<uint16_t>(j);
        ready_event.tile_k = static_cast<uint16_t>(k);
        ready_event.buffer_id = buffer;
        ready_event.loop_state = {loop_i_, loop_j_, loop_k_};
        ready_event.name = "TILE_READY B[" + std::to_string(k) + "," + std::to_string(j) +
                          "] @ L3[" + std::to_string(buffer) + "]";
        trace_.push_back(ready_event);

        // Buffer is occupied until consumed by BlockMover
        l3_available_[buffer] = end;

        stats_.dma_loads++;
        stats_.dma_bytes += config_.b_tile_bytes();

        return end;
    }

    /// Count how many L3 buffers are currently occupied (timing-based)
    uint8_t count_l3_occupied() const {
        uint8_t count = 0;
        for (uint8_t i = 0; i < config_.num_l3_tiles; ++i) {
            if (l3_available_[i] > current_cycle_) {
                count++;
            }
        }
        return count;
    }

    /// Count how many L2 banks are currently occupied (timing-based)
    uint8_t count_l2_occupied() const {
        uint8_t count = 0;
        for (uint8_t i = 0; i < config_.num_l2_banks; ++i) {
            if (l2_available_[i] > current_cycle_) {
                count++;
            }
        }
        return count;
    }

    /// Execute one matmul operation using dataflow-semantic loading
    void execute_single_matmul_dataflow(uint32_t i, uint32_t j, uint32_t k,
                                        bool is_first_k, bool is_last_k) {
        loop_i_ = i;
        loop_j_ = j;
        loop_k_ = k;

        // Load C for first k iteration if accumulating
        if (is_first_k && config_.accumulate_c) {
            load_c_tile(i, j);
        }

        // Dataflow tile loading (DMA waits for credit, then pushes)
        uint64_t a_ready = load_a_tile_dataflow(i, k);
        uint64_t b_ready = load_b_tile_dataflow(k, j);

        // BlockMover: Push to L2
        uint64_t a_l2_ready = push_to_l2_a(i, k, a_ready);
        uint64_t b_l2_ready = push_to_l2_b(k, j, b_ready);

        // Streamer: Feed to L1
        uint64_t a_l1_ready = feed_west(i, k, a_l2_ready);
        uint64_t b_l1_ready = feed_north(k, j, b_l2_ready);

        // Compute: MATMUL
        uint64_t compute_done = execute_matmul(i, j, k,
            std::max(a_l1_ready, b_l1_ready));
        compute_available_ = compute_done;

        // Drain and store on last k iteration
        if (is_last_k) {
            uint64_t d_l2_ready = drain_result(i, j, compute_available_);
            uint64_t d_l3_ready = pull_from_l2(i, j, d_l2_ready);
            store_d_tile(i, j, d_l3_ready);

            // Record tile completion
            TraceEvent complete_event;
            complete_event.cycle = current_cycle_;
            complete_event.level = TraceLevel::L3;
            complete_event.operation = TraceOperation::TILE_COMPLETE;
            complete_event.operand = TraceOperandType::TILE_D;
            complete_event.tile_i = static_cast<uint16_t>(i);
            complete_event.tile_j = static_cast<uint16_t>(j);
            complete_event.loop_state = {loop_i_, loop_j_, loop_k_};
            complete_event.name = "COMPLETE D[" + std::to_string(i) + "," + std::to_string(j) + "]";
            trace_.push_back(complete_event);
        }
    }

    // ========================================================================
    // Loop Order Implementations
    // ========================================================================

    /// IJK: for i: for j: for k: - A-row stays in L3 buffer for K iterations
    void execute_ijk() {
        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        for (uint32_t i = 0; i < m_tiles; ++i) {
            for (uint32_t j = 0; j < n_tiles; ++j) {
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    execute_single_matmul_dataflow(i, j, k,
                        k == 0, k == k_tiles - 1);
                }
            }
        }
    }

    /// JIK: for j: for i: for k: - B-column stays in L3 buffer for K iterations
    void execute_jik() {
        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        for (uint32_t j = 0; j < n_tiles; ++j) {
            for (uint32_t i = 0; i < m_tiles; ++i) {
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    execute_single_matmul_dataflow(i, j, k,
                        k == 0, k == k_tiles - 1);
                }
            }
        }
    }

    /// IKJ: for i: for k: for j: - A[i,k] reused across all j (best A reuse)
    void execute_ikj() {
        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        for (uint32_t i = 0; i < m_tiles; ++i) {
            for (uint32_t k = 0; k < k_tiles; ++k) {
                for (uint32_t j = 0; j < n_tiles; ++j) {
                    execute_single_matmul_dataflow(i, j, k,
                        k == 0, k == k_tiles - 1);
                }
            }
        }
    }

    /// KIJ: for k: for i: for j: - B[k,:] reused across all i
    void execute_kij() {
        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        for (uint32_t k = 0; k < k_tiles; ++k) {
            for (uint32_t i = 0; i < m_tiles; ++i) {
                for (uint32_t j = 0; j < n_tiles; ++j) {
                    execute_single_matmul_dataflow(i, j, k,
                        k == 0, k == k_tiles - 1);
                }
            }
        }
    }

    /// KJI: for k: for j: for i: - A[:,k] column reused across all j
    void execute_kji() {
        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        for (uint32_t k = 0; k < k_tiles; ++k) {
            for (uint32_t j = 0; j < n_tiles; ++j) {
                for (uint32_t i = 0; i < m_tiles; ++i) {
                    execute_single_matmul_dataflow(i, j, k,
                        k == 0, k == k_tiles - 1);
                }
            }
        }
    }

    /// JKI: for j: for k: for i: - B[:,j] column reused across all k
    void execute_jki() {
        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        for (uint32_t j = 0; j < n_tiles; ++j) {
            for (uint32_t k = 0; k < k_tiles; ++k) {
                for (uint32_t i = 0; i < m_tiles; ++i) {
                    execute_single_matmul_dataflow(i, j, k,
                        k == 0, k == k_tiles - 1);
                }
            }
        }
    }

    /// BLOCKED: 2-level blocking for better cache utilization
    void execute_blocked() {
        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        uint32_t bi = config_.block_i;
        uint32_t bj = config_.block_j;
        uint32_t bk = config_.block_k;

        // Outer loops over blocks
        for (uint32_t ii = 0; ii < m_tiles; ii += bi) {
            for (uint32_t jj = 0; jj < n_tiles; jj += bj) {
                for (uint32_t kk = 0; kk < k_tiles; kk += bk) {
                    // Inner loops within block
                    uint32_t i_end = std::min(ii + bi, m_tiles);
                    uint32_t j_end = std::min(jj + bj, n_tiles);
                    uint32_t k_end = std::min(kk + bk, k_tiles);

                    for (uint32_t i = ii; i < i_end; ++i) {
                        for (uint32_t k = kk; k < k_end; ++k) {
                            for (uint32_t j = jj; j < j_end; ++j) {
                                // is_first_k and is_last_k need to consider
                                // the global k range, not just this block
                                bool is_first = (kk == 0 && k == kk);
                                bool is_last = (k == k_tiles - 1);

                                execute_single_matmul_dataflow(i, j, k,
                                    is_first, is_last);
                            }
                        }
                    }
                }
            }
        }
    }

    /// IKJ with pipelining - overlaps DMA with compute
    /// Uses credit-based dataflow: DMA waits for buffer credit, then pushes tile
    /// See docs/kpu-execution-model.md for the dataflow model
    void execute_ikj_pipelined() {
        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        for (uint32_t i = 0; i < m_tiles; ++i) {
            loop_i_ = i;

            for (uint32_t k = 0; k < k_tiles; ++k) {
                loop_k_ = k;

                // DATAFLOW: DMA pushes A[i,k] to L3 buffer
                // DMA waits for BUFFER_AVAILABLE credit implicitly via l3_available_
                // Buffer occupancy is tracked by l3_available_[buffer] > current_cycle_
                uint64_t a_l3_ready = load_a_tile_dataflow(i, k);
                stats_.l3_tile_ready_events++;

                // Push A to L2 (A tile stays in L3 for all j iterations - buffer reuse)
                uint64_t a_l2_ready = push_to_l2_a(i, k, a_l3_ready);

                // Stream through all j columns - A[i,k] is reused for all j
                for (uint32_t j = 0; j < n_tiles; ++j) {
                    loop_j_ = j;

                    // Load C for first k iteration
                    if (k == 0 && config_.accumulate_c) {
                        load_c_tile(i, j);
                    }

                    // DATAFLOW: DMA pushes B[k,j] to L3 buffer
                    uint64_t b_l3_ready = load_b_tile_dataflow(k, j);
                    stats_.l3_tile_ready_events++;

                    uint64_t b_l2_ready = push_to_l2_b(k, j, b_l3_ready);

                    // Feed tiles to L1 and compute
                    uint64_t a_l1 = feed_west(i, k, a_l2_ready);
                    uint64_t b_l1 = feed_north(k, j, b_l2_ready);

                    uint64_t ready_time = std::max({a_l1, b_l1, compute_available_});
                    compute_available_ = execute_matmul(i, j, k, ready_time);

                    // B buffer freed after B tile is pushed to L2 - emit credit upstream
                    // (The BlockMover returns credit when it consumes from L3)
                    emit_buffer_available(TraceLevel::L3, config_.l3_b_buffers[(next_b_buffer_ + 1) % 2]);

                    // Drain on last k iteration
                    if (k == k_tiles - 1) {
                        uint64_t d_l2_ready = drain_result(i, j, compute_available_);
                        uint64_t d_l3_ready = pull_from_l2(i, j, d_l2_ready);
                        store_d_tile(i, j, d_l3_ready);

                        TraceEvent complete_event;
                        complete_event.cycle = current_cycle_;
                        complete_event.level = TraceLevel::L3;
                        complete_event.operation = TraceOperation::TILE_COMPLETE;
                        complete_event.operand = TraceOperandType::TILE_D;
                        complete_event.tile_i = static_cast<uint16_t>(i);
                        complete_event.tile_j = static_cast<uint16_t>(j);
                        complete_event.loop_state = {loop_i_, loop_j_, loop_k_};
                        complete_event.l3_buffers_occupied = count_l3_occupied();
                        complete_event.name = "COMPLETE D[" + std::to_string(i) + "," + std::to_string(j) + "]";
                        trace_.push_back(complete_event);
                    }
                }

                // A buffer freed after all j iterations complete - emit credit upstream
                emit_buffer_available(TraceLevel::L3, config_.l3_a_buffers[(next_a_buffer_ + 1) % 2]);
            }
        }
    }

    /// Emit BUFFER_AVAILABLE credit event (dataflow: credit flows upstream)
    void emit_buffer_available(TraceLevel level, uint8_t buffer_id) {
        TraceEvent credit_event;
        credit_event.cycle = current_cycle_;
        credit_event.level = level;
        credit_event.operation = TraceOperation::BUFFER_AVAILABLE;
        credit_event.operand = TraceOperandType::BUFFER;
        credit_event.buffer_id = buffer_id;
        credit_event.l3_buffers_occupied = count_l3_occupied();
        credit_event.l2_buffers_occupied = count_l2_occupied();
        credit_event.loop_state = {loop_i_, loop_j_, loop_k_};
        credit_event.name = (level == TraceLevel::L3 ? "L3" : "L2") +
                           std::string("[") + std::to_string(buffer_id) + "] CREDIT";
        trace_.push_back(credit_event);

        if (level == TraceLevel::L3) {
            stats_.l3_buffer_available_events++;
        } else {
            stats_.l2_buffer_available_events++;
        }
    }
};

} // namespace sw::kpu::behavioral
