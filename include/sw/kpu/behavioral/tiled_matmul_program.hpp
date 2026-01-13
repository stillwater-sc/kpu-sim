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
// - Double-buffering for L3 and L2
// - Timing approximations for behavioral simulation
// - Trace generation for visualization
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
    uint8_t num_l3_tiles = 4;   // Total L3 tile buffers
    uint8_t num_l2_banks = 8;   // Total L2 bank buffers

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

/// Operation types in trace
enum class TraceOperation : uint8_t {
    LOAD,           // DMA load from host
    STORE,          // DMA store to host
    PUSH_TO_L2,     // BlockMover L3 -> L2
    PULL_FROM_L2,   // BlockMover L2 -> L3
    FEED_WEST,      // Streamer L2 -> L1 west
    FEED_NORTH,     // Streamer L2 -> L1 north
    MATMUL,         // Compute operation
    DRAIN,          // Streamer accumulator -> L2
    TILE_READY,     // Event: tile arrived
    BUFFER_FREE     // Event: buffer available
};

inline const char* to_string(TraceOperation op) {
    switch (op) {
        case TraceOperation::LOAD: return "LOAD";
        case TraceOperation::STORE: return "STORE";
        case TraceOperation::PUSH_TO_L2: return "PUSH_TO_L2";
        case TraceOperation::PULL_FROM_L2: return "PULL_FROM_L2";
        case TraceOperation::FEED_WEST: return "FEED_WEST";
        case TraceOperation::FEED_NORTH: return "FEED_NORTH";
        case TraceOperation::MATMUL: return "MATMUL";
        case TraceOperation::DRAIN: return "DRAIN";
        case TraceOperation::TILE_READY: return "TILE_READY";
        case TraceOperation::BUFFER_FREE: return "BUFFER_FREE";
        default: return "UNKNOWN";
    }
}

/// Trace event for visualization
struct TraceEvent {
    uint64_t cycle;             // When event occurs
    TraceLevel level;           // Which level
    TraceOperation operation;   // What operation
    TraceOperandType operand;   // Which operand
    uint16_t tile_i;            // Tile row
    uint16_t tile_j;            // Tile column
    uint16_t tile_k;            // Tile k (for A, B)
    uint8_t src_location;       // Source node/bank
    uint8_t dst_location;       // Destination node/bank
    uint64_t duration;          // Duration in cycles (0 for instant events)
    std::string name;           // Human-readable name

    std::string operand_id() const {
        std::string id = to_string(operand);
        id += "[" + std::to_string(tile_i) + "," + std::to_string(tile_j);
        if (operand == TraceOperandType::TILE_A || operand == TraceOperandType::TILE_B) {
            id += "," + std::to_string(tile_k);
        }
        id += "]";
        return id;
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

    // Utilization
    double compute_utilization() const {
        return total_cycles > 0 ?
            static_cast<double>(compute_cycles) / total_cycles : 0.0;
    }

    double effective_tflops(double clock_ghz) const {
        return total_cycles > 0 ?
            (static_cast<double>(flops) / total_cycles) * clock_ghz : 0.0;
    }
};

// ============================================================================
// Tiled Matrix Multiplication Program
// ============================================================================

/// Generates and executes tiled matmul with timing model
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

        // Initialize resource availability
        for (int i = 0; i < 4; ++i) {
            l3_available_[i] = 0;
        }
        for (int i = 0; i < 8; ++i) {
            l2_available_[i] = 0;
        }

        compute_available_ = 0;
        next_a_buffer_ = 0;
        next_b_buffer_ = 0;
    }

    /// Execute the full tiled matmul
    void execute() {
        reset();

        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        // C-stationary execution: iterate over output tiles
        for (uint32_t i = 0; i < m_tiles; ++i) {
            for (uint32_t j = 0; j < n_tiles; ++j) {
                execute_output_tile(i, j, k_tiles);
            }
        }

        stats_.total_cycles = current_cycle_;
        stats_.flops = config_.total_flops();
    }

    /// Execute with pipelined output tiles (more realistic)
    void execute_pipelined() {
        reset();

        uint32_t m_tiles = config_.m_tiles();
        uint32_t n_tiles = config_.n_tiles();
        uint32_t k_tiles = config_.k_tiles();

        // Pipeline: start loading next output tile while computing current
        // This is a simplified model - real hardware would have more overlap

        for (uint32_t i = 0; i < m_tiles; ++i) {
            for (uint32_t j = 0; j < n_tiles; ++j) {
                // For first k iteration, we must wait for data
                // For subsequent k, we can overlap
                execute_output_tile_pipelined(i, j, k_tiles);
            }
        }

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

    // Resource availability tracking (cycle when available)
    uint64_t l3_available_[4] = {0};
    uint64_t l2_available_[8] = {0};
    uint64_t compute_available_ = 0;

    // Double-buffer indices (0 or 1)
    int next_a_buffer_ = 0;
    int next_b_buffer_ = 0;

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

    uint64_t push_to_l2_a(uint32_t i, uint32_t k, uint64_t l3_ready) {
        uint8_t l2_bank = config_.l2_a_banks[k % 2];
        uint64_t start = std::max(l3_ready, l2_available_[l2_bank]);
        uint64_t latency = config_.bm_push_latency +
            (config_.a_tile_bytes() / config_.bm_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::BLOCK_MOVER, TraceOperation::PUSH_TO_L2,
            TraceOperandType::TILE_A,
            static_cast<uint16_t>(i), 0, static_cast<uint16_t>(k),
            config_.l3_a_buffers[(k + 1) % 2], l2_bank, latency,
            "PUSH A[" + std::to_string(i) + "," + std::to_string(k) + "] -> L2"
        });

        trace_.push_back({
            end, TraceLevel::L2, TraceOperation::TILE_READY,
            TraceOperandType::TILE_A,
            static_cast<uint16_t>(i), 0, static_cast<uint16_t>(k),
            l2_bank, l2_bank, 0, ""
        });

        l2_available_[l2_bank] = end;
        stats_.bm_pushes++;
        stats_.bm_bytes += config_.a_tile_bytes();

        return end;
    }

    uint64_t push_to_l2_b(uint32_t k, uint32_t j, uint64_t l3_ready) {
        uint8_t l2_bank = config_.l2_b_banks[k % 2];
        uint64_t start = std::max(l3_ready, l2_available_[l2_bank]);
        uint64_t latency = config_.bm_push_latency +
            (config_.b_tile_bytes() / config_.bm_bandwidth);
        uint64_t end = start + latency;

        trace_.push_back({
            start, TraceLevel::BLOCK_MOVER, TraceOperation::PUSH_TO_L2,
            TraceOperandType::TILE_B,
            0, static_cast<uint16_t>(j), static_cast<uint16_t>(k),
            config_.l3_b_buffers[(k + 1) % 2], l2_bank, latency,
            "PUSH B[" + std::to_string(k) + "," + std::to_string(j) + "] -> L2"
        });

        trace_.push_back({
            end, TraceLevel::L2, TraceOperation::TILE_READY,
            TraceOperandType::TILE_B,
            0, static_cast<uint16_t>(j), static_cast<uint16_t>(k),
            l2_bank, l2_bank, 0, ""
        });

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
};

} // namespace sw::kpu::behavioral
