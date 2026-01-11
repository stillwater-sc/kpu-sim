// ============================================================================
// include/sw/kpu/models/temporal/datamovement/stateful_block_mover.hpp
// Stateful BlockMover - Executes tile movement programs
// ============================================================================

#pragma once

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>
#include <sw/kpu/models/temporal/datamovement/l3_interconnect.hpp>
#include <sw/kpu/dataflow/tile_flow_tracer.hpp>
#include <sw/kpu/models/temporal/noc/wormhole_router.hpp>

namespace sw::kpu {

// Forward declaration
namespace dataflow {
class TileFlowTracer;
}

// Forward declarations
class L3Tile;
class L2Bank;

// ============================================================================
// BlockMover Configuration
// ============================================================================

struct StatefulBlockMoverConfig {
    uint8_t id = 0;                             // BlockMover/L3 tile ID

    // Mesh position (derived from id if not set)
    uint8_t row = 0;
    uint8_t col = 0;
    uint8_t mesh_cols = 4;                      // For position calculation

    // Transfer characteristics
    uint32_t l3_to_l2_bandwidth_bytes_per_cycle = 64;   // 32 GB/s @ 500MHz
    uint32_t l3_to_l3_bandwidth_bytes_per_cycle = 64;   // Per-link bandwidth
    uint32_t l3_to_l2_latency_cycles = 4;               // Pipeline latency
    uint32_t l3_to_l3_latency_cycles = 2;               // Per-hop latency

    // Command queue
    size_t max_pending_transfers = 4;           // Max concurrent transfers
    size_t receive_buffer_depth = 4;            // Incoming packet buffer

    // Trigger channels
    static constexpr size_t NUM_TRIGGER_CHANNELS = 16;

    // Neighbor IDs (-1 if no neighbor in that direction)
    int neighbor_north = -1;
    int neighbor_south = -1;
    int neighbor_east = -1;
    int neighbor_west = -1;

    /// Configure neighbors based on mesh position
    void configure_from_mesh(uint8_t tile_id, uint8_t rows, uint8_t cols) {
        id = tile_id;
        mesh_cols = cols;
        row = tile_id / cols;
        col = tile_id % cols;

        neighbor_north = (row > 0) ? static_cast<int>(tile_id - cols) : -1;
        neighbor_south = (row < rows - 1) ? static_cast<int>(tile_id + cols) : -1;
        neighbor_west = (col > 0) ? static_cast<int>(tile_id - 1) : -1;
        neighbor_east = (col < cols - 1) ? static_cast<int>(tile_id + 1) : -1;
    }

    int get_neighbor(Direction dir) const {
        switch (dir) {
            case Direction::NORTH: return neighbor_north;
            case Direction::SOUTH: return neighbor_south;
            case Direction::EAST: return neighbor_east;
            case Direction::WEST: return neighbor_west;
            default: return -1;
        }
    }
};

// ============================================================================
// BlockMover Execution State
// ============================================================================

/// State of the BlockMover program execution
enum class BlockMoverState : uint8_t {
    IDLE,               // No program loaded or execution complete
    WAITING_START,      // Waiting for start_delay to elapse before execution
    RUNNING,            // Actively executing commands
    WAITING_TRIGGER,    // Blocked waiting for trigger(s)
    WAITING_TRANSFER,   // Blocked waiting for transfer to complete
    WAITING_RECEIVE,    // Blocked waiting for incoming packet
    WAITING_DELIVERY,   // Blocked waiting for NoC delivery confirmation
    HALTED,             // Explicitly halted (for debugging)
    ERROR,              // Execution error
};

inline const char* to_string(BlockMoverState state) {
    switch (state) {
        case BlockMoverState::IDLE: return "IDLE";
        case BlockMoverState::WAITING_START: return "WAITING_START";
        case BlockMoverState::RUNNING: return "RUNNING";
        case BlockMoverState::WAITING_TRIGGER: return "WAITING_TRIGGER";
        case BlockMoverState::WAITING_TRANSFER: return "WAITING_TRANSFER";
        case BlockMoverState::WAITING_RECEIVE: return "WAITING_RECEIVE";
        case BlockMoverState::WAITING_DELIVERY: return "WAITING_DELIVERY";
        case BlockMoverState::HALTED: return "HALTED";
        case BlockMoverState::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Pending Transfer
// ============================================================================

/// Tracks an in-progress data transfer
struct PendingTransfer {
    enum class Type {
        L3_TO_L2,       // Pushing to L2
        L2_TO_L3,       // Pulling from L2 (drain)
        L3_TO_L3,       // Sending to another L3
    };

    Type type = Type::L3_TO_L2;
    TileDescriptor tile;
    uint8_t dest_id = 0;            // Destination L3/L2 ID
    uint32_t bytes_total = 0;
    uint32_t bytes_transferred = 0;
    uint64_t start_cycle = 0;
    uint64_t complete_cycle = 0;

    bool is_complete(uint64_t cycle) const {
        return cycle >= complete_cycle;
    }

    float progress() const {
        return bytes_total > 0 ? static_cast<float>(bytes_transferred) / bytes_total : 1.0f;
    }
};

// ============================================================================
// StatefulBlockMover
// ============================================================================

/// Stateful BlockMover that executes tile movement programs
class StatefulBlockMover {
public:
    // Callbacks for external interactions
    using TransferCallback = std::function<void(const TileDescriptor&, uint8_t dest_l3)>;
    using TriggerCallback = std::function<void(uint8_t channel, uint16_t dest_mask)>;
    using TraceCallback = std::function<void(uint32_t trace_id, uint64_t cycle)>;
    using DmaArrivalCallback = std::function<void(const TileDescriptor& tile, uint64_t cycle)>;

    explicit StatefulBlockMover(const StatefulBlockMoverConfig& config = StatefulBlockMoverConfig{});

    // ========== Program Loading ==========

    /// Load a program for execution
    void load_program(const BlockMoverProgram& program);

    /// Load program from command vector
    void load_program(const std::vector<BlockMoverCommand>& commands);

    /// Clear current program
    void clear_program();

    // ========== Execution Control ==========

    /// Execute one cycle of the program
    /// Returns true if still executing, false if idle/halted
    bool step(uint64_t cycle);

    /// Reset execution state (restart program from beginning)
    void reset();

    /// Halt execution
    void halt();

    /// Resume from halt
    void resume();

    // ========== External Events ==========

    /// Receive a trigger from another BlockMover or component
    void receive_trigger(uint8_t channel);

    /// Receive a data packet from another L3
    void receive_packet(const L3TransferPacket& packet);

    /// Notify that a transfer to L2 has completed
    void notify_l2_transfer_complete(const TileDescriptor& tile);

    /// Notify that a NoC packet sent by this BlockMover has been delivered
    void notify_noc_delivery();

    // ========== Callbacks ==========

    /// Set callback for L3→L3 transfers (called when SEND_* completes)
    void set_transfer_callback(TransferCallback callback) {
        transfer_callback_ = std::move(callback);
    }

    /// Set callback for trigger emissions
    void set_trigger_callback(TriggerCallback callback) {
        trigger_callback_ = std::move(callback);
    }

    /// Set callback for trace markers
    void set_trace_callback(TraceCallback callback) {
        trace_callback_ = std::move(callback);
    }

    /// Set callback for DMA arrivals (called when TRACE_MARKER with tile is executed)
    void set_dma_arrival_callback(DmaArrivalCallback callback) {
        dma_arrival_callback_ = std::move(callback);
    }

    /// Set tracer for recording tile flow events
    void set_tracer(dataflow::TileFlowTracer* tracer) {
        tracer_ = tracer;
    }

    // ========== State Query ==========

    uint8_t id() const { return config_.id; }
    BlockMoverState state() const { return state_; }
    bool is_idle() const { return state_ == BlockMoverState::IDLE; }
    bool is_running() const { return state_ == BlockMoverState::RUNNING; }
    bool is_waiting() const {
        return state_ == BlockMoverState::WAITING_START ||
               state_ == BlockMoverState::WAITING_TRIGGER ||
               state_ == BlockMoverState::WAITING_TRANSFER ||
               state_ == BlockMoverState::WAITING_RECEIVE ||
               state_ == BlockMoverState::WAITING_DELIVERY;
    }

    size_t program_counter() const { return pc_; }
    size_t program_size() const { return program_.size(); }
    size_t commands_remaining() const {
        return pc_ < program_.size() ? program_.size() - pc_ : 0;
    }
    uint64_t current_cycle() const { return current_cycle_; }

    const BlockMoverCommand* current_command() const {
        return pc_ < program_.size() ? &program_[pc_] : nullptr;
    }

    // Trigger state
    bool is_trigger_set(uint8_t channel) const {
        return channel < 16 && triggers_received_.test(channel);
    }
    uint16_t get_trigger_mask() const {
        return static_cast<uint16_t>(triggers_received_.to_ulong());
    }

    // Transfer state
    size_t pending_transfers() const { return pending_transfers_.size(); }
    size_t receive_queue_size() const { return receive_queue_.size(); }

    // ========== Statistics ==========

    struct Stats {
        uint64_t commands_executed = 0;
        uint64_t cycles_active = 0;
        uint64_t cycles_waiting_trigger = 0;
        uint64_t cycles_waiting_transfer = 0;
        uint64_t cycles_waiting_receive = 0;

        uint64_t l3_to_l2_transfers = 0;
        uint64_t l3_to_l2_bytes = 0;
        uint64_t l3_to_l3_transfers = 0;
        uint64_t l3_to_l3_bytes = 0;

        uint64_t triggers_sent = 0;
        uint64_t triggers_received = 0;

        uint64_t loops_executed = 0;

        double utilization() const {
            uint64_t total = cycles_active + cycles_waiting_trigger +
                            cycles_waiting_transfer + cycles_waiting_receive;
            return total > 0 ? static_cast<double>(cycles_active) / total : 0.0;
        }
    };

    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

    // ========== Debug ==========

    std::string to_string() const;
    std::string dump_state() const;

    /// Get detailed info about receive state (expected vs queued sources)
    std::string debug_receive_state() const;

private:
    StatefulBlockMoverConfig config_;
    BlockMoverState state_ = BlockMoverState::IDLE;

    // Program
    std::vector<BlockMoverCommand> program_;
    size_t pc_ = 0;                             // Program counter

    // Loop state
    struct LoopContext {
        size_t start_pc;        // PC of LOOP_START
        uint16_t remaining;     // Iterations remaining
    };
    std::stack<LoopContext> loop_stack_;

    // Trigger state
    std::bitset<16> triggers_received_;
    uint16_t trigger_wait_mask_ = 0;            // What we're waiting for

    // Transfer state
    std::queue<PendingTransfer> pending_transfers_;
    std::deque<L3TransferPacket> receive_queue_;  // deque for selective receive
    std::optional<TileDescriptor> expected_receive_;

    // Current L2 target (set by SET_L2_BANK)
    uint8_t current_l2_bank_ = 0;

    // Callbacks
    TransferCallback transfer_callback_;
    TriggerCallback trigger_callback_;
    TraceCallback trace_callback_;
    DmaArrivalCallback dma_arrival_callback_;

    // Tracer
    dataflow::TileFlowTracer* tracer_ = nullptr;

    // Statistics
    Stats stats_;

    // Current cycle (set by step() for use by execute_current)
    uint64_t current_cycle_ = 0;

    // Start delay for staggered wavefront execution
    uint64_t start_delay_cycles_ = 0;

    // NoC delivery tracking (for WAIT_DELIVERY command)
    size_t pending_noc_deliveries_ = 0;

    // ========== Internal Execution ==========

    /// Execute the current command
    /// Returns true if execution should continue, false if blocked
    bool execute_current();

    /// Advance PC to next command
    void advance_pc();

    /// Check if waiting conditions are satisfied
    bool check_wait_conditions(uint64_t cycle);

    /// Process completed transfers
    void process_completed_transfers(uint64_t cycle);

    // Command executors
    bool exec_push_to_l2(const BlockMoverCommand& cmd, uint64_t cycle);
    bool exec_pull_from_l2(const BlockMoverCommand& cmd, uint64_t cycle);
    bool exec_send(const BlockMoverCommand& cmd, Direction dir, uint64_t cycle);
    bool exec_send_to(const BlockMoverCommand& cmd, uint64_t cycle);
    bool exec_receive(const BlockMoverCommand& cmd, uint64_t cycle);
    bool exec_wait_trigger(const BlockMoverCommand& cmd);
    bool exec_emit_trigger(const BlockMoverCommand& cmd);
    bool exec_barrier(const BlockMoverCommand& cmd, uint64_t cycle);
    bool exec_wait_delivery(const BlockMoverCommand& cmd);
    bool exec_loop_start(const BlockMoverCommand& cmd);
    bool exec_loop_end(const BlockMoverCommand& cmd);
    bool exec_jump(const BlockMoverCommand& cmd);
    bool exec_set_l2_bank(const BlockMoverCommand& cmd);

    /// Calculate transfer completion cycle
    uint64_t calc_transfer_complete_cycle(PendingTransfer::Type type, size_t bytes, uint64_t start) const;
};

// ============================================================================
// Systolic Order Checker
// ============================================================================

/// Validates systolic recurrence ordering for tile arrivals
/// A tiles: A[m,k] at L3[row,col] must arrive AFTER A[m,k] at L3[row,col-1]
/// B tiles: B[k,n] at L3[row,col] must arrive AFTER B[k,n] at L3[row-1,col]
class SystolicOrderChecker {
public:
    struct Config {
        uint8_t rows = 4;
        uint8_t cols = 4;
        bool enabled = true;
        bool stop_on_violation = true;  // Stop simulation on first violation
    };

    struct Violation {
        uint64_t cycle;
        TensorId tensor;
        uint16_t m_tile, n_tile, k_tile;
        uint8_t arrived_at_l3;      // Where tile arrived
        uint8_t expected_from_l3;   // Where it should have arrived first
        uint64_t expected_arrival_cycle;  // When it arrived at expected_from
        std::string message;
    };

    SystolicOrderChecker() : config_{} {
        arrival_times_.resize(config_.rows * config_.cols);
    }

    explicit SystolicOrderChecker(const Config& config)
        : config_(config) {
        // Initialize arrival tracking: [l3_id][tile_key] -> arrival_cycle
        arrival_times_.resize(config.rows * config.cols);
    }

    /// Record a tile arrival at an L3
    /// Returns true if order is valid, false if violation detected
    bool record_arrival(uint8_t l3_id, const TileDescriptor& tile, uint64_t cycle) {
        if (!config_.enabled) return true;

        uint8_t row = l3_id / config_.cols;
        uint8_t col = l3_id % config_.cols;

        // Create tile key for lookup
        std::string tile_key = make_tile_key(tile);

        // Check recurrence based on tensor type
        bool valid = true;
        if (tile.tensor == TensorId::A && col > 0) {
            // A tiles flow East: must arrive at [row, col-1] first
            uint8_t prev_l3 = row * config_.cols + (col - 1);
            valid = check_predecessor(tile_key, prev_l3, l3_id, tile, cycle);
        } else if (tile.tensor == TensorId::B && row > 0) {
            // B tiles flow South: must arrive at [row-1, col] first
            uint8_t prev_l3 = (row - 1) * config_.cols + col;
            valid = check_predecessor(tile_key, prev_l3, l3_id, tile, cycle);
        }
        // DMA arrivals (col=0 for A, row=0 for B) don't need predecessor check

        // Record this arrival
        arrival_times_[l3_id][tile_key] = cycle;

        return valid;
    }

    /// Check if there are any violations
    bool has_violations() const { return !violations_.empty(); }

    /// Get all violations
    const std::vector<Violation>& violations() const { return violations_; }

    /// Get number of violations
    size_t num_violations() const { return violations_.size(); }

    /// Clear recorded arrivals and violations
    void reset() {
        for (auto& map : arrival_times_) {
            map.clear();
        }
        violations_.clear();
    }

    /// Print violation summary
    void print_violations(std::ostream& os = std::cout) const {
        if (violations_.empty()) {
            os << "Systolic order check: PASSED (no violations)\n";
            return;
        }

        os << "\n╔══════════════════════════════════════════════════════════════════╗\n";
        os << "║           SYSTOLIC ORDER VIOLATIONS DETECTED                     ║\n";
        os << "╚══════════════════════════════════════════════════════════════════╝\n\n";

        for (const auto& v : violations_) {
            os << "  [Cycle " << v.cycle << "] " << v.message << "\n";
        }
        os << "\nTotal violations: " << violations_.size() << "\n";
    }

    /// Enable/disable checking
    void set_enabled(bool enabled) { config_.enabled = enabled; }
    bool is_enabled() const { return config_.enabled; }

private:
    Config config_;
    std::vector<std::unordered_map<std::string, uint64_t>> arrival_times_;
    std::vector<Violation> violations_;

    std::string make_tile_key(const TileDescriptor& tile) const {
        // Create unique key based on tensor type and indices
        char buf[64];
        switch (tile.tensor) {
            case TensorId::A:
                snprintf(buf, sizeof(buf), "A[%u,%u]", tile.m_tile, tile.k_tile);
                break;
            case TensorId::B:
                snprintf(buf, sizeof(buf), "B[%u,%u]", tile.k_tile, tile.n_tile);
                break;
            case TensorId::C:
                snprintf(buf, sizeof(buf), "C[%u,%u]", tile.m_tile, tile.n_tile);
                break;
            default:
                snprintf(buf, sizeof(buf), "?[%u,%u,%u]", tile.m_tile, tile.n_tile, tile.k_tile);
        }
        return std::string(buf);
    }

    bool check_predecessor(const std::string& tile_key, uint8_t prev_l3, uint8_t curr_l3,
                          const TileDescriptor& tile, uint64_t cycle) {
        auto it = arrival_times_[prev_l3].find(tile_key);
        if (it == arrival_times_[prev_l3].end()) {
            // Tile hasn't arrived at predecessor yet - violation!
            Violation v;
            v.cycle = cycle;
            v.tensor = tile.tensor;
            v.m_tile = tile.m_tile;
            v.n_tile = tile.n_tile;
            v.k_tile = tile.k_tile;
            v.arrived_at_l3 = curr_l3;
            v.expected_from_l3 = prev_l3;
            v.expected_arrival_cycle = 0;  // Never arrived

            uint8_t prev_row = prev_l3 / config_.cols;
            uint8_t prev_col = prev_l3 % config_.cols;
            uint8_t curr_row = curr_l3 / config_.cols;
            uint8_t curr_col = curr_l3 % config_.cols;

            char buf[256];
            snprintf(buf, sizeof(buf),
                     "%s arrived at L3[%u,%u] before arriving at L3[%u,%u] (systolic violation)",
                     tile_key.c_str(), curr_row, curr_col, prev_row, prev_col);
            v.message = buf;

            violations_.push_back(v);
            return false;
        }
        // Predecessor arrival exists - order is correct
        return true;
    }
};

// ============================================================================
// BlockMover Array (all 16 BlockMovers)
// ============================================================================

/// Manages all BlockMovers in the system
class BlockMoverArray {
public:
    struct Config {
        uint8_t rows = 4;
        uint8_t cols = 4;
        StatefulBlockMoverConfig mover_config;  // Template for individual configs
    };

    /// Default constructor uses 4x4 mesh
    BlockMoverArray();

    /// Constructor with custom configuration
    explicit BlockMoverArray(const Config& config);

    // Access individual BlockMovers
    StatefulBlockMover& operator[](size_t id) { return *movers_[id]; }
    const StatefulBlockMover& operator[](size_t id) const { return *movers_[id]; }

    StatefulBlockMover& at(uint8_t row, uint8_t col) {
        return *movers_[row * config_.cols + col];
    }

    // Load programs
    void load_programs(const std::vector<BlockMoverProgram>& programs);

    // Execution
    void step(uint64_t cycle);
    void reset();

    // State
    bool all_idle() const;
    bool any_running() const;
    size_t num_running() const;
    size_t num_waiting() const;

    // Inter-mover communication
    noc::WormholeNoC& noc() { return noc_; }
    const noc::WormholeNoC& noc() const { return noc_; }
    TriggerNetwork& triggers() { return triggers_; }

    // NoC tracing
    void set_noc_tracer(noc::WormholeTracer* tracer) {
        noc_tracer_ = tracer;
        noc_.set_tracer(tracer);
    }
    noc::WormholeTracer* noc_tracer() const { return noc_tracer_; }

    // Statistics
    struct AggregateStats {
        uint64_t total_commands = 0;
        uint64_t total_l3_l3_transfers = 0;
        uint64_t total_l3_l2_transfers = 0;
        uint64_t total_bytes_moved = 0;
        double avg_utilization = 0.0;
    };
    AggregateStats get_aggregate_stats() const;

    // Tracing support
    void set_tracer(dataflow::TileFlowTracer* tracer) { tracer_ = tracer; }
    dataflow::TileFlowTracer* tracer() const { return tracer_; }

    // Systolic order checking
    SystolicOrderChecker& order_checker() { return order_checker_; }
    const SystolicOrderChecker& order_checker() const { return order_checker_; }
    void enable_order_checking(bool enabled) { order_checker_.set_enabled(enabled); }
    bool has_order_violations() const { return order_checker_.has_violations(); }

private:
    Config config_;
    std::vector<std::unique_ptr<StatefulBlockMover>> movers_;
    noc::WormholeNoC noc_;
    noc::WormholeTracer* noc_tracer_ = nullptr;
    TriggerNetwork triggers_;
    dataflow::TileFlowTracer* tracer_ = nullptr;
    SystolicOrderChecker order_checker_;

    // Pending NoC injections (for retry on BUFFER_FULL)
    struct PendingNoCInjection {
        uint8_t src_l3_id;
        uint8_t dst_l3_id;
        TileDescriptor tile;
        uint64_t first_attempt_cycle;
    };
    std::queue<PendingNoCInjection> pending_noc_injections_;

    void setup_callbacks();
};

} // namespace sw::kpu
