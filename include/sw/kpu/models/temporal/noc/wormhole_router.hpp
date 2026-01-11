// ============================================================================
// include/sw/kpu/noc/wormhole_router.hpp
// Wormhole Router with Correct Bandwidth Modeling
//
// Key features:
// - Flit-level simulation with proper serialization
// - Credit-based flow control
// - Multi-hop XY (dimension-ordered) routing
// - Path reservation (HEAD reserves, TAIL releases)
// - Simultaneous independent link activity
// ============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <queue>
#include <vector>

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>

namespace sw::kpu::noc {

// ============================================================================
// Forward declarations
// ============================================================================
class WormholeRouter;
class WormholeNoC;

// ============================================================================
// Constants
// ============================================================================
constexpr size_t FLIT_SIZE_BYTES = 64;
constexpr size_t INPUT_BUFFER_FLITS = 8;   // 512 bytes
constexpr size_t OUTPUT_BUFFER_FLITS = 4;  // 256 bytes

// ============================================================================
// Port Direction
// ============================================================================
enum class PortDir : uint8_t {
    NORTH = 0,
    SOUTH = 1,
    EAST = 2,
    WEST = 3,
    LOCAL = 4,
    DMA = 5,
    NUM_PORTS = 6
};

inline const char* to_cstring(PortDir dir) {
    switch (dir) {
        case PortDir::NORTH: return "N";
        case PortDir::SOUTH: return "S";
        case PortDir::EAST: return "E";
        case PortDir::WEST: return "W";
        case PortDir::LOCAL: return "L";
        case PortDir::DMA: return "D";
        default: return "?";
    }
}

inline PortDir opposite_dir(PortDir dir) {
    switch (dir) {
        case PortDir::NORTH: return PortDir::SOUTH;
        case PortDir::SOUTH: return PortDir::NORTH;
        case PortDir::EAST: return PortDir::WEST;
        case PortDir::WEST: return PortDir::EAST;
        default: return dir;
    }
}

// ============================================================================
// Flit Types
// ============================================================================
enum class FlitType : uint8_t {
    HEAD = 0,       // First flit, contains routing info
    BODY = 1,       // Middle flits
    TAIL = 2,       // Last flit, releases path
    HEAD_TAIL = 3   // Single-flit packet (small tiles)
};

// ============================================================================
// Flit Structure
// ============================================================================
struct Flit {
    FlitType type = FlitType::BODY;
    uint16_t packet_seq = 0;      // Packet identifier

    // Routing info (valid for HEAD/HEAD_TAIL only)
    uint8_t src_router = 0;
    uint8_t dst_router = 0;
    uint16_t total_flits = 0;

    // Tile descriptor (valid for HEAD/HEAD_TAIL only)
    TileDescriptor tile;

    // Timing
    uint64_t inject_cycle = 0;    // When this flit was injected

    bool is_head() const {
        return type == FlitType::HEAD || type == FlitType::HEAD_TAIL;
    }

    bool is_tail() const {
        return type == FlitType::TAIL || type == FlitType::HEAD_TAIL;
    }
};

// ============================================================================
// Circular Flit Buffer
// ============================================================================
class FlitBuffer {
public:
    explicit FlitBuffer(size_t capacity = INPUT_BUFFER_FLITS);

    bool can_accept() const { return count_ < capacity_; }
    bool is_empty() const { return count_ == 0; }
    size_t count() const { return count_; }
    size_t capacity() const { return capacity_; }
    size_t free_space() const { return capacity_ - count_; }

    bool push(const Flit& flit);
    std::optional<Flit> pop();
    const Flit* peek() const;

    void clear();

private:
    size_t capacity_;
    size_t count_ = 0;
    size_t head_ = 0;
    size_t tail_ = 0;
    std::vector<Flit> buffer_;
};

// ============================================================================
// Path Reservation State
// ============================================================================
struct PathReservation {
    enum class State { IDLE, RESERVED };

    State state = State::IDLE;
    PortDir input_port = PortDir::LOCAL;
    uint16_t packet_seq = 0;
    uint16_t flits_remaining = 0;

    bool is_idle() const { return state == State::IDLE; }
    bool is_reserved() const { return state == State::RESERVED; }
    bool is_reserved_by(PortDir port) const {
        return state == State::RESERVED && input_port == port;
    }

    void reserve(PortDir in_port, uint16_t seq, uint16_t total_flits) {
        state = State::RESERVED;
        input_port = in_port;
        packet_seq = seq;
        flits_remaining = total_flits;
    }

    void flit_passed() {
        if (flits_remaining > 0) flits_remaining--;
    }

    void release() {
        state = State::IDLE;
        flits_remaining = 0;
    }
};

// ============================================================================
// Injection State (for LOCAL/DMA ports)
// ============================================================================
struct InjectionState {
    enum class State { IDLE, INJECTING };

    State state = State::IDLE;
    uint16_t packet_seq = 0;
    TileDescriptor tile;
    uint8_t dst_router = 0;
    uint64_t start_cycle = 0;
    uint16_t total_flits = 0;
    uint16_t flits_injected = 0;

    bool is_idle() const { return state == State::IDLE; }
    bool is_active() const { return state == State::INJECTING; }

    bool is_complete() const {
        return state == State::INJECTING && flits_injected >= total_flits;
    }

    void start(uint16_t seq, const TileDescriptor& t, uint8_t dst,
               uint64_t cycle, uint16_t num_flits) {
        state = State::INJECTING;
        packet_seq = seq;
        tile = t;
        dst_router = dst;
        start_cycle = cycle;
        total_flits = num_flits;
        flits_injected = 0;
    }

    Flit generate_next_flit(uint8_t src_router);

    void reset() {
        state = State::IDLE;
        flits_injected = 0;
    }
};

// ============================================================================
// Input Port
// ============================================================================
class InputPort {
public:
    InputPort() = default;
    explicit InputPort(PortDir dir);

    PortDir direction() const { return direction_; }

    // Flit reception from link
    bool can_receive() const { return buffer_.can_accept(); }
    /// Receive a flit from upstream. Returns true if flit was accepted.
    bool receive_flit(const Flit& flit, uint64_t cycle);

    // Flit access for switch
    bool has_flit() const { return !buffer_.is_empty(); }
    const Flit* peek_flit() const { return buffer_.peek(); }
    Flit consume_flit();

    // Credit management (returns credits to upstream)
    uint32_t credits_to_return() const { return credits_to_return_; }
    void clear_returned_credits() { credits_to_return_ = 0; }

    // Active packet tracking (for path reservation continuity)
    bool has_active_packet() const { return active_packet_.has_value(); }
    uint16_t active_packet_seq() const {
        return active_packet_.value_or(0);
    }
    void set_active_packet(uint16_t seq) { active_packet_ = seq; }
    void clear_active_packet() { active_packet_.reset(); }

    // Statistics
    struct Stats {
        uint64_t flits_received = 0;
        uint64_t packets_received = 0;
        uint64_t buffer_full_cycles = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    PortDir direction_ = PortDir::LOCAL;
    FlitBuffer buffer_{INPUT_BUFFER_FLITS};
    uint32_t credits_to_return_ = 0;
    std::optional<uint16_t> active_packet_;
    Stats stats_;
};

// ============================================================================
// Output Port
// ============================================================================
class OutputPort {
public:
    OutputPort() = default;
    explicit OutputPort(PortDir dir);

    PortDir direction() const { return direction_; }

    // Downstream connection
    void set_downstream_router(WormholeRouter* router) {
        downstream_router_ = router;
    }
    void set_initial_credits(uint32_t credits) { credits_ = credits; }

    // Flit acceptance from crossbar
    bool can_accept() const;
    void accept_flit(const Flit& flit, uint64_t cycle);

    // Transmission step (sends one flit if possible)
    void step(uint64_t cycle);

    // Credit from downstream
    void receive_credit() { credits_++; }
    uint32_t credits() const { return credits_; }

    // Path reservation
    PathReservation& reservation() { return reservation_; }
    const PathReservation& reservation() const { return reservation_; }

    // Pending flit for link (after step)
    bool has_flit_for_link() const { return pending_flit_.has_value(); }
    Flit take_pending_flit();

    // Statistics
    struct Stats {
        uint64_t flits_sent = 0;
        uint64_t credit_stall_cycles = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    PortDir direction_ = PortDir::LOCAL;
    FlitBuffer buffer_{OUTPUT_BUFFER_FLITS};
    WormholeRouter* downstream_router_ = nullptr;
    uint32_t credits_ = INPUT_BUFFER_FLITS;  // Initial credits
    PathReservation reservation_;
    std::optional<Flit> pending_flit_;
    Stats stats_;
};

// ============================================================================
// Wormhole Router
// ============================================================================
class WormholeRouter {
public:
    WormholeRouter() = default;
    WormholeRouter(uint8_t id, uint8_t row, uint8_t col,
                   uint8_t mesh_rows, uint8_t mesh_cols, bool has_dma = false);

    // Identification
    uint8_t id() const { return id_; }
    uint8_t row() const { return row_; }
    uint8_t col() const { return col_; }
    uint8_t mesh_rows() const { return mesh_rows_; }
    uint8_t mesh_cols() const { return mesh_cols_; }

    // Port access
    InputPort& input(PortDir dir) { return inputs_[static_cast<size_t>(dir)]; }
    OutputPort& output(PortDir dir) { return outputs_[static_cast<size_t>(dir)]; }

    const InputPort& input(PortDir dir) const {
        return inputs_[static_cast<size_t>(dir)];
    }
    const OutputPort& output(PortDir dir) const {
        return outputs_[static_cast<size_t>(dir)];
    }

    // L3 injection interface
    bool can_inject() const;
    bool start_injection(uint16_t packet_seq, const TileDescriptor& tile,
                        uint8_t dst_router, uint64_t cycle);

    // L3 ejection interface
    bool has_ejection() const;
    Flit eject_flit();

    // DMA interface (edge routers only)
    bool has_dma() const { return has_dma_; }
    bool can_inject_dma() const;
    bool start_dma_injection(uint16_t packet_seq, const TileDescriptor& tile,
                            uint8_t dst_router, uint64_t cycle);

    // Connection setup
    void connect_output(PortDir dir, WormholeRouter* neighbor);

    // Simulation step
    void step(uint64_t cycle);

    // Check if router is idle (no active transfers)
    bool is_idle() const;

    // Statistics
    struct Stats {
        uint64_t flits_switched = 0;
        uint64_t packets_injected = 0;
        uint64_t packets_ejected = 0;
        uint64_t arbitration_conflicts = 0;
        uint64_t paths_reserved = 0;
        uint64_t paths_released = 0;
    };
    const Stats& stats() const { return stats_; }
    void reset_stats();

private:
    uint8_t id_ = 0;
    uint8_t row_ = 0;
    uint8_t col_ = 0;
    uint8_t mesh_rows_ = 4;
    uint8_t mesh_cols_ = 4;
    bool has_dma_ = false;

    std::array<InputPort, 6> inputs_;    // N, S, E, W, L, DMA
    std::array<OutputPort, 6> outputs_;  // N, S, E, W, L, DMA

    // Injection state for LOCAL port
    InjectionState local_injection_;

    // Injection state for DMA port
    InjectionState dma_injection_;

    // Ejection buffer (flits arriving at LOCAL output)
    std::queue<Flit> ejection_queue_;

    Stats stats_;

    // XY (dimension-ordered) routing for multi-hop support
    // Routes first in X (columns), then in Y (rows)
    PortDir route(uint8_t dst_router) const;

    // Process injection (LOCAL/DMA → input buffer)
    void process_injections(uint64_t cycle);

    // Arbitrate and switch flits
    void arbitrate_and_switch(uint64_t cycle);

    // Process link transmissions
    void process_outputs(uint64_t cycle);

    // Deliver flits from incoming links
    void receive_from_links(uint64_t cycle);
};

// ============================================================================
// Trace Event Types
// ============================================================================
enum class WormholeEventType : uint8_t {
    INJECT_START,       // Started injecting a tile
    INJECT_COMPLETE,    // Finished injecting all flits
    PATH_RESERVE,       // HEAD flit reserved output path
    PATH_RELEASE,       // TAIL flit released path
    FLIT_HOP,           // Flit traversed a link
    DELIVER_START,      // First flit arrived at destination
    DELIVER_COMPLETE,   // Last flit delivered
    ARBITRATION_BLOCK,  // HEAD lost arbitration
    BACKPRESSURE,       // Credit exhaustion
};

struct WormholeTraceEvent {
    uint64_t cycle;
    WormholeEventType type;
    uint8_t router_id;
    PortDir port;
    uint16_t packet_seq;
    FlitType flit_type;
    uint16_t flit_index;
    uint16_t total_flits;
    TileDescriptor tile;
    uint8_t src_router;
    uint8_t dst_router;
};

// ============================================================================
// Wormhole Tracer
// ============================================================================
class WormholeTracer {
public:
    void record(const WormholeTraceEvent& event);
    void clear() { events_.clear(); }

    const std::vector<WormholeTraceEvent>& events() const { return events_; }
    size_t num_events() const { return events_.size(); }

    bool export_csv(const std::string& filename) const;
    bool export_chrome_trace(const std::string& filename,
                            uint8_t mesh_rows, uint8_t mesh_cols) const;

private:
    std::vector<WormholeTraceEvent> events_;
};

// ============================================================================
// Wormhole NoC Top Level
// ============================================================================
class WormholeNoC {
public:
    struct Config {
        uint8_t rows = 4;
        uint8_t cols = 4;
        bool dma_on_north_edge = true;
        bool dma_on_south_edge = false;
        bool dma_on_east_edge = false;
        bool dma_on_west_edge = false;
    };

    explicit WormholeNoC(const Config& config);

    // Configuration
    const Config& config() const { return config_; }
    uint8_t num_routers() const { return config_.rows * config_.cols; }
    uint8_t router_id(uint8_t row, uint8_t col) const {
        return row * config_.cols + col;
    }

    // Router access
    WormholeRouter& router(uint8_t id) { return routers_[id]; }
    const WormholeRouter& router(uint8_t id) const { return routers_[id]; }

    // Tile injection (from L3 cache)
    enum class InjectResult { SUCCESS, BUSY, INVALID };
    InjectResult inject_tile(uint8_t src_router, uint8_t dst_router,
                            const TileDescriptor& tile, uint64_t cycle);

    // Check if injection is possible
    bool can_inject(uint8_t router_id) const;

    // Delivery callback
    using DeliveryCallback = std::function<void(const TileDescriptor& tile,
                                                 uint8_t src_router,
                                                 uint64_t cycle)>;
    void set_delivery_callback(uint8_t router_id, DeliveryCallback cb);

    // DMA interface
    InjectResult dma_inject(uint8_t router_id, uint8_t dst_router,
                           const TileDescriptor& tile, uint64_t cycle);

    // Simulation
    void step(uint64_t cycle);
    bool is_idle() const;

    // Tracer
    void set_tracer(WormholeTracer* tracer) { tracer_ = tracer; }
    WormholeTracer* tracer() { return tracer_; }

    // Statistics
    struct Stats {
        uint64_t tiles_injected = 0;
        uint64_t tiles_delivered = 0;
        uint64_t total_flits = 0;
        uint64_t total_bytes = 0;
        uint64_t total_latency_cycles = 0;  // Sum of all tile latencies
        uint64_t max_latency_cycles = 0;
    };
    const Stats& stats() const { return stats_; }
    void reset_stats();

private:
    Config config_;
    std::vector<WormholeRouter> routers_;
    std::vector<DeliveryCallback> delivery_callbacks_;
    WormholeTracer* tracer_ = nullptr;

    // Active transfers tracking
    struct ActiveTransfer {
        uint16_t packet_seq;
        uint8_t src_router;
        uint8_t dst_router;
        TileDescriptor tile;
        uint64_t inject_start_cycle;
        uint16_t total_flits;
        uint16_t flits_delivered;
    };
    std::map<uint16_t, ActiveTransfer> active_transfers_;
    uint16_t next_packet_seq_ = 0;

    Stats stats_;

    // Internal helpers
    void setup_mesh();
    void check_deliveries(uint64_t cycle);
    void return_credits_to_upstream(uint64_t cycle);
    void record_event(WormholeEventType type, uint8_t router_id, PortDir port,
                     uint16_t packet_seq, FlitType flit_type,
                     uint16_t flit_index, uint16_t total_flits,
                     const TileDescriptor& tile,
                     uint8_t src_router, uint8_t dst_router, uint64_t cycle);
};

}  // namespace sw::kpu::noc
