// ============================================================================
// include/sw/kpu/noc/noc.hpp
// Network-on-Chip for KPU CheckerBoard Architecture
//
// The NoC connects:
// - DMA engines (external memory interface)
// - L3 tiles (16 tiles in 4x4 mesh)
// - Enables BlockMovers to move tiles with proper contention modeling
//
// Data flows supported:
// - DMA → L3: Load tiles from external memory to specific L3 tiles
// - L3 → L3: Systolic data flow (A tiles East, B tiles South)
// - L3 → L2: Push tiles to L2 banks for Streamer consumption
// - L2 → L3: Pull computed results back to L3
// - L3 → DMA: Store tiles to external memory
// ============================================================================

#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>

namespace sw::kpu::noc {

// Forward declarations
class NoC;
class NoCRouter;
class NoCLink;
class NoCTracer;

// ============================================================================
// NoC Configuration
// ============================================================================

/// Routing algorithm
enum class RoutingAlgorithm : uint8_t {
    XY,             // X first, then Y (deterministic, deadlock-free)
    YX,             // Y first, then X
    WEST_FIRST,     // Adaptive, west first
    NEGATIVE_FIRST, // Adaptive, negative directions first
};

/// Flow control mechanism
enum class FlowControl : uint8_t {
    STORE_AND_FORWARD,  // Complete packet buffered at each hop
    WORMHOLE,           // Flits flow through, head reserves path
    VIRTUAL_CUT_THROUGH,// Like wormhole but waits for full packet at dest
};

/// NoC topology
enum class NoCTopology : uint8_t {
    MESH_2D,    // Standard 2D mesh
    TORUS_2D,   // 2D mesh with wraparound
    RING,       // Ring topology
};

/// Port direction
enum class PortDirection : uint8_t {
    NORTH = 0,
    SOUTH = 1,
    EAST = 2,
    WEST = 3,
    LOCAL = 4,      // Injection/ejection from attached tile
    DMA = 5,        // DMA engine port (edge tiles only)
    NUM_PORTS = 6
};

inline const char* to_string(PortDirection dir) {
    switch (dir) {
        case PortDirection::NORTH: return "N";
        case PortDirection::SOUTH: return "S";
        case PortDirection::EAST: return "E";
        case PortDirection::WEST: return "W";
        case PortDirection::LOCAL: return "L";
        case PortDirection::DMA: return "D";
        default: return "?";
    }
}

inline PortDirection opposite(PortDirection dir) {
    switch (dir) {
        case PortDirection::NORTH: return PortDirection::SOUTH;
        case PortDirection::SOUTH: return PortDirection::NORTH;
        case PortDirection::EAST: return PortDirection::WEST;
        case PortDirection::WEST: return PortDirection::EAST;
        default: return dir;
    }
}

/// NoC configuration parameters
struct NoCConfig {
    // Topology
    NoCTopology topology = NoCTopology::MESH_2D;
    uint8_t rows = 4;
    uint8_t cols = 4;

    // Routing and flow control
    RoutingAlgorithm routing = RoutingAlgorithm::XY;
    FlowControl flow_control = FlowControl::STORE_AND_FORWARD;

    // Timing (in cycles)
    uint32_t router_latency = 1;        // Cycles to route through a router
    uint32_t link_latency = 1;          // Cycles to traverse a link

    // Bandwidth (bytes per cycle)
    uint32_t link_bandwidth = 64;       // 64 B/cycle = 32 GB/s @ 500 MHz
    uint32_t injection_bandwidth = 64;  // Local port bandwidth
    uint32_t dma_bandwidth = 32;        // DMA port bandwidth

    // Buffering
    uint32_t input_buffer_flits = 8;    // Input buffer depth per port
    uint32_t output_buffer_flits = 4;   // Output buffer depth per port

    // Packet size
    uint32_t flit_size_bytes = 64;      // Flit size
    uint32_t max_packet_flits = 4096;   // Max packet size in flits (256KB / 64B)

    // Virtual channels (for deadlock avoidance with adaptive routing)
    uint8_t num_virtual_channels = 1;

    // DMA configuration
    bool dma_at_edges = true;           // DMA ports on edge routers only
    uint8_t dma_ports_per_edge = 1;     // DMA ports per edge router

    // Derived values
    size_t num_routers() const { return rows * cols; }
    size_t num_links() const {
        // Horizontal + vertical links (bidirectional)
        return 2 * (rows * (cols - 1) + (rows - 1) * cols);
    }

    uint8_t get_router_id(uint8_t row, uint8_t col) const {
        return row * cols + col;
    }

    std::pair<uint8_t, uint8_t> get_router_pos(uint8_t id) const {
        return {id / cols, id % cols};
    }

    bool has_neighbor(uint8_t id, PortDirection dir) const {
        auto [row, col] = get_router_pos(id);
        switch (dir) {
            case PortDirection::NORTH: return row > 0 || topology == NoCTopology::TORUS_2D;
            case PortDirection::SOUTH: return row < rows - 1 || topology == NoCTopology::TORUS_2D;
            case PortDirection::EAST:  return col < cols - 1 || topology == NoCTopology::TORUS_2D;
            case PortDirection::WEST:  return col > 0 || topology == NoCTopology::TORUS_2D;
            default: return false;
        }
    }

    int get_neighbor_id(uint8_t id, PortDirection dir) const {
        if (!has_neighbor(id, dir)) return -1;
        auto [row, col] = get_router_pos(id);
        switch (dir) {
            case PortDirection::NORTH:
                return get_router_id((row == 0 ? rows - 1 : row - 1), col);
            case PortDirection::SOUTH:
                return get_router_id((row == rows - 1 ? 0 : row + 1), col);
            case PortDirection::EAST:
                return get_router_id(row, (col == cols - 1 ? 0 : col + 1));
            case PortDirection::WEST:
                return get_router_id(row, (col == 0 ? cols - 1 : col - 1));
            default: return -1;
        }
    }

    bool is_edge_router(uint8_t id) const {
        auto [row, col] = get_router_pos(id);
        return row == 0 || row == rows - 1 || col == 0 || col == cols - 1;
    }
};

// ============================================================================
// NoC Packet
// ============================================================================

/// Traffic class for QoS
enum class TrafficClass : uint8_t {
    BULK = 0,       // Normal tile transfers
    LOW_LATENCY = 1,// Control messages, triggers
    DMA = 2,        // DMA traffic
};

/// A packet being transferred through the NoC
struct NoCPacket {
    // Routing info
    uint8_t src_router = 0;         // Source router ID
    uint8_t dst_router = 0;         // Destination router ID
    PortDirection src_port = PortDirection::LOCAL;  // Source port type
    PortDirection dst_port = PortDirection::LOCAL;  // Destination port type

    // Payload description
    TileDescriptor tile;            // Tile being transferred
    uint32_t size_bytes = 0;        // Total size
    uint32_t num_flits = 0;         // Number of flits

    // Traffic class
    TrafficClass traffic_class = TrafficClass::BULK;

    // Timing
    uint64_t inject_cycle = 0;      // When packet was injected
    uint64_t head_arrive_cycle = 0; // When head flit arrived at destination
    uint64_t tail_arrive_cycle = 0; // When tail flit arrived (transfer complete)

    // Routing state
    uint8_t current_router = 0;     // Current position
    uint8_t hops_taken = 0;         // Hops traversed
    uint8_t flits_sent = 0;         // Flits sent (for wormhole)
    uint8_t flits_received = 0;     // Flits received

    // Sequence number for ordering
    uint64_t sequence_num = 0;

    // State
    bool is_complete() const { return flits_received >= num_flits; }
    bool is_head_arrived() const { return head_arrive_cycle > 0; }

    std::string to_string() const;
};

// ============================================================================
// NoC Link
// ============================================================================

/// A unidirectional link between two routers
class NoCLink {
public:
    NoCLink(uint8_t src_router, uint8_t dst_router,
            PortDirection src_port, uint32_t bandwidth, uint32_t latency);

    // Configuration
    uint8_t src_router_id() const { return src_router_; }
    uint8_t dst_router_id() const { return dst_router_; }
    PortDirection direction() const { return direction_; }

    // Availability
    bool is_available(uint64_t cycle) const;
    uint64_t available_at() const { return busy_until_cycle_; }

    // Transfer a flit
    bool send_flit(uint64_t cycle, uint32_t flit_bytes);

    // Query current state
    bool is_busy() const { return busy_; }
    const NoCPacket* current_packet() const { return current_packet_; }

    // Set packet being transferred (for visualization)
    void set_current_packet(const NoCPacket* pkt) { current_packet_ = pkt; }
    void clear_current_packet() { current_packet_ = nullptr; }

    // Statistics
    struct Stats {
        uint64_t flits_transferred = 0;
        uint64_t bytes_transferred = 0;
        uint64_t busy_cycles = 0;
        uint64_t idle_cycles = 0;
        uint64_t contention_events = 0;  // Times we were busy when requested
    };

    const Stats& stats() const { return stats_; }
    void update_stats(uint64_t cycle);

    std::string to_string() const;

private:
    uint8_t src_router_;
    uint8_t dst_router_;
    PortDirection direction_;

    uint32_t bandwidth_bytes_per_cycle_;
    uint32_t latency_cycles_;

    bool busy_ = false;
    uint64_t busy_until_cycle_ = 0;
    uint64_t last_update_cycle_ = 0;

    const NoCPacket* current_packet_ = nullptr;

    Stats stats_;
};

// ============================================================================
// NoC Router Input/Output Ports
// ============================================================================

/// Input port buffer
struct NoCInputPort {
    PortDirection direction = PortDirection::LOCAL;
    std::queue<NoCPacket> buffer;
    uint32_t buffer_capacity = 8;

    // Virtual channel buffers (if enabled)
    std::vector<std::queue<NoCPacket>> vc_buffers;

    // Credit tracking (for flow control)
    uint32_t credits_available = 0;

    bool can_accept() const { return buffer.size() < buffer_capacity; }
    bool has_packet() const { return !buffer.empty(); }

    // Statistics
    uint64_t packets_received = 0;
    uint64_t flits_received = 0;
    uint64_t buffer_full_events = 0;
};

/// Output port
struct NoCOutputPort {
    PortDirection direction = PortDirection::LOCAL;
    NoCLink* link = nullptr;            // Link to next router (nullptr for local)
    std::queue<NoCPacket> buffer;       // Output buffer
    uint32_t buffer_capacity = 4;

    // Credits from downstream
    uint32_t credits_available = 0;

    bool can_send(uint64_t cycle) const {
        return link == nullptr || (link->is_available(cycle) && credits_available > 0);
    }

    // Statistics
    uint64_t packets_sent = 0;
    uint64_t flits_sent = 0;
};

// ============================================================================
// NoC Router
// ============================================================================

/// A router in the NoC mesh
class NoCRouter {
public:
    explicit NoCRouter(uint8_t id, const NoCConfig& config);

    // Identification
    uint8_t id() const { return id_; }
    uint8_t row() const { return row_; }
    uint8_t col() const { return col_; }

    // Port access
    NoCInputPort& input_port(PortDirection dir) {
        return input_ports_[static_cast<size_t>(dir)];
    }
    NoCOutputPort& output_port(PortDirection dir) {
        return output_ports_[static_cast<size_t>(dir)];
    }

    // Connect output port to link
    void connect_output(PortDirection dir, NoCLink* link);

    // Packet injection (from local tile or DMA)
    bool can_inject(PortDirection port, uint64_t cycle) const;
    bool inject_packet(NoCPacket packet, PortDirection port, uint64_t cycle);

    // Packet ejection (to local tile or DMA)
    bool has_packet_for_local() const;
    std::optional<NoCPacket> eject_packet();

    // Simulation step
    void step(uint64_t cycle);

    // Routing decision
    PortDirection route(uint8_t dst_router) const;

    // Statistics
    struct Stats {
        uint64_t packets_routed = 0;
        uint64_t packets_injected = 0;
        uint64_t packets_ejected = 0;
        uint64_t routing_conflicts = 0;
        uint64_t cycles_active = 0;
    };

    const Stats& stats() const { return stats_; }

    std::string to_string() const;

private:
    uint8_t id_;
    uint8_t row_;
    uint8_t col_;
    const NoCConfig& config_;

    std::array<NoCInputPort, 6> input_ports_;   // N, S, E, W, Local, DMA
    std::array<NoCOutputPort, 6> output_ports_; // N, S, E, W, Local, DMA

    // Pending packets to route this cycle
    std::vector<std::pair<PortDirection, NoCPacket*>> route_requests_;

    Stats stats_;

    // Internal routing
    void process_inputs(uint64_t cycle);
    void arbitrate_and_forward(uint64_t cycle);
    PortDirection xy_route(uint8_t dst_router) const;
};

// ============================================================================
// NoC Interface for External Components
// ============================================================================

/// Callback when packet is delivered
using PacketDeliveryCallback = std::function<void(const NoCPacket&, uint64_t cycle)>;

/// Transfer request result
enum class TransferResult : uint8_t {
    SUCCESS,            // Transfer accepted
    BUFFER_FULL,        // Injection buffer full, try later
    LINK_BUSY,          // Output link busy, try later
    INVALID_ROUTE,      // No route to destination
    ERROR,              // Other error
};

// ============================================================================
// NoC Main Class
// ============================================================================

/// The Network-on-Chip
class NoC {
public:
    explicit NoC(const NoCConfig& config = NoCConfig{});

    // Configuration
    const NoCConfig& config() const { return config_; }

    // Tracing
    void set_tracer(NoCTracer* tracer) { tracer_ = tracer; }
    NoCTracer* tracer() const { return tracer_; }

    // Register delivery callbacks
    void set_l3_delivery_callback(uint8_t l3_id, PacketDeliveryCallback callback);
    void set_dma_delivery_callback(PacketDeliveryCallback callback);

    // ========== Transfer Interface ==========

    /// Check if injection is possible
    bool can_inject(uint8_t src_router, PortDirection port, uint64_t cycle) const;

    /// Inject a tile transfer (L3 to L3)
    TransferResult inject_l3_to_l3(
        uint8_t src_l3, uint8_t dst_l3,
        const TileDescriptor& tile,
        uint64_t cycle);

    /// Inject a DMA load (external memory to L3)
    TransferResult inject_dma_to_l3(
        uint8_t dst_l3,
        const TileDescriptor& tile,
        uint64_t cycle);

    /// Inject a DMA store (L3 to external memory)
    TransferResult inject_l3_to_dma(
        uint8_t src_l3,
        const TileDescriptor& tile,
        uint64_t cycle);

    // ========== Simulation ==========

    /// Advance simulation by one cycle
    void step(uint64_t cycle);

    /// Check if network is idle (no packets in flight)
    bool is_idle() const;

    /// Get number of packets in flight
    size_t packets_in_flight() const;

    // ========== Query ==========

    /// Estimate transfer latency (without contention)
    uint64_t estimate_latency(uint8_t src, uint8_t dst, uint32_t size_bytes) const;

    /// Get link state for tracing
    const NoCLink* get_link(uint8_t src, PortDirection dir) const;

    /// Get router
    const NoCRouter& router(uint8_t id) const { return *routers_[id]; }
    NoCRouter& router(uint8_t id) { return *routers_[id]; }

    // ========== Statistics ==========

    struct Stats {
        uint64_t total_packets = 0;
        uint64_t total_flits = 0;
        uint64_t total_bytes = 0;
        uint64_t total_hops = 0;
        uint64_t total_latency_cycles = 0;
        uint64_t max_latency_cycles = 0;
        uint64_t injection_blocked_cycles = 0;

        double avg_latency() const {
            return total_packets > 0 ?
                   static_cast<double>(total_latency_cycles) / static_cast<double>(total_packets) : 0.0;
        }
        double avg_hops() const {
            return total_packets > 0 ?
                   static_cast<double>(total_hops) / static_cast<double>(total_packets) : 0.0;
        }
    };

    const Stats& stats() const { return stats_; }
    void reset_stats();

    // ========== Debug ==========

    std::string to_string() const;
    void print_topology() const;
    std::string dump_state() const;

private:
    NoCConfig config_;

    // Routers
    std::vector<std::unique_ptr<NoCRouter>> routers_;

    // Links: [router_id][direction] -> link to neighbor
    std::vector<std::array<std::unique_ptr<NoCLink>, 4>> links_;

    // Delivery callbacks
    std::vector<PacketDeliveryCallback> l3_callbacks_;
    PacketDeliveryCallback dma_callback_;

    // Packets in flight
    std::vector<NoCPacket> in_flight_packets_;

    // Sequence counter
    uint64_t next_sequence_ = 0;

    // Statistics
    Stats stats_;

    // Tracer (optional)
    NoCTracer* tracer_ = nullptr;

    // Internal helpers
    void initialize_routers();
    void initialize_links();
    void connect_routers();
    NoCPacket create_packet(uint8_t src, uint8_t dst,
                            PortDirection src_port, PortDirection dst_port,
                            const TileDescriptor& tile, uint64_t cycle);
    void deliver_packets(uint64_t cycle);
    void update_stats(const NoCPacket& packet);
};

// ============================================================================
// NoC Tracer Integration
// ============================================================================

/// Trace event types for NoC
enum class NoCEventType : uint8_t {
    INJECT,         // Packet injected into network
    HOP,            // Packet moved to next router
    EJECT,          // Packet delivered to destination
    BLOCKED,        // Packet blocked (contention)
    LINK_BUSY,      // Link became busy
    LINK_IDLE,      // Link became idle
    FLIT_SEND,      // Individual FLIT sent on link (for progressive fill)
    FLIT_ARRIVE,    // Individual FLIT arrived at destination
};

/// A NoC trace event
struct NoCTraceEvent {
    uint64_t cycle = 0;
    NoCEventType type;
    uint8_t router_id = 0;
    PortDirection port = PortDirection::LOCAL;
    uint64_t packet_seq = 0;        // Packet sequence number
    TileDescriptor tile;            // Tile being transferred

    // FLIT-level information
    uint16_t flit_index = 0;        // Current FLIT index (0 to num_flits-1)
    uint16_t num_flits = 0;         // Total FLITs in packet
    uint8_t src_router = 0;         // Source router (for link tracking)
    uint8_t dst_router = 0;         // Destination router (for link tracking)

    std::string to_string() const;
};

/// NoC tracer for visualization
class NoCTracer {
public:
    NoCTracer() = default;

    void record_event(const NoCTraceEvent& event);
    void clear();

    const std::vector<NoCTraceEvent>& events() const { return events_; }
    size_t num_events() const { return events_.size(); }

    bool export_csv(const std::string& filename) const;
    bool export_chrome_trace(const std::string& filename, const NoCConfig& config) const;

private:
    std::vector<NoCTraceEvent> events_;
};

} // namespace sw::kpu::noc
