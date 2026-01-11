// ============================================================================
// include/sw/kpu/models/temporal/datamovement/l3_interconnect.hpp
// L3 Tile Mesh Interconnect for inter-tile data movement
// ============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "block_mover_isa.hpp"

namespace sw::kpu {

// Forward declarations
class StatefulBlockMover;

// ============================================================================
// Interconnect Topology
// ============================================================================

/// Supported interconnect topologies for L3 tiles
enum class InterconnectTopology : uint8_t {
    MESH_2D,        // 2D mesh (4×4 default) - N/S/E/W connections
    RING,           // Ring topology - each tile connects to 2 neighbors
    TORUS,          // 2D torus - mesh with wraparound
    CROSSBAR,       // Full crossbar - any-to-any
};

/// Direction for mesh-based transfers
enum class Direction : uint8_t {
    NORTH = 0,
    SOUTH = 1,
    EAST = 2,
    WEST = 3,
    LOCAL = 4,      // Same tile (no transfer needed)
};

inline const char* to_string(Direction dir) {
    switch (dir) {
        case Direction::NORTH: return "NORTH";
        case Direction::SOUTH: return "SOUTH";
        case Direction::EAST: return "EAST";
        case Direction::WEST: return "WEST";
        case Direction::LOCAL: return "LOCAL";
        default: return "UNKNOWN";
    }
}

inline Direction opposite(Direction dir) {
    switch (dir) {
        case Direction::NORTH: return Direction::SOUTH;
        case Direction::SOUTH: return Direction::NORTH;
        case Direction::EAST: return Direction::WEST;
        case Direction::WEST: return Direction::EAST;
        default: return Direction::LOCAL;
    }
}

// ============================================================================
// L3 Tile Position
// ============================================================================

/// Position of an L3 tile in the mesh
struct L3Position {
    uint8_t row = 0;    // 0 to rows-1
    uint8_t col = 0;    // 0 to cols-1

    L3Position() = default;
    L3Position(uint8_t r, uint8_t c) : row(r), col(c) {}

    /// Convert flat ID to position
    static L3Position from_id(uint8_t id, uint8_t cols) {
        return L3Position(id / cols, id % cols);
    }

    /// Convert position to flat ID
    uint8_t to_id(uint8_t cols) const {
        return row * cols + col;
    }

    bool operator==(const L3Position& other) const {
        return row == other.row && col == other.col;
    }

    std::string to_string() const {
        return "(" + std::to_string(row) + "," + std::to_string(col) + ")";
    }
};

// ============================================================================
// Transfer Packet
// ============================================================================

/// A packet of data being transferred between L3 tiles
struct L3TransferPacket {
    uint8_t src_l3_id = 0;          // Source L3 tile
    uint8_t dst_l3_id = 0;          // Destination L3 tile
    TileDescriptor tile;             // What tile is being transferred
    std::vector<uint8_t> data;       // The actual data (optional for simulation)

    // Routing
    Direction ingress_dir = Direction::LOCAL;   // Direction packet arrived from
    uint8_t hops_remaining = 0;                  // For multi-hop routing

    // Timing
    uint64_t inject_cycle = 0;       // When packet was injected
    uint64_t arrival_cycle = 0;      // When packet arrived at destination

    size_t size_bytes() const { return data.size(); }
};

// ============================================================================
// Interconnect Link
// ============================================================================

/// A single directional link between two L3 tiles
struct InterconnectLink {
    uint8_t src_l3_id = 0;
    uint8_t dst_l3_id = 0;
    Direction direction = Direction::LOCAL;

    // Link characteristics
    uint32_t bandwidth_bytes_per_cycle = 64;    // 64 bytes/cycle = 32 GB/s @ 500MHz
    uint32_t latency_cycles = 2;                 // Wire latency

    // Current state
    bool busy = false;
    uint64_t busy_until_cycle = 0;

    // Statistics
    uint64_t bytes_transferred = 0;
    uint64_t packets_transferred = 0;

    bool is_available(uint64_t cycle) const {
        return !busy || cycle >= busy_until_cycle;
    }

    uint32_t transfer_cycles(size_t bytes) const {
        return static_cast<uint32_t>(latency_cycles + (bytes + bandwidth_bytes_per_cycle - 1) / bandwidth_bytes_per_cycle);
    }
};

// ============================================================================
// L3 Interconnect Configuration
// ============================================================================

struct L3InterconnectConfig {
    InterconnectTopology topology = InterconnectTopology::MESH_2D;

    // Mesh dimensions (for MESH_2D and TORUS)
    uint8_t rows = 4;
    uint8_t cols = 4;

    // Link parameters
    uint32_t link_bandwidth_bytes_per_cycle = 64;   // 64B/cycle per link
    uint32_t link_latency_cycles = 2;                // Per-hop latency

    // Buffering
    size_t input_buffer_depth = 4;      // Packets buffered at each input port
    size_t output_buffer_depth = 4;     // Packets buffered at each output port

    // Routing
    bool allow_multi_hop = true;        // Route through intermediate tiles
    bool use_wormhole = false;          // Wormhole vs store-and-forward

    size_t num_tiles() const { return rows * cols; }

    L3Position get_position(uint8_t id) const {
        return L3Position::from_id(id, cols);
    }

    uint8_t get_id(L3Position pos) const {
        return pos.to_id(cols);
    }

    /// Get neighbor in given direction (returns -1 if no neighbor)
    int get_neighbor(uint8_t id, Direction dir) const {
        L3Position pos = get_position(id);

        switch (dir) {
            case Direction::NORTH:
                if (pos.row == 0) {
                    return (topology == InterconnectTopology::TORUS) ?
                           get_id(L3Position(rows - 1, pos.col)) : -1;
                }
                return get_id(L3Position(pos.row - 1, pos.col));

            case Direction::SOUTH:
                if (pos.row == rows - 1) {
                    return (topology == InterconnectTopology::TORUS) ?
                           get_id(L3Position(0, pos.col)) : -1;
                }
                return get_id(L3Position(pos.row + 1, pos.col));

            case Direction::EAST:
                if (pos.col == cols - 1) {
                    return (topology == InterconnectTopology::TORUS) ?
                           get_id(L3Position(pos.row, 0)) : -1;
                }
                return get_id(L3Position(pos.row, pos.col + 1));

            case Direction::WEST:
                if (pos.col == 0) {
                    return (topology == InterconnectTopology::TORUS) ?
                           get_id(L3Position(pos.row, cols - 1)) : -1;
                }
                return get_id(L3Position(pos.row, pos.col - 1));

            default:
                return -1;
        }
    }

    /// Check if two tiles are neighbors
    bool are_neighbors(uint8_t id1, uint8_t id2) const {
        for (auto dir : {Direction::NORTH, Direction::SOUTH, Direction::EAST, Direction::WEST}) {
            if (get_neighbor(id1, dir) == static_cast<int>(id2)) {
                return true;
            }
        }
        return false;
    }

    /// Get direction from id1 to id2 (if neighbors)
    std::optional<Direction> get_direction(uint8_t from_id, uint8_t to_id) const {
        for (auto dir : {Direction::NORTH, Direction::SOUTH, Direction::EAST, Direction::WEST}) {
            if (get_neighbor(from_id, dir) == static_cast<int>(to_id)) {
                return dir;
            }
        }
        return std::nullopt;
    }

    /// Compute Manhattan distance between two tiles
    uint8_t manhattan_distance(uint8_t id1, uint8_t id2) const {
        L3Position p1 = get_position(id1);
        L3Position p2 = get_position(id2);

        uint8_t row_dist = (p1.row > p2.row) ? (p1.row - p2.row) : (p2.row - p1.row);
        uint8_t col_dist = (p1.col > p2.col) ? (p1.col - p2.col) : (p2.col - p1.col);

        if (topology == InterconnectTopology::TORUS) {
            // Consider wraparound
            row_dist = std::min(row_dist, static_cast<uint8_t>(rows - row_dist));
            col_dist = std::min(col_dist, static_cast<uint8_t>(cols - col_dist));
        }

        return row_dist + col_dist;
    }
};

// ============================================================================
// L3 Interconnect
// ============================================================================

/// The L3 tile interconnect network
class L3Interconnect {
public:
    using PacketCallback = std::function<void(const L3TransferPacket&)>;

    explicit L3Interconnect(const L3InterconnectConfig& config = L3InterconnectConfig{});

    // Configuration
    const L3InterconnectConfig& config() const { return config_; }

    // Register callbacks for packet delivery
    void set_delivery_callback(uint8_t l3_id, PacketCallback callback);

    // Inject a packet into the network
    bool inject_packet(const L3TransferPacket& packet, uint64_t cycle);

    // Advance simulation by one cycle
    void step(uint64_t cycle);

    // Query state
    bool is_idle() const;
    size_t packets_in_flight() const;

    // Get next hop for routing (XY routing for mesh)
    std::optional<Direction> route_next_hop(uint8_t current_id, uint8_t dest_id) const;

    // Compute transfer latency estimate
    uint32_t estimate_transfer_cycles(uint8_t src_id, uint8_t dst_id, size_t bytes) const;

    // Statistics
    struct Stats {
        uint64_t total_packets = 0;
        uint64_t total_bytes = 0;
        uint64_t total_hops = 0;
        uint64_t max_latency_cycles = 0;
        uint64_t total_latency_cycles = 0;

        double avg_latency_cycles() const {
            return total_packets > 0 ? static_cast<double>(total_latency_cycles) / total_packets : 0.0;
        }
        double avg_hops() const {
            return total_packets > 0 ? static_cast<double>(total_hops) / total_packets : 0.0;
        }
    };

    const Stats& stats() const { return stats_; }
    void reset_stats();

    // Debug
    std::string to_string() const;
    void print_topology() const;

private:
    L3InterconnectConfig config_;

    // Links indexed by [src_id][direction]
    std::vector<std::array<InterconnectLink, 4>> links_;

    // Input buffers at each tile, indexed by [l3_id][direction]
    std::vector<std::array<std::queue<L3TransferPacket>, 4>> input_buffers_;

    // Packets waiting to be injected
    std::queue<std::pair<uint64_t, L3TransferPacket>> injection_queue_;

    // Callbacks for packet delivery
    std::vector<PacketCallback> delivery_callbacks_;

    // Statistics
    Stats stats_;

    // Internal helpers
    void initialize_links();
    void process_link(InterconnectLink& link, uint64_t cycle);
    void deliver_packet(L3TransferPacket& packet, uint64_t cycle);
};

// ============================================================================
// Trigger Network
// ============================================================================

/// Lightweight trigger/event network between L3 tiles
class TriggerNetwork {
public:
    explicit TriggerNetwork(size_t num_tiles = 16);

    /// Send a trigger from one tile to others
    void send_trigger(uint8_t from_l3, uint8_t channel, uint16_t dest_mask);

    /// Check if a trigger is pending at a tile
    bool is_trigger_pending(uint8_t l3_id, uint8_t channel) const;

    /// Check if any of the masked triggers are pending
    bool any_trigger_pending(uint8_t l3_id, uint16_t channel_mask) const;

    /// Clear a trigger (after handling)
    void clear_trigger(uint8_t l3_id, uint8_t channel);

    /// Clear all triggers matching mask
    void clear_triggers(uint8_t l3_id, uint16_t channel_mask);

    /// Advance by one cycle (triggers propagate with 1-cycle latency)
    void step(uint64_t cycle);

    /// Reset all triggers
    void reset();

private:
    size_t num_tiles_;

    // Pending triggers: [l3_id] -> bitmask of pending trigger channels
    std::vector<uint16_t> pending_triggers_;

    // Triggers in flight (1-cycle latency)
    struct InFlightTrigger {
        uint8_t dest_l3_id;
        uint8_t channel;
        uint64_t delivery_cycle;
    };
    std::queue<InFlightTrigger> in_flight_;
};

} // namespace sw::kpu
