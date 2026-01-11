// ============================================================================
// include/sw/kpu/noc/dataflow_noc.hpp
// Unified Dataflow Network-on-Chip
//
// This NoC treats routing as another layer of the Domain Flow Graph:
// - Tagged tokens (flits carry tile identity + destination)
// - Local decisions ("is this for me?" → consume or forward)
// - Stateless forwarding (no path reservation)
// - Tag-based reassembly at destination
//
// This aligns with the compute tile model: push-based, self-organizing,
// local tag matching triggers action.
// ============================================================================

#pragma once

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace sw::kpu::noc {

// ============================================================================
// TileTag - The identity signature of a tile
// ============================================================================

/// Compact tile identity for dataflow matching
/// This is the "tag" that tiles carry through the network
struct TileTag {
    TensorId tensor = TensorId::A;
    uint16_t m_tile = 0;
    uint16_t n_tile = 0;
    uint16_t k_tile = 0;

    TileTag() = default;

    TileTag(TensorId t, uint16_t m, uint16_t n, uint16_t k)
        : tensor(t), m_tile(m), n_tile(n), k_tile(k) {}

    explicit TileTag(const TileDescriptor& desc)
        : tensor(desc.tensor), m_tile(desc.m_tile),
          n_tile(desc.n_tile), k_tile(desc.k_tile) {}

    bool operator==(const TileTag& other) const {
        return tensor == other.tensor &&
               m_tile == other.m_tile &&
               n_tile == other.n_tile &&
               k_tile == other.k_tile;
    }

    bool operator!=(const TileTag& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        return std::string(sw::kpu::to_string(tensor)) +
               "[" + std::to_string(m_tile) +
               "," + std::to_string(n_tile) +
               "," + std::to_string(k_tile) + "]";
    }
};

// Hash function for TileTag (for use in unordered_map)
struct TileTagHash {
    size_t operator()(const TileTag& tag) const {
        size_t h = static_cast<size_t>(tag.tensor);
        h = h * 31 + tag.m_tile;
        h = h * 31 + tag.n_tile;
        h = h * 31 + tag.k_tile;
        return h;
    }
};

// ============================================================================
// DataflowFlit - Self-describing network token
// ============================================================================

/// A flit in the dataflow NoC
/// Each flit carries its own identity and destination - no external state needed
struct DataflowFlit {
    // === Tile Identity (the "tag") ===
    TileTag tag;

    // === Routing ===
    uint8_t source = 0;           // Where this tile originated
    uint8_t destination = 0;      // Where this tile should be consumed

    // === Reassembly ===
    uint16_t flit_index = 0;      // Which flit of the tile (0, 1, 2, ...)
    uint16_t total_flits = 1;     // Total flits in this tile

    // === Tile metadata (only in first flit) ===
    uint32_t tile_size = 0;       // Total tile size in bytes
    uint16_t tile_height = 0;     // Tile dimensions
    uint16_t tile_width = 0;

    // === Timing ===
    uint64_t inject_cycle = 0;    // When this flit entered the network

    // === Payload ===
    static constexpr size_t HEADER_SIZE = 24;  // Bytes used by header
    static constexpr size_t FLIT_SIZE = 64;    // Total flit size
    static constexpr size_t PAYLOAD_SIZE = FLIT_SIZE - HEADER_SIZE;

    std::array<uint8_t, PAYLOAD_SIZE> payload{};

    // Helpers
    bool is_first() const { return flit_index == 0; }
    bool is_last() const { return flit_index == total_flits - 1; }
    bool is_single() const { return total_flits == 1; }

    std::string to_string() const {
        return tag.to_string() +
               " [" + std::to_string(flit_index) + "/" + std::to_string(total_flits) + "]" +
               " → L3[" + std::to_string(destination) + "]";
    }
};

// ============================================================================
// Direction - Port directions
// ============================================================================

enum class Direction : uint8_t {
    NORTH = 0,
    SOUTH = 1,
    EAST = 2,
    WEST = 3,
    LOCAL = 4,  // Injection/ejection from local L3
    COUNT = 5
};

inline Direction opposite(Direction dir) {
    switch (dir) {
        case Direction::NORTH: return Direction::SOUTH;
        case Direction::SOUTH: return Direction::NORTH;
        case Direction::EAST:  return Direction::WEST;
        case Direction::WEST:  return Direction::EAST;
        default: return Direction::LOCAL;
    }
}

inline const char* to_string(Direction dir) {
    switch (dir) {
        case Direction::NORTH: return "NORTH";
        case Direction::SOUTH: return "SOUTH";
        case Direction::EAST:  return "EAST";
        case Direction::WEST:  return "WEST";
        case Direction::LOCAL: return "LOCAL";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// DataflowFlitBuffer - Simple FIFO buffer with back-pressure
// ============================================================================

class DataflowFlitBuffer {
public:
    explicit DataflowFlitBuffer(size_t max_depth = 8) : max_depth_(max_depth) {}

    bool can_accept() const { return buffer_.size() < max_depth_; }
    bool empty() const { return buffer_.empty(); }
    size_t size() const { return buffer_.size(); }
    size_t capacity() const { return max_depth_; }

    bool push(const DataflowFlit& flit) {
        if (!can_accept()) {
            return false;  // Back-pressure: caller must retry
        }
        buffer_.push(flit);
        return true;
    }

    DataflowFlit pop() {
        DataflowFlit flit = buffer_.front();
        buffer_.pop();
        return flit;
    }

    const DataflowFlit& front() const { return buffer_.front(); }

private:
    std::queue<DataflowFlit> buffer_;
    size_t max_depth_;
};

// ============================================================================
// PartialTile - Reassembly buffer for incoming tiles
// ============================================================================

/// Collects flits belonging to a tile until complete
class PartialTile {
public:
    PartialTile() = default;

    void init(uint16_t total_flits, uint32_t tile_size) {
        total_flits_ = total_flits;
        tile_size_ = tile_size;
        received_.assign(total_flits, false);
        flits_.resize(total_flits);
        flits_received_ = 0;
    }

    bool add_flit(const DataflowFlit& flit) {
        if (flit.flit_index >= total_flits_) return false;
        if (received_[flit.flit_index]) return false;  // Duplicate

        received_[flit.flit_index] = true;
        flits_[flit.flit_index] = flit;
        flits_received_++;
        return true;
    }

    bool complete() const { return total_flits_ > 0 && flits_received_ == total_flits_; }
    bool empty() const { return total_flits_ == 0; }
    uint16_t flits_received() const { return flits_received_; }
    uint16_t flits_expected() const { return total_flits_; }
    uint32_t tile_size() const { return tile_size_; }

    // Get the assembled tile data
    const std::vector<DataflowFlit>& flits() const { return flits_; }

private:
    uint16_t total_flits_ = 0;
    uint32_t tile_size_ = 0;
    uint16_t flits_received_ = 0;
    std::vector<bool> received_;
    std::vector<DataflowFlit> flits_;
};

// ============================================================================
// DataflowRouter - A dataflow node in the network
// ============================================================================

/// Router that operates on dataflow principles:
/// - Receives tagged flits
/// - Makes local decision: "is destination == my_id?"
/// - If yes: deliver to local L3
/// - If no: forward toward destination
///
/// No path reservation, no credits, no complex state machines.
/// Just like a compute tile matching tags and triggering actions.
class DataflowRouter {
public:
    struct Config {
        uint8_t id = 0;
        uint8_t row = 0;
        uint8_t col = 0;
        uint8_t mesh_rows = 4;
        uint8_t mesh_cols = 4;
        size_t buffer_depth = 8;      // Flits per input buffer
        uint32_t forward_latency = 1; // Cycles to forward a flit
    };

    struct Stats {
        uint64_t flits_received = 0;
        uint64_t flits_forwarded = 0;
        uint64_t flits_delivered = 0;
        uint64_t tiles_completed = 0;
        uint64_t back_pressure_cycles = 0;
    };

    explicit DataflowRouter(const Config& config);

    // === Core dataflow operations ===

    /// Inject a flit from a neighbor or local injection
    /// Returns false if buffer is full (back-pressure)
    bool inject(Direction from, const DataflowFlit& flit);

    /// Process one cycle: forward flits toward their destinations
    void step(uint64_t cycle);

    /// Check if a tile has been fully received
    bool has_complete_tile() const;

    /// Get and remove a completed tile (returns empty vector if none)
    std::vector<DataflowFlit> pop_complete_tile(TileTag* tag_out = nullptr);

    // === Neighbor connections ===
    void set_neighbor(Direction dir, DataflowRouter* neighbor);
    DataflowRouter* neighbor(Direction dir) const { return neighbors_[static_cast<size_t>(dir)]; }

    // === Accessors ===
    uint8_t id() const { return config_.id; }
    uint8_t row() const { return config_.row; }
    uint8_t col() const { return config_.col; }
    const Stats& stats() const { return stats_; }

    // === Status ===
    bool can_accept(Direction from) const;
    size_t pending_flits() const;

private:
    // The core dataflow decision
    Direction toward(uint8_t destination) const;

    // Process a single flit
    void process_flit(const DataflowFlit& flit, uint64_t cycle);

    // Deliver flit to local L3 reassembly
    void deliver_locally(const DataflowFlit& flit);

    Config config_;
    Stats stats_;

    // Input buffers (one per direction including LOCAL for injection)
    std::array<DataflowFlitBuffer, static_cast<size_t>(Direction::COUNT)> inputs_;

    // Neighbor pointers (null if at mesh edge)
    std::array<DataflowRouter*, static_cast<size_t>(Direction::COUNT)> neighbors_{};

    // Tile reassembly (tag → partial tile)
    std::unordered_map<TileTag, PartialTile, TileTagHash> reassembly_;

    // Completed tiles ready for consumption
    std::queue<TileTag> completed_tiles_;
};

// ============================================================================
// DataflowNoC - The complete network
// ============================================================================

/// Network of DataflowRouters operating on unified dataflow principles
class DataflowNoC {
public:
    struct Config {
        uint8_t rows = 4;
        uint8_t cols = 4;
        size_t buffer_depth = 8;
        uint32_t flit_size = 64;
        uint32_t link_bandwidth_bytes_per_cycle = 64;
    };

    struct Stats {
        uint64_t total_tiles_injected = 0;
        uint64_t total_tiles_delivered = 0;
        uint64_t total_flits_transferred = 0;
        uint64_t total_latency_cycles = 0;  // Sum of all tile latencies
        uint64_t max_latency_cycles = 0;
    };

    /// Callback when a tile is fully delivered
    using DeliveryCallback = std::function<void(
        uint8_t dst_router,
        const TileTag& tag,
        uint64_t inject_cycle,
        uint64_t complete_cycle
    )>;

    DataflowNoC();  // Uses default Config
    explicit DataflowNoC(const Config& config);

    // === Tile injection ===

    /// Inject a tile into the network
    /// The tile is split into flits and pushed toward destination
    /// Returns false if injection buffer is full
    bool inject_tile(uint8_t src_router, uint8_t dst_router,
                     const TileDescriptor& tile, uint64_t cycle);

    // === Simulation ===

    /// Advance simulation by one cycle
    void step(uint64_t cycle);

    /// Run until all in-flight tiles are delivered
    void drain(uint64_t& cycle);

    // === Callbacks ===
    void set_delivery_callback(DeliveryCallback cb) { delivery_callback_ = std::move(cb); }

    // === Accessors ===
    DataflowRouter& router(uint8_t id);
    DataflowRouter& router(uint8_t row, uint8_t col);
    const DataflowRouter& router(uint8_t id) const;

    uint8_t rows() const { return config_.rows; }
    uint8_t cols() const { return config_.cols; }
    uint8_t num_routers() const { return config_.rows * config_.cols; }

    const Stats& stats() const { return stats_; }
    const Config& config() const { return config_; }

    // === Utility ===
    uint8_t router_id(uint8_t row, uint8_t col) const {
        return row * config_.cols + col;
    }

    /// Check if any tiles are in flight
    bool has_inflight_tiles() const;
    size_t inflight_count() const { return inflight_count_; }

    /// Check if there are any pending flits (in queues or router buffers)
    bool has_pending_flits() const;

private:
    void check_deliveries(uint64_t cycle);

    Config config_;
    Stats stats_;
    std::vector<DataflowRouter> routers_;
    DeliveryCallback delivery_callback_;

    // Track in-flight tiles
    struct InflightTile {
        TileTag tag;
        uint8_t dst_router;
        uint64_t inject_cycle;
    };
    std::unordered_map<TileTag, InflightTile, TileTagHash> inflight_;
    size_t inflight_count_ = 0;

    // Per-router injection queues (for pipelined injection)
    std::vector<std::queue<DataflowFlit>> injection_queues_;

    // Process pending injections
    void process_injections();
};

// ============================================================================
// Utility functions
// ============================================================================

/// Calculate number of flits needed for a tile
inline uint16_t flits_for_tile(uint32_t tile_size_bytes) {
    return static_cast<uint16_t>(
        (tile_size_bytes + DataflowFlit::PAYLOAD_SIZE - 1) / DataflowFlit::PAYLOAD_SIZE
    );
}

/// Calculate Manhattan distance between routers
inline uint8_t manhattan_distance(uint8_t src, uint8_t dst, uint8_t cols) {
    uint8_t src_row = src / cols;
    uint8_t src_col = src % cols;
    uint8_t dst_row = dst / cols;
    uint8_t dst_col = dst % cols;

    uint8_t row_dist = (src_row > dst_row) ? (src_row - dst_row) : (dst_row - src_row);
    uint8_t col_dist = (src_col > dst_col) ? (src_col - dst_col) : (dst_col - src_col);

    return row_dist + col_dist;
}

} // namespace sw::kpu::noc
