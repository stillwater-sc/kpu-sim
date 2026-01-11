// ============================================================================
// src/noc/dataflow_noc.cpp
// Unified Dataflow Network-on-Chip Implementation
// ============================================================================

#include <sw/kpu/models/temporal/noc/dataflow_noc.hpp>

#include <algorithm>
#include <stdexcept>

namespace sw::kpu::noc {

// ============================================================================
// DataflowRouter Implementation
// ============================================================================

DataflowRouter::DataflowRouter(const Config& config)
    : config_(config)
{
    // Initialize input buffers with configured depth
    for (auto& input : inputs_) {
        input = DataflowFlitBuffer(config_.buffer_depth);
    }
}

bool DataflowRouter::inject(Direction from, const DataflowFlit& flit) {
    size_t idx = static_cast<size_t>(from);
    if (!inputs_[idx].can_accept()) {
        stats_.back_pressure_cycles++;
        return false;  // Back-pressure
    }

    inputs_[idx].push(flit);
    stats_.flits_received++;
    return true;
}

void DataflowRouter::step(uint64_t /*cycle*/) {
    // Process all input buffers
    // This is the core dataflow loop: receive, match, act

    for (size_t i = 0; i < static_cast<size_t>(Direction::COUNT); ++i) {
        auto& input = inputs_[i];

        // Process up to one flit per input per cycle (can be tuned)
        if (!input.empty()) {
            DataflowFlit flit = input.front();

            // Can we forward/deliver this flit?
            if (flit.destination == config_.id) {
                // Local delivery - always possible
                input.pop();
                deliver_locally(flit);
            } else {
                // Need to forward - check if neighbor can accept
                Direction dir = toward(flit.destination);
                DataflowRouter* next = neighbors_[static_cast<size_t>(dir)];

                if (next && next->can_accept(opposite(dir))) {
                    input.pop();
                    next->inject(opposite(dir), flit);
                    stats_.flits_forwarded++;
                } else {
                    // Back-pressure from downstream
                    stats_.back_pressure_cycles++;
                }
            }
        }
    }
}

bool DataflowRouter::has_complete_tile() const {
    return !completed_tiles_.empty();
}

std::vector<DataflowFlit> DataflowRouter::pop_complete_tile(TileTag* tag_out) {
    if (completed_tiles_.empty()) {
        return {};
    }

    TileTag tag = completed_tiles_.front();
    completed_tiles_.pop();

    auto it = reassembly_.find(tag);
    if (it == reassembly_.end()) {
        return {};
    }

    if (tag_out) {
        *tag_out = tag;
    }

    std::vector<DataflowFlit> flits = it->second.flits();
    reassembly_.erase(it);
    return flits;
}

void DataflowRouter::set_neighbor(Direction dir, DataflowRouter* neighbor) {
    neighbors_[static_cast<size_t>(dir)] = neighbor;
}

bool DataflowRouter::can_accept(Direction from) const {
    return inputs_[static_cast<size_t>(from)].can_accept();
}

size_t DataflowRouter::pending_flits() const {
    size_t count = 0;
    for (const auto& input : inputs_) {
        count += input.size();
    }
    return count;
}

Direction DataflowRouter::toward(uint8_t destination) const {
    // Simple XY routing: move in X (columns) first, then Y (rows)
    // This matches the compute tile's systolic flow patterns

    uint8_t dst_row = destination / config_.mesh_cols;
    uint8_t dst_col = destination % config_.mesh_cols;

    // X dimension first (columns)
    if (dst_col > config_.col) return Direction::EAST;
    if (dst_col < config_.col) return Direction::WEST;

    // Y dimension (rows)
    if (dst_row > config_.row) return Direction::SOUTH;
    if (dst_row < config_.row) return Direction::NORTH;

    // Should never reach here if destination != id
    return Direction::LOCAL;
}

void DataflowRouter::deliver_locally(const DataflowFlit& flit) {
    stats_.flits_delivered++;

    // Find or create reassembly entry
    auto& partial = reassembly_[flit.tag];

    if (partial.empty()) {
        // First flit of this tile
        partial.init(flit.total_flits, flit.tile_size);
    }

    // Add this flit
    partial.add_flit(flit);

    // Check if tile is complete
    if (partial.complete()) {
        stats_.tiles_completed++;
        completed_tiles_.push(flit.tag);
    }
}

// ============================================================================
// DataflowNoC Implementation
// ============================================================================

DataflowNoC::DataflowNoC()
    : DataflowNoC(Config{})
{
}

DataflowNoC::DataflowNoC(const Config& config)
    : config_(config)
{
    // Create all routers
    routers_.reserve(config_.rows * config_.cols);

    // Create per-router injection queues
    injection_queues_.resize(config_.rows * config_.cols);

    for (uint8_t row = 0; row < config_.rows; ++row) {
        for (uint8_t col = 0; col < config_.cols; ++col) {
            DataflowRouter::Config router_config;
            router_config.id = router_id(row, col);
            router_config.row = row;
            router_config.col = col;
            router_config.mesh_rows = config_.rows;
            router_config.mesh_cols = config_.cols;
            router_config.buffer_depth = config_.buffer_depth;

            routers_.emplace_back(router_config);
        }
    }

    // Connect neighbors
    for (uint8_t row = 0; row < config_.rows; ++row) {
        for (uint8_t col = 0; col < config_.cols; ++col) {
            DataflowRouter& r = router(row, col);

            // North neighbor (row - 1)
            if (row > 0) {
                r.set_neighbor(Direction::NORTH, &router(row - 1, col));
            }
            // South neighbor (row + 1)
            if (row < config_.rows - 1) {
                r.set_neighbor(Direction::SOUTH, &router(row + 1, col));
            }
            // West neighbor (col - 1)
            if (col > 0) {
                r.set_neighbor(Direction::WEST, &router(row, col - 1));
            }
            // East neighbor (col + 1)
            if (col < config_.cols - 1) {
                r.set_neighbor(Direction::EAST, &router(row, col + 1));
            }
        }
    }
}

bool DataflowNoC::inject_tile(uint8_t src_router, uint8_t dst_router,
                               const TileDescriptor& tile, uint64_t cycle) {
    if (src_router >= routers_.size()) {
        return false;
    }

    // Create the tile tag
    TileTag tag(tile);

    // Calculate number of flits
    uint16_t num_flits = flits_for_tile(tile.size);

    // Track this tile as in-flight
    InflightTile inflight;
    inflight.tag = tag;
    inflight.dst_router = dst_router;
    inflight.inject_cycle = cycle;
    inflight_[tag] = inflight;
    inflight_count_++;

    // Generate flits and add to injection queue
    // Flits will be injected into the router as buffer space becomes available
    auto& queue = injection_queues_[src_router];
    for (uint16_t i = 0; i < num_flits; ++i) {
        DataflowFlit flit;
        flit.tag = tag;
        flit.source = src_router;
        flit.destination = dst_router;
        flit.flit_index = i;
        flit.total_flits = num_flits;
        flit.tile_size = tile.size;
        flit.tile_height = tile.height;
        flit.tile_width = tile.width;
        flit.inject_cycle = cycle;

        queue.push(flit);
    }

    stats_.total_tiles_injected++;
    stats_.total_flits_transferred += num_flits;

    return true;
}

void DataflowNoC::step(uint64_t cycle) {
    // First, inject pending flits from queues into routers
    process_injections();

    // Step all routers
    for (auto& r : routers_) {
        r.step(cycle);
    }

    // Check for completed deliveries
    check_deliveries(cycle);
}

void DataflowNoC::process_injections() {
    // For each router, inject as many flits as possible from its queue
    for (size_t i = 0; i < injection_queues_.size(); ++i) {
        auto& queue = injection_queues_[i];
        auto& r = routers_[i];

        // Inject flits while queue is non-empty and router can accept
        while (!queue.empty() && r.can_accept(Direction::LOCAL)) {
            r.inject(Direction::LOCAL, queue.front());
            queue.pop();
        }
    }
}

void DataflowNoC::drain(uint64_t& cycle) {
    uint64_t start_cycle = cycle;

    while (has_inflight_tiles() || has_pending_flits()) {
        step(cycle);
        cycle++;

        // Safety limit to prevent infinite loops
        if (cycle - start_cycle > 1000000) {
            throw std::runtime_error("DataflowNoC::drain() timeout");
        }
    }
}

bool DataflowNoC::has_inflight_tiles() const {
    return inflight_count_ > 0;
}

bool DataflowNoC::has_pending_flits() const {
    // Check injection queues
    for (const auto& queue : injection_queues_) {
        if (!queue.empty()) return true;
    }

    // Check router buffers
    for (const auto& r : routers_) {
        if (r.pending_flits() > 0) return true;
    }

    return false;
}

DataflowRouter& DataflowNoC::router(uint8_t id) {
    if (id >= routers_.size()) {
        throw std::out_of_range("Router ID out of range");
    }
    return routers_[id];
}

DataflowRouter& DataflowNoC::router(uint8_t row, uint8_t col) {
    return router(router_id(row, col));
}

const DataflowRouter& DataflowNoC::router(uint8_t id) const {
    if (id >= routers_.size()) {
        throw std::out_of_range("Router ID out of range");
    }
    return routers_[id];
}

void DataflowNoC::check_deliveries(uint64_t cycle) {
    // Check each router for completed tiles
    for (auto& r : routers_) {
        while (r.has_complete_tile()) {
            TileTag tag;
            auto flits = r.pop_complete_tile(&tag);

            if (flits.empty()) continue;

            // Find the in-flight record
            auto it = inflight_.find(tag);
            if (it != inflight_.end()) {
                uint64_t inject_cycle = it->second.inject_cycle;
                uint64_t latency = cycle - inject_cycle;

                stats_.total_tiles_delivered++;
                stats_.total_latency_cycles += latency;
                stats_.max_latency_cycles = std::max(stats_.max_latency_cycles, latency);

                // Invoke callback if set
                if (delivery_callback_) {
                    delivery_callback_(r.id(), tag, inject_cycle, cycle);
                }

                inflight_.erase(it);
                inflight_count_--;
            }
        }
    }
}

} // namespace sw::kpu::noc
