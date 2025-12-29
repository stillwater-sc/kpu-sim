// ============================================================================
// src/components/datamovement/l3_interconnect.cpp
// L3 Tile Mesh Interconnect Implementation
// ============================================================================

#include "sw/kpu/components/l3_interconnect.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace sw::kpu {

// ============================================================================
// L3Interconnect Implementation
// ============================================================================

L3Interconnect::L3Interconnect(const L3InterconnectConfig& config)
    : config_(config)
    , delivery_callbacks_(config.num_tiles())
{
    initialize_links();
}

void L3Interconnect::initialize_links() {
    size_t num_tiles = config_.num_tiles();

    // Initialize link arrays
    links_.resize(num_tiles);
    input_buffers_.resize(num_tiles);

    // Create links for each tile
    for (size_t id = 0; id < num_tiles; ++id) {
        for (size_t dir = 0; dir < 4; ++dir) {
            Direction d = static_cast<Direction>(dir);
            int neighbor = config_.get_neighbor(static_cast<uint8_t>(id), d);

            InterconnectLink& link = links_[id][dir];
            link.src_l3_id = static_cast<uint8_t>(id);
            link.dst_l3_id = (neighbor >= 0) ? static_cast<uint8_t>(neighbor) : 0;
            link.direction = d;
            link.bandwidth_bytes_per_cycle = config_.link_bandwidth_bytes_per_cycle;
            link.latency_cycles = config_.link_latency_cycles;
            link.busy = false;
        }
    }
}

void L3Interconnect::set_delivery_callback(uint8_t l3_id, PacketCallback callback) {
    if (l3_id < delivery_callbacks_.size()) {
        delivery_callbacks_[l3_id] = std::move(callback);
    }
}

bool L3Interconnect::inject_packet(const L3TransferPacket& packet, uint64_t cycle) {
    if (packet.src_l3_id >= config_.num_tiles() ||
        packet.dst_l3_id >= config_.num_tiles()) {
        return false;
    }

    // Local transfer (same tile)
    if (packet.src_l3_id == packet.dst_l3_id) {
        L3TransferPacket local_packet = packet;
        local_packet.arrival_cycle = cycle;
        deliver_packet(local_packet, cycle);
        return true;
    }

    // Get next hop direction
    auto next_dir = route_next_hop(packet.src_l3_id, packet.dst_l3_id);
    if (!next_dir) {
        return false;  // No route
    }

    // Check if link is available
    InterconnectLink& link = links_[packet.src_l3_id][static_cast<size_t>(*next_dir)];
    if (!link.is_available(cycle)) {
        // Queue for later injection - use next cycle to avoid infinite loop
        L3TransferPacket queued = packet;
        queued.inject_cycle = cycle;
        injection_queue_.push({cycle + 1, queued});
        return true;
    }

    // Start transfer on link
    L3TransferPacket in_flight = packet;
    in_flight.inject_cycle = cycle;
    in_flight.ingress_dir = *next_dir;
    in_flight.hops_remaining = config_.manhattan_distance(packet.src_l3_id, packet.dst_l3_id);

    uint32_t transfer_cycles = link.transfer_cycles(packet.size_bytes());
    link.busy = true;
    link.busy_until_cycle = cycle + transfer_cycles;
    link.bytes_transferred += packet.size_bytes();
    link.packets_transferred++;

    // Queue packet at destination's input buffer
    uint8_t next_hop_id = static_cast<uint8_t>(config_.get_neighbor(packet.src_l3_id, *next_dir));
    Direction ingress = opposite(*next_dir);
    in_flight.arrival_cycle = cycle + transfer_cycles;

    input_buffers_[next_hop_id][static_cast<size_t>(ingress)].push(in_flight);

    stats_.total_packets++;
    stats_.total_bytes += packet.size_bytes();
    stats_.total_hops++;

    return true;
}

void L3Interconnect::step(uint64_t cycle) {
    // Process queued injections
    while (!injection_queue_.empty()) {
        auto& [inject_time, packet] = injection_queue_.front();
        if (inject_time <= cycle) {
            L3TransferPacket p = packet;
            injection_queue_.pop();
            inject_packet(p, cycle);
        } else {
            break;
        }
    }

    // Process packets at each tile
    for (size_t id = 0; id < config_.num_tiles(); ++id) {
        // Check each input direction
        for (size_t dir = 0; dir < 4; ++dir) {
            auto& buffer = input_buffers_[id][dir];
            if (buffer.empty()) continue;

            L3TransferPacket& packet = buffer.front();
            if (packet.arrival_cycle > cycle) continue;  // Not arrived yet

            // Packet has arrived at this tile
            if (packet.dst_l3_id == id) {
                // Final destination
                L3TransferPacket delivered = packet;
                buffer.pop();
                deliver_packet(delivered, cycle);
            } else if (config_.allow_multi_hop) {
                // Need to forward to next hop
                auto next_dir = route_next_hop(static_cast<uint8_t>(id), packet.dst_l3_id);
                if (next_dir) {
                    InterconnectLink& link = links_[id][static_cast<size_t>(*next_dir)];
                    if (link.is_available(cycle)) {
                        // Forward packet
                        L3TransferPacket forwarded = packet;
                        buffer.pop();

                        uint32_t transfer_cycles = link.transfer_cycles(forwarded.size_bytes());
                        link.busy = true;
                        link.busy_until_cycle = cycle + transfer_cycles;
                        link.bytes_transferred += forwarded.size_bytes();
                        link.packets_transferred++;

                        uint8_t next_id = static_cast<uint8_t>(config_.get_neighbor(static_cast<uint8_t>(id), *next_dir));
                        Direction ingress = opposite(*next_dir);
                        forwarded.arrival_cycle = cycle + transfer_cycles;
                        forwarded.hops_remaining--;

                        input_buffers_[next_id][static_cast<size_t>(ingress)].push(forwarded);
                        stats_.total_hops++;
                    }
                }
            }
        }
    }

    // Update link busy states
    for (auto& tile_links : links_) {
        for (auto& link : tile_links) {
            if (link.busy && cycle >= link.busy_until_cycle) {
                link.busy = false;
            }
        }
    }
}

bool L3Interconnect::is_idle() const {
    // Check injection queue
    if (!injection_queue_.empty()) return false;

    // Check all input buffers
    for (const auto& tile_buffers : input_buffers_) {
        for (const auto& buffer : tile_buffers) {
            if (!buffer.empty()) return false;
        }
    }

    // Check all links
    for (const auto& tile_links : links_) {
        for (const auto& link : tile_links) {
            if (link.busy) return false;
        }
    }

    return true;
}

size_t L3Interconnect::packets_in_flight() const {
    size_t count = injection_queue_.size();
    for (const auto& tile_buffers : input_buffers_) {
        for (const auto& buffer : tile_buffers) {
            count += buffer.size();
        }
    }
    return count;
}

std::optional<Direction> L3Interconnect::route_next_hop(uint8_t current_id, uint8_t dest_id) const {
    if (current_id == dest_id) {
        return Direction::LOCAL;
    }

    L3Position current = config_.get_position(current_id);
    L3Position dest = config_.get_position(dest_id);

    // XY routing: first move in X (column), then Y (row)
    if (current.col < dest.col && config_.get_neighbor(current_id, Direction::EAST) >= 0) {
        return Direction::EAST;
    }
    if (current.col > dest.col && config_.get_neighbor(current_id, Direction::WEST) >= 0) {
        return Direction::WEST;
    }
    if (current.row < dest.row && config_.get_neighbor(current_id, Direction::SOUTH) >= 0) {
        return Direction::SOUTH;
    }
    if (current.row > dest.row && config_.get_neighbor(current_id, Direction::NORTH) >= 0) {
        return Direction::NORTH;
    }

    // For torus, try wraparound
    if (config_.topology == InterconnectTopology::TORUS) {
        // Check if wraparound is shorter
        int east_dist = (dest.col >= current.col) ?
                        (dest.col - current.col) :
                        (config_.cols - current.col + dest.col);
        int west_dist = config_.cols - east_dist;

        if (west_dist < east_dist && config_.get_neighbor(current_id, Direction::WEST) >= 0) {
            return Direction::WEST;
        }
        if (east_dist <= west_dist && config_.get_neighbor(current_id, Direction::EAST) >= 0) {
            return Direction::EAST;
        }
    }

    return std::nullopt;  // No route found
}

uint32_t L3Interconnect::estimate_transfer_cycles(uint8_t src_id, uint8_t dst_id, size_t bytes) const {
    uint8_t hops = config_.manhattan_distance(src_id, dst_id);
    if (hops == 0) return 0;  // Same tile

    uint32_t per_hop = config_.link_latency_cycles +
                       (bytes + config_.link_bandwidth_bytes_per_cycle - 1) /
                       config_.link_bandwidth_bytes_per_cycle;

    // Pipelined: first hop takes full time, subsequent hops overlap
    return per_hop + (hops - 1) * config_.link_latency_cycles;
}

void L3Interconnect::deliver_packet(L3TransferPacket& packet, uint64_t cycle) {
    packet.arrival_cycle = cycle;

    uint64_t latency = cycle - packet.inject_cycle;
    stats_.total_latency_cycles += latency;
    stats_.max_latency_cycles = std::max(stats_.max_latency_cycles, latency);

    if (delivery_callbacks_[packet.dst_l3_id]) {
        delivery_callbacks_[packet.dst_l3_id](packet);
    }
}

void L3Interconnect::reset_stats() {
    stats_ = Stats{};
    for (auto& tile_links : links_) {
        for (auto& link : tile_links) {
            link.bytes_transferred = 0;
            link.packets_transferred = 0;
        }
    }
}

std::string L3Interconnect::to_string() const {
    std::ostringstream oss;
    oss << "L3Interconnect[" << static_cast<int>(config_.rows) << "x"
        << static_cast<int>(config_.cols) << " ";

    switch (config_.topology) {
        case InterconnectTopology::MESH_2D: oss << "MESH"; break;
        case InterconnectTopology::TORUS: oss << "TORUS"; break;
        case InterconnectTopology::RING: oss << "RING"; break;
        case InterconnectTopology::CROSSBAR: oss << "XBAR"; break;
    }

    oss << "] packets_in_flight=" << packets_in_flight();
    return oss.str();
}

void L3Interconnect::print_topology() const {
    std::cout << "\nL3 Interconnect Topology (" << static_cast<int>(config_.rows)
              << "x" << static_cast<int>(config_.cols) << "):\n";

    for (uint8_t row = 0; row < config_.rows; ++row) {
        // Print horizontal connections
        std::cout << "  ";
        for (uint8_t col = 0; col < config_.cols; ++col) {
            uint8_t id = row * config_.cols + col;
            std::cout << std::setw(2) << static_cast<int>(id);
            if (col < config_.cols - 1) {
                std::cout << " -- ";
            }
        }
        std::cout << "\n";

        // Print vertical connections
        if (row < config_.rows - 1) {
            std::cout << "  ";
            for (uint8_t col = 0; col < config_.cols; ++col) {
                std::cout << " |  ";
                if (col < config_.cols - 1) {
                    std::cout << "   ";
                }
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

// ============================================================================
// TriggerNetwork Implementation
// ============================================================================

TriggerNetwork::TriggerNetwork(size_t num_tiles)
    : num_tiles_(num_tiles)
    , pending_triggers_(num_tiles, 0)
{
}

void TriggerNetwork::send_trigger(uint8_t from_l3, uint8_t channel, uint16_t dest_mask) {
    (void)from_l3;  // Source not needed for delivery

    for (size_t id = 0; id < num_tiles_; ++id) {
        if (dest_mask & (1 << id)) {
            // Add to in-flight (1-cycle latency)
            // For simplicity, deliver immediately for now
            pending_triggers_[id] |= (1 << channel);
        }
    }
}

bool TriggerNetwork::is_trigger_pending(uint8_t l3_id, uint8_t channel) const {
    if (l3_id >= num_tiles_ || channel >= 16) return false;
    return (pending_triggers_[l3_id] & (1 << channel)) != 0;
}

bool TriggerNetwork::any_trigger_pending(uint8_t l3_id, uint16_t channel_mask) const {
    if (l3_id >= num_tiles_) return false;
    return (pending_triggers_[l3_id] & channel_mask) != 0;
}

void TriggerNetwork::clear_trigger(uint8_t l3_id, uint8_t channel) {
    if (l3_id < num_tiles_ && channel < 16) {
        pending_triggers_[l3_id] &= ~(1 << channel);
    }
}

void TriggerNetwork::clear_triggers(uint8_t l3_id, uint16_t channel_mask) {
    if (l3_id < num_tiles_) {
        pending_triggers_[l3_id] &= ~channel_mask;
    }
}

void TriggerNetwork::step(uint64_t cycle) {
    (void)cycle;
    // Process in-flight triggers (currently immediate delivery)
    // Future: add latency modeling
}

void TriggerNetwork::reset() {
    for (auto& triggers : pending_triggers_) {
        triggers = 0;
    }
    while (!in_flight_.empty()) {
        in_flight_.pop();
    }
}

} // namespace sw::kpu
