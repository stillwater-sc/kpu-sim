// ============================================================================
// src/noc/noc.cpp
// Network-on-Chip Implementation
// ============================================================================

#include "sw/kpu/noc/noc.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace sw::kpu::noc {

// ============================================================================
// NoCPacket
// ============================================================================

std::string NoCPacket::to_string() const {
    std::ostringstream oss;

    const char* tensor_names[] = {"A", "B", "C", "Bias", "Workspace", "Custom"};
    uint8_t tidx = static_cast<uint8_t>(tile.tensor);
    const char* tname = tidx < 6 ? tensor_names[tidx] : "?";

    oss << "Pkt[" << sequence_num << "] "
        << tname << "[";

    switch (tile.tensor) {
        case TensorId::A:
            oss << tile.m_tile << "," << tile.k_tile;
            break;
        case TensorId::B:
            oss << tile.k_tile << "," << tile.n_tile;
            break;
        case TensorId::C:
            oss << tile.m_tile << "," << tile.n_tile;
            break;
        default:
            oss << tile.m_tile << "," << tile.n_tile << "," << tile.k_tile;
    }

    oss << "] R" << static_cast<int>(src_router)
        << "→R" << static_cast<int>(dst_router)
        << " " << size_bytes << "B"
        << " hops=" << static_cast<int>(hops_taken);

    return oss.str();
}

// ============================================================================
// NoCLink
// ============================================================================

NoCLink::NoCLink(uint8_t src_router, uint8_t dst_router,
                 PortDirection direction, uint32_t bandwidth, uint32_t latency)
    : src_router_(src_router)
    , dst_router_(dst_router)
    , direction_(direction)
    , bandwidth_bytes_per_cycle_(bandwidth)
    , latency_cycles_(latency)
{
}

bool NoCLink::is_available(uint64_t cycle) const {
    return !busy_ || cycle >= busy_until_cycle_;
}

bool NoCLink::send_flit(uint64_t cycle, uint32_t flit_bytes) {
    if (!is_available(cycle)) {
        stats_.contention_events++;
        return false;
    }

    // Calculate how long link will be busy
    uint32_t transfer_cycles = (flit_bytes + bandwidth_bytes_per_cycle_ - 1) /
                               bandwidth_bytes_per_cycle_;
    busy_ = true;
    busy_until_cycle_ = cycle + latency_cycles_ + transfer_cycles;

    stats_.flits_transferred++;
    stats_.bytes_transferred += flit_bytes;

    return true;
}

void NoCLink::update_stats(uint64_t cycle) {
    if (cycle > last_update_cycle_) {
        uint64_t elapsed = cycle - last_update_cycle_;
        if (busy_ && cycle < busy_until_cycle_) {
            stats_.busy_cycles += elapsed;
        } else {
            stats_.idle_cycles += elapsed;
            busy_ = false;
        }
        last_update_cycle_ = cycle;
    }
}

std::string NoCLink::to_string() const {
    std::ostringstream oss;
    oss << "Link R" << static_cast<int>(src_router_)
        << "->" << ::sw::kpu::noc::to_string(direction_)
        << "->R" << static_cast<int>(dst_router_);
    if (busy_) {
        oss << " BUSY until " << busy_until_cycle_;
    } else {
        oss << " IDLE";
    }
    return oss.str();
}

// ============================================================================
// NoCRouter
// ============================================================================

NoCRouter::NoCRouter(uint8_t id, const NoCConfig& config)
    : id_(id)
    , row_(id / config.cols)
    , col_(id % config.cols)
    , config_(config)
{
    // Initialize ports
    for (size_t i = 0; i < input_ports_.size(); i++) {
        input_ports_[i].direction = static_cast<PortDirection>(i);
        input_ports_[i].buffer_capacity = config.input_buffer_flits;
        input_ports_[i].credits_available = config.input_buffer_flits;
    }

    for (size_t i = 0; i < output_ports_.size(); i++) {
        output_ports_[i].direction = static_cast<PortDirection>(i);
        output_ports_[i].buffer_capacity = config.output_buffer_flits;
    }
}

void NoCRouter::connect_output(PortDirection dir, NoCLink* link) {
    output_ports_[static_cast<size_t>(dir)].link = link;
    // Initialize credits based on downstream buffer capacity
    output_ports_[static_cast<size_t>(dir)].credits_available =
        config_.input_buffer_flits;
}

bool NoCRouter::can_inject(PortDirection port, uint64_t cycle) const {
    const auto& inp = input_ports_[static_cast<size_t>(port)];
    return inp.can_accept();
}

bool NoCRouter::inject_packet(NoCPacket packet, PortDirection port, uint64_t cycle) {
    auto& inp = input_ports_[static_cast<size_t>(port)];

    if (!inp.can_accept()) {
        inp.buffer_full_events++;
        return false;
    }

    packet.current_router = id_;
    packet.inject_cycle = cycle;

    inp.buffer.push(std::move(packet));
    inp.packets_received++;

    stats_.packets_injected++;

    return true;
}

bool NoCRouter::has_packet_for_local() const {
    // Check output port for LOCAL direction
    const auto& out = output_ports_[static_cast<size_t>(PortDirection::LOCAL)];
    return !out.buffer.empty();
}

std::optional<NoCPacket> NoCRouter::eject_packet() {
    auto& out = output_ports_[static_cast<size_t>(PortDirection::LOCAL)];

    if (out.buffer.empty()) {
        return std::nullopt;
    }

    NoCPacket pkt = std::move(out.buffer.front());
    out.buffer.pop();
    stats_.packets_ejected++;

    return pkt;
}

void NoCRouter::step(uint64_t cycle) {
    process_inputs(cycle);
    arbitrate_and_forward(cycle);
    stats_.cycles_active++;
}

PortDirection NoCRouter::route(uint8_t dst_router) const {
    if (dst_router == id_) {
        return PortDirection::LOCAL;
    }

    switch (config_.routing) {
        case RoutingAlgorithm::XY:
        default:
            return xy_route(dst_router);
    }
}

PortDirection NoCRouter::xy_route(uint8_t dst_router) const {
    auto [dst_row, dst_col] = config_.get_router_pos(dst_router);

    // X first (East/West), then Y (North/South)
    if (dst_col > col_) {
        return PortDirection::EAST;
    } else if (dst_col < col_) {
        return PortDirection::WEST;
    } else if (dst_row > row_) {
        return PortDirection::SOUTH;
    } else if (dst_row < row_) {
        return PortDirection::NORTH;
    }

    return PortDirection::LOCAL;
}

void NoCRouter::process_inputs(uint64_t cycle) {
    route_requests_.clear();

    // Collect packets from all input ports that need routing
    for (size_t i = 0; i < input_ports_.size(); i++) {
        auto& inp = input_ports_[i];

        if (!inp.buffer.empty()) {
            NoCPacket* pkt = &inp.buffer.front();

            // Determine output port
            PortDirection out_dir = route(pkt->dst_router);

            route_requests_.push_back({static_cast<PortDirection>(i), pkt});
        }
    }
}

void NoCRouter::arbitrate_and_forward(uint64_t cycle) {
    // Simple round-robin arbitration per output port
    // For each output port, pick one input that wants it

    std::array<bool, 6> output_used = {false};

    for (auto& [in_dir, pkt] : route_requests_) {
        PortDirection out_dir = route(pkt->dst_router);
        size_t out_idx = static_cast<size_t>(out_dir);

        if (output_used[out_idx]) {
            stats_.routing_conflicts++;
            continue;
        }

        auto& out = output_ports_[out_idx];

        // For LOCAL, just move to output buffer
        if (out_dir == PortDirection::LOCAL) {
            if (out.buffer.size() < out.buffer_capacity) {
                auto& inp = input_ports_[static_cast<size_t>(in_dir)];

                NoCPacket moved_pkt = std::move(inp.buffer.front());
                inp.buffer.pop();

                moved_pkt.hops_taken++;
                out.buffer.push(std::move(moved_pkt));

                output_used[out_idx] = true;
                out.packets_sent++;
                stats_.packets_routed++;
            }
            continue;
        }

        // For network ports, check if link is available
        if (out.link && out.link->is_available(cycle)) {
            if (out.buffer.size() < out.buffer_capacity) {
                auto& inp = input_ports_[static_cast<size_t>(in_dir)];

                NoCPacket moved_pkt = std::move(inp.buffer.front());
                inp.buffer.pop();

                // Update packet state
                moved_pkt.current_router = config_.get_neighbor_id(id_, out_dir);
                moved_pkt.hops_taken++;

                // Send through link
                out.link->send_flit(cycle, moved_pkt.size_bytes);
                out.link->set_current_packet(&moved_pkt);

                // Move to output buffer (will be picked up by next router)
                out.buffer.push(std::move(moved_pkt));

                output_used[out_idx] = true;
                out.packets_sent++;
                stats_.packets_routed++;
            }
        }
    }
}

std::string NoCRouter::to_string() const {
    std::ostringstream oss;
    oss << "Router[" << static_cast<int>(id_) << "] ("
        << static_cast<int>(row_) << "," << static_cast<int>(col_) << ")";

    // Show buffer occupancy
    for (size_t i = 0; i < 5; i++) {  // N, S, E, W, Local
        const auto& inp = input_ports_[i];
        if (!inp.buffer.empty()) {
            oss << " " << ::sw::kpu::noc::to_string(static_cast<PortDirection>(i))
                << ":" << inp.buffer.size();
        }
    }

    return oss.str();
}

// ============================================================================
// NoC
// ============================================================================

NoC::NoC(const NoCConfig& config)
    : config_(config)
{
    initialize_routers();
    initialize_links();
    connect_routers();

    l3_callbacks_.resize(config_.num_routers());
}

void NoC::initialize_routers() {
    routers_.reserve(config_.num_routers());
    for (uint8_t i = 0; i < config_.num_routers(); i++) {
        routers_.push_back(std::make_unique<NoCRouter>(i, config_));
    }
}

void NoC::initialize_links() {
    links_.resize(config_.num_routers());

    for (uint8_t i = 0; i < config_.num_routers(); i++) {
        for (size_t d = 0; d < 4; d++) {  // N, S, E, W
            PortDirection dir = static_cast<PortDirection>(d);

            if (config_.has_neighbor(i, dir)) {
                int neighbor = config_.get_neighbor_id(i, dir);
                links_[i][d] = std::make_unique<NoCLink>(
                    i, static_cast<uint8_t>(neighbor),
                    dir,
                    config_.link_bandwidth,
                    config_.link_latency);
            }
        }
    }
}

void NoC::connect_routers() {
    for (uint8_t i = 0; i < config_.num_routers(); i++) {
        for (size_t d = 0; d < 4; d++) {
            PortDirection dir = static_cast<PortDirection>(d);

            if (links_[i][d]) {
                routers_[i]->connect_output(dir, links_[i][d].get());
            }
        }
    }
}

bool NoC::can_inject(uint8_t src_router, PortDirection port, uint64_t cycle) const {
    if (src_router >= routers_.size()) return false;
    return routers_[src_router]->can_inject(port, cycle);
}

NoCPacket NoC::create_packet(uint8_t src, uint8_t dst,
                             PortDirection src_port, PortDirection dst_port,
                             const TileDescriptor& tile, uint64_t cycle) {
    NoCPacket pkt;
    pkt.src_router = src;
    pkt.dst_router = dst;
    pkt.src_port = src_port;
    pkt.dst_port = dst_port;
    pkt.tile = tile;
    pkt.size_bytes = tile.size;
    pkt.num_flits = (tile.size + config_.flit_size_bytes - 1) / config_.flit_size_bytes;
    pkt.sequence_num = next_sequence_++;
    pkt.inject_cycle = cycle;
    pkt.current_router = src;

    return pkt;
}

TransferResult NoC::inject_l3_to_l3(
    uint8_t src_l3, uint8_t dst_l3,
    const TileDescriptor& tile,
    uint64_t cycle)
{
    if (src_l3 >= routers_.size() || dst_l3 >= routers_.size()) {
        return TransferResult::ERROR;
    }

    if (!can_inject(src_l3, PortDirection::LOCAL, cycle)) {
        stats_.injection_blocked_cycles++;

        // Record blocked event
        if (tracer_) {
            NoCTraceEvent event;
            event.cycle = cycle;
            event.type = NoCEventType::BLOCKED;
            event.router_id = src_l3;
            event.port = PortDirection::LOCAL;
            event.packet_seq = next_sequence_;  // Would-be packet
            event.tile = tile;
            tracer_->record_event(event);
        }

        return TransferResult::BUFFER_FULL;
    }

    NoCPacket pkt = create_packet(src_l3, dst_l3,
                                  PortDirection::LOCAL, PortDirection::LOCAL,
                                  tile, cycle);

    if (!routers_[src_l3]->inject_packet(std::move(pkt), PortDirection::LOCAL, cycle)) {
        return TransferResult::BUFFER_FULL;
    }

    // Record injection event
    if (tracer_) {
        uint32_t num_flits = (tile.size + config_.flit_size_bytes - 1) / config_.flit_size_bytes;

        NoCTraceEvent event;
        event.cycle = cycle;
        event.type = NoCEventType::INJECT;
        event.router_id = src_l3;
        event.port = PortDirection::LOCAL;
        event.packet_seq = next_sequence_ - 1;  // We already incremented in create_packet
        event.tile = tile;
        event.flit_index = 0;  // First FLIT
        event.num_flits = num_flits;
        event.src_router = src_l3;
        event.dst_router = dst_l3;
        tracer_->record_event(event);
    }

    stats_.total_packets++;
    stats_.total_bytes += tile.size;

    return TransferResult::SUCCESS;
}

TransferResult NoC::inject_dma_to_l3(
    uint8_t dst_l3,
    const TileDescriptor& tile,
    uint64_t cycle)
{
    if (dst_l3 >= routers_.size()) {
        return TransferResult::ERROR;
    }

    // DMA injects at edge routers
    // For simplicity, DMA can inject directly at any router via LOCAL port
    // In a real implementation, DMA would connect to edge routers

    if (!can_inject(dst_l3, PortDirection::LOCAL, cycle)) {
        return TransferResult::BUFFER_FULL;
    }

    // DMA load goes directly to destination (simplified model)
    NoCPacket pkt = create_packet(dst_l3, dst_l3,
                                  PortDirection::DMA, PortDirection::LOCAL,
                                  tile, cycle);
    pkt.traffic_class = TrafficClass::DMA;

    if (!routers_[dst_l3]->inject_packet(std::move(pkt), PortDirection::LOCAL, cycle)) {
        return TransferResult::BUFFER_FULL;
    }

    stats_.total_packets++;
    stats_.total_bytes += tile.size;

    return TransferResult::SUCCESS;
}

TransferResult NoC::inject_l3_to_dma(
    uint8_t src_l3,
    const TileDescriptor& tile,
    uint64_t cycle)
{
    // Similar to DMA to L3, simplified model
    // In reality, would need to route to edge router with DMA port

    if (src_l3 >= routers_.size()) {
        return TransferResult::ERROR;
    }

    // For now, just track it as a transfer that completes locally
    stats_.total_packets++;
    stats_.total_bytes += tile.size;

    return TransferResult::SUCCESS;
}

void NoC::step(uint64_t cycle) {
    // Update all links
    for (auto& router_links : links_) {
        for (auto& link : router_links) {
            if (link) {
                link->update_stats(cycle);
            }
        }
    }

    // Step all routers
    for (auto& router : routers_) {
        router->step(cycle);
    }

    // Deliver packets that have reached their destination
    deliver_packets(cycle);

    // Move packets between routers (through links)
    // In store-and-forward, output buffers of one router become
    // input buffers of the next router
    for (uint8_t i = 0; i < config_.num_routers(); i++) {
        for (size_t d = 0; d < 4; d++) {
            PortDirection dir = static_cast<PortDirection>(d);

            auto& out = routers_[i]->output_port(dir);
            if (!out.buffer.empty() && out.link) {
                // Check if link transfer is complete
                if (out.link->is_available(cycle)) {
                    // Move packet to next router's input buffer
                    int neighbor = config_.get_neighbor_id(i, dir);
                    if (neighbor >= 0) {
                        PortDirection in_dir = opposite(dir);
                        auto& next_inp = routers_[neighbor]->input_port(in_dir);

                        if (next_inp.can_accept()) {
                            NoCPacket pkt = std::move(out.buffer.front());
                            out.buffer.pop();
                            out.link->clear_current_packet();

                            // Record hop event
                            if (tracer_) {
                                NoCTraceEvent event;
                                event.cycle = cycle;
                                event.type = NoCEventType::HOP;
                                event.router_id = static_cast<uint8_t>(neighbor);
                                event.port = in_dir;
                                event.packet_seq = pkt.sequence_num;
                                event.tile = pkt.tile;
                                event.flit_index = pkt.num_flits - 1;
                                event.num_flits = pkt.num_flits;
                                event.src_router = i;
                                event.dst_router = static_cast<uint8_t>(neighbor);
                                tracer_->record_event(event);

                                // Emit sampled FLIT_SEND events for link occupancy visualization
                                // FLITs are sent over num_flits cycles
                                uint64_t send_start_cycle = cycle > pkt.num_flits ?
                                                            cycle - pkt.num_flits : 0;
                                const uint16_t sample_interval = std::max<uint16_t>(1, pkt.num_flits / 8);
                                for (uint16_t flit = 0; flit < pkt.num_flits; flit += sample_interval) {
                                    NoCTraceEvent flit_event;
                                    flit_event.cycle = send_start_cycle + flit;
                                    flit_event.type = NoCEventType::FLIT_SEND;
                                    flit_event.router_id = i;
                                    flit_event.port = dir;
                                    flit_event.packet_seq = pkt.sequence_num;
                                    flit_event.tile = pkt.tile;
                                    flit_event.flit_index = flit;
                                    flit_event.num_flits = pkt.num_flits;
                                    flit_event.src_router = i;
                                    flit_event.dst_router = static_cast<uint8_t>(neighbor);
                                    tracer_->record_event(flit_event);
                                }
                            }

                            next_inp.buffer.push(std::move(pkt));
                            next_inp.packets_received++;
                        }
                    }
                }
            }
        }
    }
}

void NoC::deliver_packets(uint64_t cycle) {
    for (uint8_t i = 0; i < config_.num_routers(); i++) {
        while (routers_[i]->has_packet_for_local()) {
            auto pkt_opt = routers_[i]->eject_packet();
            if (pkt_opt) {
                NoCPacket& pkt = *pkt_opt;
                pkt.tail_arrive_cycle = cycle;

                // Record eject event
                if (tracer_) {
                    NoCTraceEvent event;
                    event.cycle = cycle;
                    event.type = NoCEventType::EJECT;
                    event.router_id = i;
                    event.port = PortDirection::LOCAL;
                    event.packet_seq = pkt.sequence_num;
                    event.tile = pkt.tile;
                    event.flit_index = pkt.num_flits - 1;  // Last FLIT
                    event.num_flits = pkt.num_flits;
                    event.src_router = pkt.src_router;
                    event.dst_router = i;
                    tracer_->record_event(event);

                    // Emit FLIT_ARRIVE events for progressive fill visualization
                    // FLITs arrive over num_flits cycles (at 1 FLIT/cycle bandwidth)
                    // First FLIT arrived at (cycle - num_flits + 1)
                    uint64_t first_flit_cycle = cycle > pkt.num_flits ?
                                                cycle - pkt.num_flits + 1 : pkt.inject_cycle;

                    // Emit sampled FLIT arrivals (every 16th FLIT to avoid trace bloat)
                    // This gives ~16 progressive fill updates per tile
                    const uint16_t sample_interval = std::max<uint16_t>(1, pkt.num_flits / 16);
                    for (uint16_t flit = 0; flit < pkt.num_flits; flit += sample_interval) {
                        NoCTraceEvent flit_event;
                        flit_event.cycle = first_flit_cycle + flit;
                        flit_event.type = NoCEventType::FLIT_ARRIVE;
                        flit_event.router_id = i;
                        flit_event.port = PortDirection::LOCAL;
                        flit_event.packet_seq = pkt.sequence_num;
                        flit_event.tile = pkt.tile;
                        flit_event.flit_index = flit;
                        flit_event.num_flits = pkt.num_flits;
                        flit_event.src_router = pkt.src_router;
                        flit_event.dst_router = i;
                        tracer_->record_event(flit_event);
                    }
                }

                // Update statistics
                update_stats(pkt);

                // Deliver via callback
                if (pkt.dst_port == PortDirection::LOCAL && i < l3_callbacks_.size()) {
                    if (l3_callbacks_[i]) {
                        l3_callbacks_[i](pkt, cycle);
                    }
                } else if (pkt.dst_port == PortDirection::DMA && dma_callback_) {
                    dma_callback_(pkt, cycle);
                }
            }
        }
    }
}

void NoC::update_stats(const NoCPacket& packet) {
    uint64_t latency = packet.tail_arrive_cycle - packet.inject_cycle;

    stats_.total_flits += packet.num_flits;
    stats_.total_hops += packet.hops_taken;
    stats_.total_latency_cycles += latency;

    if (latency > stats_.max_latency_cycles) {
        stats_.max_latency_cycles = latency;
    }
}

void NoC::set_l3_delivery_callback(uint8_t l3_id, PacketDeliveryCallback callback) {
    if (l3_id < l3_callbacks_.size()) {
        l3_callbacks_[l3_id] = std::move(callback);
    }
}

void NoC::set_dma_delivery_callback(PacketDeliveryCallback callback) {
    dma_callback_ = std::move(callback);
}

bool NoC::is_idle() const {
    for (const auto& router : routers_) {
        for (size_t i = 0; i < 6; i++) {
            if (!router->input_port(static_cast<PortDirection>(i)).buffer.empty()) {
                return false;
            }
            if (!router->output_port(static_cast<PortDirection>(i)).buffer.empty()) {
                return false;
            }
        }
    }
    return true;
}

size_t NoC::packets_in_flight() const {
    size_t count = 0;
    for (const auto& router : routers_) {
        for (size_t i = 0; i < 6; i++) {
            count += router->input_port(static_cast<PortDirection>(i)).buffer.size();
            count += router->output_port(static_cast<PortDirection>(i)).buffer.size();
        }
    }
    return count;
}

uint64_t NoC::estimate_latency(uint8_t src, uint8_t dst, uint32_t size_bytes) const {
    auto [src_row, src_col] = config_.get_router_pos(src);
    auto [dst_row, dst_col] = config_.get_router_pos(dst);

    uint8_t hops = std::abs(static_cast<int>(src_row) - dst_row) +
                   std::abs(static_cast<int>(src_col) - dst_col);

    uint32_t transfer_cycles = (size_bytes + config_.link_bandwidth - 1) /
                               config_.link_bandwidth;

    // Total: injection + per-hop (router + link) + transfer time
    return 1 + hops * (config_.router_latency + config_.link_latency) + transfer_cycles;
}

const NoCLink* NoC::get_link(uint8_t src, PortDirection dir) const {
    if (src >= links_.size()) return nullptr;
    size_t idx = static_cast<size_t>(dir);
    if (idx >= 4) return nullptr;
    return links_[src][idx].get();
}

void NoC::reset_stats() {
    stats_ = Stats{};
    for (auto& router : routers_) {
        // Could add router stat reset if needed
    }
    for (auto& router_links : links_) {
        for (auto& link : router_links) {
            if (link) {
                // Could add link stat reset if needed
            }
        }
    }
}

std::string NoC::to_string() const {
    std::ostringstream oss;
    oss << "NoC " << static_cast<int>(config_.rows) << "x"
        << static_cast<int>(config_.cols) << " mesh\n";
    oss << "  Routers: " << config_.num_routers() << "\n";
    oss << "  Packets in flight: " << packets_in_flight() << "\n";
    oss << "  Total packets: " << stats_.total_packets << "\n";
    oss << "  Avg latency: " << std::fixed << std::setprecision(1)
        << stats_.avg_latency() << " cycles\n";
    return oss.str();
}

void NoC::print_topology() const {
    std::cout << "\nNoC Topology (" << static_cast<int>(config_.rows)
              << "x" << static_cast<int>(config_.cols) << " mesh):\n\n";

    for (uint8_t row = 0; row < config_.rows; row++) {
        // Print routers in this row
        std::cout << "  ";
        for (uint8_t col = 0; col < config_.cols; col++) {
            uint8_t id = config_.get_router_id(row, col);
            std::cout << "[R" << std::setw(2) << static_cast<int>(id) << "]";
            if (col < config_.cols - 1) {
                std::cout << "---";
            }
        }
        std::cout << "\n";

        // Print vertical links
        if (row < config_.rows - 1) {
            std::cout << "  ";
            for (uint8_t col = 0; col < config_.cols; col++) {
                std::cout << "  |  ";
                if (col < config_.cols - 1) {
                    std::cout << "   ";
                }
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

std::string NoC::dump_state() const {
    std::ostringstream oss;

    oss << "=== NoC State Dump ===\n\n";

    for (const auto& router : routers_) {
        oss << router->to_string() << "\n";
    }

    oss << "\nLink States:\n";
    for (uint8_t i = 0; i < config_.num_routers(); i++) {
        for (size_t d = 0; d < 4; d++) {
            if (links_[i][d]) {
                oss << "  " << links_[i][d]->to_string() << "\n";
            }
        }
    }

    return oss.str();
}

// ============================================================================
// NoCTraceEvent
// ============================================================================

std::string NoCTraceEvent::to_string() const {
    std::ostringstream oss;

    const char* type_names[] = {"INJECT", "HOP", "EJECT", "BLOCKED", "LINK_BUSY", "LINK_IDLE"};
    oss << cycle << "," << type_names[static_cast<int>(type)]
        << ",R" << static_cast<int>(router_id)
        << "," << ::sw::kpu::noc::to_string(port)
        << ",Pkt" << packet_seq;

    return oss.str();
}

// ============================================================================
// NoCTracer
// ============================================================================

void NoCTracer::record_event(const NoCTraceEvent& event) {
    events_.push_back(event);
}

void NoCTracer::clear() {
    events_.clear();
}

bool NoCTracer::export_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) return false;

    file << "cycle,type,router_id,port,packet_seq,tensor,m_tile,n_tile,k_tile,"
         << "flit_index,num_flits,src_router,dst_router\n";

    for (const auto& e : events_) {
        file << e.cycle << ","
             << static_cast<int>(e.type) << ","
             << static_cast<int>(e.router_id) << ","
             << ::sw::kpu::noc::to_string(e.port) << ","
             << e.packet_seq << ","
             << static_cast<int>(e.tile.tensor) << ","
             << e.tile.m_tile << ","
             << e.tile.n_tile << ","
             << e.tile.k_tile << ","
             << e.flit_index << ","
             << e.num_flits << ","
             << static_cast<int>(e.src_router) << ","
             << static_cast<int>(e.dst_router) << "\n";
    }

    return true;
}

bool NoCTracer::export_chrome_trace(const std::string& filename,
                                    const NoCConfig& config) const {
    std::ofstream file(filename);
    if (!file) return false;

    // Similar to TileFlowTracer but for NoC events
    file << "{\"traceEvents\":[\n";

    // Process/thread metadata for routers and links
    // ... (similar structure to tile_flow_tracer)

    file << "]}\n";

    return true;
}

} // namespace sw::kpu::noc
