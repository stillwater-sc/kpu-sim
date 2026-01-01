// ============================================================================
// src/noc/wormhole_router.cpp
// Wormhole Router Implementation
// ============================================================================

#include "sw/kpu/noc/wormhole_router.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace sw::kpu::noc {

// ============================================================================
// FlitBuffer
// ============================================================================

FlitBuffer::FlitBuffer(size_t capacity)
    : capacity_(capacity)
    , buffer_(capacity)
{}

bool FlitBuffer::push(const Flit& flit) {
    if (count_ >= capacity_) {
        return false;
    }
    buffer_[tail_] = flit;
    tail_ = (tail_ + 1) % capacity_;
    count_++;
    return true;
}

std::optional<Flit> FlitBuffer::pop() {
    if (count_ == 0) {
        return std::nullopt;
    }
    Flit flit = buffer_[head_];
    head_ = (head_ + 1) % capacity_;
    count_--;
    return flit;
}

const Flit* FlitBuffer::peek() const {
    if (count_ == 0) {
        return nullptr;
    }
    return &buffer_[head_];
}

void FlitBuffer::clear() {
    head_ = 0;
    tail_ = 0;
    count_ = 0;
}

// ============================================================================
// InjectionState
// ============================================================================

Flit InjectionState::generate_next_flit(uint8_t src_router) {
    Flit flit;
    flit.packet_seq = packet_seq;
    flit.src_router = src_router;
    flit.dst_router = dst_router;
    flit.inject_cycle = start_cycle;

    if (total_flits == 1) {
        // Single-flit packet
        flit.type = FlitType::HEAD_TAIL;
        flit.total_flits = 1;
        flit.tile = tile;
    } else if (flits_injected == 0) {
        // First flit
        flit.type = FlitType::HEAD;
        flit.total_flits = total_flits;
        flit.tile = tile;
    } else if (flits_injected == total_flits - 1) {
        // Last flit
        flit.type = FlitType::TAIL;
        flit.total_flits = total_flits;
    } else {
        // Middle flit
        flit.type = FlitType::BODY;
        flit.total_flits = total_flits;
    }

    flits_injected++;
    return flit;
}

// ============================================================================
// InputPort
// ============================================================================

InputPort::InputPort(PortDir dir)
    : direction_(dir)
{}

bool InputPort::receive_flit(const Flit& flit, uint64_t cycle) {
    if (!buffer_.can_accept()) {
        stats_.buffer_full_cycles++;
        return false;  // Buffer full - caller should not return credit
    }

    buffer_.push(flit);
    stats_.flits_received++;

    if (flit.is_head()) {
        stats_.packets_received++;
        active_packet_ = flit.packet_seq;
    }

    if (flit.is_tail()) {
        active_packet_.reset();
    }

    return true;
}

Flit InputPort::consume_flit() {
    auto flit = buffer_.pop();
    if (flit) {
        credits_to_return_++;
        return *flit;
    }
    return Flit{};
}

// ============================================================================
// OutputPort
// ============================================================================

OutputPort::OutputPort(PortDir dir)
    : direction_(dir)
{}

bool OutputPort::can_accept() const {
    return buffer_.can_accept() && credits_ > 0;
}

void OutputPort::accept_flit(const Flit& flit, uint64_t cycle) {
    buffer_.push(flit);

    // Update path reservation
    if (flit.is_head()) {
        // Path should already be reserved by arbitration
    }
    if (flit.is_tail()) {
        reservation_.flit_passed();
    } else {
        reservation_.flit_passed();
    }
}

void OutputPort::step(uint64_t cycle) {
    pending_flit_.reset();

    if (buffer_.is_empty()) {
        return;
    }

    if (credits_ == 0) {
        stats_.credit_stall_cycles++;
        return;
    }

    // Send one flit
    auto flit = buffer_.pop();
    if (flit) {
        pending_flit_ = *flit;
        credits_--;
        stats_.flits_sent++;

        // Release path on tail
        if (flit->is_tail()) {
            reservation_.release();
        }
    }
}

Flit OutputPort::take_pending_flit() {
    if (pending_flit_) {
        Flit f = *pending_flit_;
        pending_flit_.reset();
        return f;
    }
    return Flit{};
}

// ============================================================================
// WormholeRouter
// ============================================================================

WormholeRouter::WormholeRouter(uint8_t id, uint8_t row, uint8_t col, bool has_dma)
    : id_(id)
    , row_(row)
    , col_(col)
    , has_dma_(has_dma)
{
    // Initialize ports
    for (size_t i = 0; i < 6; i++) {
        inputs_[i] = InputPort(static_cast<PortDir>(i));
        outputs_[i] = OutputPort(static_cast<PortDir>(i));
    }
}

bool WormholeRouter::can_inject() const {
    return local_injection_.is_idle() &&
           inputs_[static_cast<size_t>(PortDir::LOCAL)].can_receive();
}

bool WormholeRouter::start_injection(uint16_t packet_seq,
                                     const TileDescriptor& tile,
                                     uint8_t dst_router,
                                     uint64_t cycle) {
    if (!can_inject()) {
        return false;
    }

    uint16_t num_flits = (tile.size + FLIT_SIZE_BYTES - 1) / FLIT_SIZE_BYTES;
    if (num_flits == 0) num_flits = 1;

    local_injection_.start(packet_seq, tile, dst_router, cycle, num_flits);
    stats_.packets_injected++;

    return true;
}

bool WormholeRouter::has_ejection() const {
    return !ejection_queue_.empty();
}

Flit WormholeRouter::eject_flit() {
    if (ejection_queue_.empty()) {
        return Flit{};
    }
    Flit f = ejection_queue_.front();
    ejection_queue_.pop();
    return f;
}

bool WormholeRouter::can_inject_dma() const {
    if (!has_dma_) return false;
    return dma_injection_.is_idle() &&
           inputs_[static_cast<size_t>(PortDir::DMA)].can_receive();
}

bool WormholeRouter::start_dma_injection(uint16_t packet_seq,
                                         const TileDescriptor& tile,
                                         uint8_t dst_router,
                                         uint64_t cycle) {
    if (!can_inject_dma()) {
        return false;
    }

    uint16_t num_flits = (tile.size + FLIT_SIZE_BYTES - 1) / FLIT_SIZE_BYTES;
    if (num_flits == 0) num_flits = 1;

    dma_injection_.start(packet_seq, tile, dst_router, cycle, num_flits);

    return true;
}

void WormholeRouter::connect_output(PortDir dir, WormholeRouter* neighbor) {
    outputs_[static_cast<size_t>(dir)].set_downstream_router(neighbor);
    outputs_[static_cast<size_t>(dir)].set_initial_credits(INPUT_BUFFER_FLITS);
}

void WormholeRouter::step(uint64_t cycle) {
    // Phase 1: Process injections (L3/DMA → input buffers)
    process_injections(cycle);

    // Phase 2: Arbitrate and switch flits through crossbar
    arbitrate_and_switch(cycle);

    // Phase 3: Output ports send flits to links
    process_outputs(cycle);
}

bool WormholeRouter::is_idle() const {
    // Check injection states
    if (local_injection_.is_active()) return false;
    if (has_dma_ && dma_injection_.is_active()) return false;

    // Check input buffers
    for (size_t i = 0; i < 5; i++) {
        if (!inputs_[i].has_flit()) continue;
        return false;
    }

    // Check output buffers
    for (size_t i = 0; i < 5; i++) {
        if (outputs_[i].has_flit_for_link()) return false;
    }

    // Check ejection queue
    if (!ejection_queue_.empty()) return false;

    // Check path reservations
    for (size_t i = 0; i < 5; i++) {
        if (outputs_[i].reservation().is_reserved()) return false;
    }

    return true;
}

void WormholeRouter::reset_stats() {
    stats_ = Stats{};
}

PortDir WormholeRouter::route(uint8_t dst_router) const {
    // Single-hop nearest-neighbor routing
    // Destination must be adjacent

    if (dst_router == id_) {
        return PortDir::LOCAL;  // Arrived at destination
    }

    // Calculate destination position
    uint8_t dst_row = dst_router / 4;  // Assuming 4 columns
    uint8_t dst_col = dst_router % 4;

    // Determine direction (must be single hop)
    if (dst_col == col_ + 1 && dst_row == row_) {
        return PortDir::EAST;
    } else if (dst_col == col_ - 1 && dst_row == row_) {
        return PortDir::WEST;
    } else if (dst_row == row_ + 1 && dst_col == col_) {
        return PortDir::SOUTH;
    } else if (dst_row == row_ - 1 && dst_col == col_) {
        return PortDir::NORTH;
    }

    // If not a direct neighbor, default to LOCAL (error case)
    return PortDir::LOCAL;
}

void WormholeRouter::process_injections(uint64_t cycle) {
    // Process LOCAL injection (from L3 cache)
    if (local_injection_.is_active() && !local_injection_.is_complete()) {
        auto& local_input = inputs_[static_cast<size_t>(PortDir::LOCAL)];

        if (local_input.can_receive()) {
            Flit flit = local_injection_.generate_next_flit(id_);
            local_input.receive_flit(flit, cycle);

            if (local_injection_.is_complete()) {
                local_injection_.reset();
            }
        }
    }

    // Process DMA injection (if this is an edge router)
    if (has_dma_ && dma_injection_.is_active() && !dma_injection_.is_complete()) {
        auto& dma_input = inputs_[static_cast<size_t>(PortDir::DMA)];

        if (dma_input.can_receive()) {
            Flit flit = dma_injection_.generate_next_flit(id_);
            dma_input.receive_flit(flit, cycle);

            if (dma_injection_.is_complete()) {
                dma_injection_.reset();
            }
        }
    }
}

void WormholeRouter::arbitrate_and_switch(uint64_t cycle) {
    // For each output port, determine which input (if any) can send

    // First, continue existing reservations (body/tail flits)
    for (size_t out_idx = 0; out_idx < 5; out_idx++) {
        auto& out_port = outputs_[out_idx];

        if (!out_port.can_accept()) continue;

        if (out_port.reservation().is_reserved()) {
            // Path already reserved - let the flit through
            PortDir in_dir = out_port.reservation().input_port;
            auto& in_port = inputs_[static_cast<size_t>(in_dir)];

            if (in_port.has_flit()) {
                const Flit* flit = in_port.peek_flit();

                // Verify this is the same packet
                if (flit->packet_seq == out_port.reservation().packet_seq) {
                    Flit f = in_port.consume_flit();
                    out_port.accept_flit(f, cycle);
                    stats_.flits_switched++;
                }
            }
        }
    }

    // Second, arbitrate new HEAD flits
    for (size_t in_idx = 0; in_idx < 6; in_idx++) {
        auto& in_port = inputs_[in_idx];
        if (!in_port.has_flit()) continue;

        const Flit* flit = in_port.peek_flit();
        if (!flit->is_head()) continue;  // Only HEAD flits need arbitration

        // Determine output port for this packet
        PortDir out_dir = route(flit->dst_router);
        auto& out_port = outputs_[static_cast<size_t>(out_dir)];

        // Check if output is available
        if (!out_port.can_accept()) {
            stats_.arbitration_conflicts++;
            continue;  // Blocked
        }

        if (out_port.reservation().is_reserved()) {
            stats_.arbitration_conflicts++;
            continue;  // Another packet has the path
        }

        // Reserve the path
        out_port.reservation().reserve(
            static_cast<PortDir>(in_idx),
            flit->packet_seq,
            flit->total_flits
        );
        stats_.paths_reserved++;

        // Send the HEAD flit
        Flit f = in_port.consume_flit();
        out_port.accept_flit(f, cycle);
        stats_.flits_switched++;
    }
}

void WormholeRouter::process_outputs(uint64_t cycle) {
    // Each output port attempts to send one flit
    for (size_t i = 0; i < 5; i++) {
        outputs_[i].step(cycle);
    }

    // Handle LOCAL output (ejection to L3)
    auto& local_out = outputs_[static_cast<size_t>(PortDir::LOCAL)];
    if (local_out.has_flit_for_link()) {
        Flit f = local_out.take_pending_flit();
        ejection_queue_.push(f);

        // Return credit to LOCAL output (L3 consumed the flit)
        local_out.receive_credit();

        if (f.is_tail()) {
            stats_.packets_ejected++;
        }
    }
}

void WormholeRouter::receive_from_links(uint64_t cycle) {
    // Called by NoC to deliver flits from neighboring routers
    // This is handled through the output port's downstream connection
}

// ============================================================================
// WormholeTracer
// ============================================================================

void WormholeTracer::record(const WormholeTraceEvent& event) {
    events_.push_back(event);
}

bool WormholeTracer::export_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) return false;

    file << "cycle,type,router_id,port,packet_seq,flit_type,flit_index,"
         << "total_flits,src_router,dst_router,tensor,m_tile,n_tile,k_tile,size\n";

    for (const auto& e : events_) {
        file << e.cycle << ","
             << static_cast<int>(e.type) << ","
             << static_cast<int>(e.router_id) << ","
             << to_cstring(e.port) << ","
             << e.packet_seq << ","
             << static_cast<int>(e.flit_type) << ","
             << e.flit_index << ","
             << e.total_flits << ","
             << static_cast<int>(e.src_router) << ","
             << static_cast<int>(e.dst_router) << ","
             << static_cast<int>(e.tile.tensor) << ","
             << e.tile.m_tile << ","
             << e.tile.n_tile << ","
             << e.tile.k_tile << ","
             << e.tile.size << "\n";
    }

    return true;
}

bool WormholeTracer::export_chrome_trace(const std::string& filename,
                                         uint8_t mesh_rows,
                                         uint8_t mesh_cols) const {
    std::ofstream file(filename);
    if (!file) return false;

    file << "{\"traceEvents\":[\n";

    bool first = true;
    auto emit = [&](const std::string& json) {
        if (!first) file << ",\n";
        first = false;
        file << json;
    };

    // Process metadata
    emit("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,"
         "\"args\":{\"name\":\"Tile Transfers\"}}");

    // Thread per router
    for (uint8_t r = 0; r < mesh_rows; r++) {
        for (uint8_t c = 0; c < mesh_cols; c++) {
            int id = r * mesh_cols + c;
            std::ostringstream oss;
            oss << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":"
                << id << ",\"args\":{\"name\":\"R[" << r << "," << c << "]\"}}";
            emit(oss.str());
        }
    }

    // Track active injections for duration events
    std::map<uint16_t, uint64_t> inject_start;  // packet_seq -> start_cycle
    std::map<uint16_t, TileDescriptor> packet_tiles;

    for (const auto& e : events_) {
        std::ostringstream oss;

        switch (e.type) {
            case WormholeEventType::INJECT_START: {
                inject_start[e.packet_seq] = e.cycle;
                packet_tiles[e.packet_seq] = e.tile;

                // Tile name
                std::string tname;
                switch (e.tile.tensor) {
                    case TensorId::A:
                        tname = "A[" + std::to_string(e.tile.m_tile) + "," +
                                std::to_string(e.tile.k_tile) + "]";
                        break;
                    case TensorId::B:
                        tname = "B[" + std::to_string(e.tile.k_tile) + "," +
                                std::to_string(e.tile.n_tile) + "]";
                        break;
                    default:
                        tname = "Tile";
                }

                oss << "{\"name\":\"" << tname << " inject\","
                    << "\"cat\":\"inject\","
                    << "\"ph\":\"B\","
                    << "\"ts\":" << e.cycle << ","
                    << "\"pid\":1,"
                    << "\"tid\":" << static_cast<int>(e.router_id) << ","
                    << "\"args\":{\"flits\":" << e.total_flits << "}}";
                emit(oss.str());
                break;
            }

            case WormholeEventType::INJECT_COMPLETE: {
                auto it = inject_start.find(e.packet_seq);
                if (it != inject_start.end()) {
                    auto& tile = packet_tiles[e.packet_seq];
                    std::string tname;
                    switch (tile.tensor) {
                        case TensorId::A:
                            tname = "A[" + std::to_string(tile.m_tile) + "," +
                                    std::to_string(tile.k_tile) + "]";
                            break;
                        case TensorId::B:
                            tname = "B[" + std::to_string(tile.k_tile) + "," +
                                    std::to_string(tile.n_tile) + "]";
                            break;
                        default:
                            tname = "Tile";
                    }

                    oss << "{\"name\":\"" << tname << " inject\","
                        << "\"cat\":\"inject\","
                        << "\"ph\":\"E\","
                        << "\"ts\":" << e.cycle << ","
                        << "\"pid\":1,"
                        << "\"tid\":" << static_cast<int>(e.src_router) << "}";
                    emit(oss.str());
                }
                break;
            }

            case WormholeEventType::DELIVER_COMPLETE: {
                auto& tile = e.tile;
                std::string tname;
                switch (tile.tensor) {
                    case TensorId::A:
                        tname = "A[" + std::to_string(tile.m_tile) + "," +
                                std::to_string(tile.k_tile) + "]";
                        break;
                    case TensorId::B:
                        tname = "B[" + std::to_string(tile.k_tile) + "," +
                                std::to_string(tile.n_tile) + "]";
                        break;
                    default:
                        tname = "Tile";
                }

                oss << "{\"name\":\"" << tname << " deliver\","
                    << "\"cat\":\"deliver\","
                    << "\"ph\":\"i\","
                    << "\"ts\":" << e.cycle << ","
                    << "\"pid\":1,"
                    << "\"tid\":" << static_cast<int>(e.router_id) << ","
                    << "\"s\":\"t\"}";
                emit(oss.str());
                break;
            }

            default:
                break;
        }
    }

    file << "\n]}\n";
    return true;
}

// ============================================================================
// WormholeNoC
// ============================================================================

WormholeNoC::WormholeNoC(const Config& config)
    : config_(config)
    , routers_(config.rows * config.cols)
    , delivery_callbacks_(config.rows * config.cols)
{
    setup_mesh();
}

void WormholeNoC::setup_mesh() {
    // Create routers
    for (uint8_t row = 0; row < config_.rows; row++) {
        for (uint8_t col = 0; col < config_.cols; col++) {
            uint8_t id = router_id(row, col);

            // Determine if this router has DMA
            bool has_dma = false;
            if (config_.dma_on_north_edge && row == 0) has_dma = true;
            if (config_.dma_on_south_edge && row == config_.rows - 1) has_dma = true;
            if (config_.dma_on_west_edge && col == 0) has_dma = true;
            if (config_.dma_on_east_edge && col == config_.cols - 1) has_dma = true;

            routers_[id] = WormholeRouter(id, row, col, has_dma);
        }
    }

    // Connect routers
    for (uint8_t row = 0; row < config_.rows; row++) {
        for (uint8_t col = 0; col < config_.cols; col++) {
            uint8_t id = router_id(row, col);

            // Connect to North neighbor
            if (row > 0) {
                uint8_t north_id = router_id(row - 1, col);
                routers_[id].connect_output(PortDir::NORTH, &routers_[north_id]);
            }

            // Connect to South neighbor
            if (row < config_.rows - 1) {
                uint8_t south_id = router_id(row + 1, col);
                routers_[id].connect_output(PortDir::SOUTH, &routers_[south_id]);
            }

            // Connect to East neighbor
            if (col < config_.cols - 1) {
                uint8_t east_id = router_id(row, col + 1);
                routers_[id].connect_output(PortDir::EAST, &routers_[east_id]);
            }

            // Connect to West neighbor
            if (col > 0) {
                uint8_t west_id = router_id(row, col - 1);
                routers_[id].connect_output(PortDir::WEST, &routers_[west_id]);
            }
        }
    }
}

WormholeNoC::InjectResult WormholeNoC::inject_tile(
    uint8_t src_router, uint8_t dst_router,
    const TileDescriptor& tile, uint64_t cycle) {

    if (src_router >= routers_.size() || dst_router >= routers_.size()) {
        return InjectResult::INVALID;
    }

    auto& router = routers_[src_router];
    if (!router.can_inject()) {
        return InjectResult::BUSY;
    }

    uint16_t seq = next_packet_seq_++;
    uint16_t num_flits = (tile.size + FLIT_SIZE_BYTES - 1) / FLIT_SIZE_BYTES;
    if (num_flits == 0) num_flits = 1;

    if (!router.start_injection(seq, tile, dst_router, cycle)) {
        return InjectResult::BUSY;
    }

    // Track active transfer
    active_transfers_[seq] = ActiveTransfer{
        seq, src_router, dst_router, tile, cycle, num_flits, 0
    };

    stats_.tiles_injected++;
    stats_.total_flits += num_flits;
    stats_.total_bytes += tile.size;

    // Record trace event
    record_event(WormholeEventType::INJECT_START, src_router, PortDir::LOCAL,
                 seq, FlitType::HEAD, 0, num_flits, tile, src_router, dst_router, cycle);

    return InjectResult::SUCCESS;
}

bool WormholeNoC::can_inject(uint8_t router_id) const {
    if (router_id >= routers_.size()) return false;
    return routers_[router_id].can_inject();
}

void WormholeNoC::set_delivery_callback(uint8_t router_id, DeliveryCallback cb) {
    if (router_id < delivery_callbacks_.size()) {
        delivery_callbacks_[router_id] = std::move(cb);
    }
}

WormholeNoC::InjectResult WormholeNoC::dma_inject(
    uint8_t router_id, uint8_t dst_router,
    const TileDescriptor& tile, uint64_t cycle) {

    if (router_id >= routers_.size() || dst_router >= routers_.size()) {
        return InjectResult::INVALID;
    }

    auto& router = routers_[router_id];
    if (!router.has_dma()) {
        return InjectResult::INVALID;
    }

    if (!router.can_inject_dma()) {
        return InjectResult::BUSY;
    }

    uint16_t seq = next_packet_seq_++;
    uint16_t num_flits = (tile.size + FLIT_SIZE_BYTES - 1) / FLIT_SIZE_BYTES;

    if (!router.start_dma_injection(seq, tile, dst_router, cycle)) {
        return InjectResult::BUSY;
    }

    active_transfers_[seq] = ActiveTransfer{
        seq, router_id, dst_router, tile, cycle, num_flits, 0
    };

    stats_.tiles_injected++;
    stats_.total_flits += num_flits;
    stats_.total_bytes += tile.size;

    return InjectResult::SUCCESS;
}

void WormholeNoC::step(uint64_t cycle) {
    // Step 1: All routers process (injection, arbitration, output)
    for (auto& router : routers_) {
        router.step(cycle);
    }

    // Step 2: Transfer flits between routers via output ports
    for (uint8_t id = 0; id < routers_.size(); id++) {
        auto& router = routers_[id];

        for (size_t dir = 0; dir < 4; dir++) {  // N, S, E, W only
            auto& out_port = router.output(static_cast<PortDir>(dir));

            if (out_port.has_flit_for_link()) {
                Flit flit = out_port.take_pending_flit();

                // Find destination router
                uint8_t dst_id = flit.dst_router;

                // For single-hop, the immediate neighbor is the destination
                // But we need to get the actual neighbor based on direction
                int neighbor_id = -1;
                switch (static_cast<PortDir>(dir)) {
                    case PortDir::NORTH:
                        if (router.row() > 0) {
                            neighbor_id = router_id(router.row() - 1, router.col());
                        }
                        break;
                    case PortDir::SOUTH:
                        if (router.row() < config_.rows - 1) {
                            neighbor_id = router_id(router.row() + 1, router.col());
                        }
                        break;
                    case PortDir::EAST:
                        if (router.col() < config_.cols - 1) {
                            neighbor_id = router_id(router.row(), router.col() + 1);
                        }
                        break;
                    case PortDir::WEST:
                        if (router.col() > 0) {
                            neighbor_id = router_id(router.row(), router.col() - 1);
                        }
                        break;
                    default:
                        break;
                }

                if (neighbor_id >= 0) {
                    // Deliver to neighbor's input port
                    PortDir in_dir = opposite_dir(static_cast<PortDir>(dir));
                    bool accepted = routers_[neighbor_id].input(in_dir).receive_flit(flit, cycle);

                    if (!accepted) {
                        // Flit dropped - this is a critical error in credit flow
                        // Should not happen with properly sized buffers
                        std::cerr << "[NOC ERROR] Flit dropped at router " << neighbor_id
                                  << " from " << static_cast<int>(flit.src_router)
                                  << " seq=" << flit.packet_seq << "\n";
                    }
                    // Note: Credits are returned when flits are CONSUMED (switched),
                    // not when received. See return_credits_to_upstream() below.

                    // Record flit hop
                    if (tracer_) {
                        record_event(WormholeEventType::FLIT_HOP, neighbor_id, in_dir,
                                    flit.packet_seq, flit.type, 0, flit.total_flits,
                                    flit.tile, flit.src_router, flit.dst_router, cycle);
                    }
                }
            }
        }
    }

    // Step 2b: Return credits based on consumed flits
    // Credits are returned when flits are CONSUMED (switched to output), not received.
    // This ensures proper back-pressure when a packet is blocked at an intermediate router.
    return_credits_to_upstream(cycle);

    // Step 3: Check for completed deliveries
    check_deliveries(cycle);
}

void WormholeNoC::check_deliveries(uint64_t cycle) {
    // Check each router for ejected flits
    for (uint8_t id = 0; id < routers_.size(); id++) {
        auto& router = routers_[id];

        while (router.has_ejection()) {
            Flit flit = router.eject_flit();

            // Track delivery progress
            auto it = active_transfers_.find(flit.packet_seq);
            if (it != active_transfers_.end()) {
                it->second.flits_delivered++;

                // First flit
                if (it->second.flits_delivered == 1) {
                    record_event(WormholeEventType::DELIVER_START, id, PortDir::LOCAL,
                                flit.packet_seq, flit.type, 0, flit.total_flits,
                                it->second.tile, it->second.src_router,
                                it->second.dst_router, cycle);
                }

                // All flits delivered
                if (it->second.flits_delivered >= it->second.total_flits) {
                    uint64_t latency = cycle - it->second.inject_start_cycle;
                    stats_.tiles_delivered++;
                    stats_.total_latency_cycles += latency;
                    if (latency > stats_.max_latency_cycles) {
                        stats_.max_latency_cycles = latency;
                    }

                    // Record completion
                    record_event(WormholeEventType::DELIVER_COMPLETE, id, PortDir::LOCAL,
                                flit.packet_seq, FlitType::TAIL, it->second.total_flits - 1,
                                it->second.total_flits, it->second.tile,
                                it->second.src_router, it->second.dst_router, cycle);

                    // Also record inject complete (when last flit left source)
                    record_event(WormholeEventType::INJECT_COMPLETE,
                                it->second.src_router, PortDir::LOCAL,
                                flit.packet_seq, FlitType::TAIL, it->second.total_flits - 1,
                                it->second.total_flits, it->second.tile,
                                it->second.src_router, it->second.dst_router,
                                it->second.inject_start_cycle + it->second.total_flits);

                    // Invoke callback
                    if (delivery_callbacks_[id]) {
                        delivery_callbacks_[id](it->second.tile,
                                               it->second.src_router, cycle);
                    }

                    active_transfers_.erase(it);
                }
            }
        }
    }
}

void WormholeNoC::return_credits_to_upstream(uint64_t cycle) {
    (void)cycle;  // Not currently used but available for tracing

    // For each router, check each input port for credits to return
    for (uint8_t id = 0; id < routers_.size(); id++) {
        auto& router = routers_[id];

        // Check each input direction (except LOCAL and DMA which don't have upstream routers)
        for (size_t dir = 0; dir < 4; dir++) {  // N, S, E, W only
            auto& in_port = router.input(static_cast<PortDir>(dir));
            uint32_t credits = in_port.credits_to_return();

            if (credits > 0) {
                // Find the upstream router and output port
                int upstream_id = -1;
                PortDir upstream_out_dir = PortDir::LOCAL;

                switch (static_cast<PortDir>(dir)) {
                    case PortDir::NORTH:  // Came from south of upstream router
                        if (router.row() > 0) {
                            upstream_id = router_id(router.row() - 1, router.col());
                            upstream_out_dir = PortDir::SOUTH;
                        }
                        break;
                    case PortDir::SOUTH:  // Came from north of upstream router
                        if (router.row() < config_.rows - 1) {
                            upstream_id = router_id(router.row() + 1, router.col());
                            upstream_out_dir = PortDir::NORTH;
                        }
                        break;
                    case PortDir::EAST:  // Came from west of upstream router
                        if (router.col() < config_.cols - 1) {
                            upstream_id = router_id(router.row(), router.col() + 1);
                            upstream_out_dir = PortDir::WEST;
                        }
                        break;
                    case PortDir::WEST:  // Came from east of upstream router
                        if (router.col() > 0) {
                            upstream_id = router_id(router.row(), router.col() - 1);
                            upstream_out_dir = PortDir::EAST;
                        }
                        break;
                    default:
                        break;
                }

                if (upstream_id >= 0) {
                    // Return credits to the upstream router's output port
                    for (uint32_t c = 0; c < credits; c++) {
                        routers_[upstream_id].output(upstream_out_dir).receive_credit();
                    }
                }

                in_port.clear_returned_credits();
            }
        }
    }
}

bool WormholeNoC::is_idle() const {
    if (!active_transfers_.empty()) return false;

    for (const auto& router : routers_) {
        if (!router.is_idle()) return false;
    }

    return true;
}

void WormholeNoC::reset_stats() {
    stats_ = Stats{};
    for (auto& router : routers_) {
        router.reset_stats();
    }
}

void WormholeNoC::record_event(WormholeEventType type, uint8_t router_id,
                               PortDir port, uint16_t packet_seq,
                               FlitType flit_type, uint16_t flit_index,
                               uint16_t total_flits, const TileDescriptor& tile,
                               uint8_t src_router, uint8_t dst_router,
                               uint64_t cycle) {
    if (!tracer_) return;

    WormholeTraceEvent event;
    event.cycle = cycle;
    event.type = type;
    event.router_id = router_id;
    event.port = port;
    event.packet_seq = packet_seq;
    event.flit_type = flit_type;
    event.flit_index = flit_index;
    event.total_flits = total_flits;
    event.tile = tile;
    event.src_router = src_router;
    event.dst_router = dst_router;

    tracer_->record(event);
}

}  // namespace sw::kpu::noc
