// ============================================================================
// src/components/datamovement/cycle_accurate_dma_engine.cpp
// Cycle-accurate DMA engine implementation
// ============================================================================

#include <sw/kpu/components/dma/cycle_accurate_dma_engine.hpp>

#include <algorithm>
#include <stdexcept>

namespace sw::kpu {

// ============================================================================
// Constructor
// ============================================================================

CycleAccurateDMAEngine::CycleAccurateDMAEngine(const CycleAccurateDMAConfig& config)
    : config_(config)
{
    // Initialize channels
    channels_.resize(config_.num_channels);
    for (uint32_t i = 0; i < config_.num_channels; ++i) {
        channels_[i].channel_id = i;
        channels_[i].reset();
    }

    // Initialize channel statistics
    stats_.channel_stats.resize(config_.num_channels);

    // Set default tile mapper
    tile_mapper_ = [this](const DMATransfer& transfer, uint8_t& dst_router) {
        return default_tile_mapper(transfer, dst_router);
    };
}

// ============================================================================
// External Bindings
// ============================================================================

void CycleAccurateDMAEngine::bind_memory_controller(IMemoryController* controller) {
    memory_controller_ = controller;
}

void CycleAccurateDMAEngine::bind_noc(noc::INoC* noc, uint8_t edge_router_id) {
    noc_ = noc;
    bound_router_id_ = edge_router_id;
}

void CycleAccurateDMAEngine::set_tile_mapper(TileMapperFunc mapper) {
    if (mapper) {
        tile_mapper_ = mapper;
    }
}

// ============================================================================
// Request Interface
// ============================================================================

std::optional<uint64_t> CycleAccurateDMAEngine::submit(
    const DMATransfer& transfer,
    std::function<void()> callback)
{
    // Check if we can accept
    if (!can_accept()) {
        return std::nullopt;
    }

    // Generate transfer ID
    uint64_t transfer_id = next_transfer_id_++;

    // Try to find an idle channel
    auto idle_ch = select_idle_channel();

    if (idle_ch.has_value()) {
        // Assign directly to channel
        uint32_t ch_id = *idle_ch;
        auto& ch = channels_[ch_id];

        ch.transfer_id = transfer_id;
        ch.request = transfer;
        ch.callback = callback;
        ch.submit_cycle = current_cycle_;
        ch.state = DMAChannelState::IDLE;  // Will transition on first tick
        ch.data_buffer.resize(transfer.size);

        // Map transfer to tile
        ch.tile = tile_mapper_(transfer, ch.target_router_id);
        ch.packet_seq = next_packet_seq_++;

        transfer_to_channel_[transfer_id] = ch_id;

        // Immediately transition to first state
        if (transfer.src_type == DMAMemoryType::DEVICE_MEMORY ||
            transfer.src_type == DMAMemoryType::HOST_MEMORY) {
            // Load operation: read from memory, inject into NoC
            ch.state = DMAChannelState::STALLED_MEMORY_FULL;  // Will try on tick
        } else {
            // Other transfer types - handle as appropriate
            ch.state = DMAChannelState::STALLED_MEMORY_FULL;
        }
    } else {
        // Queue for later
        pending_queue_.push_back({transfer_id, transfer, callback});
    }

    pending_count_++;
    stats_.transfers_pending++;

    trace_event("submit", idle_ch.value_or(UINT32_MAX), transfer_id);

    return transfer_id;
}

bool CycleAccurateDMAEngine::can_accept() const {
    // Check if any channel is idle or queue has space
    for (const auto& ch : channels_) {
        if (!ch.is_busy()) {
            return true;
        }
    }

    // Also accept if pending queue has space
    return pending_queue_.size() < config_.queue_depth_per_channel;
}

bool CycleAccurateDMAEngine::wait_for(uint64_t transfer_id) {
    // Find the channel with this transfer
    auto it = transfer_to_channel_.find(transfer_id);
    if (it == transfer_to_channel_.end()) {
        // Check pending queue
        for (const auto& pt : pending_queue_) {
            if (pt.transfer_id == transfer_id) {
                // Still pending - process until complete
                while (true) {
                    tick();
                    auto it2 = transfer_to_channel_.find(transfer_id);
                    if (it2 == transfer_to_channel_.end()) {
                        // Completed and removed
                        return true;
                    }
                    if (!channels_[it2->second].is_busy()) {
                        return true;
                    }
                }
            }
        }
        return false;  // Not found
    }

    // Process until transfer completes
    uint32_t ch_id = it->second;
    while (channels_[ch_id].is_busy() &&
           channels_[ch_id].transfer_id == transfer_id) {
        tick();
    }

    return true;
}

bool CycleAccurateDMAEngine::cancel(uint64_t transfer_id) {
    // Check pending queue first
    for (auto it = pending_queue_.begin(); it != pending_queue_.end(); ++it) {
        if (it->transfer_id == transfer_id) {
            pending_queue_.erase(it);
            pending_count_--;
            stats_.transfers_pending--;
            stats_.transfers_failed++;
            return true;
        }
    }

    // Check active transfers
    auto it = transfer_to_channel_.find(transfer_id);
    if (it != transfer_to_channel_.end()) {
        uint32_t ch_id = it->second;
        auto& ch = channels_[ch_id];

        if (ch.transfer_id == transfer_id) {
            ch.reset();
            transfer_to_channel_.erase(it);
            pending_count_--;
            stats_.transfers_pending--;
            stats_.transfers_failed++;
            return true;
        }
    }

    return false;
}

// ============================================================================
// Simulation Interface
// ============================================================================

void CycleAccurateDMAEngine::tick() {
    current_cycle_++;

    // Process each channel
    for (uint32_t ch = 0; ch < channels_.size(); ++ch) {
        process_channel(ch);
    }

    // Update channel statistics
    update_channel_stats();
}

void CycleAccurateDMAEngine::drain() {
    while (has_pending()) {
        tick();
    }
}

void CycleAccurateDMAEngine::reset() {
    current_cycle_ = 0;
    next_transfer_id_ = 1;
    next_packet_seq_ = 1;
    pending_count_ = 0;

    for (auto& ch : channels_) {
        ch.reset();
    }

    pending_queue_.clear();
    transfer_to_channel_.clear();
    stats_.reset();
}

// ============================================================================
// Configuration Queries
// ============================================================================

IDMAEngine::ChannelState CycleAccurateDMAEngine::get_channel_state(uint32_t channel) const {
    if (channel >= channels_.size()) {
        return ChannelState::ERROR;
    }

    switch (channels_[channel].state) {
        case DMAChannelState::IDLE:
            return ChannelState::IDLE;
        case DMAChannelState::ERROR:
            return ChannelState::ERROR;
        case DMAChannelState::STALLED_MEMORY_FULL:
        case DMAChannelState::STALLED_NOC_FULL:
            return ChannelState::STALLED;
        default:
            return ChannelState::BUSY;
    }
}

DMAChannelState CycleAccurateDMAEngine::get_detailed_channel_state(uint32_t channel) const {
    if (channel >= channels_.size()) {
        return DMAChannelState::ERROR;
    }
    return channels_[channel].state;
}

// ============================================================================
// State Machine Processing
// ============================================================================

void CycleAccurateDMAEngine::process_channel(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    switch (ch.state) {
        case DMAChannelState::IDLE:
            // Try to assign pending transfer
            try_assign_pending_transfer(ch_id);
            break;

        case DMAChannelState::STALLED_MEMORY_FULL:
            // Determine if this is a read or write based on transfer direction
            if (ch.request.src_type == DMAMemoryType::DEVICE_MEMORY ||
                ch.request.src_type == DMAMemoryType::HOST_MEMORY) {
                // Load operation: read from device memory
                issue_memory_read(ch_id);
            } else if (ch.request.dst_type == DMAMemoryType::DEVICE_MEMORY ||
                       ch.request.dst_type == DMAMemoryType::HOST_MEMORY) {
                // Store operation: write to device memory
                issue_memory_write(ch_id);
            } else {
                // Local transfer (L3 to L3, etc.) - skip memory operations
                ch.memory_complete = true;
                ch.state = DMAChannelState::COMPLETE;
            }
            break;

        case DMAChannelState::WAITING_MEMORY_READ:
            // Check for completion
            process_waiting_memory_read(ch_id);
            break;

        case DMAChannelState::MEMORY_READ_COMPLETE:
            // Try NoC injection
            attempt_noc_injection(ch_id);
            break;

        case DMAChannelState::STALLED_NOC_FULL:
            // Retry NoC injection after delay
            if (current_cycle_ >= ch.last_noc_inject_cycle + config_.noc_retry_delay) {
                attempt_noc_injection(ch_id);
            }
            break;

        case DMAChannelState::NOC_INJECTING:
            // Monitor injection completion
            process_noc_injecting(ch_id);
            break;

        case DMAChannelState::WAITING_MEMORY_WRITE:
            // Check for completion
            process_waiting_memory_write(ch_id);
            break;

        case DMAChannelState::COMPLETE:
            // Finalize transfer
            complete_transfer(ch_id);
            break;

        case DMAChannelState::ERROR:
            // Stay in error state
            break;

        default:
            break;
    }
}

void CycleAccurateDMAEngine::try_assign_pending_transfer(uint32_t ch_id) {
    if (pending_queue_.empty()) {
        return;
    }

    auto& ch = channels_[ch_id];
    auto pending = std::move(pending_queue_.front());
    pending_queue_.pop_front();

    ch.transfer_id = pending.transfer_id;
    ch.request = pending.request;
    ch.callback = pending.callback;
    ch.submit_cycle = current_cycle_;
    ch.data_buffer.resize(pending.request.size);

    // Map transfer to tile
    ch.tile = tile_mapper_(pending.request, ch.target_router_id);
    ch.packet_seq = next_packet_seq_++;

    transfer_to_channel_[pending.transfer_id] = ch_id;

    // Start with memory read
    ch.state = DMAChannelState::STALLED_MEMORY_FULL;

    trace_event("assign", ch_id, pending.transfer_id);
}

void CycleAccurateDMAEngine::issue_memory_read(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    // Check timing constraint
    if (current_cycle_ < ch.last_memory_issue_cycle + config_.memory_issue_interval) {
        return;
    }

    // Check if memory controller is bound
    if (!memory_controller_) {
        // No memory controller - skip to NoC injection
        ch.memory_complete = true;
        ch.state = DMAChannelState::MEMORY_READ_COMPLETE;
        return;
    }

    // Check if memory controller can accept
    if (!memory_controller_->can_accept()) {
        ch.memory_retry_count++;
        stats_.stall_cycles++;

        if (ch.memory_retry_count > ch.max_retries) {
            handle_error(ch_id, "Memory retry limit exceeded");
        }
        return;
    }

    // Issue read with callback
    auto req_id = memory_controller_->submit_read(
        ch.request.src_addr,
        ch.request.size,
        [this, ch_id]() {
            if (ch_id < channels_.size()) {
                channels_[ch_id].memory_complete = true;
            }
        });

    if (req_id) {
        ch.memory_request_id = *req_id;
        ch.state = DMAChannelState::WAITING_MEMORY_READ;
        ch.last_memory_issue_cycle = current_cycle_;
        ch.memory_retry_count = 0;

        trace_event("memory_read_issued", ch_id, ch.transfer_id);
    }
}

void CycleAccurateDMAEngine::process_waiting_memory_read(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    if (ch.memory_complete) {
        ch.state = DMAChannelState::MEMORY_READ_COMPLETE;
        trace_event("memory_read_complete", ch_id, ch.transfer_id);
    }
}

void CycleAccurateDMAEngine::attempt_noc_injection(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    // Check timing constraint
    if (current_cycle_ < ch.last_noc_inject_cycle + config_.noc_inject_interval) {
        return;
    }

    // Check if NoC is bound
    if (!noc_) {
        // No NoC - complete immediately
        ch.state = DMAChannelState::COMPLETE;
        return;
    }

    // Attempt injection
    auto result = noc_->inject_tile(
        bound_router_id_,
        ch.target_router_id,
        ch.tile,
        current_cycle_);

    ch.last_noc_inject_cycle = current_cycle_;

    switch (result) {
        case noc::InjectResult::SUCCESS:
            ch.state = DMAChannelState::NOC_INJECTING;
            ch.noc_inject_started = true;
            ch.noc_retry_count = 0;
            trace_event("noc_inject_success", ch_id, ch.transfer_id);
            break;

        case noc::InjectResult::BUSY:
            ch.state = DMAChannelState::STALLED_NOC_FULL;
            ch.noc_retry_count++;
            stats_.stall_cycles++;

            if (ch.noc_retry_count > ch.max_retries) {
                handle_error(ch_id, "NoC retry limit exceeded");
            }
            trace_event("noc_inject_busy", ch_id, ch.transfer_id);
            break;

        case noc::InjectResult::INVALID:
            handle_error(ch_id, "Invalid NoC injection parameters");
            break;
    }
}

void CycleAccurateDMAEngine::process_noc_injecting(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    // For now, assume single-flit injection completes in one cycle
    // In a more detailed model, we would track multi-flit packet progress
    ch.state = DMAChannelState::COMPLETE;
    trace_event("noc_inject_complete", ch_id, ch.transfer_id);
}

void CycleAccurateDMAEngine::issue_memory_write(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    if (!memory_controller_) {
        ch.memory_complete = true;
        ch.state = DMAChannelState::COMPLETE;
        return;
    }

    if (!memory_controller_->can_accept()) {
        ch.memory_retry_count++;
        stats_.stall_cycles++;
        return;
    }

    auto req_id = memory_controller_->submit_write(
        ch.request.dst_addr,
        ch.data_buffer.data(),
        ch.request.size,
        [this, ch_id]() {
            if (ch_id < channels_.size()) {
                channels_[ch_id].memory_complete = true;
            }
        });

    if (req_id) {
        ch.memory_request_id = *req_id;
        ch.state = DMAChannelState::WAITING_MEMORY_WRITE;
        ch.last_memory_issue_cycle = current_cycle_;
        ch.memory_retry_count = 0;
    }
}

void CycleAccurateDMAEngine::process_waiting_memory_write(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    if (ch.memory_complete) {
        ch.state = DMAChannelState::COMPLETE;
    }
}

void CycleAccurateDMAEngine::complete_transfer(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    // Update statistics
    update_stats(ch);

    // Invoke callback
    if (ch.callback) {
        ch.callback();
    }

    trace_event("complete", ch_id, ch.transfer_id);

    // Remove from tracking
    transfer_to_channel_.erase(ch.transfer_id);
    pending_count_--;

    // Reset channel
    ch.reset();
}

void CycleAccurateDMAEngine::handle_error(uint32_t ch_id, const std::string& message) {
    auto& ch = channels_[ch_id];

    ch.state = DMAChannelState::ERROR;
    stats_.transfers_failed++;

    trace_event("error: " + message, ch_id, ch.transfer_id);
}

// ============================================================================
// Arbitration
// ============================================================================

std::optional<uint32_t> CycleAccurateDMAEngine::select_idle_channel() {
    switch (config_.arbitration) {
        case CycleAccurateDMAConfig::ArbitrationPolicy::ROUND_ROBIN: {
            for (uint32_t i = 0; i < channels_.size(); ++i) {
                uint32_t ch = (next_channel_rr_ + i) % channels_.size();
                if (!channels_[ch].is_busy()) {
                    next_channel_rr_ = (ch + 1) % channels_.size();
                    return ch;
                }
            }
            break;
        }

        case CycleAccurateDMAConfig::ArbitrationPolicy::PRIORITY:
        case CycleAccurateDMAConfig::ArbitrationPolicy::OLDEST_FIRST:
        default:
            // Fall back to simple scan
            for (uint32_t ch = 0; ch < channels_.size(); ++ch) {
                if (!channels_[ch].is_busy()) {
                    return ch;
                }
            }
            break;
    }

    return std::nullopt;
}

// ============================================================================
// Helpers
// ============================================================================

TileDescriptor CycleAccurateDMAEngine::default_tile_mapper(
    const DMATransfer& transfer,
    uint8_t& dst_router_id)
{
    TileDescriptor tile;

    // Simple mapping based on destination type and ID
    tile.tensor = TensorId::A;  // Default
    tile.m_tile = 0;
    tile.n_tile = 0;
    tile.k_tile = 0;
    tile.l3_offset = static_cast<uint32_t>(transfer.dst_addr);
    tile.size = transfer.size;
    tile.height = 1;
    tile.width = transfer.size;

    // Default destination router based on dst_id
    dst_router_id = static_cast<uint8_t>(transfer.dst_id % 16);

    return tile;
}

void CycleAccurateDMAEngine::update_stats(const DMAChannel& channel) {
    uint64_t latency = current_cycle_ - channel.submit_cycle;

    stats_.transfers_completed++;
    stats_.transfers_pending--;
    stats_.bytes_transferred += channel.request.size;
    stats_.total_latency_cycles += latency;

    if (latency < stats_.min_latency) {
        stats_.min_latency = latency;
    }
    if (latency > stats_.max_latency) {
        stats_.max_latency = latency;
    }

    // Categorize by transfer type
    if (channel.request.src_type == DMAMemoryType::HOST_MEMORY) {
        stats_.bytes_host_to_device += channel.request.size;
    } else if (channel.request.dst_type == DMAMemoryType::HOST_MEMORY) {
        stats_.bytes_device_to_host += channel.request.size;
    } else {
        stats_.bytes_device_to_device += channel.request.size;
    }

    // Update channel-specific stats
    if (channel.channel_id < stats_.channel_stats.size()) {
        auto& cs = stats_.channel_stats[channel.channel_id];
        cs.transfers++;
        cs.bytes += channel.request.size;
    }
}

void CycleAccurateDMAEngine::update_channel_stats() {
    bool any_busy = false;

    for (uint32_t ch = 0; ch < channels_.size(); ++ch) {
        if (channels_[ch].is_busy()) {
            any_busy = true;
            if (ch < stats_.channel_stats.size()) {
                stats_.channel_stats[ch].busy_cycles++;
            }
        }
    }

    if (any_busy) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void CycleAccurateDMAEngine::trace_event(const std::string& event,
                                          uint32_t ch_id,
                                          uint64_t transfer_id) {
    if (!tracing_enabled_ || !resource_tracker_) {
        return;
    }

    // Tracing would be implemented here using resource_tracker_
    // For now, this is a placeholder
    (void)event;
    (void)ch_id;
    (void)transfer_id;
}

} // namespace sw::kpu
