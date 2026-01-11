// ============================================================================
// src/components/dma/transactional_dma_engine.cpp
// Transactional (bandwidth-limited) DMA engine implementation
// ============================================================================

#include <sw/kpu/models/transactional/datamovement/dma_engine.hpp>

#include <algorithm>
#include <limits>

namespace sw::kpu {

// ============================================================================
// Constructor / Reset
// ============================================================================

TransactionalDMAEngine::TransactionalDMAEngine(const DMAEngineConfig& config)
    : config_(config)
{
    // Calculate bytes per cycle from bandwidth
    // bandwidth_gbps = GB/s, assume 1 GHz clock for simplicity
    // bytes_per_cycle = (bandwidth_gbps * 1e9 bytes/s) / (1e9 cycles/s) = bandwidth_gbps bytes/cycle
    bytes_per_cycle_ = static_cast<double>(config.bandwidth_gbps);

    // Initialize channels
    channels_.resize(config.num_channels);
    stats_.channel_stats.resize(config.num_channels);
}

void TransactionalDMAEngine::reset() {
    current_cycle_ = 0;
    next_transfer_id_ = 1;
    pending_count_ = 0;

    // Clear channels
    for (auto& ch : channels_) {
        ch.busy_until_cycle = 0;
        ch.pending_transfer_ids.clear();
    }

    // Clear transfers
    transfers_.clear();

    // Clear completion queue
    while (!completion_queue_.empty()) {
        completion_queue_.pop();
    }

    // Reset stats
    stats_.reset();
    stats_.channel_stats.resize(config_.num_channels);
}

// ============================================================================
// Request Interface
// ============================================================================

std::optional<uint64_t> TransactionalDMAEngine::submit(
    const DMATransfer& transfer,
    std::function<void()> callback)
{
    // Check queue depth
    if (config_.queue_depth_per_channel > 0) {
        // Check if any channel can accept
        bool can_accept_any = false;
        for (const auto& ch : channels_) {
            if (ch.pending_transfer_ids.size() < config_.queue_depth_per_channel) {
                can_accept_any = true;
                break;
            }
        }
        if (!can_accept_any) {
            return std::nullopt;
        }
    }

    // Assign transfer ID
    uint64_t transfer_id = next_transfer_id_++;

    // Select channel
    uint32_t channel = select_channel(transfer);
    auto& ch = channels_[channel];

    // Calculate when this transfer can start
    uint64_t start_cycle = std::max(current_cycle_, ch.busy_until_cycle);

    // Calculate duration
    uint32_t duration = compute_duration(transfer);

    // Calculate completion cycle
    uint64_t completion_cycle = start_cycle + duration;

    // Create transfer info
    TransferInfo info{
        .transfer_id = transfer_id,
        .submit_cycle = current_cycle_,
        .completion_cycle = completion_cycle,
        .channel = channel,
        .size = transfer.size,
        .transfer = transfer,
        .callback = callback,
        .completed = false,
        .cancelled = false
    };

    transfers_[transfer_id] = info;

    // Update channel state
    ch.busy_until_cycle = completion_cycle;
    ch.pending_transfer_ids.push_back(transfer_id);

    // Add to completion queue
    completion_queue_.push(CompletionEvent{
        .transfer_id = transfer_id,
        .completion_cycle = completion_cycle
    });

    pending_count_++;

    return transfer_id;
}

bool TransactionalDMAEngine::can_accept() const {
    if (config_.queue_depth_per_channel == 0) return true;

    for (const auto& ch : channels_) {
        if (ch.pending_transfer_ids.size() < config_.queue_depth_per_channel) {
            return true;
        }
    }
    return false;
}

bool TransactionalDMAEngine::wait_for(uint64_t transfer_id) {
    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end()) {
        return false;
    }

    while (!it->second.completed && !it->second.cancelled && has_pending()) {
        tick();
    }

    return it->second.completed;
}

bool TransactionalDMAEngine::cancel(uint64_t transfer_id) {
    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end() || it->second.completed) {
        return false;
    }

    it->second.cancelled = true;
    return true;
}

// ============================================================================
// Simulation Interface
// ============================================================================

void TransactionalDMAEngine::tick() {
    current_cycle_++;

    // Process completed transfers
    while (!completion_queue_.empty() &&
           completion_queue_.top().completion_cycle <= current_cycle_) {

        CompletionEvent event = completion_queue_.top();
        completion_queue_.pop();

        auto it = transfers_.find(event.transfer_id);
        if (it == transfers_.end()) continue;

        TransferInfo& info = it->second;

        if (info.cancelled) {
            // Remove from channel queue
            auto& ch = channels_[info.channel];
            auto q_it = std::find(ch.pending_transfer_ids.begin(),
                                  ch.pending_transfer_ids.end(),
                                  info.transfer_id);
            if (q_it != ch.pending_transfer_ids.end()) {
                ch.pending_transfer_ids.erase(q_it);
            }
            pending_count_--;
            continue;
        }

        // Mark completed
        info.completed = true;

        // Update statistics
        update_stats(info);

        // Remove from channel queue
        auto& ch = channels_[info.channel];
        auto q_it = std::find(ch.pending_transfer_ids.begin(),
                              ch.pending_transfer_ids.end(),
                              info.transfer_id);
        if (q_it != ch.pending_transfer_ids.end()) {
            ch.pending_transfer_ids.erase(q_it);
        }

        pending_count_--;

        // Invoke callback
        if (info.callback) {
            info.callback();
        }
    }

    // Update utilization stats
    bool any_busy = false;
    for (size_t i = 0; i < channels_.size(); ++i) {
        if (!channels_[i].is_idle(current_cycle_)) {
            any_busy = true;
            stats_.channel_stats[i].busy_cycles++;
        }
    }

    if (any_busy) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void TransactionalDMAEngine::drain() {
    while (has_pending()) {
        tick();
    }
}

// ============================================================================
// Configuration Queries
// ============================================================================

IDMAEngine::ChannelState TransactionalDMAEngine::get_channel_state(uint32_t channel) const {
    if (channel >= channels_.size()) {
        return ChannelState::ERROR;
    }

    const auto& ch = channels_[channel];
    if (ch.is_idle(current_cycle_)) {
        return ChannelState::IDLE;
    }
    return ChannelState::BUSY;
}

// ============================================================================
// Helpers
// ============================================================================

uint32_t TransactionalDMAEngine::select_channel(const DMATransfer& transfer) {
    // Simple round-robin with priority to less loaded channels
    uint32_t best_channel = 0;
    uint64_t earliest_available = std::numeric_limits<uint64_t>::max();

    for (uint32_t i = 0; i < config_.num_channels; ++i) {
        const auto& ch = channels_[i];

        // Skip full channels
        if (config_.queue_depth_per_channel > 0 &&
            ch.pending_transfer_ids.size() >= config_.queue_depth_per_channel) {
            continue;
        }

        // Find earliest available
        uint64_t available = ch.busy_until_cycle;
        if (available < earliest_available) {
            earliest_available = available;
            best_channel = i;
        }
    }

    // Use priority to break ties (not implemented yet)
    (void)transfer;

    return best_channel;
}

uint32_t TransactionalDMAEngine::compute_duration(const DMATransfer& transfer) const {
    // Duration = size / bandwidth
    // Account for strided transfers
    uint32_t total_bytes = transfer.size * transfer.count;

    if (bytes_per_cycle_ <= 0) {
        return 1;  // Minimum latency
    }

    uint32_t cycles = static_cast<uint32_t>(total_bytes / bytes_per_cycle_);
    return std::max(cycles, 1u);  // At least 1 cycle
}

void TransactionalDMAEngine::update_stats(const TransferInfo& info) {
    stats_.transfers_completed++;
    stats_.bytes_transferred += info.size;

    // Categorize transfer
    if (info.transfer.src_type == DMAMemoryType::HOST_MEMORY) {
        stats_.bytes_host_to_device += info.size;
    } else if (info.transfer.dst_type == DMAMemoryType::HOST_MEMORY) {
        stats_.bytes_device_to_host += info.size;
    } else {
        stats_.bytes_device_to_device += info.size;
    }

    // Latency stats
    uint64_t latency = info.completion_cycle - info.submit_cycle;
    stats_.total_latency_cycles += latency;
    stats_.min_latency = std::min(stats_.min_latency, latency);
    stats_.max_latency = std::max(stats_.max_latency, latency);

    // Per-channel stats
    if (info.channel < stats_.channel_stats.size()) {
        stats_.channel_stats[info.channel].transfers++;
        stats_.channel_stats[info.channel].bytes += info.size;
    }
}

} // namespace sw::kpu
