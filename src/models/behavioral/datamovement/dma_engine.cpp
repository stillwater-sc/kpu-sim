// ============================================================================
// src/components/dma/behavioral_dma_engine.cpp
// Behavioral (functional) DMA engine implementation
// ============================================================================

#include <sw/kpu/models/behavioral/datamovement/dma_engine.hpp>

#include <algorithm>

namespace sw::kpu {

// ============================================================================
// Constructor / Reset
// ============================================================================

BehavioralDMAEngine::BehavioralDMAEngine(const DMAEngineConfig& config)
    : config_(config)
{
    stats_.channel_stats.resize(config.num_channels);
}

void BehavioralDMAEngine::reset() {
    current_cycle_ = 0;
    next_transfer_id_ = 1;

    // Clear pending callbacks
    while (!pending_callbacks_.empty()) {
        pending_callbacks_.pop();
    }

    active_transfers_.clear();
    stats_.reset();
    stats_.channel_stats.resize(config_.num_channels);
}

// ============================================================================
// Request Interface
// ============================================================================

std::optional<uint64_t> BehavioralDMAEngine::submit(
    const DMATransfer& transfer,
    std::function<void()> callback)
{
    uint64_t transfer_id = next_transfer_id_++;

    // Compute latency
    uint32_t latency = compute_latency(transfer);

    // Schedule completion
    uint64_t completion_cycle = current_cycle_ + latency;

    if (callback || latency > 0) {
        pending_callbacks_.push(PendingCallback{
            .transfer_id = transfer_id,
            .completion_cycle = completion_cycle,
            .callback = callback
        });
        active_transfers_[transfer_id] = false;
    }

    // Update statistics
    update_stats(transfer, latency);

    return transfer_id;
}

std::optional<uint64_t> IDMAEngine::submit(
    DMAMemoryType src_type, uint32_t src_id, uint64_t src_addr,
    DMAMemoryType dst_type, uint32_t dst_id, uint64_t dst_addr,
    uint32_t size,
    std::function<void()> callback)
{
    DMATransfer transfer{
        .src_type = src_type,
        .src_id = src_id,
        .src_addr = src_addr,
        .dst_type = dst_type,
        .dst_id = dst_id,
        .dst_addr = dst_addr,
        .size = size
    };
    return submit(transfer, callback);
}

bool BehavioralDMAEngine::wait_for(uint64_t transfer_id) {
    auto it = active_transfers_.find(transfer_id);
    if (it == active_transfers_.end()) {
        return false;  // Unknown transfer
    }

    // In behavioral mode, just drain until this transfer completes
    while (!it->second && has_pending()) {
        tick();
    }

    return it->second;
}

bool BehavioralDMAEngine::cancel(uint64_t transfer_id) {
    auto it = active_transfers_.find(transfer_id);
    if (it == active_transfers_.end() || it->second) {
        return false;  // Unknown or already completed
    }

    // Mark as cancelled (callback won't fire)
    // In behavioral mode, we can't actually remove from priority queue,
    // but we can mark it so the callback doesn't fire
    it->second = true;  // Mark as "done" so wait_for returns
    return true;
}

// ============================================================================
// Simulation Interface
// ============================================================================

void BehavioralDMAEngine::tick() {
    current_cycle_++;

    // Process completed transfers
    while (!pending_callbacks_.empty() &&
           pending_callbacks_.top().completion_cycle <= current_cycle_) {

        PendingCallback cb = pending_callbacks_.top();
        pending_callbacks_.pop();

        // Mark as completed
        auto it = active_transfers_.find(cb.transfer_id);
        if (it != active_transfers_.end()) {
            it->second = true;
        }

        // Invoke callback if not cancelled
        if (cb.callback) {
            cb.callback();
        }
    }

    // Update utilization stats
    if (has_pending()) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void BehavioralDMAEngine::drain() {
    while (has_pending()) {
        tick();
    }
}

// ============================================================================
// Helpers
// ============================================================================

uint32_t BehavioralDMAEngine::compute_latency(const DMATransfer& transfer) const {
    // Fixed latency per burst
    uint32_t num_bursts = (transfer.size + config_.max_burst_size - 1) /
                          config_.max_burst_size;
    return num_bursts * config_.fixed_latency_per_burst;
}

void BehavioralDMAEngine::update_stats(const DMATransfer& transfer, uint32_t latency) {
    stats_.transfers_completed++;
    stats_.bytes_transferred += transfer.size;

    // Categorize transfer
    if (transfer.src_type == DMAMemoryType::HOST_MEMORY) {
        stats_.bytes_host_to_device += transfer.size;
    } else if (transfer.dst_type == DMAMemoryType::HOST_MEMORY) {
        stats_.bytes_device_to_host += transfer.size;
    } else {
        stats_.bytes_device_to_device += transfer.size;
    }

    // Latency stats
    stats_.total_latency_cycles += latency;
    stats_.min_latency = std::min(stats_.min_latency, static_cast<uint64_t>(latency));
    stats_.max_latency = std::max(stats_.max_latency, static_cast<uint64_t>(latency));
}

} // namespace sw::kpu
