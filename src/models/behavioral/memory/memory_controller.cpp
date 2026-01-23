// ============================================================================
// src/components/memory/behavioral_memory_controller.cpp
// Behavioral (functional) memory controller implementation
// ============================================================================

#include <sw/kpu/models/behavioral/memory/memory_controller.hpp>
#include <sw/xue/event_collector.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sw::kpu {

// Static empty violations list
const std::vector<IMemoryController::InvariantViolation>
    BehavioralMemoryController::empty_violations_;

BehavioralMemoryController::BehavioralMemoryController(const MemoryControllerConfig& config)
    : config_(config)
{
    // Calculate capacity based on configuration
    // Default: 1GB per channel
    uint64_t bytes_per_channel = 1ULL << 30;  // 1GB
    capacity_ = static_cast<uint64_t>(config_.num_channels) * bytes_per_channel;
}

std::optional<uint64_t> BehavioralMemoryController::submit_read(
    [[maybe_unused]]uint64_t address,
    uint32_t size,
    std::function<void()> callback)
{
    uint64_t request_id = next_request_id_++;

    // Update statistics
    stats_.reads++;

    // In behavioral mode, reads always hit
    stats_.page_hits++;

    // Calculate completion cycle
    uint64_t latency = config_.timing.fixed_read_latency;
    uint64_t completion_cycle = current_cycle_ + latency;

    // Track latency
    stats_.total_latency += latency;
    stats_.min_latency = std::min(stats_.min_latency, latency);
    stats_.max_latency = std::max(stats_.max_latency, latency);

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_dram_read(size, latency);

    // If zero latency (instant mode), invoke callback immediately
    if (latency == 0) {
        if (callback) {
            callback();
        }
    } else {
        // Schedule callback for later
        if (callback) {
            pending_callbacks_.push({completion_cycle, std::move(callback)});
        }
    }

    return request_id;
}

std::optional<uint64_t> BehavioralMemoryController::submit_write(
    uint64_t address,
    const void* data,
    uint32_t size,
    std::function<void()> callback)
{
    uint64_t request_id = next_request_id_++;

    // Actually store the data
    if (data && size > 0) {
        write_direct(address, data, size);
    }

    // Update statistics
    stats_.writes++;
    stats_.page_hits++;

    // Calculate completion cycle
    uint64_t latency = config_.timing.fixed_write_latency;
    uint64_t completion_cycle = current_cycle_ + latency;

    // Track latency
    stats_.total_latency += latency;
    stats_.min_latency = std::min(stats_.min_latency, latency);
    stats_.max_latency = std::max(stats_.max_latency, latency);

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_dram_write(size, latency);

    // If zero latency, invoke callback immediately
    if (latency == 0) {
        if (callback) {
            callback();
        }
    } else {
        if (callback) {
            pending_callbacks_.push({completion_cycle, std::move(callback)});
        }
    }

    return request_id;
}

void BehavioralMemoryController::tick() {
    current_cycle_++;

    // Process any callbacks that are due
    while (!pending_callbacks_.empty() &&
           pending_callbacks_.top().completion_cycle <= current_cycle_) {
        auto callback = std::move(const_cast<PendingCallback&>(pending_callbacks_.top()).callback);
        pending_callbacks_.pop();
        if (callback) {
            callback();
        }
    }

    // Update utilization stats
    if (pending_callbacks_.empty()) {
        stats_.idle_cycles++;
    } else {
        stats_.busy_cycles++;
    }
}

void BehavioralMemoryController::drain() {
    // Process all pending callbacks
    while (!pending_callbacks_.empty()) {
        tick();
    }
}

void BehavioralMemoryController::reset() {
    current_cycle_ = 0;
    next_request_id_ = 1;

    // Clear pending callbacks
    while (!pending_callbacks_.empty()) {
        pending_callbacks_.pop();
    }

    // Clear backing store
    pages_.clear();

    // Reset statistics
    stats_.reset();
}

void BehavioralMemoryController::read_direct(uint64_t address, void* data, uint32_t size) const {
    if (!data || size == 0) return;

    uint8_t* dst = static_cast<uint8_t*>(data);
    uint64_t remaining = size;
    uint64_t current_addr = address;

    while (remaining > 0) {
        uint64_t page_addr = (current_addr / PAGE_SIZE) * PAGE_SIZE;
        size_t page_offset = current_addr % PAGE_SIZE;
        size_t bytes_in_page = std::min(remaining, static_cast<uint64_t>(PAGE_SIZE - page_offset));

        const std::vector<uint8_t>* page = find_page(page_addr);
        if (page) {
            std::memcpy(dst, page->data() + page_offset, bytes_in_page);
        } else {
            // Unwritten memory reads as zero
            std::memset(dst, 0, bytes_in_page);
        }

        dst += bytes_in_page;
        current_addr += bytes_in_page;
        remaining -= bytes_in_page;
    }
}

void BehavioralMemoryController::write_direct(uint64_t address, const void* data, uint32_t size) {
    if (!data || size == 0) return;

    const uint8_t* src = static_cast<const uint8_t*>(data);
    uint64_t remaining = size;
    uint64_t current_addr = address;

    while (remaining > 0) {
        uint64_t page_addr = (current_addr / PAGE_SIZE) * PAGE_SIZE;
        size_t page_offset = current_addr % PAGE_SIZE;
        size_t bytes_in_page = std::min(remaining, static_cast<uint64_t>(PAGE_SIZE - page_offset));

        std::vector<uint8_t>& page = get_page(page_addr);
        std::memcpy(page.data() + page_offset, src, bytes_in_page);

        src += bytes_in_page;
        current_addr += bytes_in_page;
        remaining -= bytes_in_page;
    }
}

std::vector<uint8_t>& BehavioralMemoryController::get_page(uint64_t page_addr) {
    auto it = pages_.find(page_addr);
    if (it == pages_.end()) {
        // Allocate new page, initialized to zero
        auto [new_it, _] = pages_.emplace(page_addr, std::vector<uint8_t>(PAGE_SIZE, 0));
        return new_it->second;
    }
    return it->second;
}

const std::vector<uint8_t>* BehavioralMemoryController::find_page(uint64_t page_addr) const {
    auto it = pages_.find(page_addr);
    return it != pages_.end() ? &it->second : nullptr;
}

} // namespace sw::kpu
