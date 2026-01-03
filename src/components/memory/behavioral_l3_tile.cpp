// ============================================================================
// src/components/memory/behavioral_l3_tile.cpp
// Behavioral (functional) L3 tile implementation
// ============================================================================

#include <sw/kpu/components/memory/behavioral_l3_tile.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sw::kpu {

// ============================================================================
// Constructor / Reset
// ============================================================================

BehavioralL3Tile::BehavioralL3Tile(const L3TileConfig& config, uint32_t tile_id)
    : config_(config)
    , tile_id_(tile_id)
    , capacity_(static_cast<uint64_t>(config.capacity_kb) * 1024)
{
    memory_.resize(capacity_, 0);
}

void BehavioralL3Tile::reset() {
    current_cycle_ = 0;
    next_request_id_ = 1;

    // Clear pending callbacks
    while (!pending_callbacks_.empty()) {
        pending_callbacks_.pop();
    }

    // Clear memory
    std::fill(memory_.begin(), memory_.end(), 0);

    // Reset statistics
    stats_.reset();
}

// ============================================================================
// Basic Memory Operations
// ============================================================================

std::optional<uint64_t> BehavioralL3Tile::read(
    uint64_t addr,
    void* data,
    uint32_t size,
    std::function<void()> callback)
{
    if (!validate_address(addr, size)) {
        return std::nullopt;
    }

    // Perform read immediately
    std::memcpy(data, memory_.data() + addr, size);

    uint64_t request_id = next_request_id_++;

    // Update statistics
    stats_.reads++;
    stats_.bytes_read += size;

    // Handle callback with fixed latency
    if (callback) {
        uint64_t latency = config_.access_latency_cycles;
        pending_callbacks_.push(PendingCallback{
            .completion_cycle = current_cycle_ + latency,
            .callback = callback
        });
        stats_.total_read_latency += latency;
        stats_.min_latency = std::min(stats_.min_latency, latency);
        stats_.max_latency = std::max(stats_.max_latency, latency);
    }

    return request_id;
}

std::optional<uint64_t> BehavioralL3Tile::write(
    uint64_t addr,
    const void* data,
    uint32_t size,
    std::function<void()> callback)
{
    if (!validate_address(addr, size)) {
        return std::nullopt;
    }

    // Perform write immediately
    std::memcpy(memory_.data() + addr, data, size);

    uint64_t request_id = next_request_id_++;

    // Update statistics
    stats_.writes++;
    stats_.bytes_written += size;

    // Handle callback with fixed latency
    if (callback) {
        uint64_t latency = config_.access_latency_cycles;
        pending_callbacks_.push(PendingCallback{
            .completion_cycle = current_cycle_ + latency,
            .callback = callback
        });
        stats_.total_write_latency += latency;
        stats_.min_latency = std::min(stats_.min_latency, latency);
        stats_.max_latency = std::max(stats_.max_latency, latency);
    }

    return request_id;
}

// ============================================================================
// Block Operations
// ============================================================================

std::optional<uint64_t> BehavioralL3Tile::read_block(
    uint64_t base_addr,
    void* data,
    uint32_t height,
    uint32_t width,
    uint32_t element_size,
    uint32_t stride,
    std::function<void()> callback)
{
    if (stride == 0) {
        stride = width * element_size;
    }

    uint32_t row_bytes = width * element_size;
    uint8_t* dest = static_cast<uint8_t*>(data);

    // Validate entire block
    uint64_t max_addr = base_addr + (height - 1) * stride + row_bytes;
    if (max_addr > capacity_) {
        return std::nullopt;
    }

    // Read each row
    for (uint32_t row = 0; row < height; ++row) {
        uint64_t src_addr = base_addr + row * stride;
        std::memcpy(dest + row * row_bytes, memory_.data() + src_addr, row_bytes);
    }

    uint64_t request_id = next_request_id_++;

    // Update statistics
    stats_.block_reads++;
    stats_.bytes_read += height * row_bytes;

    // Handle callback
    if (callback) {
        uint64_t latency = config_.access_latency_cycles * height;
        pending_callbacks_.push(PendingCallback{
            .completion_cycle = current_cycle_ + latency,
            .callback = callback
        });
    }

    return request_id;
}

std::optional<uint64_t> BehavioralL3Tile::write_block(
    uint64_t base_addr,
    const void* data,
    uint32_t height,
    uint32_t width,
    uint32_t element_size,
    uint32_t stride,
    std::function<void()> callback)
{
    if (stride == 0) {
        stride = width * element_size;
    }

    uint32_t row_bytes = width * element_size;
    const uint8_t* src = static_cast<const uint8_t*>(data);

    // Validate entire block
    uint64_t max_addr = base_addr + (height - 1) * stride + row_bytes;
    if (max_addr > capacity_) {
        return std::nullopt;
    }

    // Write each row
    for (uint32_t row = 0; row < height; ++row) {
        uint64_t dst_addr = base_addr + row * stride;
        std::memcpy(memory_.data() + dst_addr, src + row * row_bytes, row_bytes);
    }

    uint64_t request_id = next_request_id_++;

    // Update statistics
    stats_.block_writes++;
    stats_.bytes_written += height * row_bytes;

    // Handle callback
    if (callback) {
        uint64_t latency = config_.access_latency_cycles * height;
        pending_callbacks_.push(PendingCallback{
            .completion_cycle = current_cycle_ + latency,
            .callback = callback
        });
    }

    return request_id;
}

// ============================================================================
// Simulation Interface
// ============================================================================

void BehavioralL3Tile::tick() {
    current_cycle_++;

    // Process completed callbacks
    while (!pending_callbacks_.empty() &&
           pending_callbacks_.top().completion_cycle <= current_cycle_) {

        PendingCallback cb = pending_callbacks_.top();
        pending_callbacks_.pop();

        if (cb.callback) {
            cb.callback();
        }
    }

    // Update utilization stats
    if (!pending_callbacks_.empty()) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void BehavioralL3Tile::drain() {
    while (has_pending()) {
        tick();
    }
}

// ============================================================================
// Direct Memory Access
// ============================================================================

void BehavioralL3Tile::read_direct(uint64_t addr, void* data, uint32_t size) const {
    if (addr + size <= capacity_) {
        std::memcpy(data, memory_.data() + addr, size);
    }
}

void BehavioralL3Tile::write_direct(uint64_t addr, const void* data, uint32_t size) {
    if (addr + size <= capacity_) {
        std::memcpy(memory_.data() + addr, data, size);
    }
}

// ============================================================================
// Helpers
// ============================================================================

bool BehavioralL3Tile::validate_address(uint64_t addr, uint32_t size) const {
    return addr + size <= capacity_;
}

} // namespace sw::kpu
