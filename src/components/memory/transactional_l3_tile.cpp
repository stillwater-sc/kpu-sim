// ============================================================================
// src/components/memory/transactional_l3_tile.cpp
// Transactional L3 tile implementation
// ============================================================================

#include <sw/kpu/components/memory/transactional_l3_tile.hpp>

#include <algorithm>
#include <cstring>

namespace sw::kpu {

// ============================================================================
// Constructor / Reset
// ============================================================================

TransactionalL3Tile::TransactionalL3Tile(const L3TileConfig& config, uint32_t tile_id)
    : config_(config)
    , tile_id_(tile_id)
    , capacity_(static_cast<uint64_t>(config.capacity_kb) * 1024)
{
    memory_.resize(capacity_, 0);
    banks_.resize(config.num_banks);
}

void TransactionalL3Tile::reset() {
    current_cycle_ = 0;
    next_request_id_ = 1;
    pending_count_ = 0;

    // Clear completion queue
    while (!completion_queue_.empty()) {
        completion_queue_.pop();
    }

    // Reset banks
    for (auto& bank : banks_) {
        bank.busy_until_cycle = 0;
        bank.state = BankState::IDLE;
    }

    // Clear memory
    std::fill(memory_.begin(), memory_.end(), static_cast<uint8_t>(0));

    // Reset statistics
    stats_.reset();
}

// ============================================================================
// Basic Memory Operations
// ============================================================================

std::optional<uint64_t> TransactionalL3Tile::read(
    uint64_t addr,
    void* data,
    uint32_t size,
    std::function<void()> callback)
{
    if (!validate_address(addr, size)) {
        return std::nullopt;
    }

    // Perform read immediately (data is always available)
    std::memcpy(data, memory_.data() + addr, size);

    uint64_t request_id = next_request_id_++;

    // Determine bank and schedule access
    uint8_t bank = address_to_bank(addr);
    uint64_t completion_cycle = schedule_bank_access(bank, false);

    // Track request
    if (callback) {
        completion_queue_.push(PendingRequest{
            .request_id = request_id,
            .completion_cycle = completion_cycle,
            .submit_cycle = current_cycle_,
            .bank = bank,
            .is_write = false,
            .callback = callback
        });
        pending_count_++;
    }

    // Update statistics
    stats_.reads++;
    stats_.bytes_read += size;
    uint64_t latency = completion_cycle - current_cycle_;
    stats_.total_read_latency += latency;
    stats_.min_latency = std::min(stats_.min_latency, latency);
    stats_.max_latency = std::max(stats_.max_latency, latency);

    return request_id;
}

std::optional<uint64_t> TransactionalL3Tile::write(
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

    // Determine bank and schedule access
    uint8_t bank = address_to_bank(addr);
    uint64_t completion_cycle = schedule_bank_access(bank, true);

    // Track request
    if (callback) {
        completion_queue_.push(PendingRequest{
            .request_id = request_id,
            .completion_cycle = completion_cycle,
            .submit_cycle = current_cycle_,
            .bank = bank,
            .is_write = true,
            .callback = callback
        });
        pending_count_++;
    }

    // Update statistics
    stats_.writes++;
    stats_.bytes_written += size;
    uint64_t latency = completion_cycle - current_cycle_;
    stats_.total_write_latency += latency;
    stats_.min_latency = std::min(stats_.min_latency, latency);
    stats_.max_latency = std::max(stats_.max_latency, latency);

    return request_id;
}

// ============================================================================
// Block Operations
// ============================================================================

std::optional<uint64_t> TransactionalL3Tile::read_block(
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

    // Schedule bank accesses for all touched banks
    uint64_t latest_completion = current_cycle_;
    for (uint32_t row = 0; row < height; ++row) {
        uint64_t row_addr = base_addr + row * stride;
        uint8_t bank = address_to_bank(row_addr);
        uint64_t completion = schedule_bank_access(bank, false);
        latest_completion = std::max(latest_completion, completion);
    }

    // Track request
    if (callback) {
        completion_queue_.push(PendingRequest{
            .request_id = request_id,
            .completion_cycle = latest_completion,
            .submit_cycle = current_cycle_,
            .bank = 0,  // Multiple banks
            .is_write = false,
            .callback = callback
        });
        pending_count_++;
    }

    // Update statistics
    stats_.block_reads++;
    stats_.bytes_read += height * row_bytes;

    return request_id;
}

std::optional<uint64_t> TransactionalL3Tile::write_block(
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

    // Schedule bank accesses for all touched banks
    uint64_t latest_completion = current_cycle_;
    for (uint32_t row = 0; row < height; ++row) {
        uint64_t row_addr = base_addr + row * stride;
        uint8_t bank = address_to_bank(row_addr);
        uint64_t completion = schedule_bank_access(bank, true);
        latest_completion = std::max(latest_completion, completion);
    }

    // Track request
    if (callback) {
        completion_queue_.push(PendingRequest{
            .request_id = request_id,
            .completion_cycle = latest_completion,
            .submit_cycle = current_cycle_,
            .bank = 0,
            .is_write = true,
            .callback = callback
        });
        pending_count_++;
    }

    // Update statistics
    stats_.block_writes++;
    stats_.bytes_written += height * row_bytes;

    return request_id;
}

// ============================================================================
// Status Queries
// ============================================================================

bool TransactionalL3Tile::can_accept() const {
    // Check if any bank is available
    for (const auto& bank : banks_) {
        if (bank.is_idle(current_cycle_)) {
            return true;
        }
    }
    return true;  // Always accept in transactional mode
}

IL3Tile::BankState TransactionalL3Tile::get_bank_state(uint8_t bank) const {
    if (bank >= banks_.size()) {
        return BankState::IDLE;
    }
    return banks_[bank].state;
}

bool TransactionalL3Tile::is_bank_busy(uint8_t bank) const {
    if (bank >= banks_.size()) {
        return false;
    }
    return !banks_[bank].is_idle(current_cycle_);
}

// ============================================================================
// Simulation Interface
// ============================================================================

void TransactionalL3Tile::tick() {
    current_cycle_++;

    // Update bank states
    for (auto& bank : banks_) {
        if (bank.is_idle(current_cycle_)) {
            bank.state = BankState::IDLE;
        }
    }

    // Process completed requests
    while (!completion_queue_.empty() &&
           completion_queue_.top().completion_cycle <= current_cycle_) {

        PendingRequest req = completion_queue_.top();
        completion_queue_.pop();
        pending_count_--;

        // Invoke callback
        if (req.callback) {
            req.callback();
        }
    }

    // Update utilization stats
    bool any_busy = false;
    for (const auto& bank : banks_) {
        if (!bank.is_idle(current_cycle_)) {
            any_busy = true;
            break;
        }
    }

    if (any_busy) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void TransactionalL3Tile::drain() {
    while (has_pending()) {
        tick();
    }
}

// ============================================================================
// Direct Memory Access
// ============================================================================

void TransactionalL3Tile::read_direct(uint64_t addr, void* data, uint32_t size) const {
    if (addr + size <= capacity_) {
        std::memcpy(data, memory_.data() + addr, size);
    }
}

void TransactionalL3Tile::write_direct(uint64_t addr, const void* data, uint32_t size) {
    if (addr + size <= capacity_) {
        std::memcpy(memory_.data() + addr, data, size);
    }
}

// ============================================================================
// Helpers
// ============================================================================

uint8_t TransactionalL3Tile::address_to_bank(uint64_t addr) const {
    // Simple interleaving by bank width
    return static_cast<uint8_t>((addr / config_.bank_width_bytes) % config_.num_banks);
}

bool TransactionalL3Tile::validate_address(uint64_t addr, uint32_t size) const {
    return addr + size <= capacity_;
}

uint64_t TransactionalL3Tile::schedule_bank_access(uint8_t bank, bool is_write) {
    auto& bank_info = banks_[bank];

    // Check for bank conflict
    if (!bank_info.is_idle(current_cycle_)) {
        stats_.bank_conflicts++;
    }

    // Schedule access
    uint64_t start_cycle = std::max(current_cycle_, bank_info.busy_until_cycle);
    uint64_t completion_cycle = start_cycle + config_.access_latency_cycles;

    // Update bank state
    bank_info.busy_until_cycle = completion_cycle;
    bank_info.state = is_write ? BankState::WRITING : BankState::READING;

    return completion_cycle;
}

} // namespace sw::kpu
