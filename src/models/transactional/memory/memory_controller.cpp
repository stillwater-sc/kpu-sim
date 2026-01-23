// ============================================================================
// src/components/memory/transactional_memory_controller.cpp
// Transactional (queue-based) memory controller implementation
// ============================================================================

#include <sw/kpu/models/transactional/memory/memory_controller.hpp>
#include <sw/xue/event_collector.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sw::kpu {

// ============================================================================
// Constructor / Reset
// ============================================================================

TransactionalMemoryController::TransactionalMemoryController(
    const MemoryControllerConfig& config)
    : config_(config)
    , rng_(std::random_device{}())
{
    // Calculate capacity from address bits
    // capacity = 2^(row_bits + col_bits + bank_bits + channel_bits) * burst_size
    // Default to 1GB if that seems too small
    uint64_t computed_capacity =
        (1ULL << (config.row_bits + config.col_bits + config.bank_bits + config.channel_bits))
        * config.timing.tBurst;
    capacity_ = std::max(computed_capacity, static_cast<uint64_t>(1) << 30);  // At least 1GB

    // Initialize per-bank state
    banks_.resize(config.num_channels);
    for (auto& channel : banks_) {
        channel.resize(config.banks_per_channel);
    }
}

void TransactionalMemoryController::reset() {
    current_cycle_ = 0;
    next_request_id_ = 1;
    pending_count_ = 0;

    // Clear completion queue
    while (!completion_queue_.empty()) {
        completion_queue_.pop();
    }

    // Reset bank state
    for (auto& channel : banks_) {
        for (auto& bank : channel) {
            bank.busy_until_cycle = 0;
            bank.open_row = UINT32_MAX;
            bank.queue.clear();
        }
    }

    // Reset last operation tracking
    last_op_type_ = MemoryRequestType::READ;
    last_op_cycle_ = 0;

    // Clear backing store
    pages_.clear();

    // Reset statistics
    stats_.reset();

    // Clear violations
    violations_.clear();
}

// ============================================================================
// Request Interface
// ============================================================================

std::optional<uint64_t> TransactionalMemoryController::submit_read(
    uint64_t address,
    uint32_t size,
    std::function<void()> callback)
{
    return submit_request(address, size, MemoryRequestType::READ, nullptr, callback);
}

std::optional<uint64_t> TransactionalMemoryController::submit_write(
    uint64_t address,
    const void* data,
    uint32_t size,
    std::function<void()> callback)
{
    return submit_request(address, size, MemoryRequestType::WRITE, data, callback);
}

std::optional<uint64_t> TransactionalMemoryController::submit_request(
    uint64_t address,
    uint32_t size,
    MemoryRequestType type,
    const void* data,
    std::function<void()> callback)
{
    // Check queue depth limit
    if (config_.queue_depth > 0 && pending_count_ >= config_.queue_depth) {
        return std::nullopt;
    }

    // Decode address
    uint8_t channel, bank;
    uint32_t row;
    decode_address(address, channel, bank, row);

    // Validate channel/bank
    if (channel >= config_.num_channels || bank >= config_.banks_per_channel) {
        if (invariants_enabled_) {
            report_violation("ADDR_BOUNDS", "Address maps to invalid channel/bank",
                           channel, bank);
        }
        return std::nullopt;
    }

    // Calculate when this request can start (after bank becomes free)
    auto& bank_info = banks_[channel][bank];
    uint64_t start_cycle = std::max(current_cycle_, bank_info.busy_until_cycle);

    // Note: busy_until_cycle already tracks bank serialization, so no additional
    // queueing delay needed here. The queue is for tracking pending request IDs.

    // Calculate latency (includes row buffer effects)
    uint32_t latency = calculate_latency(channel, bank, row, type);

    // Sample with variance
    latency = sample_latency(latency);

    // Complete cycle
    uint64_t completion_cycle = start_cycle + latency;

    // Create request
    uint64_t request_id = next_request_id_++;
    PendingRequest req{
        .request_id = request_id,
        .completion_cycle = completion_cycle,
        .submit_cycle = current_cycle_,
        .channel = channel,
        .bank = bank,
        .type = type,
        .callback = callback
    };

    // Add to completion queue
    completion_queue_.push(req);
    pending_count_++;

    // Track in bank queue
    bank_info.queue.push_back(request_id);

    // Update bank busy time
    bank_info.busy_until_cycle = completion_cycle;

    // Update open row (simplified: assume row stays open after access)
    bank_info.open_row = row;

    // Handle write data
    if (type == MemoryRequestType::WRITE && data != nullptr) {
        write_direct(address, data, size);
    }

    // Update statistics
    if (type == MemoryRequestType::READ) {
        stats_.reads++;
    } else {
        stats_.writes++;
    }

    // Track turnarounds
    if (last_op_cycle_ == current_cycle_ - 1 && last_op_type_ != type) {
        if (type == MemoryRequestType::WRITE) {
            stats_.read_to_write_turnarounds++;
        } else {
            stats_.write_to_read_turnarounds++;
        }
    }
    last_op_type_ = type;
    last_op_cycle_ = current_cycle_;

    return request_id;
}

bool TransactionalMemoryController::can_accept() const {
    if (config_.queue_depth == 0) return true;
    return pending_count_ < config_.queue_depth;
}

// ============================================================================
// Simulation Interface
// ============================================================================

void TransactionalMemoryController::tick() {
    current_cycle_++;

    // Process completed requests
    while (!completion_queue_.empty() &&
           completion_queue_.top().completion_cycle <= current_cycle_) {

        PendingRequest req = completion_queue_.top();
        completion_queue_.pop();
        pending_count_--;

        // Remove from bank queue
        auto& bank_info = banks_[req.channel][req.bank];
        auto it = std::find(bank_info.queue.begin(), bank_info.queue.end(),
                           req.request_id);
        if (it != bank_info.queue.end()) {
            bank_info.queue.erase(it);
        }

        // Update latency statistics
        uint64_t latency = req.completion_cycle - req.submit_cycle;
        stats_.total_latency += latency;
        stats_.min_latency = std::min(stats_.min_latency, latency);
        stats_.max_latency = std::max(stats_.max_latency, latency);

        // Record XUE event on completion with actual latency
        sw::xue::xue().set_cycle(current_cycle_);
        // Use burst size as approximate bytes (actual size not tracked in PendingRequest)
        uint64_t bytes = config_.timing.tBurst * 8;  // Burst length * 8 bytes per beat
        if (req.type == MemoryRequestType::READ) {
            sw::xue::xue().record_dram_read(bytes, latency);
        } else {
            sw::xue::xue().record_dram_write(bytes, latency);
        }

        // Invoke callback
        if (req.callback) {
            req.callback();
        }
    }

    // Update utilization stats
    if (pending_count_ > 0) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void TransactionalMemoryController::drain() {
    while (has_pending()) {
        tick();
    }
}

// ============================================================================
// Bank State Queries
// ============================================================================

IMemoryController::BankState TransactionalMemoryController::get_bank_state(
    uint8_t channel, uint8_t bank) const
{
    if (channel >= config_.num_channels || bank >= config_.banks_per_channel) {
        return BankState::IDLE;
    }

    const auto& bank_info = banks_[channel][bank];
    if (bank_info.is_idle(current_cycle_)) {
        return bank_info.has_row_open() ? BankState::ACTIVE : BankState::IDLE;
    }
    return BankState::ACTIVE;  // Busy = effectively active
}

bool TransactionalMemoryController::is_row_open(
    uint8_t channel, uint8_t bank, uint32_t row) const
{
    if (channel >= config_.num_channels || bank >= config_.banks_per_channel) {
        return false;
    }
    return banks_[channel][bank].open_row == row;
}

// ============================================================================
// Address Decoding
// ============================================================================

void TransactionalMemoryController::decode_address(
    uint64_t address, uint8_t& channel, uint8_t& bank, uint32_t& row) const
{
    // Simple interleaving: low bits select channel, then bank, then row
    // This can be made configurable for different address mapping schemes

    // Cache line size (typically 64 bytes)
    constexpr uint32_t CACHE_LINE_BITS = 6;

    // Extract fields from address after cache line offset
    uint64_t addr_shifted = address >> CACHE_LINE_BITS;

    // Channel interleaving
    uint32_t channel_bits = 0;
    for (uint8_t n = config_.num_channels; n > 1; n >>= 1) channel_bits++;
    channel = addr_shifted & ((1 << channel_bits) - 1);
    addr_shifted >>= channel_bits;

    // Bank interleaving
    uint32_t bank_bits = 0;
    for (uint8_t n = config_.banks_per_channel; n > 1; n >>= 1) bank_bits++;
    bank = addr_shifted & ((1 << bank_bits) - 1);
    addr_shifted >>= bank_bits;

    // Row is the rest
    row = static_cast<uint32_t>(addr_shifted);
}

// ============================================================================
// Latency Calculation
// ============================================================================

uint32_t TransactionalMemoryController::calculate_latency(
    uint8_t channel, uint8_t bank, uint32_t row, MemoryRequestType type)
{
    const auto& bank_info = banks_[channel][bank];
    const auto& timing = config_.timing;

    // Use physical timing parameters for service time (preferred for throughput modeling)
    // Per-scenario latencies from CA may include queueing delays which double-count
    // contention when combined with the transactional model's per-bank busy tracking.

    uint32_t cas_latency = (type == MemoryRequestType::READ)
                           ? timing.tCL + timing.tBurst
                           : timing.tWL + timing.tBurst;

    uint32_t latency;

    if (!bank_info.has_row_open()) {
        // Page empty - need ACT + CAS
        // Service time: tRCD + CAS latency
        latency = timing.tRCD + cas_latency;
        stats_.page_empty++;
    } else if (bank_info.open_row == row) {
        // Page hit - just CAS
        // Service time: CAS latency only
        latency = cas_latency;
        stats_.page_hits++;
    } else {
        // Page conflict - need PRE + ACT + CAS
        // Service time: tRP + tRCD + CAS latency
        latency = timing.tRP + timing.tRCD + cas_latency;
        stats_.page_conflicts++;
    }

    // Ensure minimum latency of 1
    return std::max(1u, latency);
}

uint32_t TransactionalMemoryController::sample_latency(uint32_t base_latency) {
    if (latency_variance_ <= 0.0) {
        return base_latency;
    }

    // Sample from normal distribution with given variance
    double stddev = base_latency * latency_variance_;
    std::normal_distribution<double> dist(base_latency, stddev);

    double sample = dist(rng_);
    // Clamp to positive
    return static_cast<uint32_t>(std::max(1.0, sample));
}

// ============================================================================
// Direct Memory Access
// ============================================================================

void TransactionalMemoryController::read_direct(
    uint64_t address, void* data, uint32_t size) const
{
    uint8_t* dest = static_cast<uint8_t*>(data);

    while (size > 0) {
        uint64_t page_addr = address / PAGE_SIZE * PAGE_SIZE;
        uint32_t page_offset = address % PAGE_SIZE;
        uint32_t chunk_size = std::min(size, static_cast<uint32_t>(PAGE_SIZE - page_offset));

        const std::vector<uint8_t>* page = find_page(page_addr);
        if (page) {
            std::memcpy(dest, page->data() + page_offset, chunk_size);
        } else {
            // Page not written - return zeros
            std::memset(dest, 0, chunk_size);
        }

        address += chunk_size;
        dest += chunk_size;
        size -= chunk_size;
    }
}

void TransactionalMemoryController::write_direct(
    uint64_t address, const void* data, uint32_t size)
{
    const uint8_t* src = static_cast<const uint8_t*>(data);

    while (size > 0) {
        uint64_t page_addr = address / PAGE_SIZE * PAGE_SIZE;
        uint32_t page_offset = address % PAGE_SIZE;
        uint32_t chunk_size = std::min(size, static_cast<uint32_t>(PAGE_SIZE - page_offset));

        std::vector<uint8_t>& page = get_page(page_addr);
        std::memcpy(page.data() + page_offset, src, chunk_size);

        address += chunk_size;
        src += chunk_size;
        size -= chunk_size;
    }
}

std::vector<uint8_t>& TransactionalMemoryController::get_page(uint64_t page_addr) {
    auto it = pages_.find(page_addr);
    if (it == pages_.end()) {
        auto [inserted_it, success] = pages_.emplace(page_addr, std::vector<uint8_t>(PAGE_SIZE, 0));
        return inserted_it->second;
    }
    return it->second;
}

const std::vector<uint8_t>* TransactionalMemoryController::find_page(uint64_t page_addr) const {
    auto it = pages_.find(page_addr);
    return it != pages_.end() ? &it->second : nullptr;
}

// ============================================================================
// Invariant Checking
// ============================================================================

void TransactionalMemoryController::report_violation(
    const std::string& id, const std::string& msg,
    uint8_t channel, uint8_t bank)
{
    violations_.push_back(InvariantViolation{
        .cycle = current_cycle_,
        .invariant_id = id,
        .message = msg,
        .channel = channel,
        .bank = bank
    });
}

} // namespace sw::kpu
