// ============================================================================
// src/components/memory/lpddr5_memory_controller.cpp
// LPDDR5 Memory Controller implementation with formal invariant checking
// ============================================================================

#include <sw/kpu/components/lpddr5_memory_controller.hpp>
#include <sw/trace/trace_entry.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <cassert>

namespace sw::kpu::lpddr5 {

// Note: We use fully qualified lpddr5::BankState:: to avoid ambiguity with
// IMemoryController::BankState which is brought in through inheritance.

// ============================================================================
// Constructor
// ============================================================================

LPDDR5MemoryController::LPDDR5MemoryController(const Config& config)
    : lpddr5_config_(config)
{
    // Initialize bank groups for each bank
    for (uint8_t ch = 0; ch < lpddr5_config_.num_channels; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            channels_[ch].banks[b].bank_group = b / 4;
        }
    }

    // Initialize interface config from LPDDR5 config
    interface_config_.fidelity = sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    interface_config_.technology = sw::kpu::MemoryTechnology::LPDDR5;
    interface_config_.verification = sw::kpu::VerificationLevel::INVARIANTS;
    interface_config_.num_channels = config.num_channels;
    interface_config_.banks_per_channel = config.banks_per_channel;
    interface_config_.bank_groups = config.bank_groups;
    interface_config_.queue_depth = config.queue_depth;
    interface_config_.row_bits = config.row_bits;
    interface_config_.col_bits = config.col_bits;
    interface_config_.bank_bits = config.bank_bits;
    interface_config_.channel_bits = config.channel_bits;
}

LPDDR5MemoryController::LPDDR5MemoryController(const sw::kpu::MemoryControllerConfig& config)
    : LPDDR5MemoryController(Config{})  // Start with defaults
{
    // Copy relevant fields from interface config
    lpddr5_config_.num_channels = config.num_channels;
    lpddr5_config_.banks_per_channel = config.banks_per_channel;
    lpddr5_config_.bank_groups = config.bank_groups;
    lpddr5_config_.queue_depth = config.queue_depth;
    lpddr5_config_.row_bits = config.row_bits;
    lpddr5_config_.col_bits = config.col_bits;
    lpddr5_config_.bank_bits = config.bank_bits;
    lpddr5_config_.channel_bits = config.channel_bits;

    // Copy timing parameters
    lpddr5_config_.timing.tRCD = config.timing.tRCD;
    lpddr5_config_.timing.tRP = config.timing.tRP;
    lpddr5_config_.timing.tRAS = config.timing.tRAS;
    lpddr5_config_.timing.tRC = config.timing.tRC;
    lpddr5_config_.timing.tCL = config.timing.tCL;
    lpddr5_config_.timing.tWL = config.timing.tWL;
    lpddr5_config_.timing.tWR = config.timing.tWR;
    lpddr5_config_.timing.tRTP = config.timing.tRTP;
    lpddr5_config_.timing.tRRD_L = config.timing.tRRD_L;
    lpddr5_config_.timing.tRRD_S = config.timing.tRRD_S;
    lpddr5_config_.timing.tCCD_L = config.timing.tCCD_L;
    lpddr5_config_.timing.tCCD_S = config.timing.tCCD_S;
    lpddr5_config_.timing.tWTR_L = config.timing.tWTR_L;
    lpddr5_config_.timing.tWTR_S = config.timing.tWTR_S;
    lpddr5_config_.timing.tRTW = config.timing.tRTW;
    lpddr5_config_.timing.tFAW = config.timing.tFAW;

    // Store interface config
    interface_config_ = config;
    interface_config_.fidelity = sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    interface_config_.technology = sw::kpu::MemoryTechnology::LPDDR5;

    // Set verification level based on config
    check_invariants_ = (config.verification == sw::kpu::VerificationLevel::INVARIANTS ||
                        config.verification == sw::kpu::VerificationLevel::PROTOCOL);
    tracing_enabled_ = config.enable_tracing;

    // Reinitialize bank groups
    for (uint8_t ch = 0; ch < lpddr5_config_.num_channels; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            channels_[ch].banks[b].bank_group = b / 4;
        }
    }
}

// ============================================================================
// Request submission
// ============================================================================

std::optional<uint64_t> LPDDR5MemoryController::submit_read(
    uint64_t address, uint32_t size, std::function<void()> callback)
{
    if (!can_accept()) {
        return std::nullopt;
    }

    MemoryRequest req;
    req.id = next_request_id_++;
    req.type = RequestType::READ;
    req.address = address;
    req.size = size;
    req.submit_cycle = current_cycle_;
    req.callback = std::move(callback);
    req.data.resize(size);

    decode_address(address, req.channel, req.bank, req.row, req.col);

    pending_queue_.push(std::move(req));
    stats_.reads++;

    return req.id;
}

std::optional<uint64_t> LPDDR5MemoryController::submit_write(
    uint64_t address, const void* data, uint32_t size, std::function<void()> callback)
{
    if (!can_accept()) {
        return std::nullopt;
    }

    MemoryRequest req;
    req.id = next_request_id_++;
    req.type = RequestType::WRITE;
    req.address = address;
    req.size = size;
    req.submit_cycle = current_cycle_;
    req.callback = std::move(callback);

    req.data.resize(size);
    if (data) {
        std::memcpy(req.data.data(), data, size);
    }

    decode_address(address, req.channel, req.bank, req.row, req.col);

    pending_queue_.push(std::move(req));
    stats_.writes++;

    return req.id;
}

// ============================================================================
// Simulation interface
// ============================================================================

void LPDDR5MemoryController::tick() {
    // Check invariants at start of cycle
    if (check_invariants_) {
        check_all_invariants();
    }

    // Update state machines
    update_bank_states();
    update_bus_states();

    // Handle refresh
    handle_refresh();

    // Schedule new requests
    schedule_requests();

    // Complete finished requests
    complete_requests();

    // Finalize bus traces - only trace IDLE if bus is still IDLE after scheduling
    // This avoids spurious BUSY->IDLE->BUSY transitions on the same cycle
    finalize_bus_traces();

    current_cycle_++;

    // Sync interface stats (lazily - only sync when accessed would be more efficient,
    // but syncing every tick ensures consistency)
    sync_interface_stats();
    if (check_invariants_ && !violations_.empty()) {
        sync_interface_violations();
    }
}

void LPDDR5MemoryController::drain() {
    const uint64_t MAX_DRAIN_CYCLES = 100000;
    uint64_t start = current_cycle_;

    while (has_pending() && (current_cycle_ - start) < MAX_DRAIN_CYCLES) {
        tick();
    }
}

void LPDDR5MemoryController::reset() {
    for (uint8_t ch = 0; ch < 2; ++ch) {
        channels_[ch] = Channel{};
        for (uint8_t b = 0; b < 16; ++b) {
            channels_[ch].banks[b].bank_group = b / 4;
        }
    }

    while (!pending_queue_.empty()) pending_queue_.pop();
    active_requests_.clear();

    current_cycle_ = 0;
    next_request_id_ = 1;
    stats_ = Statistics{};
    interface_stats_.reset();
    violations_.clear();
    interface_violations_.clear();
}

// ============================================================================
// State queries
// ============================================================================

bool LPDDR5MemoryController::has_pending() const {
    return !pending_queue_.empty() || !active_requests_.empty();
}

bool LPDDR5MemoryController::can_accept() const {
    return pending_queue_.size() < lpddr5_config_.queue_depth;
}

size_t LPDDR5MemoryController::pending_count() const {
    return pending_queue_.size() + active_requests_.size();
}

// Convert LPDDR5 BankState to IMemoryController::BankState
static sw::kpu::IMemoryController::BankState to_interface_state(lpddr5::BankState state) {
    switch (state) {
        case lpddr5::BankState::IDLE:        return sw::kpu::IMemoryController::BankState::IDLE;
        case lpddr5::BankState::ACTIVATING:  return sw::kpu::IMemoryController::BankState::ACTIVATING;
        case lpddr5::BankState::ACTIVE:      return sw::kpu::IMemoryController::BankState::ACTIVE;
        case lpddr5::BankState::READING:     return sw::kpu::IMemoryController::BankState::READING;
        case lpddr5::BankState::WRITING:     return sw::kpu::IMemoryController::BankState::WRITING;
        case lpddr5::BankState::PRECHARGING: return sw::kpu::IMemoryController::BankState::PRECHARGING;
        case lpddr5::BankState::REFRESHING:  return sw::kpu::IMemoryController::BankState::REFRESHING;
        default:                             return sw::kpu::IMemoryController::BankState::IDLE;
    }
}

sw::kpu::IMemoryController::BankState LPDDR5MemoryController::get_bank_state(
    uint8_t channel, uint8_t bank) const
{
    return to_interface_state(get_lpddr5_bank_state(channel, bank));
}

lpddr5::BankState LPDDR5MemoryController::get_lpddr5_bank_state(uint8_t channel, uint8_t bank) const {
    if (channel >= lpddr5_config_.num_channels || bank >= 16) {
        return lpddr5::BankState::IDLE;
    }
    return channels_[channel].banks[bank].state;
}

bool LPDDR5MemoryController::is_bank_idle(uint8_t channel, uint8_t bank) const {
    return get_lpddr5_bank_state(channel, bank) == lpddr5::BankState::IDLE;
}

bool LPDDR5MemoryController::is_row_open(uint8_t channel, uint8_t bank, uint32_t row) const {
    if (channel >= lpddr5_config_.num_channels || bank >= 16) {
        return false;
    }
    const Bank& b = channels_[channel].banks[bank];
    return (b.state == lpddr5::BankState::ACTIVE ||
            b.state == lpddr5::BankState::READING ||
            b.state == lpddr5::BankState::WRITING) &&
           b.open_row == row;
}

void LPDDR5MemoryController::reset_stats() {
    stats_ = Statistics{};
    interface_stats_.reset();
}

void LPDDR5MemoryController::sync_interface_stats() {
    interface_stats_.reads = stats_.reads;
    interface_stats_.writes = stats_.writes;
    interface_stats_.page_hits = stats_.page_hits;
    interface_stats_.page_empty = stats_.page_empty;
    interface_stats_.page_conflicts = stats_.page_conflicts;
    interface_stats_.total_latency = stats_.total_latency;
    interface_stats_.stall_cycles = stats_.stall_cycles;
    interface_stats_.refreshes = stats_.refreshes;
    interface_stats_.read_to_write_turnarounds = stats_.read_to_write_turnarounds;
    interface_stats_.write_to_read_turnarounds = stats_.write_to_read_turnarounds;
}

void LPDDR5MemoryController::sync_interface_violations() {
    interface_violations_.clear();
    for (const auto& v : violations_) {
        interface_violations_.push_back({
            v.cycle,
            v.invariant_id,
            v.message,
            v.channel,
            v.bank
        });
    }
}

// ============================================================================
// Address decoding
// ============================================================================

void LPDDR5MemoryController::decode_address(
    uint64_t address, uint8_t& channel, uint8_t& bank,
    uint32_t& row, uint32_t& col) const
{
    // Simple interleaved address mapping:
    // [row | bank | col | channel | byte_offset]
    uint32_t byte_offset_bits = 6;  // 64-byte cache line
    uint32_t col_bits = lpddr5_config_.col_bits;
    uint32_t bank_bits = lpddr5_config_.bank_bits;
    uint32_t channel_bits = (lpddr5_config_.num_channels > 1) ? 1 : 0;

    uint64_t addr = address >> byte_offset_bits;

    if (channel_bits > 0) {
        channel = addr & ((1 << channel_bits) - 1);
        addr >>= channel_bits;
    } else {
        channel = 0;
    }

    col = addr & ((1 << col_bits) - 1);
    addr >>= col_bits;

    bank = addr & ((1 << bank_bits) - 1);
    addr >>= bank_bits;

    row = addr & ((1 << lpddr5_config_.row_bits) - 1);
}

// ============================================================================
// Burst cycles
// ============================================================================

uint32_t LPDDR5MemoryController::burst_cycles() const {
    return (lpddr5_config_.burst_length == BurstLength::BL16) ?
           lpddr5_config_.timing.tBurst_BL16 : lpddr5_config_.timing.tBurst_BL32;
}

// ============================================================================
// tFAW tracking
// ============================================================================

bool LPDDR5MemoryController::check_tFAW(uint8_t channel) const {
    const auto& ch = channels_[channel];
    const auto& timing = lpddr5_config_.timing;

    // Find oldest activate in window
    uint64_t oldest = ch.activate_window[0];
    for (int i = 1; i < 4; ++i) {
        if (ch.activate_window[i] < oldest && ch.activate_window[i] > 0) {
            oldest = ch.activate_window[i];
        }
    }

    // If oldest activate is within tFAW, we can't issue another
    if (oldest > 0 && current_cycle_ < oldest + timing.tFAW) {
        // Count activates in window
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            if (ch.activate_window[i] > 0 &&
                current_cycle_ - ch.activate_window[i] < timing.tFAW) {
                count++;
            }
        }
        if (count >= 4) {
            return false;
        }
    }
    return true;
}

void LPDDR5MemoryController::record_activate(uint8_t channel) {
    auto& ch = channels_[channel];
    ch.activate_window[ch.activate_index] = current_cycle_;
    ch.activate_index = (ch.activate_index + 1) % 4;
}

// ============================================================================
// Bank operation checks
// ============================================================================

bool LPDDR5MemoryController::can_activate(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    // INV-BANK-1: Bank must be IDLE
    if (b.state != lpddr5::BankState::IDLE) {
        return false;
    }

    // INV-BANK-3: tRP must have elapsed since last precharge
    if (cycle < b.state_until) {
        return false;
    }

    // tRC: time since last activate on same bank
    if (cycle < b.last_activate + timing.tRC) {
        return false;
    }

    // Bank group timing
    uint8_t bg = bank / 4;
    const BankGroup& bank_group = ch.bank_groups[bg];

    // tRRD_L: same bank group
    if (cycle < bank_group.last_activate + timing.tRRD_L) {
        return false;
    }

    // Check tFAW
    if (!check_tFAW(channel)) {
        return false;
    }

    // Command bus available
    if (ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool LPDDR5MemoryController::can_read(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    // INV-BANK-7: Bank must be ACTIVE
    if (b.state != lpddr5::BankState::ACTIVE) {
        return false;
    }

    // INV-BANK-2: tRCD must have elapsed
    if (cycle < b.state_until) {
        return false;
    }

    // Bank group CAS timing
    uint8_t bg = bank / 4;
    const BankGroup& bank_group = ch.bank_groups[bg];
    if (cycle < bank_group.last_cas + timing.tCCD_L) {
        return false;
    }

    // Write-to-read turnaround (tWTR)
    if (ch.last_was_write) {
        if (cycle < bank_group.last_write + timing.tWL + burst_cycles() + timing.tWTR_L) {
            return false;
        }
    }

    // Data bus available
    if (ch.data_bus_state != DataBusState::IDLE) {
        return false;
    }

    // Command bus available
    if (ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool LPDDR5MemoryController::can_write(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    // INV-BANK-7: Bank must be ACTIVE
    if (b.state != lpddr5::BankState::ACTIVE) {
        return false;
    }

    // INV-BANK-2: tRCD must have elapsed
    if (cycle < b.state_until) {
        return false;
    }

    // Bank group CAS timing
    uint8_t bg = bank / 4;
    const BankGroup& bank_group = ch.bank_groups[bg];
    if (cycle < bank_group.last_cas + timing.tCCD_L) {
        return false;
    }

    // Read-to-write turnaround (tRTW)
    if (!ch.last_was_write && ch.data_bus_until > 0) {
        if (cycle < ch.data_bus_until + timing.tRTW) {
            return false;
        }
    }

    // Data bus available
    if (ch.data_bus_state != DataBusState::IDLE) {
        return false;
    }

    // Command bus available
    if (ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool LPDDR5MemoryController::can_precharge(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    // INV-BANK-8: Bank must be ACTIVE (not during burst)
    if (b.state != lpddr5::BankState::ACTIVE) {
        return false;
    }

    // INV-BANK-4: tRAS must have elapsed
    if (cycle < b.last_activate + timing.tRAS) {
        return false;
    }

    // INV-BANK-5: tWR must have elapsed after write
    if (b.last_write_cmd > 0) {
        uint64_t write_done = b.last_write_cmd + timing.tWL + burst_cycles();
        if (cycle < write_done + timing.tWR) {
            return false;
        }
    }

    // INV-BANK-6: tRTP must have elapsed after read
    if (b.last_read_cmd > 0) {
        if (cycle < b.last_read_cmd + timing.tRTP) {
            return false;
        }
    }

    return true;
}

bool LPDDR5MemoryController::can_refresh(uint8_t channel, uint8_t bank) const {
    // INV-BANK-9: Bank must be IDLE
    return channels_[channel].banks[bank].state == lpddr5::BankState::IDLE;
}

// ============================================================================
// Bank operations
// ============================================================================

void LPDDR5MemoryController::do_activate(uint8_t channel, uint8_t bank, uint32_t row) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    b.state = lpddr5::BankState::ACTIVATING;
    b.open_row = row;
    b.state_until = current_cycle_ + timing.tRCD;
    b.last_activate = current_cycle_;

    // Update bank group
    uint8_t bg = bank / 4;
    ch.bank_groups[bg].last_activate = current_cycle_;

    // Record for tFAW
    record_activate(channel);

    // Command bus busy for 1 cycle
    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation
    trace_bank_state_change(channel, bank, lpddr5::BankState::ACTIVATING, "ACTIVATE row " + std::to_string(row));
    trace_command(channel, bank, "ACTIVATE", timing.tRCD, 0);
    trace_bus_state_change(channel, false, "BUSY", "ACT cmd");
}

void LPDDR5MemoryController::do_read(uint8_t channel, uint8_t bank, MemoryRequest& req) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    b.state = lpddr5::BankState::READING;
    b.last_read_cmd = current_cycle_;
    b.burst_end = current_cycle_ + timing.tCL + burst_cycles();

    // Update bank group
    uint8_t bg = bank / 4;
    ch.bank_groups[bg].last_cas = current_cycle_;

    // Track turnarounds (must check BEFORE updating last_was_write)
    if (ch.last_was_write) {
        stats_.write_to_read_turnarounds++;
    }

    // Data bus will be busy after tCL
    ch.data_bus_state = DataBusState::READ_BURST;
    ch.data_bus_until = b.burst_end;
    ch.last_was_write = false;

    // Command bus busy for 1 cycle
    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    // Set request completion time
    req.issue_cycle = current_cycle_;
    req.complete_cycle = b.burst_end;

    // Trace the operation
    uint64_t duration = timing.tCL + burst_cycles();
    trace_bank_state_change(channel, bank, lpddr5::BankState::READING, "READ burst");
    trace_command(channel, bank, "READ", duration, req.id);
    trace_bus_state_change(channel, true, "BUSY", "READ burst");
    trace_bus_state_change(channel, false, "BUSY", "RD cmd");
}

void LPDDR5MemoryController::do_write(uint8_t channel, uint8_t bank, MemoryRequest& req) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    b.state = lpddr5::BankState::WRITING;
    b.last_write_cmd = current_cycle_;
    b.burst_end = current_cycle_ + timing.tWL + burst_cycles();

    // Update bank group
    uint8_t bg = bank / 4;
    ch.bank_groups[bg].last_cas = current_cycle_;
    ch.bank_groups[bg].last_write = current_cycle_;

    // Data bus busy after tWL
    ch.data_bus_state = DataBusState::WRITE_BURST;
    ch.data_bus_until = b.burst_end;

    // Track turnarounds
    if (!ch.last_was_write) {
        stats_.read_to_write_turnarounds++;
    }
    ch.last_was_write = true;

    // Command bus busy for 1 cycle
    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    // Set request completion time
    req.issue_cycle = current_cycle_;
    req.complete_cycle = b.burst_end;

    // Trace the operation
    uint64_t duration = timing.tWL + burst_cycles();
    trace_bank_state_change(channel, bank, lpddr5::BankState::WRITING, "WRITE burst");
    trace_command(channel, bank, "WRITE", duration, req.id);
    trace_bus_state_change(channel, true, "BUSY", "WRITE burst");
    trace_bus_state_change(channel, false, "BUSY", "WR cmd");
}

void LPDDR5MemoryController::do_precharge(uint8_t channel, uint8_t bank) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    b.state = lpddr5::BankState::PRECHARGING;
    b.state_until = current_cycle_ + timing.tRP;

    // Command bus busy for 1 cycle
    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation
    trace_bank_state_change(channel, bank, lpddr5::BankState::PRECHARGING, "PRECHARGE");
    trace_command(channel, bank, "PRECHARGE", timing.tRP, 0);
    trace_bus_state_change(channel, false, "BUSY", "PRE cmd");
}

void LPDDR5MemoryController::do_refresh(uint8_t channel, uint8_t bank) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = lpddr5_config_.timing;

    b.state = lpddr5::BankState::REFRESHING;
    b.state_until = current_cycle_ + timing.tRFCpb;
    b.last_refresh = current_cycle_;

    stats_.refreshes++;

    // Command bus busy for 1 cycle
    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation
    trace_bank_state_change(channel, bank, lpddr5::BankState::REFRESHING, "REFRESH");
    trace_command(channel, bank, "REFRESH", timing.tRFCpb, 0);
    trace_bus_state_change(channel, false, "BUSY", "REF cmd");
}

// ============================================================================
// State updates
// ============================================================================

void LPDDR5MemoryController::update_bank_states() {
    for (uint8_t ch = 0; ch < lpddr5_config_.num_channels; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            Bank& bank = channels_[ch].banks[b];

            if (current_cycle_ >= bank.state_until) {
                switch (bank.state) {
                    case lpddr5::BankState::ACTIVATING:
                        bank.state = lpddr5::BankState::ACTIVE;
                        trace_bank_state_change(ch, b, lpddr5::BankState::ACTIVE, "tRCD complete");
                        break;
                    case lpddr5::BankState::PRECHARGING:
                        bank.state = lpddr5::BankState::IDLE;
                        bank.open_row = 0;
                        bank.last_read_cmd = 0;
                        bank.last_write_cmd = 0;
                        trace_bank_state_change(ch, b, lpddr5::BankState::IDLE, "tRP complete");
                        break;
                    case lpddr5::BankState::REFRESHING:
                        bank.state = lpddr5::BankState::IDLE;
                        trace_bank_state_change(ch, b, lpddr5::BankState::IDLE, "tRFCpb complete");
                        break;
                    default:
                        break;
                }
            }

            // Burst completion
            if (current_cycle_ >= bank.burst_end) {
                if (bank.state == lpddr5::BankState::READING || bank.state == lpddr5::BankState::WRITING) {
                    bank.state = lpddr5::BankState::ACTIVE;
                    trace_bank_state_change(ch, b, lpddr5::BankState::ACTIVE, "burst complete");
                }
            }
        }
    }
}

void LPDDR5MemoryController::update_bus_states() {
    for (uint8_t ch = 0; ch < lpddr5_config_.num_channels; ++ch) {
        Channel& channel = channels_[ch];

        // Command bus
        if (current_cycle_ >= channel.cmd_bus_until &&
            channel.cmd_bus_state != CommandBusState::IDLE) {
            channel.cmd_bus_state = CommandBusState::IDLE;
            // Defer IDLE tracing until end of tick to avoid spurious transitions
            channel.deferred_cmd_bus_idle_trace = true;
        }

        // Data bus
        if (current_cycle_ >= channel.data_bus_until &&
            channel.data_bus_state != DataBusState::IDLE) {
            channel.data_bus_state = DataBusState::IDLE;
            // Defer IDLE tracing until end of tick to avoid spurious transitions
            channel.deferred_data_bus_idle_trace = true;
        }
    }
}

void LPDDR5MemoryController::finalize_bus_traces() {
    // Trace deferred IDLE transitions only if bus is still IDLE
    // This avoids spurious IDLE traces when a new operation starts on the same cycle
    for (uint8_t ch = 0; ch < lpddr5_config_.num_channels; ++ch) {
        Channel& channel = channels_[ch];

        if (channel.deferred_cmd_bus_idle_trace) {
            if (channel.cmd_bus_state == CommandBusState::IDLE) {
                trace_bus_state_change(ch, false, "IDLE", "cmd complete");
            }
            channel.deferred_cmd_bus_idle_trace = false;
        }

        if (channel.deferred_data_bus_idle_trace) {
            if (channel.data_bus_state == DataBusState::IDLE) {
                trace_bus_state_change(ch, true, "IDLE", "burst complete");
            }
            channel.deferred_data_bus_idle_trace = false;
        }
    }
}

void LPDDR5MemoryController::handle_refresh() {
    const auto& timing = lpddr5_config_.timing;

    for (uint8_t ch = 0; ch < lpddr5_config_.num_channels; ++ch) {
        Channel& channel = channels_[ch];

        // Check if any bank needs refresh
        for (uint8_t b = 0; b < 16; ++b) {
            Bank& bank = channel.banks[b];

            // Check if refresh deadline is approaching
            uint64_t deadline = bank.last_refresh + 16 * timing.tREFIpb;
            if (current_cycle_ >= deadline - timing.tRFCpb) {
                // Urgent refresh needed
                if (can_refresh(ch, b)) {
                    do_refresh(ch, b);
                    return;  // Only one refresh per cycle
                }
            }
        }

        // Background refresh (round-robin)
        uint8_t next_bank = channel.next_refresh_bank;
        Bank& bank = channel.banks[next_bank];

        if (current_cycle_ >= bank.last_refresh + timing.tREFIpb) {
            if (can_refresh(ch, next_bank)) {
                do_refresh(ch, next_bank);
                channel.next_refresh_bank = (next_bank + 1) % 16;
            }
        }
    }
}

// ============================================================================
// Request scheduling
// ============================================================================

void LPDDR5MemoryController::schedule_requests() {
    if (pending_queue_.empty()) {
        return;
    }

    // Simple FCFS scheduling for now
    MemoryRequest& req = pending_queue_.front();

    if (try_issue_request(req)) {
        active_requests_.push_back(std::move(pending_queue_.front()));
        pending_queue_.pop();
    } else {
        stats_.stall_cycles++;
    }
}

bool LPDDR5MemoryController::try_issue_request(MemoryRequest& req) {
    uint8_t ch = req.channel;
    uint8_t bank = req.bank;
    Bank& b = channels_[ch].banks[bank];

    // Check current bank state
    switch (b.state) {
        case lpddr5::BankState::IDLE:
            // Page empty - need to activate
            if (can_activate(ch, bank, current_cycle_)) {
                do_activate(ch, bank, req.row);
                // Only count as page_empty if this wasn't preceded by a page conflict
                if (!req.triggered_conflict) {
                    stats_.page_empty++;
                }
                req.triggered_activate = true;  // Mark that this request triggered activate
                return false;  // Will issue R/W next cycle
            }
            return false;

        case lpddr5::BankState::ACTIVATING:
            // Wait for activation to complete
            return false;

        case lpddr5::BankState::ACTIVE:
            if (b.open_row == req.row) {
                // Row is open - issue R/W
                // Only count as page_hit if we didn't trigger the activate
                if (req.type == RequestType::READ) {
                    if (can_read(ch, bank, current_cycle_)) {
                        do_read(ch, bank, req);
                        if (!req.triggered_activate) {
                            stats_.page_hits++;
                        }

                        // Track bank group usage
                        // (simplified - would need to track previous access bank)
                        return true;
                    }
                } else {
                    if (can_write(ch, bank, current_cycle_)) {
                        do_write(ch, bank, req);
                        if (!req.triggered_activate) {
                            stats_.page_hits++;
                        }
                        return true;
                    }
                }
            } else {
                // Page conflict - need to precharge
                if (can_precharge(ch, bank, current_cycle_)) {
                    do_precharge(ch, bank);
                    stats_.page_conflicts++;
                    req.triggered_conflict = true;  // Mark that this request triggered a conflict
                }
            }
            return false;

        case lpddr5::BankState::READING:
        case lpddr5::BankState::WRITING:
            // Wait for burst to complete
            return false;

        case lpddr5::BankState::PRECHARGING:
            // Wait for precharge to complete
            return false;

        case lpddr5::BankState::REFRESHING:
            // Wait for refresh to complete
            return false;

        default:
            return false;
    }
}

// ============================================================================
// Request completion
// ============================================================================

void LPDDR5MemoryController::complete_requests() {
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        if (current_cycle_ >= it->complete_cycle && it->complete_cycle > 0) {
            // Request completed
            uint64_t latency = current_cycle_ - it->submit_cycle;
            stats_.total_latency += latency;

            // Call completion callback
            if (it->callback) {
                it->callback();
            }

            it = active_requests_.erase(it);
        } else {
            ++it;
        }
    }
}

// ============================================================================
// Invariant checking
// ============================================================================

void LPDDR5MemoryController::check_all_invariants() {
    for (uint8_t ch = 0; ch < lpddr5_config_.num_channels; ++ch) {
        check_timing_invariants(ch);
        check_bus_invariants(ch);

        for (uint8_t b = 0; b < 16; ++b) {
            check_bank_invariants(ch, b);
        }
    }
}

void LPDDR5MemoryController::check_bank_invariants(uint8_t channel, uint8_t bank) {
    const Bank& b = channels_[channel].banks[bank];
    const auto& timing = lpddr5_config_.timing;

    // INV-BANK-1: Only one row can be active per bank at any time
    // (This is implicit in our state machine - enforced by design)

    // INV-BANK-2: tRCD check for ACTIVE state
    if (b.state == lpddr5::BankState::ACTIVE && current_cycle_ < b.state_until) {
        // This shouldn't happen - we transitioned too early
        report_violation("INV-BANK-2",
            "Bank transitioned to ACTIVE before tRCD elapsed",
            channel, bank);
    }

    // INV-BANK-4: tRAS check
    if (b.state == lpddr5::BankState::PRECHARGING) {
        if (b.last_activate > 0 &&
            current_cycle_ < b.last_activate + timing.tRAS) {
            report_violation("INV-BANK-4",
                "PRECHARGE issued before tRAS elapsed",
                channel, bank);
        }
    }

    // INV-BANK-5: tWR check (write recovery)
    if (b.state == lpddr5::BankState::PRECHARGING && b.last_write_cmd > 0) {
        uint64_t write_done = b.last_write_cmd + timing.tWL + burst_cycles();
        if (current_cycle_ < write_done + timing.tWR) {
            report_violation("INV-BANK-5",
                "PRECHARGE issued before tWR elapsed after write",
                channel, bank);
        }
    }

    // INV-BANK-6: tRTP check (read to precharge)
    if (b.state == lpddr5::BankState::PRECHARGING && b.last_read_cmd > 0) {
        if (current_cycle_ < b.last_read_cmd + timing.tRTP) {
            report_violation("INV-BANK-6",
                "PRECHARGE issued before tRTP elapsed after read",
                channel, bank);
        }
    }
}

void LPDDR5MemoryController::check_timing_invariants(uint8_t channel) {
    const auto& ch = channels_[channel];
    const auto& timing = lpddr5_config_.timing;

    // INV-TIME-4: tFAW check
    int activate_count = 0;
    uint64_t window_start = (current_cycle_ > timing.tFAW) ?
                            current_cycle_ - timing.tFAW : 0;

    for (int i = 0; i < 4; ++i) {
        if (ch.activate_window[i] >= window_start) {
            activate_count++;
        }
    }

    if (activate_count > 4) {
        report_violation("INV-TIME-4",
            "More than 4 ACTIVATEs in tFAW window",
            channel, 0);
    }
}

void LPDDR5MemoryController::check_bus_invariants(uint8_t channel) {
    const auto& ch = channels_[channel];

    // INV-CMD-1: One command per cycle
    // (Enforced by our command bus state machine)

    // INV-DATA-1: Data bus exclusive
    // (Enforced by our data bus state machine)

    // Check for impossible state combinations
    if (ch.data_bus_state == DataBusState::READ_BURST &&
        ch.last_was_write) {
        // Data bus says reading but we think we're writing
        // This could indicate a state tracking bug
    }
}

void LPDDR5MemoryController::report_violation(
    const std::string& id, const std::string& msg,
    uint8_t channel, uint8_t bank)
{
    lpddr5::InvariantViolation v;
    v.cycle = current_cycle_;
    v.invariant_id = id;
    v.message = msg;
    v.channel = channel;
    v.bank = bank;

    violations_.push_back(v);
}

// ============================================================================
// Tracing helpers
// ============================================================================

void LPDDR5MemoryController::trace_bank_state_change(
    uint8_t channel, uint8_t bank, lpddr5::BankState new_state, const std::string& reason)
{
    if (!tracing_enabled_) return;

    // Track resource state changes
    if (resource_tracker_) {
        // Compute unique bank ID: channel * 16 + bank
        uint32_t bank_id = channel * 16 + bank;

        sw::trace::ResourceState rs;
        switch (new_state) {
            case lpddr5::BankState::IDLE:
                rs = sw::trace::ResourceState::IDLE;
                break;
            case lpddr5::BankState::ACTIVATING:
            case lpddr5::BankState::PRECHARGING:
            case lpddr5::BankState::REFRESHING:
                rs = sw::trace::ResourceState::STALLED;
                break;
            case lpddr5::BankState::ACTIVE:
            case lpddr5::BankState::READING:
            case lpddr5::BankState::WRITING:
                rs = sw::trace::ResourceState::BUSY;
                break;
            default:
                rs = sw::trace::ResourceState::IDLE;
                break;
        }

        resource_tracker_->transition(
            sw::trace::ComponentType::LPDDR5_BANK,
            bank_id,
            rs,
            current_cycle_,
            0,
            reason
        );
    }
}

void LPDDR5MemoryController::trace_bus_state_change(
    uint8_t channel, bool is_data_bus, const std::string& state, const std::string& reason)
{
    if (!tracing_enabled_) return;

    if (resource_tracker_) {
        sw::trace::ComponentType bus_type = is_data_bus ?
            sw::trace::ComponentType::LPDDR5_DATA_BUS :
            sw::trace::ComponentType::LPDDR5_CMD_BUS;

        sw::trace::ResourceState rs = (state == "IDLE") ?
            sw::trace::ResourceState::IDLE :
            sw::trace::ResourceState::BUSY;

        resource_tracker_->transition(
            bus_type,
            channel,
            rs,
            current_cycle_,
            0,
            reason
        );
    }
}

void LPDDR5MemoryController::trace_command(
    uint8_t channel, uint8_t bank, const std::string& cmd,
    uint64_t duration, uint64_t request_id)
{
    if (!tracing_enabled_) return;

    // Create trace entry for the command
    uint32_t bank_id = channel * 16 + bank;

    // Determine transaction type
    sw::trace::TransactionType trans_type;
    if (cmd == "ACTIVATE") {
        trans_type = sw::trace::TransactionType::ACTIVATE;
    } else if (cmd == "READ") {
        trans_type = sw::trace::TransactionType::BURST_READ;
    } else if (cmd == "WRITE") {
        trans_type = sw::trace::TransactionType::BURST_WRITE;
    } else if (cmd == "PRECHARGE") {
        trans_type = sw::trace::TransactionType::PRECHARGE;
    } else if (cmd == "REFRESH") {
        trans_type = sw::trace::TransactionType::REFRESH;
    } else {
        trans_type = sw::trace::TransactionType::UNKNOWN;
    }

    sw::trace::TraceEntry entry(
        current_cycle_,
        sw::trace::ComponentType::LPDDR5_BANK,
        bank_id,
        trans_type,
        request_id
    );

    entry.complete(current_cycle_ + duration);
    entry.description = cmd + " Ch" + std::to_string(channel) + " Bank" + std::to_string(bank);

    // Add memory payload
    sw::trace::MemoryPayload payload;
    payload.location = sw::trace::MemoryLocation(0, 0, bank_id, sw::trace::ComponentType::LPDDR5_BANK);
    payload.latency_cycles = static_cast<uint32_t>(duration);
    entry.payload = payload;

    trace_entries_.push_back(entry);
}

} // namespace sw::kpu::lpddr5
