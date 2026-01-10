// ============================================================================
// src/components/memory/gddr6_memory_controller.cpp
// GDDR6 Memory Controller implementation with formal invariant checking
// ============================================================================

#include <sw/kpu/components/gddr6_memory_controller.hpp>
#include <sw/trace/trace_entry.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <cassert>

namespace sw::kpu::gddr6 {

// Note: We use fully qualified gddr6::BankState:: to avoid ambiguity with
// IMemoryController::BankState which is brought in through inheritance.

// ============================================================================
// Constructor
// ============================================================================

GDDR6MemoryController::GDDR6MemoryController(const Config& config)
    : gddr6_config_(config)
{
    // Initialize bank groups for each bank
    const uint8_t num_ch = std::min<uint8_t>(gddr6_config_.num_channels, 2);
    for (uint8_t ch = 0; ch < num_ch; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            channels_[ch].banks[b].bank_group = b / 4;
        }
    }

    // Initialize interface config from GDDR6 config
    interface_config_.fidelity = sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    interface_config_.technology = sw::kpu::MemoryTechnology::GDDR6;
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

GDDR6MemoryController::GDDR6MemoryController(const sw::kpu::MemoryControllerConfig& config)
    : GDDR6MemoryController(Config{})  // Start with defaults
{
    // Copy relevant fields from interface config
    gddr6_config_.num_channels = config.num_channels;
    gddr6_config_.banks_per_channel = config.banks_per_channel;
    gddr6_config_.bank_groups = config.bank_groups;
    gddr6_config_.queue_depth = config.queue_depth;
    gddr6_config_.row_bits = config.row_bits;
    gddr6_config_.col_bits = config.col_bits;
    gddr6_config_.bank_bits = config.bank_bits;
    gddr6_config_.channel_bits = config.channel_bits;

    // Copy timing parameters (GDDR6 uses tRCDRD/tRCDWR instead of single tRCD)
    gddr6_config_.timing.tRCDRD = config.timing.tRCD;
    gddr6_config_.timing.tRCDWR = config.timing.tRCD;
    gddr6_config_.timing.tRP = config.timing.tRP;
    gddr6_config_.timing.tRAS = config.timing.tRAS;
    gddr6_config_.timing.tRC = config.timing.tRC;
    gddr6_config_.timing.tRL = config.timing.tCL;
    gddr6_config_.timing.tWL = config.timing.tWL;
    gddr6_config_.timing.tWR = config.timing.tWR;
    gddr6_config_.timing.tRTP = config.timing.tRTP;
    gddr6_config_.timing.tRRD_L = config.timing.tRRD_L;
    gddr6_config_.timing.tRRD_S = config.timing.tRRD_S;
    gddr6_config_.timing.tCCD_L = config.timing.tCCD_L;
    gddr6_config_.timing.tCCD_S = config.timing.tCCD_S;
    gddr6_config_.timing.tWTR_L = config.timing.tWTR_L;
    gddr6_config_.timing.tWTR_S = config.timing.tWTR_S;
    gddr6_config_.timing.tRTW = config.timing.tRTW;
    gddr6_config_.timing.tFAW = config.timing.tFAW;

    // Store interface config
    interface_config_ = config;
    interface_config_.fidelity = sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    interface_config_.technology = sw::kpu::MemoryTechnology::GDDR6;

    // Set verification level based on config
    check_invariants_ = (config.verification == sw::kpu::VerificationLevel::INVARIANTS ||
                        config.verification == sw::kpu::VerificationLevel::PROTOCOL);
    tracing_enabled_ = config.enable_tracing;

    // Reinitialize bank groups
    for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            channels_[ch].banks[b].bank_group = b / 4;
        }
    }
}

// ============================================================================
// Request submission
// ============================================================================

std::optional<uint64_t> GDDR6MemoryController::submit_read(
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

    // Track channel usage
    if (req.channel == 0) {
        stats_.channel_a_accesses++;
    } else {
        stats_.channel_b_accesses++;
    }

    return req.id;
}

std::optional<uint64_t> GDDR6MemoryController::submit_write(
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

    // Track channel usage
    if (req.channel == 0) {
        stats_.channel_a_accesses++;
    } else {
        stats_.channel_b_accesses++;
    }

    return req.id;
}

// ============================================================================
// Simulation interface
// ============================================================================

void GDDR6MemoryController::tick() {
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

    // Finalize bus traces
    finalize_bus_traces();

    current_cycle_++;

    // Sync interface stats
    sync_interface_stats();
    if (check_invariants_ && !violations_.empty()) {
        sync_interface_violations();
    }
}

void GDDR6MemoryController::drain() {
    const uint64_t MAX_DRAIN_CYCLES = 100000;
    uint64_t start = current_cycle_;

    while (has_pending() && (current_cycle_ - start) < MAX_DRAIN_CYCLES) {
        tick();
    }
}

void GDDR6MemoryController::reset() {
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

bool GDDR6MemoryController::has_pending() const {
    return !pending_queue_.empty() || !active_requests_.empty();
}

bool GDDR6MemoryController::can_accept() const {
    return pending_queue_.size() < gddr6_config_.queue_depth;
}

size_t GDDR6MemoryController::pending_count() const {
    return pending_queue_.size() + active_requests_.size();
}

// Convert GDDR6 BankState to IMemoryController::BankState
static sw::kpu::IMemoryController::BankState to_interface_state(gddr6::BankState state) {
    switch (state) {
        case gddr6::BankState::IDLE:        return sw::kpu::IMemoryController::BankState::IDLE;
        case gddr6::BankState::ACTIVATING:  return sw::kpu::IMemoryController::BankState::ACTIVATING;
        case gddr6::BankState::ACTIVE:      return sw::kpu::IMemoryController::BankState::ACTIVE;
        case gddr6::BankState::READING:     return sw::kpu::IMemoryController::BankState::READING;
        case gddr6::BankState::WRITING:     return sw::kpu::IMemoryController::BankState::WRITING;
        case gddr6::BankState::PRECHARGING: return sw::kpu::IMemoryController::BankState::PRECHARGING;
        case gddr6::BankState::REFRESHING:  return sw::kpu::IMemoryController::BankState::REFRESHING;
        default:                            return sw::kpu::IMemoryController::BankState::IDLE;
    }
}

sw::kpu::IMemoryController::BankState GDDR6MemoryController::get_bank_state(
    uint8_t channel, uint8_t bank) const
{
    return to_interface_state(get_gddr6_bank_state(channel, bank));
}

gddr6::BankState GDDR6MemoryController::get_gddr6_bank_state(uint8_t channel, uint8_t bank) const {
    if (channel >= gddr6_config_.num_channels || bank >= 16) {
        return gddr6::BankState::IDLE;
    }
    return channels_[channel].banks[bank].state;
}

bool GDDR6MemoryController::is_bank_idle(uint8_t channel, uint8_t bank) const {
    return get_gddr6_bank_state(channel, bank) == gddr6::BankState::IDLE;
}

bool GDDR6MemoryController::is_row_open(uint8_t channel, uint8_t bank, uint32_t row) const {
    if (channel >= gddr6_config_.num_channels || bank >= 16) {
        return false;
    }
    const Bank& b = channels_[channel].banks[bank];
    return (b.state == gddr6::BankState::ACTIVE ||
            b.state == gddr6::BankState::READING ||
            b.state == gddr6::BankState::WRITING) &&
           b.open_row == row;
}

void GDDR6MemoryController::reset_stats() {
    stats_ = Statistics{};
    interface_stats_.reset();
}

void GDDR6MemoryController::sync_interface_stats() {
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

    interface_stats_.min_latency = std::min(stats_.read_latency_min, stats_.write_latency_min);
    interface_stats_.max_latency = std::max(stats_.read_latency_max, stats_.write_latency_max);
}

void GDDR6MemoryController::sync_interface_violations() {
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

void GDDR6MemoryController::decode_address(
    uint64_t address, uint8_t& channel, uint8_t& bank,
    uint32_t& row, uint32_t& col) const
{
    // GDDR6 address mapping:
    // [row | bank | col | channel | byte_offset]
    uint32_t byte_offset_bits = 5;  // 32-byte minimum access
    uint32_t col_bits = gddr6_config_.col_bits;
    uint32_t bank_bits = gddr6_config_.bank_bits;
    uint32_t channel_bits = gddr6_config_.channel_bits;

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

    row = addr & ((1 << gddr6_config_.row_bits) - 1);
}

// ============================================================================
// Burst cycles
// ============================================================================

uint32_t GDDR6MemoryController::burst_cycles() const {
    // GDDR6 BL16: 4 WCK cycles = 4 CK cycles for burst
    return gddr6_config_.timing.tBurst;
}

// ============================================================================
// tFAW tracking
// ============================================================================

bool GDDR6MemoryController::check_tFAW(uint8_t channel) const {
    const auto& ch = channels_[channel];
    const auto& timing = gddr6_config_.timing;

    uint64_t oldest = ch.activate_window[0];
    for (int i = 1; i < 4; ++i) {
        if (ch.activate_window[i] < oldest && ch.activate_window[i] > 0) {
            oldest = ch.activate_window[i];
        }
    }

    if (oldest > 0 && current_cycle_ < oldest + timing.tFAW) {
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

void GDDR6MemoryController::record_activate(uint8_t channel) {
    auto& ch = channels_[channel];
    ch.activate_window[ch.activate_index] = current_cycle_;
    ch.activate_index = (ch.activate_index + 1) % 4;
}

// ============================================================================
// Bank operation checks
// ============================================================================

bool GDDR6MemoryController::can_activate(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    // Bank must be IDLE
    if (b.state != gddr6::BankState::IDLE) {
        return false;
    }

    // tRP must have elapsed since last precharge
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

bool GDDR6MemoryController::can_read(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    // Bank must be ACTIVE
    if (b.state != gddr6::BankState::ACTIVE) {
        return false;
    }

    // tRCDRD must have elapsed
    if (cycle < b.state_until) {
        return false;
    }

    // Bank group CAS timing (tCCD_L)
    uint8_t bg = bank / 4;
    const BankGroup& bank_group = ch.bank_groups[bg];
    if (cycle < bank_group.last_cas + timing.tCCD_L) {
        return false;
    }

    // Write-to-read turnaround (tWTR_L)
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

bool GDDR6MemoryController::can_write(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    // Bank must be ACTIVE
    if (b.state != gddr6::BankState::ACTIVE) {
        return false;
    }

    // tRCDWR must have elapsed
    if (cycle < b.state_until) {
        return false;
    }

    // Bank group CAS timing (tCCD_L)
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

bool GDDR6MemoryController::can_precharge(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    // Bank must be ACTIVE
    if (b.state != gddr6::BankState::ACTIVE) {
        return false;
    }

    // tRAS must have elapsed
    if (cycle < b.last_activate + timing.tRAS) {
        return false;
    }

    // tWR must have elapsed after write
    if (b.last_write_cmd > 0) {
        uint64_t write_done = b.last_write_cmd + timing.tWL + burst_cycles();
        if (cycle < write_done + timing.tWR) {
            return false;
        }
    }

    // tRTP must have elapsed after read
    if (b.last_read_cmd > 0) {
        if (cycle < b.last_read_cmd + timing.tRTP) {
            return false;
        }
    }

    return true;
}

bool GDDR6MemoryController::can_refresh(uint8_t channel, uint8_t bank) const {
    return channels_[channel].banks[bank].state == gddr6::BankState::IDLE;
}

// ============================================================================
// Bank operations
// ============================================================================

void GDDR6MemoryController::do_activate(uint8_t channel, uint8_t bank, uint32_t row, uint64_t request_id) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    b.state = gddr6::BankState::ACTIVATING;
    b.open_row = row;
    // Use tRCDRD as the default activation time (will complete before read/write)
    b.state_until = current_cycle_ + timing.tRCDRD;
    b.last_activate = current_cycle_;

    // Track which request opened this page
    b.page_opener_request_id = request_id;

    // Update bank group
    uint8_t bg = bank / 4;
    ch.bank_groups[bg].last_activate = current_cycle_;

    // Record for tFAW
    record_activate(channel);

    // Command bus busy for 1 cycle
    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation
    trace_bank_state_change(channel, bank, gddr6::BankState::ACTIVATING, "ACTIVATE row " + std::to_string(row));
    trace_command(channel, bank, "ACTIVATE", timing.tRCDRD, request_id);
    trace_bus_state_change(channel, false, "BUSY", "ACT cmd");
}

void GDDR6MemoryController::do_read(uint8_t channel, uint8_t bank, MemoryRequest& req) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    b.state = gddr6::BankState::READING;
    b.last_read_cmd = current_cycle_;
    b.burst_end = current_cycle_ + timing.tRL + burst_cycles();

    // Update bank group
    uint8_t bg = bank / 4;
    ch.bank_groups[bg].last_cas = current_cycle_;

    // Track turnarounds
    if (ch.last_was_write) {
        stats_.write_to_read_turnarounds++;
    }

    // Data bus will be busy after tRL
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
    uint64_t duration = timing.tRL + burst_cycles();
    trace_bank_state_change(channel, bank, gddr6::BankState::READING, "READ burst");
    trace_command(channel, bank, "READ", duration, req.id);
    trace_bus_state_change(channel, true, "BUSY", "READ burst");
    trace_bus_state_change(channel, false, "BUSY", "RD cmd");
}

void GDDR6MemoryController::do_write(uint8_t channel, uint8_t bank, MemoryRequest& req) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    b.state = gddr6::BankState::WRITING;
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
    trace_bank_state_change(channel, bank, gddr6::BankState::WRITING, "WRITE burst");
    trace_command(channel, bank, "WRITE", duration, req.id);
    trace_bus_state_change(channel, true, "BUSY", "WRITE burst");
    trace_bus_state_change(channel, false, "BUSY", "WR cmd");
}

void GDDR6MemoryController::do_precharge(uint8_t channel, uint8_t bank) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    // Capture the request_id before clearing
    uint64_t request_id = b.page_opener_request_id;

    b.state = gddr6::BankState::PRECHARGING;
    b.state_until = current_cycle_ + timing.tRP;

    // Command bus busy for 1 cycle
    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation with the correct request_id
    trace_bank_state_change(channel, bank, gddr6::BankState::PRECHARGING, "PRECHARGE");
    trace_command(channel, bank, "PRECHARGE", timing.tRP, request_id);
    trace_bus_state_change(channel, false, "BUSY", "PRE cmd");
}

void GDDR6MemoryController::do_refresh(uint8_t channel, uint8_t bank) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr6_config_.timing;

    b.state = gddr6::BankState::REFRESHING;
    b.state_until = current_cycle_ + timing.tRFCpb;
    b.last_refresh = current_cycle_;

    stats_.refreshes++;

    // Command bus busy for 1 cycle
    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation
    trace_bank_state_change(channel, bank, gddr6::BankState::REFRESHING, "REFRESH");
    trace_command(channel, bank, "REFRESH", timing.tRFCpb, 0);
    trace_bus_state_change(channel, false, "BUSY", "REF cmd");
}

// ============================================================================
// State updates
// ============================================================================

void GDDR6MemoryController::update_bank_states() {
    for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            Bank& bank = channels_[ch].banks[b];

            if (current_cycle_ >= bank.state_until) {
                switch (bank.state) {
                    case gddr6::BankState::ACTIVATING:
                        bank.state = gddr6::BankState::ACTIVE;
                        trace_bank_state_change(ch, b, gddr6::BankState::ACTIVE, "tRCD complete");
                        break;
                    case gddr6::BankState::PRECHARGING:
                        bank.state = gddr6::BankState::IDLE;
                        bank.open_row = 0;
                        bank.last_read_cmd = 0;
                        bank.last_write_cmd = 0;
                        trace_bank_state_change(ch, b, gddr6::BankState::IDLE, "tRP complete");
                        break;
                    case gddr6::BankState::REFRESHING:
                        bank.state = gddr6::BankState::IDLE;
                        trace_bank_state_change(ch, b, gddr6::BankState::IDLE, "tRFCpb complete");
                        break;
                    default:
                        break;
                }
            }

            // Burst completion
            if (current_cycle_ >= bank.burst_end) {
                if (bank.state == gddr6::BankState::READING || bank.state == gddr6::BankState::WRITING) {
                    bank.state = gddr6::BankState::ACTIVE;
                    trace_bank_state_change(ch, b, gddr6::BankState::ACTIVE, "burst complete");
                }
            }
        }
    }
}

void GDDR6MemoryController::update_bus_states() {
    for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
        Channel& channel = channels_[ch];

        // Command bus
        if (current_cycle_ >= channel.cmd_bus_until &&
            channel.cmd_bus_state != CommandBusState::IDLE) {
            channel.cmd_bus_state = CommandBusState::IDLE;
            channel.deferred_cmd_bus_idle_trace = true;
        }

        // Data bus
        if (current_cycle_ >= channel.data_bus_until &&
            channel.data_bus_state != DataBusState::IDLE) {
            channel.data_bus_state = DataBusState::IDLE;
            channel.deferred_data_bus_idle_trace = true;
        }
    }
}

void GDDR6MemoryController::finalize_bus_traces() {
    for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
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

void GDDR6MemoryController::handle_refresh() {
    // DISABLED mode: no refresh at all
    if (refresh_mode_ == RefreshMode::DISABLED) {
        return;
    }

    const auto& timing = gddr6_config_.timing;

    // Deadline enforcement (applies to all modes except DISABLED)
    if (deadline_enforcement_) {
        for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
            Channel& channel = channels_[ch];

            for (uint8_t b = 0; b < 16; ++b) {
                Bank& bank = channel.banks[b];

                // Check if refresh deadline is approaching
                uint64_t deadline = bank.last_refresh + 16 * timing.tREFI;
                if (current_cycle_ >= deadline - timing.tRFCpb) {
                    // Urgent refresh needed - force it regardless of mode
                    if (can_refresh(ch, b)) {
                        do_refresh(ch, b);
                        return;  // Only one refresh per cycle
                    }
                }
            }
        }
    }

    // Mode-specific scheduling
    switch (refresh_mode_) {
        case RefreshMode::AUTOMATIC:
            // Original behavior: opportunistic + proactive round-robin
            for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
                Channel& channel = channels_[ch];
                uint8_t next_bank = channel.next_refresh_bank;
                Bank& bank = channel.banks[next_bank];

                if (current_cycle_ >= bank.last_refresh + timing.tREFI) {
                    if (can_refresh(ch, next_bank)) {
                        do_refresh(ch, next_bank);
                        channel.next_refresh_bank = (next_bank + 1) % 16;
                        return;
                    }
                }
            }
            break;

        case RefreshMode::INTERVAL: {
            // Fixed interval mode: refresh at user-specified intervals
            uint64_t interval = refresh_interval_;
            if (interval == 0) {
                interval = timing.tREFI;  // Default to tREFI
            }

            if (current_cycle_ >= last_interval_refresh_ + interval) {
                // Do round-robin refresh across banks
                for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
                    Channel& channel = channels_[ch];
                    uint8_t next_bank = channel.next_refresh_bank;

                    if (can_refresh(ch, next_bank)) {
                        do_refresh(ch, next_bank);
                        channel.next_refresh_bank = (next_bank + 1) % 16;
                        last_interval_refresh_ = current_cycle_;
                        return;
                    }
                }
            }
            break;
        }

        case RefreshMode::OPPORTUNISTIC:
            // Only refresh when bus is idle (opportunistic only)
            for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
                Channel& channel = channels_[ch];

                // Check if bus is idle
                if (channel.cmd_bus_state == CommandBusState::IDLE &&
                    channel.data_bus_state == DataBusState::IDLE) {

                    uint8_t next_bank = channel.next_refresh_bank;
                    Bank& bank = channel.banks[next_bank];

                    if (current_cycle_ >= bank.last_refresh + timing.tREFI) {
                        if (can_refresh(ch, next_bank)) {
                            do_refresh(ch, next_bank);
                            channel.next_refresh_bank = (next_bank + 1) % 16;
                            return;
                        }
                    }
                }
            }
            break;

        case RefreshMode::EXPLICIT:
            // Do nothing - wait for inject_refresh() calls
            break;

        case RefreshMode::DISABLED:
            // Already handled above
            break;
    }
}

// ============================================================================
// Request scheduling
// ============================================================================

void GDDR6MemoryController::schedule_requests() {
    if (pending_queue_.empty()) {
        return;
    }

    // Simple FCFS scheduling
    MemoryRequest& req = pending_queue_.front();

    if (try_issue_request(req)) {
        active_requests_.push_back(std::move(pending_queue_.front()));
        pending_queue_.pop();
    } else {
        stats_.stall_cycles++;
    }
}

bool GDDR6MemoryController::try_issue_request(MemoryRequest& req) {
    uint8_t ch = req.channel;
    uint8_t bank = req.bank;
    Bank& b = channels_[ch].banks[bank];

    switch (b.state) {
        case gddr6::BankState::IDLE:
            // Page empty - need to activate
            if (can_activate(ch, bank, current_cycle_)) {
                do_activate(ch, bank, req.row, req.id);
                if (!req.triggered_conflict) {
                    stats_.page_empty++;
                }
                req.triggered_activate = true;
                return false;
            }
            return false;

        case gddr6::BankState::ACTIVATING:
            // Wait for activation to complete
            return false;

        case gddr6::BankState::ACTIVE:
            if (b.open_row == req.row) {
                // Row is open - issue R/W
                if (req.type == RequestType::READ) {
                    if (can_read(ch, bank, current_cycle_)) {
                        do_read(ch, bank, req);
                        if (!req.triggered_activate) {
                            stats_.page_hits++;
                        }
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
                    req.triggered_conflict = true;
                }
            }
            return false;

        case gddr6::BankState::READING:
        case gddr6::BankState::WRITING:
            // Wait for burst to complete
            return false;

        case gddr6::BankState::PRECHARGING:
            // Wait for precharge to complete
            return false;

        case gddr6::BankState::REFRESHING:
            // Wait for refresh to complete
            return false;

        default:
            return false;
    }
}

// ============================================================================
// Request completion
// ============================================================================

void GDDR6MemoryController::complete_requests() {
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        if (current_cycle_ >= it->complete_cycle && it->complete_cycle > 0) {
            // Request completed
            uint64_t latency = current_cycle_ - it->submit_cycle;
            stats_.total_latency += latency;

            // Track separate read/write latencies
            if (it->type == RequestType::READ) {
                stats_.read_latency_total += latency;
                stats_.read_latency_min = std::min(stats_.read_latency_min, latency);
                stats_.read_latency_max = std::max(stats_.read_latency_max, latency);
            } else {
                stats_.write_latency_total += latency;
                stats_.write_latency_min = std::min(stats_.write_latency_min, latency);
                stats_.write_latency_max = std::max(stats_.write_latency_max, latency);
            }

            // Track per-scenario latencies
            if (it->triggered_conflict) {
                stats_.page_conflict_latency_total += latency;
                stats_.page_conflict_count++;
            } else if (it->triggered_activate) {
                stats_.page_empty_latency_total += latency;
                stats_.page_empty_count++;
            } else {
                stats_.page_hit_latency_total += latency;
                stats_.page_hit_count++;
            }

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

void GDDR6MemoryController::check_all_invariants() {
    for (uint8_t ch = 0; ch < gddr6_config_.num_channels; ++ch) {
        check_timing_invariants(ch);
        check_bus_invariants(ch);

        for (uint8_t b = 0; b < 16; ++b) {
            check_bank_invariants(ch, b);
        }
    }
}

void GDDR6MemoryController::check_bank_invariants(uint8_t channel, uint8_t bank) {
    const Bank& b = channels_[channel].banks[bank];
    const auto& timing = gddr6_config_.timing;

    // INV-100: tRCD check for ACTIVE state
    if (b.state == gddr6::BankState::ACTIVE && current_cycle_ < b.state_until) {
        report_violation("INV-100",
            "Bank transitioned to ACTIVE before tRCD elapsed",
            channel, bank);
    }

    // INV-107: tRAS check
    if (b.state == gddr6::BankState::PRECHARGING) {
        if (b.last_activate > 0 &&
            current_cycle_ < b.last_activate + timing.tRAS) {
            report_violation("INV-107",
                "PRECHARGE issued before tRAS elapsed",
                channel, bank);
        }
    }

    // tWR check (write recovery)
    if (b.state == gddr6::BankState::PRECHARGING && b.last_write_cmd > 0) {
        uint64_t write_done = b.last_write_cmd + timing.tWL + burst_cycles();
        if (current_cycle_ < write_done + timing.tWR) {
            report_violation("INV-BANK-WR",
                "PRECHARGE issued before tWR elapsed after write",
                channel, bank);
        }
    }

    // tRTP check (read to precharge)
    if (b.state == gddr6::BankState::PRECHARGING && b.last_read_cmd > 0) {
        if (current_cycle_ < b.last_read_cmd + timing.tRTP) {
            report_violation("INV-BANK-RTP",
                "PRECHARGE issued before tRTP elapsed after read",
                channel, bank);
        }
    }
}

void GDDR6MemoryController::check_timing_invariants(uint8_t channel) {
    const auto& ch = channels_[channel];
    const auto& timing = gddr6_config_.timing;

    // INV-103: tFAW check
    int activate_count = 0;
    uint64_t window_start = (current_cycle_ > timing.tFAW) ?
                            current_cycle_ - timing.tFAW : 0;

    for (int i = 0; i < 4; ++i) {
        if (ch.activate_window[i] >= window_start) {
            activate_count++;
        }
    }

    if (activate_count > 4) {
        report_violation("INV-103",
            "More than 4 ACTIVATEs in tFAW window",
            channel, 0);
    }
}

void GDDR6MemoryController::check_bus_invariants(uint8_t channel) {
    const auto& ch = channels_[channel];

    // Check for impossible state combinations
    if (ch.data_bus_state == DataBusState::READ_BURST &&
        ch.last_was_write) {
        // Data bus says reading but we think we're writing
    }
}

void GDDR6MemoryController::report_violation(
    const std::string& id, const std::string& msg,
    uint8_t channel, uint8_t bank)
{
    gddr6::InvariantViolation v;
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

void GDDR6MemoryController::trace_bank_state_change(
    uint8_t channel, uint8_t bank, gddr6::BankState new_state, const std::string& reason)
{
    if (!tracing_enabled_) return;

    if (resource_tracker_) {
        // Use GDDR6_BANK for proper trace categorization
        uint32_t bank_id = channel * 16 + bank;

        sw::trace::ResourceState rs;
        switch (new_state) {
            case gddr6::BankState::IDLE:
                rs = sw::trace::ResourceState::IDLE;
                break;
            case gddr6::BankState::ACTIVATING:
            case gddr6::BankState::PRECHARGING:
            case gddr6::BankState::REFRESHING:
                rs = sw::trace::ResourceState::STALLED;
                break;
            case gddr6::BankState::ACTIVE:
            case gddr6::BankState::READING:
            case gddr6::BankState::WRITING:
                rs = sw::trace::ResourceState::BUSY;
                break;
            default:
                rs = sw::trace::ResourceState::IDLE;
                break;
        }

        resource_tracker_->transition(
            sw::trace::ComponentType::GDDR6_BANK,
            bank_id,
            rs,
            current_cycle_,
            0,
            reason
        );
    }
}

void GDDR6MemoryController::trace_bus_state_change(
    uint8_t channel, bool is_data_bus, const std::string& state, const std::string& reason)
{
    if (!tracing_enabled_) return;

    if (resource_tracker_) {
        // Use GDDR6-specific bus types
        sw::trace::ComponentType bus_type = is_data_bus ?
            sw::trace::ComponentType::GDDR6_DATA_BUS :
            sw::trace::ComponentType::GDDR6_CMD_BUS;

        sw::trace::ResourceState rs = (state == "IDLE") ?
            sw::trace::ResourceState::IDLE :
            sw::trace::ResourceState::BUSY;

        // Use a different ID range for buses (100+ for data, 200+ for cmd)
        uint32_t bus_id = is_data_bus ? (100 + channel) : (200 + channel);

        resource_tracker_->transition(
            bus_type,
            bus_id,
            rs,
            current_cycle_,
            0,
            reason
        );
    }
}

void GDDR6MemoryController::trace_command(
    uint8_t channel, uint8_t bank, const std::string& cmd,
    uint64_t duration, uint64_t request_id)
{
    if (!tracing_enabled_) return;

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
        sw::trace::ComponentType::GDDR6_BANK,
        bank_id,
        trans_type,
        request_id
    );

    entry.complete(current_cycle_ + duration);
    entry.description = cmd + " Ch" + std::to_string(channel) + " Bank" + std::to_string(bank);

    // Add memory payload
    sw::trace::MemoryPayload payload;
    payload.location = sw::trace::MemoryLocation(0, 0, bank_id, sw::trace::ComponentType::GDDR6_BANK);
    payload.latency_cycles = static_cast<uint32_t>(duration);
    entry.payload = payload;

    trace_entries_.push_back(entry);
}

// ============================================================================
// Refresh Control Implementation
// ============================================================================

void GDDR6MemoryController::set_refresh_mode(RefreshMode mode) {
    refresh_mode_ = mode;
    if (mode == RefreshMode::INTERVAL) {
        last_interval_refresh_ = current_cycle_;
    }
}

void GDDR6MemoryController::set_refresh_interval(uint64_t cycles) {
    refresh_interval_ = cycles;
    last_interval_refresh_ = current_cycle_;
}

uint64_t GDDR6MemoryController::cycles_until_deadline(uint8_t channel, uint8_t bank) const {
    if (channel >= gddr6_config_.num_channels || bank >= 16) {
        return UINT64_MAX;
    }

    const auto& timing = gddr6_config_.timing;
    const Bank& b = channels_[channel].banks[bank];

    // Deadline is 16 * tREFI after last refresh
    uint64_t deadline = b.last_refresh + 16 * timing.tREFI;

    if (current_cycle_ >= deadline) {
        return 0;
    }

    return deadline - current_cycle_;
}

bool GDDR6MemoryController::refresh_pending(uint8_t channel, uint8_t bank) const {
    if (channel >= gddr6_config_.num_channels || bank >= 16) {
        return false;
    }

    const auto& timing = gddr6_config_.timing;
    const Bank& b = channels_[channel].banks[bank];

    return current_cycle_ >= b.last_refresh + timing.tREFI;
}

uint32_t GDDR6MemoryController::refresh_debt(uint8_t channel, uint8_t bank) const {
    if (channel >= gddr6_config_.num_channels || bank >= 16) {
        return 0;
    }

    const auto& timing = gddr6_config_.timing;
    const Bank& b = channels_[channel].banks[bank];

    if (current_cycle_ <= b.last_refresh) {
        return 0;
    }

    uint64_t elapsed = current_cycle_ - b.last_refresh;
    return static_cast<uint32_t>(elapsed / timing.tREFI);
}

bool GDDR6MemoryController::inject_refresh(uint8_t channel, int8_t bank) {
    if (channel >= gddr6_config_.num_channels) {
        return false;
    }

    if (bank == -1) {
        uint8_t next_bank = channels_[channel].next_refresh_bank;
        if (can_refresh(channel, next_bank)) {
            do_refresh(channel, next_bank);
            channels_[channel].next_refresh_bank = (next_bank + 1) % 16;
            return true;
        }
        return false;
    }

    if (bank >= 16) {
        return false;
    }

    if (can_refresh(channel, static_cast<uint8_t>(bank))) {
        do_refresh(channel, static_cast<uint8_t>(bank));
        return true;
    }

    return false;
}

} // namespace sw::kpu::gddr6
