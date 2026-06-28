// ============================================================================
// src/components/memory/gddr7_memory_controller.cpp
// GDDR7 Memory Controller implementation with formal invariant checking
// ============================================================================

#include <sw/kpu/models/temporal/memory/controllers/gddr7_controller.hpp>
#include <sw/trace/trace_entry.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <cassert>

namespace sw::kpu::gddr7 {

// Note: We use fully qualified gddr7::BankState:: to avoid ambiguity with
// IMemoryController::BankState which is brought in through inheritance.

// ============================================================================
// Constructor
// ============================================================================

GDDR7MemoryController::GDDR7MemoryController(const Config& config)
    : gddr7_config_(config)
{
    // Initialize bank groups for each bank
    // GDDR7: 16 banks = 4 bank groups × 4 banks per group
    const uint8_t num_ch = std::min<uint8_t>(gddr7_config_.num_channels, 2);
    for (uint8_t ch = 0; ch < num_ch; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            channels_[ch].banks[b].bank_group = b / 4;
        }
    }

    // Initialize interface config from GDDR7 config
    interface_config_.fidelity = sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    interface_config_.technology = sw::kpu::MemoryTechnology::GDDR7;
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

GDDR7MemoryController::GDDR7MemoryController(const sw::kpu::MemoryControllerConfig& config)
    : GDDR7MemoryController(Config{})  // Start with defaults
{
    // Copy relevant fields from interface config
    gddr7_config_.num_channels = config.num_channels;
    gddr7_config_.banks_per_channel = config.banks_per_channel;
    gddr7_config_.bank_groups = config.bank_groups;
    gddr7_config_.queue_depth = config.queue_depth;
    gddr7_config_.row_bits = config.row_bits;
    gddr7_config_.col_bits = config.col_bits;
    gddr7_config_.bank_bits = config.bank_bits;
    gddr7_config_.channel_bits = config.channel_bits;

    // Copy timing parameters
    gddr7_config_.timing.tRCDRD = config.timing.tRCD;
    gddr7_config_.timing.tRCDWR = config.timing.tRCD;
    gddr7_config_.timing.tRP = config.timing.tRP;
    gddr7_config_.timing.tRAS = config.timing.tRAS;
    gddr7_config_.timing.tRC = config.timing.tRC;
    gddr7_config_.timing.tRL = config.timing.tCL;
    gddr7_config_.timing.tWL = config.timing.tWL;
    gddr7_config_.timing.tWR = config.timing.tWR;
    gddr7_config_.timing.tRTP = config.timing.tRTP;
    gddr7_config_.timing.tRRD_L = config.timing.tRRD_L;
    gddr7_config_.timing.tRRD_S = config.timing.tRRD_S;
    gddr7_config_.timing.tCCD_L = config.timing.tCCD_L;
    gddr7_config_.timing.tCCD_S = config.timing.tCCD_S;
    gddr7_config_.timing.tWTR_L = config.timing.tWTR_L;
    gddr7_config_.timing.tWTR_S = config.timing.tWTR_S;
    gddr7_config_.timing.tRTW = config.timing.tRTW;
    gddr7_config_.timing.tFAW = config.timing.tFAW;

    // Store interface config
    interface_config_ = config;
    interface_config_.fidelity = sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    interface_config_.technology = sw::kpu::MemoryTechnology::GDDR7;

    // Set verification level based on config
    check_invariants_ = (config.verification == sw::kpu::VerificationLevel::INVARIANTS ||
                        config.verification == sw::kpu::VerificationLevel::PROTOCOL);
    tracing_enabled_ = config.enable_tracing;

    // Reinitialize bank groups
    for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            channels_[ch].banks[b].bank_group = b / 4;
        }
    }
}

// ============================================================================
// Request submission
// ============================================================================

std::optional<uint64_t> GDDR7MemoryController::submit_read(
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

std::optional<uint64_t> GDDR7MemoryController::submit_write(
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

void GDDR7MemoryController::tick() {
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

void GDDR7MemoryController::drain() {
    const uint64_t MAX_DRAIN_CYCLES = 100000;
    uint64_t start = current_cycle_;

    while (has_pending() && (current_cycle_ - start) < MAX_DRAIN_CYCLES) {
        tick();
    }
}

void GDDR7MemoryController::reset() {
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

bool GDDR7MemoryController::has_pending() const {
    return !pending_queue_.empty() || !active_requests_.empty();
}

bool GDDR7MemoryController::can_accept() const {
    return pending_queue_.size() < gddr7_config_.queue_depth;
}

size_t GDDR7MemoryController::pending_count() const {
    return pending_queue_.size() + active_requests_.size();
}

// Convert GDDR7 BankState to IMemoryController::BankState
static sw::kpu::IMemoryController::BankState to_interface_state(gddr7::BankState state) {
    switch (state) {
        case gddr7::BankState::IDLE:        return sw::kpu::IMemoryController::BankState::IDLE;
        case gddr7::BankState::ACTIVATING:  return sw::kpu::IMemoryController::BankState::ACTIVATING;
        case gddr7::BankState::ACTIVE:      return sw::kpu::IMemoryController::BankState::ACTIVE;
        case gddr7::BankState::READING:     return sw::kpu::IMemoryController::BankState::READING;
        case gddr7::BankState::WRITING:     return sw::kpu::IMemoryController::BankState::WRITING;
        case gddr7::BankState::PRECHARGING: return sw::kpu::IMemoryController::BankState::PRECHARGING;
        case gddr7::BankState::REFRESHING:  return sw::kpu::IMemoryController::BankState::REFRESHING;
        default:                            return sw::kpu::IMemoryController::BankState::IDLE;
    }
}

sw::kpu::IMemoryController::BankState GDDR7MemoryController::get_bank_state(
    uint8_t channel, uint8_t bank) const
{
    return to_interface_state(get_gddr7_bank_state(channel, bank));
}

gddr7::BankState GDDR7MemoryController::get_gddr7_bank_state(uint8_t channel, uint8_t bank) const {
    if (channel >= gddr7_config_.num_channels || bank >= 16) {
        return gddr7::BankState::IDLE;
    }
    return channels_[channel].banks[bank].state;
}

bool GDDR7MemoryController::is_bank_idle(uint8_t channel, uint8_t bank) const {
    return get_gddr7_bank_state(channel, bank) == gddr7::BankState::IDLE;
}

bool GDDR7MemoryController::is_row_open(uint8_t channel, uint8_t bank, uint32_t row) const {
    if (channel >= gddr7_config_.num_channels || bank >= 16) {
        return false;
    }
    const Bank& b = channels_[channel].banks[bank];
    return (b.state == gddr7::BankState::ACTIVE ||
            b.state == gddr7::BankState::READING ||
            b.state == gddr7::BankState::WRITING) &&
           b.open_row == row;
}

void GDDR7MemoryController::reset_stats() {
    stats_ = Statistics{};
    interface_stats_.reset();
}

void GDDR7MemoryController::sync_interface_stats() {
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

void GDDR7MemoryController::sync_interface_violations() {
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

void GDDR7MemoryController::decode_address(
    uint64_t address, uint8_t& channel, uint8_t& bank,
    uint32_t& row, uint32_t& col) const
{
    uint32_t byte_offset_bits = 6;  // 64-byte cache line
    uint32_t col_bits = gddr7_config_.col_bits;
    uint32_t bank_bits = gddr7_config_.bank_bits;
    uint32_t channel_bits = (gddr7_config_.num_channels > 1) ? 1 : 0;

    uint64_t addr = address >> byte_offset_bits;

    if (channel_bits > 0) {
        channel = static_cast<uint8_t>(addr & ((1 << channel_bits) - 1));
        addr >>= channel_bits;
    } else {
        channel = 0;
    }

    col = addr & ((1 << col_bits) - 1);
    addr >>= col_bits;

    bank = static_cast<uint8_t>(addr & ((1 << bank_bits) - 1));
    addr >>= bank_bits;

    row = addr & ((1 << gddr7_config_.row_bits) - 1);
}

// ============================================================================
// Burst cycles
// ============================================================================

uint32_t GDDR7MemoryController::burst_cycles() const {
    return gddr7_config_.timing.tBurst;
}

// ============================================================================
// tFAW tracking
// ============================================================================

bool GDDR7MemoryController::check_tFAW(uint8_t channel) const {
    const auto& ch = channels_[channel];
    const auto& timing = gddr7_config_.timing;

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

void GDDR7MemoryController::record_activate(uint8_t channel) {
    auto& ch = channels_[channel];
    ch.activate_window[ch.activate_index] = current_cycle_;
    ch.activate_index = (ch.activate_index + 1) % 4;
}

// ============================================================================
// Bank operation checks
// ============================================================================

bool GDDR7MemoryController::can_activate(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    if (b.state != gddr7::BankState::IDLE) {
        return false;
    }

    if (cycle < b.state_until) {
        return false;
    }

    if (cycle < b.last_activate + timing.tRC) {
        return false;
    }

    uint8_t bg = bank / 4;
    const BankGroup& bank_group = ch.bank_groups[bg];

    if (cycle < bank_group.last_activate + timing.tRRD_L) {
        return false;
    }

    if (!check_tFAW(channel)) {
        return false;
    }

    if (ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool GDDR7MemoryController::can_read(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    if (b.state != gddr7::BankState::ACTIVE) {
        return false;
    }

    if (cycle < b.state_until) {
        return false;
    }

    uint8_t bg = bank / 4;
    const BankGroup& bank_group = ch.bank_groups[bg];
    if (cycle < bank_group.last_cas + timing.tCCD_L) {
        return false;
    }

    if (ch.last_was_write) {
        if (cycle < bank_group.last_write + timing.tWL + burst_cycles() + timing.tWTR_L) {
            return false;
        }
    }

    if (ch.data_bus_state != DataBusState::IDLE) {
        return false;
    }

    if (ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool GDDR7MemoryController::can_write(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    if (b.state != gddr7::BankState::ACTIVE) {
        return false;
    }

    if (cycle < b.state_until) {
        return false;
    }

    uint8_t bg = bank / 4;
    const BankGroup& bank_group = ch.bank_groups[bg];
    if (cycle < bank_group.last_cas + timing.tCCD_L) {
        return false;
    }

    if (!ch.last_was_write && ch.data_bus_until > 0) {
        if (cycle < ch.data_bus_until + timing.tRTW) {
            return false;
        }
    }

    if (ch.data_bus_state != DataBusState::IDLE) {
        return false;
    }

    if (ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool GDDR7MemoryController::can_precharge(uint8_t channel, uint8_t bank, uint64_t cycle) const {
    const auto& ch = channels_[channel];
    const Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    if (b.state != gddr7::BankState::ACTIVE) {
        return false;
    }

    if (cycle < b.last_activate + timing.tRAS) {
        return false;
    }

    if (b.last_write_cmd > 0) {
        uint64_t write_done = b.last_write_cmd + timing.tWL + burst_cycles();
        if (cycle < write_done + timing.tWR) {
            return false;
        }
    }

    if (b.last_read_cmd > 0) {
        if (cycle < b.last_read_cmd + timing.tRTP) {
            return false;
        }
    }

    return true;
}

bool GDDR7MemoryController::can_refresh(uint8_t channel, uint8_t bank) const {
    return channels_[channel].banks[bank].state == gddr7::BankState::IDLE;
}

// ============================================================================
// Bank operations
// ============================================================================

void GDDR7MemoryController::do_activate(uint8_t channel, uint8_t bank, uint32_t row, uint64_t request_id) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    b.state = gddr7::BankState::ACTIVATING;
    b.open_row = row;
    b.state_until = current_cycle_ + timing.tRCDRD;
    b.last_activate = current_cycle_;
    b.page_opener_request_id = request_id;

    uint8_t bg = bank / 4;
    ch.bank_groups[bg].last_activate = current_cycle_;

    record_activate(channel);

    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    trace_bank_state_change(channel, bank, gddr7::BankState::ACTIVATING, "ACTIVATE row " + std::to_string(row));
    trace_command(channel, bank, "ACTIVATE", timing.tRCDRD, request_id);
    trace_bus_state_change(channel, false, "BUSY", "ACT cmd");
}

void GDDR7MemoryController::do_read(uint8_t channel, uint8_t bank, MemoryRequest& req) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    b.state = gddr7::BankState::READING;
    b.last_read_cmd = current_cycle_;
    b.burst_end = current_cycle_ + timing.tRL + burst_cycles();

    uint8_t bg = bank / 4;
    ch.bank_groups[bg].last_cas = current_cycle_;

    if (ch.last_was_write) {
        stats_.write_to_read_turnarounds++;
    }

    ch.data_bus_state = DataBusState::READ_BURST;
    ch.data_bus_until = b.burst_end;
    ch.last_was_write = false;

    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    req.issue_cycle = current_cycle_;
    req.complete_cycle = b.burst_end;

    uint64_t duration = timing.tRL + burst_cycles();
    trace_bank_state_change(channel, bank, gddr7::BankState::READING, "READ burst");
    trace_command(channel, bank, "READ", duration, req.id);
    trace_bus_state_change(channel, true, "BUSY", "READ burst");
    trace_bus_state_change(channel, false, "BUSY", "RD cmd");
}

void GDDR7MemoryController::do_write(uint8_t channel, uint8_t bank, MemoryRequest& req) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    b.state = gddr7::BankState::WRITING;
    b.last_write_cmd = current_cycle_;
    b.burst_end = current_cycle_ + timing.tWL + burst_cycles();

    uint8_t bg = bank / 4;
    ch.bank_groups[bg].last_cas = current_cycle_;
    ch.bank_groups[bg].last_write = current_cycle_;

    ch.data_bus_state = DataBusState::WRITE_BURST;
    ch.data_bus_until = b.burst_end;

    if (!ch.last_was_write) {
        stats_.read_to_write_turnarounds++;
    }
    ch.last_was_write = true;

    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    req.issue_cycle = current_cycle_;
    req.complete_cycle = b.burst_end;

    uint64_t duration = timing.tWL + burst_cycles();
    trace_bank_state_change(channel, bank, gddr7::BankState::WRITING, "WRITE burst");
    trace_command(channel, bank, "WRITE", duration, req.id);
    trace_bus_state_change(channel, true, "BUSY", "WRITE burst");
    trace_bus_state_change(channel, false, "BUSY", "WR cmd");
}

void GDDR7MemoryController::do_precharge(uint8_t channel, uint8_t bank) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    uint64_t request_id = b.page_opener_request_id;

    b.state = gddr7::BankState::PRECHARGING;
    b.state_until = current_cycle_ + timing.tRP;

    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    trace_bank_state_change(channel, bank, gddr7::BankState::PRECHARGING, "PRECHARGE");
    trace_command(channel, bank, "PRECHARGE", timing.tRP, request_id);
    trace_bus_state_change(channel, false, "BUSY", "PRE cmd");
}

void GDDR7MemoryController::do_refresh(uint8_t channel, uint8_t bank) {
    auto& ch = channels_[channel];
    Bank& b = ch.banks[bank];
    const auto& timing = gddr7_config_.timing;

    b.state = gddr7::BankState::REFRESHING;
    b.state_until = current_cycle_ + timing.tRFCpb;
    b.last_refresh = current_cycle_;

    stats_.refreshes++;

    ch.cmd_bus_state = CommandBusState::BUSY;
    ch.cmd_bus_until = current_cycle_ + 1;

    trace_bank_state_change(channel, bank, gddr7::BankState::REFRESHING, "REFRESH");
    trace_command(channel, bank, "REFRESH", timing.tRFCpb, 0);
    trace_bus_state_change(channel, false, "BUSY", "REF cmd");
}

// ============================================================================
// State updates
// ============================================================================

void GDDR7MemoryController::update_bank_states() {
    for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
        for (uint8_t b = 0; b < 16; ++b) {
            Bank& bank = channels_[ch].banks[b];

            if (current_cycle_ >= bank.state_until) {
                switch (bank.state) {
                    case gddr7::BankState::ACTIVATING:
                        bank.state = gddr7::BankState::ACTIVE;
                        trace_bank_state_change(ch, b, gddr7::BankState::ACTIVE, "tRCD complete");
                        break;
                    case gddr7::BankState::PRECHARGING:
                        bank.state = gddr7::BankState::IDLE;
                        bank.open_row = 0;
                        bank.last_read_cmd = 0;
                        bank.last_write_cmd = 0;
                        trace_bank_state_change(ch, b, gddr7::BankState::IDLE, "tRP complete");
                        break;
                    case gddr7::BankState::REFRESHING:
                        bank.state = gddr7::BankState::IDLE;
                        trace_bank_state_change(ch, b, gddr7::BankState::IDLE, "tRFCpb complete");
                        break;
                    default:
                        break;
                }
            }

            if (current_cycle_ >= bank.burst_end) {
                if (bank.state == gddr7::BankState::READING || bank.state == gddr7::BankState::WRITING) {
                    bank.state = gddr7::BankState::ACTIVE;
                    trace_bank_state_change(ch, b, gddr7::BankState::ACTIVE, "burst complete");
                }
            }
        }
    }
}

void GDDR7MemoryController::update_bus_states() {
    for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
        Channel& channel = channels_[ch];

        if (current_cycle_ >= channel.cmd_bus_until &&
            channel.cmd_bus_state != CommandBusState::IDLE) {
            channel.cmd_bus_state = CommandBusState::IDLE;
            channel.deferred_cmd_bus_idle_trace = true;
        }

        if (current_cycle_ >= channel.data_bus_until &&
            channel.data_bus_state != DataBusState::IDLE) {
            channel.data_bus_state = DataBusState::IDLE;
            channel.deferred_data_bus_idle_trace = true;
        }
    }
}

void GDDR7MemoryController::finalize_bus_traces() {
    for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
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

void GDDR7MemoryController::handle_refresh() {
    // DISABLED mode: no refresh at all
    if (refresh_mode_ == RefreshMode::DISABLED) {
        return;
    }

    const auto& timing = gddr7_config_.timing;

    // Deadline enforcement (applies to all modes except DISABLED)
    if (deadline_enforcement_) {
        for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
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
            for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
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
                for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
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
            for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
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

void GDDR7MemoryController::schedule_requests() {
    if (pending_queue_.empty()) {
        return;
    }

    MemoryRequest& req = pending_queue_.front();

    if (try_issue_request(req)) {
        active_requests_.push_back(std::move(pending_queue_.front()));
        pending_queue_.pop();
    } else {
        stats_.stall_cycles++;
    }
}

bool GDDR7MemoryController::try_issue_request(MemoryRequest& req) {
    uint8_t ch = req.channel;
    uint8_t bank = req.bank;
    Bank& b = channels_[ch].banks[bank];

    switch (b.state) {
        case gddr7::BankState::IDLE:
            if (can_activate(ch, bank, current_cycle_)) {
                do_activate(ch, bank, req.row, req.id);
                if (!req.triggered_conflict) {
                    stats_.page_empty++;
                }
                req.triggered_activate = true;
                return false;
            }
            return false;

        case gddr7::BankState::ACTIVATING:
            return false;

        case gddr7::BankState::ACTIVE:
            if (b.open_row == req.row) {
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
                if (can_precharge(ch, bank, current_cycle_)) {
                    do_precharge(ch, bank);
                    stats_.page_conflicts++;
                    req.triggered_conflict = true;
                }
            }
            return false;

        case gddr7::BankState::READING:
        case gddr7::BankState::WRITING:
        case gddr7::BankState::PRECHARGING:
        case gddr7::BankState::REFRESHING:
            return false;

        default:
            return false;
    }
}

// ============================================================================
// Request completion
// ============================================================================

void GDDR7MemoryController::complete_requests() {
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        if (current_cycle_ >= it->complete_cycle && it->complete_cycle > 0) {
            uint64_t latency = current_cycle_ - it->submit_cycle;
            stats_.total_latency += latency;

            if (it->type == RequestType::READ) {
                stats_.read_latency_total += latency;
                stats_.read_latency_min = std::min(stats_.read_latency_min, latency);
                stats_.read_latency_max = std::max(stats_.read_latency_max, latency);
            } else {
                stats_.write_latency_total += latency;
                stats_.write_latency_min = std::min(stats_.write_latency_min, latency);
                stats_.write_latency_max = std::max(stats_.write_latency_max, latency);
            }

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

void GDDR7MemoryController::check_all_invariants() {
    for (uint8_t ch = 0; ch < gddr7_config_.num_channels; ++ch) {
        check_timing_invariants(ch);
        check_bus_invariants(ch);

        for (uint8_t b = 0; b < 16; ++b) {
            check_bank_invariants(ch, b);
        }
    }
}

void GDDR7MemoryController::check_bank_invariants(uint8_t channel, uint8_t bank) {
    const Bank& b = channels_[channel].banks[bank];
    const auto& timing = gddr7_config_.timing;

    if (b.state == gddr7::BankState::ACTIVE && current_cycle_ < b.state_until) {
        report_violation("INV-BANK-2",
            "Bank transitioned to ACTIVE before tRCD elapsed",
            channel, bank);
    }

    if (b.state == gddr7::BankState::PRECHARGING) {
        if (b.last_activate > 0 &&
            current_cycle_ < b.last_activate + timing.tRAS) {
            report_violation("INV-BANK-4",
                "PRECHARGE issued before tRAS elapsed",
                channel, bank);
        }
    }

    if (b.state == gddr7::BankState::PRECHARGING && b.last_write_cmd > 0) {
        uint64_t write_done = b.last_write_cmd + timing.tWL + burst_cycles();
        if (current_cycle_ < write_done + timing.tWR) {
            report_violation("INV-BANK-5",
                "PRECHARGE issued before tWR elapsed after write",
                channel, bank);
        }
    }

    if (b.state == gddr7::BankState::PRECHARGING && b.last_read_cmd > 0) {
        if (current_cycle_ < b.last_read_cmd + timing.tRTP) {
            report_violation("INV-BANK-6",
                "PRECHARGE issued before tRTP elapsed after read",
                channel, bank);
        }
    }
}

void GDDR7MemoryController::check_timing_invariants(uint8_t channel) {
    const auto& ch = channels_[channel];
    const auto& timing = gddr7_config_.timing;

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

void GDDR7MemoryController::check_bus_invariants(uint8_t channel) {
    const auto& ch = channels_[channel];

    if (ch.data_bus_state == DataBusState::READ_BURST &&
        ch.last_was_write) {
        // State tracking inconsistency
    }
}

void GDDR7MemoryController::report_violation(
    const std::string& id, const std::string& msg,
    uint8_t channel, uint8_t bank)
{
    gddr7::InvariantViolation v;
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

void GDDR7MemoryController::trace_bank_state_change(
    uint8_t channel, uint8_t bank, gddr7::BankState new_state, const std::string& reason)
{
    if (!tracing_enabled_) return;

    if (resource_tracker_) {
        uint32_t bank_id = channel * 16 + bank;

        sw::trace::ResourceState rs;
        switch (new_state) {
            case gddr7::BankState::IDLE:
                rs = sw::trace::ResourceState::IDLE;
                break;
            case gddr7::BankState::ACTIVATING:
            case gddr7::BankState::PRECHARGING:
            case gddr7::BankState::REFRESHING:
                rs = sw::trace::ResourceState::STALLED;
                break;
            case gddr7::BankState::ACTIVE:
            case gddr7::BankState::READING:
            case gddr7::BankState::WRITING:
                rs = sw::trace::ResourceState::BUSY;
                break;
            default:
                rs = sw::trace::ResourceState::IDLE;
                break;
        }

        resource_tracker_->transition(
            sw::trace::ComponentType::GDDR7_BANK,
            bank_id,
            rs,
            current_cycle_,
            0,
            reason
        );
    }
}

void GDDR7MemoryController::trace_bus_state_change(
    uint8_t channel, bool is_data_bus, const std::string& state, const std::string& reason)
{
    if (!tracing_enabled_) return;

    if (resource_tracker_) {
        sw::trace::ComponentType bus_type = is_data_bus ?
            sw::trace::ComponentType::GDDR7_DATA_BUS :
            sw::trace::ComponentType::GDDR7_CMD_BUS;

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

void GDDR7MemoryController::trace_command(
    uint8_t channel, uint8_t bank, const std::string& cmd,
    uint64_t duration, uint64_t request_id)
{
    if (!tracing_enabled_) return;

    uint32_t bank_id = channel * 16 + bank;

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
        sw::trace::ComponentType::GDDR7_BANK,
        bank_id,
        trans_type,
        request_id
    );

    entry.complete(current_cycle_ + duration);
    entry.description = cmd + " Ch" + std::to_string(channel) + " Bank" + std::to_string(bank);

    sw::trace::MemoryPayload payload;
    payload.location = sw::trace::MemoryLocation(0, 0, bank_id, sw::trace::ComponentType::GDDR7_BANK);
    payload.latency_cycles = static_cast<uint32_t>(duration);
    entry.payload = payload;

    trace_entries_.push_back(entry);
}

// ============================================================================
// Refresh Control Implementation
// ============================================================================

void GDDR7MemoryController::set_refresh_mode(RefreshMode mode) {
    refresh_mode_ = mode;
    if (mode == RefreshMode::INTERVAL) {
        last_interval_refresh_ = current_cycle_;
    }
}

void GDDR7MemoryController::set_refresh_interval(uint64_t cycles) {
    refresh_interval_ = cycles;
    last_interval_refresh_ = current_cycle_;
}

uint64_t GDDR7MemoryController::cycles_until_deadline(uint8_t channel, uint8_t bank) const {
    if (channel >= gddr7_config_.num_channels || bank >= 16) {
        return UINT64_MAX;
    }

    const auto& timing = gddr7_config_.timing;
    const Bank& b = channels_[channel].banks[bank];

    // Deadline is 16 * tREFI after last refresh
    uint64_t deadline = b.last_refresh + 16 * timing.tREFI;

    if (current_cycle_ >= deadline) {
        return 0;
    }

    return deadline - current_cycle_;
}

bool GDDR7MemoryController::refresh_pending(uint8_t channel, uint8_t bank) const {
    if (channel >= gddr7_config_.num_channels || bank >= 16) {
        return false;
    }

    const auto& timing = gddr7_config_.timing;
    const Bank& b = channels_[channel].banks[bank];

    return current_cycle_ >= b.last_refresh + timing.tREFI;
}

uint32_t GDDR7MemoryController::refresh_debt(uint8_t channel, uint8_t bank) const {
    if (channel >= gddr7_config_.num_channels || bank >= 16) {
        return 0;
    }

    const auto& timing = gddr7_config_.timing;
    const Bank& b = channels_[channel].banks[bank];

    if (current_cycle_ <= b.last_refresh) {
        return 0;
    }

    uint64_t elapsed = current_cycle_ - b.last_refresh;
    return static_cast<uint32_t>(elapsed / timing.tREFI);
}

bool GDDR7MemoryController::inject_refresh(uint8_t channel, int8_t bank) {
    if (channel >= gddr7_config_.num_channels) {
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

} // namespace sw::kpu::gddr7
