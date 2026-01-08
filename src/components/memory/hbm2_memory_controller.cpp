// ============================================================================
// src/components/memory/hbm2_memory_controller.cpp
// HBM2 Memory Controller implementation with formal invariant checking
// ============================================================================

#include <sw/kpu/components/hbm2_memory_controller.hpp>
#include <sw/trace/trace_entry.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <cassert>

namespace sw::kpu::hbm2 {

// ============================================================================
// Constructor
// ============================================================================

HBM2MemoryController::HBM2MemoryController(const Config& config)
    : hbm2_config_(config)
{
    // Initialize bank groups for each bank in each pseudo-channel
    for (uint8_t ch = 0; ch < hbm2_config_.num_channels; ++ch) {
        for (uint8_t pc = 0; pc < hbm2_config_.pseudo_channels_per_channel; ++pc) {
            for (uint8_t b = 0; b < 16; ++b) {
                channels_[ch].pseudo_channels[pc].banks[b].bank_group = b / 4;
            }
        }
    }

    // Initialize interface config from HBM2 config
    interface_config_.fidelity = sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    interface_config_.technology = sw::kpu::MemoryTechnology::HBM2;
    interface_config_.verification = sw::kpu::VerificationLevel::INVARIANTS;
    interface_config_.num_channels = config.num_channels;
    interface_config_.banks_per_channel = config.pseudo_channels_per_channel * config.banks_per_pseudo_channel;
    interface_config_.bank_groups = config.bank_groups;
    interface_config_.queue_depth = config.queue_depth;
    interface_config_.row_bits = config.row_bits;
    interface_config_.col_bits = config.col_bits;
    interface_config_.bank_bits = config.bank_bits;
    interface_config_.channel_bits = config.channel_bits;
}

HBM2MemoryController::HBM2MemoryController(const sw::kpu::MemoryControllerConfig& config)
    : HBM2MemoryController(Config{})
{
    // Copy relevant fields from interface config
    hbm2_config_.num_channels = config.num_channels;
    hbm2_config_.queue_depth = config.queue_depth;
    hbm2_config_.row_bits = config.row_bits;
    hbm2_config_.col_bits = config.col_bits;
    hbm2_config_.bank_bits = config.bank_bits;
    hbm2_config_.channel_bits = config.channel_bits;

    // Copy timing parameters
    hbm2_config_.timing.tRCDRD = config.timing.tRCD;
    hbm2_config_.timing.tRCDWR = config.timing.tRCD / 2;  // HBM2 has faster write path
    hbm2_config_.timing.tRP = config.timing.tRP;
    hbm2_config_.timing.tRAS = config.timing.tRAS;
    hbm2_config_.timing.tRC = config.timing.tRC;
    hbm2_config_.timing.tRL = config.timing.tCL;
    hbm2_config_.timing.tWL = config.timing.tWL;
    hbm2_config_.timing.tWR = config.timing.tWR;
    hbm2_config_.timing.tRTP = config.timing.tRTP;
    hbm2_config_.timing.tRRD_L = config.timing.tRRD_L;
    hbm2_config_.timing.tRRD_S = config.timing.tRRD_S;
    hbm2_config_.timing.tCCD_L = config.timing.tCCD_L;
    hbm2_config_.timing.tCCD_S = config.timing.tCCD_S;
    hbm2_config_.timing.tWTR_L = config.timing.tWTR_L;
    hbm2_config_.timing.tWTR_S = config.timing.tWTR_S;
    hbm2_config_.timing.tRTW = config.timing.tRTW;
    hbm2_config_.timing.tFAW = config.timing.tFAW;

    // Store interface config
    interface_config_ = config;
    interface_config_.fidelity = sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    interface_config_.technology = sw::kpu::MemoryTechnology::HBM2;

    // Set verification level based on config
    check_invariants_ = (config.verification == sw::kpu::VerificationLevel::INVARIANTS ||
                        config.verification == sw::kpu::VerificationLevel::PROTOCOL);
    tracing_enabled_ = config.enable_tracing;

    // Reinitialize bank groups
    for (uint8_t ch = 0; ch < hbm2_config_.num_channels; ++ch) {
        for (uint8_t pc = 0; pc < hbm2_config_.pseudo_channels_per_channel; ++pc) {
            for (uint8_t b = 0; b < 16; ++b) {
                channels_[ch].pseudo_channels[pc].banks[b].bank_group = b / 4;
            }
        }
    }
}

// ============================================================================
// Request submission
// ============================================================================

std::optional<uint64_t> HBM2MemoryController::submit_read(
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

    decode_address(address, req.channel, req.pseudo_channel, req.bank, req.row, req.col);

    pending_queue_.push(std::move(req));
    stats_.reads++;

    // Track channel and pseudo-channel usage
    if (req.channel < 8) {
        stats_.channel_accesses[req.channel]++;
        uint8_t pc_idx = req.channel * 2 + req.pseudo_channel;
        if (pc_idx < 16) {
            stats_.pseudo_channel_accesses[pc_idx]++;
        }
    }

    return req.id;
}

std::optional<uint64_t> HBM2MemoryController::submit_write(
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

    decode_address(address, req.channel, req.pseudo_channel, req.bank, req.row, req.col);

    pending_queue_.push(std::move(req));
    stats_.writes++;

    // Track channel and pseudo-channel usage
    if (req.channel < 8) {
        stats_.channel_accesses[req.channel]++;
        uint8_t pc_idx = req.channel * 2 + req.pseudo_channel;
        if (pc_idx < 16) {
            stats_.pseudo_channel_accesses[pc_idx]++;
        }
    }

    return req.id;
}

// ============================================================================
// Simulation interface
// ============================================================================

void HBM2MemoryController::tick() {
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

void HBM2MemoryController::drain() {
    const uint64_t MAX_DRAIN_CYCLES = 100000;
    uint64_t start = current_cycle_;

    while (has_pending() && (current_cycle_ - start) < MAX_DRAIN_CYCLES) {
        tick();
    }
}

void HBM2MemoryController::reset() {
    for (uint8_t ch = 0; ch < 8; ++ch) {
        for (uint8_t pc = 0; pc < 2; ++pc) {
            channels_[ch].pseudo_channels[pc] = PseudoChannel{};
            for (uint8_t b = 0; b < 16; ++b) {
                channels_[ch].pseudo_channels[pc].banks[b].bank_group = b / 4;
            }
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

bool HBM2MemoryController::has_pending() const {
    return !pending_queue_.empty() || !active_requests_.empty();
}

bool HBM2MemoryController::can_accept() const {
    return pending_queue_.size() < hbm2_config_.queue_depth;
}

size_t HBM2MemoryController::pending_count() const {
    return pending_queue_.size() + active_requests_.size();
}

// Convert HBM2 BankState to IMemoryController::BankState
static sw::kpu::IMemoryController::BankState to_interface_state(hbm2::BankState state) {
    switch (state) {
        case hbm2::BankState::IDLE:        return sw::kpu::IMemoryController::BankState::IDLE;
        case hbm2::BankState::ACTIVATING:  return sw::kpu::IMemoryController::BankState::ACTIVATING;
        case hbm2::BankState::ACTIVE:      return sw::kpu::IMemoryController::BankState::ACTIVE;
        case hbm2::BankState::READING:     return sw::kpu::IMemoryController::BankState::READING;
        case hbm2::BankState::WRITING:     return sw::kpu::IMemoryController::BankState::WRITING;
        case hbm2::BankState::PRECHARGING: return sw::kpu::IMemoryController::BankState::PRECHARGING;
        case hbm2::BankState::REFRESHING:  return sw::kpu::IMemoryController::BankState::REFRESHING;
        default:                           return sw::kpu::IMemoryController::BankState::IDLE;
    }
}

sw::kpu::IMemoryController::BankState HBM2MemoryController::get_bank_state(
    uint8_t channel, uint8_t bank) const
{
    // Map flat bank index to pseudo-channel + bank
    uint8_t pc = bank / 16;
    uint8_t local_bank = bank % 16;
    return to_interface_state(get_hbm2_bank_state(channel, pc, local_bank));
}

hbm2::BankState HBM2MemoryController::get_hbm2_bank_state(uint8_t channel, uint8_t pc, uint8_t bank) const {
    if (channel >= hbm2_config_.num_channels || pc >= 2 || bank >= 16) {
        return hbm2::BankState::IDLE;
    }
    return channels_[channel].pseudo_channels[pc].banks[bank].state;
}

bool HBM2MemoryController::is_bank_idle(uint8_t channel, uint8_t pc, uint8_t bank) const {
    return get_hbm2_bank_state(channel, pc, bank) == hbm2::BankState::IDLE;
}

bool HBM2MemoryController::is_row_open(uint8_t channel, uint8_t bank, uint32_t row) const {
    // Map flat bank index to pseudo-channel + bank
    uint8_t pc = bank / 16;
    uint8_t local_bank = bank % 16;

    if (channel >= hbm2_config_.num_channels || pc >= 2 || local_bank >= 16) {
        return false;
    }
    const Bank& b = channels_[channel].pseudo_channels[pc].banks[local_bank];
    return (b.state == hbm2::BankState::ACTIVE ||
            b.state == hbm2::BankState::READING ||
            b.state == hbm2::BankState::WRITING) &&
           b.open_row == row;
}

void HBM2MemoryController::reset_stats() {
    stats_ = Statistics{};
    interface_stats_.reset();
}

void HBM2MemoryController::sync_interface_stats() {
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

void HBM2MemoryController::sync_interface_violations() {
    interface_violations_.clear();
    for (const auto& v : violations_) {
        // Map pseudo-channel + bank to flat bank index
        uint8_t flat_bank = v.pseudo_channel * 16 + v.bank;
        interface_violations_.push_back({
            v.cycle,
            v.invariant_id,
            v.message,
            v.channel,
            flat_bank
        });
    }
}

// ============================================================================
// Address decoding
// ============================================================================

void HBM2MemoryController::decode_address(
    uint64_t address, uint8_t& channel, uint8_t& pc, uint8_t& bank,
    uint32_t& row, uint32_t& col) const
{
    // HBM2 address mapping:
    // [row | bank | col | pseudo_channel | channel | byte_offset]
    uint32_t byte_offset_bits = 5;  // 32-byte minimum access
    uint32_t col_bits = hbm2_config_.col_bits;
    uint32_t bank_bits = hbm2_config_.bank_bits;
    uint32_t pc_bits = hbm2_config_.pc_bits;
    uint32_t channel_bits = hbm2_config_.channel_bits;

    uint64_t addr = address >> byte_offset_bits;

    // Extract channel
    if (channel_bits > 0) {
        channel = addr & ((1 << channel_bits) - 1);
        addr >>= channel_bits;
    } else {
        channel = 0;
    }

    // Extract pseudo-channel
    if (pc_bits > 0) {
        pc = addr & ((1 << pc_bits) - 1);
        addr >>= pc_bits;
    } else {
        pc = 0;
    }

    // Extract column
    col = addr & ((1 << col_bits) - 1);
    addr >>= col_bits;

    // Extract bank
    bank = addr & ((1 << bank_bits) - 1);
    addr >>= bank_bits;

    // Extract row
    row = addr & ((1 << hbm2_config_.row_bits) - 1);
}

uint64_t HBM2MemoryController::encode_address(uint8_t channel, uint8_t pc, uint8_t bank,
                                               uint32_t row, uint32_t col) const
{
    uint32_t byte_offset_bits = 5;
    uint32_t col_bits = hbm2_config_.col_bits;
    uint32_t bank_bits = hbm2_config_.bank_bits;
    uint32_t pc_bits = hbm2_config_.pc_bits;
    uint32_t channel_bits = hbm2_config_.channel_bits;

    uint64_t addr = row;
    addr = (addr << bank_bits) | bank;
    addr = (addr << col_bits) | col;
    addr = (addr << pc_bits) | pc;
    addr = (addr << channel_bits) | channel;
    addr <<= byte_offset_bits;

    return addr;
}

// ============================================================================
// Burst cycles
// ============================================================================

uint32_t HBM2MemoryController::burst_cycles() const {
    // HBM2 BL4: 2 CK cycles for burst
    return hbm2_config_.timing.tBurst;
}

// ============================================================================
// tFAW tracking
// ============================================================================

bool HBM2MemoryController::check_tFAW(uint8_t channel, uint8_t pc) const {
    const auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    const auto& timing = hbm2_config_.timing;

    uint64_t oldest = pseudo_ch.activate_window[0];
    for (int i = 1; i < 4; ++i) {
        if (pseudo_ch.activate_window[i] < oldest && pseudo_ch.activate_window[i] > 0) {
            oldest = pseudo_ch.activate_window[i];
        }
    }

    if (oldest > 0 && current_cycle_ < oldest + timing.tFAW) {
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            if (pseudo_ch.activate_window[i] > 0 &&
                current_cycle_ - pseudo_ch.activate_window[i] < timing.tFAW) {
                count++;
            }
        }
        if (count >= 4) {
            return false;
        }
    }
    return true;
}

void HBM2MemoryController::record_activate(uint8_t channel, uint8_t pc) {
    auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    pseudo_ch.activate_window[pseudo_ch.activate_index] = current_cycle_;
    pseudo_ch.activate_index = (pseudo_ch.activate_index + 1) % 4;
}

// ============================================================================
// Bank operation checks
// ============================================================================

bool HBM2MemoryController::can_activate(uint8_t channel, uint8_t pc, uint8_t bank, uint64_t cycle) const {
    const auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    const Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    // Bank must be IDLE
    if (b.state != hbm2::BankState::IDLE) {
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
    const BankGroup& bank_group = pseudo_ch.bank_groups[bg];

    // tRRD_L: same bank group
    if (cycle < bank_group.last_activate + timing.tRRD_L) {
        return false;
    }

    // Check tFAW
    if (!check_tFAW(channel, pc)) {
        return false;
    }

    // Command bus available
    if (pseudo_ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool HBM2MemoryController::can_read(uint8_t channel, uint8_t pc, uint8_t bank, uint64_t cycle) const {
    const auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    const Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    // Bank must be ACTIVE
    if (b.state != hbm2::BankState::ACTIVE) {
        return false;
    }

    // tRCDRD must have elapsed
    if (cycle < b.state_until) {
        return false;
    }

    // Bank group CAS timing (tCCD_L)
    uint8_t bg = bank / 4;
    const BankGroup& bank_group = pseudo_ch.bank_groups[bg];
    if (cycle < bank_group.last_cas + timing.tCCD_L) {
        return false;
    }

    // Write-to-read turnaround (tWTR_L)
    if (pseudo_ch.last_was_write) {
        if (cycle < bank_group.last_write + timing.tWL + burst_cycles() + timing.tWTR_L) {
            return false;
        }
    }

    // Data bus available
    if (pseudo_ch.data_bus_state != DataBusState::IDLE) {
        return false;
    }

    // Command bus available
    if (pseudo_ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool HBM2MemoryController::can_write(uint8_t channel, uint8_t pc, uint8_t bank, uint64_t cycle) const {
    const auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    const Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    // Bank must be ACTIVE
    if (b.state != hbm2::BankState::ACTIVE) {
        return false;
    }

    // tRCDWR must have elapsed
    if (cycle < b.state_until) {
        return false;
    }

    // Bank group CAS timing (tCCD_L)
    uint8_t bg = bank / 4;
    const BankGroup& bank_group = pseudo_ch.bank_groups[bg];
    if (cycle < bank_group.last_cas + timing.tCCD_L) {
        return false;
    }

    // Read-to-write turnaround (tRTW)
    if (!pseudo_ch.last_was_write && pseudo_ch.data_bus_until > 0) {
        if (cycle < pseudo_ch.data_bus_until + timing.tRTW) {
            return false;
        }
    }

    // Data bus available
    if (pseudo_ch.data_bus_state != DataBusState::IDLE) {
        return false;
    }

    // Command bus available
    if (pseudo_ch.cmd_bus_state != CommandBusState::IDLE) {
        return false;
    }

    return true;
}

bool HBM2MemoryController::can_precharge(uint8_t channel, uint8_t pc, uint8_t bank, uint64_t cycle) const {
    const auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    const Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    // Bank must be ACTIVE
    if (b.state != hbm2::BankState::ACTIVE) {
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

bool HBM2MemoryController::can_refresh(uint8_t channel, uint8_t pc, uint8_t bank) const {
    return channels_[channel].pseudo_channels[pc].banks[bank].state == hbm2::BankState::IDLE;
}

// ============================================================================
// Bank operations
// ============================================================================

void HBM2MemoryController::do_activate(uint8_t channel, uint8_t pc, uint8_t bank, uint32_t row, uint64_t request_id) {
    auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    b.state = hbm2::BankState::ACTIVATING;
    b.open_row = row;
    b.state_until = current_cycle_ + timing.tRCDRD;
    b.last_activate = current_cycle_;

    // Track which request opened this page
    b.page_opener_request_id = request_id;

    // Update bank group
    uint8_t bg = bank / 4;
    pseudo_ch.bank_groups[bg].last_activate = current_cycle_;

    // Record for tFAW
    record_activate(channel, pc);

    // Command bus busy for 1 cycle
    pseudo_ch.cmd_bus_state = CommandBusState::BUSY;
    pseudo_ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation
    trace_bank_state_change(channel, pc, bank, hbm2::BankState::ACTIVATING, "ACTIVATE row " + std::to_string(row));
    trace_command(channel, pc, bank, "ACTIVATE", timing.tRCDRD, request_id);
    trace_bus_state_change(channel, pc, false, "BUSY", "ACT cmd");
}

void HBM2MemoryController::do_read(uint8_t channel, uint8_t pc, uint8_t bank, MemoryRequest& req) {
    auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    b.state = hbm2::BankState::READING;
    b.last_read_cmd = current_cycle_;
    b.burst_end = current_cycle_ + timing.tRL + burst_cycles();

    // Update bank group
    uint8_t bg = bank / 4;
    pseudo_ch.bank_groups[bg].last_cas = current_cycle_;

    // Track turnarounds
    if (pseudo_ch.last_was_write) {
        stats_.write_to_read_turnarounds++;
    }

    // Data bus will be busy after tRL
    pseudo_ch.data_bus_state = DataBusState::READ_BURST;
    pseudo_ch.data_bus_until = b.burst_end;
    pseudo_ch.last_was_write = false;

    // Command bus busy for 1 cycle
    pseudo_ch.cmd_bus_state = CommandBusState::BUSY;
    pseudo_ch.cmd_bus_until = current_cycle_ + 1;

    // Set request completion time
    req.issue_cycle = current_cycle_;
    req.complete_cycle = b.burst_end;

    // Trace the operation
    uint64_t duration = timing.tRL + burst_cycles();
    trace_bank_state_change(channel, pc, bank, hbm2::BankState::READING, "READ burst");
    trace_command(channel, pc, bank, "READ", duration, req.id);
    trace_bus_state_change(channel, pc, true, "BUSY", "READ burst");
    trace_bus_state_change(channel, pc, false, "BUSY", "RD cmd");
}

void HBM2MemoryController::do_write(uint8_t channel, uint8_t pc, uint8_t bank, MemoryRequest& req) {
    auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    b.state = hbm2::BankState::WRITING;
    b.last_write_cmd = current_cycle_;
    b.burst_end = current_cycle_ + timing.tWL + burst_cycles();

    // Update bank group
    uint8_t bg = bank / 4;
    pseudo_ch.bank_groups[bg].last_cas = current_cycle_;
    pseudo_ch.bank_groups[bg].last_write = current_cycle_;

    // Data bus busy after tWL
    pseudo_ch.data_bus_state = DataBusState::WRITE_BURST;
    pseudo_ch.data_bus_until = b.burst_end;

    // Track turnarounds
    if (!pseudo_ch.last_was_write) {
        stats_.read_to_write_turnarounds++;
    }
    pseudo_ch.last_was_write = true;

    // Command bus busy for 1 cycle
    pseudo_ch.cmd_bus_state = CommandBusState::BUSY;
    pseudo_ch.cmd_bus_until = current_cycle_ + 1;

    // Set request completion time
    req.issue_cycle = current_cycle_;
    req.complete_cycle = b.burst_end;

    // Trace the operation
    uint64_t duration = timing.tWL + burst_cycles();
    trace_bank_state_change(channel, pc, bank, hbm2::BankState::WRITING, "WRITE burst");
    trace_command(channel, pc, bank, "WRITE", duration, req.id);
    trace_bus_state_change(channel, pc, true, "BUSY", "WRITE burst");
    trace_bus_state_change(channel, pc, false, "BUSY", "WR cmd");
}

void HBM2MemoryController::do_precharge(uint8_t channel, uint8_t pc, uint8_t bank) {
    auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    // Capture the request_id before clearing
    uint64_t request_id = b.page_opener_request_id;

    b.state = hbm2::BankState::PRECHARGING;
    b.state_until = current_cycle_ + timing.tRP;

    // Command bus busy for 1 cycle
    pseudo_ch.cmd_bus_state = CommandBusState::BUSY;
    pseudo_ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation with the correct request_id
    trace_bank_state_change(channel, pc, bank, hbm2::BankState::PRECHARGING, "PRECHARGE");
    trace_command(channel, pc, bank, "PRECHARGE", timing.tRP, request_id);
    trace_bus_state_change(channel, pc, false, "BUSY", "PRE cmd");
}

void HBM2MemoryController::do_refresh(uint8_t channel, uint8_t pc, uint8_t bank) {
    auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    Bank& b = pseudo_ch.banks[bank];
    const auto& timing = hbm2_config_.timing;

    b.state = hbm2::BankState::REFRESHING;
    b.state_until = current_cycle_ + timing.tRFCpb;
    b.last_refresh = current_cycle_;

    stats_.refreshes++;

    // Command bus busy for 1 cycle
    pseudo_ch.cmd_bus_state = CommandBusState::BUSY;
    pseudo_ch.cmd_bus_until = current_cycle_ + 1;

    // Trace the operation
    trace_bank_state_change(channel, pc, bank, hbm2::BankState::REFRESHING, "REFRESH");
    trace_command(channel, pc, bank, "REFRESH", timing.tRFCpb, 0);
    trace_bus_state_change(channel, pc, false, "BUSY", "REF cmd");
}

// ============================================================================
// State updates
// ============================================================================

void HBM2MemoryController::update_bank_states() {
    for (uint8_t ch = 0; ch < hbm2_config_.num_channels; ++ch) {
        for (uint8_t pc = 0; pc < hbm2_config_.pseudo_channels_per_channel; ++pc) {
            for (uint8_t b = 0; b < 16; ++b) {
                Bank& bank = channels_[ch].pseudo_channels[pc].banks[b];

                if (current_cycle_ >= bank.state_until) {
                    switch (bank.state) {
                        case hbm2::BankState::ACTIVATING:
                            bank.state = hbm2::BankState::ACTIVE;
                            trace_bank_state_change(ch, pc, b, hbm2::BankState::ACTIVE, "tRCD complete");
                            break;
                        case hbm2::BankState::PRECHARGING:
                            bank.state = hbm2::BankState::IDLE;
                            bank.open_row = 0;
                            bank.last_read_cmd = 0;
                            bank.last_write_cmd = 0;
                            trace_bank_state_change(ch, pc, b, hbm2::BankState::IDLE, "tRP complete");
                            break;
                        case hbm2::BankState::REFRESHING:
                            bank.state = hbm2::BankState::IDLE;
                            trace_bank_state_change(ch, pc, b, hbm2::BankState::IDLE, "tRFCpb complete");
                            break;
                        default:
                            break;
                    }
                }

                // Burst completion
                if (current_cycle_ >= bank.burst_end) {
                    if (bank.state == hbm2::BankState::READING || bank.state == hbm2::BankState::WRITING) {
                        bank.state = hbm2::BankState::ACTIVE;
                        trace_bank_state_change(ch, pc, b, hbm2::BankState::ACTIVE, "burst complete");
                    }
                }
            }
        }
    }
}

void HBM2MemoryController::update_bus_states() {
    for (uint8_t ch = 0; ch < hbm2_config_.num_channels; ++ch) {
        for (uint8_t pc = 0; pc < hbm2_config_.pseudo_channels_per_channel; ++pc) {
            PseudoChannel& pseudo_ch = channels_[ch].pseudo_channels[pc];

            // Command bus
            if (current_cycle_ >= pseudo_ch.cmd_bus_until &&
                pseudo_ch.cmd_bus_state != CommandBusState::IDLE) {
                pseudo_ch.cmd_bus_state = CommandBusState::IDLE;
                pseudo_ch.deferred_cmd_bus_idle_trace = true;
            }

            // Data bus
            if (current_cycle_ >= pseudo_ch.data_bus_until &&
                pseudo_ch.data_bus_state != DataBusState::IDLE) {
                pseudo_ch.data_bus_state = DataBusState::IDLE;
                pseudo_ch.deferred_data_bus_idle_trace = true;
            }
        }
    }
}

void HBM2MemoryController::finalize_bus_traces() {
    for (uint8_t ch = 0; ch < hbm2_config_.num_channels; ++ch) {
        for (uint8_t pc = 0; pc < hbm2_config_.pseudo_channels_per_channel; ++pc) {
            PseudoChannel& pseudo_ch = channels_[ch].pseudo_channels[pc];

            if (pseudo_ch.deferred_cmd_bus_idle_trace) {
                if (pseudo_ch.cmd_bus_state == CommandBusState::IDLE) {
                    trace_bus_state_change(ch, pc, false, "IDLE", "cmd complete");
                }
                pseudo_ch.deferred_cmd_bus_idle_trace = false;
            }

            if (pseudo_ch.deferred_data_bus_idle_trace) {
                if (pseudo_ch.data_bus_state == DataBusState::IDLE) {
                    trace_bus_state_change(ch, pc, true, "IDLE", "burst complete");
                }
                pseudo_ch.deferred_data_bus_idle_trace = false;
            }
        }
    }
}

void HBM2MemoryController::handle_refresh() {
    const auto& timing = hbm2_config_.timing;

    for (uint8_t ch = 0; ch < hbm2_config_.num_channels; ++ch) {
        for (uint8_t pc = 0; pc < hbm2_config_.pseudo_channels_per_channel; ++pc) {
            PseudoChannel& pseudo_ch = channels_[ch].pseudo_channels[pc];

            // Check if any bank needs refresh
            uint8_t bank = pseudo_ch.next_refresh_bank;
            Bank& b = pseudo_ch.banks[bank];

            if (b.state == hbm2::BankState::IDLE &&
                current_cycle_ >= b.last_refresh + timing.tREFI) {
                do_refresh(ch, pc, bank);
                pseudo_ch.next_refresh_bank = (bank + 1) % 16;
                return;  // One refresh per cycle
            }
        }
    }
}

// ============================================================================
// Request scheduling
// ============================================================================

void HBM2MemoryController::schedule_requests() {
    if (pending_queue_.empty()) {
        return;
    }

    // Try to issue requests from the front of the queue
    std::queue<MemoryRequest> retry_queue;

    while (!pending_queue_.empty()) {
        MemoryRequest req = std::move(pending_queue_.front());
        pending_queue_.pop();

        if (try_issue_request(req)) {
            active_requests_.push_back(std::move(req));
        } else {
            retry_queue.push(std::move(req));
        }
    }

    pending_queue_ = std::move(retry_queue);
}

bool HBM2MemoryController::try_issue_request(MemoryRequest& req) {
    uint8_t ch = req.channel;
    uint8_t pc = req.pseudo_channel;
    uint8_t bank = req.bank;
    uint32_t row = req.row;

    auto& pseudo_ch = channels_[ch].pseudo_channels[pc];
    Bank& b = pseudo_ch.banks[bank];

    // Check bank state
    switch (b.state) {
        case hbm2::BankState::IDLE:
            // Need to activate row first
            if (can_activate(ch, pc, bank, current_cycle_)) {
                do_activate(ch, pc, bank, row, req.id);
                req.triggered_activate = true;
                stats_.page_empty++;
                stats_.page_empty_count++;
                return false;  // Will issue data command on next cycle
            }
            return false;

        case hbm2::BankState::ACTIVATING:
            // Wait for activation to complete
            return false;

        case hbm2::BankState::ACTIVE:
        case hbm2::BankState::READING:
        case hbm2::BankState::WRITING:
            if (b.open_row == row) {
                // Page hit - can issue read/write
                if (req.type == RequestType::READ) {
                    if (can_read(ch, pc, bank, current_cycle_)) {
                        do_read(ch, pc, bank, req);
                        if (!req.triggered_activate) {
                            stats_.page_hits++;
                            stats_.page_hit_count++;
                        }
                        return true;
                    }
                } else {
                    if (can_write(ch, pc, bank, current_cycle_)) {
                        do_write(ch, pc, bank, req);
                        if (!req.triggered_activate) {
                            stats_.page_hits++;
                            stats_.page_hit_count++;
                        }
                        return true;
                    }
                }
            } else {
                // Page conflict - need to precharge first
                if (can_precharge(ch, pc, bank, current_cycle_)) {
                    do_precharge(ch, pc, bank);
                    req.triggered_conflict = true;
                    stats_.page_conflicts++;
                    stats_.page_conflict_count++;
                }
            }
            return false;

        case hbm2::BankState::PRECHARGING:
        case hbm2::BankState::REFRESHING:
            // Wait for operation to complete
            return false;

        default:
            return false;
    }
}

// ============================================================================
// Request completion
// ============================================================================

void HBM2MemoryController::complete_requests() {
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        if (current_cycle_ >= it->complete_cycle && it->complete_cycle > 0) {
            // Calculate latency
            uint64_t latency = it->complete_cycle - it->submit_cycle;
            stats_.total_latency += latency;

            // Track per-type latency
            if (it->type == RequestType::READ) {
                stats_.read_latency_total += latency;
                stats_.read_latency_min = std::min(stats_.read_latency_min, latency);
                stats_.read_latency_max = std::max(stats_.read_latency_max, latency);
            } else {
                stats_.write_latency_total += latency;
                stats_.write_latency_min = std::min(stats_.write_latency_min, latency);
                stats_.write_latency_max = std::max(stats_.write_latency_max, latency);
            }

            // Track scenario latency
            if (it->triggered_conflict) {
                stats_.page_conflict_latency_total += latency;
            } else if (it->triggered_activate) {
                stats_.page_empty_latency_total += latency;
            } else {
                stats_.page_hit_latency_total += latency;
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

void HBM2MemoryController::check_all_invariants() {
    for (uint8_t ch = 0; ch < hbm2_config_.num_channels; ++ch) {
        for (uint8_t pc = 0; pc < hbm2_config_.pseudo_channels_per_channel; ++pc) {
            for (uint8_t bank = 0; bank < 16; ++bank) {
                check_bank_invariants(ch, pc, bank);
            }
            check_timing_invariants(ch, pc);
            check_bus_invariants(ch, pc);
        }
    }
}

void HBM2MemoryController::check_bank_invariants(uint8_t channel, uint8_t pc, uint8_t bank) {
    const auto& b = channels_[channel].pseudo_channels[pc].banks[bank];
    const auto& timing = hbm2_config_.timing;

    // INV-BANK-1: Only one row can be active per bank at any time
    // (Implicit in our state machine - enforced by design)

    // INV-BANK-2: tRCD check - bank should not be ACTIVE before tRCD elapsed
    if (b.state == hbm2::BankState::ACTIVE && current_cycle_ < b.state_until) {
        report_violation("HBM2-INV-001",
            "Bank transitioned to ACTIVE before tRCD elapsed",
            channel, pc, bank);
    }

    // INV-BANK-3: tRAS check - PRECHARGE should not happen before tRAS
    if (b.state == hbm2::BankState::PRECHARGING) {
        if (b.last_activate > 0 &&
            current_cycle_ < b.last_activate + timing.tRAS) {
            report_violation("HBM2-INV-002",
                "PRECHARGE issued before tRAS elapsed",
                channel, pc, bank);
        }
    }

    // INV-BANK-4: tWR check - write recovery before PRECHARGE
    if (b.state == hbm2::BankState::PRECHARGING && b.last_write_cmd > 0) {
        uint64_t write_done = b.last_write_cmd + timing.tWL + burst_cycles();
        if (current_cycle_ < write_done + timing.tWR) {
            report_violation("HBM2-INV-003",
                "PRECHARGE issued before tWR elapsed after write",
                channel, pc, bank);
        }
    }

    // INV-BANK-5: tRTP check - read to precharge
    if (b.state == hbm2::BankState::PRECHARGING && b.last_read_cmd > 0) {
        if (current_cycle_ < b.last_read_cmd + timing.tRTP) {
            report_violation("HBM2-INV-004",
                "PRECHARGE issued before tRTP elapsed after read",
                channel, pc, bank);
        }
    }
}

void HBM2MemoryController::check_timing_invariants(uint8_t channel, uint8_t pc) {
    // Check tFAW violations
    const auto& pseudo_ch = channels_[channel].pseudo_channels[pc];
    const auto& timing = hbm2_config_.timing;

    int count = 0;
    for (int i = 0; i < 4; ++i) {
        if (pseudo_ch.activate_window[i] > 0 &&
            current_cycle_ - pseudo_ch.activate_window[i] < timing.tFAW) {
            count++;
        }
    }
    if (count > 4) {
        report_violation("HBM2-INV-100",
            "tFAW violation: " + std::to_string(count) + " activates in window",
            channel, pc, 0);
    }
}

void HBM2MemoryController::check_bus_invariants(uint8_t channel, uint8_t pc) {
    // Check that command and data buses are consistent
    const auto& pseudo_ch = channels_[channel].pseudo_channels[pc];

    if (pseudo_ch.cmd_bus_state == CommandBusState::BUSY &&
        pseudo_ch.cmd_bus_until < current_cycle_) {
        report_violation("HBM2-INV-200",
            "Command bus stuck in BUSY state past deadline",
            channel, pc, 0);
    }

    if (pseudo_ch.data_bus_state != DataBusState::IDLE &&
        pseudo_ch.data_bus_until < current_cycle_) {
        report_violation("HBM2-INV-201",
            "Data bus stuck in non-IDLE state past deadline",
            channel, pc, 0);
    }
}

void HBM2MemoryController::report_violation(const std::string& id, const std::string& msg,
                                             uint8_t channel, uint8_t pc, uint8_t bank) {
    violations_.push_back({
        current_cycle_,
        id,
        msg,
        channel,
        pc,
        bank
    });
}

// ============================================================================
// Tracing
// ============================================================================

void HBM2MemoryController::trace_bank_state_change(uint8_t channel, uint8_t pc, uint8_t bank,
                                                    hbm2::BankState new_state, const std::string& reason) {
    if (!tracing_enabled_) return;
    (void)channel; (void)pc; (void)bank; (void)new_state; (void)reason;
    // Tracing implementation would go here
}

void HBM2MemoryController::trace_bus_state_change(uint8_t channel, uint8_t pc, bool is_data_bus,
                                                   const std::string& state, const std::string& reason) {
    if (!tracing_enabled_) return;
    (void)channel; (void)pc; (void)is_data_bus; (void)state; (void)reason;
    // Tracing implementation would go here
}

void HBM2MemoryController::trace_command(uint8_t channel, uint8_t pc, uint8_t bank,
                                          const std::string& cmd, uint64_t duration, uint64_t request_id) {
    if (!tracing_enabled_) return;

    // Use global pseudo-channel ID for trace tid
    uint8_t pc_id = channel * 2 + pc;

    sw::trace::TraceEntry entry(
        current_cycle_,
        sw::trace::ComponentType::HBM2_BANK,
        pc_id,  // Use pseudo-channel as component_id for grouping
        cmd == "READ" || cmd == "BURST_READ" ? sw::trace::TransactionType::BURST_READ :
        cmd == "WRITE" || cmd == "BURST_WRITE" ? sw::trace::TransactionType::BURST_WRITE :
        cmd == "ACTIVATE" ? sw::trace::TransactionType::ACTIVATE :
        cmd == "PRECHARGE" ? sw::trace::TransactionType::PRECHARGE :
        cmd == "REFRESH" ? sw::trace::TransactionType::REFRESH :
        sw::trace::TransactionType::UNKNOWN,
        request_id
    );

    entry.complete(current_cycle_ + duration);
    entry.clock_freq_ghz = 1.0;  // 1 GHz for HBM2

    // Store bank info in description
    std::ostringstream ss;
    ss << cmd << " ch=" << (int)channel << " pc=" << (int)pc << " bank=" << (int)bank;
    entry.description = ss.str();

    trace_entries_.push_back(std::move(entry));
}

} // namespace sw::kpu::hbm2
