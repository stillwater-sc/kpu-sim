// ============================================================================
// src/components/memory/memory_controller.cpp
// Memory Controller with cycle-accurate DRAM timing simulation
// ============================================================================

#include <sw/kpu/components/memory_controller.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sw::kpu {

MemoryController::MemoryController(const Config& config)
    : config_(config)
    , banks_(config.num_banks)
    , next_refresh_cycle_(config.timing.tREFI)
{
}

std::optional<uint64_t> MemoryController::submit_read(
    uint64_t address, uint32_t size, std::function<void()> callback)
{
    if (!can_accept()) {
        trace_event(dataflow::TileEventType::MC_QUEUE_FULL, 0);
        return std::nullopt;
    }

    MemoryRequest req;
    req.id = next_request_id_++;
    req.type = MemoryRequestType::READ;
    req.address = address;
    req.size = size;
    req.issue_cycle = current_cycle_;
    req.completion_cycle = 0;
    req.callback = std::move(callback);
    req.data.resize(size);

    // Decode address
    decode_address(address, req.bank_id, req.row, req.col);

    pending_queue_.push(std::move(req));
    stats_.reads++;

    return req.id;
}

std::optional<uint64_t> MemoryController::submit_write(
    uint64_t address, const void* data, uint32_t size, std::function<void()> callback)
{
    if (!can_accept()) {
        trace_event(dataflow::TileEventType::MC_QUEUE_FULL, 0);
        return std::nullopt;
    }

    MemoryRequest req;
    req.id = next_request_id_++;
    req.type = MemoryRequestType::WRITE;
    req.address = address;
    req.size = size;
    req.issue_cycle = current_cycle_;
    req.completion_cycle = 0;
    req.callback = std::move(callback);

    // Copy data
    req.data.resize(size);
    if (data) {
        std::memcpy(req.data.data(), data, size);
    }

    // Decode address
    decode_address(address, req.bank_id, req.row, req.col);

    pending_queue_.push(std::move(req));
    stats_.writes++;

    return req.id;
}

void MemoryController::tick() {
    // Update bank state transitions
    update_banks();

    // Handle refresh
    handle_refresh();

    // Schedule new requests
    schedule_request();

    // Complete finished requests
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        if (current_cycle_ >= it->completion_cycle) {
            // Request completed
            uint64_t latency = current_cycle_ - it->issue_cycle;
            stats_.total_latency += latency;
            stats_.min_latency = std::min(stats_.min_latency, latency);
            stats_.max_latency = std::max(stats_.max_latency, latency);

            // Trace completion
            if (it->type == MemoryRequestType::READ) {
                trace_event(dataflow::TileEventType::MC_READ_COMPLETE,
                           it->bank_id, it->row, it->address, it->size);

                // Actually read from backing memory
                if (memory_) {
                    memory_->read(it->address, it->data.data(), it->size);
                }
            } else {
                trace_event(dataflow::TileEventType::MC_WRITE_COMPLETE,
                           it->bank_id, it->row, it->address, it->size);

                // Actually write to backing memory
                if (memory_) {
                    memory_->write(it->address, it->data.data(), it->size);
                }
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

    current_cycle_++;
}

void MemoryController::drain() {
    while (has_pending()) {
        tick();
    }
}

void MemoryController::reset() {
    banks_.assign(config_.num_banks, Bank{});
    while (!pending_queue_.empty()) pending_queue_.pop();
    active_requests_.clear();
    current_cycle_ = 0;
    next_request_id_ = 0;
    next_refresh_cycle_ = config_.timing.tREFI;
    stats_ = Stats{};
}

void MemoryController::update_banks() {
    for (size_t i = 0; i < banks_.size(); i++) {
        Bank& bank = banks_[i];

        if (current_cycle_ >= bank.state_until) {
            switch (bank.state) {
                case BankState::ACTIVATING:
                    bank.state = BankState::ACTIVE;
                    break;
                case BankState::PRECHARGING:
                    bank.state = BankState::IDLE;
                    bank.open_row = 0;
                    break;
                case BankState::REFRESHING:
                    bank.state = BankState::IDLE;
                    break;
                default:
                    break;
            }
        }
    }
}

void MemoryController::handle_refresh() {
    if (current_cycle_ >= next_refresh_cycle_) {
        // Schedule refresh - simple model: refresh all banks
        bool can_refresh = true;
        for (const auto& bank : banks_) {
            if (bank.state != BankState::IDLE && bank.state != BankState::ACTIVE) {
                can_refresh = false;
                break;
            }
        }

        if (can_refresh) {
            trace_event(dataflow::TileEventType::MC_REFRESH_START, 0);

            for (size_t i = 0; i < banks_.size(); i++) {
                if (banks_[i].state == BankState::ACTIVE) {
                    // Precharge first
                    banks_[i].state = BankState::PRECHARGING;
                    banks_[i].state_until = current_cycle_ + config_.timing.tRP;
                }
            }

            // Set refresh complete time
            for (auto& bank : banks_) {
                bank.state = BankState::REFRESHING;
                bank.state_until = current_cycle_ + config_.timing.tRFC;
            }

            stats_.refreshes++;
            next_refresh_cycle_ = current_cycle_ + config_.timing.tREFI;
        }
    }
}

void MemoryController::schedule_request() {
    if (pending_queue_.empty()) return;

    // FR-FCFS scheduling: find best candidate
    // For now, simple FCFS - just take from front
    MemoryRequest& req = pending_queue_.front();
    Bank& bank = banks_[req.bank_id];

    // Check if bank is available
    if (bank.state == BankState::REFRESHING ||
        bank.state == BankState::ACTIVATING ||
        bank.state == BankState::PRECHARGING) {
        stats_.stall_cycles++;
        return;  // Wait for bank
    }

    // Determine access type
    if (bank.state == BankState::IDLE) {
        // Empty access - need to activate row
        stats_.page_empty++;

        trace_event(dataflow::TileEventType::MC_BANK_ACTIVATE, req.bank_id, req.row);

        bank.state = BankState::ACTIVATING;
        bank.open_row = req.row;
        bank.state_until = current_cycle_ + config_.timing.tRCD;
        bank.last_access = current_cycle_;

        // Request will be issued after activation
        req.completion_cycle = calculate_completion(req);

    } else if (bank.state == BankState::ACTIVE) {
        if (bank.open_row == req.row) {
            // Page hit!
            stats_.page_hits++;
            trace_event(dataflow::TileEventType::MC_PAGE_HIT, req.bank_id, req.row);

            // Issue immediately
            if (req.type == MemoryRequestType::READ) {
                trace_event(dataflow::TileEventType::MC_READ_ISSUE,
                           req.bank_id, req.row, req.address, req.size);
            } else {
                trace_event(dataflow::TileEventType::MC_WRITE_ISSUE,
                           req.bank_id, req.row, req.address, req.size);
            }

            req.completion_cycle = calculate_completion(req);
            bank.last_access = current_cycle_;

        } else {
            // Page conflict - need to precharge and activate new row
            stats_.page_conflicts++;
            trace_event(dataflow::TileEventType::MC_PAGE_CONFLICT,
                       req.bank_id, bank.open_row, req.address, 0);
            trace_event(dataflow::TileEventType::MC_BANK_PRECHARGE, req.bank_id);

            bank.state = BankState::PRECHARGING;
            bank.state_until = current_cycle_ + config_.timing.tRP;

            // Will need to activate after precharge - don't pop yet
            return;
        }
    }

    // Move request to active (must push before pop to avoid dangling reference)
    active_requests_.push_back(std::move(pending_queue_.front()));
    pending_queue_.pop();
}

uint64_t MemoryController::calculate_completion(const MemoryRequest& req) {
    Bank& bank = banks_[req.bank_id];
    uint64_t completion = current_cycle_;

    // Add activation delay if needed
    if (bank.state == BankState::ACTIVATING) {
        completion = bank.state_until;
    }

    // Add CAS latency
    if (req.type == MemoryRequestType::READ) {
        completion += config_.timing.tCL;
    } else {
        completion += config_.timing.tWL;
    }

    // Add burst transfer time
    uint32_t bursts = (req.size + config_.timing.bytes_per_burst() - 1) /
                      config_.timing.bytes_per_burst();
    completion += bursts * config_.timing.tBurst;

    return completion;
}

void MemoryController::trace_event(dataflow::TileEventType type, uint8_t bank_id,
                                    uint32_t row, uint64_t address, uint32_t bytes) {
    if (!tracer_) return;

    dataflow::TileFlowEvent e;
    e.cycle = current_cycle_;
    e.type = type;
    e.bank_id = bank_id;
    e.row = row;
    e.address = address;
    e.bytes = bytes;

    tracer_->record(e);
}

} // namespace sw::kpu
