// ============================================================================
// src/harness/dma_harness.cpp
// Implementation of DMA engine test harness
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/harness/dma_harness.hpp>
#include <sw/kpu/models/behavioral/datamovement/dma_engine.hpp>

#include <iostream>
#include <iomanip>
#include <algorithm>

namespace sw::kpu::harness {

// ============================================================================
// Construction
// ============================================================================

DMAHarness::DMAHarness(const DMAHarnessConfig& config)
    : PatternHarnessBase<DMAHarnessConfig>(config)
    , l3_buffers_(config.l3_buffer_count, config.l3_buffer_size)
{
    // Create DMA engines based on configuration
    DMAEngineConfig engine_config;
    engine_config.fidelity = config.fidelity;
    engine_config.enable_tracing = config.enable_tracing;
    engine_config.enable_statistics = config.enable_stats;
    engine_config.bandwidth_gbps = static_cast<uint32_t>(config.bandwidth_gbps);
    engine_config.fixed_latency_per_burst = config.fixed_latency_cycles;
    engine_config.queue_depth_per_channel = config.queue_depth;

    dma_engines_.reserve(config.num_dma_engines);
    for (uint32_t i = 0; i < config.num_dma_engines; ++i) {
        engine_config.instance_id = i;

        // Create behavioral DMA engine (will later support factory pattern)
        auto engine = std::make_unique<sw::kpu::BehavioralDMAEngine>(engine_config);

        // Set up resource tracker if tracing enabled
        if (config.enable_tracing) {
            engine->set_resource_tracker(resource_tracker_.get());
            resource_tracker_->register_resource(
                sw::trace::ComponentType::DMA_ENGINE, i);
        }

        dma_engines_.push_back(std::move(engine));
    }
}

// ============================================================================
// Simulation Control
// ============================================================================

bool DMAHarness::run() {
    while (has_pending() && check_cycle_limit()) {
        tick();
    }
    return !has_pending();
}

void DMAHarness::tick() {
    // Track if any engine is busy
    bool any_busy = false;

    // Process pending requests that can now be issued
    process_pending_requests();

    // Tick all DMA engines
    for (size_t i = 0; i < dma_engines_.size(); ++i) {
        auto& engine = dma_engines_[i];
        engine->tick();

        if (engine->has_pending()) {
            any_busy = true;
        }

        // Update stats from engine
        update_stats_from_engine(i);
    }

    // Track busy/idle cycles
    if (any_busy) {
        dma_stats_.busy_cycles++;
    } else {
        dma_stats_.idle_cycles++;
    }

    // Track credit stalls (pending queue non-empty but no credits)
    if (!pending_queue_.empty() && l3_buffers_.available_credits() == 0) {
        dma_stats_.credit_stall_cycles++;

        // Record stall for tile at front of queue
        if (!pending_queue_.empty()) {
            const auto& front = pending_queue_.front();
            if (journey_tracker_.has_journey(front.tile_id)) {
                journey_tracker_.record_credit_stall(front.tile_id, 1);
            }
        }
    }

    was_busy_last_cycle_ = any_busy;
    PatternHarnessBase::tick();  // Increment current_cycle_
}

void DMAHarness::reset() {
    PatternHarnessBase::reset();

    // Reset DMA engines
    for (auto& engine : dma_engines_) {
        engine->reset();
    }

    // Reset L3 buffer pool
    l3_buffers_.reset();

    // Clear request tracking
    while (!pending_queue_.empty()) {
        pending_queue_.pop();
    }
    in_flight_requests_.clear();
    next_transfer_id_ = 1;

    // Reset statistics
    dma_stats_.reset();

    // Reset journey tracker
    journey_tracker_.reset();

    was_busy_last_cycle_ = false;
}

bool DMAHarness::has_pending() const {
    if (!pending_queue_.empty()) return true;
    if (!in_flight_requests_.empty()) return true;

    for (const auto& engine : dma_engines_) {
        if (engine->has_pending()) return true;
    }

    return false;
}

// ============================================================================
// Request Interface
// ============================================================================

bool DMAHarness::request_tile_load(
    const TileID& tile_id,
    uint64_t dram_address,
    uint32_t l3_buffer_id,
    std::function<void()> callback)
{
    DMARequest request;
    request.tile_id = tile_id;
    request.dram_address = dram_address;
    request.l3_buffer_id = l3_buffer_id;
    request.size = config_.tile_size;
    request.is_load = true;
    request.callback = std::move(callback);

    // Start journey tracking
    auto& journey = journey_tracker_.begin_journey(tile_id, current_cycle_);
    journey.dram_request_cycle = current_cycle_;

    pending_queue_.push(std::move(request));
    return true;
}

bool DMAHarness::request_tile_store(
    const TileID& tile_id,
    uint32_t l3_buffer_id,
    uint64_t dram_address,
    std::function<void()> callback)
{
    // Verify buffer is occupied with this tile
    if (!l3_buffers_.is_occupied(l3_buffer_id)) {
        return false;
    }

    DMARequest request;
    request.tile_id = tile_id;
    request.dram_address = dram_address;
    request.l3_buffer_id = l3_buffer_id;
    request.size = config_.tile_size;
    request.is_load = false;
    request.callback = std::move(callback);

    // Track store request
    if (journey_tracker_.has_journey(tile_id)) {
        journey_tracker_.record_dram_writeback_start(tile_id, current_cycle_);
    }

    pending_queue_.push(std::move(request));
    return true;
}

size_t DMAHarness::request_tile_loads(
    const std::vector<std::pair<TileID, uint64_t>>& requests)
{
    size_t accepted = 0;
    for (const auto& [tile_id, dram_addr] : requests) {
        if (request_tile_load(tile_id, dram_addr)) {
            accepted++;
        }
    }
    return accepted;
}

// ============================================================================
// Internal Helpers
// ============================================================================

void DMAHarness::process_pending_requests() {
    // Try to issue pending requests
    while (!pending_queue_.empty()) {
        auto& request = pending_queue_.front();

        // Check for available credit (L3 buffer) for loads
        if (request.is_load) {
            uint32_t buffer_id = request.l3_buffer_id;

            // Auto-allocate buffer if not specified
            if (buffer_id == UINT32_MAX) {
                buffer_id = l3_buffers_.allocate();
                if (buffer_id == UINT32_MAX) {
                    // No credits available, stop processing
                    break;
                }
                request.l3_buffer_id = buffer_id;
            } else if (!l3_buffers_.reserve(buffer_id, request.tile_id)) {
                // Specified buffer not available
                break;
            }
        }

        // Find an engine that can accept the transfer
        IDMAEngine* selected_engine = nullptr;
        for (auto& engine : dma_engines_) {
            if (engine->can_accept()) {
                selected_engine = engine.get();
                break;
            }
        }

        if (!selected_engine) {
            // No engine available, stop processing
            break;
        }

        // Build DMA transfer descriptor
        DMATransfer transfer;
        if (request.is_load) {
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = request.dram_address;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = request.l3_buffer_id;
        } else {
            transfer.src_type = DMAMemoryType::L3_TILE;
            transfer.src_id = request.l3_buffer_id;
            transfer.dst_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.dst_addr = request.dram_address;
        }
        transfer.size = request.size;
        transfer.user_tag = next_transfer_id_;

        // Submit transfer with completion callback
        auto tile_id_copy = request.tile_id;
        auto is_load_copy = request.is_load;
        auto size_copy = request.size;
        auto user_callback = request.callback;
        auto start_cycle = current_cycle_;

        auto completion_callback = [this, tile_id_copy, is_load_copy,
                                    size_copy, user_callback, start_cycle]() {
            // Record completion timing
            Cycle latency = current_cycle_ - start_cycle;

            if (is_load_copy) {
                // Load completed: tile arrived at L3
                journey_tracker_.record_dram_fetch_complete(tile_id_copy, current_cycle_);
                journey_tracker_.record_l3_arrival(tile_id_copy, current_cycle_);
            } else {
                // Store completed
                journey_tracker_.record_dram_writeback_complete(tile_id_copy, current_cycle_);
            }

            // Update statistics
            on_transfer_complete(tile_id_copy, is_load_copy, size_copy);

            // Call user callback if provided
            if (user_callback) {
                user_callback();
            }
        };

        auto result = selected_engine->submit(transfer, completion_callback);

        if (result.has_value()) {
            // Transfer accepted
            uint64_t transfer_id = result.value();

            // Record transfer start
            DMARequest in_flight = request;
            in_flight_requests_[transfer_id] = std::move(in_flight);

            // Record DRAM fetch start
            journey_tracker_.record_dram_fetch_start(request.tile_id, current_cycle_);

            next_transfer_id_++;
            pending_queue_.pop();
        } else {
            // Transfer rejected, try again later
            break;
        }
    }
}

void DMAHarness::update_stats_from_engine(size_t engine_id) {
    // This method can collect per-engine statistics if needed
    // For now, main statistics are updated via completion callbacks
    (void)engine_id;  // Suppress unused warning
}

void DMAHarness::on_transfer_complete(const TileID& tile_id, bool is_load, uint32_t size) {
    (void)tile_id;  // Used implicitly via journey tracker

    if (is_load) {
        dma_stats_.tiles_loaded++;
        dma_stats_.bytes_loaded += size;
    } else {
        dma_stats_.tiles_stored++;
        dma_stats_.bytes_stored += size;
    }

    dma_stats_.transfers_completed++;
    dma_stats_.bytes_transferred += size;

    // Record in central stats collector
    if (is_load) {
        stats_collector_.record_external_read(size);
    } else {
        stats_collector_.record_external_write(size);
    }
}

// ============================================================================
// Statistics
// ============================================================================

void DMAHarness::print_stats(std::ostream& os) const {
    os << "\n" << dma_stats_.to_string();
    os << "\n";
    PatternHarnessBase::print_stats(os);
}

// ============================================================================
// Validation
// ============================================================================

ValidationResult DMAHarness::validate() const {
    ValidationResult result = PatternHarnessBase::validate();

    // Check for failed transfers
    if (dma_stats_.transfers_failed > 0) {
        result.add_error("DMA transfers failed: " +
                         std::to_string(dma_stats_.transfers_failed));
    }

    // Check for stuck pending requests
    if (!pending_queue_.empty()) {
        result.add_warning("Pending DMA requests remain: " +
                           std::to_string(pending_queue_.size()));
    }

    // Check utilization
    if (dma_stats_.utilization() < 0.1 && dma_stats_.transfers_completed > 0) {
        result.add_warning("Low DMA utilization: " +
                           std::to_string(dma_stats_.utilization() * 100) + "%");
    }

    return result;
}

bool DMAHarness::verify_stats(uint64_t expected_loads, uint64_t expected_stores) const {
    bool pass = true;

    if (dma_stats_.tiles_loaded != expected_loads) {
        std::cerr << "FAIL: Expected " << expected_loads
                  << " tile loads, got " << dma_stats_.tiles_loaded << std::endl;
        pass = false;
    }

    if (dma_stats_.tiles_stored != expected_stores) {
        std::cerr << "FAIL: Expected " << expected_stores
                  << " tile stores, got " << dma_stats_.tiles_stored << std::endl;
        pass = false;
    }

    return pass;
}

} // namespace sw::kpu::harness
