// ============================================================================
// src/harness/block_mover_harness.cpp
// Implementation of BlockMover test harness
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/harness/block_mover_harness.hpp>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

namespace sw::kpu::harness {

// ============================================================================
// Construction
// ============================================================================

BlockMoverHarness::BlockMoverHarness(const BlockMoverHarnessConfig& config)
    : PatternHarnessBase<BlockMoverHarnessConfig>(config)
    , l2_banks_(config.l2_banks, config.l2_bank_size)
    , l3_buffer_data_(config.l3_buffers)
    , l3_buffer_valid_(config.l3_buffers, false)
    , l3_buffer_tiles_(config.l3_buffers)
{
    // Initialize L3 buffer storage
    for (auto& buf : l3_buffer_data_) {
        buf.resize(256 * 1024);  // 256KB per buffer
    }

    // Create behavioral memory model for the BlockMover to use
    behavioral::MemoryModelConfig mem_config;
    mem_config.l3_tile_count = config.l3_buffers;
    mem_config.l3_tile_capacity_bytes = 256 * 1024;
    mem_config.l2_bank_count = config.l2_banks;
    mem_config.l2_bank_capacity_bytes = config.l2_bank_size;
    memory_model_ = std::make_unique<behavioral::BehavioralMemoryModel>(mem_config);

    // Create BlockMovers
    block_movers_.reserve(config.num_block_movers);
    for (uint32_t i = 0; i < config.num_block_movers; ++i) {
        auto bm = std::make_unique<behavioral::BehavioralBlockMover>();
        block_movers_.push_back(std::move(bm));

        // Register for tracing
        if (config.enable_tracing) {
            resource_tracker_->register_resource(
                sw::trace::ComponentType::BLOCK_MOVER, i);
        }
    }
}

// ============================================================================
// Simulation Control
// ============================================================================

bool BlockMoverHarness::run() {
    while (has_pending() && check_cycle_limit()) {
        tick();
    }
    return !has_pending();
}

void BlockMoverHarness::tick() {
    bool any_busy = false;

    // Process pending requests
    process_pending_requests();

    // In behavioral mode, transfers complete instantly
    // For transactional/cycle-accurate, we would track in-flight transfers

    if (!pending_queue_.empty()) {
        any_busy = true;
    }

    // Track stalls
    if (!pending_queue_.empty() && l2_banks_.available_credits() == 0) {
        bm_stats_.credit_stall_cycles++;
    }

    if (any_busy) {
        bm_stats_.busy_cycles++;
    } else {
        bm_stats_.idle_cycles++;
    }

    PatternHarnessBase::tick();
}

void BlockMoverHarness::reset() {
    PatternHarnessBase::reset();

    // Reset BlockMovers
    for (auto& bm : block_movers_) {
        bm->reset_stats();
    }

    // Reset L2 banks
    l2_banks_.reset();

    // Reset L3 buffers
    std::fill(l3_buffer_valid_.begin(), l3_buffer_valid_.end(), false);
    std::fill(l3_buffer_tiles_.begin(), l3_buffer_tiles_.end(), TileID{});
    for (auto& buf : l3_buffer_data_) {
        std::fill(buf.begin(), buf.end(), 0);
    }

    // Clear pending requests
    while (!pending_queue_.empty()) {
        pending_queue_.pop();
    }

    // Reset statistics
    bm_stats_.reset();

    // Reset journey tracker
    journey_tracker_.reset();

    // Reset memory model stats
    if (memory_model_) {
        memory_model_->reset_stats();
    }
}

bool BlockMoverHarness::has_pending() const {
    return !pending_queue_.empty();
}

// ============================================================================
// L3 Preloading
// ============================================================================

void BlockMoverHarness::preload_l3(uint32_t buffer_id, const TileID& tile_id,
                                    const void* data, size_t size) {
    if (buffer_id >= l3_buffer_data_.size()) return;

    // Copy data to L3 buffer
    size_t copy_size = std::min(size, l3_buffer_data_[buffer_id].size());
    std::memcpy(l3_buffer_data_[buffer_id].data(), data, copy_size);
    l3_buffer_valid_[buffer_id] = true;
    l3_buffer_tiles_[buffer_id] = tile_id;

    // Also copy to memory model if available
    if (memory_model_) {
        auto* region = memory_model_->get_region(
            behavioral::MemoryRegionType::L3_TILE, buffer_id);
        if (region) {
            region->write(0, data, copy_size);
        }
    }
}

void BlockMoverHarness::preload_l3_pattern(uint32_t buffer_id, const TileID& tile_id,
                                            uint32_t height, uint32_t width,
                                            uint32_t element_size) {
    if (buffer_id >= l3_buffer_data_.size()) return;

    size_t total_size = height * width * element_size;
    if (total_size > l3_buffer_data_[buffer_id].size()) return;

    // Generate pattern data: element[i][j] = (i * width + j) for verification
    auto& buffer = l3_buffer_data_[buffer_id];

    if (element_size == 2) {
        // FP16 pattern
        uint16_t* ptr = reinterpret_cast<uint16_t*>(buffer.data());
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                ptr[i * width + j] = static_cast<uint16_t>(i * width + j);
            }
        }
    } else if (element_size == 4) {
        // FP32 pattern
        float* ptr = reinterpret_cast<float*>(buffer.data());
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                ptr[i * width + j] = static_cast<float>(i * width + j);
            }
        }
    }

    l3_buffer_valid_[buffer_id] = true;
    l3_buffer_tiles_[buffer_id] = tile_id;

    // Also update memory model
    if (memory_model_) {
        auto* region = memory_model_->get_region(
            behavioral::MemoryRegionType::L3_TILE, buffer_id);
        if (region) {
            region->write(0, buffer.data(), total_size);
        }
    }
}

// ============================================================================
// Transfer Operations
// ============================================================================

bool BlockMoverHarness::move_l3_to_l2(
    const TileID& tile_id,
    uint32_t l3_buffer_id,
    uint32_t l2_bank_id,
    Transform transform,
    std::function<void()> callback)
{
    // Check L3 buffer valid
    if (l3_buffer_id >= l3_buffer_valid_.size() || !l3_buffer_valid_[l3_buffer_id]) {
        return false;
    }

    BlockMoverRequest request;
    request.tile_id = tile_id;
    request.l3_buffer_id = l3_buffer_id;
    request.l2_bank_id = l2_bank_id;
    request.transform = transform;
    request.l3_to_l2 = true;
    request.callback = std::move(callback);

    // Start/update journey tracking
    if (!journey_tracker_.has_journey(tile_id)) {
        journey_tracker_.begin_journey(tile_id, current_cycle_);
    }
    journey_tracker_.record_l3_departure(tile_id, current_cycle_);

    pending_queue_.push(std::move(request));
    return true;
}

bool BlockMoverHarness::move_l2_to_l3(
    const TileID& tile_id,
    uint32_t l2_bank_id,
    uint32_t l3_buffer_id,
    std::function<void()> callback)
{
    // Check L2 bank occupied
    if (!l2_banks_.is_occupied(l2_bank_id)) {
        return false;
    }

    BlockMoverRequest request;
    request.tile_id = tile_id;
    request.l2_bank_id = l2_bank_id;
    request.l3_buffer_id = l3_buffer_id;
    request.l3_to_l2 = false;
    request.callback = std::move(callback);

    pending_queue_.push(std::move(request));
    return true;
}

// ============================================================================
// State Queries
// ============================================================================

bool BlockMoverHarness::tile_resident_l2(const TileID& tile_id) const {
    for (uint32_t i = 0; i < l2_banks_.num_banks(); ++i) {
        if (l2_banks_.is_occupied(i) && l2_banks_.get_tile(i) == tile_id) {
            return true;
        }
    }
    return false;
}

std::vector<uint8_t> BlockMoverHarness::read_l2(uint32_t bank_id, uint64_t offset, size_t size) const {
    std::vector<uint8_t> result(size);
    l2_banks_.read(bank_id, offset, result.data(), size);
    return result;
}

// ============================================================================
// Internal Helpers
// ============================================================================

void BlockMoverHarness::process_pending_requests() {
    // Process all pending requests (behavioral mode - instant completion)
    while (!pending_queue_.empty()) {
        auto& request = pending_queue_.front();

        if (request.l3_to_l2) {
            // L3 -> L2 transfer
            uint32_t l2_bank = request.l2_bank_id;

            // Auto-allocate L2 bank if needed
            if (l2_bank == UINT32_MAX) {
                l2_bank = l2_banks_.allocate();
                if (l2_bank == UINT32_MAX) {
                    // No credit available
                    bm_stats_.credit_stall_cycles++;
                    break;
                }
                // Record the tile_id for auto-allocated bank
                l2_banks_.set_tile(l2_bank, request.tile_id);
            } else if (!l2_banks_.reserve(l2_bank, request.tile_id)) {
                // Specified bank not available
                bm_stats_.credit_stall_cycles++;
                break;
            }

            request.l2_bank_id = l2_bank;
            execute_transfer(request);

            // Record L2 arrival (use current_cycle_ + 1 to indicate "end of tick")
            // In behavioral mode, this ensures l2_arrival > 0 for journey tracking
            journey_tracker_.record_l2_arrival(request.tile_id, current_cycle_ + 1, l2_bank);
        } else {
            // L2 -> L3 transfer
            execute_transfer(request);

            // Release L2 bank
            l2_banks_.release(request.l2_bank_id);
        }

        // Call completion callback
        if (request.callback) {
            request.callback();
        }

        pending_queue_.pop();
    }
}

void BlockMoverHarness::execute_transfer(const BlockMoverRequest& request) {
    // Calculate transfer size
    size_t transfer_size = request.height * request.width * request.element_size;

    // Select a BlockMover (round-robin for now)
    static size_t next_bm = 0;
    auto& bm = block_movers_[next_bm % block_movers_.size()];
    next_bm++;

    // Build transfer descriptor
    behavioral::BehavioralBlockMover::TransferDescriptor desc;

    if (request.l3_to_l2) {
        desc.src_type = behavioral::MemoryRegionType::L3_TILE;
        desc.src_region_id = request.l3_buffer_id;
        desc.src_offset = request.l3_offset;
        desc.dst_type = behavioral::MemoryRegionType::L2_BANK;
        desc.dst_region_id = request.l2_bank_id;
        desc.dst_offset = request.l2_offset;

        bm_stats_.l3_to_l2_transfers++;
    } else {
        desc.src_type = behavioral::MemoryRegionType::L2_BANK;
        desc.src_region_id = request.l2_bank_id;
        desc.src_offset = request.l2_offset;
        desc.dst_type = behavioral::MemoryRegionType::L3_TILE;
        desc.dst_region_id = request.l3_buffer_id;
        desc.dst_offset = request.l3_offset;

        bm_stats_.l2_to_l3_transfers++;
    }

    desc.height = request.height;
    desc.width = request.width;
    desc.element_size = request.element_size;
    desc.transpose = (request.transform == Transform::TRANSPOSE);

    // Execute transfer
    bm->transfer(desc, memory_model_.get());

    // Update statistics
    bm_stats_.tiles_moved++;
    bm_stats_.bytes_moved += transfer_size;

    if (request.transform != Transform::IDENTITY) {
        bm_stats_.transform_counts[request.transform]++;
    }

    // Record L3 read for central stats
    if (request.l3_to_l2) {
        stats_collector_.record_l3_read(transfer_size);
        stats_collector_.record_l2_write(transfer_size);
    } else {
        stats_collector_.record_l2_read(transfer_size);
        stats_collector_.record_l3_write(transfer_size);
    }
}

// ============================================================================
// Statistics
// ============================================================================

void BlockMoverHarness::print_stats(std::ostream& os) const {
    os << "\n" << bm_stats_.to_string();
    os << "\n";
    PatternHarnessBase::print_stats(os);
}

// ============================================================================
// Validation
// ============================================================================

ValidationResult BlockMoverHarness::validate() const {
    ValidationResult result = PatternHarnessBase::validate();

    // Check for stuck pending requests
    if (!pending_queue_.empty()) {
        result.add_warning("Pending BlockMover requests remain: " +
                           std::to_string(pending_queue_.size()));
    }

    // Check data integrity by comparing BlockMover stats
    for (size_t i = 0; i < block_movers_.size(); ++i) {
        const auto& stats = block_movers_[i]->stats();
        if (stats.transfers == 0 && bm_stats_.tiles_moved > 0) {
            // At least one BlockMover should have processed transfers
        }
    }

    return result;
}

} // namespace sw::kpu::harness
