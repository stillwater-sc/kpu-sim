// ============================================================================
// src/harness/streamer_harness.cpp
// Implementation of Streamer test harness
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/harness/streamer_harness.hpp>

#include <iostream>
#include <cstring>
#include <cmath>

namespace sw::kpu::harness {

// ============================================================================
// Construction
// ============================================================================

StreamerHarness::StreamerHarness(const StreamerHarnessConfig& config)
    : PatternHarnessBase<StreamerHarnessConfig>(config)
    , l2_banks_(config.l2_banks, 64 * 1024)  // 64KB per bank
    , l1_buffers_(config.l1_depth, config.systolic_size * config.systolic_size * 2)  // FP16
    , accumulator_(config.systolic_size * config.systolic_size)
    , drain_buffer_(config.systolic_size * config.systolic_size, 0.0f)
{
    // Register streamers for tracing
    if (config.enable_tracing) {
        for (uint32_t i = 0; i < config.num_streamers; ++i) {
            resource_tracker_->register_resource(
                sw::trace::ComponentType::STREAMER, i);
        }
    }
}

// ============================================================================
// Simulation Control
// ============================================================================

bool StreamerHarness::run() {
    while (has_pending() && check_cycle_limit()) {
        tick();
    }
    return !has_pending();
}

void StreamerHarness::tick() {
    bool any_busy = false;

    // Process pending requests
    process_pending_requests();

    if (!pending_queue_.empty()) {
        any_busy = true;
    }

    // Track stalls
    if (!pending_queue_.empty()) {
        // Could track specific stall reasons here
    }

    if (any_busy) {
        str_stats_.busy_cycles++;
    } else {
        str_stats_.idle_cycles++;
    }

    PatternHarnessBase::tick();
}

void StreamerHarness::reset() {
    PatternHarnessBase::reset();

    // Reset memory
    l2_banks_.reset();
    l1_buffers_.reset();

    // Reset compute
    accumulator_.clear();
    std::fill(drain_buffer_.begin(), drain_buffer_.end(), 0.0f);

    // Clear pending requests
    while (!pending_queue_.empty()) {
        pending_queue_.pop();
    }

    // Reset statistics
    str_stats_.reset();

    // Reset journey tracker
    journey_tracker_.reset();
}

bool StreamerHarness::has_pending() const {
    return !pending_queue_.empty();
}

// ============================================================================
// L2 Preloading
// ============================================================================

void StreamerHarness::preload_l2(uint32_t bank_id, const TileID& tile_id,
                                  const void* data, size_t size) {
    if (bank_id >= l2_banks_.num_banks()) return;

    l2_banks_.reserve(bank_id, tile_id);
    l2_banks_.write(bank_id, 0, data, size);
}

void StreamerHarness::preload_l2_pattern(uint32_t bank_id, const TileID& tile_id,
                                          uint32_t height, uint32_t width,
                                          uint32_t element_size) {
    if (bank_id >= l2_banks_.num_banks()) return;

    size_t total_size = height * width * element_size;
    std::vector<uint8_t> buffer(total_size);

    // Generate pattern data
    if (element_size == 2) {
        uint16_t* ptr = reinterpret_cast<uint16_t*>(buffer.data());
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                ptr[i * width + j] = static_cast<uint16_t>(i * width + j);
            }
        }
    } else if (element_size == 4) {
        float* ptr = reinterpret_cast<float*>(buffer.data());
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                ptr[i * width + j] = static_cast<float>(i * width + j);
            }
        }
    }

    preload_l2(bank_id, tile_id, buffer.data(), total_size);
}

// ============================================================================
// Stream Operations
// ============================================================================

bool StreamerHarness::stream_rows(const TileID& tile_id, uint32_t l2_bank_id,
                                   std::function<void()> callback) {
    if (!l2_banks_.is_occupied(l2_bank_id)) {
        return false;
    }

    StreamRequest request;
    request.tile_id = tile_id;
    request.type = StreamRequest::Type::ROW_STREAM;
    request.l2_bank_id = l2_bank_id;
    request.callback = std::move(callback);

    // Track journey
    if (!journey_tracker_.has_journey(tile_id)) {
        journey_tracker_.begin_journey(tile_id, current_cycle_);
    }
    journey_tracker_.record_l2_departure(tile_id, current_cycle_);

    pending_queue_.push(std::move(request));
    return true;
}

bool StreamerHarness::stream_cols(const TileID& tile_id, uint32_t l2_bank_id,
                                   std::function<void()> callback) {
    if (!l2_banks_.is_occupied(l2_bank_id)) {
        return false;
    }

    StreamRequest request;
    request.tile_id = tile_id;
    request.type = StreamRequest::Type::COL_STREAM;
    request.l2_bank_id = l2_bank_id;
    request.callback = std::move(callback);

    // Track journey
    if (!journey_tracker_.has_journey(tile_id)) {
        journey_tracker_.begin_journey(tile_id, current_cycle_);
    }
    journey_tracker_.record_l2_departure(tile_id, current_cycle_);

    pending_queue_.push(std::move(request));
    return true;
}

bool StreamerHarness::broadcast(const TileID& tile_id, uint32_t l2_bank_id,
                                 std::function<void()> callback) {
    if (!l2_banks_.is_occupied(l2_bank_id)) {
        return false;
    }

    StreamRequest request;
    request.tile_id = tile_id;
    request.type = StreamRequest::Type::BROADCAST;
    request.l2_bank_id = l2_bank_id;
    request.callback = std::move(callback);

    pending_queue_.push(std::move(request));
    return true;
}

bool StreamerHarness::drain(const TileID& tile_id, uint32_t l2_bank_id,
                             std::function<void()> callback) {
    StreamRequest request;
    request.tile_id = tile_id;
    request.type = StreamRequest::Type::DRAIN;
    request.l2_bank_id = l2_bank_id;
    request.callback = std::move(callback);

    // Track journey
    if (journey_tracker_.has_journey(tile_id)) {
        journey_tracker_.record_drain_start(tile_id, current_cycle_);
    }

    pending_queue_.push(std::move(request));
    return true;
}

// ============================================================================
// State Queries
// ============================================================================

std::vector<float> StreamerHarness::read_drain_buffer() const {
    return drain_buffer_;
}

// ============================================================================
// Internal Helpers
// ============================================================================

void StreamerHarness::process_pending_requests() {
    // Process all pending requests (behavioral mode - instant completion)
    while (!pending_queue_.empty()) {
        auto request = pending_queue_.front();
        pending_queue_.pop();

        execute_stream(request);

        // Call completion callback
        if (request.callback) {
            request.callback();
        }
    }
}

void StreamerHarness::execute_stream(const StreamRequest& request) {
    size_t transfer_size = request.height * request.width * request.element_size;

    switch (request.type) {
        case StreamRequest::Type::ROW_STREAM: {
            // Read from L2 bank
            std::vector<uint8_t> data(transfer_size);
            l2_banks_.read(request.l2_bank_id, request.l2_offset, data.data(), transfer_size);

            // Write to L1 buffer (simplified - just use first buffer)
            l1_buffers_.write(0, data.data(), transfer_size);

            // Record L1 arrival
            journey_tracker_.record_l1_arrival(request.tile_id, current_cycle_, 0);

            // Trigger compute (simplified)
            journey_tracker_.record_compute_start(request.tile_id, current_cycle_);
            str_stats_.compute_operations += request.height * request.width;
            str_stats_.compute_cycles++;
            journey_tracker_.record_compute_end(request.tile_id, current_cycle_);

            str_stats_.rows_streamed++;
            str_stats_.bytes_l2_to_l1 += transfer_size;

            // Record in central stats
            stats_collector_.record_l2_read(transfer_size);
            stats_collector_.record_l1_write(transfer_size);
            break;
        }

        case StreamRequest::Type::COL_STREAM: {
            // Similar to row stream but for columns
            std::vector<uint8_t> data(transfer_size);
            l2_banks_.read(request.l2_bank_id, request.l2_offset, data.data(), transfer_size);
            l1_buffers_.write(1, data.data(), transfer_size);

            // Record L1 arrival
            journey_tracker_.record_l1_arrival(request.tile_id, current_cycle_, 1);

            str_stats_.cols_streamed++;
            str_stats_.bytes_l2_to_l1 += transfer_size;

            stats_collector_.record_l2_read(transfer_size);
            stats_collector_.record_l1_write(transfer_size);
            break;
        }

        case StreamRequest::Type::BROADCAST: {
            // Read scalar/bias from L2
            std::vector<uint8_t> data(transfer_size);
            l2_banks_.read(request.l2_bank_id, request.l2_offset, data.data(), transfer_size);
            l1_buffers_.write(2, data.data(), transfer_size);

            str_stats_.broadcasts++;
            str_stats_.bytes_l2_to_l1 += transfer_size;

            stats_collector_.record_l2_read(transfer_size);
            break;
        }

        case StreamRequest::Type::DRAIN: {
            // Copy accumulator to drain buffer
            size_t drain_size = accumulator_.size() * sizeof(float);
            std::memcpy(drain_buffer_.data(), accumulator_.data().data(), drain_size);

            // Write to L2
            l2_banks_.reserve(request.l2_bank_id, request.tile_id);
            l2_banks_.write(request.l2_bank_id, 0, drain_buffer_.data(), drain_size);

            // Record journey
            journey_tracker_.record_drain_complete(request.tile_id, current_cycle_);

            str_stats_.drains++;
            str_stats_.bytes_l1_to_l2 += drain_size;

            stats_collector_.record_l1_read(drain_size);
            stats_collector_.record_l2_write(drain_size);
            break;
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

void StreamerHarness::print_stats(std::ostream& os) const {
    os << "\n" << str_stats_.to_string();
    os << "\n";
    PatternHarnessBase::print_stats(os);
}

// ============================================================================
// Validation
// ============================================================================

ValidationResult StreamerHarness::validate() const {
    ValidationResult result = PatternHarnessBase::validate();

    // Check for stuck pending requests
    if (!pending_queue_.empty()) {
        result.add_warning("Pending Streamer requests remain: " +
                           std::to_string(pending_queue_.size()));
    }

    return result;
}

} // namespace sw::kpu::harness
