// ============================================================================
// src/harness/pipeline_harness.cpp
// Implementation of full data movement pipeline harness
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/harness/pipeline_harness.hpp>

#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace sw::kpu::harness {

// ============================================================================
// Construction
// ============================================================================

DataMovementPipelineHarness::DataMovementPipelineHarness(const PipelineHarnessConfig& config)
    : PatternHarnessBase<PipelineHarnessConfig>(config)
{
    // Create component harnesses with their respective configs
    DMAHarnessConfig dma_config = config.dma_config;
    dma_config.fidelity = config.fidelity;
    dma_config.enable_tracing = config.enable_tracing;
    dma_config.enable_stats = config.enable_stats;
    dma_config.clock_frequency_ghz = config.clock_frequency_ghz;
    dma_harness_ = std::make_unique<DMAHarness>(dma_config);

    BlockMoverHarnessConfig bm_config = config.block_mover_config;
    bm_config.fidelity = config.fidelity;
    bm_config.enable_tracing = config.enable_tracing;
    bm_config.enable_stats = config.enable_stats;
    bm_config.clock_frequency_ghz = config.clock_frequency_ghz;
    block_mover_harness_ = std::make_unique<BlockMoverHarness>(bm_config);

    StreamerHarnessConfig str_config = config.streamer_config;
    str_config.fidelity = config.fidelity;
    str_config.enable_tracing = config.enable_tracing;
    str_config.enable_stats = config.enable_stats;
    str_config.clock_frequency_ghz = config.clock_frequency_ghz;
    streamer_harness_ = std::make_unique<StreamerHarness>(str_config);
}

// ============================================================================
// Simulation Control
// ============================================================================

bool DataMovementPipelineHarness::run() {
    // Execute all schedule operations
    while (current_op_index_ < schedule_.size() && check_cycle_limit()) {
        if (!execute_next_operation()) {
            // Operation blocked, advance time
            tick();
        }
    }

    // Drain all pending operations
    while (has_pending() && check_cycle_limit()) {
        tick();
    }

    collect_stats();
    return !has_pending() && current_op_index_ >= schedule_.size();
}

void DataMovementPipelineHarness::tick() {
    // Tick all component harnesses
    dma_harness_->tick();
    block_mover_harness_->tick();
    streamer_harness_->tick();

    PatternHarnessBase::tick();
}

void DataMovementPipelineHarness::step(Cycle cycles) {
    for (Cycle i = 0; i < cycles; ++i) {
        tick();
    }
}

void DataMovementPipelineHarness::reset() {
    PatternHarnessBase::reset();

    // Reset component harnesses
    dma_harness_->reset();
    block_mover_harness_->reset();
    streamer_harness_->reset();

    // Clear DRAM
    dram_data_.clear();
    dram_base_addresses_.clear();

    // Clear schedule
    schedule_.clear();
    current_op_index_ = 0;

    // Clear tracking
    completed_tiles_.clear();
    tile_l3_buffers_.clear();
    tile_l2_banks_.clear();

    // Reset stats
    pipeline_stats_.reset();
}

bool DataMovementPipelineHarness::has_pending() const {
    return dma_harness_->has_pending() ||
           block_mover_harness_->has_pending() ||
           streamer_harness_->has_pending();
}

// ============================================================================
// Memory Initialization
// ============================================================================

void DataMovementPipelineHarness::init_dram(uint32_t matrix_id, const void* data,
                                             uint32_t rows, uint32_t cols,
                                             uint32_t element_size) {
    size_t total_size = rows * cols * element_size;
    dram_data_[matrix_id].resize(total_size);
    std::memcpy(dram_data_[matrix_id].data(), data, total_size);

    // Assign base address (simple linear mapping)
    dram_base_addresses_[matrix_id] = matrix_id * (1ULL << 30);  // 1GB per matrix
}

void DataMovementPipelineHarness::init_dram_pattern(uint32_t matrix_id,
                                                     uint32_t rows, uint32_t cols,
                                                     uint32_t element_size) {
    size_t total_size = rows * cols * element_size;
    dram_data_[matrix_id].resize(total_size);

    // Generate pattern data
    if (element_size == 2) {
        uint16_t* ptr = reinterpret_cast<uint16_t*>(dram_data_[matrix_id].data());
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                ptr[i * cols + j] = static_cast<uint16_t>((i * cols + j) & 0xFFFF);
            }
        }
    } else if (element_size == 4) {
        float* ptr = reinterpret_cast<float*>(dram_data_[matrix_id].data());
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                ptr[i * cols + j] = static_cast<float>(i * cols + j);
            }
        }
    }

    dram_base_addresses_[matrix_id] = matrix_id * (1ULL << 30);
}

std::vector<uint8_t> DataMovementPipelineHarness::read_dram(uint32_t matrix_id) {
    auto it = dram_data_.find(matrix_id);
    if (it != dram_data_.end()) {
        return it->second;
    }
    return {};
}

// ============================================================================
// Schedule Execution
// ============================================================================

void DataMovementPipelineHarness::load_schedule(const PipelineSchedule& schedule) {
    schedule_ = schedule;
    current_op_index_ = 0;
    completed_tiles_.clear();
}

bool DataMovementPipelineHarness::execute_next_operation() {
    if (current_op_index_ >= schedule_.size()) {
        return false;
    }

    const auto& op = schedule_.operations[current_op_index_];

    // Check dependencies
    if (!dependencies_satisfied(op)) {
        return false;
    }

    // Execute operation
    if (execute_operation(op)) {
        current_op_index_++;
        return true;
    }

    return false;
}

bool DataMovementPipelineHarness::execute_operation(const ScheduleOperation& op) {
    switch (op.type) {
        case ScheduleOperation::Type::DMA_LOAD: {
            // Find DRAM data
            uint64_t dram_addr = op.address;
            if (dram_addr == 0 && dram_base_addresses_.count(op.tile_id.matrix_id)) {
                // Calculate address from tile coordinates
                dram_addr = dram_base_addresses_[op.tile_id.matrix_id];
                // Simplified: assume tile_size bytes per tile
                dram_addr += (op.tile_id.tile_row * 16 + op.tile_id.tile_col) *
                             config_.dma_config.tile_size;
            }

            uint32_t l3_buf = op.dst_buffer;
            if (l3_buf == UINT32_MAX) {
                l3_buf = dma_harness_->l3_buffers().allocate();
                if (l3_buf == UINT32_MAX) return false;
            }

            tile_l3_buffers_[op.tile_id] = l3_buf;

            auto tile_copy = op.tile_id;
            auto callback = [this, tile_copy]() {
                completed_tiles_[tile_copy] = current_cycle_;
            };

            return dma_harness_->request_tile_load(op.tile_id, dram_addr, l3_buf, callback);
        }

        case ScheduleOperation::Type::BM_L3_TO_L2: {
            uint32_t l3_buf = op.src_buffer;
            if (l3_buf == UINT32_MAX) {
                auto it = tile_l3_buffers_.find(op.tile_id);
                if (it == tile_l3_buffers_.end()) return false;
                l3_buf = it->second;
            }

            uint32_t l2_bank = op.dst_buffer;
            if (l2_bank == UINT32_MAX) {
                l2_bank = block_mover_harness_->l2_banks().allocate();
                if (l2_bank == UINT32_MAX) return false;
            }

            tile_l2_banks_[op.tile_id] = l2_bank;

            auto tile_copy = op.tile_id;
            auto callback = [this, tile_copy]() {
                completed_tiles_[tile_copy] = current_cycle_;
            };

            return block_mover_harness_->move_l3_to_l2(op.tile_id, l3_buf, l2_bank,
                                                        op.transform, callback);
        }

        case ScheduleOperation::Type::STREAM_ROWS: {
            uint32_t l2_bank = op.src_buffer;
            if (l2_bank == UINT32_MAX) {
                auto it = tile_l2_banks_.find(op.tile_id);
                if (it == tile_l2_banks_.end()) return false;
                l2_bank = it->second;
            }

            auto tile_copy = op.tile_id;
            auto callback = [this, tile_copy]() {
                completed_tiles_[tile_copy] = current_cycle_;
            };

            return streamer_harness_->stream_rows(op.tile_id, l2_bank, callback);
        }

        case ScheduleOperation::Type::STREAM_COLS: {
            uint32_t l2_bank = op.src_buffer;
            if (l2_bank == UINT32_MAX) {
                auto it = tile_l2_banks_.find(op.tile_id);
                if (it == tile_l2_banks_.end()) return false;
                l2_bank = it->second;
            }

            auto tile_copy = op.tile_id;
            auto callback = [this, tile_copy]() {
                completed_tiles_[tile_copy] = current_cycle_;
            };

            return streamer_harness_->stream_cols(op.tile_id, l2_bank, callback);
        }

        case ScheduleOperation::Type::DRAIN: {
            uint32_t l2_bank = op.dst_buffer;
            if (l2_bank == UINT32_MAX) {
                l2_bank = block_mover_harness_->l2_banks().allocate();
                if (l2_bank == UINT32_MAX) return false;
            }

            auto tile_copy = op.tile_id;
            auto callback = [this, tile_copy]() {
                completed_tiles_[tile_copy] = current_cycle_;
            };

            return streamer_harness_->drain(op.tile_id, l2_bank, callback);
        }

        case ScheduleOperation::Type::DMA_STORE:
        case ScheduleOperation::Type::BM_L2_TO_L3:
        case ScheduleOperation::Type::COMPUTE:
        case ScheduleOperation::Type::BARRIER:
            // Not yet implemented
            return true;
    }

    return false;
}

bool DataMovementPipelineHarness::dependencies_satisfied(const ScheduleOperation& op) const {
    for (const auto& dep : op.dependencies) {
        if (completed_tiles_.find(dep) == completed_tiles_.end()) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Tile Journey Tracking
// ============================================================================

std::vector<TileJourney> DataMovementPipelineHarness::get_tile_journeys() const {
    // Merge journeys from all components
    std::vector<TileJourney> result;

    // Get journeys from DMA harness
    for (const auto& [id, journey] : dma_harness_->journey_tracker().journeys()) {
        result.push_back(journey);
    }

    // Merge/update with BlockMover journeys
    for (const auto& [id, bm_journey] : block_mover_harness_->journey_tracker().journeys()) {
        bool found = false;
        for (auto& journey : result) {
            if (journey.tile_id == id) {
                // Merge timing
                if (bm_journey.l2_arrival > 0) journey.l2_arrival = bm_journey.l2_arrival;
                if (bm_journey.l2_departure > 0) journey.l2_departure = bm_journey.l2_departure;
                found = true;
                break;
            }
        }
        if (!found) {
            result.push_back(bm_journey);
        }
    }

    // Merge/update with Streamer journeys
    for (const auto& [id, str_journey] : streamer_harness_->journey_tracker().journeys()) {
        bool found = false;
        for (auto& journey : result) {
            if (journey.tile_id == id) {
                // Merge timing
                if (str_journey.l1_arrival > 0) journey.l1_arrival = str_journey.l1_arrival;
                if (str_journey.compute_start > 0) journey.compute_start = str_journey.compute_start;
                if (str_journey.compute_end > 0) journey.compute_end = str_journey.compute_end;
                if (str_journey.drain_start > 0) journey.drain_start = str_journey.drain_start;
                if (str_journey.drain_complete > 0) journey.drain_complete = str_journey.drain_complete;
                found = true;
                break;
            }
        }
        if (!found) {
            result.push_back(str_journey);
        }
    }

    return result;
}

std::optional<TileJourney> DataMovementPipelineHarness::get_tile_journey(const TileID& tile_id) const {
    auto journeys = get_tile_journeys();
    for (const auto& j : journeys) {
        if (j.tile_id == tile_id) {
            return j;
        }
    }
    return std::nullopt;
}

TileJourneyStats DataMovementPipelineHarness::get_journey_stats() const {
    TileJourneyTracker combined_tracker;

    // Combine all journeys
    auto all_journeys = get_tile_journeys();
    for (const auto& j : all_journeys) {
        auto& journey = combined_tracker.get_or_create_journey(j.tile_id);
        journey = j;
    }

    return combined_tracker.calculate_statistics();
}

// ============================================================================
// Statistics
// ============================================================================

PipelineStats DataMovementPipelineHarness::get_pipeline_stats() const {
    PipelineStats stats;
    stats.dma_stats = dma_harness_->dma_stats();
    stats.block_mover_stats = block_mover_harness_->block_mover_stats();
    stats.streamer_stats = streamer_harness_->streamer_stats();

    stats.finalize(config_.peak_gflops,
                   config_.dma_config.bandwidth_gbps,
                   config_.clock_frequency_ghz);

    return stats;
}

void DataMovementPipelineHarness::print_stats(std::ostream& os) const {
    os << "\n" << get_pipeline_stats().to_string();

    // Print journey stats
    os << "\n" << get_journey_stats().to_string();
}

void DataMovementPipelineHarness::collect_stats() {
    pipeline_stats_ = get_pipeline_stats();
}

// ============================================================================
// Validation
// ============================================================================

ValidationResult DataMovementPipelineHarness::validate() const {
    ValidationResult result = PatternHarnessBase::validate();

    // Check component validations
    auto dma_result = dma_harness_->validate();
    if (!dma_result.passed) {
        for (const auto& err : dma_result.errors) {
            result.add_error("DMA: " + err);
        }
    }

    auto bm_result = block_mover_harness_->validate();
    if (!bm_result.passed) {
        for (const auto& err : bm_result.errors) {
            result.add_error("BlockMover: " + err);
        }
    }

    auto str_result = streamer_harness_->validate();
    if (!str_result.passed) {
        for (const auto& err : str_result.errors) {
            result.add_error("Streamer: " + err);
        }
    }

    // Check schedule completion
    if (current_op_index_ < schedule_.size()) {
        result.add_warning("Schedule not fully executed: " +
                           std::to_string(schedule_.size() - current_op_index_) +
                           " operations remaining");
    }

    return result;
}

bool DataMovementPipelineHarness::validate_output(uint32_t matrix_id, const void* expected,
                                                   size_t size, double tolerance) {
    auto it = dram_data_.find(matrix_id);
    if (it == dram_data_.end()) {
        return false;
    }

    if (it->second.size() != size) {
        return false;
    }

    // Compare as floats
    const float* actual = reinterpret_cast<const float*>(it->second.data());
    const float* exp = reinterpret_cast<const float*>(expected);
    size_t count = size / sizeof(float);

    for (size_t i = 0; i < count; ++i) {
        if (std::abs(actual[i] - exp[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

} // namespace sw::kpu::harness
