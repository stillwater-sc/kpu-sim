// Vector Engine Implementation
// Inline bias addition and activation processing during L1→L2 transfer

#include <sw/kpu/components/vector_engine.hpp>
#include <stdexcept>
#include <cstring>

namespace sw::kpu {

// ============================================================================
// Constructors
// ============================================================================

VectorEngine::VectorEngine(size_t id)
    : VectorEngine(id, VectorEngineConfig{}) {}

VectorEngine::VectorEngine(size_t id, const VectorEngineConfig& config)
    : id_(id), config_(config), sfu_(config.sfu_config) {
    bias_buffer_.resize(config_.bias_buffer_size);
    input_buffer_.resize(config_.vector_width);
    output_buffer_.resize(config_.vector_width);
}

// ============================================================================
// Configuration
// ============================================================================

void VectorEngine::configure(const VectorEngineConfig& config) {
    config_ = config;
    sfu_.configure(config.sfu_config.activation, config.sfu_config.table_size);
    bias_buffer_.resize(config_.bias_buffer_size);
    input_buffer_.resize(config_.vector_width);
    output_buffer_.resize(config_.vector_width);
}

void VectorEngine::set_activation(ActivationType activation) {
    sfu_.configure(activation);
    config_.sfu_config.activation = activation;
}

void VectorEngine::preload_bias(const float* bias_data, Size count) {
    if (count > config_.bias_buffer_size) {
        throw std::invalid_argument("VectorEngine: bias vector exceeds buffer size");
    }
    std::memcpy(bias_buffer_.data(), bias_data, count * sizeof(float));
    bias_loaded_ = true;
}

// ============================================================================
// Operation Queue
// ============================================================================

void VectorEngine::enqueue_operation(const VEOperation& op) {
    op_queue_.push(op);

    // If idle, start processing immediately
    if (state_ == State::IDLE) {
        start_operation();
    }
}

// ============================================================================
// Cycle-Accurate Simulation
// ============================================================================

bool VectorEngine::update(Cycle cycle, L1ReadFunc l1_read, L2WriteFunc l2_write) {
    if (!config_.enabled) {
        stats_.cycles_idle++;
        return false;
    }

    switch (state_) {
        case State::IDLE:
            if (!op_queue_.empty()) {
                start_operation();
                op_start_cycle_ = cycle;
            } else {
                stats_.cycles_idle++;
            }
            return false;

        case State::STARTING:
            // Transition to processing on first row
            state_ = State::PROCESSING;
            current_row_ = 0;
            stats_.cycles_active++;
            return false;

        case State::PROCESSING:
            // Process one row per cycle (pipelined)
            process_row(l1_read, l2_write);
            stats_.cycles_active++;

            current_row_++;
            if (current_row_ >= current_op_.height) {
                state_ = State::DRAINING;
            }
            return false;

        case State::DRAINING:
            // Allow pipeline to drain (pipeline_depth cycles)
            stats_.cycles_active++;
            if (cycle >= op_start_cycle_ + estimate_cycles(current_op_.height, current_op_.width)) {
                state_ = State::COMPLETING;
            }
            return false;

        case State::COMPLETING:
            finish_operation();
            return true;

        default:
            return false;
    }
}

void VectorEngine::start_operation() {
    if (op_queue_.empty()) {
        state_ = State::IDLE;
        return;
    }

    current_op_ = op_queue_.front();
    op_queue_.pop();

    // Configure SFU for this operation
    sfu_.configure(current_op_.activation);

    // Resize buffers if needed
    if (current_op_.width > input_buffer_.size()) {
        input_buffer_.resize(current_op_.width);
        output_buffer_.resize(current_op_.width);
    }

    state_ = State::STARTING;
    current_row_ = 0;
}

void VectorEngine::process_row(L1ReadFunc l1_read, L2WriteFunc l2_write) {
    // Calculate addresses for current row
    Address l1_addr = current_op_.l1_base_addr +
                      current_row_ * (current_op_.row_stride ? current_op_.row_stride :
                                      current_op_.width * current_op_.element_size);
    Address l2_addr = current_op_.l2_base_addr +
                      current_row_ * (current_op_.row_stride ? current_op_.row_stride :
                                      current_op_.width * current_op_.element_size);

    Size row_bytes = current_op_.width * current_op_.element_size;

    // Read row from L1
    l1_read(current_op_.l1_scratchpad_id, l1_addr, input_buffer_.data(), row_bytes);
    stats_.elements_processed += current_op_.width;

    // Apply bias if enabled
    if (current_op_.bias_enabled && bias_loaded_) {
        for (Size col = 0; col < current_op_.width; ++col) {
            input_buffer_[col] += bias_buffer_[col];
        }
        stats_.bias_additions += current_op_.width;
    }

    // Apply activation function via SFU
    if (current_op_.activation != ActivationType::NONE) {
        sfu_.evaluate_vector(input_buffer_.data(), output_buffer_.data(), current_op_.width);
        stats_.activations_computed += current_op_.width;
    } else {
        // No activation, just copy
        std::memcpy(output_buffer_.data(), input_buffer_.data(), row_bytes);
    }

    // Write row to L2
    l2_write(current_op_.l2_bank_id, l2_addr, output_buffer_.data(), row_bytes);
}

void VectorEngine::finish_operation() {
    stats_.operations_completed++;

    // Call completion callback if provided
    if (current_op_.completion_callback) {
        current_op_.completion_callback();
    }

    // Check for more operations
    if (!op_queue_.empty()) {
        start_operation();
    } else {
        state_ = State::IDLE;
    }
}

// ============================================================================
// Immediate Processing (Non-Pipelined)
// ============================================================================

void VectorEngine::process_row_immediate(const float* input, float* output,
                                         Size width, Size /* bias_row */) {
    // Copy input to working buffer
    std::vector<float> working(width);
    std::memcpy(working.data(), input, width * sizeof(float));

    // Apply bias if enabled and loaded
    if (bias_loaded_) {
        for (Size col = 0; col < width; ++col) {
            working[col] += bias_buffer_[col];
        }
    }

    // Apply activation via SFU
    sfu_.evaluate_vector(working.data(), output, width);
}

void VectorEngine::process_tile_immediate(const float* input, float* output,
                                          Size height, Size width) {
    for (Size row = 0; row < height; ++row) {
        const float* in_row = input + row * width;
        float* out_row = output + row * width;

        // Working buffer for this row
        std::vector<float> working(width);
        std::memcpy(working.data(), in_row, width * sizeof(float));

        // Apply bias if loaded
        if (bias_loaded_) {
            for (Size col = 0; col < width; ++col) {
                working[col] += bias_buffer_[col];
            }
        }

        // Apply activation
        sfu_.evaluate_vector(working.data(), out_row, width);
    }
}

// ============================================================================
// State Management
// ============================================================================

void VectorEngine::reset() {
    state_ = State::IDLE;
    current_row_ = 0;
    op_start_cycle_ = 0;

    // Clear operation queue
    while (!op_queue_.empty()) {
        op_queue_.pop();
    }

    // Clear buffers
    std::fill(input_buffer_.begin(), input_buffer_.end(), 0.0f);
    std::fill(output_buffer_.begin(), output_buffer_.end(), 0.0f);

    // Note: bias_buffer_ is intentionally NOT cleared (may still be valid)
}

// ============================================================================
// Statistics
// ============================================================================

void VectorEngine::reset_stats() {
    stats_ = VEStats{};
}

// ============================================================================
// Timing
// ============================================================================

Cycle VectorEngine::estimate_cycles(Size height, Size width) const {
    // Each row takes ceil(width / vector_width) chunks
    // Pipeline adds latency for first result
    Size chunks_per_row = (width + config_.vector_width - 1) / config_.vector_width;
    Size total_chunks = height * chunks_per_row;

    return total_chunks + config_.pipeline_depth;
}

} // namespace sw::kpu
