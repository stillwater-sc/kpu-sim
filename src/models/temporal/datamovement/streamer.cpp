#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <cmath>

#include <sw/kpu/components/streamer.hpp>
#include <sw/kpu/components/l2_bank.hpp>
#include <sw/kpu/components/l1_buffer.hpp>
#include <sw/trace/trace_logger.hpp>

namespace sw::kpu {

// Helper function to convert StreamDirection to string
static const char* stream_direction_to_string(Streamer::StreamDirection dir) {
    switch (dir) {
        case Streamer::StreamDirection::L2_TO_L1: return "L2_TO_L1";
        case Streamer::StreamDirection::L1_TO_L2: return "L1_TO_L2";
        default: return "UNKNOWN";
    }
}

// Helper function to convert StreamType to string
static const char* stream_type_to_string(Streamer::StreamType type) {
    switch (type) {
        case Streamer::StreamType::ROW_STREAM: return "ROW_STREAM";
        case Streamer::StreamType::COLUMN_STREAM: return "COLUMN_STREAM";
        default: return "UNKNOWN";
    }
}

Streamer::Streamer(size_t streamer_id, double clock_freq_ghz, double bandwidth_gb_s)
    : current_stream(nullptr)
    , streamer_id(streamer_id)
    , tracing_enabled_(false)
    , trace_logger_(&trace::TraceLogger::instance())
    , clock_freq_ghz_(clock_freq_ghz)
    , current_cycle_(0)
    , bandwidth_gb_s_(bandwidth_gb_s)
{
}

Streamer::Streamer(const Streamer& other)
    : current_stream(nullptr), streamer_id(other.streamer_id) {
    // Don't copy the current stream state - new instance starts fresh
    // This is appropriate since copying a streamer should create a new independent instance
}

Streamer& Streamer::operator=(const Streamer& other) {
    if (this != &other) {
        streamer_id = other.streamer_id;
        current_stream.reset(); // Reset any existing stream
        // Clear any queued streams when copying
        while (!stream_queue.empty()) {
            stream_queue.pop();
        }
    }
    return *this;
}

void Streamer::enqueue_stream(const StreamConfig& config) {
    // Validate configuration
    if (config.matrix_height == 0 || config.matrix_width == 0 ||
        config.element_size == 0 || config.compute_fabric_size == 0) {
        throw std::invalid_argument("Invalid stream configuration: zero dimensions");
    }

    if (config.cache_line_size == 0) {
        throw std::invalid_argument("Invalid cache line size");
    }

    // Get transaction ID and add timing info
    uint64_t txn_id = trace_logger_->next_transaction_id();
    StreamConfig config_with_timing = config;
    config_with_timing.start_cycle = current_cycle_;
    config_with_timing.end_cycle = 0;
    config_with_timing.transaction_id = txn_id;

    // Calculate stream size
    Size stream_size = config.matrix_height * config.matrix_width * config.element_size;

    stream_queue.push(config_with_timing);

    // Log trace entry for stream issue
    if (tracing_enabled_ && trace_logger_) {
        trace::TraceEntry entry(
            current_cycle_,
            trace::ComponentType::STREAMER,
            static_cast<uint32_t>(streamer_id),
            trace::TransactionType::TRANSFER,
            txn_id
        );

        // Set clock frequency for time conversion
        entry.clock_freq_ghz = clock_freq_ghz_;

        // Create DMA payload (Streamer is a specialized DMA)
        trace::DMAPayload payload;

        // Set source/destination based on direction
        if (config.direction == StreamDirection::L2_TO_L1) {
            payload.source = trace::MemoryLocation(
                config.l2_base_addr, stream_size, static_cast<uint32_t>(config.l2_bank_id),
                trace::ComponentType::L2_BANK
            );
            payload.destination = trace::MemoryLocation(
                config.l1_base_addr, stream_size, static_cast<uint32_t>(config.l1_buffer_id),
                trace::ComponentType::L1
            );
        } else {
            payload.source = trace::MemoryLocation(
                config.l1_base_addr, stream_size, static_cast<uint32_t>(config.l1_buffer_id),
                trace::ComponentType::L1
            );
            payload.destination = trace::MemoryLocation(
                config.l2_base_addr, stream_size, static_cast<uint32_t>(config.l2_bank_id),
                trace::ComponentType::L2_BANK
            );
        }

        payload.bytes_transferred = stream_size;
        payload.bandwidth_gb_s = bandwidth_gb_s_;

        entry.payload = payload;
        entry.description = std::string("Streamer ") +
                           stream_direction_to_string(config.direction) + " " +
                           stream_type_to_string(config.stream_type) + " enqueued";

        trace_logger_->log(std::move(entry));
    }
}

bool Streamer::update(Cycle current_cycle,
                              std::vector<L2Bank>& l2_banks,
                              std::vector<L1Buffer>& l1_buffers) {
    // Start a new stream if none is active and queue has work
    if (!current_stream && !stream_queue.empty()) {
        initialize_stream_state(stream_queue.front());
        stream_queue.pop();
    }

    if (!current_stream) {
        return false; // No work to do
    }

    // Process the current stream
    bool stream_complete = advance_stream_cycle(current_cycle, l2_banks, l1_buffers);

    if (stream_complete) {
        StreamConfig& config = current_stream->config;

        // Set completion cycle
        config.end_cycle = current_cycle;

        // Log trace entry for stream completion
        if (tracing_enabled_ && trace_logger_) {
            Size stream_size = config.matrix_height * config.matrix_width * config.element_size;

            trace::TraceEntry entry(
                config.start_cycle,
                trace::ComponentType::STREAMER,
                static_cast<uint32_t>(streamer_id),
                trace::TransactionType::TRANSFER,
                config.transaction_id
            );

            // Set clock frequency for time conversion
            entry.clock_freq_ghz = clock_freq_ghz_;

            // Complete the entry with end cycle
            entry.complete(config.end_cycle, trace::TransactionStatus::COMPLETED);

            // Create DMA payload
            trace::DMAPayload payload;

            // Set source/destination based on direction
            if (config.direction == StreamDirection::L2_TO_L1) {
                payload.source = trace::MemoryLocation(
                    config.l2_base_addr, stream_size, static_cast<uint32_t>(config.l2_bank_id),
                    trace::ComponentType::L2_BANK
                );
                payload.destination = trace::MemoryLocation(
                    config.l1_base_addr, stream_size, static_cast<uint32_t>(config.l1_buffer_id),
                    trace::ComponentType::L1
                );
            } else {
                payload.source = trace::MemoryLocation(
                    config.l1_base_addr, stream_size, static_cast<uint32_t>(config.l1_buffer_id),
                    trace::ComponentType::L1
                );
                payload.destination = trace::MemoryLocation(
                    config.l2_base_addr, stream_size, static_cast<uint32_t>(config.l2_bank_id),
                    trace::ComponentType::L2_BANK
                );
            }

            payload.bytes_transferred = stream_size;
            payload.bandwidth_gb_s = bandwidth_gb_s_;

            entry.payload = payload;
            entry.description = std::string("Streamer ") +
                               stream_direction_to_string(config.direction) + " " +
                               stream_type_to_string(config.stream_type) + " completed";

            trace_logger_->log(std::move(entry));
        }

        // Call completion callback if provided
        if (current_stream->config.completion_callback) {
            current_stream->config.completion_callback();
        }
        current_stream.reset();
        return true;
    }

    return false;
}

void Streamer::initialize_stream_state(const StreamConfig& config) {
    current_stream = std::make_unique<StreamState>();
    current_stream->config = config;
    current_stream->is_active = true;
    current_stream->start_cycle = 0; // Will be set when stream actually starts
    current_stream->current_row = 0;
    current_stream->current_col = 0;
    current_stream->elements_streamed_this_cycle = 0;

    // Initialize stagger offsets
    Size fabric_size = config.compute_fabric_size;
    current_stream->row_stagger_offset.resize(fabric_size, 0);
    current_stream->col_stagger_offset.resize(fabric_size, 0);

    // Initialize cache line buffer
    current_stream->cache_line_buffer.resize(config.cache_line_size);
    current_stream->buffer_valid = false;
    current_stream->buffered_cache_line_addr = 0;
}

bool Streamer::advance_stream_cycle(Cycle current_cycle,
                                            std::vector<L2Bank>& l2_banks,
                                            std::vector<L1Buffer>& l1_buffers) {
    const StreamConfig& config = current_stream->config;

    // Set start cycle on first call
    if (current_stream->start_cycle == 0) {
        current_stream->start_cycle = current_cycle;
    }

    // Validate indices
    if (config.l2_bank_id >= l2_banks.size() ||
        config.l1_buffer_id >= l1_buffers.size()) {
        throw std::out_of_range("Invalid L2 bank or L1 buffer ID");
    }

    bool stream_complete = false;

    // Route to appropriate streaming function based on direction and type
    switch (config.direction) {
        case StreamDirection::L2_TO_L1:
            if (config.stream_type == StreamType::ROW_STREAM) {
                stream_complete = stream_row_l2_to_l1(current_cycle, l2_banks, l1_buffers);
            } else {
                stream_complete = stream_column_l2_to_l1(current_cycle, l2_banks, l1_buffers);
            }
            break;

        case StreamDirection::L1_TO_L2:
            if (config.stream_type == StreamType::ROW_STREAM) {
                stream_complete = stream_row_l1_to_l2(current_cycle, l2_banks, l1_buffers);
            } else {
                stream_complete = stream_column_l1_to_l2(current_cycle, l2_banks, l1_buffers);
            }
            break;
    }

    return stream_complete;
}

bool Streamer::stream_row_l2_to_l1(Cycle current_cycle,
                                           std::vector<L2Bank>& l2_banks,
                                           std::vector<L1Buffer>& l1_buffers) {
    (void)current_cycle; // Suppress unused parameter warning
    const StreamConfig& config = current_stream->config;
    L2Bank& l2_bank = l2_banks[config.l2_bank_id];
    L1Buffer& l1_buffer = l1_buffers[config.l1_buffer_id];

    Size fabric_size = config.compute_fabric_size;

    // Simplified row streaming for test compatibility
    // Stream elements from the current row to consecutive L1 positions
    // FIX: Accumulate L1 address offset so we don't overwrite previous chunks

    Size elements_to_stream = std::min(fabric_size, config.matrix_width - current_stream->current_col);

    for (Size i = 0; i < elements_to_stream; ++i) {
        Size current_matrix_col = current_stream->current_col + i;

        // Calculate L2 address: row-major order (row * width + col)
        Address l2_addr = config.l2_base_addr +
                         (current_stream->current_row * config.matrix_width + current_matrix_col) * config.element_size;

        // Calculate L1 address: accumulate offset for full matrix
        // Position in the full matrix being assembled in L1
        Size offset = current_stream->current_row * config.matrix_width + current_matrix_col;
        Address l1_addr = config.l1_base_addr + offset * config.element_size;

        // Transfer data directly from L2 to L1
        std::vector<uint8_t> element_data(config.element_size);
        l2_bank.read(l2_addr, element_data.data(), config.element_size);
        l1_buffer.write(l1_addr, element_data.data(), config.element_size);
    }

    // Advance to next position
    current_stream->current_col += elements_to_stream;
    if (current_stream->current_col >= config.matrix_width) {
        current_stream->current_col = 0;
        current_stream->current_row++;
    }

    // Check if streaming is complete
    if (current_stream->current_row >= config.matrix_height) {
        return true; // Stream complete
    }

    return false; // Stream continues
}

bool Streamer::stream_column_l2_to_l1(Cycle current_cycle,
                                              std::vector<L2Bank>& l2_banks,
                                              std::vector<L1Buffer>& l1_buffers) {
    (void)current_cycle; // Suppress unused parameter warning
    const StreamConfig& config = current_stream->config;
    L2Bank& l2_bank = l2_banks[config.l2_bank_id];
    L1Buffer& l1_buffer = l1_buffers[config.l1_buffer_id];

    Size fabric_size = config.compute_fabric_size;

    // Simplified column streaming for test compatibility
    // Stream elements from the current column to consecutive L1 positions
    // FIX: Accumulate L1 address offset so we don't overwrite previous chunks

    Size elements_to_stream = std::min(fabric_size, config.matrix_height - current_stream->current_row);

    for (Size i = 0; i < elements_to_stream; ++i) {
        Size current_matrix_row = current_stream->current_row + i;

        // Calculate L2 address: row-major order (row * width + col)
        Address l2_addr = config.l2_base_addr +
                         (current_matrix_row * config.matrix_width + current_stream->current_col) * config.element_size;

        // Calculate L1 address: accumulate offset for full matrix
        // Position in the full matrix being assembled in L1
        Size offset = current_matrix_row * config.matrix_width + current_stream->current_col;
        Address l1_addr = config.l1_base_addr + offset * config.element_size;

        // Transfer data directly from L2 to L1
        std::vector<uint8_t> element_data(config.element_size);
        l2_bank.read(l2_addr, element_data.data(), config.element_size);
        l1_buffer.write(l1_addr, element_data.data(), config.element_size);
    }

    // Advance to next position
    current_stream->current_row += elements_to_stream;
    if (current_stream->current_row >= config.matrix_height) {
        current_stream->current_row = 0;
        current_stream->current_col++;
    }

    // Check if streaming is complete
    if (current_stream->current_col >= config.matrix_width) {
        return true; // Stream complete
    }

    return false; // Stream continues
}

bool Streamer::stream_row_l1_to_l2(Cycle current_cycle,
                                           std::vector<L2Bank>& l2_banks,
                                           std::vector<L1Buffer>& l1_buffers) {
    (void)current_cycle; // Suppress unused parameter warning
    const StreamConfig& config = current_stream->config;
    L2Bank& l2_bank = l2_banks[config.l2_bank_id];
    L1Buffer& l1_buffer = l1_buffers[config.l1_buffer_id];

    Size fabric_size = config.compute_fabric_size;

    // Simplified L1→L2 row streaming
    // FIX: Read from proper L1 offset (full matrix layout)
    Size elements_to_stream = std::min(fabric_size, config.matrix_width - current_stream->current_col);

    for (Size i = 0; i < elements_to_stream; ++i) {
        Size current_matrix_col = current_stream->current_col + i;

        // Calculate L1 address: read from full matrix layout
        Size offset = current_stream->current_row * config.matrix_width + current_matrix_col;
        Address l1_addr = config.l1_base_addr + offset * config.element_size;

        // Calculate L2 address: write to row in row-major matrix (row * width + col)
        Address l2_addr = config.l2_base_addr + offset * config.element_size;

        // Transfer data
        std::vector<uint8_t> element_data(config.element_size);
        l1_buffer.read(l1_addr, element_data.data(), config.element_size);
        l2_bank.write(l2_addr, element_data.data(), config.element_size);
    }

    // Advance to next position
    current_stream->current_col += elements_to_stream;
    if (current_stream->current_col >= config.matrix_width) {
        current_stream->current_col = 0;
        current_stream->current_row++;
    }

    // Check if all rows processed
    if (current_stream->current_row >= config.matrix_height) {
        return true; // Stream complete
    }

    return false;
}

bool Streamer::stream_column_l1_to_l2(Cycle current_cycle,
                                              std::vector<L2Bank>& l2_banks,
                                              std::vector<L1Buffer>& l1_buffers) {
    (void)current_cycle; // Suppress unused parameter warning
    const StreamConfig& config = current_stream->config;
    L2Bank& l2_bank = l2_banks[config.l2_bank_id];
    L1Buffer& l1_buffer = l1_buffers[config.l1_buffer_id];

    Size fabric_size = config.compute_fabric_size;

    // Simplified L1→L2 column streaming
    // FIX: Read from proper L1 offset (full matrix layout)
    Size elements_to_stream = std::min(fabric_size, config.matrix_height - current_stream->current_row);

    for (Size i = 0; i < elements_to_stream; ++i) {
        Size current_matrix_row = current_stream->current_row + i;

        // Calculate L1 address: read from full matrix layout
        Size offset = current_matrix_row * config.matrix_width + current_stream->current_col;
        Address l1_addr = config.l1_base_addr + offset * config.element_size;

        // Calculate L2 address: write to column in row-major matrix (row * width + col)
        Address l2_addr = config.l2_base_addr + offset * config.element_size;

        // Transfer data
        std::vector<uint8_t> element_data(config.element_size);
        l1_buffer.read(l1_addr, element_data.data(), config.element_size);
        l2_bank.write(l2_addr, element_data.data(), config.element_size);
    }

    // Advance to next position
    current_stream->current_row += elements_to_stream;
    if (current_stream->current_row >= config.matrix_height) {
        current_stream->current_row = 0;
        current_stream->current_col++;
    }

    // Check if all columns processed
    if (current_stream->current_col >= config.matrix_width) {
        return true; // Stream complete
    }

    return false;
}

void Streamer::fetch_cache_line_if_needed(L2Bank& l2_bank, Address addr) {
    Address cache_line_addr = (addr / current_stream->config.cache_line_size) * current_stream->config.cache_line_size;

    if (!current_stream->buffer_valid || current_stream->buffered_cache_line_addr != cache_line_addr) {
        l2_bank.read_cache_line(cache_line_addr,
                               current_stream->cache_line_buffer.data(),
                               current_stream->config.cache_line_size);
        current_stream->buffered_cache_line_addr = cache_line_addr;
        current_stream->buffer_valid = true;
    }
}

void Streamer::write_cache_line_if_needed(L2Bank& l2_bank, Address addr) {
    Address cache_line_addr = (addr / current_stream->config.cache_line_size) * current_stream->config.cache_line_size;

    if (current_stream->buffer_valid && current_stream->buffered_cache_line_addr == cache_line_addr) {
        l2_bank.write_cache_line(cache_line_addr,
                                current_stream->cache_line_buffer.data(),
                                current_stream->config.cache_line_size);
    }
}

Size Streamer::calculate_stagger_delay(Size fabric_position, StreamType type) const {
    // Systolic array staggering: each row/column starts one cycle after the previous
    // This ensures proper data flow timing for the systolic array

    switch (type) {
        case StreamType::ROW_STREAM:
            // Row streaming: each fabric row receives data one cycle after the previous
            // Row 0 gets data at cycle 0, row 1 at cycle 1, etc.
            return fabric_position;

        case StreamType::COLUMN_STREAM:
            // Column streaming: each fabric column receives data one cycle after the previous
            // Column 0 gets data at cycle 0, column 1 at cycle 1, etc.
            return fabric_position;

        default:
            return fabric_position;
    }
}

bool Streamer::should_stream_this_cycle(Size fabric_position, Cycle current_cycle) const {
    Size stagger_delay = calculate_stagger_delay(fabric_position, current_stream->config.stream_type);
    Cycle effective_cycle = current_cycle - current_stream->start_cycle;

    return effective_cycle >= stagger_delay;
}

Address Streamer::calculate_row_address(Size row, Size col) const {
    const StreamConfig& config = current_stream->config;
    return config.l2_base_addr + (row * config.matrix_width + col) * config.element_size;
}

Address Streamer::calculate_column_address(Size row, Size col) const {
    const StreamConfig& config = current_stream->config;
    return config.l2_base_addr + (row * config.matrix_width + col) * config.element_size;
}

Size Streamer::calculate_stream_cycles(Size matrix_height, Size matrix_width, Size fabric_size) {
    // Conservative estimate: each fabric-sized block takes at least one cycle
    Size row_blocks = (matrix_height + fabric_size - 1) / fabric_size;
    Size col_blocks = (matrix_width + fabric_size - 1) / fabric_size;
    return row_blocks * col_blocks;
}

Size Streamer::calculate_elements_per_cycle(Size fabric_size) {
    return fabric_size; // Up to fabric_size elements can be streamed per cycle
}

void Streamer::reset() {
    current_stream.reset();
    while (!stream_queue.empty()) {
        stream_queue.pop();
    }
    current_cycle_ = 0;
}

void Streamer::abort_current_stream() {
    current_stream.reset();
}

} // namespace sw::kpu