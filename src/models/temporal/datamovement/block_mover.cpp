#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cstring>

#include <sw/kpu/components/block_mover.hpp>
#include <sw/kpu/components/l3_tile.hpp>
#include <sw/kpu/components/l2_bank.hpp>
#include <sw/trace/trace_logger.hpp>

namespace sw::kpu {

// Helper function to convert TransformType to string
static const char* transform_type_to_string(BlockMover::TransformType type) {
    switch (type) {
        case BlockMover::TransformType::IDENTITY: return "IDENTITY";
        case BlockMover::TransformType::TRANSPOSE: return "TRANSPOSE";
        case BlockMover::TransformType::BLOCK_RESHAPE: return "BLOCK_RESHAPE";
        case BlockMover::TransformType::SHUFFLE_PATTERN: return "SHUFFLE_PATTERN";
        default: return "UNKNOWN";
    }
}

// BlockMover implementation - manages L3↔L2 data movement with transformations
BlockMover::BlockMover(size_t engine_id, size_t associated_l3_tile_id,
                       double clock_freq_ghz, double bandwidth_gb_s)
    : is_active(false)
    , engine_id(engine_id)
    , associated_l3_tile_id(associated_l3_tile_id)
    , cycles_remaining(0)
    , tracing_enabled_(false)
    , trace_logger_(&trace::TraceLogger::instance())
    , clock_freq_ghz_(clock_freq_ghz)
    , current_cycle_(0)
    , bandwidth_gb_s_(bandwidth_gb_s)
{
}

void BlockMover::enqueue_block_transfer(size_t src_l3_tile_id, Address src_offset,
                                       size_t dst_l2_bank_id, Address dst_offset,
                                       Size block_height, Size block_width, Size element_size,
                                       TransformType transform,
                                       std::function<void()> callback) {
    // Get transaction ID
    uint64_t txn_id = trace_logger_->next_transaction_id();

    // Create transfer with timing info
    BlockTransfer transfer{
        src_l3_tile_id, src_offset,
        dst_l2_bank_id, dst_offset,
        block_height, block_width, element_size,
        transform, std::move(callback),
        0,               // start_cycle (will be set when transfer actually starts)
        0,               // end_cycle (not yet completed)
        txn_id
    };

    transfer_queue.emplace_back(std::move(transfer));
}

bool BlockMover::process_transfers(std::vector<L3Tile>& l3_tiles,
                                  std::vector<L2Bank>& l2_banks) {
    if (transfer_queue.empty() && cycles_remaining == 0) {
        is_active = false;
        return false;
    }

    is_active = true;

    // Start a new transfer if none is active
    if (cycles_remaining == 0 && !transfer_queue.empty()) {
        auto& transfer = transfer_queue.front();

        // Set the actual start cycle now that processing begins
        transfer.start_cycle = current_cycle_;

        // Validate indices
        if (transfer.src_l3_tile_id >= l3_tiles.size()) {
            throw std::out_of_range("Invalid L3 tile ID: " + std::to_string(transfer.src_l3_tile_id));
        }
        if (transfer.dst_l2_bank_id >= l2_banks.size()) {
            throw std::out_of_range("Invalid L2 bank ID: " + std::to_string(transfer.dst_l2_bank_id));
        }

        // Calculate total block size and transfer cycles
        Size block_size = transfer.block_height * transfer.block_width * transfer.element_size;

        // Timing model: 1 cycle per 64 bytes (cache line), minimum 1 cycle
        constexpr Size CACHE_LINE_SIZE = 64;
        cycles_remaining = std::max<Cycle>(1, (block_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE);

        // Log trace entry for transfer issue (now that it's actually starting)
        if (tracing_enabled_ && trace_logger_) {
            trace::TraceEntry entry(
                current_cycle_,
                trace::ComponentType::BLOCK_MOVER,
                static_cast<uint32_t>(engine_id),
                trace::TransactionType::TRANSFER,
                transfer.transaction_id
            );

            // Set clock frequency for time conversion
            entry.clock_freq_ghz = clock_freq_ghz_;

            // Create DMA payload (BlockMover is a specialized DMA)
            trace::DMAPayload payload;
            payload.source = trace::MemoryLocation(
                transfer.src_offset, block_size, static_cast<uint32_t>(transfer.src_l3_tile_id),
                trace::ComponentType::L3_TILE
            );
            payload.destination = trace::MemoryLocation(
                transfer.dst_offset, block_size, static_cast<uint32_t>(transfer.dst_l2_bank_id),
                trace::ComponentType::L2_BANK
            );
            payload.bytes_transferred = block_size;
            payload.bandwidth_gb_s = bandwidth_gb_s_;

            entry.payload = payload;
            entry.description = std::string("BlockMover transfer issued (") +
                               transform_type_to_string(transfer.transform) + ")";

            trace_logger_->log(std::move(entry));
        }

        // Read block from L3 tile into buffer
        std::vector<std::uint8_t> src_buffer(block_size);
        transfer_buffer.resize(block_size);

        l3_tiles[transfer.src_l3_tile_id].read_block(
            transfer.src_offset, src_buffer.data(),
            transfer.block_height, transfer.block_width, transfer.element_size
        );

        // Apply transformation immediately (transform happens in block mover logic)
        apply_transform(src_buffer, transfer_buffer, transfer);
    }

    // Process one cycle of the current transfer
    if (cycles_remaining > 0) {
        cycles_remaining--;

        // Transfer completes when cycles reach 0
        if (cycles_remaining == 0) {
            auto& transfer = transfer_queue.front();

            // Set completion time
            transfer.end_cycle = current_cycle_;

            // Write transformed block to L2 bank
            l2_banks[transfer.dst_l2_bank_id].write_block(
                transfer.dst_offset, transfer_buffer.data(),
                transfer.block_height, transfer.block_width, transfer.element_size
            );

            // Log trace entry for transfer completion
            if (tracing_enabled_ && trace_logger_) {
                Size block_size = transfer.block_height * transfer.block_width * transfer.element_size;

                trace::TraceEntry entry(
                    transfer.start_cycle,
                    trace::ComponentType::BLOCK_MOVER,
                    static_cast<uint32_t>(engine_id),
                    trace::TransactionType::TRANSFER,
                    transfer.transaction_id
                );

                // Set clock frequency for time conversion
                entry.clock_freq_ghz = clock_freq_ghz_;

                // Complete the entry with end cycle
                entry.complete(transfer.end_cycle, trace::TransactionStatus::COMPLETED);

                // Create DMA payload
                trace::DMAPayload payload;
                payload.source = trace::MemoryLocation(
                    transfer.src_offset, block_size, static_cast<uint32_t>(transfer.src_l3_tile_id),
                    trace::ComponentType::L3_TILE
                );
                payload.destination = trace::MemoryLocation(
                    transfer.dst_offset, block_size, static_cast<uint32_t>(transfer.dst_l2_bank_id),
                    trace::ComponentType::L2_BANK
                );
                payload.bytes_transferred = block_size;
                payload.bandwidth_gb_s = bandwidth_gb_s_;

                entry.payload = payload;
                entry.description = std::string("BlockMover transfer completed (") +
                                   transform_type_to_string(transfer.transform) + ")";

                trace_logger_->log(std::move(entry));
            }

            // Call completion callback if provided
            if (transfer.completion_callback) {
                transfer.completion_callback();
            }

            // Remove completed transfer from queue
            transfer_queue.erase(transfer_queue.begin());
            transfer_buffer.clear();

            // Check if all work is done
            bool completed = transfer_queue.empty();
            if (completed) {
                is_active = false;
            }

            return completed;
        }
    }

    return false;
}

void BlockMover::apply_transform(const std::vector<uint8_t>& src_data,
                                std::vector<uint8_t>& dst_data,
                                const BlockTransfer& transfer) {
    switch (transfer.transform) {
        case TransformType::IDENTITY:
            identity_copy(src_data, dst_data);
            break;

        case TransformType::TRANSPOSE:
            transpose_block(src_data, dst_data,
                          transfer.block_height, transfer.block_width, transfer.element_size);
            break;

        case TransformType::BLOCK_RESHAPE:
        case TransformType::SHUFFLE_PATTERN:
            // For now, fall back to identity copy
            // These will be implemented in future iterations
            identity_copy(src_data, dst_data);
            break;

        default:
            throw std::runtime_error("Unknown transform type");
    }
}

void BlockMover::identity_copy(const std::vector<uint8_t>& src,
                              std::vector<uint8_t>& dst) {
    std::copy(src.begin(), src.end(), dst.begin());
}

void BlockMover::transpose_block(const std::vector<uint8_t>& src,
                                std::vector<uint8_t>& dst,
                                Size height, Size width, Size element_size) {
    // Transpose a 2D block: (i,j) -> (j,i)
    for (Size row = 0; row < height; ++row) {
        for (Size col = 0; col < width; ++col) {
            Size src_offset = (row * width + col) * element_size;
            Size dst_offset = (col * height + row) * element_size;

            // Copy element from (row,col) to (col,row)
            std::memcpy(dst.data() + dst_offset,
                       src.data() + src_offset,
                       element_size);
        }
    }
}

void BlockMover::reset() {
    transfer_queue.clear();
    transfer_buffer.clear();
    cycles_remaining = 0;
    is_active = false;
    current_cycle_ = 0;
}

} // namespace sw::kpu