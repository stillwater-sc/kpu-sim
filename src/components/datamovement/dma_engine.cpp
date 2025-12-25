#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <cmath>

#include <sw/memory/external_memory.hpp>
#include <sw/memory/address_decoder.hpp>
#include <sw/kpu/components/dma_engine.hpp>
#include <sw/kpu/components/scratchpad.hpp>
#include <sw/kpu/components/l3_tile.hpp>
#include <sw/kpu/components/l2_bank.hpp>

namespace sw::kpu {

// DMAEngine implementation - cycle-accurate multi-cycle processing like BlockMover
DMAEngine::DMAEngine(size_t engine_id, double clock_freq_ghz, double bandwidth_gb_s)
    : is_active(false)
    , engine_id(engine_id)
    , cycles_remaining(0)
    , tracing_enabled_(false)
    , trace_logger_(&trace::TraceLogger::instance())
    , clock_freq_ghz_(clock_freq_ghz)
    , bandwidth_gb_s_(bandwidth_gb_s)
    , current_cycle_(0)
    , address_decoder_(nullptr)
{
}

// ===========================================
// Address-Based API Implementation (Recommended)
// ===========================================

void DMAEngine::enqueue_transfer(Address src_addr, Address dst_addr, Size size,
                                 std::function<void()> callback) {
    // Validate that address decoder is configured
    if (!address_decoder_) {
        throw std::runtime_error(
            "AddressDecoder not configured. Call set_address_decoder() before using address-based API. "
            "See docs/dma-architecture-comparison.md for migration guide."
        );
    }

    // Validate source address range
    if (!address_decoder_->is_valid_range(src_addr, size)) {
        throw std::out_of_range(
            "Source address range [0x" + std::to_string(src_addr) + ", 0x" +
            std::to_string(src_addr + size) + ") is invalid or crosses region boundaries"
        );
    }

    // Validate destination address range
    if (!address_decoder_->is_valid_range(dst_addr, size)) {
        throw std::out_of_range(
            "Destination address range [0x" + std::to_string(dst_addr) + ", 0x" +
            std::to_string(dst_addr + size) + ") is invalid or crosses region boundaries"
        );
    }

    // Decode source and destination addresses
    auto src_route = address_decoder_->decode(src_addr);
    auto dst_route = address_decoder_->decode(dst_addr);

    // Convert sw::memory::MemoryType to sw::kpu::DMAEngine::MemoryType
    auto convert_memory_type = [](sw::memory::MemoryType type) -> MemoryType {
        switch (type) {
            case sw::memory::MemoryType::HOST_MEMORY: return MemoryType::HOST_MEMORY;
            case sw::memory::MemoryType::EXTERNAL:    return MemoryType::KPU_MEMORY;
            case sw::memory::MemoryType::L3_TILE:     return MemoryType::L3_TILE;
            case sw::memory::MemoryType::L2_BANK:     return MemoryType::L2_BANK;
            case sw::memory::MemoryType::PAGE_BUFFER: return MemoryType::PAGE_BUFFER;
            default:
                throw std::runtime_error("Unknown memory type in address decoder");
        }
    };

    // Create transfer directly with decoded routing information
    uint64_t txn_id = trace_logger_->next_transaction_id();

    Transfer transfer{
        convert_memory_type(src_route.type), src_route.id, src_route.offset,
        convert_memory_type(dst_route.type), dst_route.id, dst_route.offset,
        size, std::move(callback),
        0,  // start_cycle (will be set when transfer actually starts)
        0,  // end_cycle (not yet completed)
        txn_id
    };

    transfer_queue.emplace_back(std::move(transfer));
}

// ===========================================
// Type-Based API Implementation (Legacy - Deprecated)
// ===========================================

void DMAEngine::enqueue_transfer(MemoryType src_type, size_t src_id, Address src_addr,
                                MemoryType dst_type, size_t dst_id, Address dst_addr,
                                Size size, std::function<void()> callback) {
    // Get transaction ID
    uint64_t txn_id = trace_logger_->next_transaction_id();

    // Create transfer with timing info
    Transfer transfer{
        src_type, src_id, src_addr,
        dst_type, dst_id, dst_addr,
        size, std::move(callback),
        0,               // start_cycle (will be set when transfer actually starts)
        0,               // end_cycle (not yet completed)
        txn_id
    };

    transfer_queue.emplace_back(std::move(transfer));
}

bool DMAEngine::process_transfers(std::vector<ExternalMemory>& host_memory_regions,
                                  std::vector<ExternalMemory>& memory_banks,
                                  std::vector<L3Tile>& l3_tiles,
                                  std::vector<L2Bank>& l2_banks,
                                  std::vector<Scratchpad>& scratchpads) {
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

        // Validate destination capacity before starting the transfer
        if (transfer.dst_type == MemoryType::PAGE_BUFFER) {
            if (transfer.dst_id >= scratchpads.size()) {
                throw std::out_of_range("Invalid destination page buffer ID: " + std::to_string(transfer.dst_id));
            }
            if (transfer.dst_addr + transfer.size > scratchpads[transfer.dst_id].get_capacity()) {
                throw std::out_of_range("DMA transfer would exceed page buffer capacity: addr=" +
                    std::to_string(transfer.dst_addr) + " size=" + std::to_string(transfer.size) +
                    " capacity=" + std::to_string(scratchpads[transfer.dst_id].get_capacity()));
            }
        }

        // Calculate transfer latency in cycles based on bandwidth
        // bandwidth_gb_s_ is in GB/s, size is in bytes
        // bytes_per_cycle = (bandwidth_gb_s * 1e9) / (clock_freq_ghz * 1e9)
        //                 = bandwidth_gb_s / clock_freq_ghz
        double bytes_per_cycle = bandwidth_gb_s_ / clock_freq_ghz_;
        cycles_remaining = static_cast<trace::CycleCount>(std::ceil(transfer.size / bytes_per_cycle));
        if (cycles_remaining == 0) cycles_remaining = 1;  // Minimum 1 cycle

        // Allocate buffer for the transfer
        transfer_buffer.resize(transfer.size);

        // Log trace entry for transfer issue (now that it's actually starting)
        if (tracing_enabled_ && trace_logger_) {
            trace::TraceEntry entry(
                current_cycle_,
                trace::ComponentType::DMA_ENGINE,
                static_cast<uint32_t>(engine_id),
                trace::TransactionType::TRANSFER,
                transfer.transaction_id
            );

            // Set clock frequency for time conversion
            entry.clock_freq_ghz = clock_freq_ghz_;

            // Map MemoryType to ComponentType
            auto to_component_type = [](MemoryType type) {
                switch (type) {
                    case MemoryType::HOST_MEMORY: return trace::ComponentType::HOST_MEMORY;
                    case MemoryType::KPU_MEMORY: return trace::ComponentType::KPU_MEMORY;
                    case MemoryType::L3_TILE: return trace::ComponentType::L3_TILE;
                    case MemoryType::L2_BANK: return trace::ComponentType::L2_BANK;
                    case MemoryType::PAGE_BUFFER: return trace::ComponentType::PAGE_BUFFER;
                    default: return trace::ComponentType::UNKNOWN;
                }
            };

            // Create DMA payload
            trace::DMAPayload payload;
            payload.source = trace::MemoryLocation(
                transfer.src_addr, transfer.size, static_cast<uint32_t>(transfer.src_id),
                to_component_type(transfer.src_type)
            );
            payload.destination = trace::MemoryLocation(
                transfer.dst_addr, transfer.size, static_cast<uint32_t>(transfer.dst_id),
                to_component_type(transfer.dst_type)
            );
            payload.bytes_transferred = transfer.size;
            payload.bandwidth_gb_s = bandwidth_gb_s_;

            entry.payload = payload;
            entry.description = "DMA transfer issued";

            trace_logger_->log(std::move(entry));
        }

        // Read from source into buffer
        switch (transfer.src_type) {
            case MemoryType::HOST_MEMORY:
                if (transfer.src_id >= host_memory_regions.size()) {
                    throw std::out_of_range("Invalid source host memory region ID: " + std::to_string(transfer.src_id));
                }
                host_memory_regions[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;

            case MemoryType::KPU_MEMORY:
                if (transfer.src_id >= memory_banks.size()) {
                    throw std::out_of_range("Invalid source memory bank ID: " + std::to_string(transfer.src_id));
                }
                memory_banks[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;

            case MemoryType::L3_TILE:
                if (transfer.src_id >= l3_tiles.size()) {
                    throw std::out_of_range("Invalid source L3 tile ID: " + std::to_string(transfer.src_id));
                }
                l3_tiles[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;

            case MemoryType::L2_BANK:
                if (transfer.src_id >= l2_banks.size()) {
                    throw std::out_of_range("Invalid source L2 bank ID: " + std::to_string(transfer.src_id));
                }
                l2_banks[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;

            case MemoryType::PAGE_BUFFER:
                if (transfer.src_id >= scratchpads.size()) {
                    throw std::out_of_range("Invalid source page buffer ID: " + std::to_string(transfer.src_id));
                }
                scratchpads[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;
        }

        // Log source READ event
        if (tracing_enabled_ && trace_logger_) {
            // Map MemoryType to ComponentType
            auto to_component_type = [](MemoryType type) {
                switch (type) {
                    case MemoryType::HOST_MEMORY: return trace::ComponentType::HOST_MEMORY;
                    case MemoryType::KPU_MEMORY: return trace::ComponentType::KPU_MEMORY;
                    case MemoryType::L3_TILE: return trace::ComponentType::L3_TILE;
                    case MemoryType::L2_BANK: return trace::ComponentType::L2_BANK;
                    case MemoryType::PAGE_BUFFER: return trace::ComponentType::PAGE_BUFFER;
                    default: return trace::ComponentType::UNKNOWN;
                }
            };

            // Calculate source read latency (model realistic memory access time)
            trace::CycleCount read_latency = 1;  // Minimum 1 cycle for memory access

            trace::TraceEntry read_entry(
                current_cycle_,
                to_component_type(transfer.src_type),
                static_cast<uint32_t>(transfer.src_id),
                trace::TransactionType::READ,
                transfer.transaction_id
            );
            read_entry.clock_freq_ghz = clock_freq_ghz_;
            read_entry.complete(current_cycle_ + read_latency, trace::TransactionStatus::COMPLETED);

            trace::MemoryPayload payload;
            payload.location = trace::MemoryLocation(
                transfer.src_addr, transfer.size, static_cast<uint32_t>(transfer.src_id),
                to_component_type(transfer.src_type)
            );
            payload.is_hit = true;
            payload.latency_cycles = static_cast<uint32_t>(read_latency);
            read_entry.payload = payload;
            read_entry.description = "DMA source read";
            trace_logger_->log(std::move(read_entry));
        }
    }

    // Process one cycle of the current transfer
    if (cycles_remaining > 0) {
        cycles_remaining--;

        // Transfer completes when cycles reach 0
        if (cycles_remaining == 0) {
            auto& transfer = transfer_queue.front();

            // Set completion time
            transfer.end_cycle = current_cycle_;

            // Write to destination
            switch (transfer.dst_type) {
                case MemoryType::HOST_MEMORY:
                    if (transfer.dst_id >= host_memory_regions.size()) {
                        throw std::out_of_range("Invalid destination host memory region ID: " + std::to_string(transfer.dst_id));
                    }
                    host_memory_regions[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;

                case MemoryType::KPU_MEMORY:
                    if (transfer.dst_id >= memory_banks.size()) {
                        throw std::out_of_range("Invalid destination memory bank ID: " + std::to_string(transfer.dst_id));
                    }
                    memory_banks[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;

                case MemoryType::L3_TILE:
                    if (transfer.dst_id >= l3_tiles.size()) {
                        throw std::out_of_range("Invalid destination L3 tile ID: " + std::to_string(transfer.dst_id));
                    }
                    l3_tiles[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;

                case MemoryType::L2_BANK:
                    if (transfer.dst_id >= l2_banks.size()) {
                        throw std::out_of_range("Invalid destination L2 bank ID: " + std::to_string(transfer.dst_id));
                    }
                    l2_banks[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;

                case MemoryType::PAGE_BUFFER:
                    if (transfer.dst_id >= scratchpads.size()) {
                        throw std::out_of_range("Invalid destination page buffer ID: " + std::to_string(transfer.dst_id));
                    }
                    // Validate transfer doesn't exceed destination capacity
                    if (transfer.dst_addr + transfer.size > scratchpads[transfer.dst_id].get_capacity()) {
                        throw std::out_of_range("DMA transfer would exceed page buffer capacity: addr=" +
                            std::to_string(transfer.dst_addr) + " size=" + std::to_string(transfer.size) +
                            " capacity=" + std::to_string(scratchpads[transfer.dst_id].get_capacity()));
                    }
                    scratchpads[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;
            }

            // Log destination WRITE event
            if (tracing_enabled_ && trace_logger_) {
                // Map MemoryType to ComponentType
                auto to_component_type = [](MemoryType type) {
                    switch (type) {
                        case MemoryType::HOST_MEMORY: return trace::ComponentType::HOST_MEMORY;
                        case MemoryType::KPU_MEMORY: return trace::ComponentType::KPU_MEMORY;
                        case MemoryType::L3_TILE: return trace::ComponentType::L3_TILE;
                        case MemoryType::L2_BANK: return trace::ComponentType::L2_BANK;
                        case MemoryType::PAGE_BUFFER: return trace::ComponentType::PAGE_BUFFER;
                        default: return trace::ComponentType::UNKNOWN;
                    }
                };

                // Calculate destination write latency (model realistic memory access time)
                trace::CycleCount write_latency = 1;  // Minimum 1 cycle for memory access

                trace::TraceEntry write_entry(
                    current_cycle_,
                    to_component_type(transfer.dst_type),
                    static_cast<uint32_t>(transfer.dst_id),
                    trace::TransactionType::WRITE,
                    transfer.transaction_id
                );
                write_entry.clock_freq_ghz = clock_freq_ghz_;
                write_entry.complete(current_cycle_ + write_latency, trace::TransactionStatus::COMPLETED);

                trace::MemoryPayload payload;
                payload.location = trace::MemoryLocation(
                    transfer.dst_addr, transfer.size, static_cast<uint32_t>(transfer.dst_id),
                    to_component_type(transfer.dst_type)
                );
                payload.is_hit = true;
                payload.latency_cycles = static_cast<uint32_t>(write_latency);
                write_entry.payload = payload;
                write_entry.description = "DMA destination write";
                trace_logger_->log(std::move(write_entry));
            }

            // Log trace entry for transfer completion
            if (tracing_enabled_ && trace_logger_) {
                trace::TraceEntry entry(
                    transfer.start_cycle,
                    trace::ComponentType::DMA_ENGINE,
                    static_cast<uint32_t>(engine_id),
                    trace::TransactionType::TRANSFER,
                    transfer.transaction_id
                );

                // Set clock frequency for time conversion
                entry.clock_freq_ghz = clock_freq_ghz_;

                // Complete the entry with end cycle
                entry.complete(transfer.end_cycle, trace::TransactionStatus::COMPLETED);

                // Map MemoryType to ComponentType
                auto to_component_type = [](MemoryType type) {
                    switch (type) {
                        case MemoryType::HOST_MEMORY: return trace::ComponentType::HOST_MEMORY;
                        case MemoryType::KPU_MEMORY: return trace::ComponentType::KPU_MEMORY;
                        case MemoryType::L3_TILE: return trace::ComponentType::L3_TILE;
                        case MemoryType::L2_BANK: return trace::ComponentType::L2_BANK;
                        case MemoryType::PAGE_BUFFER: return trace::ComponentType::PAGE_BUFFER;
                        default: return trace::ComponentType::UNKNOWN;
                    }
                };

                // Create DMA payload
                trace::DMAPayload payload;
                payload.source = trace::MemoryLocation(
                    transfer.src_addr, transfer.size, static_cast<uint32_t>(transfer.src_id),
                    to_component_type(transfer.src_type)
                );
                payload.destination = trace::MemoryLocation(
                    transfer.dst_addr, transfer.size, static_cast<uint32_t>(transfer.dst_id),
                    to_component_type(transfer.dst_type)
                );
                payload.bytes_transferred = transfer.size;
                payload.bandwidth_gb_s = bandwidth_gb_s_;

                entry.payload = payload;
                entry.description = "DMA transfer completed";

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

void DMAEngine::reset() {
    transfer_queue.clear();
    transfer_buffer.clear();
    cycles_remaining = 0;
    is_active = false;
    current_cycle_ = 0;
}

} // namespace sw::kpu
