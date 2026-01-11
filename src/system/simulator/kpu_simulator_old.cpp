
#include <algorithm>
#include <random>
#include <iostream>
#include <cassert>
#include <cstring>

#include "sw/kpu/kpu_simulator.hpp"
namespace sw::kpu {

// ExternalMemory implementation - manages its own memory model
ExternalMemory::ExternalMemory(Size capacity_mb, Size bandwidth_gbps) 
    : capacity(capacity_mb * 1024 * 1024), 
      bandwidth_bytes_per_cycle(bandwidth_gbps * 1000000000 / 8 / 1000000000), // Assuming 1GHz clock
      last_access_cycle(0) {
    memory_model.resize(capacity);
    std::fill(memory_model.begin(), memory_model.end(), 0);
}

void ExternalMemory::read(Address addr, void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("Memory read out of bounds");
    }
    std::memcpy(data, memory_model.data() + addr, size);
}

void ExternalMemory::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("Memory write out of bounds");
    }
    std::memcpy(memory_model.data() + addr, data, size);
}

bool ExternalMemory::is_ready() const {
    // Simplified: assume always ready for now
    return true;
}

void ExternalMemory::reset() {
    std::fill(memory_model.begin(), memory_model.end(), 0);
    last_access_cycle = 0;
}

// Scratchpad implementation - manages its own memory model
Scratchpad::Scratchpad(Size capacity_kb) 
    : capacity(capacity_kb * 1024) {
    memory_model.resize(capacity);
    std::fill(memory_model.begin(), memory_model.end(), 0);
}

void Scratchpad::read(Address addr, void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("Scratchpad read out of bounds");
    }
    std::memcpy(data, memory_model.data() + addr, size);
}

void Scratchpad::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("Scratchpad write out of bounds");
    }
    std::memcpy(memory_model.data() + addr, data, size);
}

void Scratchpad::reset() {
    std::fill(memory_model.begin(), memory_model.end(), 0);
}

// L3Tile implementation - distributed L3 cache tiles
L3Tile::L3Tile(size_t tile_id, Size capacity_kb)
    : capacity(capacity_kb * 1024), tile_id(tile_id) {
    memory_model.resize(capacity);
    std::fill(memory_model.begin(), memory_model.end(), 0);
}

void L3Tile::read(Address addr, void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("L3Tile read out of bounds");
    }
    std::memcpy(data, memory_model.data() + addr, size);
}

void L3Tile::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("L3Tile write out of bounds");
    }
    std::memcpy(memory_model.data() + addr, data, size);
}

void L3Tile::read_block(Address base_addr, void* data,
                       Size block_height, Size block_width, Size element_size,
                       Size stride) {
    if (stride == 0) {
        stride = block_width * element_size; // Contiguous case
    }

    uint8_t* dst_ptr = static_cast<uint8_t*>(data);
    for (Size row = 0; row < block_height; ++row) {
        Address row_addr = base_addr + row * stride;
        Size row_size = block_width * element_size;
        read(row_addr, dst_ptr, row_size);
        dst_ptr += row_size;
    }
}

void L3Tile::write_block(Address base_addr, const void* data,
                        Size block_height, Size block_width, Size element_size,
                        Size stride) {
    if (stride == 0) {
        stride = block_width * element_size; // Contiguous case
    }

    const uint8_t* src_ptr = static_cast<const uint8_t*>(data);
    for (Size row = 0; row < block_height; ++row) {
        Address row_addr = base_addr + row * stride;
        Size row_size = block_width * element_size;
        write(row_addr, src_ptr, row_size);
        src_ptr += row_size;
    }
}

void L3Tile::reset() {
    std::fill(memory_model.begin(), memory_model.end(), 0);
}

// L2Bank implementation - L2 cache banks
L2Bank::L2Bank(size_t bank_id, Size capacity_kb)
    : capacity(capacity_kb * 1024), bank_id(bank_id) {
    memory_model.resize(capacity);
    std::fill(memory_model.begin(), memory_model.end(), 0);
}

void L2Bank::read(Address addr, void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("L2Bank read out of bounds");
    }
    std::memcpy(data, memory_model.data() + addr, size);
}

void L2Bank::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("L2Bank write out of bounds");
    }
    std::memcpy(memory_model.data() + addr, data, size);
}

void L2Bank::read_cache_line(Address addr, void* data, Size cache_line_size) {
    read(addr, data, cache_line_size);
}

void L2Bank::write_cache_line(Address addr, const void* data, Size cache_line_size) {
    write(addr, data, cache_line_size);
}

void L2Bank::read_block(Address base_addr, void* data,
                       Size block_height, Size block_width, Size element_size,
                       Size stride) {
    if (stride == 0) {
        stride = block_width * element_size; // Contiguous case
    }

    uint8_t* dst_ptr = static_cast<uint8_t*>(data);
    for (Size row = 0; row < block_height; ++row) {
        Address row_addr = base_addr + row * stride;
        Size row_size = block_width * element_size;
        read(row_addr, dst_ptr, row_size);
        dst_ptr += row_size;
    }
}

void L2Bank::write_block(Address base_addr, const void* data,
                        Size block_height, Size block_width, Size element_size,
                        Size stride) {
    if (stride == 0) {
        stride = block_width * element_size; // Contiguous case
    }

    const uint8_t* src_ptr = static_cast<const uint8_t*>(data);
    for (Size row = 0; row < block_height; ++row) {
        Address row_addr = base_addr + row * stride;
        Size row_size = block_width * element_size;
        write(row_addr, src_ptr, row_size);
        src_ptr += row_size;
    }
}

void L2Bank::reset() {
    std::fill(memory_model.begin(), memory_model.end(), 0);
}

// DMAEngine implementation - manages its own transfer queue
DMAEngine::DMAEngine(size_t engine_id)
    : is_active(false), engine_id(engine_id) {
}

void DMAEngine::enqueue_transfer(MemoryType src_type, size_t src_id, Address src_addr,
                                MemoryType dst_type, size_t dst_id, Address dst_addr,
                                Size size, std::function<void()> callback) {
    transfer_queue.emplace_back(Transfer{
        src_type, src_id, src_addr,
        dst_type, dst_id, dst_addr,
        size, std::move(callback)
    });
}

bool DMAEngine::process_transfers(std::vector<ExternalMemory>& memory_banks,
                                 std::vector<Scratchpad>& scratchpads) {
    if (transfer_queue.empty()) {
        is_active = false;
        return false;
    }

    is_active = true;
    auto& transfer = transfer_queue.front();

    // Allocate temporary buffer for the transfer
    std::vector<std::uint8_t> buffer(transfer.size);

    // Read from source
    if (transfer.src_type == MemoryType::EXTERNAL) {
        if (transfer.src_id >= memory_banks.size()) {
            throw std::out_of_range("Invalid source memory bank ID: " + std::to_string(transfer.src_id));
        }
        memory_banks[transfer.src_id].read(transfer.src_addr, buffer.data(), transfer.size);
    } else {
        if (transfer.src_id >= scratchpads.size()) {
            throw std::out_of_range("Invalid source scratchpad ID: " + std::to_string(transfer.src_id));
        }
        scratchpads[transfer.src_id].read(transfer.src_addr, buffer.data(), transfer.size);
    }

    // Write to destination
    if (transfer.dst_type == MemoryType::EXTERNAL) {
        if (transfer.dst_id >= memory_banks.size()) {
            throw std::out_of_range("Invalid destination memory bank ID: " + std::to_string(transfer.dst_id));
        }
        memory_banks[transfer.dst_id].write(transfer.dst_addr, buffer.data(), transfer.size);
    } else {
        if (transfer.dst_id >= scratchpads.size()) {
            throw std::out_of_range("Invalid destination scratchpad ID: " + std::to_string(transfer.dst_id));
        }
        scratchpads[transfer.dst_id].write(transfer.dst_addr, buffer.data(), transfer.size);
    }

    // Call completion callback if provided
    if (transfer.completion_callback) {
        transfer.completion_callback();
    }

    transfer_queue.erase(transfer_queue.begin());

    bool completed = transfer_queue.empty();
    if (completed) {
        is_active = false;
    }

    return completed;
}

void DMAEngine::reset() {
    transfer_queue.clear();
    is_active = false;
}

// BlockMover implementation - manages L3↔L2 data movement with transformations
BlockMover::BlockMover(size_t engine_id, size_t associated_l3_tile_id)
    : is_active(false), engine_id(engine_id), associated_l3_tile_id(associated_l3_tile_id) {
}

void BlockMover::enqueue_block_transfer(size_t src_l3_tile_id, Address src_offset,
                                       size_t dst_l2_bank_id, Address dst_offset,
                                       Size block_height, Size block_width, Size element_size,
                                       TransformType transform,
                                       std::function<void()> callback) {
    transfer_queue.emplace_back(BlockTransfer{
        src_l3_tile_id, src_offset,
        dst_l2_bank_id, dst_offset,
        block_height, block_width, element_size,
        transform, std::move(callback)
    });
}

bool BlockMover::process_transfers(std::vector<L3Tile>& l3_tiles,
                                  std::vector<L2Bank>& l2_banks) {
    if (transfer_queue.empty()) {
        is_active = false;
        return false;
    }

    is_active = true;
    auto& transfer = transfer_queue.front();

    // Validate indices
    if (transfer.src_l3_tile_id >= l3_tiles.size()) {
        throw std::out_of_range("Invalid L3 tile ID: " + std::to_string(transfer.src_l3_tile_id));
    }
    if (transfer.dst_l2_bank_id >= l2_banks.size()) {
        throw std::out_of_range("Invalid L2 bank ID: " + std::to_string(transfer.dst_l2_bank_id));
    }

    // Calculate total block size
    Size block_size = transfer.block_height * transfer.block_width * transfer.element_size;

    // Allocate buffers for the transfer
    std::vector<std::uint8_t> src_buffer(block_size);
    std::vector<std::uint8_t> dst_buffer(block_size);

    // Read block from L3 tile
    l3_tiles[transfer.src_l3_tile_id].read_block(
        transfer.src_offset, src_buffer.data(),
        transfer.block_height, transfer.block_width, transfer.element_size
    );

    // Apply transformation
    apply_transform(src_buffer, dst_buffer, transfer);

    // Write transformed block to L2 bank
    l2_banks[transfer.dst_l2_bank_id].write_block(
        transfer.dst_offset, dst_buffer.data(),
        transfer.block_height, transfer.block_width, transfer.element_size
    );

    // Call completion callback if provided
    if (transfer.completion_callback) {
        transfer.completion_callback();
    }

    transfer_queue.erase(transfer_queue.begin());

    bool completed = transfer_queue.empty();
    if (completed) {
        is_active = false;
    }

    return completed;
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
    is_active = false;
}

// ComputeFabric implementation
ComputeFabric::ComputeFabric(size_t tile_id)
    : is_computing(false), compute_start_cycle(0), tile_id(tile_id) {
}

void ComputeFabric::start_matmul(const MatMulConfig& config) {
    if (is_computing) {
        throw std::runtime_error("ComputeFabric is already busy");
    }
    
    current_op = config;
    is_computing = true;
    compute_start_cycle = 0; // Will be set by the caller
}

bool ComputeFabric::update(Cycle current_cycle, std::vector<Scratchpad>& scratchpads) {
    if (!is_computing) {
        return false;
    }
    
    if (compute_start_cycle == 0) {
        compute_start_cycle = current_cycle;
    }
    
    Cycle required_cycles = estimate_cycles(current_op.m, current_op.n, current_op.k);
    
    if (current_cycle - compute_start_cycle >= required_cycles) {
        // Operation completed
        execute_matmul(scratchpads);
        
        if (current_op.completion_callback) {
            current_op.completion_callback();
        }
        
        is_computing = false;
        return true;
    }
    
    return false;
}

void ComputeFabric::execute_matmul(std::vector<Scratchpad>& scratchpads) {
    if (current_op.scratchpad_id >= scratchpads.size()) {
        throw std::out_of_range("Invalid scratchpad ID for matmul operation");
    }
    
    auto& scratchpad = scratchpads[current_op.scratchpad_id];
    
    // Read matrices from scratchpad
    Size a_size = current_op.m * current_op.k * sizeof(float);
    Size b_size = current_op.k * current_op.n * sizeof(float);
    Size c_size = current_op.m * current_op.n * sizeof(float);
    
    std::vector<float> a(current_op.m * current_op.k);
    std::vector<float> b(current_op.k * current_op.n);
    std::vector<float> c(current_op.m * current_op.n, 0.0f);
    
    scratchpad.read(current_op.a_addr, a.data(), a_size);
    scratchpad.read(current_op.b_addr, b.data(), b_size);
    
    // Perform matrix multiplication: C = A * B
    for (Size i = 0; i < current_op.m; ++i) {
        for (Size j = 0; j < current_op.n; ++j) {
            float sum = 0.0f;
            for (Size k = 0; k < current_op.k; ++k) {
                sum += a[i * current_op.k + k] * b[k * current_op.n + j];
            }
            c[i * current_op.n + j] = sum;
        }
    }
    
    // Write result back to scratchpad
    scratchpad.write(current_op.c_addr, c.data(), c_size);
}

Cycle ComputeFabric::estimate_cycles(Size m, Size n, Size k) const {
    // Simplified model: assume 1 cycle per MAC operation
    return m * n * k;
}

void ComputeFabric::reset() {
    is_computing = false;
    compute_start_cycle = 0;
}

// KPUSimulator implementation - clean delegation-based API
KPUSimulator::KPUSimulator(const Config& config) : current_cycle(0) {
    // Initialize memory banks
    memory_banks.reserve(config.memory_bank_count);
    for (size_t i = 0; i < config.memory_bank_count; ++i) {
        memory_banks.emplace_back(config.memory_bank_capacity_mb, config.memory_bandwidth_gbps);
    }
    
    // Initialize scratchpads
    scratchpads.reserve(config.scratchpad_count);
    for (size_t i = 0; i < config.scratchpad_count; ++i) {
        scratchpads.emplace_back(config.scratchpad_capacity_kb);
    }
    
    // Initialize compute tiles
    compute_tiles.reserve(config.compute_tile_count);
    for (size_t i = 0; i < config.compute_tile_count; ++i) {
        compute_tiles.emplace_back(i);
    }
    
    // Initialize DMA engines - now bidirectional, configured per-transfer
    dma_engines.reserve(config.dma_engine_count);
    for (size_t i = 0; i < config.dma_engine_count; ++i) {
        dma_engines.emplace_back(i);  // Just pass engine ID for identification
    }

    // Initialize L3 tiles - distributed L3 cache tiles
    l3_tiles.reserve(config.l3_tile_count);
    for (size_t i = 0; i < config.l3_tile_count; ++i) {
        l3_tiles.emplace_back(i, config.l3_tile_capacity_kb);
    }

    // Initialize L2 banks - L2 cache banks
    l2_banks.reserve(config.l2_bank_count);
    for (size_t i = 0; i < config.l2_bank_count; ++i) {
        l2_banks.emplace_back(i, config.l2_bank_capacity_kb);
    }

    // Initialize BlockMovers - L3↔L2 data movement engines
    block_movers.reserve(config.block_mover_count);
    for (size_t i = 0; i < config.block_mover_count; ++i) {
        // Associate each BlockMover with an L3 tile (simple 1:1 mapping for now)
        size_t associated_l3_tile = i % config.l3_tile_count;
        block_movers.emplace_back(i, associated_l3_tile);
    }

    sim_start_time = std::chrono::high_resolution_clock::now();
}

// Memory operations - clean delegation
void KPUSimulator::read_memory_bank(size_t bank_id, Address addr, void* data, Size size) {
    validate_bank_id(bank_id);
    memory_banks[bank_id].read(addr, data, size);
}

void KPUSimulator::write_memory_bank(size_t bank_id, Address addr, const void* data, Size size) {
    validate_bank_id(bank_id);
    memory_banks[bank_id].write(addr, data, size);
}

void KPUSimulator::read_scratchpad(size_t pad_id, Address addr, void* data, Size size) {
    validate_scratchpad_id(pad_id);
    scratchpads[pad_id].read(addr, data, size);
}

void KPUSimulator::write_scratchpad(size_t pad_id, Address addr, const void* data, Size size) {
    validate_scratchpad_id(pad_id);
    scratchpads[pad_id].write(addr, data, size);
}

// L3 and L2 memory operations
void KPUSimulator::read_l3_tile(size_t tile_id, Address addr, void* data, Size size) {
    validate_l3_tile_id(tile_id);
    l3_tiles[tile_id].read(addr, data, size);
}

void KPUSimulator::write_l3_tile(size_t tile_id, Address addr, const void* data, Size size) {
    validate_l3_tile_id(tile_id);
    l3_tiles[tile_id].write(addr, data, size);
}

void KPUSimulator::read_l2_bank(size_t bank_id, Address addr, void* data, Size size) {
    validate_l2_bank_id(bank_id);
    l2_banks[bank_id].read(addr, data, size);
}

void KPUSimulator::write_l2_bank(size_t bank_id, Address addr, const void* data, Size size) {
    validate_l2_bank_id(bank_id);
    l2_banks[bank_id].write(addr, data, size);
}

// DMA operations
void KPUSimulator::start_dma_transfer(size_t dma_id,
                                     DMAEngine::MemoryType src_type, size_t src_id, Address src_addr,
                                     DMAEngine::MemoryType dst_type, size_t dst_id, Address dst_addr,
                                     Size size, std::function<void()> callback) {
    validate_dma_id(dma_id);
    dma_engines[dma_id].enqueue_transfer(src_type, src_id, src_addr,
                                        dst_type, dst_id, dst_addr,
                                        size, std::move(callback));
}

void KPUSimulator::start_dma_external_to_scratchpad(size_t dma_id, size_t bank_id, Address src_addr,
                                                    size_t pad_id, Address dst_addr, Size size,
                                                    std::function<void()> callback) {
    start_dma_transfer(dma_id, DMAEngine::MemoryType::EXTERNAL, bank_id, src_addr,
                      DMAEngine::MemoryType::SCRATCHPAD, pad_id, dst_addr,
                      size, std::move(callback));
}

void KPUSimulator::start_dma_scratchpad_to_external(size_t dma_id, size_t pad_id, Address src_addr,
                                                    size_t bank_id, Address dst_addr, Size size,
                                                    std::function<void()> callback) {
    start_dma_transfer(dma_id, DMAEngine::MemoryType::SCRATCHPAD, pad_id, src_addr,
                      DMAEngine::MemoryType::EXTERNAL, bank_id, dst_addr,
                      size, std::move(callback));
}

bool KPUSimulator::is_dma_busy(size_t dma_id) {
    validate_dma_id(dma_id);
    return dma_engines[dma_id].is_busy();
}

// BlockMover operations
void KPUSimulator::start_block_transfer(size_t block_mover_id, size_t src_l3_tile_id, Address src_offset,
                                       size_t dst_l2_bank_id, Address dst_offset,
                                       Size block_height, Size block_width, Size element_size,
                                       BlockMover::TransformType transform,
                                       std::function<void()> callback) {
    validate_block_mover_id(block_mover_id);
    validate_l3_tile_id(src_l3_tile_id);
    validate_l2_bank_id(dst_l2_bank_id);

    block_movers[block_mover_id].enqueue_block_transfer(
        src_l3_tile_id, src_offset, dst_l2_bank_id, dst_offset,
        block_height, block_width, element_size, transform, std::move(callback)
    );
}

bool KPUSimulator::is_block_mover_busy(size_t block_mover_id) {
    validate_block_mover_id(block_mover_id);
    return block_movers[block_mover_id].is_busy();
}

// Compute operations
void KPUSimulator::start_matmul(size_t tile_id, size_t scratchpad_id, Size m, Size n, Size k,
                               Address a_addr, Address b_addr, Address c_addr,
                               std::function<void()> callback) {
    validate_tile_id(tile_id);
    validate_scratchpad_id(scratchpad_id);
    
    ComputeFabric::MatMulConfig config{
        .m = m, .n = n, .k = k,
        .a_addr = a_addr, .b_addr = b_addr, .c_addr = c_addr,
        .scratchpad_id = scratchpad_id,
        .completion_callback = std::move(callback)
    };
    
    compute_tiles[tile_id].start_matmul(config);
}

bool KPUSimulator::is_compute_busy(size_t tile_id) {
    validate_tile_id(tile_id);
    return compute_tiles[tile_id].is_busy();
}

// Simulation control
void KPUSimulator::reset() {
    for (auto& bank : memory_banks) {
        bank.reset();
    }
    for (auto& pad : scratchpads) {
        pad.reset();
    }
    for (auto& dma : dma_engines) {
        dma.reset();
    }
    for (auto& tile : compute_tiles) {
        tile.reset();
    }
    for (auto& l3_tile : l3_tiles) {
        l3_tile.reset();
    }
    for (auto& l2_bank : l2_banks) {
        l2_bank.reset();
    }
    for (auto& block_mover : block_movers) {
        block_mover.reset();
    }
    current_cycle = 0;
    sim_start_time = std::chrono::high_resolution_clock::now();
}

void KPUSimulator::step() {
    ++current_cycle;

    // Update all components
    for (auto& dma : dma_engines) {
        dma.process_transfers(memory_banks, scratchpads);
    }
    for (auto& block_mover : block_movers) {
        block_mover.process_transfers(l3_tiles, l2_banks);
    }
    for (auto& tile : compute_tiles) {
        tile.update(current_cycle, scratchpads);
    }
}

void KPUSimulator::run_until_idle() {
    bool any_busy;
    do {
        any_busy = false;
        for (const auto& dma : dma_engines) {
            if (dma.is_busy()) {
                any_busy = true;
                break;
            }
        }
        for (const auto& block_mover : block_movers) {
            if (block_mover.is_busy()) {
                any_busy = true;
                break;
            }
        }
        for (const auto& tile : compute_tiles) {
            if (tile.is_busy()) {
                any_busy = true;
                break;
            }
        }
        if (any_busy) {
            step();
        }
    } while (any_busy);
}

// Configuration queries
Size KPUSimulator::get_memory_bank_capacity(size_t bank_id) const {
    validate_bank_id(bank_id);
    return memory_banks[bank_id].get_capacity();
}

Size KPUSimulator::get_scratchpad_capacity(size_t pad_id) const {
    validate_scratchpad_id(pad_id);
    return scratchpads[pad_id].get_capacity();
}

Size KPUSimulator::get_l3_tile_capacity(size_t tile_id) const {
    validate_l3_tile_id(tile_id);
    return l3_tiles[tile_id].get_capacity();
}

Size KPUSimulator::get_l2_bank_capacity(size_t bank_id) const {
    validate_l2_bank_id(bank_id);
    return l2_banks[bank_id].get_capacity();
}

// High-level test operation
bool KPUSimulator::run_matmul_test(const MatMulTest& test, size_t memory_bank_id, 
                                  size_t scratchpad_id, size_t compute_tile_id) {
    reset();
    
    Size a_size = test.m * test.k * sizeof(float);
    Size b_size = test.k * test.n * sizeof(float);
    Size c_size = test.m * test.n * sizeof(float);
    
    // Addresses in external memory
    Address ext_a_addr = 0;
    Address ext_b_addr = a_size;
    Address ext_c_addr = ext_b_addr + b_size;
    
    // Addresses in scratchpad
    Address scratch_a_addr = 0;
    Address scratch_b_addr = a_size;
    Address scratch_c_addr = scratch_b_addr + b_size;
    
    try {
        // Load test data into external memory
        write_memory_bank(memory_bank_id, ext_a_addr, test.matrix_a.data(), a_size);
        write_memory_bank(memory_bank_id, ext_b_addr, test.matrix_b.data(), b_size);
        
        // Set up computation pipeline
        bool dma_a_complete = false, dma_b_complete = false, compute_complete = false;
        
        // DMA A and B matrices to scratchpad using convenience methods
        start_dma_external_to_scratchpad(0, memory_bank_id, ext_a_addr, scratchpad_id, scratch_a_addr, a_size,
            [&dma_a_complete]() { dma_a_complete = true; });
        start_dma_external_to_scratchpad(0, memory_bank_id, ext_b_addr, scratchpad_id, scratch_b_addr, b_size,
            [&dma_b_complete]() { dma_b_complete = true; });
        
        // Wait for data to be loaded
        while (!dma_a_complete || !dma_b_complete) {
            step();
        }
        
        // Start matrix multiplication
        start_matmul(compute_tile_id, scratchpad_id, test.m, test.n, test.k,
                    scratch_a_addr, scratch_b_addr, scratch_c_addr,
                    [&compute_complete]() { compute_complete = true; });
        
        // Wait for computation to complete
        while (!compute_complete) {
            step();
        }
        
        // DMA result back to external memory using convenience method
        bool dma_c_complete = false;
        start_dma_scratchpad_to_external(0, scratchpad_id, scratch_c_addr, memory_bank_id, ext_c_addr, c_size,
            [&dma_c_complete]() { dma_c_complete = true; });
        
        // Wait for result transfer
        while (!dma_c_complete) {
            step();
        }
        
        // Verify result
        std::vector<float> result_c(test.m * test.n);
        read_memory_bank(memory_bank_id, ext_c_addr, result_c.data(), c_size);
        
        return test_utils::verify_matmul_result(test.matrix_a, test.matrix_b, result_c, 
                                               test.m, test.n, test.k);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during matmul test: " << e.what() << std::endl;
        return false;
    }
}

// Statistics and monitoring
double KPUSimulator::get_elapsed_time_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - sim_start_time);
    return duration.count() / 1000.0;
}

void KPUSimulator::print_stats() const {
    std::cout << "=== KPU Simulator Statistics ===" << std::endl;
    std::cout << "Simulation cycles: " << current_cycle << std::endl;
    std::cout << "Wall-clock time: " << get_elapsed_time_ms() << " ms" << std::endl;
    std::cout << "Memory banks: " << memory_banks.size() << std::endl;
    std::cout << "Scratchpads: " << scratchpads.size() << std::endl;
    std::cout << "L3 tiles: " << l3_tiles.size() << std::endl;
    std::cout << "L2 banks: " << l2_banks.size() << std::endl;
    std::cout << "Compute tiles: " << compute_tiles.size() << std::endl;
    std::cout << "DMA engines: " << dma_engines.size() << std::endl;
    std::cout << "Block movers: " << block_movers.size() << std::endl;
}

void KPUSimulator::print_component_status() const {
    std::cout << "=== Component Status ===" << std::endl;
    
    std::cout << "Memory Banks:" << std::endl;
    for (size_t i = 0; i < memory_banks.size(); ++i) {
        std::cout << "  Bank[" << i << "]: " << memory_banks[i].get_capacity() / (1024*1024) 
                  << " MB, Ready: " << (memory_banks[i].is_ready() ? "Yes" : "No") << std::endl;
    }
    
    std::cout << "Scratchpads:" << std::endl;
    for (size_t i = 0; i < scratchpads.size(); ++i) {
        std::cout << "  Pad[" << i << "]: " << scratchpads[i].get_capacity() / 1024 
                  << " KB, Ready: " << (scratchpads[i].is_ready() ? "Yes" : "No") << std::endl;
    }
    
    std::cout << "DMA Engines:" << std::endl;
    for (size_t i = 0; i < dma_engines.size(); ++i) {
        const auto& dma = dma_engines[i];
        std::cout << "  DMA[" << i << "]: ";
        std::cout << "Busy: " << (dma.is_busy() ? "Yes" : "No");
        std::cout << ", Queue: " << dma.get_queue_size() << " transfers" << std::endl;
    }
    
    std::cout << "L3 Tiles:" << std::endl;
    for (size_t i = 0; i < l3_tiles.size(); ++i) {
        std::cout << "  L3Tile[" << i << "]: " << l3_tiles[i].get_capacity() / 1024
                  << " KB, Ready: " << (l3_tiles[i].is_ready() ? "Yes" : "No") << std::endl;
    }

    std::cout << "L2 Banks:" << std::endl;
    for (size_t i = 0; i < l2_banks.size(); ++i) {
        std::cout << "  L2Bank[" << i << "]: " << l2_banks[i].get_capacity() / 1024
                  << " KB, Ready: " << (l2_banks[i].is_ready() ? "Yes" : "No") << std::endl;
    }

    std::cout << "Block Movers:" << std::endl;
    for (size_t i = 0; i < block_movers.size(); ++i) {
        const auto& mover = block_movers[i];
        std::cout << "  BlockMover[" << i << "]: ";
        std::cout << "Busy: " << (mover.is_busy() ? "Yes" : "No");
        std::cout << ", Queue: " << mover.get_queue_size() << " transfers";
        std::cout << ", Associated L3: " << mover.get_associated_l3_tile() << std::endl;
    }

    std::cout << "Compute Tiles:" << std::endl;
    for (size_t i = 0; i < compute_tiles.size(); ++i) {
        std::cout << "  Tile[" << i << "]: Busy: "
                  << (compute_tiles[i].is_busy() ? "Yes" : "No") << std::endl;
    }
}

// Component status queries
bool KPUSimulator::is_memory_bank_ready(size_t bank_id) const {
    validate_bank_id(bank_id);
    return memory_banks[bank_id].is_ready();
}

bool KPUSimulator::is_scratchpad_ready(size_t pad_id) const {
    validate_scratchpad_id(pad_id);
    return scratchpads[pad_id].is_ready();
}

bool KPUSimulator::is_l3_tile_ready(size_t tile_id) const {
    validate_l3_tile_id(tile_id);
    return l3_tiles[tile_id].is_ready();
}

bool KPUSimulator::is_l2_bank_ready(size_t bank_id) const {
    validate_l2_bank_id(bank_id);
    return l2_banks[bank_id].is_ready();
}

// Validation helpers
void KPUSimulator::validate_bank_id(size_t bank_id) const {
    if (bank_id >= memory_banks.size()) {
        throw std::out_of_range("Invalid memory bank ID: " + std::to_string(bank_id));
    }
}

void KPUSimulator::validate_scratchpad_id(size_t pad_id) const {
    if (pad_id >= scratchpads.size()) {
        throw std::out_of_range("Invalid scratchpad ID: " + std::to_string(pad_id));
    }
}

void KPUSimulator::validate_dma_id(size_t dma_id) const {
    if (dma_id >= dma_engines.size()) {
        throw std::out_of_range("Invalid DMA engine ID: " + std::to_string(dma_id));
    }
}

void KPUSimulator::validate_tile_id(size_t tile_id) const {
    if (tile_id >= compute_tiles.size()) {
        throw std::out_of_range("Invalid compute tile ID: " + std::to_string(tile_id));
    }
}

void KPUSimulator::validate_l3_tile_id(size_t tile_id) const {
    if (tile_id >= l3_tiles.size()) {
        throw std::out_of_range("Invalid L3 tile ID: " + std::to_string(tile_id));
    }
}

void KPUSimulator::validate_l2_bank_id(size_t bank_id) const {
    if (bank_id >= l2_banks.size()) {
        throw std::out_of_range("Invalid L2 bank ID: " + std::to_string(bank_id));
    }
}

void KPUSimulator::validate_block_mover_id(size_t mover_id) const {
    if (mover_id >= block_movers.size()) {
        throw std::out_of_range("Invalid block mover ID: " + std::to_string(mover_id));
    }
}

// Test utilities implementation
namespace test_utils {

KPUSimulator::MatMulTest generate_simple_matmul_test(Size m, Size n, Size k) {
    KPUSimulator::MatMulTest test;
    test.m = m;
    test.n = n; 
    test.k = k;
    
    test.matrix_a = generate_random_matrix(m, k, -2.0f, 2.0f);
    test.matrix_b = generate_random_matrix(k, n, -2.0f, 2.0f);
    
    // Compute expected result
    test.expected_c.resize(m * n);
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (Size ki = 0; ki < k; ++ki) {
                sum += test.matrix_a[i * k + ki] * test.matrix_b[ki * n + j];
            }
            test.expected_c[i * n + j] = sum;
        }
    }
    
    return test;
}

std::vector<float> generate_random_matrix(Size rows, Size cols, float min_val, float max_val) {
    std::vector<float> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    std::generate(matrix.begin(), matrix.end(), [&]() { return dis(gen); });
    return matrix;
}

bool verify_matmul_result(const std::vector<float>& a, const std::vector<float>& b, 
                         const std::vector<float>& c, Size m, Size n, Size k, float tolerance) {
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float expected = 0.0f;
            for (Size ki = 0; ki < k; ++ki) {
                expected += a[i * k + ki] * b[ki * n + j];
            }
            
            float actual = c[i * n + j];
            if (std::abs(actual - expected) > tolerance) {
                std::cerr << "Mismatch at (" << i << "," << j << "): expected " 
                         << expected << ", got " << actual << std::endl;
                return false;
            }
        }
    }
    return true;
}

KPUSimulator::Config generate_multi_bank_config(size_t num_banks, size_t num_tiles) {
    KPUSimulator::Config config;
    config.memory_bank_count = num_banks;
    config.memory_bank_capacity_mb = 512; // Smaller banks for multi-bank setup
	config.memory_bandwidth_gbps = 16; // Higher bandwidth per bank
    config.scratchpad_count = num_tiles; // One scratchpad per tile
    config.scratchpad_capacity_kb = 256;
    config.compute_tile_count = num_tiles;
    config.dma_engine_count = num_banks + num_tiles; // Plenty of DMA engines
    return config;
}

bool run_distributed_matmul_test(KPUSimulator& sim, Size matrix_size) {
    // Generate test case
    auto test = generate_simple_matmul_test(matrix_size, matrix_size, matrix_size);
    
    // Use multiple banks and tiles if available
    size_t num_banks = sim.get_memory_bank_count();
    size_t num_tiles = sim.get_compute_tile_count();
    
    if (num_banks < 2 || num_tiles < 1) {
        std::cout << "Warning: Not enough banks/tiles for distributed test, using defaults" << std::endl;
        return sim.run_matmul_test(test);
    }
    
    std::cout << "Running distributed matmul test with " << num_banks 
              << " banks and " << num_tiles << " tiles..." << std::endl;
    
    // For now, just use the first bank and tile (can be extended for true distribution)
    bool result = sim.run_matmul_test(test, 0, 0, 0);
    
    if (result) {
        std::cout << "Distributed test passed!" << std::endl;
        sim.print_component_status();
    }
    
    return result;
}

} // namespace test_utils

} // namespace sw::kpu