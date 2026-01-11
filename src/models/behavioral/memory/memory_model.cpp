// ============================================================================
// src/components/behavioral/memory_model.cpp
// Behavioral memory model implementation
// ============================================================================

#include <sw/kpu/behavioral/memory_model.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>

namespace sw::kpu::behavioral {

// ============================================================================
// Constructor
// ============================================================================

BehavioralMemoryModel::BehavioralMemoryModel(const MemoryModelConfig& config)
    : config_(config)
{
    // Create host memory region
    host_region_ = std::make_unique<MemoryRegion>(
        MemoryRegionType::HOST, 0, config.host_capacity_bytes
    );

    // Create L3 tiles
    l3_tiles_.reserve(config.l3_tile_count);
    l3_allocators_.resize(config.l3_tile_count);
    for (uint32_t i = 0; i < config.l3_tile_count; ++i) {
        l3_tiles_.push_back(std::make_unique<MemoryRegion>(
            MemoryRegionType::L3_TILE, i, config.l3_tile_capacity_bytes
        ));
    }

    // Create L2 banks
    l2_banks_.reserve(config.l2_bank_count);
    l2_allocators_.resize(config.l2_bank_count);
    for (uint32_t i = 0; i < config.l2_bank_count; ++i) {
        l2_banks_.push_back(std::make_unique<MemoryRegion>(
            MemoryRegionType::L2_BANK, i, config.l2_bank_capacity_bytes
        ));
    }

    // Create L1 buffers
    l1_buffers_.reserve(config.l1_buffer_count);
    l1_allocators_.resize(config.l1_buffer_count);
    for (uint32_t i = 0; i < config.l1_buffer_count; ++i) {
        l1_buffers_.push_back(std::make_unique<MemoryRegion>(
            MemoryRegionType::L1_BUFFER, i, config.l1_buffer_capacity_bytes
        ));
    }
}

// ============================================================================
// Region Access
// ============================================================================

MemoryRegion* BehavioralMemoryModel::get_region(MemoryRegionType type, uint32_t id) {
    switch (type) {
        case MemoryRegionType::HOST:
            return (id == 0) ? host_region_.get() : nullptr;
        case MemoryRegionType::L3_TILE:
            return (id < l3_tiles_.size()) ? l3_tiles_[id].get() : nullptr;
        case MemoryRegionType::L2_BANK:
            return (id < l2_banks_.size()) ? l2_banks_[id].get() : nullptr;
        case MemoryRegionType::L1_BUFFER:
            return (id < l1_buffers_.size()) ? l1_buffers_[id].get() : nullptr;
        default:
            return nullptr;
    }
}

const MemoryRegion* BehavioralMemoryModel::get_region(MemoryRegionType type, uint32_t id) const {
    return const_cast<BehavioralMemoryModel*>(this)->get_region(type, id);
}

MemoryRegion* BehavioralMemoryModel::get_region(Address addr) {
    auto type = AddressEncoder::decode_type(addr);
    auto id = AddressEncoder::decode_region_id(addr);
    return get_region(type, id);
}

const MemoryRegion* BehavioralMemoryModel::get_region(Address addr) const {
    return const_cast<BehavioralMemoryModel*>(this)->get_region(addr);
}

// ============================================================================
// Allocator Access
// ============================================================================

BehavioralMemoryModel::RegionAllocator* BehavioralMemoryModel::get_allocator(
    MemoryRegionType type, uint32_t id)
{
    switch (type) {
        case MemoryRegionType::HOST:
            return (id == 0) ? &host_allocator_ : nullptr;
        case MemoryRegionType::L3_TILE:
            return (id < l3_allocators_.size()) ? &l3_allocators_[id] : nullptr;
        case MemoryRegionType::L2_BANK:
            return (id < l2_allocators_.size()) ? &l2_allocators_[id] : nullptr;
        case MemoryRegionType::L1_BUFFER:
            return (id < l1_allocators_.size()) ? &l1_allocators_[id] : nullptr;
        default:
            return nullptr;
    }
}

// ============================================================================
// Allocation
// ============================================================================

Allocation BehavioralMemoryModel::allocate_host(Size bytes, const std::string& name) {
    auto* region = host_region_.get();
    auto* allocator = &host_allocator_;

    if (allocator->next_offset + bytes > region->capacity()) {
        throw std::runtime_error(
            "Host allocation failed: requested " + std::to_string(bytes) +
            " bytes, available " + std::to_string(region->capacity() - allocator->next_offset)
        );
    }

    Allocation alloc;
    alloc.region_type = MemoryRegionType::HOST;
    alloc.region_id = 0;
    alloc.base_address = AddressEncoder::encode(MemoryRegionType::HOST, 0, allocator->next_offset);
    alloc.size_bytes = bytes;
    alloc.name = name;

    allocator->next_offset += bytes;
    allocator->allocations.push_back(alloc);

    // Track named allocation
    if (!name.empty()) {
        named_allocations_[name] = alloc;
    }

    stats_.total_host_allocated += bytes;
    return alloc;
}

Allocation BehavioralMemoryModel::allocate_l3(uint32_t tile_id, Size bytes, const std::string& name) {
    if (tile_id >= l3_tiles_.size()) {
        throw std::out_of_range("L3 tile ID " + std::to_string(tile_id) + " out of range");
    }

    auto* region = l3_tiles_[tile_id].get();
    auto* allocator = &l3_allocators_[tile_id];

    if (allocator->next_offset + bytes > region->capacity()) {
        throw std::runtime_error(
            "L3[" + std::to_string(tile_id) + "] allocation failed: requested " +
            std::to_string(bytes) + " bytes, available " +
            std::to_string(region->capacity() - allocator->next_offset)
        );
    }

    Allocation alloc;
    alloc.region_type = MemoryRegionType::L3_TILE;
    alloc.region_id = tile_id;
    alloc.base_address = AddressEncoder::encode(MemoryRegionType::L3_TILE, tile_id, allocator->next_offset);
    alloc.size_bytes = bytes;
    alloc.name = name;

    allocator->next_offset += bytes;
    allocator->allocations.push_back(alloc);

    if (!name.empty()) {
        named_allocations_[name] = alloc;
    }

    stats_.total_l3_allocated += bytes;
    return alloc;
}

Allocation BehavioralMemoryModel::allocate_l2(uint32_t bank_id, Size bytes, const std::string& name) {
    if (bank_id >= l2_banks_.size()) {
        throw std::out_of_range("L2 bank ID " + std::to_string(bank_id) + " out of range");
    }

    auto* region = l2_banks_[bank_id].get();
    auto* allocator = &l2_allocators_[bank_id];

    if (allocator->next_offset + bytes > region->capacity()) {
        throw std::runtime_error(
            "L2[" + std::to_string(bank_id) + "] allocation failed: requested " +
            std::to_string(bytes) + " bytes, available " +
            std::to_string(region->capacity() - allocator->next_offset)
        );
    }

    Allocation alloc;
    alloc.region_type = MemoryRegionType::L2_BANK;
    alloc.region_id = bank_id;
    alloc.base_address = AddressEncoder::encode(MemoryRegionType::L2_BANK, bank_id, allocator->next_offset);
    alloc.size_bytes = bytes;
    alloc.name = name;

    allocator->next_offset += bytes;
    allocator->allocations.push_back(alloc);

    if (!name.empty()) {
        named_allocations_[name] = alloc;
    }

    stats_.total_l2_allocated += bytes;
    return alloc;
}

Allocation BehavioralMemoryModel::allocate_l1(uint32_t buffer_id, Size bytes, const std::string& name) {
    if (buffer_id >= l1_buffers_.size()) {
        throw std::out_of_range("L1 buffer ID " + std::to_string(buffer_id) + " out of range");
    }

    auto* region = l1_buffers_[buffer_id].get();
    auto* allocator = &l1_allocators_[buffer_id];

    if (allocator->next_offset + bytes > region->capacity()) {
        throw std::runtime_error(
            "L1[" + std::to_string(buffer_id) + "] allocation failed: requested " +
            std::to_string(bytes) + " bytes, available " +
            std::to_string(region->capacity() - allocator->next_offset)
        );
    }

    Allocation alloc;
    alloc.region_type = MemoryRegionType::L1_BUFFER;
    alloc.region_id = buffer_id;
    alloc.base_address = AddressEncoder::encode(MemoryRegionType::L1_BUFFER, buffer_id, allocator->next_offset);
    alloc.size_bytes = bytes;
    alloc.name = name;

    allocator->next_offset += bytes;
    allocator->allocations.push_back(alloc);

    if (!name.empty()) {
        named_allocations_[name] = alloc;
    }

    stats_.total_l1_allocated += bytes;
    return alloc;
}

void BehavioralMemoryModel::free(const Allocation& alloc) {
    // For simplicity, we use a bump allocator - free is a no-op
    // A more sophisticated implementation would track free blocks
    // and reuse them.

    // Remove from named allocations if present
    if (!alloc.name.empty()) {
        named_allocations_.erase(alloc.name);
    }
}

std::optional<Allocation> BehavioralMemoryModel::find_allocation(const std::string& name) const {
    auto it = named_allocations_.find(name);
    if (it != named_allocations_.end()) {
        return it->second;
    }
    return std::nullopt;
}

// ============================================================================
// Data Operations
// ============================================================================

void BehavioralMemoryModel::write(Address addr, const void* data, Size bytes) {
    auto* region = get_region(addr);
    if (!region) {
        throw std::runtime_error("Write to invalid address: " + std::to_string(addr));
    }

    Size offset = AddressEncoder::decode_offset(addr);
    region->write(offset, data, bytes);

    stats_.write_count++;
    stats_.bytes_written += bytes;
}

void BehavioralMemoryModel::read(Address addr, void* data, Size bytes) const {
    const auto* region = get_region(addr);
    if (!region) {
        throw std::runtime_error("Read from invalid address: " + std::to_string(addr));
    }

    Size offset = AddressEncoder::decode_offset(addr);
    region->read(offset, data, bytes);

    stats_.read_count++;
    stats_.bytes_read += bytes;
}

void BehavioralMemoryModel::copy(Address dst, Address src, Size bytes) {
    // Get source and destination regions
    const auto* src_region = get_region(src);
    auto* dst_region = get_region(dst);

    if (!src_region) {
        throw std::runtime_error("Copy from invalid source address: " + std::to_string(src));
    }
    if (!dst_region) {
        throw std::runtime_error("Copy to invalid destination address: " + std::to_string(dst));
    }

    Size src_offset = AddressEncoder::decode_offset(src);
    Size dst_offset = AddressEncoder::decode_offset(dst);

    // Read from source, write to destination
    // Use intermediate buffer for safety (handles overlapping regions)
    std::vector<uint8_t> buffer(bytes);
    src_region->read(src_offset, buffer.data(), bytes);
    dst_region->write(dst_offset, buffer.data(), bytes);

    stats_.copy_count++;
    stats_.bytes_copied += bytes;
}

void BehavioralMemoryModel::zero(Address addr, Size bytes) {
    auto* region = get_region(addr);
    if (!region) {
        throw std::runtime_error("Zero at invalid address: " + std::to_string(addr));
    }

    Size offset = AddressEncoder::decode_offset(addr);
    std::vector<uint8_t> zeros(bytes, 0);
    region->write(offset, zeros.data(), bytes);

    stats_.write_count++;
    stats_.bytes_written += bytes;
}

// ============================================================================
// Data Location Tracking
// ============================================================================

void BehavioralMemoryModel::register_tensor(const std::string& name, const Allocation& alloc) {
    auto it = tensor_locations_.find(name);
    if (it == tensor_locations_.end()) {
        DataLocation loc;
        loc.tensor_name = name;
        loc.allocations.push_back(alloc);
        tensor_locations_[name] = loc;
    } else {
        // Add this allocation to existing locations
        it->second.allocations.push_back(alloc);
    }
}

std::optional<DataLocation> BehavioralMemoryModel::locate_tensor(const std::string& name) const {
    auto it = tensor_locations_.find(name);
    if (it != tensor_locations_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void BehavioralMemoryModel::update_tensor_location(const std::string& name, const Allocation& new_alloc) {
    auto it = tensor_locations_.find(name);
    if (it != tensor_locations_.end()) {
        it->second.allocations.push_back(new_alloc);
    } else {
        register_tensor(name, new_alloc);
    }
}

void BehavioralMemoryModel::unregister_tensor(const std::string& name) {
    tensor_locations_.erase(name);
}

// ============================================================================
// Validation
// ============================================================================

bool BehavioralMemoryModel::is_valid(Address addr, Size bytes) const {
    const auto* region = get_region(addr);
    if (!region) return false;

    Size offset = AddressEncoder::decode_offset(addr);
    return (offset + bytes <= region->capacity());
}

bool BehavioralMemoryModel::verify(Address addr, const void* expected, Size bytes) const {
    if (!is_valid(addr, bytes)) return false;

    std::vector<uint8_t> actual(bytes);
    read(addr, actual.data(), bytes);

    return std::memcmp(actual.data(), expected, bytes) == 0;
}

// ============================================================================
// Debug
// ============================================================================

void BehavioralMemoryModel::dump_memory_map() const {
    std::cout << "=== Memory Map ===" << std::endl;

    // Host
    std::cout << "HOST: " << host_region_->capacity() << " bytes, "
              << host_allocator_.next_offset << " used" << std::endl;

    // L3 tiles
    for (size_t i = 0; i < l3_tiles_.size(); ++i) {
        std::cout << "L3[" << i << "]: " << l3_tiles_[i]->capacity() << " bytes, "
                  << l3_allocators_[i].next_offset << " used" << std::endl;
    }

    // L2 banks
    for (size_t i = 0; i < l2_banks_.size(); ++i) {
        std::cout << "L2[" << i << "]: " << l2_banks_[i]->capacity() << " bytes, "
                  << l2_allocators_[i].next_offset << " used" << std::endl;
    }

    // L1 buffers
    for (size_t i = 0; i < l1_buffers_.size(); ++i) {
        std::cout << "L1[" << i << "]: " << l1_buffers_[i]->capacity() << " bytes, "
                  << l1_allocators_[i].next_offset << " used" << std::endl;
    }

    // Named allocations
    if (!named_allocations_.empty()) {
        std::cout << "\nNamed Allocations:" << std::endl;
        for (const auto& [name, alloc] : named_allocations_) {
            std::cout << "  " << name << ": "
                      << to_string(alloc.region_type) << "[" << alloc.region_id << "] "
                      << alloc.size_bytes << " bytes @ 0x"
                      << std::hex << alloc.base_address << std::dec << std::endl;
        }
    }

    // Statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Host allocated: " << stats_.total_host_allocated << " bytes" << std::endl;
    std::cout << "  L3 allocated: " << stats_.total_l3_allocated << " bytes" << std::endl;
    std::cout << "  L2 allocated: " << stats_.total_l2_allocated << " bytes" << std::endl;
    std::cout << "  L1 allocated: " << stats_.total_l1_allocated << " bytes" << std::endl;
    std::cout << "  Writes: " << stats_.write_count << " (" << stats_.bytes_written << " bytes)" << std::endl;
    std::cout << "  Reads: " << stats_.read_count << " (" << stats_.bytes_read << " bytes)" << std::endl;
    std::cout << "  Copies: " << stats_.copy_count << " (" << stats_.bytes_copied << " bytes)" << std::endl;
}

void BehavioralMemoryModel::dump_hex(Address addr, Size bytes) const {
    const auto* region = get_region(addr);
    if (!region) {
        std::cout << "Invalid address: 0x" << std::hex << addr << std::dec << std::endl;
        return;
    }

    Size offset = AddressEncoder::decode_offset(addr);
    const uint8_t* data = region->data_ptr(offset);

    std::cout << "Hex dump at 0x" << std::hex << addr << " (" << std::dec << bytes << " bytes):" << std::endl;

    for (Size i = 0; i < bytes; i += 16) {
        std::cout << std::hex << std::setfill('0') << std::setw(8) << (addr + i) << ": ";

        // Hex values
        for (Size j = 0; j < 16 && (i + j) < bytes; ++j) {
            std::cout << std::setw(2) << static_cast<int>(data[i + j]) << " ";
        }

        // Padding if less than 16 bytes
        for (Size j = bytes - i; j < 16; ++j) {
            std::cout << "   ";
        }

        std::cout << " |";

        // ASCII representation
        for (Size j = 0; j < 16 && (i + j) < bytes; ++j) {
            char c = static_cast<char>(data[i + j]);
            std::cout << (std::isprint(c) ? c : '.');
        }

        std::cout << "|" << std::endl;
    }
    std::cout << std::dec;
}

} // namespace sw::kpu::behavioral
