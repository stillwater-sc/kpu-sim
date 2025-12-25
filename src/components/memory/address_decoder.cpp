#include <sw/memory/address_decoder.hpp>

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace sw::memory {

void AddressDecoder::add_region(Address base, Size size, MemoryType type, size_t id,
                               const std::string& name) {
    // Check for overlaps with existing regions
    Address end = base + size;
    for (const auto& region : regions_) {
        Address region_end = region.base + region.size;

        // Check if regions overlap
        bool overlaps = (base < region_end) && (end > region.base);
        if (overlaps) {
            std::ostringstream oss;
            oss << "Memory region [0x" << std::hex << base << "-0x" << end << ") "
                << "overlaps with existing region [0x" << region.base << "-0x" << region_end << ")";
            if (!region.name.empty()) {
                oss << " (" << region.name << ")";
            }
            throw std::invalid_argument(oss.str());
        }
    }

    // Add the region
    regions_.emplace_back(base, size, type, id, name);

    // Sort regions by base address for efficient lookup
    std::sort(regions_.begin(), regions_.end(),
             [](const Region& a, const Region& b) { return a.base < b.base; });
}

AddressDecoder::RoutingInfo AddressDecoder::decode(Address addr) const {
    // Binary search for the region containing this address
    auto it = std::upper_bound(regions_.begin(), regions_.end(), addr,
                              [](Address a, const Region& r) { return a < r.base; });

    // upper_bound returns first element with base > addr
    // So we need to check the previous element
    if (it != regions_.begin()) {
        --it;
        if (it->contains(addr)) {
            Address offset = addr - it->base;
            return RoutingInfo(it->type, it->id, offset, it->size);
        }
    }

    // Address not found in any region
    std::ostringstream oss;
    oss << "Address 0x" << std::hex << addr << " is not mapped to any memory region";
    throw std::out_of_range(oss.str());
}

bool AddressDecoder::is_valid(Address addr) const {
    try {
        decode(addr);
        return true;
    } catch (const std::out_of_range&) {
        return false;
    }
}

bool AddressDecoder::is_valid_range(Address addr, Size size) const {
    if (size == 0) return true;

    try {
        // Check if start address is valid
        auto start_route = decode(addr);

        // Check if end address is valid
        Address end_addr = addr + size - 1;  // Last byte of transfer
        auto end_route = decode(end_addr);

        // Ensure both addresses are in the same region
        return (start_route.type == end_route.type) &&
               (start_route.id == end_route.id);
    } catch (const std::out_of_range&) {
        return false;
    }
}

std::optional<AddressDecoder::Region> AddressDecoder::find_region(Address addr) const {
    try {
        auto route = decode(addr);
        // Find the actual region (we know it exists since decode succeeded)
        for (const auto& region : regions_) {
            if (region.type == route.type && region.id == route.id) {
                return region;
            }
        }
    } catch (const std::out_of_range&) {
        // Address not mapped
    }
    return std::nullopt;
}

Size AddressDecoder::get_total_mapped_size() const {
    Size total = 0;
    for (const auto& region : regions_) {
        total += region.size;
    }
    return total;
}

std::string AddressDecoder::to_string() const {
    std::ostringstream oss;

    // Find maximum address to determine hex width needed
    Address max_addr = 0;
    for (const auto& region : regions_) {
        Address end = region.base + region.size - 1;
        if (end > max_addr) max_addr = end;
    }

    // Calculate hex digits needed (minimum 8 for 32-bit compatibility)
    int hex_width = 8;
    Address temp = max_addr;
    while (temp > 0xFFFFFFFF) {
        hex_width++;
        temp >>= 4;
    }

    oss << "Memory Map (" << regions_.size() << " regions):\n";
    oss << "  Address Range";
    // Adjust header spacing based on hex width
    int addr_range_width = 2 + hex_width + 3 + 2 + hex_width; // "0x" + digits + " - " + "0x" + digits
    for (int i = 13; i < addr_range_width; ++i) oss << " ";
    oss << " | Size      | Type        | ID | Name\n";

    oss << "  ";
    for (int i = 0; i < addr_range_width; ++i) oss << "-";
    oss << " | --------- | ----------- | -- | ----\n";

    for (const auto& region : regions_) {
        // Address range with proper width
        oss << "  0x" << std::hex << std::setfill('0') << std::setw(hex_width) << region.base
            << " - 0x" << std::setw(hex_width) << (region.base + region.size - 1);

        // Size column - fixed 9 characters
        oss << " | ";
        std::ostringstream size_stream;
        size_stream << std::dec;
        if (region.size >= (1024ULL * 1024 * 1024)) {
            size_stream << (region.size / (1024ULL * 1024 * 1024)) << " GB";
        } else if (region.size >= (1024 * 1024)) {
            size_stream << (region.size / (1024 * 1024)) << " MB";
        } else if (region.size >= 1024) {
            size_stream << (region.size / 1024) << " KB";
        } else {
            size_stream << region.size << " B";
        }
        oss << std::left << std::setw(9) << std::setfill(' ') << size_stream.str();

        // Type name - fixed 11 characters
        oss << " | ";
        std::string type_str;
        switch (region.type) {
            case MemoryType::HOST_MEMORY: type_str = "HOST"; break;
            case MemoryType::EXTERNAL:    type_str = "EXTERNAL"; break;
            case MemoryType::L3_TILE:     type_str = "L3_TILE"; break;
            case MemoryType::L2_BANK:     type_str = "L2_BANK"; break;
            case MemoryType::L1:          type_str = "L1"; break;
            case MemoryType::PAGE_BUFFER: type_str = "PAGE_BUFFER"; break;
        }
        oss << std::left << std::setw(11) << type_str;

        // ID - fixed 2 characters, right-aligned
        oss << " | " << std::dec << std::right << std::setfill('0') << std::setw(2) << region.id;

        // Name
        if (!region.name.empty()) {
            oss << " | " << region.name;
        }

        oss << "\n";
    }

    oss << "\nTotal mapped: ";
    Size total = get_total_mapped_size();
    if (total >= (1024 * 1024 * 1024)) {
        oss << (total / (1024 * 1024 * 1024)) << " GB";
    } else if (total >= (1024 * 1024)) {
        oss << (total / (1024 * 1024)) << " MB";
    } else if (total >= 1024) {
        oss << (total / 1024) << " KB";
    } else {
        oss << total << " B";
    }
    oss << "\n";  // Add newline at the end

    return oss.str();
}

} // namespace sw::memory
