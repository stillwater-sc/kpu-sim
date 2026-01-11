#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <sw/kpu/components/l2_bank.hpp>

namespace sw::kpu {

// L2Bank implementation - L2 cache banks
L2Bank::L2Bank(size_t bank_id, Size capacity_kb)
    : capacity(capacity_kb * 1024), bank_id(bank_id) {
    memory_model.resize(capacity);
    std::fill(memory_model.begin(), memory_model.end(), uint8_t(0));
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
    std::fill(memory_model.begin(), memory_model.end(), uint8_t(0));
}

} // namespace sw::kpu