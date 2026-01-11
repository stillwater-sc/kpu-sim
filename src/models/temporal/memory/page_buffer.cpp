#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <sw/kpu/components/page_buffer.hpp>

namespace sw::kpu {

// PageBuffer implementation - manages its own memory model
PageBuffer::PageBuffer(Size capacity_kb)
    : capacity(capacity_kb * 1024) {
    memory_model.resize(capacity);
    std::fill(memory_model.begin(), memory_model.end(), uint8_t(0));
}

void PageBuffer::read(Address addr, void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("PageBuffer read out of bounds");
    }
    std::memcpy(data, memory_model.data() + addr, size);
}

void PageBuffer::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("PageBuffer write out of bounds");
    }
    std::memcpy(memory_model.data() + addr, data, size);
}

void PageBuffer::reset() {
    std::fill(memory_model.begin(), memory_model.end(), uint8_t(0));
}

} // namespace sw::kpu