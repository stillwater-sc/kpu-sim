#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <sw/kpu/components/scratchpad.hpp>

namespace sw::kpu {

// Scratchpad implementation - manages its own memory model
Scratchpad::Scratchpad(Size capacity_kb)
    : capacity(capacity_kb * 1024) {
    memory_model.resize(capacity);
    std::fill(memory_model.begin(), memory_model.end(), uint8_t(0));
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
    std::fill(memory_model.begin(), memory_model.end(), uint8_t(0));
}

} // namespace sw::kpu