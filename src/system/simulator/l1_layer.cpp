#include <sw/kpu/models/temporal/memory/l1_layer.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu {

L1Layer::L1Layer(const L1LayerConfig& config)
    : config_(config) {
    // Materialize the L1Buffer elements, assigning a global flat buffer_id so the
    // layer presents a single index space. Prefer the canonical buffer_groups;
    // otherwise fall back to the uniform (num_buffers x capacity_kb) convenience.
    const size_t total = config_.total_buffers();
    buffers_.reserve(total);
    size_t global_id = 0;
    if (!config_.buffer_groups.empty()) {
        for (const auto& group : config_.buffer_groups) {
            for (size_t k = 0; k < group.multiplicity; ++k) {
                buffers_.emplace_back(global_id, group.buffer.capacity_kb);
                ++global_id;
            }
        }
    } else {
        for (size_t k = 0; k < config_.num_buffers; ++k) {
            buffers_.emplace_back(global_id, config_.capacity_kb);
            ++global_id;
        }
    }
}

L1Buffer& L1Layer::buffer(size_t index) {
    if (index >= buffers_.size()) {
        throw std::out_of_range("L1Layer::buffer: index " + std::to_string(index) +
                                " out of range (" + std::to_string(buffers_.size()) + " buffers)");
    }
    return buffers_[index];
}

const L1Buffer& L1Layer::buffer(size_t index) const {
    if (index >= buffers_.size()) {
        throw std::out_of_range("L1Layer::buffer: index " + std::to_string(index) +
                                " out of range (" + std::to_string(buffers_.size()) + " buffers)");
    }
    return buffers_[index];
}

Size L1Layer::total_capacity_bytes() const {
    Size total = 0;
    for (const auto& buffer : buffers_) {
        total += buffer.get_capacity();
    }
    return total;
}

void L1Layer::reset() {
    for (auto& buffer : buffers_) {
        buffer.reset();
    }
}

} // namespace sw::kpu
