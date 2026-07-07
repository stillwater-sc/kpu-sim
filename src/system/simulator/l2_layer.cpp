#include <sw/kpu/models/temporal/memory/l2_layer.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu {

L2Layer::L2Layer(const L2LayerConfig& config)
    : config_(config) {
    // Materialize the L2Bank elements, assigning a global flat bank_id so the
    // layer presents a single index space. Prefer the canonical bank_groups;
    // otherwise fall back to the uniform (num_banks x capacity_kb) convenience.
    const size_t total_banks = config_.total_banks();
    banks_.reserve(total_banks);
    size_t global_id = 0;
    for (const auto& group : config_.bank_groups) {
        for (size_t k = 0; k < group.multiplicity; ++k) {
            banks_.emplace_back(global_id, group.bank.capacity_kb);
            ++global_id;
        }
    }
}

L2Bank& L2Layer::bank(size_t index) {
    if (index >= banks_.size()) {
        throw std::out_of_range("L2Layer::bank: index " + std::to_string(index) +
                                " out of range (" + std::to_string(banks_.size()) + " banks)");
    }
    return banks_[index];
}

const L2Bank& L2Layer::bank(size_t index) const {
    if (index >= banks_.size()) {
        throw std::out_of_range("L2Layer::bank: index " + std::to_string(index) +
                                " out of range (" + std::to_string(banks_.size()) + " banks)");
    }
    return banks_[index];
}

void L2Layer::reset() {
    for (auto& bank : banks_) {
        bank.reset();
    }
}

} // namespace sw::kpu
