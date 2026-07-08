#include <sw/kpu/models/temporal/compute/vu_layer.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu {

VULayer::VULayer(const VULayerConfig& config)
    : config_(config) {
    // Materialize the Vector Units
    const size_t total_tiles = config_.total_tiles();
    tiles_.reserve(total_tiles);
    size_t global_id = 0;
    for (const auto& group : config_.tile_groups) {
        for (size_t k = 0; k < group.multiplicity; ++k) {
            tiles_.emplace_back(global_id, group.tile);
            ++global_id;
        }
    }
}

VULayer::~VULayer() = default;
VULayer::VULayer(VULayer&&) noexcept = default;
VULayer& VULayer::operator=(VULayer&&) noexcept = default;

VUTile& VULayer::tile(size_t index) {
    if (index >= tiles_.size()) {
        throw std::out_of_range("VULayer::tile: index " + std::to_string(index) +
                                " out of range (" + std::to_string(tiles_.size()) + " tiles)");
    }
    return tiles_[index];
}

const VUTile& VULayer::tile(size_t index) const {
    if (index >= tiles_.size()) {
        throw std::out_of_range("VULayer::tile: index " + std::to_string(index) +
                                " out of range (" + std::to_string(tiles_.size()) + " tiles)");
    }
    return tiles_[index];
}

void VULayer::reset() {
    for (auto& tile : tiles_) {
        tile.reset();
    }
}

} // namespace sw::kpu
