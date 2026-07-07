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
    for (const auto& group: config_.tile_groups) {
	for (size_t k = 0; k < group.multiplicity; ++k) {
	    tiles_.emplace_back(global_id, group.tile.capacity);
	    ++global_id;
	}
    }
}

VULayer::~VULayer() = default;
VULayer::VULayer(VULayer&&) noexcept = default;
VULayer& VULayer::operator=(VULayer&&) noexcept = default;


} // namespace sw::kpu
