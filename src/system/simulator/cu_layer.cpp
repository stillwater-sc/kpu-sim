#include <sw/kpu/models/temporal/compute/cu_layer.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu {

CULayer::CULayer(const CULayerConfig& config)
    : config_(config) {
    // Materialize the CUTile elements, assigning a global flat tile_id so the
    // layer presents a single index space.
    const size_t total_tiles = config_.total_tiles();
    tiles_.reserve(total_tiles);
    size_t global_id = 0;
    for (const auto& group : config_.tile_groups) {
        for (size_t k = 0; k < group.multiplicity; ++k) {
            tiles_.emplace_back(global_id, group.tile.capacity_kb);
            ++global_id;
        }
    }

    // Materialize the Streamers the layer owns. 
    // Target micro-architecture: four streamers per tile, one for each NEWS direction.
    const size_t total_streamers = total_tiles * 4;
    config_.streamer_count = total_streamers;
    streamers_.reserve(config_.streamer_count);
    for (size_t i = 0; i < config_.streamer_count; ++i) {
        const size_t associated_tile = total_tiles > 0 ? (i / 4) : 0;
        streamers_.emplace_back(i, associated_tile,
                                   config_.streamer_clock_ghz,
                                   config_.streamer_buswidth_bits);
    }

    // Construct the vector units only when requested
    //if (config_.enable_interconnect && total_tiles > 0) {
    //    interconnect_ = std::make_unique<L3Interconnect>();
    //}
}

CULayer::~CULayer() = default;
CULayer::CULayer(CULayer&&) noexcept = default;
CULayer& CULayer::operator=(CULayer&&) noexcept = default;

CUTile& CULayer::tile(size_t index) {
    if (index >= tiles_.size()) {
        throw std::out_of_range("CULayer::tile: index " + std::to_string(index) +
                                " out of range (" + std::to_string(tiles_.size()) + " tiles)");
    }
    return tiles_[index];
}

const CUTile& CULayer::tile(size_t index) const {
    if (index >= tiles_.size()) {
        throw std::out_of_range("CULayer::tile: index " + std::to_string(index) +
                                " out of range (" + std::to_string(tiles_.size()) + " tiles)");
    }
    return tiles_[index];
}

Streamer& CULayer::streamer(size_t index) {
    if (index >= streamers_.size()) {
        throw std::out_of_range("CULayer::streamer: index " + std::to_string(index) +
                                " out of range (" + std::to_string(streamers_.size()) + " streamers)");
    }
    return block_movers_[index];
}

const Streamer& CULayer::streamer(size_t index) const {
    if (index >= streamers_.size()) {
        throw std::out_of_range("CULayer::block_mover: index " + std::to_string(index) +
                                " out of range (" + std::to_string(streamers_.size()) + " streamers)");
    }
    return streamers_[index];
}

void CULayer::process_streamers(std::vector<Streamer>& streamers) {
    for (auto& streamer : streamers_) {
        streamer.process_transfers(tiles_, l2_banks);
    }
}

void CULayer::set_streamer_cycle(Cycle cycle) {
    for (auto& streamer : streamers_) {
        streamer.set_cycle(cycle);
    }
}

bool CULayer::any_streamer_busy() const {
    for (const auto& streamer : streamers_) {
        if (streamer.is_busy()) {
            return true;
        }
    }
    return false;
}

void CULayer::reset() {
    for (auto& tile : tiles_) {
        tile.reset();
    }
    for (auto& streamer : streamers_) {
        streamer.reset();
    }
}

} // namespace sw::kpu
