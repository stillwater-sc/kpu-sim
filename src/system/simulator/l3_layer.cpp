#include <sw/kpu/models/temporal/memory/l3_layer.hpp>
#include <sw/kpu/models/temporal/memory/l2_bank.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu {

L3Layer::L3Layer(const L3LayerConfig& config)
    : config_(config) {
    // Materialize the L3Tile elements from the canonical tile_groups,
    // assigning a global flat tile_id so the layer presents a single index space.
    const size_t total_tiles = config_.total_tiles();
    tiles_.reserve(total_tiles);
    size_t global_id = 0;
    for (const auto& group : config_.tile_groups) {
        for (size_t k = 0; k < group.multiplicity; ++k) {
            tiles_.emplace_back(global_id, group.tile.capacity_kb);
            ++global_id;
        }
    }

    // Materialize the BlockMovers the layer owns.
    // Target micro-architecture: four movers per tile, one for each NEWS
    // direction; calculated automatically unless overridden in the config.
    if (config_.block_mover_count == 0) {
        config_.block_mover_count = total_tiles * 4;
    }
    // BlockMover takes an aggregate bandwidth; derive it from the configured
    // bus width and clock (GB/s = bits/8 * GHz).
    const double block_mover_bandwidth_gb_s =
        config_.block_mover_buswidth_bits / 8.0 * config_.block_mover_clock_ghz;
    block_movers_.reserve(config_.block_mover_count);
    for (size_t i = 0; i < config_.block_mover_count; ++i) {
        const size_t associated_tile = total_tiles > 0 ? (i / 4) % total_tiles : 0;
        block_movers_.emplace_back(i, associated_tile,
                                   config_.block_mover_clock_ghz,
                                   block_mover_bandwidth_gb_s);
    }

    // Construct the interconnect only when requested (ownership seam).
    if (config_.enable_interconnect && total_tiles > 0) {
        interconnect_ = std::make_unique<L3Interconnect>();
    }
}

L3Layer::~L3Layer() = default;
L3Layer::L3Layer(L3Layer&&) noexcept = default;
L3Layer& L3Layer::operator=(L3Layer&&) noexcept = default;

L3Tile& L3Layer::tile(size_t index) {
    if (index >= tiles_.size()) {
        throw std::out_of_range("L3Layer::tile: index " + std::to_string(index) +
                                " out of range (" + std::to_string(tiles_.size()) + " tiles)");
    }
    return tiles_[index];
}

const L3Tile& L3Layer::tile(size_t index) const {
    if (index >= tiles_.size()) {
        throw std::out_of_range("L3Layer::tile: index " + std::to_string(index) +
                                " out of range (" + std::to_string(tiles_.size()) + " tiles)");
    }
    return tiles_[index];
}

BlockMover& L3Layer::block_mover(size_t index) {
    if (index >= block_movers_.size()) {
        throw std::out_of_range("L3Layer::block_mover: index " + std::to_string(index) +
                                " out of range (" + std::to_string(block_movers_.size()) + " movers)");
    }
    return block_movers_[index];
}

const BlockMover& L3Layer::block_mover(size_t index) const {
    if (index >= block_movers_.size()) {
        throw std::out_of_range("L3Layer::block_mover: index " + std::to_string(index) +
                                " out of range (" + std::to_string(block_movers_.size()) + " movers)");
    }
    return block_movers_[index];
}

void L3Layer::process_block_movers(std::vector<L2Bank>& l2_banks) {
    for (auto& mover : block_movers_) {
        mover.process_transfers(tiles_, l2_banks);
    }
}

void L3Layer::set_block_mover_cycle(Cycle cycle) {
    for (auto& mover : block_movers_) {
        mover.set_cycle(cycle);
    }
}

bool L3Layer::any_block_mover_busy() const {
    for (const auto& mover : block_movers_) {
        if (mover.is_busy()) {
            return true;
        }
    }
    return false;
}

L3Interconnect& L3Layer::interconnect() {
    if (!interconnect_) {
        throw std::logic_error("L3Layer::interconnect: layer has no interconnect");
    }
    return *interconnect_;
}

const L3Interconnect& L3Layer::interconnect() const {
    if (!interconnect_) {
        throw std::logic_error("L3Layer::interconnect: layer has no interconnect");
    }
    return *interconnect_;
}

void L3Layer::reset() {
    for (auto& tile : tiles_) {
        tile.reset();
    }
    for (auto& mover : block_movers_) {
        mover.reset();
    }
}

} // namespace sw::kpu
