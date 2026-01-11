// ============================================================================
// src/components/memory/l3_tile_factory.cpp
// Factory for creating L3 tiles at various fidelity levels
// ============================================================================

#include <sw/kpu/components/memory/l3_tile_interface.hpp>
#include <sw/kpu/models/behavioral/memory/l3_tile.hpp>
#include <sw/kpu/components/memory/transactional_l3_tile.hpp>
#include <stdexcept>
#include <sstream>

namespace sw::kpu {

std::unique_ptr<IL3Tile> create_l3_tile(const L3TileConfig& config, uint32_t tile_id) {
    switch (config.fidelity) {
        case SimulationFidelity::BEHAVIORAL:
            return std::make_unique<BehavioralL3Tile>(config, tile_id);

        case SimulationFidelity::TRANSACTIONAL:
            return std::make_unique<TransactionalL3Tile>(config, tile_id);

        case SimulationFidelity::CYCLE_ACCURATE:
            // TODO: Implement CycleAccurateL3Tile
            // For now, fall back to transactional
            return std::make_unique<TransactionalL3Tile>(config, tile_id);

        default: {
            std::ostringstream oss;
            oss << "Unknown simulation fidelity: "
                << static_cast<int>(config.fidelity);
            throw std::invalid_argument(oss.str());
        }
    }
}

} // namespace sw::kpu
