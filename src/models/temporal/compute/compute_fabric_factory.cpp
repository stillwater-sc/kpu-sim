// ============================================================================
// src/components/compute/compute_fabric_factory.cpp
// Factory for creating compute fabrics at various fidelity levels
// ============================================================================

#include <sw/kpu/models/interfaces/compute_fabric_interface.hpp>
#include <sw/kpu/models/behavioral/compute/compute_fabric.hpp>
#include <sw/kpu/models/transactional/compute/compute_fabric.hpp>
#include <stdexcept>
#include <sstream>

namespace sw::kpu {

std::unique_ptr<IComputeFabric> create_compute_fabric(
    const ComputeFabricConfig& config, uint32_t tile_id)
{
    switch (config.fidelity) {
        case SimulationFidelity::BEHAVIORAL:
            return std::make_unique<BehavioralComputeFabric>(config, tile_id);

        case SimulationFidelity::TRANSACTIONAL:
            return std::make_unique<TransactionalComputeFabric>(config, tile_id);

        case SimulationFidelity::CYCLE_ACCURATE:
            // TODO: Implement CycleAccurateSystolicArray
            // For now, fall back to transactional
            return std::make_unique<TransactionalComputeFabric>(config, tile_id);

        default: {
            std::ostringstream oss;
            oss << "Unknown simulation fidelity: "
                << static_cast<int>(config.fidelity);
            throw std::invalid_argument(oss.str());
        }
    }
}

} // namespace sw::kpu
