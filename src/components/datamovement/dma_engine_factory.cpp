// ============================================================================
// src/components/dma/dma_engine_factory.cpp
// Factory for creating DMA engines at various fidelity levels
// ============================================================================

#include <sw/kpu/components/dma/dma_engine_interface.hpp>
#include <sw/kpu/components/dma/behavioral_dma_engine.hpp>
#include <sw/kpu/components/dma/transactional_dma_engine.hpp>
#include <stdexcept>
#include <sstream>

namespace sw::kpu {

std::unique_ptr<IDMAEngine> create_dma_engine(const DMAEngineConfig& config) {
    switch (config.fidelity) {
        case SimulationFidelity::BEHAVIORAL:
            return std::make_unique<BehavioralDMAEngine>(config);

        case SimulationFidelity::TRANSACTIONAL:
            return std::make_unique<TransactionalDMAEngine>(config);

        case SimulationFidelity::CYCLE_ACCURATE:
            // TODO: Implement CycleAccurateDMAEngine
            // For now, fall back to transactional
            return std::make_unique<TransactionalDMAEngine>(config);

        default: {
            std::ostringstream oss;
            oss << "Unknown simulation fidelity: "
                << static_cast<int>(config.fidelity);
            throw std::invalid_argument(oss.str());
        }
    }
}

} // namespace sw::kpu
