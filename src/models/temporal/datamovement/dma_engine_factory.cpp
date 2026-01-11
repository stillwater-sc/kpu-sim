// ============================================================================
// src/components/dma/dma_engine_factory.cpp
// Factory for creating DMA engines at various fidelity levels
// ============================================================================

#include <sw/kpu/components/dma/dma_engine_interface.hpp>
#include <sw/kpu/models/behavioral/datamovement/dma_engine.hpp>
#include <sw/kpu/models/transactional/datamovement/dma_engine.hpp>
#include <sw/kpu/components/dma/cycle_accurate_dma_engine.hpp>
#include <stdexcept>
#include <sstream>

namespace sw::kpu {

std::unique_ptr<IDMAEngine> create_dma_engine(const DMAEngineConfig& config) {
    switch (config.fidelity) {
        case SimulationFidelity::BEHAVIORAL:
            return std::make_unique<BehavioralDMAEngine>(config);

        case SimulationFidelity::TRANSACTIONAL:
            return std::make_unique<TransactionalDMAEngine>(config);

        case SimulationFidelity::CYCLE_ACCURATE: {
            // Create CycleAccurateDMAConfig from base config
            CycleAccurateDMAConfig ca_config;
            // Copy base fields
            ca_config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
            ca_config.num_engines = config.num_engines;
            ca_config.channels_per_engine = config.channels_per_engine;
            ca_config.num_channels = config.num_channels;
            ca_config.max_burst_size = config.max_burst_size;
            ca_config.bandwidth_gbps = config.bandwidth_gbps;
            ca_config.queue_depth_per_channel = config.queue_depth_per_channel;
            ca_config.fixed_latency_per_burst = config.fixed_latency_per_burst;
            // Use defaults for cycle-accurate specific fields
            return std::make_unique<CycleAccurateDMAEngine>(ca_config);
        }

        default: {
            std::ostringstream oss;
            oss << "Unknown simulation fidelity: "
                << static_cast<int>(config.fidelity);
            throw std::invalid_argument(oss.str());
        }
    }
}

} // namespace sw::kpu
