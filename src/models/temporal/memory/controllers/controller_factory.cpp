// ============================================================================
// src/components/memory/memory_controller_factory.cpp
// Factory for creating memory controllers at various fidelity levels
// ============================================================================

#include <sw/kpu/components/memory/memory_controller_interface.hpp>
#include <sw/kpu/models/behavioral/memory/memory_controller.hpp>
#include <sw/kpu/models/transactional/memory/memory_controller.hpp>
#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>
#include <sw/kpu/models/temporal/memory/controllers/ddr5_controller.hpp>
#include <sw/kpu/models/temporal/memory/controllers/gddr6_controller.hpp>
#include <sw/kpu/models/temporal/memory/controllers/gddr7_controller.hpp>
#include <sw/kpu/models/temporal/memory/controllers/hbm2_controller.hpp>
#include <sw/kpu/models/temporal/memory/controllers/hbm3_controller.hpp>
#include <stdexcept>
#include <sstream>

namespace sw::kpu {

std::unique_ptr<IMemoryController> create_memory_controller(
    const MemoryControllerConfig& config)
{
    switch (config.fidelity) {
        case SimulationFidelity::BEHAVIORAL:
            // Behavioral model - technology doesn't matter, just use simple fixed latency
            return std::make_unique<BehavioralMemoryController>(config);

        case SimulationFidelity::TRANSACTIONAL:
            // Transactional model - queue-based with statistical timing
            return std::make_unique<TransactionalMemoryController>(config);

        case SimulationFidelity::CYCLE_ACCURATE:
            // Cycle-accurate - select based on technology
            switch (config.technology) {
                case MemoryTechnology::IDEAL:
                    // IDEAL with cycle-accurate: use LPDDR5 as a reasonable default
                    return std::make_unique<lpddr5::LPDDR5MemoryController>(config);

                case MemoryTechnology::LPDDR5:
                case MemoryTechnology::LPDDR5X:
                    return std::make_unique<lpddr5::LPDDR5MemoryController>(config);

                case MemoryTechnology::DDR5:
                    return std::make_unique<ddr5::DDR5MemoryController>(config);

                case MemoryTechnology::HBM2:
                case MemoryTechnology::HBM2E:
                    return std::make_unique<hbm2::HBM2MemoryController>(config);

                case MemoryTechnology::HBM3:
                case MemoryTechnology::HBM3E:
                    return std::make_unique<hbm3::HBM3MemoryController>(config);

                case MemoryTechnology::GDDR6:
                    return std::make_unique<gddr6::GDDR6MemoryController>(config);

                case MemoryTechnology::GDDR7:
                    return std::make_unique<gddr7::GDDR7MemoryController>(config);

                default: {
                    std::ostringstream oss;
                    oss << "Unknown memory technology: "
                        << static_cast<int>(config.technology);
                    throw std::invalid_argument(oss.str());
                }
            }

        default: {
            std::ostringstream oss;
            oss << "Unknown simulation fidelity: "
                << static_cast<int>(config.fidelity);
            throw std::invalid_argument(oss.str());
        }
    }
}

// ============================================================================
// Default timing parameters for various technologies
// ============================================================================

MemoryTimingParams MemoryControllerConfig::get_default_timing(
    MemoryTechnology tech, uint32_t speed_mt_s)
{
    MemoryTimingParams timing;

    switch (tech) {
        case MemoryTechnology::IDEAL:
            // Ideal: low fixed latency
            timing.fixed_read_latency = 10;
            timing.fixed_write_latency = 10;
            break;

        case MemoryTechnology::LPDDR5:
            // LPDDR5-6400 typical timing
            if (speed_mt_s >= 7500) {
                // LPDDR5-7500
                timing.tRCD = 16;
                timing.tRP = 16;
                timing.tRAS = 32;
                timing.tRC = 48;
                timing.tCL = 16;
            } else if (speed_mt_s >= 6400) {
                // LPDDR5-6400
                timing.tRCD = 14;
                timing.tRP = 14;
                timing.tRAS = 28;
                timing.tRC = 42;
                timing.tCL = 14;
            } else {
                // LPDDR5-5500
                timing.tRCD = 12;
                timing.tRP = 12;
                timing.tRAS = 24;
                timing.tRC = 36;
                timing.tCL = 12;
            }
            timing.tWL = 8;
            timing.tWR = 24;
            timing.tRTP = 6;
            timing.tRRD_L = 6;
            timing.tRRD_S = 4;
            timing.tCCD_L = 6;
            timing.tCCD_S = 4;
            timing.tWTR_L = 10;
            timing.tWTR_S = 4;
            timing.tRTW = 14;
            timing.tFAW = 24;
            break;

        case MemoryTechnology::LPDDR5X:
            // LPDDR5X-8533 typical timing (tighter than LPDDR5)
            timing.tRCD = 18;
            timing.tRP = 18;
            timing.tRAS = 36;
            timing.tRC = 54;
            timing.tCL = 18;
            timing.tWL = 10;
            timing.tWR = 28;
            timing.tRTP = 8;
            timing.tRRD_L = 8;
            timing.tRRD_S = 4;
            timing.tCCD_L = 8;
            timing.tCCD_S = 4;
            timing.tWTR_L = 12;
            timing.tWTR_S = 4;
            timing.tRTW = 16;
            timing.tFAW = 28;
            break;

        case MemoryTechnology::DDR5:
            // DDR5-4800 typical timing
            if (speed_mt_s >= 6400) {
                timing.tRCD = 22;
                timing.tRP = 22;
                timing.tCL = 22;
            } else if (speed_mt_s >= 5600) {
                timing.tRCD = 20;
                timing.tRP = 20;
                timing.tCL = 20;
            } else {
                timing.tRCD = 16;
                timing.tRP = 16;
                timing.tCL = 16;
            }
            timing.tRAS = 32;
            timing.tRC = timing.tRAS + timing.tRP;
            timing.tWL = 8;
            timing.tWR = 24;
            timing.tRTP = 8;
            timing.tRRD_L = 8;
            timing.tRRD_S = 4;
            timing.tCCD_L = 8;
            timing.tCCD_S = 4;
            timing.tWTR_L = 12;
            timing.tWTR_S = 4;
            timing.tRTW = 16;
            timing.tFAW = 32;
            break;

        case MemoryTechnology::HBM2:
            // HBM2-2000 typical timing @ 2 Gbps (1.0 GHz CK)
            timing.tRCD = 12;
            timing.tRP = 14;
            timing.tRAS = 28;
            timing.tRC = 42;
            timing.tCL = 18;
            timing.tWL = 7;
            timing.tWR = 16;
            timing.tRTP = 6;
            timing.tRRD_L = 4;
            timing.tRRD_S = 3;
            timing.tCCD_L = 4;
            timing.tCCD_S = 2;
            timing.tWTR_L = 8;
            timing.tWTR_S = 4;
            timing.tRTW = 10;
            timing.tFAW = 16;
            break;

        case MemoryTechnology::HBM2E:
            // HBM2E-3600 timing @ 3.6 Gbps (1.8 GHz CK)
            // Scaled from HBM2: 1.0/1.8 ≈ 0.56x
            timing.tRCD = 7;    // 12 * 0.56 ≈ 7
            timing.tRP = 8;     // 14 * 0.56 ≈ 8
            timing.tRAS = 16;   // 28 * 0.56 ≈ 16
            timing.tRC = 24;    // 42 * 0.56 ≈ 24
            timing.tCL = 10;    // 18 * 0.56 ≈ 10
            timing.tWL = 4;     // 7 * 0.56 ≈ 4
            timing.tWR = 9;     // 16 * 0.56 ≈ 9
            timing.tRTP = 4;    // 6 * 0.56 ≈ 4
            timing.tRRD_L = 3;  // 4 * 0.56 ≈ 3
            timing.tRRD_S = 2;  // 3 * 0.56 ≈ 2
            timing.tCCD_L = 3;  // 4 * 0.56 ≈ 3
            timing.tCCD_S = 2;  // 2 * 0.56 ≈ 2
            timing.tWTR_L = 5;  // 8 * 0.56 ≈ 5
            timing.tWTR_S = 3;  // 4 * 0.56 ≈ 3
            timing.tRTW = 6;    // 10 * 0.56 ≈ 6
            timing.tFAW = 9;    // 16 * 0.56 ≈ 9
            break;

        case MemoryTechnology::HBM3:
            // HBM3-5600 typical timing @ 5.6 Gbps (2.8 GHz CK)
            timing.tRCD = 8;
            timing.tRP = 8;
            timing.tRAS = 16;
            timing.tRC = 24;
            timing.tCL = 8;
            timing.tWL = 4;
            timing.tWR = 12;
            timing.tRTP = 4;
            timing.tRRD_L = 4;
            timing.tRRD_S = 2;
            timing.tCCD_L = 4;
            timing.tCCD_S = 2;
            timing.tWTR_L = 6;
            timing.tWTR_S = 3;
            timing.tRTW = 8;
            timing.tFAW = 16;
            break;

        case MemoryTechnology::HBM3E:
            // HBM3E-9600 timing @ 9.6 Gbps (4.8 GHz CK)
            // Scaled from HBM3: 2.8/4.8 ≈ 0.58x
            timing.tRCD = 5;    // 8 * 0.58 ≈ 5
            timing.tRP = 5;     // 8 * 0.58 ≈ 5
            timing.tRAS = 10;   // 16 * 0.58 ≈ 10
            timing.tRC = 14;    // 24 * 0.58 ≈ 14
            timing.tCL = 5;     // 8 * 0.58 ≈ 5
            timing.tWL = 3;     // 4 * 0.58 ≈ 3
            timing.tWR = 7;     // 12 * 0.58 ≈ 7
            timing.tRTP = 3;    // 4 * 0.58 ≈ 3
            timing.tRRD_L = 3;  // 4 * 0.58 ≈ 3
            timing.tRRD_S = 2;  // 2 * 0.58 ≈ 2
            timing.tCCD_L = 3;  // 4 * 0.58 ≈ 3
            timing.tCCD_S = 2;  // 2 * 0.58 ≈ 2
            timing.tWTR_L = 4;  // 6 * 0.58 ≈ 4
            timing.tWTR_S = 2;  // 3 * 0.58 ≈ 2
            timing.tRTW = 5;    // 8 * 0.58 ≈ 5
            timing.tFAW = 10;   // 16 * 0.58 ≈ 10
            break;

        case MemoryTechnology::GDDR6:
            // GDDR6-16000 typical timing
            timing.tRCD = 12;
            timing.tRP = 12;
            timing.tRAS = 24;
            timing.tRC = 36;
            timing.tCL = 12;
            timing.tWL = 6;
            timing.tWR = 20;
            timing.tRTP = 6;
            timing.tRRD_L = 4;
            timing.tRRD_S = 4;
            timing.tCCD_L = 4;
            timing.tCCD_S = 4;
            timing.tWTR_L = 8;
            timing.tWTR_S = 8;
            timing.tRTW = 10;
            timing.tFAW = 20;
            break;

        case MemoryTechnology::GDDR7:
            // GDDR7-32000 typical timing (PAM3 signaling)
            timing.tRCD = 14;
            timing.tRP = 14;
            timing.tRAS = 24;
            timing.tRC = 38;
            timing.tCL = 16;
            timing.tWL = 6;
            timing.tWR = 14;
            timing.tRTP = 6;
            timing.tRRD_L = 4;
            timing.tRRD_S = 3;
            timing.tCCD_L = 3;
            timing.tCCD_S = 2;
            timing.tWTR_L = 5;
            timing.tWTR_S = 3;
            timing.tRTW = 12;
            timing.tFAW = 12;
            break;

        default:
            // Use LPDDR5-6400 as default
            timing.tRCD = 14;
            timing.tRP = 14;
            timing.tRAS = 28;
            timing.tRC = 42;
            timing.tCL = 14;
            timing.tWL = 8;
            timing.tWR = 24;
            timing.tRTP = 6;
            timing.tRRD_L = 6;
            timing.tRRD_S = 4;
            timing.tCCD_L = 6;
            timing.tCCD_S = 4;
            timing.tWTR_L = 10;
            timing.tWTR_S = 4;
            timing.tRTW = 14;
            timing.tFAW = 24;
            break;
    }

    // Set behavioral/transactional latencies based on cycle-accurate parameters
    timing.fixed_read_latency = timing.tRCD + timing.tCL + timing.tBurst;
    timing.fixed_write_latency = timing.tRCD + timing.tWL + timing.tBurst;
    timing.mean_read_latency = timing.fixed_read_latency;
    timing.mean_write_latency = timing.fixed_write_latency;

    return timing;
}

} // namespace sw::kpu
