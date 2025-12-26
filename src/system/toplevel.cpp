#include "sw/system/toplevel.hpp"
#include "sw/system/config_loader.hpp"
#include "sw/system/config_formatter.hpp"
#include "sw/kpu/kpu_simulator.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace sw::sim {

//=============================================================================
// Constructors and Destructor
//=============================================================================

SystemSimulator::SystemSimulator()
    : config_(SystemConfig::create_minimal_kpu()) {
    std::cout << "[SystemSimulator] Constructor called with default config\n";
}

SystemSimulator::SystemSimulator(const SystemConfig& config)
    : config_(config) {
    std::cout << "[SystemSimulator] Constructor called with custom config\n";
}

SystemSimulator::SystemSimulator(const std::filesystem::path& config_file) {
    std::cout << "[SystemSimulator] Loading configuration from: " << config_file << "\n";
    config_ = ConfigLoader::load_from_file(config_file);
}

SystemSimulator::~SystemSimulator() {
    if (initialized_) {
        shutdown();
    }
}

SystemSimulator::SystemSimulator(SystemSimulator&&) noexcept = default;
SystemSimulator& SystemSimulator::operator=(SystemSimulator&&) noexcept = default;

//=============================================================================
// Initialization and Shutdown
//=============================================================================

bool SystemSimulator::initialize() {
    if (initialized_) {
        std::cout << "[SystemSimulator] Already initialized\n";
        return true;
    }

    std::cout << "[SystemSimulator] Initializing system: " << config_.system.name << "\n";

    // Validate configuration
    if (!config_.validate()) {
        std::cerr << "[SystemSimulator] Configuration validation failed:\n";
        std::cerr << config_.get_validation_errors();
        return false;
    }

    // Create components based on configuration
    try {
        create_components_from_config();
    }
    catch (const std::exception& e) {
        std::cerr << "[SystemSimulator] Failed to create components: " << e.what() << "\n";
        destroy_components();
        return false;
    }

    initialized_ = true;
    std::cout << "[SystemSimulator] Initialization complete\n";
    print_config();
    return true;
}

bool SystemSimulator::initialize(const SystemConfig& config) {
    if (initialized_) {
        shutdown();
    }
    config_ = config;
    return initialize();
}

bool SystemSimulator::load_config_and_initialize(const std::filesystem::path& config_file) {
    if (initialized_) {
        shutdown();
    }

    try {
        config_ = ConfigLoader::load_from_file(config_file);
        return initialize();
    }
    catch (const std::exception& e) {
        std::cerr << "[SystemSimulator] Failed to load config: " << e.what() << "\n";
        return false;
    }
}

void SystemSimulator::shutdown() {
    if (!initialized_) {
        std::cout << "[SystemSimulator] Already shut down\n";
        return;
    }

    std::cout << "[SystemSimulator] Shutting down system components...\n";

    destroy_components();

    initialized_ = false;
    std::cout << "[SystemSimulator] Shutdown complete\n";
}

//=============================================================================
// Component Management
//=============================================================================

void SystemSimulator::create_components_from_config() {
    std::cout << "[SystemSimulator] Creating components from configuration...\n";

    // Create KPU instances
    for (const auto& accel_config : config_.accelerators) {
        if (accel_config.type == AcceleratorType::KPU && accel_config.kpu_config.has_value()) {
            std::cout << "[SystemSimulator] Creating KPU: " << accel_config.id << "\n";
            auto* kpu = create_kpu_from_config(accel_config.kpu_config.value());
            if (kpu) {
                kpu_instances_.emplace_back(kpu);
            }
        }
        else if (accel_config.type == AcceleratorType::GPU) {
            std::cout << "[SystemSimulator] GPU support not yet implemented: " << accel_config.id << "\n";
        }
        else if (accel_config.type == AcceleratorType::NPU) {
            std::cout << "[SystemSimulator] NPU support not yet implemented: " << accel_config.id << "\n";
        }
    }

    std::cout << "[SystemSimulator] Created " << kpu_instances_.size() << " KPU instance(s)\n";
}

void SystemSimulator::destroy_components() {
    std::cout << "[SystemSimulator] Destroying components...\n";
    kpu_instances_.clear();
}

sw::kpu::KPUSimulator* SystemSimulator::create_kpu_from_config(const KPUConfig& kpu_config) {
    // Convert KPUConfig to sw::kpu::KPUSimulator::Config
    sw::kpu::KPUSimulator::Config sim_config;

    // Memory configuration
    sim_config.memory_bank_count = kpu_config.memory.banks.size();
    if (!kpu_config.memory.banks.empty()) {
        sim_config.memory_bank_capacity_mb = kpu_config.memory.banks[0].capacity_mb;
        sim_config.memory_bandwidth_gbps = static_cast<sw::kpu::Size>(kpu_config.memory.banks[0].bandwidth_gbps);
    }

    // L1 buffer configuration (compute fabric)
    sim_config.l1_buffer_count = kpu_config.memory.l1_buffers.size();
    if (!kpu_config.memory.l1_buffers.empty()) {
        sim_config.l1_buffer_capacity_kb = kpu_config.memory.l1_buffers[0].capacity_kb;
    }

    // Page buffer configuration (memory controller scratchpads)
    sim_config.page_buffer_count = kpu_config.memory.scratchpads.size();
    if (!kpu_config.memory.scratchpads.empty()) {
        sim_config.page_buffer_capacity_kb = kpu_config.memory.scratchpads[0].capacity_kb;
    }

    // L3 and L2 configuration
    sim_config.l3_tile_count = kpu_config.memory.l3_tiles.size();
    if (!kpu_config.memory.l3_tiles.empty()) {
        sim_config.l3_tile_capacity_kb = kpu_config.memory.l3_tiles[0].capacity_kb;
    }

    sim_config.l2_bank_count = kpu_config.memory.l2_banks.size();
    if (!kpu_config.memory.l2_banks.empty()) {
        sim_config.l2_bank_capacity_kb = kpu_config.memory.l2_banks[0].capacity_kb;
    }

    // Compute configuration
    sim_config.compute_tile_count = kpu_config.compute_fabric.tiles.size();
    if (!kpu_config.compute_fabric.tiles.empty()) {
        const auto& tile = kpu_config.compute_fabric.tiles[0];
        sim_config.use_systolic_array_mode = (tile.type == "systolic");
        sim_config.processor_array_rows = tile.systolic_rows;
        sim_config.processor_array_cols = tile.systolic_cols;
    }

    // Data movement configuration
    sim_config.dma_engine_count = kpu_config.data_movement.dma_engines.size();
    sim_config.block_mover_count = kpu_config.data_movement.block_movers.size();
    sim_config.streamer_count = kpu_config.data_movement.streamers.size();

    return new sw::kpu::KPUSimulator(sim_config);
}

//=============================================================================
// Component Access
//=============================================================================

size_t SystemSimulator::get_kpu_count() const {
    return kpu_instances_.size();
}

sw::kpu::KPUSimulator* SystemSimulator::get_kpu(size_t index) {
    if (index >= kpu_instances_.size()) {
        return nullptr;
    }
    return kpu_instances_[index].get();
}

sw::kpu::KPUSimulator* SystemSimulator::get_kpu_by_id(const std::string& id) {
    size_t idx = 0;
    for (const auto& accel_config : config_.accelerators) {
        if (accel_config.type == AcceleratorType::KPU) {
            if (accel_config.id == id) {
                return get_kpu(idx);
            }
            idx++;
        }
    }
    return nullptr;
}

//=============================================================================
// Testing and Status
//=============================================================================

bool SystemSimulator::run_self_test() {
    if (!initialized_) {
        std::cout << "[SystemSimulator] Cannot run self test - not initialized\n";
        return false;
    }

    std::cout << "[SystemSimulator] Running self test...\n";

    bool test_passed = true;

    // Test KPU instances
    for (size_t i = 0; i < kpu_instances_.size(); ++i) {
        std::cout << "[SystemSimulator] Testing KPU " << i << "...\n";
        auto* kpu = kpu_instances_[i].get();

        // Simple validation
        if (kpu->get_memory_bank_count() == 0) {
            std::cout << "[SystemSimulator] KPU " << i << " has no memory banks!\n";
            test_passed = false;
        }
        if (kpu->get_compute_tile_count() == 0) {
            std::cout << "[SystemSimulator] KPU " << i << " has no compute tiles!\n";
            test_passed = false;
        }
    }

    std::cout << "[SystemSimulator] Self test "
              << (test_passed ? "PASSED" : "FAILED") << "\n";
    return test_passed;
}

void SystemSimulator::print_config() const {
    // Use the new config formatter for comprehensive output
    std::cout << config_;
}

void SystemSimulator::print_status() const {
    std::cout << "\n========================================\n";
    std::cout << "System Status\n";
    std::cout << "========================================\n";
    std::cout << "Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    std::cout << "KPU Instances: " << kpu_instances_.size() << "\n";
    std::cout << "========================================\n\n";
}

//=============================================================================
// Memory Map and System Reporting
//=============================================================================

std::string SystemSimulator::get_memory_map(size_t kpu_index) const {
    if (!initialized_) {
        return "System not initialized\n";
    }

    if (kpu_index >= kpu_instances_.size()) {
        std::ostringstream oss;
        oss << "Invalid KPU index: " << kpu_index << " (available: 0-"
            << (kpu_instances_.size() - 1) << ")\n";
        return oss.str();
    }

    auto* kpu = kpu_instances_[kpu_index].get();
    auto* decoder = kpu->get_address_decoder();

    if (!decoder) {
        return "No address decoder available for this KPU\n";
    }

    std::ostringstream oss;
    oss << "\n========================================\n";
    oss << "KPU[" << kpu_index << "] Memory Map\n";
    oss << "========================================\n";
    oss << decoder->to_string();
    oss << "========================================\n";

    return oss.str();
}

std::string SystemSimulator::get_system_report() const {
    std::ostringstream oss;

    // Configuration
    oss << config_;

    // Runtime status
    oss << "\n========================================\n";
    oss << "Runtime Status\n";
    oss << "========================================\n";
    oss << "Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    oss << "KPU Instances: " << kpu_instances_.size() << "\n";

    if (initialized_ && !kpu_instances_.empty()) {
        oss << "\nKPU Details:\n";
        for (size_t i = 0; i < kpu_instances_.size(); ++i) {
            auto* kpu = kpu_instances_[i].get();
            oss << "  KPU[" << i << "]:\n";
            oss << "    Memory Banks: " << kpu->get_memory_bank_count() << "\n";
            oss << "    L3 Tiles: " << kpu->get_l3_tile_count() << "\n";
            oss << "    L2 Banks: " << kpu->get_l2_bank_count() << "\n";
            oss << "    L1 Buffers: " << kpu->get_l1_buffer_count() << "\n";
            oss << "    Page Buffers: " << kpu->get_page_buffer_count() << "\n";
            oss << "    Compute Tiles: " << kpu->get_compute_tile_count() << "\n";
            oss << "    DMA Engines: " << kpu->get_dma_engine_count() << "\n";
            oss << "    Block Movers: " << kpu->get_block_mover_count() << "\n";
            oss << "    Streamers: " << kpu->get_streamer_count() << "\n";
        }
    }

    oss << "========================================\n";

    // Memory maps for each KPU
    if (initialized_) {
        for (size_t i = 0; i < kpu_instances_.size(); ++i) {
            oss << "\n" << get_memory_map(i);
        }
    }

    return oss.str();
}

void SystemSimulator::print_full_report(std::ostream& os) const {
    os << get_system_report();
}

} // namespace sw::sim