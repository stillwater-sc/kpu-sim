/**
 * @file system_config_demo.cpp
 * @brief Demonstrates JSON-based system configuration
 *
 * This example shows how to:
 * 1. Load configurations from JSON files
 * 2. Create configurations programmatically
 * 3. Validate configurations
 * 4. Initialize the system simulator
 * 5. Access configured accelerators
 */

#include <sw/system/toplevel.hpp>
#include <sw/system/config_loader.hpp>
#include <sw/system/config_formatter.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <iostream>
#include <filesystem>

using namespace sw::sim;

void demo_factory_configs() {
    std::cout << "\n========================================\n";
    std::cout << "Demo 1: Factory Configuration Methods\n";
    std::cout << "========================================\n";

    // Create predefined configurations
    auto minimal = SystemConfig::create_minimal_kpu();
    auto edge_ai = SystemConfig::create_edge_ai();
    auto datacenter = SystemConfig::create_datacenter();

    std::cout << "\nAvailable factory configurations:\n";
    std::cout << "1. " << minimal.system.name << "\n";
    std::cout << "   - KPUs: " << minimal.get_kpu_count() << "\n";
    std::cout << "   - GPUs: " << minimal.get_gpu_count() << "\n";
    std::cout << "   - NPUs: " << minimal.get_npu_count() << "\n";

    std::cout << "2. " << edge_ai.system.name << "\n";
    std::cout << "   - KPUs: " << edge_ai.get_kpu_count() << "\n";
    std::cout << "   - GPUs: " << edge_ai.get_gpu_count() << "\n";
    std::cout << "   - NPUs: " << edge_ai.get_npu_count() << "\n";

    std::cout << "3. " << datacenter.system.name << "\n";
    std::cout << "   - KPUs: " << datacenter.get_kpu_count() << "\n";
    std::cout << "   - GPUs: " << datacenter.get_gpu_count() << "\n";
    std::cout << "   - NPUs: " << datacenter.get_npu_count() << "\n";
}

void demo_json_file_loading() {
    std::cout << "\n========================================\n";
    std::cout << "Demo 2: Loading from JSON Files\n";
    std::cout << "========================================\n";

    std::filesystem::path examples_dir = "../../configs/examples";
    if (!std::filesystem::exists(examples_dir)) {
        examples_dir = "../configs/examples";
    }

    if (!std::filesystem::exists(examples_dir)) {
        std::cout << "Example configurations not found, skipping demo\n";
        return;
    }

    // Try to load each example configuration
    std::vector<std::string> config_files = {
        "minimal_kpu.json",
        "edge_ai.json",
        "datacenter_hbm.json"
    };

    for (const auto& filename : config_files) {
        auto config_path = examples_dir / filename;
        if (!std::filesystem::exists(config_path)) {
            continue;
        }

        std::cout << "\nLoading: " << filename << "\n";

        try {
            auto config = ConfigLoader::load_from_file(config_path);
            std::cout << "  System: " << config.system.name << "\n";
            std::cout << "  Valid: " << (config.validate() ? "Yes" : "No") << "\n";
            std::cout << "  Accelerators: " << config.accelerators.size() << "\n";

            // Get validation warnings/errors
            auto validation_msg = config.get_validation_errors();
            if (!validation_msg.empty()) {
                std::cout << "  Notes:\n" << validation_msg;
            }
        }
        catch (const std::exception& e) {
            std::cout << "  Error: " << e.what() << "\n";
        }
    }
}

void demo_programmatic_config() {
    std::cout << "\n========================================\n";
    std::cout << "Demo 3: Programmatic Configuration\n";
    std::cout << "========================================\n";

    SystemConfig config;

    // System info
    config.system.name = "Custom Demo System";
    config.system.description = "Programmatically created configuration";

    // Host configuration
    config.host.cpu.core_count = 16;
    config.host.cpu.frequency_mhz = 3000;

    MemoryModuleConfig mem;
    mem.id = "ddr5_dimm_0";
    mem.type = "DDR5";
    mem.form_factor = "DIMM";
    mem.capacity_gb = 64;
    mem.bandwidth_gbps = 51.2f;
    config.host.memory.modules.push_back(mem);

    // KPU accelerator
    AcceleratorConfig kpu_accel;
    kpu_accel.type = AcceleratorType::KPU;
    kpu_accel.id = "my_kpu";
    kpu_accel.description = "Custom configured KPU";

    KPUConfig kpu;
    kpu.memory.type = "GDDR6";
    kpu.memory.form_factor = "PCB";

    // Add memory banks
    for (int i = 0; i < 2; ++i) {
        KPUMemoryBankConfig bank;
        bank.id = "bank_" + std::to_string(i);
        bank.capacity_mb = 2048;
        bank.bandwidth_gbps = 150.0f;
        kpu.memory.banks.push_back(bank);
    }

    // Add L1 buffers (compute fabric)
    for (int i = 0; i < 4; ++i) {
        KPUL1Config l1_buf;
        l1_buf.id = "l1_buffer_" + std::to_string(i);
        l1_buf.capacity_kb = 128;
        kpu.memory.l1_buffers.push_back(l1_buf);
    }

    // Add compute tiles
    for (int i = 0; i < 4; ++i) {
        ComputeTileConfig tile;
        tile.id = "tile_" + std::to_string(i);
        tile.type = "systolic";
        tile.systolic_rows = 16;
        tile.systolic_cols = 16;
        tile.datatype = "fp32";
        kpu.compute_fabric.tiles.push_back(tile);
    }

    // Add DMA engines
    for (int i = 0; i < 4; ++i) {
        DMAEngineConfig dma;
        dma.id = "dma_" + std::to_string(i);
        dma.bandwidth_gbps = 75.0f;
        kpu.data_movement.dma_engines.push_back(dma);
    }

    kpu_accel.kpu_config = kpu;
    config.accelerators.push_back(kpu_accel);

    // Interconnect
    config.interconnect.host_to_accelerator.type = "PCIe";
    PCIeConfig pcie;
    pcie.generation = 4;
    pcie.lanes = 16;
    pcie.bandwidth_gbps = 32.0f;
    config.interconnect.host_to_accelerator.pcie_config = pcie;

    std::cout << "\nCreated configuration:\n";
    std::cout << config;
    std::cout << "Validation: " << (config.validate() ? "PASSED" : "FAILED") << "\n";
}

void demo_json_round_trip() {
    std::cout << "\n========================================\n";
    std::cout << "Demo 4: JSON Serialization Round Trip\n";
    std::cout << "========================================\n";

    // Create a configuration
    auto config = SystemConfig::create_edge_ai();

    std::cout << "\nOriginal configuration: " << config.system.name << "\n";

    // Serialize to JSON string
    std::string json_str = ConfigLoader::to_json_string(config, true);
    std::cout << "JSON size: " << json_str.length() << " characters\n";

    // Save to file
    std::filesystem::path temp_file = "demo_config_temp.json";
    ConfigLoader::save_to_file(config, temp_file);
    std::cout << "Saved to: " << temp_file << "\n";

    // Load back
    auto loaded_config = ConfigLoader::load_from_file(temp_file);
    std::cout << "Loaded configuration: " << loaded_config.system.name << "\n";
    std::cout << "Configurations match: "
              << (loaded_config.system.name == config.system.name ? "Yes" : "No") << "\n";

    // Cleanup
    std::filesystem::remove(temp_file);
    std::cout << "Cleaned up temporary file\n";
}

void demo_simulator_initialization() {
    std::cout << "\n========================================\n";
    std::cout << "Demo 5: System Simulator Initialization\n";
    std::cout << "========================================\n";

    // Create configuration
    auto config = SystemConfig::create_minimal_kpu();
    std::cout << "\nCreating simulator with: " << config.system.name << "\n";

    // Initialize simulator
    SystemSimulator sim(config);
    if (sim.initialize()) {
        std::cout << "Initialization: SUCCESS\n";

        // Access KPU
        std::cout << "\nKPU count: " << sim.get_kpu_count() << "\n";

        auto* kpu = sim.get_kpu(0);
        if (kpu) {
            std::cout << "KPU[0] details:\n";
            std::cout << "  Memory banks: " << kpu->get_memory_bank_count() << "\n";
            std::cout << "  L1 buffers: " << kpu->get_l1_buffer_count() << "\n";
            std::cout << "  Compute tiles: " << kpu->get_compute_tile_count() << "\n";
            std::cout << "  DMA engines: " << kpu->get_dma_engine_count() << "\n";
        }

        // Run self test
        std::cout << "\nRunning self test...\n";
        bool test_passed = sim.run_self_test();
        std::cout << "Self test: " << (test_passed ? "PASSED" : "FAILED") << "\n";

        // Shutdown
        sim.shutdown();
        std::cout << "Shutdown: complete\n";
    }
    else {
        std::cout << "Initialization: FAILED\n";
    }
}

int main() {
    std::cout << "===========================================\n";
    std::cout << " System Configuration Demo\n";
    std::cout << "===========================================\n";

    try {
        demo_factory_configs();
        demo_json_file_loading();
        demo_programmatic_config();
        demo_json_round_trip();
        demo_simulator_initialization();

        std::cout << "\n===========================================\n";
        std::cout << " All demos completed successfully!\n";
        std::cout << "===========================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
}
