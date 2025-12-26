/**
 * @file host_kpu.cpp
 * @brief models a host + KPU simulator configuration
 *
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

void create_system(SystemConfig& config) {

    std::cout << "========================================\n";
    std::cout << "   Creating a Host + KPU configuration\n";
    std::cout << "========================================\n";

    config.clear();

    // System info
    config.system.name = "Host+KPU Baseline System";
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
    kpu_accel.id = "T100";
    kpu_accel.description = "Custom configured KPU to deliver 100 TOPS of sustained performance";

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

    // Add L3 tiles
    for (int i = 0; i < 4; ++i) {
        KPUTileConfig tile;
        tile.id = "l3_" + std::to_string(i);
        tile.capacity_kb = 256;
        kpu.memory.l3_tiles.push_back(tile);
    }

    // Add L2 banks
    for (int i = 0; i < 8; ++i) {
        KPUTileConfig bank;
        bank.id = "l2_" + std::to_string(i);
        bank.capacity_kb = 128;
        kpu.memory.l2_banks.push_back(bank);
    }

    // Add L1 buffers
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

    // Add block movers
    for (int i = 0; i < 4; ++i) {
        BlockMoverConfig mover;
        mover.id = "block_mover_" + std::to_string(i);
        kpu.data_movement.block_movers.push_back(mover);
    }

    // Add streamers
    for (int i = 0; i < 8; ++i) {
        StreamerConfig streamer;
        streamer.id = "streamer_" + std::to_string(i);
        kpu.data_movement.streamers.push_back(streamer);
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

    // Print configuration using new formatter
    std::cout << "\nCreated configuration:\n";
    std::cout << config;

    // Validate
    std::cout << "Validation: " << (config.validate() ? "PASSED" : "FAILED") << "\n";
}

void demo_json_round_trip() {
    std::cout << "========================================\n";
    std::cout << "   JSON Serialization Round Trip\n";
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

/**
 * @brief Execute MLP layer through complete memory hierarchy
 *
 * Data flow pipeline:
 * 1. Host memory → KPU memory banks (via DMA simulation)
 * 2. Memory banks → L3 tiles (via DMA)
 * 3. L3 tiles → L2 banks (via Block Movers)
 * 4. L2 banks → L1 scratchpad (via Streamers)
 * 5. Compute on systolic array: output = input × weights + bias
 * 6. Result readback through reverse path
 */
bool execute_mlp_layer(sw::kpu::KPUSimulator* kpu,
                       size_t batch_size,
                       size_t input_dim,
                       size_t output_dim) {
    using namespace sw;
    using namespace sw::kpu;

    std::cout << "\n========================================\n";
    std::cout << "  MLP Layer Execution\n";
    std::cout << "========================================\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Input dimension: " << input_dim << "\n";
    std::cout << "Output dimension: " << output_dim << "\n";
    std::cout << "\n--- Data Movement Pipeline ---\n";

    // Step 1: Allocate and initialize tensors in host memory
    std::cout << "\n[1] Host Memory Allocation\n";

    // Input tensor: [batch_size × input_dim]
    std::vector<float> input(batch_size * input_dim);
    // Weight matrix: [input_dim × output_dim]
    std::vector<float> weights(input_dim * output_dim);
    // Bias vector: [output_dim]
    std::vector<float> bias(output_dim);
    // Output tensor: [batch_size × output_dim]
    std::vector<float> output(batch_size * output_dim, 0.0f);

    // Initialize with simple test data
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 10) * 0.1f;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>((i % 5) + 1) * 0.2f;
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = 0.5f;
    }

    std::cout << "  Input tensor allocated: " << input.size() * sizeof(float) / 1024.0f << " KB\n";
    std::cout << "  Weight matrix allocated: " << weights.size() * sizeof(float) / 1024.0f << " KB\n";
    std::cout << "  Bias vector allocated: " << bias.size() * sizeof(float) / 1024.0f << " KB\n";

    // Step 2: Transfer from host to KPU memory banks (simulated as direct write)
    std::cout << "\n[2] Host -> KPU Memory Banks (DMA simulation)\n";

    const size_t bank_id = 0;
    const Address input_addr = 0x0000;
    const Address weights_addr = input_addr + input.size() * sizeof(float);
    const Address bias_addr = weights_addr + weights.size() * sizeof(float);

    kpu->write_memory_bank(bank_id, input_addr, input.data(), input.size() * sizeof(float));
    kpu->write_memory_bank(bank_id, weights_addr, weights.data(), weights.size() * sizeof(float));
    kpu->write_memory_bank(bank_id, bias_addr, bias.data(), bias.size() * sizeof(float));

    std::cout << "  Input -> Bank[" << bank_id << "] @ 0x" << std::hex << input_addr << std::dec << "\n";
    std::cout << "  Weights -> Bank[" << bank_id << "] @ 0x" << std::hex << weights_addr << std::dec << "\n";
    std::cout << "  Bias -> Bank[" << bank_id << "] @ 0x" << std::hex << bias_addr << std::dec << "\n";

    // Step 3: Manual transfer from memory banks to L3 tiles
    // Note: DMA only supports EXTERNAL<->SCRATCHPAD, so we use direct read/write for L3
    std::cout << "\n[3] Memory Banks -> L3 Tiles (manual transfer)\n";

    const size_t l3_tile_id = 0;
    const Address l3_input_addr = 0x0000;
    const Address l3_weights_addr = 0x4000;

    // Transfer input to L3 (read from bank, write to L3)
    std::vector<uint8_t> temp_buffer(std::max(input.size(), weights.size()) * sizeof(float));
    kpu->read_memory_bank(bank_id, input_addr, temp_buffer.data(), input.size() * sizeof(float));
    kpu->write_l3_tile(l3_tile_id, l3_input_addr, temp_buffer.data(), input.size() * sizeof(float));
    std::cout << "  Input transferred to L3[" << l3_tile_id << "]\n";

    // Transfer weights to L3
    kpu->read_memory_bank(bank_id, weights_addr, temp_buffer.data(), weights.size() * sizeof(float));
    kpu->write_l3_tile(l3_tile_id, l3_weights_addr, temp_buffer.data(), weights.size() * sizeof(float));
    std::cout << "  Weights transferred to L3[" << l3_tile_id << "]\n";

    // Step 4: Block mover from L3 to L2
    std::cout << "\n[4] L3 Tiles -> L2 Banks (Block Mover)\n";

    const size_t block_mover_id = 0;
    const size_t l2_bank_id = 0;
    const Address l2_input_addr = 0x0000;
    const Address l2_weights_addr = 0x2000;

    // Transfer input blocks to L2
    kpu->start_block_transfer(block_mover_id, l3_tile_id, l3_input_addr,
        l2_bank_id, l2_input_addr,
        batch_size, input_dim, sizeof(float));
    kpu->run_until_idle();
    std::cout << "  Input blocks moved to L2[" << l2_bank_id << "]\n";

    // Transfer weight blocks to L2
    kpu->start_block_transfer(block_mover_id, l3_tile_id, l3_weights_addr,
        l2_bank_id, l2_weights_addr,
        input_dim, output_dim, sizeof(float));
    kpu->run_until_idle();
    std::cout << "  Weight blocks moved to L2[" << l2_bank_id << "]\n";

    // Step 5: Streamers from L2 to L1 buffer
    std::cout << "\n[5] L2 Banks -> L1 Buffer (Streamers)\n";

    const size_t row_streamer_id = 0;
    const size_t col_streamer_id = 1;
    const size_t l1_buffer_id = 0;
    const Address l1_input_addr = 0x0000;
    const Address l1_weights_addr = 0x1000;
    const size_t compute_fabric_size = kpu->get_systolic_array_rows();

    // Stream input rows to L1
    kpu->start_row_stream(row_streamer_id, l2_bank_id, l1_buffer_id,
        l2_input_addr, l1_input_addr,
        batch_size, input_dim, sizeof(float), compute_fabric_size);
    kpu->run_until_idle();
    std::cout << "  Input rows streamed to L1 buffer[" << l1_buffer_id << "]\n";

    // Stream weight columns to L1
    kpu->start_column_stream(col_streamer_id, l2_bank_id, l1_buffer_id,
        l2_weights_addr, l1_weights_addr,
        input_dim, output_dim, sizeof(float), compute_fabric_size);
    kpu->run_until_idle();
    std::cout << "  Weight columns streamed to L1 buffer[" << l1_buffer_id << "]\n";

    // Step 6: Execute matrix multiplication on systolic array
    std::cout << "\n[6] Systolic Array Compute\n";

    const size_t compute_tile_id = 0;
    const Address l1_output_addr = 0x2000;

    std::cout << "  Systolic array: " << kpu->get_systolic_array_rows()
              << "×" << kpu->get_systolic_array_cols()
              << " (" << kpu->get_systolic_array_total_pes() << " PEs)\n";

    kpu->start_matmul(compute_tile_id, l1_buffer_id,
        batch_size, output_dim, input_dim,
        l1_input_addr, l1_weights_addr, l1_output_addr);
    kpu->run_until_idle();
    std::cout << "  Matrix multiplication completed\n";

    // Add bias (simple operation in L1 buffer)
    std::cout << "  Adding bias...\n";
    std::vector<float> result(batch_size * output_dim);
    kpu->read_l1_buffer(l1_buffer_id, l1_output_addr, result.data(), result.size() * sizeof(float));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] += bias[i % output_dim];
    }
    kpu->write_l1_buffer(l1_buffer_id, l1_output_addr, result.data(), result.size() * sizeof(float));
    std::cout << "  Bias added\n";

    // Step 7: Result readback through reverse path
    std::cout << "\n[7] Result Readback Path\n";

    // L1 → L2 (via streamer)
    const Address l2_output_addr = 0x4000;
    kpu->start_row_stream(row_streamer_id, l2_bank_id, l1_buffer_id,
        l2_output_addr, l1_output_addr,
        batch_size, output_dim, sizeof(float), compute_fabric_size,
        sw::kpu::Streamer::StreamDirection::L1_TO_L2);
    kpu->run_until_idle();
    std::cout << "  L1 -> L2 (streamer)\n";

    // L2 → L3 (via block mover with reverse transform)
    const Address l3_output_addr = 0x8000;
    kpu->start_block_transfer(block_mover_id, l2_bank_id, l2_output_addr,
        l3_tile_id, l3_output_addr,
        batch_size, output_dim, sizeof(float));
    kpu->run_until_idle();
    std::cout << "  L2 -> L3 (block mover)\n";

    // L3 → Memory bank (manual transfer)
    const Address output_addr = 0x10000;
    kpu->read_l3_tile(l3_tile_id, l3_output_addr, temp_buffer.data(), result.size() * sizeof(float));
    kpu->write_memory_bank(bank_id, output_addr, temp_buffer.data(), result.size() * sizeof(float));
    std::cout << "  L3 -> Memory bank (manual transfer)\n";

    // Memory bank → Host (read back)
    kpu->read_memory_bank(bank_id, output_addr, output.data(), output.size() * sizeof(float));
    std::cout << "  Memory bank → Host\n";

    // Verify results
    std::cout << "\n[8] Result Verification\n";
    std::cout << "  Sample outputs (first 5):\n";
    for (size_t i = 0; i < std::min(size_t(5), output.size()); ++i) {
        std::cout << "    output[" << i << "] = " << output[i] << "\n";
    }

    std::cout << "\nMLP layer execution completed successfully!\n";
    return true;
}

// Run Built-in Self Test
bool bist(const SystemConfig& config) {
    std::cout << "========================================\n";
    std::cout << "    System Simulator BIST\n";
    std::cout << "========================================\n";

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
            std::cout << "  L3 tiles: " << kpu->get_l3_tile_count() << "\n";
            std::cout << "  L2 banks: " << kpu->get_l2_bank_count() << "\n";
            std::cout << "  Block movers: " << kpu->get_block_mover_count() << "\n";
            std::cout << "  Streamers: " << kpu->get_streamer_count() << "\n";

            // Show memory map
            std::cout << sim.get_memory_map(0);

            // Run MLP layer execution demo
            execute_mlp_layer(kpu, 4, 8, 4);  // Small test: 4 batch, 8 input dim, 4 output dim
        }

        // Run self test
        std::cout << "\nRunning self test...\n";
        bool test_passed = sim.run_self_test();
        std::cout << "Self test: " << (test_passed ? "PASSED" : "FAILED") << "\n";

        // Shutdown
        sim.shutdown();
        std::cout << "Shutdown: complete\n";
	    return true;
    }
    else {
        std::cout << "Initialization: FAILED\n";
	    return false;
    }
}

int main() {
    std::cout << "===========================================\n";
    std::cout << " Host + T100 KPU configuration\n";
    std::cout << "===========================================\n";

    try {
	bool bSaveToFile = false;
	SystemConfig config;
	create_system(config);
        // bool success = bist(config);
        bist(config);
	
	if (bSaveToFile) {
	    // Save to file
	    std::filesystem::path temp_file = "host_kpu_T100.json";
	    ConfigLoader::save_to_file(config, temp_file);
	    std::cout << "Saved to: " << temp_file << "\n";
	}

	std::cout << '\n';
        std::cout << "===========================================\n";
        std::cout << " simulation completed successfully!\n";
        std::cout << "===========================================\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
