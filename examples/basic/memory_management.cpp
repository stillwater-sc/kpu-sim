/**
 * @file memory_management.cpp
 * @brief Demonstrates memory management patterns in KPU
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

void print_memory_info(const sw::kpu::KPUSimulator& kpu) {
    std::cout << "\nMemory Configuration:\n";
    std::cout << "  Memory banks: " << kpu.get_memory_bank_count() << "\n";
    for (size_t i = 0; i < kpu.get_memory_bank_count(); ++i) {
        std::cout << "    Bank " << i << ": "
                  << (kpu.get_memory_bank_capacity(i) / (1024 * 1024)) << " MB\n";
    }

    std::cout << "  L1 buffers: " << kpu.get_l1_buffer_count() << "\n";
    for (size_t i = 0; i < kpu.get_l1_buffer_count(); ++i) {
        std::cout << "    L1 buffer " << i << ": "
                  << (kpu.get_l1_buffer_capacity(i) / 1024) << " KB\n";
    }

    std::cout << "  L3 tiles: " << kpu.get_l3_tile_count() << "\n";
    std::cout << "  L2 banks: " << kpu.get_l2_bank_count() << "\n";
}

int main() {
    std::cout << "===========================================\n";
    std::cout << " KPU Memory Management Example\n";
    std::cout << "===========================================\n";

    // Create KPU with multiple memory banks and L1 buffers
    sw::kpu::KPUSimulator::Config config;
    config.memory_bank_count = 4;
    config.memory_bank_capacity_mb = 512;
    config.memory_bandwidth_gbps = 100;
    config.l1_buffer_count = 4;
    config.l1_buffer_capacity_kb = 64;
    config.compute_tile_count = 2;
    config.dma_engine_count = 4;
    config.l3_tile_count = 4;
    config.l3_tile_capacity_kb = 256;
    config.l2_bank_count = 8;
    config.l2_bank_capacity_kb = 64;
    config.block_mover_count = 4;
    config.streamer_count = 8;

    sw::kpu::KPUSimulator kpu(config);
    print_memory_info(kpu);

    // Demonstrate memory operations
    std::cout << "\n===========================================\n";
    std::cout << " Memory Operations Demo\n";
    std::cout << "===========================================\n";

    // 1. Write to external memory bank
    std::cout << "\n1. Writing to external memory banks...\n";
    std::vector<float> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    const size_t bank_id = 0;
    const size_t addr = 0;
    kpu.write_memory_bank(bank_id, addr, data.data(), data.size() * sizeof(float));
    std::cout << "  Written " << data.size() << " floats to bank " << bank_id << "\n";

    // 2. Read back from memory
    std::cout << "\n2. Reading from external memory...\n";
    std::vector<float> read_data(1024);
    kpu.read_memory_bank(bank_id, addr, read_data.data(), read_data.size() * sizeof(float));
    std::cout << "  Read " << read_data.size() << " floats from bank " << bank_id << "\n";

    // Verify
    bool match = true;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] != read_data[i]) {
            match = false;
            break;
        }
    }
    std::cout << "  Data verification: " << (match ? "PASS" : "FAIL") << "\n";

    // 3. Transfer through memory hierarchy: External → L3 → L2 → L1
    std::cout << "\n3. Memory hierarchy transfer pipeline...\n";
    const size_t l3_tile_id = 0;
    const size_t l2_bank_id = 0;
    const size_t l1_buffer_id = 0;
    const size_t transfer_size = 256 * sizeof(float);

    // Step 3a: Write to L3 tile (simulating DMA to L3)
    std::cout << "  External memory → L3 tile...\n";
    kpu.write_l3_tile(l3_tile_id, 0, data.data(), transfer_size);
    std::cout << "  Written " << transfer_size << " bytes to L3 tile\n";

    // Step 3b: BlockMover L3 → L2
    std::cout << "  L3 tile → L2 bank (via BlockMover)...\n";
    bool block_done = false;
    kpu.start_block_transfer(0, l3_tile_id, 0, l2_bank_id, 0,
                             16, 16, sizeof(float),  // 16x16 block
                             sw::kpu::BlockMover::TransformType::IDENTITY,
                             [&]() { block_done = true; });
    kpu.run_until_idle();
    std::cout << "  BlockMover transfer complete\n";

    // Step 3c: Streamer L2 → L1
    std::cout << "  L2 bank → L1 buffer (via Streamer)...\n";
    bool stream_done = false;
    kpu.start_row_stream(0, l2_bank_id, l1_buffer_id, 0, 0,
                         16, 16, sizeof(float), 16,
                         sw::kpu::Streamer::StreamDirection::L2_TO_L1,
                         [&]() { stream_done = true; });
    kpu.run_until_idle();
    std::cout << "  Streamer transfer complete\n";

    // 4. Read from L1 buffer
    std::cout << "\n4. Reading from L1 buffer...\n";
    std::vector<float> l1_data(256);
    kpu.read_l1_buffer(l1_buffer_id, 0, l1_data.data(), l1_data.size() * sizeof(float));
    std::cout << "  Read " << l1_data.size() << " floats from L1 buffer " << l1_buffer_id << "\n";

    // Verify L1 data matches original
    match = true;
    for (size_t i = 0; i < l1_data.size(); ++i) {
        if (l1_data[i] != data[i]) {
            match = false;
            break;
        }
    }
    std::cout << "  L1 buffer data verification: " << (match ? "PASS" : "FAIL") << "\n";

    // 5. Write to L1 buffer and transfer back through hierarchy
    std::cout << "\n5. Write to L1 buffer and transfer back...\n";
    std::vector<float> new_data(256);
    for (size_t i = 0; i < new_data.size(); ++i) {
        new_data[i] = static_cast<float>(i * 2);
    }

    kpu.write_l1_buffer(l1_buffer_id, 0, new_data.data(), new_data.size() * sizeof(float));
    std::cout << "  Written " << new_data.size() << " floats to L1 buffer\n";

    // L1 → L2 (via Streamer)
    std::cout << "  L1 buffer → L2 bank (via Streamer)...\n";
    stream_done = false;
    kpu.start_row_stream(0, l2_bank_id, l1_buffer_id, 0x1000, 0,
                         16, 16, sizeof(float), 16,
                         sw::kpu::Streamer::StreamDirection::L1_TO_L2,
                         [&]() { stream_done = true; });
    kpu.run_until_idle();
    std::cout << "  Streamer transfer complete\n";

    // Read from L2 and verify
    std::vector<float> final_data(256);
    kpu.read_l2_bank(l2_bank_id, 0x1000, final_data.data(), final_data.size() * sizeof(float));

    match = true;
    for (size_t i = 0; i < final_data.size(); ++i) {
        if (final_data[i] != new_data[i]) {
            match = false;
            break;
        }
    }
    std::cout << "  Final data verification: " << (match ? "PASS" : "FAIL") << "\n";

    // Print simulation statistics
    std::cout << "\n===========================================\n";
    std::cout << " Simulation Statistics\n";
    std::cout << "===========================================\n";
    std::cout << "  Total cycles: " << kpu.get_current_cycle() << "\n";
    std::cout << "  Elapsed time: " << kpu.get_elapsed_time_ms() << " ms\n";

    kpu.print_component_status();

    std::cout << "\n===========================================\n";
    std::cout << " All memory operations completed!\n";
    std::cout << "===========================================\n";

    return 0;
}
