/**
 * @file hello_kpu.cpp
 * @brief Simple first KPU program - Hello World for KPU simulator
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "===========================================\n";
    std::cout << " Hello KPU - First KPU Program\n";
    std::cout << "===========================================\n\n";

    // Create a simple KPU configuration
    sw::kpu::KPUSimulator::Config config;
    config.memory_bank_count = 2;
    config.memory_bank_capacity_mb = 1024;
    config.memory_bandwidth_gbps = 100;
	config.memory_controller_count = 4;
    config.page_buffer_count = 4;
	config.page_buffer_capacity_kb = 4;
	config.l3_layer.tile_groups = { {"l3", {128}, 2} }; // 2 tiles of 128KB each in a uniform group
	config.l3_layer.block_mover_clock_ghz = 1.0;
	config.l3_layer.block_mover_buswidth_bits = 512;
	config.l2_layer.bank_groups = { {"l2", {64}, 8} }; // 8 banks of 64KB each in a uniform group
	config.l1_layer.buffer_groups = { {"l1", {64}, 32} }; // 32 buffers of 64B each in a uniform group
    config.compute_tile_count = 2;
    config.dma_engine_count = 2;
    config.streamer_count = 4;
    config.processor_array_rows = 16;
    config.processor_array_cols = 16;
    config.use_systolic_array_mode = true;

    // Create simulator
    sw::kpu::KPUSimulator kpu(config);
	std::cout << kpu.generate_config_report() << "\n";

    std::cout << "KPU created successfully!\n";
    std::cout << "  Using systolic arrays: " << (kpu.is_using_systolic_arrays() ? "Yes" : "No") << "\n";

    if (kpu.is_using_systolic_arrays()) {
        std::cout << "  Systolic array size: "
                  << kpu.get_systolic_array_rows() << "x"
                  << kpu.get_systolic_array_cols() << "\n";
        std::cout << "  Total PEs: " << kpu.get_systolic_array_total_pes() << "\n";
    }

    std::cout << "\n===========================================\n";
    std::cout << " KPU is ready for computation!\n";
    std::cout << "===========================================\n";

    return 0;
}
