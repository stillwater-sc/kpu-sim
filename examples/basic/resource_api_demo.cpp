// Resource API Demo
// Demonstrates the unified resource management interface for the KPU simulator
//
// This example shows how to:
// - Discover and enumerate all hardware resources
// - Allocate memory in different memory tiers
// - Read/write data to memory resources
// - Check resource status and statistics
// - Reset and clear resources

/*
 Resource API Demo (examples/basic/resource_api_demo.cpp)

  The demo demonstrates all the key capabilities of the Resource API:

  | Section            | Functionality                                         |
  |--------------------|-------------------------------------------------------|
  | 1. Configuration   | Configure simulator with custom resource counts/sizes |
  | 2. Discovery       | Enumerate all resource types and their properties     |
  | 3. Allocation      | Allocate memory in different tiers (External, L3, L2) |
  | 4. Read/Write      | Write matrix data to memory and read it back          |
  | 5. Status          | Check resource state, utilization, empty/full status  |
  | 6. System Stats    | View system-wide aggregated statistics                |
  | 7. Reset/Clear     | Reset allocations and clear memory contents           |
  | 8. Address Queries | Find resources by address, validate ranges            |

  Running the Demo

  ./build/examples/basic/example_resource_api_demo

  Key Output Highlights

  - 134 MB total memory across 6 memory tiers
  - Memory hierarchy: HOST_MEMORY → EXTERNAL_MEMORY → L3_TILE → L2_BANK → L1_BUFFER → PAGE_BUFFER
  - Unified address space: Each tier has its own address range
  - Resource tracking: Utilization percentages, allocation info, state monitoring
  - Full observability: Stats for memory, compute, and data movement resources
*/

#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/resource_api.hpp>
#include <sw/kpu/resource_stats.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>

using namespace sw::kpu;

// Helper to format bytes nicely
std::string format_bytes(Size bytes) {
    if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    return std::to_string(bytes) + " B";
}

// Print a separator line
void separator(const std::string& title = "") {
    if (title.empty()) {
        std::cout << std::string(60, '-') << "\n";
    } else {
        std::cout << "\n=== " << title << " " << std::string(55 - title.length(), '=') << "\n";
    }
}

int main() {
    std::cout << "KPU Simulator - Resource API Demo\n";
    separator();

    // =========================================================================
    // 1. Configure and create the simulator
    // =========================================================================
    separator("1. Simulator Configuration");

    KPUSimulator::Config config;

    // Configure memory hierarchy
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = 64;

    config.memory_bank_count = 4;
    config.memory_bank_capacity_mb = 16;

    config.l3_tile_count = 8;
    config.l3_tile_capacity_kb = 512;

    config.l2_bank_count = 16;
    config.l2_bank_capacity_kb = 128;

    config.l1_buffer_count = 32;
    config.l1_buffer_capacity_kb = 16;

    config.scratchpad_count = 8;
    config.scratchpad_capacity_kb = 32;

    // Configure compute and data movement
    config.compute_tile_count = 8;
    config.dma_engine_count = 4;
    config.block_mover_count = 8;
    config.streamer_count = 16;

    KPUSimulator simulator(config);
    auto rm = simulator.create_resource_manager();

    std::cout << "Simulator created with custom configuration.\n";

    // =========================================================================
    // 2. Resource Discovery
    // =========================================================================
    separator("2. Resource Discovery");

    std::cout << std::left;
    std::cout << "\nMemory Resources:\n";
    std::cout << std::setw(20) << "Type"
              << std::setw(8) << "Count"
              << std::setw(15) << "Capacity Each"
              << std::setw(15) << "Total" << "\n";
    std::cout << std::string(58, '-') << "\n";

    auto print_memory_type = [&](ResourceType type) {
        size_t count = rm->get_resource_count(type);
        if (count == 0) return;

        ResourceHandle h = rm->get_resource(type, 0);
        Size total = count * h.capacity;

        std::cout << std::setw(20) << resource_type_name(type)
                  << std::setw(8) << count
                  << std::setw(15) << format_bytes(h.capacity)
                  << std::setw(15) << format_bytes(total) << "\n";
    };

    print_memory_type(ResourceType::HOST_MEMORY);
    print_memory_type(ResourceType::EXTERNAL_MEMORY);
    print_memory_type(ResourceType::L3_TILE);
    print_memory_type(ResourceType::L2_BANK);
    print_memory_type(ResourceType::L1_BUFFER);
    print_memory_type(ResourceType::PAGE_BUFFER);

    std::cout << "\nCompute Resources:\n";
    std::cout << "  Compute Tiles: " << rm->get_resource_count(ResourceType::COMPUTE_TILE) << "\n";

    std::cout << "\nData Movement Resources:\n";
    std::cout << "  DMA Engines:   " << rm->get_resource_count(ResourceType::DMA_ENGINE) << "\n";
    std::cout << "  Block Movers:  " << rm->get_resource_count(ResourceType::BLOCK_MOVER) << "\n";
    std::cout << "  Streamers:     " << rm->get_resource_count(ResourceType::STREAMER) << "\n";

    // =========================================================================
    // 3. Memory Allocation
    // =========================================================================
    separator("3. Memory Allocation");

    // Get handles to different memory resources
    ResourceHandle external_mem = rm->get_resource(ResourceType::EXTERNAL_MEMORY, 0);
    ResourceHandle l3_tile = rm->get_resource(ResourceType::L3_TILE, 0);
    ResourceHandle l2_bank = rm->get_resource(ResourceType::L2_BANK, 0);

    std::cout << "\nAllocating memory in different tiers:\n";

    // Allocate in external memory (for large tensors)
    auto ext_alloc = rm->allocate(external_mem, 1024 * 1024, 64, "tensor_A");  // 1 MB
    if (ext_alloc.has_value()) {
        std::cout << "  External Memory: Allocated 1 MB at address 0x"
                  << std::hex << *ext_alloc << std::dec << "\n";
    }

    // Allocate in L3 (for tile caching)
    auto l3_alloc = rm->allocate(l3_tile, 64 * 1024, 64, "tile_cache");  // 64 KB
    if (l3_alloc.has_value()) {
        std::cout << "  L3 Tile:         Allocated 64 KB at address 0x"
                  << std::hex << *l3_alloc << std::dec << "\n";
    }

    // Allocate in L2 (for working set)
    auto l2_alloc = rm->allocate(l2_bank, 16 * 1024, 64, "working_set");  // 16 KB
    if (l2_alloc.has_value()) {
        std::cout << "  L2 Bank:         Allocated 16 KB at address 0x"
                  << std::hex << *l2_alloc << std::dec << "\n";
    }

    // Show allocation tracking
    std::cout << "\nAllocation tracking:\n";
    std::cout << "  External Memory: " << format_bytes(rm->get_allocated_bytes(external_mem))
              << " / " << format_bytes(external_mem.capacity) << " used\n";
    std::cout << "  L3 Tile:         " << format_bytes(rm->get_allocated_bytes(l3_tile))
              << " / " << format_bytes(l3_tile.capacity) << " used\n";
    std::cout << "  L2 Bank:         " << format_bytes(rm->get_allocated_bytes(l2_bank))
              << " / " << format_bytes(l2_bank.capacity) << " used\n";

    // =========================================================================
    // 4. Memory Read/Write Operations
    // =========================================================================
    separator("4. Memory Read/Write Operations");

    if (l2_alloc.has_value()) {
        Address addr = *l2_alloc;

        // Write a matrix of floats
        std::vector<float> matrix(1024);  // 4 KB of floats
        for (size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] = static_cast<float>(i) * 0.1f;
        }

        std::cout << "\nWriting 4 KB matrix to L2 bank...\n";
        rm->write(addr, matrix.data(), matrix.size() * sizeof(float));

        // Read it back
        std::vector<float> read_matrix(1024);
        rm->read(addr, read_matrix.data(), read_matrix.size() * sizeof(float));

        // Verify
        bool match = true;
        for (size_t i = 0; i < matrix.size(); ++i) {
            if (matrix[i] != read_matrix[i]) {
                match = false;
                break;
            }
        }
        std::cout << "Read back and verified: " << (match ? "SUCCESS" : "FAILED") << "\n";

        // Show first few values
        std::cout << "First 5 values: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << read_matrix[i] << " ";
        }
        std::cout << "...\n";
    }

    // =========================================================================
    // 5. Resource Status and Statistics
    // =========================================================================
    separator("5. Resource Status and Statistics");

    std::cout << "\nMemory Resource Status:\n";
    for (auto type : {ResourceType::EXTERNAL_MEMORY, ResourceType::L3_TILE, ResourceType::L2_BANK}) {
        ResourceHandle h = rm->get_resource(type, 0);
        ResourceStatus status = rm->get_status(h);

        std::cout << "\n  " << resource_type_name(type) << "[0]:\n";
        std::cout << "    State:       " << resource_state_name(status.state) << "\n";
        std::cout << "    Utilization: " << std::fixed << std::setprecision(2)
                  << rm->get_utilization(h) << "%\n";
        std::cout << "    Empty:       " << (rm->is_empty(h) ? "yes" : "no") << "\n";
        std::cout << "    Full:        " << (rm->is_full(h) ? "yes" : "no") << "\n";
    }

    std::cout << "\nCompute Resource Status:\n";
    for (size_t i = 0; i < 2; ++i) {
        ResourceHandle h = rm->get_resource(ResourceType::COMPUTE_TILE, i);
        std::cout << "  compute_tile[" << i << "]: "
                  << resource_state_name(rm->get_state(h)) << "\n";
    }

    std::cout << "\nData Movement Resource Status:\n";
    for (size_t i = 0; i < 2; ++i) {
        ResourceHandle h = rm->get_resource(ResourceType::DMA_ENGINE, i);
        std::cout << "  dma_engine[" << i << "]:  "
                  << resource_state_name(rm->get_state(h))
                  << ", busy=" << (rm->is_busy(h) ? "yes" : "no") << "\n";
    }

    // =========================================================================
    // 6. System-wide Statistics
    // =========================================================================
    separator("6. System-wide Statistics");

    SystemStats sys_stats = rm->get_system_stats();

    std::cout << "\nSystem Totals:\n";
    std::cout << "  Total Memory Capacity:  " << format_bytes(sys_stats.total_memory_capacity) << "\n";
    std::cout << "  Total Memory Allocated: " << format_bytes(sys_stats.total_memory_allocated) << "\n";
    std::cout << "  Total Bytes Read:       " << sys_stats.total_memory_read_bytes << "\n";
    std::cout << "  Total Bytes Written:    " << sys_stats.total_memory_write_bytes << "\n";
    std::cout << "  Total Compute Ops:      " << sys_stats.total_compute_ops << "\n";
    std::cout << "  Total FLOPs:            " << sys_stats.total_flops << "\n";
    std::cout << "  Total Transfers:        " << sys_stats.total_transfers << "\n";
    std::cout << "  Total Bytes Moved:      " << sys_stats.total_bytes_moved << "\n";

    // =========================================================================
    // 7. Reset and Clear Operations
    // =========================================================================
    separator("7. Reset and Clear Operations");

    std::cout << "\nBefore reset:\n";
    std::cout << "  L2 Bank allocated: " << format_bytes(rm->get_allocated_bytes(l2_bank)) << "\n";
    std::cout << "  L2 Bank is_empty:  " << (rm->is_empty(l2_bank) ? "yes" : "no") << "\n";

    // Reset allocations (free all allocations but keep data)
    rm->reset_allocations(l2_bank);

    std::cout << "\nAfter reset_allocations:\n";
    std::cout << "  L2 Bank allocated: " << format_bytes(rm->get_allocated_bytes(l2_bank)) << "\n";
    std::cout << "  L2 Bank is_empty:  " << (rm->is_empty(l2_bank) ? "yes" : "no") << "\n";

    // Full reset (clear memory and allocations)
    rm->reset(external_mem);
    std::cout << "\nAfter full reset on external_memory:\n";
    std::cout << "  External Memory allocated: " << format_bytes(rm->get_allocated_bytes(external_mem)) << "\n";

    // =========================================================================
    // 8. Address Space Queries
    // =========================================================================
    separator("8. Address Space Queries");

    // Allocate fresh memory
    auto fresh_alloc = rm->allocate(ResourceType::L3_TILE, 1024);
    if (fresh_alloc.has_value()) {
        Address addr = *fresh_alloc;

        std::cout << "\nAllocated 1 KB at address 0x" << std::hex << addr << std::dec << "\n";

        // Find which resource contains this address
        ResourceHandle containing = rm->find_resource_for_address(addr);
        std::cout << "Address belongs to: " << containing.to_string() << "\n";

        // Check address validity
        std::cout << "Is valid address:   " << (rm->is_valid_address(addr) ? "yes" : "no") << "\n";
        std::cout << "Is valid range:     " << (rm->is_valid_range(addr, 1024) ? "yes" : "no") << "\n";

        // Get allocation info
        auto info = rm->get_allocation_info(addr);
        if (info.has_value()) {
            std::cout << "\nAllocation info:\n";
            std::cout << "  Address:   0x" << std::hex << info->address << std::dec << "\n";
            std::cout << "  Size:      " << info->size << " bytes\n";
            std::cout << "  Alignment: " << info->alignment << " bytes\n";
            std::cout << "  Resource:  " << info->resource.to_string() << "\n";
        }
    }

    separator();
    std::cout << "\nDemo complete!\n";

    return 0;
}
