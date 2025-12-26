// Comprehensive test suite for all resource types
// Tests resource discovery, read/write, status, reset, and statistics for each resource type

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/resource_api.hpp>
#include <sw/kpu/resource_stats.hpp>

#include <vector>
#include <cstring>
#include <iostream>
#include <iomanip>

using namespace sw::kpu;
using Catch::Approx;

// Test fixture with all resource types configured
class AllResourcesFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> simulator;
    std::unique_ptr<ResourceManager> rm;

    AllResourcesFixture() {
        // Configure all resource types
        config.host_memory_region_count = 1;
        config.host_memory_region_capacity_mb = 16;

        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 8;

        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 256;

        config.l2_bank_count = 8;
        config.l2_bank_capacity_kb = 64;

        config.l1_buffer_count = 16;
        config.l1_buffer_capacity_kb = 8;

        config.page_buffer_count = 4;
        config.page_buffer_capacity_kb = 16;

        config.compute_tile_count = 4;
        config.dma_engine_count = 2;
        config.block_mover_count = 4;
        config.streamer_count = 8;

        simulator = std::make_unique<KPUSimulator>(config);
        rm = simulator->create_resource_manager();
    }
};

// =============================================================================
// Test each resource type's discovery
// =============================================================================

TEST_CASE_METHOD(AllResourcesFixture, "All Resources: Discovery and enumeration", "[all_resources][discovery]") {
    SECTION("Count each resource type") {
        REQUIRE(rm->get_resource_count(ResourceType::HOST_MEMORY) == 1);
        REQUIRE(rm->get_resource_count(ResourceType::EXTERNAL_MEMORY) == 2);
        REQUIRE(rm->get_resource_count(ResourceType::L3_TILE) == 4);
        REQUIRE(rm->get_resource_count(ResourceType::L2_BANK) == 8);
        REQUIRE(rm->get_resource_count(ResourceType::L1_BUFFER) == 16);
        REQUIRE(rm->get_resource_count(ResourceType::PAGE_BUFFER) == 4);
        REQUIRE(rm->get_resource_count(ResourceType::COMPUTE_TILE) == 4);
        REQUIRE(rm->get_resource_count(ResourceType::DMA_ENGINE) == 2);
        REQUIRE(rm->get_resource_count(ResourceType::BLOCK_MOVER) == 4);
        REQUIRE(rm->get_resource_count(ResourceType::STREAMER) == 8);
    }

    SECTION("Get handles for each resource type") {
        // Test each resource type can be accessed by index
        for (size_t i = 0; i < rm->get_resource_count(ResourceType::HOST_MEMORY); ++i) {
            auto h = rm->get_resource(ResourceType::HOST_MEMORY, i);
            REQUIRE(h.is_valid());
            REQUIRE(h.type == ResourceType::HOST_MEMORY);
            REQUIRE(h.id == i);
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::EXTERNAL_MEMORY); ++i) {
            auto h = rm->get_resource(ResourceType::EXTERNAL_MEMORY, i);
            REQUIRE(h.is_valid());
            REQUIRE(h.is_memory());
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::L3_TILE); ++i) {
            auto h = rm->get_resource(ResourceType::L3_TILE, i);
            REQUIRE(h.is_valid());
            REQUIRE(h.capacity > 0);
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::L2_BANK); ++i) {
            auto h = rm->get_resource(ResourceType::L2_BANK, i);
            REQUIRE(h.is_valid());
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::L1_BUFFER); ++i) {
            auto h = rm->get_resource(ResourceType::L1_BUFFER, i);
            REQUIRE(h.is_valid());
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::PAGE_BUFFER); ++i) {
            auto h = rm->get_resource(ResourceType::PAGE_BUFFER, i);
            REQUIRE(h.is_valid());
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::COMPUTE_TILE); ++i) {
            auto h = rm->get_resource(ResourceType::COMPUTE_TILE, i);
            REQUIRE(h.is_valid());
            REQUIRE(h.is_compute());
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::DMA_ENGINE); ++i) {
            auto h = rm->get_resource(ResourceType::DMA_ENGINE, i);
            REQUIRE(h.is_valid());
            REQUIRE(h.is_data_movement());
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::BLOCK_MOVER); ++i) {
            auto h = rm->get_resource(ResourceType::BLOCK_MOVER, i);
            REQUIRE(h.is_valid());
            REQUIRE(h.is_data_movement());
        }

        for (size_t i = 0; i < rm->get_resource_count(ResourceType::STREAMER); ++i) {
            auto h = rm->get_resource(ResourceType::STREAMER, i);
            REQUIRE(h.is_valid());
            REQUIRE(h.is_data_movement());
        }
    }

    SECTION("Aggregate resource lists") {
        auto mem = rm->get_memory_resources();
        REQUIRE(mem.size() == (1 + 2 + 4 + 8 + 16 + 4));  // All memory types

        auto compute = rm->get_compute_resources();
        REQUIRE(compute.size() == 4);

        auto dm = rm->get_data_movement_resources();
        REQUIRE(dm.size() == (2 + 4 + 8));  // DMA + BlockMover + Streamer
    }
}

// =============================================================================
// Test memory resource operations (read, write, clear, reset)
// =============================================================================

TEST_CASE_METHOD(AllResourcesFixture, "All Resources: Memory operations on each type", "[all_resources][memory]") {
    // Test each memory resource type
    std::vector<ResourceType> memory_types = {
        ResourceType::HOST_MEMORY,
        ResourceType::EXTERNAL_MEMORY,
        ResourceType::L3_TILE,
        ResourceType::L2_BANK,
        ResourceType::L1_BUFFER,
        ResourceType::PAGE_BUFFER
    };

    for (auto type : memory_types) {
        DYNAMIC_SECTION("Memory operations for " << resource_type_name(type)) {
            ResourceHandle handle = rm->get_resource(type, 0);
            REQUIRE(handle.is_valid());
            REQUIRE(handle.is_memory());
            REQUIRE(handle.capacity > 0);

            // Test allocation
            Size alloc_size = std::min(handle.capacity / 4, Size(4096));
            auto alloc_result = rm->allocate(handle, alloc_size, 64, "test_" + resource_type_name(type));
            REQUIRE(alloc_result.has_value());
            Address addr = *alloc_result;
            REQUIRE(rm->is_empty(handle) == false);

            // Test write
            std::vector<uint8_t> write_data(alloc_size);
            for (size_t i = 0; i < alloc_size; ++i) {
                write_data[i] = static_cast<uint8_t>(i % 256);
            }
            rm->write(addr, write_data.data(), alloc_size);

            // Test read
            std::vector<uint8_t> read_data(alloc_size);
            rm->read(addr, read_data.data(), alloc_size);
            REQUIRE(read_data == write_data);

            // Test get allocation info
            auto info = rm->get_allocation_info(addr);
            REQUIRE(info.has_value());
            REQUIRE(info->size == alloc_size);

            // Test is_valid_address
            REQUIRE(rm->is_valid_address(addr) == true);
            REQUIRE(rm->is_valid_range(addr, alloc_size) == true);

            // Test utilization
            double util = rm->get_utilization(handle);
            REQUIRE(util > 0.0);

            // Test reset_allocations (keeps data, clears allocator)
            rm->reset_allocations(handle);
            REQUIRE(rm->is_empty(handle) == true);

            // Test clear (zeros memory)
            // First write some data
            auto addr_result = rm->allocate(handle, 256);
            REQUIRE(addr_result.has_value());
            addr = *addr_result;
            std::vector<uint8_t> pattern(256, 0xAB);
            rm->write(addr, pattern.data(), 256);

            rm->clear(handle);

            // Verify memory is zeroed
            std::vector<uint8_t> check(256);
            rm->read(handle.base_address, check.data(), 256);
            for (size_t i = 0; i < 256; ++i) {
                REQUIRE(check[i] == 0);
            }
        }
    }
}

// =============================================================================
// Test resource status for each type
// =============================================================================

TEST_CASE_METHOD(AllResourcesFixture, "All Resources: Status checks", "[all_resources][status]") {
    SECTION("Memory resources start idle") {
        for (auto type : {ResourceType::HOST_MEMORY, ResourceType::EXTERNAL_MEMORY,
                          ResourceType::L3_TILE, ResourceType::L2_BANK,
                          ResourceType::L1_BUFFER, ResourceType::PAGE_BUFFER}) {
            ResourceHandle h = rm->get_resource(type, 0);
            REQUIRE(rm->get_state(h) == ResourceState::IDLE);
            REQUIRE(rm->is_busy(h) == false);
            REQUIRE(rm->is_ready(h) == true);
        }
    }

    SECTION("Compute resources start idle") {
        for (size_t i = 0; i < rm->get_resource_count(ResourceType::COMPUTE_TILE); ++i) {
            ResourceHandle h = rm->get_resource(ResourceType::COMPUTE_TILE, i);
            REQUIRE(rm->get_state(h) == ResourceState::IDLE);
            REQUIRE(rm->is_busy(h) == false);
        }
    }

    SECTION("Data movement resources start idle") {
        for (auto type : {ResourceType::DMA_ENGINE, ResourceType::BLOCK_MOVER, ResourceType::STREAMER}) {
            for (size_t i = 0; i < rm->get_resource_count(type); ++i) {
                ResourceHandle h = rm->get_resource(type, i);
                REQUIRE(rm->get_state(h) == ResourceState::IDLE);
            }
        }
    }

    SECTION("Get comprehensive status") {
        // Memory resource status
        ResourceHandle mem = rm->get_resource(ResourceType::EXTERNAL_MEMORY, 0);
        ResourceStatus mem_status = rm->get_status(mem);
        REQUIRE(mem_status.state == ResourceState::IDLE);
        REQUIRE(mem_status.is_healthy() == true);
        REQUIRE(mem_status.is_available() == true);
        REQUIRE(mem_status.memory_stats.capacity_bytes > 0);

        // Compute resource status
        ResourceHandle compute = rm->get_resource(ResourceType::COMPUTE_TILE, 0);
        ResourceStatus compute_status = rm->get_status(compute);
        REQUIRE(compute_status.state == ResourceState::IDLE);

        // Data movement resource status
        ResourceHandle dma = rm->get_resource(ResourceType::DMA_ENGINE, 0);
        ResourceStatus dma_status = rm->get_status(dma);
        REQUIRE(dma_status.state == ResourceState::IDLE);
    }
}

// =============================================================================
// Test statistics for each resource type
// =============================================================================

TEST_CASE_METHOD(AllResourcesFixture, "All Resources: Statistics", "[all_resources][stats]") {
    SECTION("Memory resource stats") {
        ResourceHandle mem = rm->get_resource(ResourceType::EXTERNAL_MEMORY, 0);

        MemoryResourceStats stats = rm->get_memory_stats(mem);
        REQUIRE(stats.capacity_bytes == config.memory_bank_capacity_mb * 1024 * 1024);
        REQUIRE(stats.allocated_bytes == 0);
        REQUIRE(stats.utilization_percent() == Approx(0.0));

        // Allocate and check stats update
        rm->allocate(mem, 1024);
        stats = rm->get_memory_stats(mem);
        REQUIRE(stats.allocated_bytes >= 1024);
        REQUIRE(stats.utilization_percent() > 0.0);
    }

    SECTION("Compute resource stats") {
        ResourceHandle compute = rm->get_resource(ResourceType::COMPUTE_TILE, 0);
        ComputeResourceStats stats = rm->get_compute_stats(compute);
        REQUIRE(stats.matmul_count == 0);
        REQUIRE(stats.total_flops == 0);
    }

    SECTION("Data movement resource stats") {
        ResourceHandle dma = rm->get_resource(ResourceType::DMA_ENGINE, 0);
        DataMovementStats stats = rm->get_data_movement_stats(dma);
        REQUIRE(stats.transfer_count == 0);
        REQUIRE(stats.bytes_transferred == 0);
    }

    SECTION("System-wide stats") {
        SystemStats sys = rm->get_system_stats();
        REQUIRE(sys.total_memory_capacity > 0);

        // Allocate in multiple resources
        rm->allocate(ResourceType::EXTERNAL_MEMORY, 1024);
        rm->allocate(ResourceType::L3_TILE, 512);

        sys = rm->get_system_stats();
        REQUIRE(sys.total_memory_allocated >= 1536);
    }

    SECTION("Reset stats") {
        ResourceHandle mem = rm->get_resource(ResourceType::EXTERNAL_MEMORY, 0);
        rm->allocate(mem, 1024);

        // Stats should show allocation
        MemoryResourceStats before = rm->get_memory_stats(mem);
        REQUIRE(before.allocated_bytes > 0);

        // Reset stats
        rm->reset_stats(mem);

        // Counters should be reset (but allocation remains)
        MemoryResourceStats after = rm->get_memory_stats(mem);
        REQUIRE(after.read_count == 0);
        REQUIRE(after.write_count == 0);
        // Note: allocated_bytes is not reset, only counters
    }

    SECTION("Reset all stats") {
        rm->allocate(ResourceType::EXTERNAL_MEMORY, 1024);
        rm->allocate(ResourceType::L3_TILE, 512);

        rm->reset_all_stats();

        // All counters should be reset
        for (auto type : {ResourceType::EXTERNAL_MEMORY, ResourceType::L3_TILE}) {
            ResourceHandle h = rm->get_resource(type, 0);
            MemoryResourceStats stats = rm->get_memory_stats(h);
            REQUIRE(stats.read_count == 0);
            REQUIRE(stats.write_count == 0);
        }
    }
}

// =============================================================================
// Test empty/full checks
// =============================================================================

TEST_CASE_METHOD(AllResourcesFixture, "All Resources: Empty and full checks", "[all_resources][capacity]") {
    SECTION("is_empty on fresh resource") {
        for (auto type : {ResourceType::HOST_MEMORY, ResourceType::EXTERNAL_MEMORY,
                          ResourceType::L3_TILE, ResourceType::L2_BANK,
                          ResourceType::L1_BUFFER, ResourceType::PAGE_BUFFER}) {
            ResourceHandle h = rm->get_resource(type, 0);
            REQUIRE(rm->is_empty(h) == true);
            REQUIRE(rm->is_full(h) == false);
        }
    }

    SECTION("is_full after filling") {
        // Use a small resource (PAGE_BUFFER)
        ResourceHandle pb = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        // Keep allocating until full
        while (!rm->is_full(pb)) {
            auto addr = rm->allocate(pb, 1024);
            if (!addr.has_value()) break;  // Allocation failed
        }

        REQUIRE(rm->is_empty(pb) == false);
    }
}

// =============================================================================
// Print resource inventory (informational)
// =============================================================================

TEST_CASE_METHOD(AllResourcesFixture, "All Resources: Print inventory", "[all_resources][inventory]") {
    std::cout << "\n=== KPU Resource Inventory ===" << std::endl;
    std::cout << std::left;

    // Memory resources
    std::cout << "\nMemory Resources:" << std::endl;
    std::cout << std::setw(20) << "Type" << std::setw(8) << "Count"
              << std::setw(15) << "Capacity Each" << std::setw(15) << "Total" << std::endl;
    std::cout << std::string(58, '-') << std::endl;

    auto print_memory = [&](ResourceType type) {
        size_t count = rm->get_resource_count(type);
        if (count == 0) return;

        ResourceHandle h = rm->get_resource(type, 0);
        Size total = count * h.capacity;

        std::cout << std::setw(20) << resource_type_name(type)
                  << std::setw(8) << count;

        if (h.capacity >= 1024*1024) {
            std::cout << std::setw(15) << (std::to_string(h.capacity/(1024*1024)) + " MB");
            std::cout << std::setw(15) << (std::to_string(total/(1024*1024)) + " MB");
        } else {
            std::cout << std::setw(15) << (std::to_string(h.capacity/1024) + " KB");
            std::cout << std::setw(15) << (std::to_string(total/1024) + " KB");
        }
        std::cout << std::endl;
    };

    print_memory(ResourceType::HOST_MEMORY);
    print_memory(ResourceType::EXTERNAL_MEMORY);
    print_memory(ResourceType::L3_TILE);
    print_memory(ResourceType::L2_BANK);
    print_memory(ResourceType::L1_BUFFER);
    print_memory(ResourceType::PAGE_BUFFER);

    // Compute resources
    std::cout << "\nCompute Resources:" << std::endl;
    std::cout << std::setw(20) << "Type" << std::setw(8) << "Count" << std::endl;
    std::cout << std::string(28, '-') << std::endl;
    std::cout << std::setw(20) << "compute_tile"
              << std::setw(8) << rm->get_resource_count(ResourceType::COMPUTE_TILE) << std::endl;

    // Data movement resources
    std::cout << "\nData Movement Resources:" << std::endl;
    std::cout << std::setw(20) << "Type" << std::setw(8) << "Count" << std::endl;
    std::cout << std::string(28, '-') << std::endl;
    std::cout << std::setw(20) << "dma_engine"
              << std::setw(8) << rm->get_resource_count(ResourceType::DMA_ENGINE) << std::endl;
    std::cout << std::setw(20) << "block_mover"
              << std::setw(8) << rm->get_resource_count(ResourceType::BLOCK_MOVER) << std::endl;
    std::cout << std::setw(20) << "streamer"
              << std::setw(8) << rm->get_resource_count(ResourceType::STREAMER) << std::endl;

    // System totals
    SystemStats sys = rm->get_system_stats();
    std::cout << "\nSystem Totals:" << std::endl;
    std::cout << "  Total memory capacity: " << (sys.total_memory_capacity / (1024*1024)) << " MB" << std::endl;
    std::cout << "  Total compute tiles: " << rm->get_resource_count(ResourceType::COMPUTE_TILE) << std::endl;
    std::cout << "  Total data movers: " << rm->get_data_movement_resources().size() << std::endl;

    REQUIRE(true);  // This test is for output only
}
