// Test suite for Resource API
// Tests ResourceHandle, ResourceManager, and resource operations

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/resource_api.hpp>
#include <sw/kpu/allocator.hpp>

#include <vector>
#include <cstring>

using namespace sw::kpu;

TEST_CASE("ResourceType classification", "[resource_api]") {
    SECTION("Memory resources") {
        REQUIRE(is_memory_resource(ResourceType::HOST_MEMORY) == true);
        REQUIRE(is_memory_resource(ResourceType::EXTERNAL_MEMORY) == true);
        REQUIRE(is_memory_resource(ResourceType::L3_TILE) == true);
        REQUIRE(is_memory_resource(ResourceType::L2_BANK) == true);
        REQUIRE(is_memory_resource(ResourceType::L1_BUFFER) == true);
        REQUIRE(is_memory_resource(ResourceType::PAGE_BUFFER) == true);
        REQUIRE(is_memory_resource(ResourceType::COMPUTE_TILE) == false);
        REQUIRE(is_memory_resource(ResourceType::DMA_ENGINE) == false);
    }

    SECTION("Compute resources") {
        REQUIRE(is_compute_resource(ResourceType::COMPUTE_TILE) == true);
        REQUIRE(is_compute_resource(ResourceType::HOST_MEMORY) == false);
        REQUIRE(is_compute_resource(ResourceType::DMA_ENGINE) == false);
    }

    SECTION("Data movement resources") {
        REQUIRE(is_data_movement_resource(ResourceType::DMA_ENGINE) == true);
        REQUIRE(is_data_movement_resource(ResourceType::BLOCK_MOVER) == true);
        REQUIRE(is_data_movement_resource(ResourceType::STREAMER) == true);
        REQUIRE(is_data_movement_resource(ResourceType::COMPUTE_TILE) == false);
        REQUIRE(is_data_movement_resource(ResourceType::HOST_MEMORY) == false);
    }
}

TEST_CASE("ResourceHandle validity", "[resource_api]") {
    SECTION("Default handle is invalid") {
        ResourceHandle handle;
        REQUIRE(handle.is_valid() == false);
    }

    SECTION("Constructed handle is valid") {
        ResourceHandle handle(ResourceType::EXTERNAL_MEMORY, 0, 0x1000, 1024);
        REQUIRE(handle.is_valid() == true);
        REQUIRE(handle.type == ResourceType::EXTERNAL_MEMORY);
        REQUIRE(handle.id == 0);
        REQUIRE(handle.base_address == 0x1000);
        REQUIRE(handle.capacity == 1024);
    }

    SECTION("Handle classification methods") {
        ResourceHandle mem_handle(ResourceType::EXTERNAL_MEMORY, 0);
        REQUIRE(mem_handle.is_memory() == true);
        REQUIRE(mem_handle.is_compute() == false);
        REQUIRE(mem_handle.is_data_movement() == false);

        ResourceHandle compute_handle(ResourceType::COMPUTE_TILE, 0);
        REQUIRE(compute_handle.is_memory() == false);
        REQUIRE(compute_handle.is_compute() == true);
        REQUIRE(compute_handle.is_data_movement() == false);

        ResourceHandle dma_handle(ResourceType::DMA_ENGINE, 0);
        REQUIRE(dma_handle.is_memory() == false);
        REQUIRE(dma_handle.is_compute() == false);
        REQUIRE(dma_handle.is_data_movement() == true);
    }

    SECTION("Handle equality") {
        ResourceHandle h1(ResourceType::EXTERNAL_MEMORY, 0);
        ResourceHandle h2(ResourceType::EXTERNAL_MEMORY, 0);
        ResourceHandle h3(ResourceType::EXTERNAL_MEMORY, 1);
        ResourceHandle h4(ResourceType::L3_TILE, 0);

        REQUIRE(h1 == h2);
        REQUIRE(h1 != h3);
        REQUIRE(h1 != h4);
    }

    SECTION("Handle to_string") {
        ResourceHandle handle(ResourceType::EXTERNAL_MEMORY, 2);
        REQUIRE(handle.to_string() == "external_memory[2]");
    }
}

TEST_CASE("ResourceManager resource discovery", "[resource_api]") {
    // Create a simulator with known configuration
    KPUSimulator::Config config;
    config.host_memory_region_count = 1;
    config.memory_bank_count = 2;
    config.l3_tile_count = 4;
    config.l2_bank_count = 8;
    config.l1_buffer_count = 4;
    config.scratchpad_count = 2;
    config.compute_tile_count = 2;
    config.dma_engine_count = 4;
    config.block_mover_count = 4;
    config.streamer_count = 8;

    KPUSimulator simulator(config);
    auto rm = simulator.create_resource_manager();

    SECTION("Resource counts") {
        REQUIRE(rm->get_resource_count(ResourceType::HOST_MEMORY) == 1);
        REQUIRE(rm->get_resource_count(ResourceType::EXTERNAL_MEMORY) == 2);
        REQUIRE(rm->get_resource_count(ResourceType::L3_TILE) == 4);
        REQUIRE(rm->get_resource_count(ResourceType::L2_BANK) == 8);
        REQUIRE(rm->get_resource_count(ResourceType::L1_BUFFER) == 4);
        REQUIRE(rm->get_resource_count(ResourceType::PAGE_BUFFER) == 2);
        REQUIRE(rm->get_resource_count(ResourceType::COMPUTE_TILE) == 2);
        REQUIRE(rm->get_resource_count(ResourceType::DMA_ENGINE) == 4);
        REQUIRE(rm->get_resource_count(ResourceType::BLOCK_MOVER) == 4);
        REQUIRE(rm->get_resource_count(ResourceType::STREAMER) == 8);
    }

    SECTION("Get individual resource") {
        ResourceHandle handle = rm->get_resource(ResourceType::EXTERNAL_MEMORY, 0);
        REQUIRE(handle.is_valid() == true);
        REQUIRE(handle.type == ResourceType::EXTERNAL_MEMORY);
        REQUIRE(handle.id == 0);
        REQUIRE(handle.capacity > 0);
    }

    SECTION("Get resource out of range throws") {
        REQUIRE_THROWS_AS(rm->get_resource(ResourceType::EXTERNAL_MEMORY, 100),
                          std::out_of_range);
    }

    SECTION("Get all resources of type") {
        auto tiles = rm->get_all_resources(ResourceType::L3_TILE);
        REQUIRE(tiles.size() == 4);
        for (size_t i = 0; i < tiles.size(); ++i) {
            REQUIRE(tiles[i].type == ResourceType::L3_TILE);
            REQUIRE(tiles[i].id == i);
        }
    }

    SECTION("Get memory resources") {
        auto mem_resources = rm->get_memory_resources();
        // Should include all memory resource types
        REQUIRE(mem_resources.size() == (1 + 2 + 4 + 8 + 4 + 2));  // host + external + l3 + l2 + l1 + scratch
    }

    SECTION("Get compute resources") {
        auto compute_resources = rm->get_compute_resources();
        REQUIRE(compute_resources.size() == 2);
    }

    SECTION("Get data movement resources") {
        auto dm_resources = rm->get_data_movement_resources();
        REQUIRE(dm_resources.size() == (4 + 4 + 8));  // dma + block_mover + streamer
    }
}

TEST_CASE("ResourceManager memory allocation", "[resource_api]") {
    KPUSimulator::Config config;
    config.memory_bank_count = 1;
    config.memory_bank_capacity_mb = 1;  // 1 MB
    config.scratchpad_count = 1;
    config.scratchpad_capacity_kb = 64;  // 64 KB

    KPUSimulator simulator(config);
    auto rm = simulator.create_resource_manager();

    SECTION("Allocate in specific resource") {
        ResourceHandle handle = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        Address addr = rm->allocate(handle, 1024, 64, "test_alloc");
        REQUIRE(addr != 0);
        REQUIRE((addr % 64) == 0);  // Check alignment
    }

    SECTION("Allocate with default alignment") {
        ResourceHandle handle = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        Address addr = rm->allocate(handle, 256);
        REQUIRE(addr != 0);
        REQUIRE((addr % 64) == 0);  // Default alignment is 64
    }

    SECTION("Allocate zero size returns zero") {
        ResourceHandle handle = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        Address addr = rm->allocate(handle, 0);
        REQUIRE(addr == 0);
    }

    SECTION("Allocate by resource type") {
        Address addr = rm->allocate(ResourceType::PAGE_BUFFER, 512, 64, "by_type");
        REQUIRE(addr != 0);
    }

    SECTION("Track allocations") {
        ResourceHandle handle = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        Size initial_allocated = rm->get_allocated_bytes(handle);

        rm->allocate(handle, 1024);

        Size after_alloc = rm->get_allocated_bytes(handle);
        REQUIRE(after_alloc >= initial_allocated + 1024);
    }

    SECTION("Get allocation info") {
        ResourceHandle handle = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        Address addr = rm->allocate(handle, 2048, 128, "labeled_alloc");

        auto info = rm->get_allocation_info(addr);
        REQUIRE(info.has_value());
        REQUIRE(info->address == addr);
        REQUIRE(info->size == 2048);
        REQUIRE(info->alignment == 128);
        REQUIRE(info->label == "labeled_alloc");
    }

    SECTION("Invalid alignment throws") {
        ResourceHandle handle = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        REQUIRE_THROWS_AS(rm->allocate(handle, 1024, 3), std::invalid_argument);  // Not power of 2
        REQUIRE_THROWS_AS(rm->allocate(handle, 1024, 0), std::invalid_argument);  // Zero alignment
    }

    SECTION("Non-memory resource throws") {
        ResourceHandle handle = rm->get_resource(ResourceType::COMPUTE_TILE, 0);

        REQUIRE_THROWS_AS(rm->allocate(handle, 1024), std::invalid_argument);
    }
}

TEST_CASE("ResourceManager memory operations", "[resource_api]") {
    KPUSimulator::Config config;
    config.scratchpad_count = 1;
    config.scratchpad_capacity_kb = 64;

    KPUSimulator simulator(config);
    auto rm = simulator.create_resource_manager();

    ResourceHandle scratch = rm->get_resource(ResourceType::PAGE_BUFFER, 0);
    Address addr = rm->allocate(scratch, 1024);
    REQUIRE(addr != 0);

    SECTION("Write and read back data") {
        std::vector<float> write_data = {1.0f, 2.0f, 3.0f, 4.0f};
        rm->write(addr, write_data.data(), write_data.size() * sizeof(float));

        std::vector<float> read_data(4);
        rm->read(addr, read_data.data(), read_data.size() * sizeof(float));

        REQUIRE(read_data == write_data);
    }

    SECTION("Memset operation") {
        rm->memset(addr, 0xAB, 256);

        std::vector<uint8_t> read_data(256);
        rm->read(addr, read_data.data(), 256);

        for (size_t i = 0; i < 256; ++i) {
            REQUIRE(read_data[i] == 0xAB);
        }
    }

    SECTION("Copy operation") {
        // Write source data
        std::vector<uint8_t> source_data(256);
        for (size_t i = 0; i < 256; ++i) {
            source_data[i] = static_cast<uint8_t>(i);
        }
        rm->write(addr, source_data.data(), 256);

        // Allocate destination and copy
        Address dst = rm->allocate(scratch, 256);
        rm->copy(addr, dst, 256);

        // Read back and verify
        std::vector<uint8_t> read_data(256);
        rm->read(dst, read_data.data(), 256);

        REQUIRE(read_data == source_data);
    }
}

TEST_CASE("ResourceManager address space queries", "[resource_api]") {
    KPUSimulator::Config config;
    config.memory_bank_count = 2;
    config.memory_bank_capacity_mb = 1;
    config.scratchpad_count = 2;
    config.scratchpad_capacity_kb = 64;

    KPUSimulator simulator(config);
    auto rm = simulator.create_resource_manager();

    SECTION("Find resource for address") {
        // Get known addresses
        ResourceHandle bank0 = rm->get_resource(ResourceType::EXTERNAL_MEMORY, 0);
        ResourceHandle bank1 = rm->get_resource(ResourceType::EXTERNAL_MEMORY, 1);

        // Test finding resources
        ResourceHandle found = rm->find_resource_for_address(bank0.base_address + 100);
        REQUIRE(found.is_valid());
        REQUIRE(found.type == ResourceType::EXTERNAL_MEMORY);
        REQUIRE(found.id == 0);

        found = rm->find_resource_for_address(bank1.base_address + 100);
        REQUIRE(found.is_valid());
        REQUIRE(found.type == ResourceType::EXTERNAL_MEMORY);
        REQUIRE(found.id == 1);
    }

    SECTION("Invalid address returns invalid handle") {
        ResourceHandle found = rm->find_resource_for_address(0xFFFFFFFFFFFFFFFF);
        REQUIRE(found.is_valid() == false);
    }

    SECTION("Valid address check") {
        ResourceHandle scratch = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        REQUIRE(rm->is_valid_address(scratch.base_address) == true);
        REQUIRE(rm->is_valid_address(scratch.base_address + 100) == true);
        REQUIRE(rm->is_valid_address(0xFFFFFFFFFFFFFFFF) == false);
    }

    SECTION("Valid range check") {
        ResourceHandle scratch = rm->get_resource(ResourceType::PAGE_BUFFER, 0);

        // Valid range within resource
        REQUIRE(rm->is_valid_range(scratch.base_address, 1024) == true);

        // Zero size is valid
        REQUIRE(rm->is_valid_range(scratch.base_address, 0) == true);

        // Range extending past resource boundary
        REQUIRE(rm->is_valid_range(scratch.base_address, scratch.capacity + 1) == false);
    }
}

TEST_CASE("ResourceManager resource status", "[resource_api]") {
    KPUSimulator::Config config;
    config.compute_tile_count = 1;
    config.dma_engine_count = 1;

    KPUSimulator simulator(config);
    auto rm = simulator.create_resource_manager();

    SECTION("Initial state is not busy") {
        ResourceHandle compute = rm->get_resource(ResourceType::COMPUTE_TILE, 0);
        REQUIRE(rm->is_busy(compute) == false);
        REQUIRE(rm->is_ready(compute) == true);

        ResourceHandle dma = rm->get_resource(ResourceType::DMA_ENGINE, 0);
        REQUIRE(rm->is_busy(dma) == false);
    }
}

TEST_CASE("BumpAllocator basic operations", "[allocator]") {
    BumpAllocator allocator(0x1000, 4096);  // 4KB starting at 0x1000

    SECTION("Basic allocation") {
        Address addr = allocator.allocate(256);
        REQUIRE(addr == 0x1000);  // First allocation at base

        Address addr2 = allocator.allocate(256);
        REQUIRE(addr2 == 0x1100);  // Next allocation follows
    }

    SECTION("Aligned allocation") {
        Address addr = allocator.allocate(10, 64);
        REQUIRE((addr % 64) == 0);

        Address addr2 = allocator.allocate(10, 128);
        REQUIRE((addr2 % 128) == 0);
    }

    SECTION("Out of memory returns zero") {
        Address addr = allocator.allocate(5000);  // More than capacity
        REQUIRE(addr == 0);
    }

    SECTION("Reset frees all memory") {
        allocator.allocate(1024);
        allocator.allocate(1024);

        Size used_before = allocator.get_used_bytes();
        REQUIRE(used_before >= 2048);

        allocator.reset();

        Size used_after = allocator.get_used_bytes();
        REQUIRE(used_after == 0);
    }

    SECTION("Peak usage tracking") {
        allocator.allocate(1024);
        REQUIRE(allocator.get_peak_usage() >= 1024);

        allocator.allocate(512);
        REQUIRE(allocator.get_peak_usage() >= 1536);

        allocator.reset();
        REQUIRE(allocator.get_peak_usage() >= 1536);  // Peak is preserved
    }

    SECTION("Contains check") {
        REQUIRE(allocator.contains(0x1000) == true);
        REQUIRE(allocator.contains(0x1FFF) == true);
        REQUIRE(allocator.contains(0x0FFF) == false);
        REQUIRE(allocator.contains(0x2000) == false);
    }
}

TEST_CASE("TrackingAllocator operations", "[allocator]") {
    TrackingAllocator allocator(0x2000, 8192);  // 8KB starting at 0x2000

    SECTION("Allocate and deallocate") {
        Address addr = allocator.allocate(256, 64, "test1");
        REQUIRE(addr != 0);

        bool result = allocator.deallocate(addr);
        REQUIRE(result == true);
    }

    SECTION("Get allocation info") {
        Address addr = allocator.allocate(512, 128, "my_alloc");

        auto info = allocator.get_allocation(addr);
        REQUIRE(info.has_value());
        REQUIRE(info->address == addr);
        REQUIRE(info->size == 512);
        REQUIRE(info->alignment == 128);
        REQUIRE(info->label == "my_alloc");
    }

    SECTION("Deallocate invalid address returns false") {
        bool result = allocator.deallocate(0xDEADBEEF);
        REQUIRE(result == false);
    }

    SECTION("Get all allocations") {
        allocator.allocate(100, 64, "a");
        allocator.allocate(200, 64, "b");
        allocator.allocate(300, 64, "c");

        auto allocs = allocator.get_all_allocations();
        REQUIRE(allocs.size() == 3);
    }

    SECTION("Allocation tracking after deallocation") {
        Address addr = allocator.allocate(256);
        Size before = allocator.get_allocated_bytes();

        allocator.deallocate(addr);
        Size after = allocator.get_allocated_bytes();

        REQUIRE(after < before);
    }
}

TEST_CASE("PoolAllocator operations", "[allocator]") {
    PoolAllocator pool(0x3000, 64, 10);  // 10 blocks of 64 bytes each

    SECTION("Basic allocation") {
        Address addr = pool.allocate();
        REQUIRE(addr != 0);  // Allocation succeeded
        REQUIRE(addr >= 0x3000);  // Within pool range
        REQUIRE(addr < 0x3000 + 64 * 10);  // Within pool range

        Address addr2 = pool.allocate();
        REQUIRE(addr2 != 0);  // Second allocation succeeded
        REQUIRE(addr2 != addr);  // Different from first
    }

    SECTION("Deallocate and reuse") {
        Address addr1 = pool.allocate();
        Address addr2 = pool.allocate();

        REQUIRE(pool.get_allocated_count() == 2);
        REQUIRE(pool.get_free_count() == 8);

        pool.deallocate(addr1);

        REQUIRE(pool.get_allocated_count() == 1);
        REQUIRE(pool.get_free_count() == 9);

        // Next allocation should reuse freed block
        Address addr3 = pool.allocate();
        REQUIRE(addr3 == addr1);
    }

    SECTION("Pool exhaustion") {
        for (int i = 0; i < 10; ++i) {
            Address addr = pool.allocate();
            REQUIRE(addr != 0);
        }

        // Pool is now full
        Address addr = pool.allocate();
        REQUIRE(addr == 0);
    }

    SECTION("Invalid deallocation") {
        bool result = pool.deallocate(0x4000);  // Outside pool
        REQUIRE(result == false);

        result = pool.deallocate(0x3010);  // Inside pool but misaligned
        REQUIRE(result == false);
    }

    SECTION("Reset pool") {
        pool.allocate();
        pool.allocate();
        pool.allocate();

        REQUIRE(pool.get_allocated_count() == 3);

        pool.reset();

        REQUIRE(pool.get_allocated_count() == 0);
        REQUIRE(pool.get_free_count() == 10);
    }
}
