#include <vector>
#include <algorithm>
#include <numeric>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/components/dma_engine.hpp>
#include <sw/memory/address_decoder.hpp>
#include <sw/memory/external_memory.hpp>
#include <sw/kpu/components/l3_tile.hpp>

using namespace sw::kpu;
using namespace sw::memory;

/**
 * Test fixture for address-based DMA API
 *
 * Demonstrates industry-standard DMA usage where:
 * - Memory map is configured once during initialization
 * - DMA commands use pure addresses (like Intel IOAT, ARM PL330, AMD SDMA)
 * - Applications are decoupled from physical memory topology
 *
 * DMA supports transfers between: HOST_MEMORY, KPU_MEMORY (external), L3_TILE
 * Note: L2 banks accessed via BlockMover, L1 buffers via Streamers (not DMA)
 */
class AddressBasedDMAFixture {
public:
    // Hardware components
    std::vector<ExternalMemory> host_memory_regions;  // Empty for these tests
    std::vector<ExternalMemory> memory_banks;
    std::vector<L3Tile> l3_tiles;

    // DMA engine and address decoder
    DMAEngine dma_engine;
    AddressDecoder decoder;

    // Memory map addresses (configured like real hardware)
    static constexpr Address EXTERNAL_BANK0_BASE = 0x0000'0000;
    static constexpr Address EXTERNAL_BANK1_BASE = 0x2000'0000;  // 512 MB offset
    static constexpr Address L3_TILE0_BASE       = 0x8000'0000;
    static constexpr Address L3_TILE1_BASE       = 0x8008'0000;  // 512 KB offset

    AddressBasedDMAFixture()
        : dma_engine(0, 1.0, 100.0)  // Engine 0, 1 GHz, 100 GB/s
    {
        // Create hardware components
        memory_banks.emplace_back(512, 100);  // 512 MB capacity
        memory_banks.emplace_back(512, 100);
        l3_tiles.emplace_back(512);           // 512 KB L3 tile
        l3_tiles.emplace_back(512);

        // Configure memory map (done ONCE during initialization)
        // This is the key advantage: applications use addresses, not (type, id) tuples
        decoder.add_region(EXTERNAL_BANK0_BASE, 512 * 1024 * 1024, MemoryType::EXTERNAL, 0, "External Bank 0");
        decoder.add_region(EXTERNAL_BANK1_BASE, 512 * 1024 * 1024, MemoryType::EXTERNAL, 1, "External Bank 1");
        decoder.add_region(L3_TILE0_BASE, 512 * 1024, MemoryType::L3_TILE, 0, "L3 Tile 0");
        decoder.add_region(L3_TILE1_BASE, 512 * 1024, MemoryType::L3_TILE, 1, "L3 Tile 1");

        // Connect address decoder to DMA engine
        dma_engine.set_address_decoder(&decoder);

        // Enable tracing for debugging
        dma_engine.enable_tracing(true);
    }

    // Helper: generate test data pattern
    std::vector<uint8_t> generate_pattern(size_t size, uint8_t start = 0) {
        std::vector<uint8_t> data(size);
        std::iota(data.begin(), data.end(), start);
        return data;
    }

    // Helper: verify data at address
    bool verify_external_data(Address addr, const std::vector<uint8_t>& expected) {
        auto route = decoder.decode(addr);
        if (route.type != MemoryType::EXTERNAL) return false;

        std::vector<uint8_t> actual(expected.size());
        memory_banks[route.id].read(route.offset, actual.data(), expected.size());
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }

    bool verify_l3_data(Address addr, const std::vector<uint8_t>& expected) {
        auto route = decoder.decode(addr);
        if (route.type != MemoryType::L3_TILE) return false;

        std::vector<uint8_t> actual(expected.size());
        l3_tiles[route.id].read(route.offset, actual.data(), expected.size());
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }

    // Helper: write data to address
    void write_external_data(Address addr, const std::vector<uint8_t>& data) {
        auto route = decoder.decode(addr);
        memory_banks[route.id].write(route.offset, data.data(), data.size());
    }

    void write_l3_data(Address addr, const std::vector<uint8_t>& data) {
        auto route = decoder.decode(addr);
        l3_tiles[route.id].write(route.offset, data.data(), data.size());
    }
};

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Basic Transfer", "[dma][address][basic]") {
    const size_t transfer_size = 4096;

    // Source and destination as pure addresses (no type/id needed!)
    Address src_addr = EXTERNAL_BANK0_BASE + 0x1000;
    Address dst_addr = L3_TILE0_BASE;

    // Write test data to source
    auto test_data = generate_pattern(transfer_size, 0xAA);
    write_external_data(src_addr, test_data);

    // DMA transfer using pure addresses (like Intel IOAT, ARM PL330)
    bool complete = false;
    dma_engine.enqueue_transfer(src_addr, dst_addr, transfer_size,
        [&complete]() { complete = true; });

    // Process until complete
    while (!complete) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Verify data integrity
    REQUIRE(verify_l3_data(dst_addr, test_data));
    REQUIRE_FALSE(dma_engine.is_busy());
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Hardware Topology Independent", "[dma][address][portable]") {
    // This test demonstrates the key advantage: code doesn't care about physical topology

    const size_t transfer_size = 2048;

    // Addresses for two different data locations
    Address matrix_a_addr = EXTERNAL_BANK0_BASE + 0x10000;  // In Bank 0
    Address matrix_b_addr = EXTERNAL_BANK1_BASE + 0x20000;  // In Bank 1 (different bank!)

    Address l3_a = L3_TILE0_BASE;
    Address l3_b = L3_TILE0_BASE + transfer_size;

    // Write test data
    auto data_a = generate_pattern(transfer_size, 0x11);
    auto data_b = generate_pattern(transfer_size, 0x22);
    write_external_data(matrix_a_addr, data_a);
    write_external_data(matrix_b_addr, data_b);

    // Transfer both matrices - code is IDENTICAL regardless of which bank they're in!
    // This is the key benefit: applications don't need to know physical layout
    int completions = 0;
    auto callback = [&completions]() { completions++; };

    dma_engine.enqueue_transfer(matrix_a_addr, l3_a, transfer_size, callback);
    dma_engine.enqueue_transfer(matrix_b_addr, l3_b, transfer_size, callback);

    // Process transfers
    while (completions < 2) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Verify both transfers succeeded
    REQUIRE(verify_l3_data(l3_a, data_a));
    REQUIRE(verify_l3_data(l3_b, data_b));
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Memory Map Visualization", "[dma][address][config]") {
    // Print memory map for documentation/debugging
    std::string map = decoder.to_string();

    INFO("Memory Map:\n" << map);

    // Verify map contains expected regions
    REQUIRE(decoder.get_regions().size() == 4);
    REQUIRE(decoder.get_total_mapped_size() > 0);

    // Verify specific address decoding
    auto route0 = decoder.decode(EXTERNAL_BANK0_BASE + 0x1000);
    REQUIRE(route0.type == MemoryType::EXTERNAL);
    REQUIRE(route0.id == 0);
    REQUIRE(route0.offset == 0x1000);

    auto route1 = decoder.decode(EXTERNAL_BANK1_BASE + 0x5000);
    REQUIRE(route1.type == MemoryType::EXTERNAL);
    REQUIRE(route1.id == 1);
    REQUIRE(route1.offset == 0x5000);

    auto route_l3 = decoder.decode(L3_TILE0_BASE + 0x100);
    REQUIRE(route_l3.type == MemoryType::L3_TILE);
    REQUIRE(route_l3.id == 0);
    REQUIRE(route_l3.offset == 0x100);
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Error Handling - Unmapped Address", "[dma][address][error]") {
    const size_t transfer_size = 1024;

    // Try to transfer from an unmapped address
    Address invalid_src = 0x7000'0000;  // Not in any region
    Address valid_dst = L3_TILE0_BASE;

    REQUIRE_THROWS_AS(
        dma_engine.enqueue_transfer(invalid_src, valid_dst, transfer_size),
        std::out_of_range
    );
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Error Handling - Cross-Region Transfer", "[dma][address][error]") {
    // A transfer that starts in one region but extends into unmapped space
    Address src_addr = L3_TILE0_BASE + (512 * 1024) - 512;  // Near end of L3 tile
    Address dst_addr = EXTERNAL_BANK0_BASE;

    size_t oversized = 2048;  // Would extend past end of L3 tile

    REQUIRE_THROWS_AS(
        dma_engine.enqueue_transfer(src_addr, dst_addr, oversized),
        std::out_of_range
    );
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Error Handling - Decoder Not Configured", "[dma][address][error]") {
    // Create a DMA engine without address decoder configured
    DMAEngine unconfigured_dma(1, 1.0, 100.0);

    bool exception_thrown = false;
    try {
        unconfigured_dma.enqueue_transfer(0x1000, 0x2000, 1024);
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        exception_thrown = (msg.find("AddressDecoder not configured") != std::string::npos);
    }
    REQUIRE(exception_thrown);
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Multiple Transfers Across Hierarchy", "[dma][address][hierarchy]") {
    const size_t transfer_size = 1024;

    // Test data movement: External -> L3 -> External (different location)
    Address ext_addr = EXTERNAL_BANK0_BASE + 0x1000;
    Address l3_addr  = L3_TILE0_BASE + 0x100;
    Address ext_addr2 = EXTERNAL_BANK1_BASE + 0x2000;

    // Prepare test data
    auto test_data = generate_pattern(transfer_size, 0x55);
    write_external_data(ext_addr, test_data);

    int completions = 0;
    auto callback = [&completions]() { completions++; };

    // Transfer: External -> L3
    dma_engine.enqueue_transfer(ext_addr, l3_addr, transfer_size, callback);

    // Process first transfer
    while (completions < 1) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Verify data in L3
    REQUIRE(verify_l3_data(l3_addr, test_data));

    // Transfer: L3 -> External (different bank)
    dma_engine.enqueue_transfer(l3_addr, ext_addr2, transfer_size, callback);

    while (completions < 2) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Verify data made it through entire hierarchy
    REQUIRE(verify_external_data(ext_addr2, test_data));
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Bidirectional Transfers", "[dma][address][bidirectional]") {
    // Test transfers in both directions: External <-> L3

    const size_t transfer_size = 2048;
    Address ext_addr = EXTERNAL_BANK0_BASE + 0x1000;
    Address l3_addr = L3_TILE0_BASE;

    auto test_data = generate_pattern(transfer_size, 0x77);
    write_external_data(ext_addr, test_data);

    // External -> L3
    bool complete = false;
    dma_engine.enqueue_transfer(ext_addr, l3_addr, transfer_size,
        [&complete]() { complete = true; });

    while (!complete) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    REQUIRE(verify_l3_data(l3_addr, test_data));

    // Modify data in L3
    auto modified_data = generate_pattern(transfer_size, 0xBB);
    write_l3_data(l3_addr, modified_data);

    // L3 -> External (different location)
    Address ext_addr2 = EXTERNAL_BANK0_BASE + 0x10000;
    complete = false;
    dma_engine.enqueue_transfer(l3_addr, ext_addr2, transfer_size,
        [&complete]() { complete = true; });

    while (!complete) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    REQUIRE(verify_external_data(ext_addr2, modified_data));
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Virtual Memory Simulation", "[dma][address][virtual]") {
    // This test simulates how virtual memory would work with address-based API

    const size_t transfer_size = 4096;

    // Simulate a "virtual address space" where logical addresses map to physical
    struct VirtualMapping {
        Address virtual_addr;
        Address physical_addr;
    };

    // Application uses virtual addresses
    VirtualMapping tensor_a = {0x0000'1000, EXTERNAL_BANK0_BASE + 0x10000};
    VirtualMapping tensor_b = {0x0000'2000, EXTERNAL_BANK1_BASE + 0x20000};

    auto data_a = generate_pattern(transfer_size, 0xAA);
    auto data_b = generate_pattern(transfer_size, 0xBB);

    write_external_data(tensor_a.physical_addr, data_a);
    write_external_data(tensor_b.physical_addr, data_b);

    // Application code uses "virtual addresses" (in real system, MMU translates)
    // For simulation, we translate manually
    auto translate = [](const VirtualMapping& vm) { return vm.physical_addr; };

    Address l3_a = L3_TILE0_BASE;
    Address l3_b = L3_TILE0_BASE + transfer_size;

    int completions = 0;
    auto callback = [&completions]() { completions++; };

    // Application code - uses virtual addresses, doesn't care about physical layout
    dma_engine.enqueue_transfer(translate(tensor_a), l3_a, transfer_size, callback);
    dma_engine.enqueue_transfer(translate(tensor_b), l3_b, transfer_size, callback);

    // If virtual memory remaps tensor_a to a different bank, only the mapping changes
    // The DMA code stays the same! This is the key benefit.

    while (completions < 2) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    REQUIRE(verify_l3_data(l3_a, data_a));
    REQUIRE(verify_l3_data(l3_b, data_b));
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Dynamic Memory Allocation Pattern", "[dma][address][allocation]") {
    // Demonstrates how address-based API enables flexible memory allocation

    // Simulate a simple allocator that uses any available memory
    auto allocate = [this](size_t size) -> Address {
        // In a real system, this would search for free space
        // For this test, we'll use different regions based on size
        if (size <= 128 * 1024) {
            return L3_TILE0_BASE;  // Small allocations in L3
        } else {
            return EXTERNAL_BANK0_BASE;  // Large allocations in external
        }
    };

    // Allocate tensors - allocator chooses optimal location
    size_t small_tensor_size = 16 * 1024;
    size_t large_tensor_size = 256 * 1024;

    Address small_tensor = allocate(small_tensor_size);
    Address large_tensor = allocate(large_tensor_size);

    // Application code doesn't care where allocator placed the data!
    // Just uses addresses
    auto small_data = generate_pattern(small_tensor_size, 0x11);
    auto large_data = generate_pattern(large_tensor_size, 0x22);

    // Write data to allocated locations
    auto small_route = decoder.decode(small_tensor);
    if (small_route.type == MemoryType::L3_TILE) {
        l3_tiles[small_route.id].write(small_route.offset, small_data.data(), small_data.size());
    }

    auto large_route = decoder.decode(large_tensor);
    if (large_route.type == MemoryType::EXTERNAL) {
        memory_banks[large_route.id].write(large_route.offset, large_data.data(), large_data.size());
    }

    // DMA transfers work regardless of where allocator placed data
    // Transfer small tensor from L3 to external
    Address ext_dest = EXTERNAL_BANK0_BASE + 0x100000;

    bool complete = false;
    dma_engine.enqueue_transfer(small_tensor, ext_dest, small_tensor_size,
        [&complete]() { complete = true; });

    while (!complete) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    REQUIRE(verify_external_data(ext_dest, small_data));
}
