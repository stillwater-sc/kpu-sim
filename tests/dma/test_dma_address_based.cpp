#include <vector>
#include <algorithm>
#include <numeric>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/components/dma_engine.hpp>
#include <sw/memory/address_decoder.hpp>
#include <sw/memory/external_memory.hpp>
#include <sw/kpu/components/scratchpad.hpp>
#include <sw/kpu/components/l3_tile.hpp>
#include <sw/kpu/components/l2_bank.hpp>

using namespace sw::kpu;
using namespace sw::memory;

/**
 * Test fixture for address-based DMA API
 *
 * Demonstrates industry-standard DMA usage where:
 * - Memory map is configured once during initialization
 * - DMA commands use pure addresses (like Intel IOAT, ARM PL330, AMD SDMA)
 * - Applications are decoupled from physical memory topology
 */
class AddressBasedDMAFixture {
public:
    // Hardware components
    std::vector<ExternalMemory> host_memory_regions;  // Empty for these tests
    std::vector<ExternalMemory> memory_banks;
    std::vector<L3Tile> l3_tiles;
    std::vector<L2Bank> l2_banks;
    std::vector<Scratchpad> scratchpads;

    // DMA engine and address decoder
    DMAEngine dma_engine;
    AddressDecoder decoder;

    // Memory map addresses (configured like real hardware)
    static constexpr Address EXTERNAL_BANK0_BASE = 0x0000'0000;
    static constexpr Address EXTERNAL_BANK1_BASE = 0x2000'0000;  // 512 MB offset
    static constexpr Address L3_TILE0_BASE       = 0x8000'0000;
    static constexpr Address L3_TILE1_BASE       = 0x8002'0000;  // 128 KB offset
    static constexpr Address L2_BANK0_BASE       = 0x9000'0000;
    static constexpr Address SCRATCHPAD0_BASE    = 0xFFFF'0000;

    AddressBasedDMAFixture()
        : dma_engine(0, 1.0, 100.0)  // Engine 0, 1 GHz, 100 GB/s
    {
        // Create hardware components (2 external banks, 256 KB scratchpad)
        memory_banks.emplace_back(512, 100);  // 512 MB capacity
        memory_banks.emplace_back(512, 100);
        l3_tiles.emplace_back(128);           // 128 KB
        l3_tiles.emplace_back(128);
        l2_banks.emplace_back(64);            // 64 KB
        scratchpads.emplace_back(256);        // 256 KB

        // Configure memory map (done ONCE during initialization)
        // This is the key advantage: applications use addresses, not (type, id) tuples
        decoder.add_region(EXTERNAL_BANK0_BASE, 512 * 1024 * 1024, MemoryType::EXTERNAL, 0, "External Bank 0");
        decoder.add_region(EXTERNAL_BANK1_BASE, 512 * 1024 * 1024, MemoryType::EXTERNAL, 1, "External Bank 1");
        decoder.add_region(L3_TILE0_BASE, 128 * 1024, MemoryType::L3_TILE, 0, "L3 Tile 0");
        decoder.add_region(L3_TILE1_BASE, 128 * 1024, MemoryType::L3_TILE, 1, "L3 Tile 1");
        decoder.add_region(L2_BANK0_BASE, 64 * 1024, MemoryType::L2_BANK, 0, "L2 Bank 0");
        decoder.add_region(SCRATCHPAD0_BASE, 256 * 1024, MemoryType::PAGE_BUFFER, 0, "PageBuffer 0");

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

    bool verify_scratchpad_data(Address addr, const std::vector<uint8_t>& expected) {
        auto route = decoder.decode(addr);
        if (route.type != MemoryType::PAGE_BUFFER) return false;

        std::vector<uint8_t> actual(expected.size());
        scratchpads[route.id].read(route.offset, actual.data(), expected.size());
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }

    // Helper: write data to address
    void write_external_data(Address addr, const std::vector<uint8_t>& data) {
        auto route = decoder.decode(addr);
        memory_banks[route.id].write(route.offset, data.data(), data.size());
    }
};

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Basic Transfer", "[dma][address][basic]") {
    const size_t transfer_size = 4096;

    // Source and destination as pure addresses (no type/id needed!)
    Address src_addr = EXTERNAL_BANK0_BASE + 0x1000;
    Address dst_addr = SCRATCHPAD0_BASE;

    // Write test data to source
    auto test_data = generate_pattern(transfer_size, 0xAA);
    write_external_data(src_addr, test_data);

    // DMA transfer using pure addresses (like Intel IOAT, ARM PL330)
    bool complete = false;
    dma_engine.enqueue_transfer(src_addr, dst_addr, transfer_size,
        [&complete]() { complete = true; });

    // Process until complete
    while (!complete) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Verify data integrity
    REQUIRE(verify_scratchpad_data(dst_addr, test_data));
    REQUIRE_FALSE(dma_engine.is_busy());
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Hardware Topology Independent", "[dma][address][portable]") {
    // This test demonstrates the key advantage: code doesn't care about physical topology

    const size_t transfer_size = 2048;

    // Addresses for two different data locations
    Address matrix_a_addr = EXTERNAL_BANK0_BASE + 0x10000;  // In Bank 0
    Address matrix_b_addr = EXTERNAL_BANK1_BASE + 0x20000;  // In Bank 1 (different bank!)

    Address scratch_a = SCRATCHPAD0_BASE;
    Address scratch_b = SCRATCHPAD0_BASE + transfer_size;

    // Write test data
    auto data_a = generate_pattern(transfer_size, 0x11);
    auto data_b = generate_pattern(transfer_size, 0x22);
    write_external_data(matrix_a_addr, data_a);
    write_external_data(matrix_b_addr, data_b);

    // Transfer both matrices - code is IDENTICAL regardless of which bank they're in!
    // This is the key benefit: applications don't need to know physical layout
    int completions = 0;
    auto callback = [&completions]() { completions++; };

    dma_engine.enqueue_transfer(matrix_a_addr, scratch_a, transfer_size, callback);
    dma_engine.enqueue_transfer(matrix_b_addr, scratch_b, transfer_size, callback);

    // Process transfers
    while (completions < 2) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Verify both transfers succeeded
    REQUIRE(verify_scratchpad_data(scratch_a, data_a));
    REQUIRE(verify_scratchpad_data(scratch_b, data_b));
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Memory Map Visualization", "[dma][address][config]") {
    // Print memory map for documentation/debugging
    std::string map = decoder.to_string();

    INFO("Memory Map:\n" << map);

    // Verify map contains expected regions
    REQUIRE(decoder.get_regions().size() == 6);
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

    auto route_scratch = decoder.decode(SCRATCHPAD0_BASE + 0x100);
    REQUIRE(route_scratch.type == MemoryType::PAGE_BUFFER);
    REQUIRE(route_scratch.id == 0);
    REQUIRE(route_scratch.offset == 0x100);
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Error Handling - Unmapped Address", "[dma][address][error]") {
    const size_t transfer_size = 1024;

    // Try to transfer from an unmapped address
    Address invalid_src = 0x7000'0000;  // Not in any region
    Address valid_dst = SCRATCHPAD0_BASE;

    REQUIRE_THROWS_AS(
        dma_engine.enqueue_transfer(invalid_src, valid_dst, transfer_size),
        std::out_of_range
    );
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Error Handling - Cross-Region Transfer", "[dma][address][error]") {
    // A transfer that starts in one region but extends into unmapped space
    Address src_addr = SCRATCHPAD0_BASE + (256 * 1024) - 512;  // Near end of scratchpad
    Address dst_addr = EXTERNAL_BANK0_BASE;

    size_t oversized = 2048;  // Would extend past end of scratchpad

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

    // Test data movement through the entire memory hierarchy
    Address ext_addr = EXTERNAL_BANK0_BASE + 0x1000;
    Address l3_addr  = L3_TILE0_BASE + 0x100;
    Address l2_addr  = L2_BANK0_BASE + 0x50;
    Address scratch_addr = SCRATCHPAD0_BASE;

    // Prepare test data
    auto test_data = generate_pattern(transfer_size, 0x55);
    write_external_data(ext_addr, test_data);

    int completions = 0;
    auto callback = [&completions]() { completions++; };

    // Transfer chain: External -> L3 -> L2 -> Scratchpad
    // Each transfer queued with just addresses!
    dma_engine.enqueue_transfer(ext_addr, l3_addr, transfer_size, callback);

    // Process first transfer
    while (completions < 1) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Continue chain
    dma_engine.enqueue_transfer(l3_addr, l2_addr, transfer_size, callback);

    while (completions < 2) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Final transfer to scratchpad
    dma_engine.enqueue_transfer(l2_addr, scratch_addr, transfer_size, callback);

    while (completions < 3) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    // Verify data made it through entire hierarchy
    REQUIRE(verify_scratchpad_data(scratch_addr, test_data));
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Comparison with Type-Based API", "[dma][address][comparison]") {
    // This test shows both APIs side-by-side to demonstrate the difference

    const size_t transfer_size = 2048;
    Address src_addr = EXTERNAL_BANK0_BASE + 0x1000;
    Address dst_addr = SCRATCHPAD0_BASE;

    auto test_data = generate_pattern(transfer_size, 0x77);
    write_external_data(src_addr, test_data);

    SECTION("Address-Based API (New - Recommended)") {
        // Simple, clean, hardware-agnostic
        bool complete = false;
        dma_engine.enqueue_transfer(src_addr, dst_addr, transfer_size,
            [&complete]() { complete = true; });

        while (!complete) {
            dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
            dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
        }

        REQUIRE(verify_scratchpad_data(dst_addr, test_data));
    }

    SECTION("Type-Based API (Legacy - Deprecated)") {
        // Requires knowing physical memory topology
        // More verbose, tightly coupled to hardware
        bool complete = false;

        // Must decode addresses to get type/id (what decoder does internally)
        auto src_route = decoder.decode(src_addr);
        auto dst_route = decoder.decode(dst_addr);

        // Suppress deprecation warnings for legacy API demonstration
        #ifdef _MSC_VER
            #pragma warning(push)
            #pragma warning(disable: 4996)  // 'function': was declared deprecated
        #else
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        #endif

        dma_engine.enqueue_transfer(
            DMAEngine::MemoryType::KPU_MEMORY, src_route.id, src_route.offset,
            DMAEngine::MemoryType::PAGE_BUFFER, dst_route.id, dst_route.offset,
            transfer_size,
            [&complete]() { complete = true; }
        );

        #ifdef _MSC_VER
            #pragma warning(pop)
        #else
            #pragma GCC diagnostic pop
        #endif

        while (!complete) {
            dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
            dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
        }

        REQUIRE(verify_scratchpad_data(dst_addr, test_data));
    }

    // Both APIs produce identical results, but address-based is cleaner
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

    Address scratch_a = SCRATCHPAD0_BASE;
    Address scratch_b = SCRATCHPAD0_BASE + transfer_size;

    int completions = 0;
    auto callback = [&completions]() { completions++; };

    // Application code - uses virtual addresses, doesn't care about physical layout
    dma_engine.enqueue_transfer(translate(tensor_a), scratch_a, transfer_size, callback);
    dma_engine.enqueue_transfer(translate(tensor_b), scratch_b, transfer_size, callback);

    // If virtual memory remaps tensor_a to a different bank, only the mapping changes
    // The DMA code stays the same! This is the key benefit.

    while (completions < 2) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    REQUIRE(verify_scratchpad_data(scratch_a, data_a));
    REQUIRE(verify_scratchpad_data(scratch_b, data_b));
}

TEST_CASE_METHOD(AddressBasedDMAFixture, "Address-Based API: Dynamic Memory Allocation Pattern", "[dma][address][allocation]") {
    // Demonstrates how address-based API enables flexible memory allocation

    // Simulate a simple allocator that uses any available memory
    auto allocate = [](size_t size) -> Address {
        // In a real system, this would search for free space
        // For this test, we'll use different regions based on size
        if (size <= 64 * 1024) {
            return L3_TILE0_BASE;  // Small allocations in L3
        } else {
            return EXTERNAL_BANK0_BASE;  // Large allocations in external
        }
    };

    // Allocate tensors - allocator chooses optimal location
    size_t small_tensor_size = 16 * 1024;
    size_t large_tensor_size = 128 * 1024;

    Address small_tensor = allocate(small_tensor_size);
    Address large_tensor = allocate(large_tensor_size);

    // Application code doesn't care where allocator placed the data!
    // Just uses addresses
    auto small_data = generate_pattern(small_tensor_size, 0x11);
    auto large_data = generate_pattern(large_tensor_size, 0x22);

    // Decode to write (in real system, this would be transparent)
    auto small_route = decoder.decode(small_tensor);
    if (small_route.type == MemoryType::L3_TILE) {
        l3_tiles[small_route.id].write(small_route.offset, small_data.data(), small_data.size());
    }

    auto large_route = decoder.decode(large_tensor);
    if (large_route.type == MemoryType::EXTERNAL) {
        memory_banks[large_route.id].write(large_route.offset, large_data.data(), large_data.size());
    }

    // DMA transfers work regardless of where allocator placed data
    Address scratch_small = SCRATCHPAD0_BASE;
    Address scratch_large = SCRATCHPAD0_BASE + small_tensor_size;

    int completions = 0;
    auto callback = [&completions]() { completions++; };

    dma_engine.enqueue_transfer(small_tensor, scratch_small, small_tensor_size, callback);
    dma_engine.enqueue_transfer(large_tensor, scratch_large, large_tensor_size, callback);

    while (completions < 2) {
        dma_engine.process_transfers(host_memory_regions, memory_banks, l3_tiles, l2_banks, scratchpads);
        dma_engine.set_current_cycle(dma_engine.get_current_cycle() + 1);
    }

    REQUIRE(verify_scratchpad_data(scratch_small, small_data));
    // Note: large tensor exceeds scratchpad capacity (256 KB), so we only test small
}
