// ============================================================================
// tests/runtime/test_behavioral_memory.cpp
// Unit tests for behavioral memory model
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/models/behavioral/memory/memory_model.hpp>

#include <cmath>
#include <vector>

using namespace sw::kpu::behavioral;
using Catch::Matchers::WithinRel;

// ============================================================================
// Address Encoding Tests
// ============================================================================

TEST_CASE("AddressEncoder encodes and decodes correctly", "[behavioral][memory]") {
    SECTION("Host address encoding") {
        Address addr = AddressEncoder::encode(MemoryRegionType::HOST, 0, 0x1000);
        REQUIRE(AddressEncoder::decode_type(addr) == MemoryRegionType::HOST);
        REQUIRE(AddressEncoder::decode_region_id(addr) == 0);
        REQUIRE(AddressEncoder::decode_offset(addr) == 0x1000);
    }

    SECTION("L3 tile address encoding") {
        Address addr = AddressEncoder::encode(MemoryRegionType::L3_TILE, 3, 0x2000);
        REQUIRE(AddressEncoder::decode_type(addr) == MemoryRegionType::L3_TILE);
        REQUIRE(AddressEncoder::decode_region_id(addr) == 3);
        REQUIRE(AddressEncoder::decode_offset(addr) == 0x2000);
    }

    SECTION("L2 bank address encoding") {
        Address addr = AddressEncoder::encode(MemoryRegionType::L2_BANK, 7, 0x500);
        REQUIRE(AddressEncoder::decode_type(addr) == MemoryRegionType::L2_BANK);
        REQUIRE(AddressEncoder::decode_region_id(addr) == 7);
        REQUIRE(AddressEncoder::decode_offset(addr) == 0x500);
    }

    SECTION("L1 buffer address encoding") {
        Address addr = AddressEncoder::encode(MemoryRegionType::L1_BUFFER, 15, 0x100);
        REQUIRE(AddressEncoder::decode_type(addr) == MemoryRegionType::L1_BUFFER);
        REQUIRE(AddressEncoder::decode_region_id(addr) == 15);
        REQUIRE(AddressEncoder::decode_offset(addr) == 0x100);
    }

    SECTION("Large offset encoding") {
        Size large_offset = (1ULL << 31) - 1;  // Max 32-bit offset
        Address addr = AddressEncoder::encode(MemoryRegionType::HOST, 0, large_offset);
        REQUIRE(AddressEncoder::decode_offset(addr) == large_offset);
    }
}

// ============================================================================
// Memory Region Tests
// ============================================================================

TEST_CASE("MemoryRegion basic operations", "[behavioral][memory]") {
    MemoryRegion region(MemoryRegionType::HOST, 0, 1024);

    SECTION("Initial state") {
        REQUIRE(region.type() == MemoryRegionType::HOST);
        REQUIRE(region.id() == 0);
        REQUIRE(region.capacity() == 1024);
        REQUIRE(region.used() == 0);
        REQUIRE(region.available() == 1024);
    }

    SECTION("Write and read") {
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        region.write(0, data, sizeof(data));

        float result[4];
        region.read(0, result, sizeof(result));

        for (int i = 0; i < 4; ++i) {
            REQUIRE_THAT(result[i], WithinRel(data[i], 1e-6f));
        }
    }

    SECTION("Write at offset") {
        float data1[] = {1.0f, 2.0f};
        float data2[] = {3.0f, 4.0f};

        region.write(0, data1, sizeof(data1));
        region.write(100, data2, sizeof(data2));

        float result1[2], result2[2];
        region.read(0, result1, sizeof(result1));
        region.read(100, result2, sizeof(result2));

        REQUIRE_THAT(result1[0], WithinRel(1.0f, 1e-6f));
        REQUIRE_THAT(result1[1], WithinRel(2.0f, 1e-6f));
        REQUIRE_THAT(result2[0], WithinRel(3.0f, 1e-6f));
        REQUIRE_THAT(result2[1], WithinRel(4.0f, 1e-6f));
    }

    SECTION("Out of bounds write throws") {
        float data[256];
        REQUIRE_THROWS_AS(region.write(900, data, sizeof(data)), std::out_of_range);
    }

    SECTION("Out of bounds read throws") {
        float data[256];
        REQUIRE_THROWS_AS(region.read(900, data, sizeof(data)), std::out_of_range);
    }

    SECTION("Clear zeros memory") {
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        region.write(0, data, sizeof(data));
        region.clear();

        float result[4];
        region.read(0, result, sizeof(result));
        for (int i = 0; i < 4; ++i) {
            REQUIRE(result[i] == 0.0f);
        }
    }

    SECTION("Data pointer access") {
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        region.write(0, data, sizeof(data));

        const float* ptr = reinterpret_cast<const float*>(region.data_ptr(0));
        for (int i = 0; i < 4; ++i) {
            REQUIRE_THAT(ptr[i], WithinRel(data[i], 1e-6f));
        }
    }
}

// ============================================================================
// Memory Model Allocation Tests
// ============================================================================

TEST_CASE("BehavioralMemoryModel allocation", "[behavioral][memory]") {
    MemoryModelConfig config;
    config.host_capacity_bytes = 1024 * 1024;  // 1 MB
    config.l3_tile_count = 4;
    config.l3_tile_capacity_bytes = 64 * 1024;  // 64 KB
    config.l2_bank_count = 8;
    config.l2_bank_capacity_bytes = 32 * 1024;  // 32 KB
    config.l1_buffer_count = 16;
    config.l1_buffer_capacity_bytes = 8 * 1024;  // 8 KB

    BehavioralMemoryModel memory(config);

    SECTION("Host allocation") {
        auto alloc = memory.allocate_host(1024, "test_tensor");
        REQUIRE(alloc.is_valid());
        REQUIRE(alloc.region_type == MemoryRegionType::HOST);
        REQUIRE(alloc.region_id == 0);
        REQUIRE(alloc.size_bytes == 1024);
        REQUIRE(alloc.name == "test_tensor");

        // Find by name
        auto found = memory.find_allocation("test_tensor");
        REQUIRE(found.has_value());
        REQUIRE(found->base_address == alloc.base_address);
    }

    SECTION("L3 tile allocation") {
        auto alloc = memory.allocate_l3(2, 4096, "l3_data");
        REQUIRE(alloc.is_valid());
        REQUIRE(alloc.region_type == MemoryRegionType::L3_TILE);
        REQUIRE(alloc.region_id == 2);
        REQUIRE(alloc.size_bytes == 4096);
    }

    SECTION("L2 bank allocation") {
        auto alloc = memory.allocate_l2(5, 2048, "l2_buffer");
        REQUIRE(alloc.is_valid());
        REQUIRE(alloc.region_type == MemoryRegionType::L2_BANK);
        REQUIRE(alloc.region_id == 5);
        REQUIRE(alloc.size_bytes == 2048);
    }

    SECTION("L1 buffer allocation") {
        auto alloc = memory.allocate_l1(10, 512, "l1_stream");
        REQUIRE(alloc.is_valid());
        REQUIRE(alloc.region_type == MemoryRegionType::L1_BUFFER);
        REQUIRE(alloc.region_id == 10);
        REQUIRE(alloc.size_bytes == 512);
    }

    SECTION("Multiple allocations are sequential") {
        auto alloc1 = memory.allocate_host(100);
        auto alloc2 = memory.allocate_host(200);
        auto alloc3 = memory.allocate_host(300);

        // Allocations should be sequential (bump allocator)
        Size offset1 = AddressEncoder::decode_offset(alloc1.base_address);
        Size offset2 = AddressEncoder::decode_offset(alloc2.base_address);
        Size offset3 = AddressEncoder::decode_offset(alloc3.base_address);

        REQUIRE(offset1 == 0);
        REQUIRE(offset2 == 100);
        REQUIRE(offset3 == 300);
    }

    SECTION("Allocation overflow throws") {
        // Try to allocate more than L3 tile capacity
        REQUIRE_THROWS_AS(
            memory.allocate_l3(0, 100 * 1024),  // 100 KB > 64 KB
            std::runtime_error
        );
    }

    SECTION("Invalid region ID throws") {
        REQUIRE_THROWS_AS(memory.allocate_l3(10, 1024), std::out_of_range);
        REQUIRE_THROWS_AS(memory.allocate_l2(20, 1024), std::out_of_range);
        REQUIRE_THROWS_AS(memory.allocate_l1(30, 1024), std::out_of_range);
    }
}

// ============================================================================
// Memory Model Data Operations Tests
// ============================================================================

TEST_CASE("BehavioralMemoryModel data operations", "[behavioral][memory]") {
    MemoryModelConfig config;
    config.host_capacity_bytes = 64 * 1024;
    config.l3_tile_count = 2;
    config.l3_tile_capacity_bytes = 16 * 1024;
    config.l2_bank_count = 4;
    config.l2_bank_capacity_bytes = 8 * 1024;

    BehavioralMemoryModel memory(config);

    SECTION("Write and read floats") {
        auto alloc = memory.allocate_host(16 * sizeof(float));

        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
        memory.write_floats(alloc.base_address, data);

        auto result = memory.read_floats(alloc.base_address, 16);
        REQUIRE(result.size() == 16);
        for (size_t i = 0; i < 16; ++i) {
            REQUIRE_THAT(result[i], WithinRel(data[i], 1e-6f));
        }
    }

    SECTION("Copy between regions") {
        auto host_alloc = memory.allocate_host(1024);
        auto l3_alloc = memory.allocate_l3(0, 1024);

        // Write to host
        std::vector<float> data(256, 3.14f);
        memory.write(host_alloc.base_address, data.data(), data.size() * sizeof(float));

        // Copy host -> L3 (simulates DMA)
        memory.copy(l3_alloc.base_address, host_alloc.base_address, 1024);

        // Read from L3
        std::vector<float> result(256);
        memory.read(l3_alloc.base_address, result.data(), 1024);

        for (size_t i = 0; i < 256; ++i) {
            REQUIRE_THAT(result[i], WithinRel(3.14f, 1e-6f));
        }
    }

    SECTION("Copy L3 to L2") {
        auto l3_alloc = memory.allocate_l3(1, 2048);
        auto l2_alloc = memory.allocate_l2(2, 2048);

        // Write to L3
        std::vector<float> data(512);
        for (size_t i = 0; i < 512; ++i) {
            data[i] = static_cast<float>(i);
        }
        memory.write(l3_alloc.base_address, data.data(), data.size() * sizeof(float));

        // Copy L3 -> L2 (simulates BlockMover)
        memory.copy(l2_alloc.base_address, l3_alloc.base_address, 2048);

        // Read from L2
        std::vector<float> result(512);
        memory.read(l2_alloc.base_address, result.data(), 2048);

        for (size_t i = 0; i < 512; ++i) {
            REQUIRE_THAT(result[i], WithinRel(static_cast<float>(i), 1e-6f));
        }
    }

    SECTION("Zero memory") {
        auto alloc = memory.allocate_host(1024);

        // Write non-zero data
        std::vector<float> data(256, 1.0f);
        memory.write(alloc.base_address, data.data(), 1024);

        // Zero it
        memory.zero(alloc.base_address, 1024);

        // Read and verify zeros
        std::vector<float> result(256);
        memory.read(alloc.base_address, result.data(), 1024);

        for (size_t i = 0; i < 256; ++i) {
            REQUIRE(result[i] == 0.0f);
        }
    }

    SECTION("Get typed pointer") {
        auto alloc = memory.allocate_host(16 * sizeof(float));

        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        memory.write(alloc.base_address, data.data(), data.size() * sizeof(float));

        // Get pointer and access directly
        const float* ptr = memory.get_ptr<float>(alloc.base_address);
        REQUIRE_THAT(ptr[0], WithinRel(1.0f, 1e-6f));
        REQUIRE_THAT(ptr[1], WithinRel(2.0f, 1e-6f));
        REQUIRE_THAT(ptr[2], WithinRel(3.0f, 1e-6f));
        REQUIRE_THAT(ptr[3], WithinRel(4.0f, 1e-6f));
    }
}

// ============================================================================
// Tensor Location Tracking Tests
// ============================================================================

TEST_CASE("BehavioralMemoryModel tensor tracking", "[behavioral][memory]") {
    BehavioralMemoryModel memory;

    SECTION("Register and locate tensor") {
        auto alloc = memory.allocate_host(1024, "weights");
        memory.register_tensor("weights", alloc);

        auto location = memory.locate_tensor("weights");
        REQUIRE(location.has_value());
        REQUIRE(location->tensor_name == "weights");
        REQUIRE(location->allocations.size() == 1);
        REQUIRE(location->exists_in(MemoryRegionType::HOST));
    }

    SECTION("Tensor in multiple locations") {
        auto host_alloc = memory.allocate_host(1024, "input_host");
        auto l3_alloc = memory.allocate_l3(0, 1024, "input_l3");

        memory.register_tensor("input", host_alloc);
        memory.update_tensor_location("input", l3_alloc);

        auto location = memory.locate_tensor("input");
        REQUIRE(location.has_value());
        REQUIRE(location->allocations.size() == 2);
        REQUIRE(location->exists_in(MemoryRegionType::HOST));
        REQUIRE(location->exists_in(MemoryRegionType::L3_TILE));

        auto host_loc = location->find_in(MemoryRegionType::HOST);
        REQUIRE(host_loc.has_value());
        REQUIRE(host_loc->base_address == host_alloc.base_address);

        auto l3_loc = location->find_in(MemoryRegionType::L3_TILE);
        REQUIRE(l3_loc.has_value());
        REQUIRE(l3_loc->base_address == l3_alloc.base_address);
    }

    SECTION("Unregister tensor") {
        auto alloc = memory.allocate_host(1024);
        memory.register_tensor("temp", alloc);

        auto location = memory.locate_tensor("temp");
        REQUIRE(location.has_value());

        memory.unregister_tensor("temp");
        location = memory.locate_tensor("temp");
        REQUIRE_FALSE(location.has_value());
    }

    SECTION("Locate nonexistent tensor") {
        auto location = memory.locate_tensor("nonexistent");
        REQUIRE_FALSE(location.has_value());
    }
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_CASE("BehavioralMemoryModel validation", "[behavioral][memory]") {
    BehavioralMemoryModel memory;

    SECTION("is_valid for valid address") {
        auto alloc = memory.allocate_host(1024);
        REQUIRE(memory.is_valid(alloc.base_address, 512));
        REQUIRE(memory.is_valid(alloc.base_address, 1024));
    }

    SECTION("is_valid for region overflow") {
        // is_valid checks against region capacity, not allocation size
        // Create a small config to test region overflow
        MemoryModelConfig small_config;
        small_config.l1_buffer_capacity_bytes = 512;
        BehavioralMemoryModel small_memory(small_config);

        auto alloc = small_memory.allocate_l1(0, 256);
        // Reading 256 bytes is valid (within region)
        REQUIRE(small_memory.is_valid(alloc.base_address, 256));
        // Reading 1024 bytes exceeds L1 buffer capacity of 512
        REQUIRE_FALSE(small_memory.is_valid(alloc.base_address, 1024));
    }

    SECTION("verify data matches") {
        auto alloc = memory.allocate_host(16);

        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        memory.write(alloc.base_address, data, sizeof(data));

        REQUIRE(memory.verify(alloc.base_address, data, sizeof(data)));

        float wrong[] = {1.0f, 2.0f, 3.0f, 5.0f};  // Last element different
        REQUIRE_FALSE(memory.verify(alloc.base_address, wrong, sizeof(wrong)));
    }
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("BehavioralMemoryModel statistics", "[behavioral][memory]") {
    BehavioralMemoryModel memory;

    SECTION("Track allocation statistics") {
        memory.allocate_host(1000);
        memory.allocate_host(500);
        memory.allocate_l3(0, 2000);
        memory.allocate_l2(0, 800);
        memory.allocate_l1(0, 400);

        const auto& stats = memory.stats();
        REQUIRE(stats.total_host_allocated == 1500);
        REQUIRE(stats.total_l3_allocated == 2000);
        REQUIRE(stats.total_l2_allocated == 800);
        REQUIRE(stats.total_l1_allocated == 400);
    }

    SECTION("Track read/write statistics") {
        auto alloc = memory.allocate_host(1024);

        float data[64];
        memory.write(alloc.base_address, data, sizeof(data));
        memory.write(alloc.base_address, data, sizeof(data));
        memory.read(alloc.base_address, data, sizeof(data));

        const auto& stats = memory.stats();
        REQUIRE(stats.write_count == 2);
        REQUIRE(stats.read_count == 1);
        REQUIRE(stats.bytes_written == 2 * sizeof(data));
        REQUIRE(stats.bytes_read == sizeof(data));
    }

    SECTION("Track copy statistics") {
        auto alloc1 = memory.allocate_host(1024);
        auto alloc2 = memory.allocate_l3(0, 1024);

        memory.copy(alloc2.base_address, alloc1.base_address, 512);
        memory.copy(alloc2.base_address, alloc1.base_address, 256);

        const auto& stats = memory.stats();
        REQUIRE(stats.copy_count == 2);
        REQUIRE(stats.bytes_copied == 768);
    }

    SECTION("Reset statistics") {
        memory.allocate_host(1000);

        auto alloc = memory.allocate_host(100);
        float data[25];
        memory.write(alloc.base_address, data, sizeof(data));

        memory.reset_stats();

        const auto& stats = memory.stats();
        REQUIRE(stats.total_host_allocated == 0);
        REQUIRE(stats.write_count == 0);
        REQUIRE(stats.bytes_written == 0);
    }
}

// ============================================================================
// MLP-Specific Data Layout Tests
// ============================================================================

TEST_CASE("BehavioralMemoryModel MLP data layout", "[behavioral][memory][mlp]") {
    MemoryModelConfig config;
    config.host_capacity_bytes = 1024 * 1024;  // 1 MB
    config.l3_tile_count = 4;
    config.l3_tile_capacity_bytes = 128 * 1024;  // 128 KB per tile
    config.l2_bank_count = 8;
    config.l2_bank_capacity_bytes = 64 * 1024;  // 64 KB per bank

    BehavioralMemoryModel memory(config);

    SECTION("Store XOR MLP weights and inputs") {
        // XOR network: 2 -> 4 -> 1

        // Hidden layer: weights [2,4], bias [4]
        const float hidden_weights[2 * 4] = {
            2.5f, -2.5f, 2.0f, -2.0f,
            -2.5f, 2.5f, 2.0f, -2.0f
        };
        const float hidden_bias[4] = {-0.5f, -0.5f, -3.0f, 3.0f};

        // Output layer: weights [4,1], bias [1]
        const float output_weights[4 * 1] = {2.0f, 2.0f, -1.0f, -1.0f};
        const float output_bias[1] = {-0.5f};

        // Inputs [4,2] - all 4 XOR test cases
        const float inputs[4 * 2] = {
            0.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f
        };

        // Allocate in host memory
        auto hw_alloc = memory.allocate_host(sizeof(hidden_weights), "hidden_weights");
        auto hb_alloc = memory.allocate_host(sizeof(hidden_bias), "hidden_bias");
        auto ow_alloc = memory.allocate_host(sizeof(output_weights), "output_weights");
        auto ob_alloc = memory.allocate_host(sizeof(output_bias), "output_bias");
        auto in_alloc = memory.allocate_host(sizeof(inputs), "inputs");
        auto out_alloc = memory.allocate_host(4 * sizeof(float), "outputs");

        // Write data
        memory.write(hw_alloc.base_address, hidden_weights, sizeof(hidden_weights));
        memory.write(hb_alloc.base_address, hidden_bias, sizeof(hidden_bias));
        memory.write(ow_alloc.base_address, output_weights, sizeof(output_weights));
        memory.write(ob_alloc.base_address, output_bias, sizeof(output_bias));
        memory.write(in_alloc.base_address, inputs, sizeof(inputs));

        // Verify weights can be read back
        auto read_hw = memory.read_floats(hw_alloc.base_address, 8);
        REQUIRE_THAT(read_hw[0], WithinRel(2.5f, 1e-6f));
        REQUIRE_THAT(read_hw[1], WithinRel(-2.5f, 1e-6f));

        auto read_inputs = memory.read_floats(in_alloc.base_address, 8);
        REQUIRE(read_inputs[0] == 0.0f);
        REQUIRE(read_inputs[1] == 0.0f);
        REQUIRE(read_inputs[2] == 0.0f);
        REQUIRE(read_inputs[3] == 1.0f);
    }

    SECTION("Simulate DMA transfer Host -> L3") {
        // Allocate weight matrix in host
        const size_t weight_size = 64 * 64 * sizeof(float);  // 64x64 matrix
        auto host_alloc = memory.allocate_host(weight_size, "weights_host");

        // Initialize with some pattern
        std::vector<float> weights(64 * 64);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<float>(i) * 0.01f;
        }
        memory.write(host_alloc.base_address, weights.data(), weight_size);

        // Allocate in L3 (simulates DMA target)
        auto l3_alloc = memory.allocate_l3(0, weight_size, "weights_l3");

        // DMA transfer (copy)
        memory.copy(l3_alloc.base_address, host_alloc.base_address, weight_size);

        // Register tensor location update
        memory.register_tensor("weights", host_alloc);
        memory.update_tensor_location("weights", l3_alloc);

        // Verify data is in L3
        auto l3_weights = memory.read_floats(l3_alloc.base_address, 64 * 64);
        REQUIRE_THAT(l3_weights[0], WithinRel(0.0f, 1e-6f));
        REQUIRE_THAT(l3_weights[100], WithinRel(1.0f, 1e-6f));
        REQUIRE_THAT(l3_weights[1000], WithinRel(10.0f, 1e-6f));

        // Verify tensor can be found in both locations
        auto location = memory.locate_tensor("weights");
        REQUIRE(location.has_value());
        REQUIRE(location->exists_in(MemoryRegionType::HOST));
        REQUIRE(location->exists_in(MemoryRegionType::L3_TILE));
    }

    SECTION("Simulate BlockMover transfer L3 -> L2") {
        // Tile in L3
        const size_t tile_size = 16 * 16 * sizeof(float);  // 16x16 tile
        auto l3_alloc = memory.allocate_l3(1, tile_size, "tile_l3");

        // Initialize tile
        std::vector<float> tile(16 * 16);
        for (size_t i = 0; i < tile.size(); ++i) {
            tile[i] = static_cast<float>(i);
        }
        memory.write(l3_alloc.base_address, tile.data(), tile_size);

        // Allocate in L2 (simulates BlockMover target)
        auto l2_alloc = memory.allocate_l2(3, tile_size, "tile_l2");

        // BlockMover transfer
        memory.copy(l2_alloc.base_address, l3_alloc.base_address, tile_size);

        // Verify data is in L2 and can be used for compute
        const float* l2_ptr = memory.get_ptr<float>(l2_alloc.base_address);
        REQUIRE_THAT(l2_ptr[0], WithinRel(0.0f, 1e-6f));
        REQUIRE_THAT(l2_ptr[127], WithinRel(127.0f, 1e-6f));
        REQUIRE_THAT(l2_ptr[255], WithinRel(255.0f, 1e-6f));
    }
}
