#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <sstream>

using namespace sw::kpu;

namespace {

KPUSimulator::Config make_test_config() {
    KPUSimulator::Config config;
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = 64;
    config.host_memory_bandwidth_gbps = 100;
    config.memory_bank_count = 2;
    config.memory_bank_capacity_mb = 4;
    config.memory_bandwidth_gbps = 100;
    config.memory_controller_count = 1;
    config.page_buffer_count = 2;
    config.page_buffer_capacity_kb = 64;
    config.l3_layer.num_tiles = 4;
    config.l3_layer.capacity_kb = 128;
    config.l3_layer.block_mover_count = 4;
    config.l2_layer.num_banks = 8;
    config.l2_layer.capacity_kb = 64;
    config.l1_layer.num_buffers = 8;
    config.l1_layer.capacity_kb = 32;
    config.dma_engine_count = 2;
    config.streamer_count = 4;
    config.compute_tile_count = 1;
    config.processor_array_rows = 16;
    config.processor_array_cols = 16;
    config.use_systolic_array_mode = true;
    return config;
}

} // namespace

TEST_CASE("KPUSimulator retains its construction config", "[system][config-report]") {
    auto config = make_test_config();
    KPUSimulator sim(config);

    // I1: snapshot matches the constructed SoC
    const auto& cfg = sim.get_config();
    REQUIRE(cfg.memory_bank_count == config.memory_bank_count);
    REQUIRE(cfg.l3_layer.num_tiles == config.l3_layer.num_tiles);
    REQUIRE(cfg.l2_layer.num_banks == config.l2_layer.num_banks);
    REQUIRE(cfg.l1_layer.num_buffers == config.l1_layer.num_buffers);
    REQUIRE(cfg.l3_layer.total_tiles() == sim.get_l3_tile_count());
    REQUIRE(cfg.l2_layer.total_banks() == sim.get_l2_bank_count());
    REQUIRE(cfg.l1_layer.total_buffers() == sim.get_l1_buffer_count());
    REQUIRE(cfg.dma_engine_count == sim.get_dma_engine_count());
    REQUIRE(cfg.streamer_count == sim.get_streamer_count());
}

TEST_CASE("Config report renders floorplan and all asset sections", "[system][config-report]") {
    KPUSimulator sim(make_test_config());
    std::string report = sim.generate_config_report();

    // Post: non-empty, one section per asset class
    REQUIRE_FALSE(report.empty());
    REQUIRE(report.find("KPU SoC Configuration Report") != std::string::npos);

    SECTION("floorplan sketch") {
        REQUIRE(report.find("HOST") != std::string::npos);
        REQUIRE(report.find("KPU SoC") != std::string::npos);
        REQUIRE(report.find("credits flow UP, data flows DOWN") != std::string::npos);
        REQUIRE(report.find("L3 Layer") != std::string::npos);
        REQUIRE(report.find("L2 Layer") != std::string::npos);
        REQUIRE(report.find("L1 Layer") != std::string::npos);
        REQUIRE(report.find("Compute Fabric") != std::string::npos);
    }

    SECTION("asset attribute sections") {
        REQUIRE(report.find("Host memory regions (1)") != std::string::npos);
        REQUIRE(report.find("External memory banks (2)") != std::string::npos);
        REQUIRE(report.find("page buffers (2)") != std::string::npos);
        REQUIRE(report.find("L3 layer (4 tile(s)") != std::string::npos);
        REQUIRE(report.find("Block movers (4") != std::string::npos);
        REQUIRE(report.find("L2 layer (8 bank(s)") != std::string::npos);
        REQUIRE(report.find("L1 layer (8 stream buffer(s)") != std::string::npos);
        REQUIRE(report.find("DMA engines (2)") != std::string::npos);
        REQUIRE(report.find("Streamers (4)") != std::string::npos);
        REQUIRE(report.find("Compute fabric (1 tile(s)") != std::string::npos);
        REQUIRE(report.find("Memory map (") != std::string::npos);
    }

    SECTION("asset attributes present") {
        REQUIRE(report.find("128 KB") != std::string::npos);   // L3 tile capacity
        REQUIRE(report.find("systolic 16x16 (256 PEs)") != std::string::npos);
        REQUIRE(report.find("100 GB/s") != std::string::npos); // bandwidth
        REQUIRE(report.find("base 0x") != std::string::npos);  // memory-map bases
    }

    SECTION("non-uniform group summary") {
        auto config = make_test_config();
        config.l3_layer.num_tiles = 0;
        config.l3_layer.tile_groups = {
            {"default", {128}, 2},
            {"hbw", {256}, 2},
        };
        KPUSimulator hetero(config);
        std::string hetero_report = hetero.generate_config_report();
        REQUIRE(hetero_report.find("default x2 @ 128 KB") != std::string::npos);
        REQUIRE(hetero_report.find("hbw x2 @ 256 KB") != std::string::npos);
    }
}

TEST_CASE("Config report generation is const and repeatable", "[system][config-report]") {
    KPUSimulator sim(make_test_config());

    // I2: purity — repeated generation yields identical output, state untouched
    std::string first = sim.generate_config_report();
    std::string second = sim.generate_config_report();
    REQUIRE(first == second);
    REQUIRE(sim.get_current_cycle() == 0);

    // print_config_report(ostream) streams exactly the generated report
    std::ostringstream oss;
    sim.print_config_report(oss);
    REQUIRE(oss.str() == first);
}

TEST_CASE("Config report handles an empty default configuration", "[system][config-report]") {
    KPUSimulator sim{KPUSimulator::Config{}};
    std::string report = sim.generate_config_report();
    REQUIRE_FALSE(report.empty());
    REQUIRE(report.find("(none)") != std::string::npos);
}
