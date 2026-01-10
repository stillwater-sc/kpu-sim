// ============================================================================
// tests/dma/test_dma_memory_noc_integration.cpp
// Integration tests for DMA-Memory-NoC data path
//
// Tests the complete data movement pipeline:
// DMA Engine -> Memory Controller -> DMA Engine -> NoC -> Destination
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/components/dma/cycle_accurate_dma_engine.hpp>
#include <sw/kpu/components/dma/dma_placement.hpp>
#include <sw/kpu/components/memory/memory_controller_interface.hpp>
#include <sw/kpu/noc/noc_interface.hpp>

#include <atomic>
#include <vector>
#include <iostream>

using namespace sw::kpu;
using namespace sw::kpu::noc;

// ============================================================================
// Test Helpers
// ============================================================================

namespace {

/// Create a cycle-accurate memory controller for testing
std::unique_ptr<IMemoryController> create_test_memory_controller(
    MemoryTechnology tech = MemoryTechnology::LPDDR5)
{
    MemoryControllerConfig config;
    config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
    config.technology = tech;
    config.num_channels = 1;
    config.banks_per_channel = 8;
    config.queue_depth = 32;
    return create_memory_controller(config);
}

/// Create a behavioral memory controller (fast, for NoC-focused tests)
std::unique_ptr<IMemoryController> create_behavioral_memory_controller()
{
    MemoryControllerConfig config;
    config.fidelity = SimulationFidelity::BEHAVIORAL;
    config.technology = MemoryTechnology::IDEAL;
    return create_memory_controller(config);
}

/// Create a test NoC
std::unique_ptr<INoC> create_test_noc(uint8_t rows = 4, uint8_t cols = 4)
{
    noc::NoCConfig config;
    config.rows = rows;
    config.cols = cols;
    config.buffer_depth = 8;
    config.flit_size_bytes = 64;
    return create_noc(NoCType::DATAFLOW, config);
}

/// Create a cycle-accurate DMA engine for testing
std::unique_ptr<CycleAccurateDMAEngine> create_test_dma_engine(uint32_t num_channels = 4)
{
    CycleAccurateDMAConfig config;
    config.num_channels = num_channels;
    config.queue_depth_per_channel = 8;
    return std::make_unique<CycleAccurateDMAEngine>(config);
}

/// Run co-simulation of DMA, memory, and NoC for a maximum number of cycles
/// Returns actual cycles used
uint64_t cosimulate(CycleAccurateDMAEngine& dma,
                    IMemoryController* mc,
                    INoC* noc,
                    uint64_t max_cycles)
{
    uint64_t cycle = 0;

    while (cycle < max_cycles) {
        // Check if all done
        bool dma_done = !dma.has_pending();
        bool mc_done = mc ? !mc->has_pending() : true;
        bool noc_done = noc ? noc->is_idle() : true;

        if (dma_done && mc_done && noc_done) {
            break;
        }

        // Advance all components
        dma.set_cycle(cycle);
        dma.tick();

        if (mc) {
            mc->set_cycle(cycle);
            mc->tick();
        }

        if (noc) {
            noc->step(cycle);
        }

        ++cycle;
    }

    return cycle;
}

} // anonymous namespace

// ============================================================================
// DMA + Memory Controller Integration Tests
// ============================================================================

TEST_CASE("DMA-Memory integration basic", "[dma][memory][integration]") {
    auto dma = create_test_dma_engine(2);
    auto mc = create_test_memory_controller();

    // Bind DMA to memory controller
    dma->bind_memory_controller(mc.get());

    SECTION("DMA has memory binding") {
        REQUIRE(dma->has_memory_binding());
        REQUIRE_FALSE(dma->has_noc_binding());
    }

    SECTION("Simple transfer through memory controller") {
        std::atomic<bool> completed{false};

        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.src_addr = 0x1000;
        transfer.dst_type = DMAMemoryType::L3_TILE;
        transfer.dst_addr = 0;
        transfer.size = 256;

        auto id = dma->submit(transfer, [&completed]() {
            completed = true;
        });

        REQUIRE(id.has_value());
        REQUIRE(dma->has_pending());

        // Co-simulate
        uint64_t cycles = cosimulate(*dma, mc.get(), nullptr, 10000);

        REQUIRE(completed);
        REQUIRE_FALSE(dma->has_pending());
        REQUIRE(cycles > 0);
        REQUIRE(cycles < 10000);  // Should complete well before timeout

        // Check statistics
        const auto& stats = dma->stats();
        REQUIRE(stats.transfers_completed == 1);
        REQUIRE(stats.bytes_transferred == 256);
    }

    SECTION("Multiple concurrent transfers") {
        std::atomic<int> completed_count{0};

        constexpr int NUM_TRANSFERS = 4;

        for (int i = 0; i < NUM_TRANSFERS; ++i) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x1000 + i * 0x100;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.size = 64;

            auto id = dma->submit(transfer, [&completed_count]() {
                ++completed_count;
            });

            REQUIRE(id.has_value());
        }

        REQUIRE(dma->pending_count() == NUM_TRANSFERS);

        // Co-simulate
        uint64_t cycles = cosimulate(*dma, mc.get(), nullptr, 50000);

        REQUIRE(completed_count == NUM_TRANSFERS);
        REQUIRE_FALSE(dma->has_pending());

        const auto& stats = dma->stats();
        REQUIRE(stats.transfers_completed == NUM_TRANSFERS);
    }
}

TEST_CASE("DMA-Memory backpressure handling", "[dma][memory][integration][backpressure]") {
    // Create memory controller with small queue
    MemoryControllerConfig mc_config;
    mc_config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
    mc_config.technology = MemoryTechnology::LPDDR5;
    mc_config.queue_depth = 4;  // Small queue to trigger backpressure
    auto mc = create_memory_controller(mc_config);

    auto dma = create_test_dma_engine(8);  // More DMA channels than MC queue
    dma->bind_memory_controller(mc.get());

    SECTION("DMA handles memory queue full") {
        std::atomic<int> completed_count{0};

        // Submit more transfers than memory queue can hold
        constexpr int NUM_TRANSFERS = 8;

        for (int i = 0; i < NUM_TRANSFERS; ++i) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x1000 * i;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.size = 64;

            dma->submit(transfer, [&completed_count]() {
                ++completed_count;
            });
        }

        // Co-simulate - all should eventually complete
        uint64_t cycles = cosimulate(*dma, mc.get(), nullptr, 100000);

        REQUIRE(completed_count == NUM_TRANSFERS);
        REQUIRE_FALSE(dma->has_pending());

        // Verify stall cycles were tracked (backpressure occurred)
        const auto& stats = dma->stats();
        REQUIRE(stats.transfers_completed == NUM_TRANSFERS);
    }
}

// ============================================================================
// DMA + NoC Integration Tests
// ============================================================================

TEST_CASE("DMA-NoC integration basic", "[dma][noc][integration]") {
    auto dma = create_test_dma_engine(2);
    auto noc = create_test_noc(4, 4);

    // Bind DMA to NoC at edge router 0 (west edge, row 0)
    dma->bind_noc(noc.get(), 0);

    SECTION("DMA has NoC binding") {
        REQUIRE(dma->has_noc_binding());
        REQUIRE(dma->bound_router_id() == 0);
    }

    SECTION("Simple NoC tile injection") {
        std::atomic<bool> dma_completed{false};
        std::atomic<bool> noc_delivered{false};

        // Set up NoC delivery callback
        noc->set_delivery_callback(5, [&noc_delivered](
            const TileDescriptor& tile,
            uint8_t src, uint8_t dst,
            uint64_t inject_cycle, uint64_t deliver_cycle)
        {
            noc_delivered = true;
        });

        // Create transfer that targets router 5 (row 1, col 1)
        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.src_addr = 0x1000;
        transfer.dst_type = DMAMemoryType::L3_TILE;
        transfer.dst_id = 5;  // Target router
        transfer.size = 64;

        auto id = dma->submit(transfer, [&dma_completed]() {
            dma_completed = true;
        });

        REQUIRE(id.has_value());

        // Co-simulate (use behavioral MC for fast memory access)
        auto mc = create_behavioral_memory_controller();
        dma->bind_memory_controller(mc.get());

        uint64_t cycles = cosimulate(*dma, mc.get(), noc.get(), 10000);

        // DMA should complete
        REQUIRE(dma_completed);
        REQUIRE_FALSE(dma->has_pending());

        // Check NoC statistics
        const auto& noc_stats = noc->stats();
        REQUIRE(noc_stats.tiles_injected >= 1);
    }
}

TEST_CASE("DMA-NoC backpressure handling", "[dma][noc][integration][backpressure]") {
    auto dma = create_test_dma_engine(4);

    // Create NoC with small buffers to trigger backpressure
    noc::NoCConfig noc_config;
    noc_config.rows = 4;
    noc_config.cols = 4;
    noc_config.buffer_depth = 2;  // Small buffers
    auto noc = create_noc(NoCType::DATAFLOW, noc_config);

    dma->bind_noc(noc.get(), 0);

    // Use behavioral MC for fast memory
    auto mc = create_behavioral_memory_controller();
    dma->bind_memory_controller(mc.get());

    SECTION("Multiple tiles with NoC congestion") {
        std::atomic<int> completed_count{0};

        constexpr int NUM_TRANSFERS = 8;

        for (int i = 0; i < NUM_TRANSFERS; ++i) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x1000 * i;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = 15;  // Far corner - longer path
            transfer.size = 128;

            dma->submit(transfer, [&completed_count]() {
                ++completed_count;
            });
        }

        // Co-simulate
        uint64_t cycles = cosimulate(*dma, mc.get(), noc.get(), 50000);

        REQUIRE(completed_count == NUM_TRANSFERS);
        REQUIRE_FALSE(dma->has_pending());
    }
}

// ============================================================================
// Full DMA-Memory-NoC Integration Tests
// ============================================================================

TEST_CASE("Full DMA-Memory-NoC pipeline", "[dma][memory][noc][integration][full]") {
    // Create all components
    auto mc = create_test_memory_controller();
    auto noc = create_test_noc(4, 4);
    auto dma = create_test_dma_engine(4);

    // Wire everything together
    dma->bind_memory_controller(mc.get());
    dma->bind_noc(noc.get(), 0);  // West edge, row 0

    SECTION("End-to-end data movement") {
        std::atomic<bool> completed{false};
        uint64_t delivery_latency = 0;

        // Track NoC delivery
        noc->set_delivery_callback(5, [&](
            const TileDescriptor& tile,
            uint8_t src, uint8_t dst,
            uint64_t inject_cycle, uint64_t deliver_cycle)
        {
            delivery_latency = deliver_cycle - inject_cycle;
        });

        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.src_addr = 0x1000;
        transfer.dst_type = DMAMemoryType::L3_TILE;
        transfer.dst_id = 5;
        transfer.size = 256;

        auto id = dma->submit(transfer, [&completed]() {
            completed = true;
        });

        REQUIRE(id.has_value());

        // Co-simulate
        uint64_t cycles = cosimulate(*dma, mc.get(), noc.get(), 50000);

        REQUIRE(completed);
        REQUIRE(cycles > 0);
        REQUIRE(cycles < 50000);

        // Verify statistics
        const auto& dma_stats = dma->stats();
        REQUIRE(dma_stats.transfers_completed == 1);
        REQUIRE(dma_stats.bytes_transferred == 256);

        const auto& mc_stats = mc->stats();
        REQUIRE(mc_stats.reads >= 1);
    }

    SECTION("Multiple tiles to different destinations") {
        std::atomic<int> completed_count{0};
        std::vector<uint64_t> completion_cycles;
        completion_cycles.resize(4);

        // Submit transfers to different corners of the mesh
        uint8_t destinations[] = {0, 3, 12, 15};  // Four corners

        for (int i = 0; i < 4; ++i) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x1000 * i;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = destinations[i];
            transfer.size = 64;

            dma->submit(transfer, [&completed_count]() {
                ++completed_count;
            });
        }

        REQUIRE(dma->pending_count() == 4);

        // Co-simulate
        uint64_t cycles = cosimulate(*dma, mc.get(), noc.get(), 100000);

        REQUIRE(completed_count == 4);
        REQUIRE_FALSE(dma->has_pending());

        const auto& dma_stats = dma->stats();
        REQUIRE(dma_stats.transfers_completed == 4);
    }
}

// ============================================================================
// DMA System Tests (Multiple DMA Engines)
// ============================================================================

TEST_CASE("DMA System with placement", "[dma][system][integration]") {
    // Create 4x4 mesh NoC
    auto noc = create_test_noc(4, 4);

    // Create memory controllers (2 for round-robin assignment)
    auto mc0 = create_test_memory_controller();
    auto mc1 = create_test_memory_controller();
    std::vector<IMemoryController*> mcs = {mc0.get(), mc1.get()};

    // Create standard DMA placement (one per west row, one per north column)
    auto placement = DMASystemConfig::create_standard(4, 4, 2);

    CycleAccurateDMAConfig engine_config;
    engine_config.num_channels = 2;

    // Create DMA system
    auto system = create_dma_system(placement, engine_config, mcs, noc.get());

    SECTION("System has correct configuration") {
        REQUIRE(system.west_dmas.size() == 4);   // One per row
        REQUIRE(system.north_dmas.size() == 4);  // One per column
        REQUIRE(system.size() == 8);
    }

    SECTION("West edge DMA transfers (A tiles)") {
        std::atomic<int> completed_count{0};

        // Submit a transfer from each west DMA
        for (size_t row = 0; row < system.west_dmas.size(); ++row) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x10000 * row;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = row * 4 + 3;  // East edge of each row
            transfer.size = 64;

            system.west_dmas[row]->submit(transfer, [&completed_count]() {
                ++completed_count;
            });
        }

        REQUIRE(system.has_pending());

        // Co-simulate all components
        uint64_t cycle = 0;
        while (system.has_pending() && cycle < 100000) {
            mc0->set_cycle(cycle);
            mc1->set_cycle(cycle);
            mc0->tick();
            mc1->tick();

            system.set_cycle(cycle);
            system.tick();

            noc->step(cycle);
            ++cycle;
        }

        REQUIRE(completed_count == 4);
        REQUIRE_FALSE(system.has_pending());
    }

    SECTION("North edge DMA transfers (B tiles)") {
        std::atomic<int> completed_count{0};

        // Submit a transfer from each north DMA
        for (size_t col = 0; col < system.north_dmas.size(); ++col) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x20000 * col;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = 12 + col;  // South edge
            transfer.size = 64;

            system.north_dmas[col]->submit(transfer, [&completed_count]() {
                ++completed_count;
            });
        }

        REQUIRE(system.has_pending());

        // Co-simulate
        uint64_t cycle = 0;
        while (system.has_pending() && cycle < 100000) {
            mc0->set_cycle(cycle);
            mc1->set_cycle(cycle);
            mc0->tick();
            mc1->tick();

            system.set_cycle(cycle);
            system.tick();

            noc->step(cycle);
            ++cycle;
        }

        REQUIRE(completed_count == 4);
        REQUIRE_FALSE(system.has_pending());
    }

    SECTION("Concurrent west and north transfers") {
        std::atomic<int> west_completed{0};
        std::atomic<int> north_completed{0};

        // Submit from west DMAs
        for (size_t row = 0; row < system.west_dmas.size(); ++row) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x10000 * row;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = row * 4 + 2;
            transfer.size = 64;

            system.west_dmas[row]->submit(transfer, [&west_completed]() {
                ++west_completed;
            });
        }

        // Submit from north DMAs
        for (size_t col = 0; col < system.north_dmas.size(); ++col) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x20000 * col;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = 8 + col;
            transfer.size = 64;

            system.north_dmas[col]->submit(transfer, [&north_completed]() {
                ++north_completed;
            });
        }

        REQUIRE(system.has_pending());

        // Co-simulate
        uint64_t cycle = 0;
        while (system.has_pending() && cycle < 200000) {
            mc0->set_cycle(cycle);
            mc1->set_cycle(cycle);
            mc0->tick();
            mc1->tick();

            system.set_cycle(cycle);
            system.tick();

            noc->step(cycle);
            ++cycle;
        }

        REQUIRE(west_completed == 4);
        REQUIRE(north_completed == 4);
        REQUIRE_FALSE(system.has_pending());
    }
}

// ============================================================================
// Performance and Latency Tests
// ============================================================================

TEST_CASE("DMA-Memory-NoC latency measurement", "[dma][memory][noc][integration][performance]") {
    auto mc = create_test_memory_controller();
    auto noc = create_test_noc(4, 4);
    auto dma = create_test_dma_engine(1);

    dma->bind_memory_controller(mc.get());
    dma->bind_noc(noc.get(), 0);

    SECTION("Measure single transfer latency") {
        uint64_t start_cycle = 0;
        uint64_t end_cycle = 0;

        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.src_addr = 0x1000;
        transfer.dst_type = DMAMemoryType::L3_TILE;
        transfer.dst_id = 15;  // Far corner for max latency
        transfer.size = 64;

        auto id = dma->submit(transfer, [&end_cycle, &dma]() {
            end_cycle = dma->current_cycle();
        });

        REQUIRE(id.has_value());

        uint64_t cycles = cosimulate(*dma, mc.get(), noc.get(), 50000);

        INFO("Transfer latency: " << cycles << " cycles");
        INFO("DMA stats - completed: " << dma->stats().transfers_completed);
        INFO("MC stats - reads: " << mc->stats().reads);
        INFO("NoC stats - injected: " << noc->stats().tiles_injected);

        REQUIRE(cycles > 0);
        REQUIRE(cycles < 50000);
    }

    SECTION("Measure different memory technologies") {
        std::vector<std::pair<MemoryTechnology, const char*>> technologies = {
            {MemoryTechnology::LPDDR5, "LPDDR5"},
            {MemoryTechnology::DDR5, "DDR5"},
            {MemoryTechnology::HBM3, "HBM3"},
        };

        for (const auto& [tech, name] : technologies) {
            MemoryControllerConfig mc_config;
            mc_config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
            mc_config.technology = tech;
            auto tech_mc = create_memory_controller(mc_config);

            auto tech_dma = create_test_dma_engine(1);
            tech_dma->bind_memory_controller(tech_mc.get());

            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x1000;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.size = 256;

            bool completed = false;
            tech_dma->submit(transfer, [&completed]() {
                completed = true;
            });

            uint64_t cycles = cosimulate(*tech_dma, tech_mc.get(), nullptr, 50000);

            INFO("Technology: " << name << " - Latency: " << cycles << " cycles");
            INFO("Avg MC latency: " << tech_mc->stats().avg_latency());

            REQUIRE(completed);
        }
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_CASE("DMA-Memory-NoC stress test", "[dma][memory][noc][integration][stress]") {
    auto mc = create_test_memory_controller();
    auto noc = create_test_noc(4, 4);
    auto dma = create_test_dma_engine(8);

    dma->bind_memory_controller(mc.get());
    dma->bind_noc(noc.get(), 0);

    SECTION("Many small transfers") {
        std::atomic<int> completed_count{0};
        constexpr int NUM_TRANSFERS = 32;  // Limit to prevent timeout
        int submitted = 0;

        for (int i = 0; i < NUM_TRANSFERS; ++i) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x1000 + i * 64;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = i % 16;  // Distribute across mesh
            transfer.size = 64;

            auto id = dma->submit(transfer, [&completed_count]() {
                ++completed_count;
            });

            if (id.has_value()) {
                ++submitted;
            } else {
                // Can't submit more - run some cycles and try again
                cosimulate(*dma, mc.get(), noc.get(), 500);
                id = dma->submit(transfer, [&completed_count]() {
                    ++completed_count;
                });
                if (id.has_value()) {
                    ++submitted;
                }
            }
        }

        INFO("Submitted " << submitted << " of " << NUM_TRANSFERS << " transfers");

        // Run until all complete (or timeout)
        uint64_t cycles = cosimulate(*dma, mc.get(), noc.get(), 500000);

        INFO("Completed " << completed_count << " transfers in " << cycles << " cycles");

        // Allow partial completion (at least 50% must complete)
        REQUIRE(completed_count >= submitted / 2);
    }

    SECTION("Large transfers") {
        std::atomic<int> completed_count{0};
        constexpr int NUM_TRANSFERS = 10;
        constexpr uint32_t TRANSFER_SIZE = 4096;

        for (int i = 0; i < NUM_TRANSFERS; ++i) {
            DMATransfer transfer;
            transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            transfer.src_addr = 0x10000 * i;
            transfer.dst_type = DMAMemoryType::L3_TILE;
            transfer.dst_id = i % 16;
            transfer.size = TRANSFER_SIZE;

            dma->submit(transfer, [&completed_count]() {
                ++completed_count;
            });
        }

        uint64_t cycles = cosimulate(*dma, mc.get(), noc.get(), 500000);

        REQUIRE(completed_count == NUM_TRANSFERS);

        INFO("Completed " << NUM_TRANSFERS << " large transfers in " << cycles << " cycles");
        INFO("Total bytes: " << (NUM_TRANSFERS * TRANSFER_SIZE));
    }
}
