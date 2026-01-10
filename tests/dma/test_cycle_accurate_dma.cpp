// Test suite for Cycle-Accurate DMA Engine
// Tests DMA channel state machine, memory binding, and NoC integration

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/components/dma/cycle_accurate_dma_engine.hpp>
#include <sw/kpu/components/dma/dma_placement.hpp>
#include <sw/kpu/components/dma/dma_engine_interface.hpp>

using namespace sw::kpu;

// ============================================================================
// Basic DMA Engine Tests
// ============================================================================

TEST_CASE("CycleAccurateDMAEngine creation", "[dma][cycle_accurate]") {
    CycleAccurateDMAConfig config;
    config.num_channels = 4;
    config.queue_depth_per_channel = 8;

    SECTION("Engine creates successfully") {
        CycleAccurateDMAEngine engine(config);

        REQUIRE(engine.fidelity() == SimulationFidelity::CYCLE_ACCURATE);
        REQUIRE(engine.num_channels() == 4);
        REQUIRE(engine.current_cycle() == 0);
        REQUIRE_FALSE(engine.has_pending());
    }

    SECTION("Engine can be created via factory") {
        DMAEngineConfig base_config;
        base_config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        base_config.num_channels = 2;

        auto engine = create_dma_engine(base_config);
        REQUIRE(engine != nullptr);
        REQUIRE(engine->fidelity() == SimulationFidelity::CYCLE_ACCURATE);
    }
}

TEST_CASE("CycleAccurateDMAEngine submit and channel state", "[dma][cycle_accurate]") {
    CycleAccurateDMAConfig config;
    config.num_channels = 2;
    config.queue_depth_per_channel = 4;

    CycleAccurateDMAEngine engine(config);

    SECTION("Submit transfer assigns to idle channel") {
        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.dst_type = DMAMemoryType::L3_TILE;
        transfer.size = 1024;

        REQUIRE(engine.can_accept());

        auto id = engine.submit(transfer);
        REQUIRE(id.has_value());
        REQUIRE(engine.has_pending());
        REQUIRE(engine.pending_count() == 1);
    }

    SECTION("Multiple transfers use different channels") {
        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.dst_type = DMAMemoryType::L3_TILE;
        transfer.size = 512;

        auto id1 = engine.submit(transfer);
        auto id2 = engine.submit(transfer);

        REQUIRE(id1.has_value());
        REQUIRE(id2.has_value());
        REQUIRE(*id1 != *id2);
        REQUIRE(engine.pending_count() == 2);
    }

    SECTION("Submit with callback") {
        bool callback_invoked = false;

        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.dst_type = DMAMemoryType::L3_TILE;
        transfer.size = 256;

        auto id = engine.submit(transfer, [&callback_invoked]() {
            callback_invoked = true;
        });

        REQUIRE(id.has_value());

        // Drain to complete transfer
        engine.drain();

        REQUIRE(callback_invoked);
        REQUIRE_FALSE(engine.has_pending());
    }
}

TEST_CASE("CycleAccurateDMAEngine tick and state machine", "[dma][cycle_accurate]") {
    CycleAccurateDMAConfig config;
    config.num_channels = 1;

    CycleAccurateDMAEngine engine(config);

    SECTION("Tick advances cycle") {
        REQUIRE(engine.current_cycle() == 0);

        engine.tick();
        REQUIRE(engine.current_cycle() == 1);

        engine.tick();
        engine.tick();
        REQUIRE(engine.current_cycle() == 3);
    }

    SECTION("Transfer completes after ticks") {
        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.dst_type = DMAMemoryType::L3_TILE;
        transfer.size = 64;

        bool completed = false;
        engine.submit(transfer, [&completed]() { completed = true; });

        // Without memory controller or NoC, transfer should complete quickly
        int max_ticks = 100;
        while (!completed && max_ticks-- > 0) {
            engine.tick();
        }

        REQUIRE(completed);
        REQUIRE_FALSE(engine.has_pending());
    }
}

TEST_CASE("CycleAccurateDMAEngine channel states", "[dma][cycle_accurate]") {
    CycleAccurateDMAConfig config;
    config.num_channels = 4;

    CycleAccurateDMAEngine engine(config);

    SECTION("All channels start IDLE") {
        for (uint32_t i = 0; i < 4; ++i) {
            REQUIRE(engine.get_channel_state(i) == IDMAEngine::ChannelState::IDLE);
            REQUIRE(engine.get_detailed_channel_state(i) == DMAChannelState::IDLE);
        }
    }

    SECTION("Channel becomes BUSY after submit") {
        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.size = 128;

        engine.submit(transfer);

        // At least one channel should not be IDLE
        bool any_non_idle = false;
        for (uint32_t i = 0; i < 4; ++i) {
            if (engine.get_detailed_channel_state(i) != DMAChannelState::IDLE) {
                any_non_idle = true;
                break;
            }
        }
        REQUIRE(any_non_idle);
    }

    SECTION("Invalid channel returns ERROR") {
        REQUIRE(engine.get_channel_state(100) == IDMAEngine::ChannelState::ERROR);
        REQUIRE(engine.get_detailed_channel_state(100) == DMAChannelState::ERROR);
    }
}

TEST_CASE("CycleAccurateDMAEngine reset", "[dma][cycle_accurate]") {
    CycleAccurateDMAConfig config;
    config.num_channels = 2;

    CycleAccurateDMAEngine engine(config);

    // Submit some transfers
    DMATransfer transfer;
    transfer.src_type = DMAMemoryType::DEVICE_MEMORY;
    transfer.size = 100;

    engine.submit(transfer);
    engine.submit(transfer);
    engine.tick();
    engine.tick();

    REQUIRE(engine.current_cycle() > 0);

    // Reset
    engine.reset();

    REQUIRE(engine.current_cycle() == 0);
    REQUIRE_FALSE(engine.has_pending());
    REQUIRE(engine.pending_count() == 0);

    for (uint32_t i = 0; i < 2; ++i) {
        REQUIRE(engine.get_channel_state(i) == IDMAEngine::ChannelState::IDLE);
    }
}

TEST_CASE("CycleAccurateDMAEngine statistics", "[dma][cycle_accurate]") {
    CycleAccurateDMAConfig config;
    config.num_channels = 2;

    CycleAccurateDMAEngine engine(config);

    SECTION("Statistics initially zero") {
        const auto& stats = engine.stats();
        REQUIRE(stats.transfers_completed == 0);
        REQUIRE(stats.bytes_transferred == 0);
    }

    SECTION("Statistics update after transfer") {
        DMATransfer transfer;
        transfer.src_type = DMAMemoryType::HOST_MEMORY;
        transfer.dst_type = DMAMemoryType::DEVICE_MEMORY;
        transfer.size = 1024;

        engine.submit(transfer);
        engine.drain();

        const auto& stats = engine.stats();
        REQUIRE(stats.transfers_completed == 1);
        REQUIRE(stats.bytes_transferred == 1024);
        REQUIRE(stats.bytes_host_to_device == 1024);
    }

    SECTION("Reset statistics") {
        DMATransfer transfer;
        transfer.size = 512;
        engine.submit(transfer);
        engine.drain();

        engine.reset_stats();

        const auto& stats = engine.stats();
        REQUIRE(stats.transfers_completed == 0);
        REQUIRE(stats.bytes_transferred == 0);
    }
}

TEST_CASE("CycleAccurateDMAEngine cancel transfer", "[dma][cycle_accurate]") {
    CycleAccurateDMAConfig config;
    config.num_channels = 2;

    CycleAccurateDMAEngine engine(config);

    DMATransfer transfer;
    transfer.size = 256;

    auto id = engine.submit(transfer);
    REQUIRE(id.has_value());
    REQUIRE(engine.pending_count() == 1);

    bool cancelled = engine.cancel(*id);
    REQUIRE(cancelled);
    REQUIRE(engine.pending_count() == 0);
}

// ============================================================================
// DMA Placement Tests
// ============================================================================

TEST_CASE("DMAPlacement configuration", "[dma][placement]") {
    SECTION("Standard 4x4 mesh placement") {
        auto config = DMASystemConfig::create_standard(4, 4, 2);

        // Should have 4 west DMAs (one per row)
        REQUIRE(config.west_edge_dmas.size() == 4);

        // Should have 4 north DMAs (one per column)
        REQUIRE(config.north_edge_dmas.size() == 4);

        REQUIRE(config.total_dmas() == 8);
    }

    SECTION("Router IDs are correct") {
        auto config = DMASystemConfig::create_standard(4, 4, 1);

        // West edge: router IDs should be 0, 4, 8, 12 (column 0)
        REQUIRE(config.west_edge_dmas[0].router_id == 0);
        REQUIRE(config.west_edge_dmas[1].router_id == 4);
        REQUIRE(config.west_edge_dmas[2].router_id == 8);
        REQUIRE(config.west_edge_dmas[3].router_id == 12);

        // North edge: router IDs should be 0, 1, 2, 3 (row 0)
        REQUIRE(config.north_edge_dmas[0].router_id == 0);
        REQUIRE(config.north_edge_dmas[1].router_id == 1);
        REQUIRE(config.north_edge_dmas[2].router_id == 2);
        REQUIRE(config.north_edge_dmas[3].router_id == 3);
    }

    SECTION("Memory controller round-robin assignment") {
        auto config = DMASystemConfig::create_standard(4, 4, 2);

        // Memory controllers should alternate
        REQUIRE(config.west_edge_dmas[0].memory_controller_id == 0);
        REQUIRE(config.west_edge_dmas[1].memory_controller_id == 1);
        REQUIRE(config.west_edge_dmas[2].memory_controller_id == 0);
        REQUIRE(config.west_edge_dmas[3].memory_controller_id == 1);
    }

    SECTION("Minimal configuration") {
        auto config = DMASystemConfig::create_minimal(4, 4);

        REQUIRE(config.west_edge_dmas.size() == 1);
        REQUIRE(config.north_edge_dmas.size() == 1);
        REQUIRE(config.total_dmas() == 2);
    }

    SECTION("Compute router ID helper") {
        // West edge
        REQUIRE(DMAPlacement::compute_router_id(
            DMAPlacement::Edge::WEST, 0, 4, 4) == 0);
        REQUIRE(DMAPlacement::compute_router_id(
            DMAPlacement::Edge::WEST, 3, 4, 4) == 12);

        // North edge
        REQUIRE(DMAPlacement::compute_router_id(
            DMAPlacement::Edge::NORTH, 0, 4, 4) == 0);
        REQUIRE(DMAPlacement::compute_router_id(
            DMAPlacement::Edge::NORTH, 3, 4, 4) == 3);

        // East edge
        REQUIRE(DMAPlacement::compute_router_id(
            DMAPlacement::Edge::EAST, 0, 4, 4) == 3);
        REQUIRE(DMAPlacement::compute_router_id(
            DMAPlacement::Edge::EAST, 3, 4, 4) == 15);

        // South edge
        REQUIRE(DMAPlacement::compute_router_id(
            DMAPlacement::Edge::SOUTH, 0, 4, 4) == 12);
        REQUIRE(DMAPlacement::compute_router_id(
            DMAPlacement::Edge::SOUTH, 3, 4, 4) == 15);
    }
}

// ============================================================================
// DMA System Tests
// ============================================================================

TEST_CASE("DMASystem operations", "[dma][system]") {
    CycleAccurateDMAConfig engine_config;
    engine_config.num_channels = 2;

    auto placement = DMASystemConfig::create_minimal(4, 4);

    // Create system without memory controllers or NoC for basic tests
    auto system = create_dma_system(placement, engine_config, {}, nullptr);

    SECTION("System has correct number of DMAs") {
        REQUIRE(system.west_dmas.size() == 1);
        REQUIRE(system.north_dmas.size() == 1);
        REQUIRE(system.size() == 2);
    }

    SECTION("System tick advances all engines") {
        system.west_dmas[0]->submit(DMATransfer{});
        system.north_dmas[0]->submit(DMATransfer{});

        REQUIRE(system.has_pending());

        system.tick();
        system.tick();

        // All engines should have advanced
        REQUIRE(system.west_dmas[0]->current_cycle() == 2);
        REQUIRE(system.north_dmas[0]->current_cycle() == 2);
    }

    SECTION("System drain completes all transfers") {
        system.west_dmas[0]->submit(DMATransfer{});
        system.north_dmas[0]->submit(DMATransfer{});

        REQUIRE(system.has_pending());

        system.drain();

        REQUIRE_FALSE(system.has_pending());
    }

    SECTION("System reset clears all state") {
        system.west_dmas[0]->submit(DMATransfer{});
        system.tick();
        system.tick();

        system.reset();

        REQUIRE(system.west_dmas[0]->current_cycle() == 0);
        REQUIRE_FALSE(system.has_pending());
    }

    SECTION("System set_cycle synchronizes all engines") {
        system.set_cycle(100);

        REQUIRE(system.west_dmas[0]->current_cycle() == 100);
        REQUIRE(system.north_dmas[0]->current_cycle() == 100);
    }
}

// ============================================================================
// Memory Binding Tests (without actual memory controller)
// ============================================================================

TEST_CASE("CycleAccurateDMAEngine binding", "[dma][cycle_accurate][binding]") {
    CycleAccurateDMAConfig config;
    config.num_channels = 1;

    CycleAccurateDMAEngine engine(config);

    SECTION("Initially not bound") {
        REQUIRE_FALSE(engine.has_memory_binding());
        REQUIRE_FALSE(engine.has_noc_binding());
    }

    SECTION("Transfer works without bindings") {
        // Should still complete (using default behavior)
        DMATransfer transfer;
        transfer.size = 64;

        bool completed = false;
        engine.submit(transfer, [&completed]() { completed = true; });

        engine.drain();
        REQUIRE(completed);
    }
}
