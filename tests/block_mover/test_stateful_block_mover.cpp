// ============================================================================
// tests/block_mover/test_stateful_block_mover.cpp
// Tests for Stateful BlockMover and Tile Data Flow infrastructure
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>
#include <sw/kpu/models/temporal/datamovement/l3_interconnect.hpp>
#include <sw/kpu/models/temporal/datamovement/stateful_block_mover.hpp>
#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "sw/kpu/dataflow/block_mover_compiler.hpp"

using namespace sw::kpu;
using namespace sw::kpu::dataflow;

// ============================================================================
// BlockMover ISA Tests
// ============================================================================

TEST_CASE("BlockMover ISA - TileDescriptor", "[block_mover][isa]") {
    SECTION("Create and compare tile descriptors") {
        TileDescriptor tile1;
        tile1.tensor = TensorId::A;
        tile1.m_tile = 2;
        tile1.n_tile = 3;
        tile1.k_tile = 1;
        tile1.size = 1024;

        TileDescriptor tile2 = tile1;
        REQUIRE(tile1 == tile2);

        tile2.k_tile = 2;
        REQUIRE_FALSE(tile1 == tile2);
    }

    SECTION("TileDescriptor to_string") {
        TileDescriptor tile;
        tile.tensor = TensorId::B;
        tile.m_tile = 1;
        tile.n_tile = 2;
        tile.k_tile = 3;

        std::string s = tile.to_string();
        REQUIRE(s.find("B") != std::string::npos);
        REQUIRE(s.find("1") != std::string::npos);
        REQUIRE(s.find("2") != std::string::npos);
        REQUIRE(s.find("3") != std::string::npos);
    }
}

TEST_CASE("BlockMover ISA - Program Builder", "[block_mover][isa]") {
    SECTION("Build a simple program") {
        BlockMoverProgram prog;
        prog.l3_id = 5;

        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.size = 4096;

        prog.wait_trigger(TriggerChannel::DMA_LOAD_DONE)
            .push_to_l2(tile, 0)
            .emit_trigger(TriggerChannel::A_TILE_READY)
            .barrier();

        REQUIRE(prog.size() == 4);
        REQUIRE(prog.commands[0].op == BlockMoverOp::WAIT_TRIGGER);
        REQUIRE(prog.commands[1].op == BlockMoverOp::PUSH_TO_L2);
        REQUIRE(prog.commands[2].op == BlockMoverOp::EMIT_TRIGGER);
        REQUIRE(prog.commands[3].op == BlockMoverOp::BARRIER);
    }

    SECTION("Program with loops") {
        BlockMoverProgram prog;

        TileDescriptor tile;
        tile.tensor = TensorId::B;
        tile.size = 2048;

        prog.loop_start(4)
            .receive()
            .push_to_l2(tile, 0)
            .send_south(tile)
            .loop_end();

        prog.assemble();

        REQUIRE(prog.size() == 5);
        REQUIRE(prog.commands[0].op == BlockMoverOp::LOOP_START);
        REQUIRE(prog.commands[0].loop_count == 4);
        REQUIRE(prog.commands[4].op == BlockMoverOp::LOOP_END);
        // Loop end should jump back to loop start
        REQUIRE(prog.commands[4].jump_offset == -4);
    }
}

// ============================================================================
// L3 Interconnect Tests
// ============================================================================

TEST_CASE("L3 Interconnect - Configuration", "[block_mover][interconnect]") {
    SECTION("Default 4x4 mesh") {
        L3InterconnectConfig config;
        REQUIRE(config.rows == 4);
        REQUIRE(config.cols == 4);
        REQUIRE(config.num_tiles() == 16);
        REQUIRE(config.topology == InterconnectTopology::MESH_2D);
    }

    SECTION("Position conversion") {
        L3InterconnectConfig config;

        // ID 0 is at (0,0)
        L3Position pos0 = config.get_position(0);
        REQUIRE(pos0.row == 0);
        REQUIRE(pos0.col == 0);

        // ID 5 is at (1,1) in 4x4 mesh
        L3Position pos5 = config.get_position(5);
        REQUIRE(pos5.row == 1);
        REQUIRE(pos5.col == 1);

        // ID 15 is at (3,3)
        L3Position pos15 = config.get_position(15);
        REQUIRE(pos15.row == 3);
        REQUIRE(pos15.col == 3);
    }

    SECTION("Neighbor lookup") {
        L3InterconnectConfig config;

        // Tile 5 (center area) has all 4 neighbors
        REQUIRE(config.get_neighbor(5, Direction::NORTH) == 1);
        REQUIRE(config.get_neighbor(5, Direction::SOUTH) == 9);
        REQUIRE(config.get_neighbor(5, Direction::EAST) == 6);
        REQUIRE(config.get_neighbor(5, Direction::WEST) == 4);

        // Tile 0 (corner) has only SOUTH and EAST
        REQUIRE(config.get_neighbor(0, Direction::NORTH) == -1);
        REQUIRE(config.get_neighbor(0, Direction::WEST) == -1);
        REQUIRE(config.get_neighbor(0, Direction::SOUTH) == 4);
        REQUIRE(config.get_neighbor(0, Direction::EAST) == 1);
    }

    SECTION("Manhattan distance") {
        L3InterconnectConfig config;

        REQUIRE(config.manhattan_distance(0, 0) == 0);
        REQUIRE(config.manhattan_distance(0, 1) == 1);
        REQUIRE(config.manhattan_distance(0, 5) == 2);  // (0,0) to (1,1)
        REQUIRE(config.manhattan_distance(0, 15) == 6); // (0,0) to (3,3)
    }
}

TEST_CASE("L3 Interconnect - Routing", "[block_mover][interconnect]") {
    L3InterconnectConfig config;
    L3Interconnect interconnect(config);

    SECTION("XY routing next hop") {
        // From 0 to 15: should go EAST first (X routing)
        auto next = interconnect.route_next_hop(0, 15);
        REQUIRE(next.has_value());
        REQUIRE(*next == Direction::EAST);

        // From 3 to 15: should go SOUTH (already at right column)
        next = interconnect.route_next_hop(3, 15);
        REQUIRE(next.has_value());
        REQUIRE(*next == Direction::SOUTH);

        // From 12 to 15: should go EAST
        next = interconnect.route_next_hop(12, 15);
        REQUIRE(next.has_value());
        REQUIRE(*next == Direction::EAST);
    }

    SECTION("Transfer latency estimation") {
        // Same tile: 0 cycles
        uint32_t lat = interconnect.estimate_transfer_cycles(5, 5, 1024);
        REQUIRE(lat == 0);

        // Neighbor (1 hop)
        lat = interconnect.estimate_transfer_cycles(5, 6, 64);
        REQUIRE(lat > 0);

        // Distant tile (multiple hops)
        lat = interconnect.estimate_transfer_cycles(0, 15, 64);
        REQUIRE(lat > interconnect.estimate_transfer_cycles(0, 1, 64));
    }
}

// ============================================================================
// StatefulBlockMover Tests
// ============================================================================

TEST_CASE("StatefulBlockMover - Basic Execution", "[block_mover][stateful]") {
    SECTION("Empty program") {
        StatefulBlockMover mover;
        REQUIRE(mover.is_idle());
        REQUIRE(mover.program_size() == 0);

        bool running = mover.step(0);
        REQUIRE_FALSE(running);
        REQUIRE(mover.is_idle());
    }

    SECTION("Simple program execution") {
        StatefulBlockMoverConfig config;
        config.configure_from_mesh(5, 4, 4);
        StatefulBlockMover mover(config);

        BlockMoverProgram prog;
        prog.l3_id = 5;
        prog.append(BlockMoverOp::NOP);
        prog.append(BlockMoverOp::NOP);
        prog.append(BlockMoverOp::NOP);

        mover.load_program(prog);
        REQUIRE(mover.is_running());
        REQUIRE(mover.program_size() == 3);

        // Execute all NOPs
        mover.step(0);
        mover.step(1);
        mover.step(2);

        REQUIRE(mover.is_idle());
        REQUIRE(mover.stats().commands_executed == 3);
    }

    SECTION("Trigger wait and receive") {
        StatefulBlockMover mover;

        BlockMoverProgram prog;
        prog.wait_trigger(TriggerChannel::A_TILE_READY);
        prog.append(BlockMoverOp::NOP);

        mover.load_program(prog);

        // First step should set up wait
        mover.step(0);
        REQUIRE(mover.state() == BlockMoverState::WAITING_TRIGGER);

        // Receive trigger before next step
        mover.receive_trigger(TriggerChannel::A_TILE_READY);

        // Next step should process the trigger and continue
        mover.step(1);
        mover.step(2);  // Execute NOP
        mover.step(3);  // Should complete

        // Should eventually finish
        REQUIRE(mover.stats().triggers_received >= 1);
    }
}

TEST_CASE("StatefulBlockMover - Loop Execution", "[block_mover][stateful]") {
    StatefulBlockMover mover;

    BlockMoverProgram prog;
    prog.loop_start(3);
    prog.append(BlockMoverOp::NOP);
    prog.loop_end();
    prog.assemble();

    mover.load_program(prog);

    // Execute loop - give it enough cycles
    size_t cycles = 0;
    while (cycles < 50) {
        bool still_running = mover.step(cycles++);
        if (!still_running) break;
    }

    // Loop body (NOP) should execute 3 times
    // Plus LOOP_START and LOOP_END overhead
    REQUIRE(mover.stats().loops_executed >= 1);
    REQUIRE(mover.stats().commands_executed >= 3);  // At least the loop iterations
}

TEST_CASE("BlockMoverArray - Multi-tile coordination", "[block_mover][array]") {
    BlockMoverArray array;

    REQUIRE(array.all_idle());

    // Load simple programs to tiles 0 and 1
    std::vector<BlockMoverProgram> programs(2);

    programs[0].l3_id = 0;
    programs[0].append(BlockMoverOp::NOP);
    programs[0].emit_trigger(TriggerChannel::A_TILE_READY, 0x0002);  // Send to tile 1

    programs[1].l3_id = 1;
    programs[1].wait_trigger(TriggerChannel::A_TILE_READY);
    programs[1].append(BlockMoverOp::NOP);

    array.load_programs(programs);

    // Execute until done - give more cycles for trigger propagation
    size_t cycles = 0;
    while (cycles < 200) {
        array.step(cycles++);
        if (array.all_idle()) break;
    }

    auto stats = array.get_aggregate_stats();
    REQUIRE(stats.total_commands > 0);
}

// ============================================================================
// TileDataFlowGraph Tests
// ============================================================================

TEST_CASE("TileDataFlowGraph - Node Creation", "[dataflow][dfg]") {
    TileDataFlowGraph dfg;

    SECTION("Add DMA load node") {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.size = 4096;

        size_t id = dfg.add_dma_load(tile, 0);
        REQUIRE(id == 0);
        REQUIRE(dfg.num_nodes() == 1);

        const DFNode& node = dfg.node(0);
        REQUIRE(node.type == DFNodeType::DMA_LOAD);
        REQUIRE(node.l3_id == 0);
    }

    SECTION("Add multiple nodes with edges") {
        TileDescriptor a_tile, b_tile, c_tile;
        a_tile.tensor = TensorId::A;
        a_tile.size = 1024;
        b_tile.tensor = TensorId::B;
        b_tile.size = 1024;
        c_tile.tensor = TensorId::C;
        c_tile.size = 1024;

        size_t load_a = dfg.add_dma_load(a_tile, 0);
        size_t load_b = dfg.add_dma_load(b_tile, 0);
        size_t compute = dfg.add_matmul(0, 32, 32, 32, a_tile, b_tile, c_tile);

        dfg.add_edge(load_a, compute);
        dfg.add_edge(load_b, compute);

        REQUIRE(dfg.num_nodes() == 3);
        REQUIRE(dfg.num_edges() == 2);

        // Check predecessors
        REQUIRE(dfg.node(compute).predecessors.size() == 2);

        // Check sources and sinks
        auto sources = dfg.get_sources();
        REQUIRE(sources.size() == 2);

        auto sinks = dfg.get_sinks();
        REQUIRE(sinks.size() == 1);
        REQUIRE(sinks[0] == compute);
    }
}

TEST_CASE("TileDataFlowGraph - Topological Order", "[dataflow][dfg]") {
    TileDataFlowGraph dfg;

    // Create a diamond: A → B, A → C, B → D, C → D
    size_t a = dfg.add_barrier(0);
    size_t b = dfg.add_barrier(0);
    size_t c = dfg.add_barrier(0);
    size_t d = dfg.add_barrier(0);

    dfg.add_edge(a, b);
    dfg.add_edge(a, c);
    dfg.add_edge(b, d);
    dfg.add_edge(c, d);

    REQUIRE(dfg.is_acyclic());

    auto order = dfg.topological_order();
    REQUIRE(order.size() == 4);

    // A must come before B, C
    auto pos_a = std::find(order.begin(), order.end(), a) - order.begin();
    auto pos_b = std::find(order.begin(), order.end(), b) - order.begin();
    auto pos_c = std::find(order.begin(), order.end(), c) - order.begin();
    auto pos_d = std::find(order.begin(), order.end(), d) - order.begin();

    REQUIRE(pos_a < pos_b);
    REQUIRE(pos_a < pos_c);
    REQUIRE(pos_b < pos_d);
    REQUIRE(pos_c < pos_d);
}

TEST_CASE("TileDataFlowGraph - Build Matmul DFG", "[dataflow][dfg][matmul]") {
    SECTION("Simple 2x2 matmul") {
        TileDataFlowGraph dfg = TileDataFlowGraph::build_matmul_output_stationary(
            128, 128, 128,  // M, N, K
            2, 2, 2          // m_tiles, n_tiles, k_tiles
        );

        REQUIRE(dfg.num_nodes() > 0);
        REQUIRE(dfg.is_acyclic());

        // Should have DMA loads for A and B tiles
        auto dma_loads = dfg.get_nodes_by_type(DFNodeType::DMA_LOAD);
        REQUIRE(dma_loads.size() > 0);

        // Should have compute nodes
        auto computes = dfg.get_nodes_by_type(DFNodeType::MATMUL);
        REQUIRE(computes.size() == 2 * 2 * 2);  // m × n × k

        // Should have DMA stores for C tiles
        auto dma_stores = dfg.get_nodes_by_type(DFNodeType::DMA_STORE);
        REQUIRE(dma_stores.size() == 2 * 2);  // m × n
    }

    SECTION("Critical path") {
        TileDataFlowGraph dfg = TileDataFlowGraph::build_matmul_output_stationary(
            64, 64, 64, 2, 2, 1);

        int64_t cp = dfg.critical_path_length();
        REQUIRE(cp > 0);
    }
}

// ============================================================================
// BlockMoverCompiler Tests
// ============================================================================

TEST_CASE("BlockMoverCompiler - Basic Compilation", "[dataflow][compiler]") {
    SECTION("Compile simple DFG") {
        TileDataFlowGraph dfg;

        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.size = 4096;

        dfg.add_dma_load(tile, 0);

        BlockMoverCompiler compiler;
        CompiledSchedule schedule = compiler.compile(dfg);

        REQUIRE(schedule.has_work());
        REQUIRE(schedule.stats.total_commands > 0);
    }

    SECTION("Compile matmul") {
        TileDataFlowGraph dfg = TileDataFlowGraph::build_matmul_output_stationary(
            64, 64, 64, 2, 2, 1);

        BlockMoverCompiler compiler;
        CompiledSchedule schedule = compiler.compile(dfg);

        REQUIRE(schedule.has_work());
        REQUIRE(schedule.estimated_cycles > 0);
        REQUIRE(schedule.compute_cycles > 0);

        // Check that multiple L3 tiles have programs
        auto active = schedule.active_programs();
        REQUIRE(active.size() > 0);
    }
}

TEST_CASE("BlockMoverCompiler - Convenience Function", "[dataflow][compiler]") {
    auto schedule = compile_matmul(
        256, 256, 256,  // M, N, K
        4, 4, 4          // tiles
    );

    REQUIRE(schedule.has_work());
    REQUIRE(schedule.stats.total_compute_ops == 4 * 4 * 4);
    REQUIRE(schedule.stats.total_dma_ops > 0);
}

// ============================================================================
// DFG Scheduler Tests
// ============================================================================

TEST_CASE("DFGScheduler - ASAP Scheduling", "[dataflow][scheduler]") {
    TileDataFlowGraph dfg = TileDataFlowGraph::build_matmul_output_stationary(
        64, 64, 64, 2, 2, 1);

    DFGScheduler::Config config;
    config.algorithm = DFGScheduler::Algorithm::ASAP;
    DFGScheduler scheduler(config);

    DFGSchedule schedule = scheduler.schedule(dfg);

    REQUIRE(schedule.makespan > 0);
    REQUIRE(schedule.nodes.size() == dfg.num_nodes());

    // Validate schedule
    REQUIRE(schedule.validate(dfg));
}

TEST_CASE("DFGScheduler - List Scheduling", "[dataflow][scheduler]") {
    TileDataFlowGraph dfg = TileDataFlowGraph::build_matmul_output_stationary(
        128, 128, 128, 4, 4, 2);

    DFGScheduler scheduler;  // Default: LIST_SCHEDULING
    DFGSchedule schedule = scheduler.schedule(dfg);

    REQUIRE(schedule.makespan > 0);
    REQUIRE(schedule.validate(dfg));

    // Compare with ASAP scheduling (both should produce valid schedules)
    DFGScheduler::Config asap_config;
    asap_config.algorithm = DFGScheduler::Algorithm::ASAP;
    DFGScheduler asap_scheduler(asap_config);
    DFGSchedule asap_schedule = asap_scheduler.schedule(dfg);

    // Both should produce valid schedules
    REQUIRE(asap_schedule.makespan > 0);
    REQUIRE(asap_schedule.validate(dfg));
}
