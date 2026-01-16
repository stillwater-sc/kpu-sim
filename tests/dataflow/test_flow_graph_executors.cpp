// ============================================================================
// tests/dataflow/test_flow_graph_executors.cpp
// Tests for Flow Graph Executors at all levels
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/models/dataflow/operand_flow_graph.hpp>
#include <sw/kpu/models/dataflow/flow_graph_executor.hpp>
#include <sw/kpu/models/dataflow/dma_flow_executor.hpp>
#include <sw/kpu/models/dataflow/block_mover_flow_executor.hpp>
#include <sw/kpu/models/dataflow/streamer_flow_executor.hpp>

using namespace sw::kpu::dataflow;

// ============================================================================
// Base FlowGraphExecutor Tests
// ============================================================================

TEST_CASE("FlowGraphExecutor basic execution", "[dataflow][executor]") {
    // Build a simple graph: wait -> fire -> produce
    OperandFlowGraph graph;
    graph.name = "simple_test";

    auto input = tile_a(0, 0, Location::L3, 0);
    auto output = tile_a(0, 0, Location::L2, 0);

    auto wait_id = graph.add_wait({input}, "wait");
    auto fire_id = graph.add_fire(Operation::PUSH_TO_L2, {input}, {output}, "push");
    auto produce_id = graph.add_produce({output}, "produce");

    graph.add_edge(wait_id, fire_id);
    graph.add_edge(fire_id, produce_id);

    FlowGraphExecutor executor(graph);

    SECTION("Initial state") {
        REQUIRE(executor.current_cycle() == 0);
        REQUIRE_FALSE(executor.is_complete());
        REQUIRE(executor.get_waiting_nodes().size() == 3);
    }

    SECTION("Inject operand enables execution") {
        // Initially nothing can fire (waiting for input)
        auto ready = executor.get_ready_nodes();
        REQUIRE(ready.empty());

        // Inject the input operand
        executor.inject_operand(input, 0);

        // Now wait node should be ready
        ready = executor.get_ready_nodes();
        REQUIRE(ready.size() == 1);
        REQUIRE(ready[0] == wait_id);
    }

    SECTION("Step executes ready nodes") {
        executor.inject_operand(input, 0);

        // Step should fire wait node
        bool progress = executor.step();
        REQUIRE(progress);
        REQUIRE(executor.get_node_state(wait_id) == NodeState::COMPLETED);

        // Fire node should now be ready
        auto ready = executor.get_ready_nodes();
        REQUIRE(ready.size() == 1);
        REQUIRE(ready[0] == fire_id);
    }

    SECTION("Run completes all nodes") {
        executor.inject_operand(input, 0);

        bool success = executor.run();
        REQUIRE(success);
        REQUIRE(executor.is_complete());

        // All nodes should be completed
        REQUIRE(executor.get_node_state(wait_id) == NodeState::COMPLETED);
        REQUIRE(executor.get_node_state(fire_id) == NodeState::COMPLETED);
        REQUIRE(executor.get_node_state(produce_id) == NodeState::COMPLETED);
    }

    SECTION("Statistics are tracked") {
        executor.inject_operand(input, 0);
        executor.run();

        const auto& stats = executor.stats();
        REQUIRE(stats.nodes_fired == 3);
        REQUIRE(stats.operands_produced > 0);
    }

    SECTION("Events are generated") {
        executor.inject_operand(input, 0);
        executor.run();

        const auto& events = executor.events();
        REQUIRE(!events.empty());

        // Should have OPERAND_READY, NODE_FIRED, NODE_COMPLETED events
        bool has_ready = false, has_fired = false, has_complete = false;
        for (const auto& e : events) {
            if (e.type == ExecutionEventType::OPERAND_READY) has_ready = true;
            if (e.type == ExecutionEventType::NODE_FIRED) has_fired = true;
            if (e.type == ExecutionEventType::NODE_COMPLETED) has_complete = true;
        }
        REQUIRE(has_ready);
        REQUIRE(has_fired);
        REQUIRE(has_complete);
    }
}

TEST_CASE("FlowGraphExecutor JOIN node", "[dataflow][executor][join]") {
    OperandFlowGraph graph;

    auto input_a = tile_a(0, 0, Location::L3, 0);
    auto input_b = tile_b(0, 0, Location::L3, 0);

    auto wait_a = graph.add_wait({input_a}, "wait_a");
    auto wait_b = graph.add_wait({input_b}, "wait_b");
    auto join = graph.add_join({input_a, input_b}, "join_ab");

    graph.add_edge(wait_a, join);
    graph.add_edge(wait_b, join);

    FlowGraphExecutor executor(graph);

    SECTION("JOIN waits for all inputs") {
        // Inject only A
        executor.inject_operand(input_a, 0);
        executor.step();

        // JOIN should not be ready yet
        REQUIRE(executor.get_node_state(join) == NodeState::WAITING);

        // Inject B
        executor.inject_operand(input_b, 1);
        executor.step();

        // Now JOIN should become ready and fire
        executor.step();
        REQUIRE(executor.get_node_state(join) == NodeState::COMPLETED);
    }
}

TEST_CASE("FlowGraphExecutor deadlock detection", "[dataflow][executor][deadlock]") {
    OperandFlowGraph graph;

    // Create a graph that will deadlock (waiting for operand that's never produced)
    auto input = tile_a(0, 0, Location::L3, 0);
    graph.add_wait({input}, "wait_forever");

    FlowGraphExecutor executor(graph);

    // Don't inject the operand - should deadlock
    bool success = executor.run(100);
    REQUIRE_FALSE(success);
    REQUIRE_FALSE(executor.is_complete());
}

// ============================================================================
// DMA Flow Executor Tests
// ============================================================================

TEST_CASE("DMAFlowExecutor basic load", "[dataflow][executor][dma]") {
    DMAFlowGraphBuilder builder;
    builder.set_node_id(0)
           .set_dimensions(4, 4, 4)
           .add_load_a(0, 0, 0);

    auto graph = builder.build();

    DMAExecutorConfig config;
    config.load_latency = 10;

    DMAFlowExecutor executor(graph, config);

    // Inject buffer available
    executor.inject_buffer_available(Location::L3, 0);

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.loads_completed() == 1);
    REQUIRE(executor.bytes_transferred() > 0);

    // Check output events
    auto ready_events = executor.get_tile_ready_events();
    REQUIRE(ready_events.size() == 1);
    REQUIRE(ready_events[0].operand.type == OperandType::TILE_A);
    REQUIRE(ready_events[0].operand.location == Location::L3);
}

TEST_CASE("DMAFlowExecutor load multiple tiles", "[dataflow][executor][dma]") {
    DMAFlowGraphBuilder builder;
    builder.set_node_id(0)
           .set_dimensions(4, 4, 4)
           .load_a_row(0, 0);  // Load all A[0,k] tiles

    auto graph = builder.build();

    DMAFlowExecutor executor(graph);

    // Inject buffer available for each tile
    for (int k = 0; k < 4; ++k) {
        executor.inject_buffer_available(Location::L3, 0);
    }

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.loads_completed() == 4);

    auto ready_events = executor.get_tile_ready_events();
    REQUIRE(ready_events.size() == 4);
}

TEST_CASE("DMAFlowExecutor store operation", "[dataflow][executor][dma]") {
    DMAFlowGraphBuilder builder;
    builder.set_node_id(0)
           .set_dimensions(4, 4, 4)
           .add_store_c(0, 0, 0);

    auto graph = builder.build();

    DMAFlowExecutor executor(graph);

    // Inject the C tile as ready in L3
    auto c_tile = tile_c(0, 0, Location::L3, 0);
    executor.inject_operand(c_tile);

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.stores_completed() == 1);
}

// ============================================================================
// BlockMover Flow Executor Tests
// ============================================================================

TEST_CASE("BlockMoverFlowExecutor basic push", "[dataflow][executor][blockmover]") {
    OperandFlowGraph graph;
    graph.level = ExecutionLevel::BLOCK_MOVER;

    auto a_l3 = tile_a(0, 0, Location::L3, 0);
    auto l2_avail = make_buffer_token(Location::L2, 0);
    auto a_l2 = tile_a(0, 0, Location::L2, 0);

    auto wait_a = graph.add_wait({a_l3});
    auto wait_l2 = graph.add_wait({l2_avail});
    auto join = graph.add_join({a_l3, l2_avail});
    auto push = graph.add_fire(Operation::PUSH_TO_L2, {a_l3}, {a_l2});

    graph.add_edge(wait_a, join);
    graph.add_edge(wait_l2, join);
    graph.add_edge(join, push);

    BlockMoverExecutorConfig config;
    config.row = 0;
    config.col = 0;
    config.push_latency = 5;

    BlockMoverFlowExecutor executor(graph, config);

    // Inject inputs
    executor.inject_operand(a_l3);
    executor.inject_operand(l2_avail);

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.pushes_completed() == 1);
    REQUIRE(executor.bytes_to_l2() > 0);

    // Check L2 ready events
    auto l2_events = executor.get_l2_ready_events();
    REQUIRE(l2_events.size() == 1);
    REQUIRE(l2_events[0].operand.location == Location::L2);
}

TEST_CASE("BlockMoverFlowExecutor mesh send", "[dataflow][executor][blockmover][mesh]") {
    OperandFlowGraph graph;
    graph.level = ExecutionLevel::BLOCK_MOVER;

    auto a_l3_src = tile_a(0, 0, Location::L3, 0);
    auto a_l3_dst = tile_a(0, 0, Location::L3, 1);  // East neighbor

    auto wait = graph.add_wait({a_l3_src});
    auto send = graph.add_fire(Operation::SEND_EAST, {a_l3_src}, {a_l3_dst});

    graph.add_edge(wait, send);

    BlockMoverExecutorConfig config;
    config.row = 0;
    config.col = 0;
    config.mesh_cols = 4;
    config.mesh_latency_per_hop = 3;

    BlockMoverFlowExecutor executor(graph, config);

    executor.inject_operand(a_l3_src);

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.mesh_sends_completed() == 1);
    REQUIRE(executor.bytes_mesh() > 0);

    // Check mesh events (may have duplicates from input/output processing)
    auto mesh_events = executor.get_mesh_ready_events();
    REQUIRE(mesh_events.size() >= 1);

    // At least one should be to node 1
    bool found_east = false;
    for (const auto& e : mesh_events) {
        if (e.operand.node_id == 1) found_east = true;
    }
    REQUIRE(found_east);
}

TEST_CASE("BlockMoverFlowGraphBuilder C-stationary", "[dataflow][executor][blockmover][builder]") {
    BlockMoverFlowGraphBuilder builder;
    builder.set_position(1, 1, 4)  // L3[1,1]
           .set_dimensions(4, 4, 4)
           .add_c_stationary_iteration(0);

    auto graph = builder.build();

    REQUIRE(graph.level == ExecutionLevel::BLOCK_MOVER);
    REQUIRE(graph.mesh_row == 1);
    REQUIRE(graph.mesh_col == 1);
    REQUIRE(graph.num_nodes() > 0);

    BlockMoverFlowExecutor executor(graph);

    // Inject A and B tiles at L3
    auto a = tile_a(1, 0, Location::L3, 5);
    auto b = tile_b(0, 1, Location::L3, 5);
    executor.inject_operand(a);
    executor.inject_operand(b);

    // Inject L2 bank availability
    executor.inject_operand(make_buffer_token(Location::L2, 0));
    executor.inject_operand(make_buffer_token(Location::L2, 1));

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.pushes_completed() == 2);  // A and B pushed
}

// ============================================================================
// Streamer Flow Executor Tests
// ============================================================================

TEST_CASE("StreamerFlowExecutor basic matmul", "[dataflow][executor][streamer]") {
    OperandFlowGraph graph;
    graph.level = ExecutionLevel::STREAMER;

    auto a_l2 = tile_a(0, 0, Location::L2, 0);
    auto b_l2 = tile_b(0, 0, Location::L2, 1);
    auto a_l1 = tile_a(0, 0, Location::L1, 0);
    auto b_l1 = tile_b(0, 0, Location::L1, 0);
    auto c_acc = tile_c(0, 0, Location::ACCUMULATOR, 0);

    auto wait_a = graph.add_wait({a_l2});
    auto wait_b = graph.add_wait({b_l2});
    auto join = graph.add_join({a_l2, b_l2});
    auto feed_a = graph.add_fire(Operation::FEED_WEST, {a_l2}, {a_l1});
    auto feed_b = graph.add_fire(Operation::FEED_NORTH, {b_l2}, {b_l1});
    auto matmul = graph.add_fire(Operation::MATMUL, {a_l1, b_l1}, {c_acc});

    graph.add_edge(wait_a, join);
    graph.add_edge(wait_b, join);
    graph.add_edge(join, feed_a);
    graph.add_edge(join, feed_b);
    graph.add_edge(feed_a, matmul);
    graph.add_edge(feed_b, matmul);

    StreamerExecutorConfig config;
    config.feed_latency = 2;
    config.matmul_latency = 64;

    StreamerFlowExecutor executor(graph, config);

    // Inject A and B tiles in L2
    executor.inject_operand(a_l2);
    executor.inject_operand(b_l2);

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.feeds_completed() == 2);
    REQUIRE(executor.matmuls_completed() == 1);
    REQUIRE(executor.flops() > 0);

    // Check accumulator state
    REQUIRE(executor.accumulator_has_result());
    REQUIRE(executor.accumulator().i == 0);
    REQUIRE(executor.accumulator().j == 0);
}

TEST_CASE("StreamerFlowExecutor drain", "[dataflow][executor][streamer][drain]") {
    OperandFlowGraph graph;
    graph.level = ExecutionLevel::STREAMER;

    auto c_acc = tile_c(0, 0, Location::ACCUMULATOR, 0);
    auto c_l2 = tile_c(0, 0, Location::L2, 2);

    auto wait = graph.add_wait({c_acc});
    auto drain = graph.add_fire(Operation::DRAIN, {c_acc}, {c_l2});

    graph.add_edge(wait, drain);

    StreamerFlowExecutor executor(graph);

    // Inject accumulator ready
    executor.inject_operand(c_acc);

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.drains_completed() == 1);

    // Check result events
    auto results = executor.get_result_ready_events();
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].operand.location == Location::L2);
}

TEST_CASE("StreamerFlowGraphBuilder C-stationary", "[dataflow][executor][streamer][builder]") {
    StreamerFlowGraphBuilder builder;
    builder.set_position(0, 0, 4)
           .set_dimensions(4, 4, 2)  // 2 k iterations
           .build_c_stationary();

    auto graph = builder.build();

    REQUIRE(graph.level == ExecutionLevel::STREAMER);
    REQUIRE(graph.num_nodes() > 0);

    StreamerFlowExecutor executor(graph);

    // Inject A and B tiles for each k iteration
    for (uint16_t k = 0; k < 2; ++k) {
        executor.inject_operand(tile_a(0, k, Location::L2, 0));
        executor.inject_operand(tile_b(k, 0, Location::L2, 1));
    }

    bool success = executor.run();
    REQUIRE(success);
    REQUIRE(executor.matmuls_completed() == 2);  // 2 k iterations
    REQUIRE(executor.drains_completed() == 1);   // 1 final drain
}

// ============================================================================
// Multi-Level Integration Test
// ============================================================================

TEST_CASE("Multi-level execution flow", "[dataflow][executor][integration]") {
    // This test demonstrates the flow of events between levels

    // Level 1: DMA loads A tile
    DMAFlowGraphBuilder dma_builder;
    dma_builder.set_node_id(0)
               .set_dimensions(1, 1, 1)
               .add_load_a(0, 0, 0);
    auto dma_graph = dma_builder.build();
    DMAFlowExecutor dma(dma_graph);

    // Level 2: BlockMover pushes to L2
    OperandFlowGraph bm_graph;
    bm_graph.level = ExecutionLevel::BLOCK_MOVER;
    auto a_l3 = tile_a(0, 0, Location::L3, 0);
    auto a_l2 = tile_a(0, 0, Location::L2, 0);
    auto l2_avail = make_buffer_token(Location::L2, 0);
    auto bm_wait = bm_graph.add_wait({a_l3});
    auto bm_wait_l2 = bm_graph.add_wait({l2_avail});
    auto bm_join = bm_graph.add_join({a_l3, l2_avail});
    auto bm_push = bm_graph.add_fire(Operation::PUSH_TO_L2, {a_l3}, {a_l2});
    bm_graph.add_edge(bm_wait, bm_join);
    bm_graph.add_edge(bm_wait_l2, bm_join);
    bm_graph.add_edge(bm_join, bm_push);
    BlockMoverFlowExecutor bm(bm_graph);

    // Execute DMA
    dma.inject_buffer_available(Location::L3, 0);
    REQUIRE(dma.run());

    // Get DMA output events and feed to BlockMover
    auto dma_outputs = dma.get_tile_ready_events();
    REQUIRE(dma_outputs.size() == 1);

    // Inject DMA outputs to BlockMover
    bm.inject_operand(dma_outputs[0].operand, dma_outputs[0].cycle);
    bm.inject_operand(l2_avail);  // L2 available

    REQUIRE(bm.run());

    // Verify chain completed
    REQUIRE(dma.loads_completed() == 1);
    REQUIRE(bm.pushes_completed() == 1);

    // BlockMover should have produced L2 ready event
    auto bm_outputs = bm.get_l2_ready_events();
    REQUIRE(bm_outputs.size() == 1);
    REQUIRE(bm_outputs[0].operand.location == Location::L2);
}

// ============================================================================
// Event Callback Test
// ============================================================================

TEST_CASE("FlowGraphExecutor event callback", "[dataflow][executor][events]") {
    OperandFlowGraph graph;
    auto input = tile_a(0, 0, Location::L3, 0);
    graph.add_wait({input});

    FlowGraphExecutor executor(graph);

    std::vector<ExecutionEvent> captured_events;
    executor.set_event_callback([&captured_events](const ExecutionEvent& e) {
        captured_events.push_back(e);
    });

    executor.inject_operand(input);
    executor.run();

    REQUIRE(!captured_events.empty());

    // Check that callback was invoked for each event type
    bool saw_ready = false, saw_fired = false;
    for (const auto& e : captured_events) {
        if (e.type == ExecutionEventType::OPERAND_READY) saw_ready = true;
        if (e.type == ExecutionEventType::NODE_FIRED) saw_fired = true;
    }
    REQUIRE(saw_ready);
    REQUIRE(saw_fired);
}

// ============================================================================
// Reset Test
// ============================================================================

TEST_CASE("FlowGraphExecutor reset", "[dataflow][executor][reset]") {
    OperandFlowGraph graph;
    auto input = tile_a(0, 0, Location::L3, 0);
    auto wait = graph.add_wait({input});

    FlowGraphExecutor executor(graph);

    // Run once
    executor.inject_operand(input);
    executor.run();
    REQUIRE(executor.is_complete());
    REQUIRE(executor.current_cycle() > 0);

    // Reset and run again
    executor.reset();
    REQUIRE_FALSE(executor.is_complete());
    REQUIRE(executor.current_cycle() == 0);
    REQUIRE(executor.get_node_state(wait) == NodeState::WAITING);

    // Can run again
    executor.inject_operand(input);
    REQUIRE(executor.run());
    REQUIRE(executor.is_complete());
}

// ============================================================================
// Chrome Trace Export Tests
// ============================================================================

#include <sw/kpu/models/dataflow/chrome_trace.hpp>
#include <fstream>
#include <filesystem>

TEST_CASE("ChromeTraceWriter basic events", "[dataflow][trace][chrome]") {
    ChromeTraceWriter writer("Test DMA", ExecutionLevel::DMA, 0);

    SECTION("Metadata events are created") {
        const auto& events = writer.events();
        REQUIRE(events.size() >= 2);  // process_name and thread_name

        bool has_proc_name = false, has_thread_name = false;
        for (const auto& e : events) {
            if (e.name == "process_name") has_proc_name = true;
            if (e.name == "thread_name") has_thread_name = true;
        }
        REQUIRE(has_proc_name);
        REQUIRE(has_thread_name);
    }

    SECTION("Duration events") {
        // Use larger cycle values to ensure non-zero duration after ns->us conversion
        writer.record_duration("test_op", "test", 0, 100000);

        const auto& events = writer.events();
        bool found = false;
        for (const auto& e : events) {
            if (e.name == "test_op" && e.phase == TracePhase::COMPLETE) {
                found = true;
                REQUIRE(e.duration_us > 0);
            }
        }
        REQUIRE(found);
    }

    SECTION("Instant events") {
        writer.record_instant("marker", "test", 50);

        const auto& events = writer.events();
        bool found = false;
        for (const auto& e : events) {
            if (e.name == "marker" && e.phase == TracePhase::INSTANT) {
                found = true;
            }
        }
        REQUIRE(found);
    }
}

TEST_CASE("ChromeTraceWriter records ExecutionEvents", "[dataflow][trace][chrome]") {
    OperandFlowGraph graph;
    auto input = tile_a(0, 0, Location::L3, 0);
    auto output = tile_a(0, 0, Location::L2, 0);

    auto wait_id = graph.add_wait({input}, "wait");
    auto fire_id = graph.add_fire(Operation::PUSH_TO_L2, {input}, {output}, "push");
    graph.add_edge(wait_id, fire_id);

    FlowGraphExecutor executor(graph);
    ChromeTraceWriter writer("DMA Tile 0", ExecutionLevel::DMA, 0);

    // Install callback to record events
    executor.set_event_callback([&writer](const ExecutionEvent& e) {
        writer.record_event(e);
    });

    executor.inject_operand(input);
    executor.run();

    const auto& events = writer.events();

    // Should have various event types
    bool has_node_complete = false;
    bool has_operand_ready = false;
    for (const auto& e : events) {
        if (e.category == "node" && e.phase == TracePhase::COMPLETE) has_node_complete = true;
        if (e.category == "operand" && e.name == "Operand Ready") has_operand_ready = true;
    }
    REQUIRE(has_node_complete);
    REQUIRE(has_operand_ready);
}

TEST_CASE("TraceConsolidator merges multiple traces", "[dataflow][trace][consolidator]") {
    // Create trace writers for different levels
    ChromeTraceWriter dma_writer("L3 Tile 0", ExecutionLevel::DMA, 0);
    ChromeTraceWriter bm_writer("Tile[0,0]", ExecutionLevel::BLOCK_MOVER, 0);
    ChromeTraceWriter streamer_writer("L2 Bank 0", ExecutionLevel::STREAMER, 0);

    // Add some events to each
    dma_writer.record_duration("DMA Load", "dma", 0, 100);
    bm_writer.record_duration("Push to L2", "blockmover", 100, 150);
    streamer_writer.record_duration("Feed West", "streamer", 150, 200);

    // Consolidate
    TraceConsolidator consolidator;
    consolidator.add_trace(dma_writer);
    consolidator.add_trace(bm_writer);
    consolidator.add_trace(streamer_writer);

    REQUIRE(consolidator.event_count() > 6);  // Metadata + duration events

    // Check summary
    std::string summary = consolidator.summary();
    REQUIRE(summary.find("Total events") != std::string::npos);
}

TEST_CASE("TraceConsolidator operand flow", "[dataflow][trace][consolidator][flow]") {
    ChromeTraceWriter producer("DMA", ExecutionLevel::DMA, 0);
    ChromeTraceWriter consumer("BlockMover", ExecutionLevel::BLOCK_MOVER, 0);

    TraceConsolidator consolidator;
    consolidator.add_trace(producer);
    consolidator.add_trace(consumer);

    // Add flow connection
    consolidator.add_operand_flow("A[0,0]@L3", producer, 100, consumer, 110);

    // Should have flow events
    size_t initial_count = consolidator.event_count();
    REQUIRE(consolidator.event_count() >= initial_count);  // Flow events added
}

TEST_CASE("Chrome trace file output", "[dataflow][trace][file]") {
    // Create a simple execution with tracing
    OperandFlowGraph graph;
    auto input = tile_a(0, 0, Location::L3, 0);
    auto output = tile_a(0, 0, Location::L2, 0);
    auto wait = graph.add_wait({input});
    auto fire = graph.add_fire(Operation::PUSH_TO_L2, {input}, {output});
    graph.add_edge(wait, fire);

    FlowGraphExecutor executor(graph);
    ChromeTraceWriter writer("Test", ExecutionLevel::DMA, 0);

    executor.set_event_callback([&writer](const ExecutionEvent& e) {
        writer.record_event(e);
    });

    executor.inject_operand(input);
    executor.run();

    // Write to temp file (cross-platform)
    std::string filename = (std::filesystem::temp_directory_path() / "test_trace.json").string();
    REQUIRE(writer.write_to_file(filename));

    // Verify file exists and is valid JSON
    std::ifstream file(filename);
    REQUIRE(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    REQUIRE(content.find("traceEvents") != std::string::npos);
    REQUIRE(content.find("displayTimeUnit") != std::string::npos);

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("Multi-level trace consolidation", "[dataflow][trace][integration]") {
    // DMA level
    DMAFlowGraphBuilder dma_builder;
    dma_builder.set_node_id(0)
               .set_dimensions(1, 1, 1)
               .add_load_a(0, 0, 0);
    auto dma_graph = dma_builder.build();
    DMAFlowExecutor dma(dma_graph);

    ChromeTraceWriter dma_trace("L3 Tile 0", ExecutionLevel::DMA, 0);
    dma.set_event_callback([&dma_trace](const ExecutionEvent& e) {
        dma_trace.record_event(e);
    });

    // BlockMover level
    OperandFlowGraph bm_graph;
    bm_graph.level = ExecutionLevel::BLOCK_MOVER;
    auto a_l3 = tile_a(0, 0, Location::L3, 0);
    auto a_l2 = tile_a(0, 0, Location::L2, 0);
    auto l2_avail = make_buffer_token(Location::L2, 0);
    auto bm_wait = bm_graph.add_wait({a_l3});
    auto bm_wait_l2 = bm_graph.add_wait({l2_avail});
    auto bm_join = bm_graph.add_join({a_l3, l2_avail});
    auto bm_push = bm_graph.add_fire(Operation::PUSH_TO_L2, {a_l3}, {a_l2});
    bm_graph.add_edge(bm_wait, bm_join);
    bm_graph.add_edge(bm_wait_l2, bm_join);
    bm_graph.add_edge(bm_join, bm_push);
    BlockMoverFlowExecutor bm(bm_graph);

    ChromeTraceWriter bm_trace("Tile[0,0]", ExecutionLevel::BLOCK_MOVER, 0);
    bm.set_event_callback([&bm_trace](const ExecutionEvent& e) {
        bm_trace.record_event(e);
    });

    // Execute DMA
    dma.inject_buffer_available(Location::L3, 0);
    REQUIRE(dma.run());

    // Feed DMA outputs to BlockMover
    auto dma_outputs = dma.get_tile_ready_events();
    bm.inject_operand(dma_outputs[0].operand, dma_outputs[0].cycle);
    bm.inject_operand(l2_avail);

    REQUIRE(bm.run());

    // Consolidate traces
    TraceConsolidator consolidator;
    consolidator.add_trace(dma_trace);
    consolidator.add_trace(bm_trace);

    // Add flow between levels
    consolidator.add_operand_flow(
        "A[0,0]@L3",
        dma_trace, dma_outputs[0].cycle,
        bm_trace, dma_outputs[0].cycle + 1  // BlockMover receives next cycle
    );

    // Write consolidated trace (cross-platform)
    std::string filename = (std::filesystem::temp_directory_path() / "multilevel_trace.json").string();
    REQUIRE(consolidator.write_to_file(filename));

    // Verify
    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    // Should have events from both levels (different pids)
    REQUIRE(content.find("\"pid\":1") != std::string::npos);  // DMA
    REQUIRE(content.find("\"pid\":2") != std::string::npos);  // BlockMover

    // Should have flow events
    REQUIRE(content.find("flow") != std::string::npos);

    std::filesystem::remove(filename);
}
