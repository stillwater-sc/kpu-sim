// ============================================================================
// examples/behavioral/ofg_trace_demo.cpp
// Demonstrates multi-level OFG execution with Chrome Trace export
// ============================================================================
//
// This example shows how to:
// 1. Create OFG executors for DMA, BlockMover, and Streamer levels
// 2. Connect Chrome trace writers to each executor
// 3. Execute the flow graphs and capture execution events
// 4. Consolidate all traces into a single Chrome Trace file
// 5. Add flow events to visualize operand movement between levels
//
// Output: traces/ofg_multilevel_trace.json
// View in: chrome://tracing or https://ui.perfetto.dev
//
// ============================================================================

#include <sw/kpu/models/dataflow/operand_flow_graph.hpp>
#include <sw/kpu/models/dataflow/flow_graph_executor.hpp>
#include <sw/kpu/models/dataflow/dma_flow_executor.hpp>
#include <sw/kpu/models/dataflow/block_mover_flow_executor.hpp>
#include <sw/kpu/models/dataflow/streamer_flow_executor.hpp>
#include <sw/kpu/models/dataflow/chrome_trace.hpp>

#include <iostream>
#include <filesystem>

using namespace sw::kpu::dataflow;

int main() {
    std::cout << "=== OFG Multi-Level Trace Demo ===" << std::endl;
    std::cout << std::endl;

    // Configure tracing with 1000ns (1us) per cycle for visible trace events
    // This makes small cycle counts visible in the trace viewer
    TraceConfig trace_config;
    trace_config.cycle_time_ns = 1000;  // 1 microsecond per cycle

    // Create trace consolidator
    TraceConsolidator consolidator(trace_config);

    // ========================================================================
    // Level 1: DMA - Load A[0,0] and B[0,0] tiles from host to L3
    // ========================================================================
    std::cout << "Level 1: DMA (Host -> L3)" << std::endl;

    DMAFlowGraphBuilder dma_builder;
    dma_builder.set_node_id(0)           // L3 tile 0
               .set_dimensions(2, 2, 2)   // 2x2x2 tile dimensions
               .add_load_a(0, 0, 0)       // Load A[0,0]
               .add_load_b(0, 0, 0);      // Load B[0,0]

    auto dma_graph = dma_builder.build();

    DMAExecutorConfig dma_config;
    dma_config.load_latency = 100;  // 100 cycles per DMA transfer
    dma_config.store_latency = 100;

    DMAFlowExecutor dma_executor(dma_graph, dma_config);

    // Create trace writer for DMA
    ChromeTraceWriter dma_trace("DMA Engine", ExecutionLevel::DMA, 0, trace_config);
    dma_executor.set_event_callback([&dma_trace](const ExecutionEvent& e) {
        dma_trace.record_event(e);
    });

    // Inject buffer availability (L3 has space)
    dma_executor.inject_buffer_available(Location::L3, 0, 0);
    dma_executor.inject_buffer_available(Location::L3, 0, 0);

    bool dma_success = dma_executor.run();
    std::cout << "  DMA completed: " << (dma_success ? "YES" : "NO") << std::endl;
    std::cout << "  Loads: " << dma_executor.loads_completed() << std::endl;
    std::cout << "  Cycles: " << dma_executor.current_cycle() << std::endl;

    auto dma_outputs = dma_executor.get_tile_ready_events();
    std::cout << "  Output events: " << dma_outputs.size() << std::endl;

    consolidator.add_trace(dma_trace);

    // ========================================================================
    // Level 2: BlockMover - Push tiles from L3 to L2
    // ========================================================================
    std::cout << std::endl << "Level 2: BlockMover (L3 -> L2)" << std::endl;

    OperandFlowGraph bm_graph;
    bm_graph.level = ExecutionLevel::BLOCK_MOVER;
    bm_graph.name = "BlockMover L3->L2";

    // Create nodes for pushing A and B to L2
    auto a_l3 = tile_a(0, 0, Location::L3, 0);
    auto b_l3 = tile_b(0, 0, Location::L3, 0);
    auto a_l2 = tile_a(0, 0, Location::L2, 0);
    auto b_l2 = tile_b(0, 0, Location::L2, 1);
    auto l2_bank0_avail = make_buffer_token(Location::L2, 0);
    auto l2_bank1_avail = make_buffer_token(Location::L2, 1);

    // Build graph: wait for inputs, join, push
    auto wait_a = bm_graph.add_wait({a_l3}, "wait_A");
    auto wait_b = bm_graph.add_wait({b_l3}, "wait_B");
    auto wait_l2_0 = bm_graph.add_wait({l2_bank0_avail}, "wait_L2_0");
    auto wait_l2_1 = bm_graph.add_wait({l2_bank1_avail}, "wait_L2_1");

    auto join_a = bm_graph.add_join({a_l3, l2_bank0_avail}, "join_A_L2");
    auto join_b = bm_graph.add_join({b_l3, l2_bank1_avail}, "join_B_L2");

    auto push_a = bm_graph.add_fire(Operation::PUSH_TO_L2, {a_l3}, {a_l2}, "push_A");
    auto push_b = bm_graph.add_fire(Operation::PUSH_TO_L2, {b_l3}, {b_l2}, "push_B");

    bm_graph.add_edge(wait_a, join_a);
    bm_graph.add_edge(wait_l2_0, join_a);
    bm_graph.add_edge(join_a, push_a);

    bm_graph.add_edge(wait_b, join_b);
    bm_graph.add_edge(wait_l2_1, join_b);
    bm_graph.add_edge(join_b, push_b);

    BlockMoverExecutorConfig bm_config;
    bm_config.push_latency = 20;  // 20 cycles to push tile

    BlockMoverFlowExecutor bm_executor(bm_graph, bm_config);

    // Create trace writer for BlockMover
    ChromeTraceWriter bm_trace("BlockMover Tile[0,0]", ExecutionLevel::BLOCK_MOVER, 0, trace_config);
    bm_executor.set_event_callback([&bm_trace](const ExecutionEvent& e) {
        bm_trace.record_event(e);
    });

    // Inject DMA outputs as inputs to BlockMover
    uint64_t bm_start_cycle = dma_executor.current_cycle();
    for (const auto& out : dma_outputs) {
        bm_executor.inject_operand(out.operand, bm_start_cycle + out.cycle);

        // Add flow event showing data movement from DMA to BlockMover
        std::string flow_id = out.operand.to_string();
        consolidator.add_operand_flow(flow_id, dma_trace, out.cycle,
                                       bm_trace, bm_start_cycle + out.cycle);
    }

    // L2 banks are available
    bm_executor.inject_operand(l2_bank0_avail, bm_start_cycle);
    bm_executor.inject_operand(l2_bank1_avail, bm_start_cycle);

    bool bm_success = bm_executor.run();
    std::cout << "  BlockMover completed: " << (bm_success ? "YES" : "NO") << std::endl;
    std::cout << "  Pushes: " << bm_executor.pushes_completed() << std::endl;
    std::cout << "  Cycles: " << bm_executor.current_cycle() << std::endl;

    auto bm_outputs = bm_executor.get_l2_ready_events();
    std::cout << "  Output events: " << bm_outputs.size() << std::endl;

    consolidator.add_trace(bm_trace);

    // ========================================================================
    // Level 3: Streamer - Feed tiles to compute and execute matmul
    // ========================================================================
    std::cout << std::endl << "Level 3: Streamer (L2 -> Compute)" << std::endl;

    OperandFlowGraph str_graph;
    str_graph.level = ExecutionLevel::STREAMER;
    str_graph.name = "Streamer Matmul";

    // Create nodes for feeding and computing
    auto a_l1 = tile_a(0, 0, Location::L1, 0);
    auto b_l1 = tile_b(0, 0, Location::L1, 0);
    auto c_acc = tile_c(0, 0, Location::ACCUMULATOR, 0);
    auto c_l2 = tile_c(0, 0, Location::L2, 2);

    auto str_wait_a = str_graph.add_wait({a_l2}, "wait_A_L2");
    auto str_wait_b = str_graph.add_wait({b_l2}, "wait_B_L2");
    auto str_join = str_graph.add_join({a_l2, b_l2}, "join_AB");

    auto feed_a = str_graph.add_fire(Operation::FEED_WEST, {a_l2}, {a_l1}, "feed_A");
    auto feed_b = str_graph.add_fire(Operation::FEED_NORTH, {b_l2}, {b_l1}, "feed_B");
    auto matmul = str_graph.add_fire(Operation::MATMUL, {a_l1, b_l1}, {c_acc}, "matmul");
    auto drain = str_graph.add_fire(Operation::DRAIN, {c_acc}, {c_l2}, "drain_C");

    str_graph.add_edge(str_wait_a, str_join);
    str_graph.add_edge(str_wait_b, str_join);
    str_graph.add_edge(str_join, feed_a);
    str_graph.add_edge(str_join, feed_b);
    str_graph.add_edge(feed_a, matmul);
    str_graph.add_edge(feed_b, matmul);
    str_graph.add_edge(matmul, drain);

    StreamerExecutorConfig str_config;
    str_config.feed_latency = 8;      // 8 cycles to feed tile
    str_config.matmul_latency = 64;   // 64 cycles for matmul (systolic latency)
    str_config.drain_latency = 16;    // 16 cycles to drain result

    StreamerFlowExecutor str_executor(str_graph, str_config);

    // Create trace writer for Streamer
    ChromeTraceWriter str_trace("Streamer PE[0,0]", ExecutionLevel::STREAMER, 0, trace_config);
    str_executor.set_event_callback([&str_trace](const ExecutionEvent& e) {
        str_trace.record_event(e);
    });

    // Inject BlockMover outputs as inputs to Streamer
    uint64_t str_start_cycle = bm_start_cycle + bm_executor.current_cycle();
    for (const auto& out : bm_outputs) {
        str_executor.inject_operand(out.operand, str_start_cycle + out.cycle);

        // Add flow event showing data movement from BlockMover to Streamer
        std::string flow_id = out.operand.to_string();
        consolidator.add_operand_flow(flow_id, bm_trace, out.cycle,
                                       str_trace, str_start_cycle + out.cycle);
    }

    bool str_success = str_executor.run();
    std::cout << "  Streamer completed: " << (str_success ? "YES" : "NO") << std::endl;
    std::cout << "  Feeds: " << str_executor.feeds_completed() << std::endl;
    std::cout << "  Matmuls: " << str_executor.matmuls_completed() << std::endl;
    std::cout << "  Drains: " << str_executor.drains_completed() << std::endl;
    std::cout << "  Cycles: " << str_executor.current_cycle() << std::endl;

    consolidator.add_trace(str_trace);

    // ========================================================================
    // Write consolidated trace
    // ========================================================================
    std::cout << std::endl << "=== Writing Trace ===" << std::endl;

    // Create output directory if needed
    std::filesystem::create_directories("traces");

    std::string trace_file = "traces/ofg_multilevel_trace.json";
    bool write_success = consolidator.write_to_file(trace_file);

    if (write_success) {
        std::cout << "Trace written to: " << trace_file << std::endl;
        std::cout << std::endl << consolidator.summary() << std::endl;
        std::cout << "View in:" << std::endl;
        std::cout << "  - Chrome: chrome://tracing (load file)" << std::endl;
        std::cout << "  - Perfetto: https://ui.perfetto.dev (open file)" << std::endl;
    } else {
        std::cerr << "Failed to write trace file!" << std::endl;
        return 1;
    }

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << std::endl << "=== Execution Summary ===" << std::endl;
    uint64_t total_cycles = str_start_cycle + str_executor.current_cycle();
    std::cout << "Total cycles: " << total_cycles << std::endl;
    std::cout << "  DMA:        " << dma_executor.current_cycle() << " cycles" << std::endl;
    std::cout << "  BlockMover: " << bm_executor.current_cycle() << " cycles" << std::endl;
    std::cout << "  Streamer:   " << str_executor.current_cycle() << " cycles" << std::endl;

    return 0;
}
