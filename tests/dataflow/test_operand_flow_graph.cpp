// ============================================================================
// tests/dataflow/test_operand_flow_graph.cpp
// Tests for Operand Flow Graph data structure
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/models/dataflow/operand_flow_graph.hpp>

using namespace sw::kpu::dataflow;

// ============================================================================
// Basic Type Tests
// ============================================================================

TEST_CASE("OperandType to_string", "[dataflow][operand]") {
    REQUIRE(std::string(to_string(OperandType::TILE_A)) == "TILE_A");
    REQUIRE(std::string(to_string(OperandType::TILE_B)) == "TILE_B");
    REQUIRE(std::string(to_string(OperandType::TILE_C)) == "TILE_C");
    REQUIRE(std::string(to_string(OperandType::BUFFER_TOKEN)) == "BUFFER_TOKEN");
}

TEST_CASE("Location to_string", "[dataflow][operand]") {
    REQUIRE(std::string(to_string(Location::MEMORY)) == "MEMORY");
    REQUIRE(std::string(to_string(Location::L3)) == "L3");
    REQUIRE(std::string(to_string(Location::L2)) == "L2");
    REQUIRE(std::string(to_string(Location::L1)) == "L1");
    REQUIRE(std::string(to_string(Location::ACCUMULATOR)) == "ACCUMULATOR");
}

TEST_CASE("Operation to_string", "[dataflow][operand]") {
    REQUIRE(std::string(to_string(Operation::LOAD)) == "LOAD");
    REQUIRE(std::string(to_string(Operation::PUSH_TO_L2)) == "PUSH_TO_L2");
    REQUIRE(std::string(to_string(Operation::SEND_EAST)) == "SEND_EAST");
    REQUIRE(std::string(to_string(Operation::MATMUL)) == "MATMUL");
}

// ============================================================================
// TileCoord Tests
// ============================================================================

TEST_CASE("TileCoord equality", "[dataflow][coord]") {
    TileCoord c1{1, 2, 3};
    TileCoord c2{1, 2, 3};
    TileCoord c3{1, 2, 4};

    REQUIRE(c1 == c2);
    REQUIRE(c1 != c3);
}

TEST_CASE("TileCoord to_string", "[dataflow][coord]") {
    TileCoord c{1, 2, 3};
    REQUIRE(c.to_string() == "[1,2,3]");
}

TEST_CASE("TileCoordHash works", "[dataflow][coord]") {
    TileCoordHash hasher;
    TileCoord c1{1, 2, 3};
    TileCoord c2{1, 2, 3};
    TileCoord c3{3, 2, 1};

    REQUIRE(hasher(c1) == hasher(c2));
    REQUIRE(hasher(c1) != hasher(c3));
}

// ============================================================================
// Operand Tests
// ============================================================================

TEST_CASE("Operand creation and equality", "[dataflow][operand]") {
    Operand op1{OperandType::TILE_A, {1, 0, 2}, Location::L3, 5};
    Operand op2{OperandType::TILE_A, {1, 0, 2}, Location::L3, 5};
    Operand op3{OperandType::TILE_A, {1, 0, 2}, Location::L2, 5};

    REQUIRE(op1 == op2);
    REQUIRE(op1 != op3);  // Different location
}

TEST_CASE("Operand to_string", "[dataflow][operand]") {
    Operand op{OperandType::TILE_A, {1, 0, 2}, Location::L3, 5};
    std::string s = op.to_string();

    REQUIRE(s.find("TILE_A") != std::string::npos);
    REQUIRE(s.find("[1,0,2]") != std::string::npos);
    REQUIRE(s.find("L3") != std::string::npos);
    REQUIRE(s.find("[5]") != std::string::npos);
}

TEST_CASE("Helper functions for operand creation", "[dataflow][operand]") {
    SECTION("tile_a creates A tile operand") {
        auto op = tile_a(1, 2, Location::L3, 5);
        REQUIRE(op.type == OperandType::TILE_A);
        REQUIRE(op.coord.i == 1);
        REQUIRE(op.coord.k == 2);
        REQUIRE(op.location == Location::L3);
        REQUIRE(op.node_id == 5);
    }

    SECTION("tile_b creates B tile operand") {
        auto op = tile_b(2, 3, Location::L2, 1);
        REQUIRE(op.type == OperandType::TILE_B);
        REQUIRE(op.coord.k == 2);
        REQUIRE(op.coord.j == 3);
        REQUIRE(op.location == Location::L2);
    }

    SECTION("tile_c creates C tile operand") {
        auto op = tile_c(1, 2, Location::ACCUMULATOR, 0);
        REQUIRE(op.type == OperandType::TILE_C);
        REQUIRE(op.coord.i == 1);
        REQUIRE(op.coord.j == 2);
        REQUIRE(op.location == Location::ACCUMULATOR);
    }

    SECTION("make_buffer_token creates buffer token") {
        auto op = make_buffer_token(Location::L2, 3);
        REQUIRE(op.type == OperandType::BUFFER_TOKEN);
        REQUIRE(op.location == Location::L2);
        REQUIRE(op.node_id == 3);
    }
}

// ============================================================================
// FlowNode Tests
// ============================================================================

TEST_CASE("FlowNode creation", "[dataflow][node]") {
    FlowNode node;
    node.id = 0;
    node.type = FlowNodeType::FIRE;
    node.operation = Operation::PUSH_TO_L2;
    node.name = "push_a";
    node.inputs = {tile_a(1, 0, Location::L3, 5)};
    node.outputs = {tile_a(1, 0, Location::L2, 0)};

    REQUIRE(node.type == FlowNodeType::FIRE);
    REQUIRE(node.operation == Operation::PUSH_TO_L2);
    REQUIRE(node.inputs.size() == 1);
    REQUIRE(node.outputs.size() == 1);
}

TEST_CASE("FlowNode to_string", "[dataflow][node]") {
    FlowNode node;
    node.id = 0;
    node.type = FlowNodeType::FIRE;
    node.operation = Operation::PUSH_TO_L2;
    node.name = "push_a";
    node.inputs = {tile_a(1, 0, Location::L3, 5)};
    node.outputs = {tile_a(1, 0, Location::L2, 0)};

    std::string s = node.to_string();
    REQUIRE(s.find("[0]") != std::string::npos);
    REQUIRE(s.find("FIRE") != std::string::npos);
    REQUIRE(s.find("PUSH_TO_L2") != std::string::npos);
    REQUIRE(s.find("push_a") != std::string::npos);
}

// ============================================================================
// OperandFlowGraph Construction Tests
// ============================================================================

TEST_CASE("OperandFlowGraph empty graph", "[dataflow][graph]") {
    OperandFlowGraph graph;
    graph.name = "empty";
    graph.level = ExecutionLevel::BLOCK_MOVER;

    REQUIRE(graph.num_nodes() == 0);
    REQUIRE(graph.num_edges() == 0);
    REQUIRE(graph.get_entry_nodes().empty());
    REQUIRE(graph.get_exit_nodes().empty());
}

TEST_CASE("OperandFlowGraph add_node", "[dataflow][graph]") {
    OperandFlowGraph graph;

    FlowNode node;
    node.type = FlowNodeType::WAIT;
    node.inputs = {tile_a(0, 0, Location::L3, 0)};

    uint32_t id = graph.add_node(node);

    REQUIRE(id == 0);
    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.nodes[0].id == 0);
}

TEST_CASE("OperandFlowGraph builder methods", "[dataflow][graph]") {
    OperandFlowGraph graph;

    SECTION("add_wait") {
        auto id = graph.add_wait({tile_a(0, 0, Location::L3, 0)}, "wait_a");
        REQUIRE(graph.nodes[id].type == FlowNodeType::WAIT);
        REQUIRE(graph.nodes[id].inputs.size() == 1);
    }

    SECTION("add_fire") {
        auto id = graph.add_fire(
            Operation::PUSH_TO_L2,
            {tile_a(0, 0, Location::L3, 0)},
            {tile_a(0, 0, Location::L2, 0)},
            "push_a"
        );
        REQUIRE(graph.nodes[id].type == FlowNodeType::FIRE);
        REQUIRE(graph.nodes[id].operation == Operation::PUSH_TO_L2);
    }

    SECTION("add_produce") {
        auto id = graph.add_produce({tile_a(0, 0, Location::L2, 0)}, "produce_a");
        REQUIRE(graph.nodes[id].type == FlowNodeType::PRODUCE);
        REQUIRE(graph.nodes[id].outputs.size() == 1);
    }

    SECTION("add_join") {
        auto id = graph.add_join({
            tile_a(0, 0, Location::L3, 0),
            tile_b(0, 0, Location::L3, 0)
        }, "join_ab");
        REQUIRE(graph.nodes[id].type == FlowNodeType::JOIN);
        REQUIRE(graph.nodes[id].inputs.size() == 2);
    }

    SECTION("add_fork") {
        auto id = graph.add_fork(
            tile_a(0, 0, Location::L3, 0),
            {tile_a(0, 0, Location::L3, 1), tile_a(0, 0, Location::L3, 2)},
            "fork_a"
        );
        REQUIRE(graph.nodes[id].type == FlowNodeType::FORK);
        REQUIRE(graph.nodes[id].outputs.size() == 2);
    }
}

TEST_CASE("OperandFlowGraph add_edge", "[dataflow][graph]") {
    OperandFlowGraph graph;

    auto n0 = graph.add_wait({tile_a(0, 0, Location::L3, 0)});
    auto n1 = graph.add_fire(Operation::PUSH_TO_L2, {}, {tile_a(0, 0, Location::L2, 0)});

    graph.add_edge(n0, n1);

    REQUIRE(graph.num_edges() == 1);
    REQUIRE(graph.edges[0].from_node == n0);
    REQUIRE(graph.edges[0].to_node == n1);
}

TEST_CASE("OperandFlowGraph add_edge with operand", "[dataflow][graph]") {
    OperandFlowGraph graph;

    auto n0 = graph.add_wait({tile_a(0, 0, Location::L3, 0)});
    auto n1 = graph.add_fire(Operation::PUSH_TO_L2, {}, {});

    auto operand = tile_a(0, 0, Location::L3, 0);
    graph.add_edge(n0, n1, operand);

    REQUIRE(graph.edges[0].operand.has_value());
    REQUIRE(graph.edges[0].operand->type == OperandType::TILE_A);
}

// ============================================================================
// OperandFlowGraph Query Tests
// ============================================================================

TEST_CASE("OperandFlowGraph find_producers", "[dataflow][graph][query]") {
    OperandFlowGraph graph;

    auto operand = tile_a(0, 0, Location::L2, 0);

    // Node that produces the operand
    graph.add_fire(Operation::PUSH_TO_L2, {}, {operand}, "producer");

    // Node that doesn't produce it
    graph.add_fire(Operation::PUSH_TO_L2, {}, {tile_b(0, 0, Location::L2, 1)}, "other");

    auto producers = graph.find_producers(operand);
    REQUIRE(producers.size() == 1);
    REQUIRE(producers[0] == 0);
}

TEST_CASE("OperandFlowGraph find_consumers", "[dataflow][graph][query]") {
    OperandFlowGraph graph;

    auto operand = tile_a(0, 0, Location::L3, 0);

    // Node that consumes the operand
    graph.add_wait({operand}, "consumer");

    // Node that consumes a different operand
    graph.add_wait({tile_b(0, 0, Location::L3, 0)}, "other");

    auto consumers = graph.find_consumers(operand);
    REQUIRE(consumers.size() == 1);
    REQUIRE(consumers[0] == 0);
}

TEST_CASE("OperandFlowGraph predecessor/successor queries", "[dataflow][graph][query]") {
    OperandFlowGraph graph;

    // Build a simple chain: n0 -> n1 -> n2
    auto n0 = graph.add_wait({tile_a(0, 0, Location::L3, 0)});
    auto n1 = graph.add_fire(Operation::PUSH_TO_L2, {}, {});
    auto n2 = graph.add_produce({tile_a(0, 0, Location::L2, 0)});

    graph.add_edge(n0, n1);
    graph.add_edge(n1, n2);

    SECTION("get_predecessors") {
        auto preds = graph.get_predecessors(n1);
        REQUIRE(preds.size() == 1);
        REQUIRE(preds[0] == n0);

        preds = graph.get_predecessors(n0);
        REQUIRE(preds.empty());
    }

    SECTION("get_successors") {
        auto succs = graph.get_successors(n1);
        REQUIRE(succs.size() == 1);
        REQUIRE(succs[0] == n2);

        succs = graph.get_successors(n2);
        REQUIRE(succs.empty());
    }

    SECTION("get_entry_nodes") {
        auto entries = graph.get_entry_nodes();
        REQUIRE(entries.size() == 1);
        REQUIRE(entries[0] == n0);
    }

    SECTION("get_exit_nodes") {
        auto exits = graph.get_exit_nodes();
        REQUIRE(exits.size() == 1);
        REQUIRE(exits[0] == n2);
    }
}

// ============================================================================
// OperandFlowGraph Validation Tests
// ============================================================================

TEST_CASE("OperandFlowGraph validation", "[dataflow][graph][validation]") {
    SECTION("Valid graph passes validation") {
        OperandFlowGraph graph;
        graph.add_wait({tile_a(0, 0, Location::L3, 0)});
        graph.add_fire(Operation::PUSH_TO_L2, {}, {tile_a(0, 0, Location::L2, 0)});
        graph.add_edge(0, 1);

        auto result = graph.validate();
        REQUIRE(result.valid);
        REQUIRE(result.errors.empty());
    }

    SECTION("WAIT node without inputs generates warning") {
        OperandFlowGraph graph;
        graph.add_wait({}, "empty_wait");

        auto result = graph.validate();
        REQUIRE(result.valid);  // Still valid, but has warning
        REQUIRE(!result.warnings.empty());
    }

    SECTION("PRODUCE node without outputs generates warning") {
        OperandFlowGraph graph;
        graph.add_produce({}, "empty_produce");

        auto result = graph.validate();
        REQUIRE(result.valid);
        REQUIRE(!result.warnings.empty());
    }
}

// ============================================================================
// OperandFlowGraph Statistics Tests
// ============================================================================

TEST_CASE("OperandFlowGraph statistics", "[dataflow][graph][stats]") {
    OperandFlowGraph graph;

    graph.add_wait({tile_a(0, 0, Location::L3, 0)});
    graph.add_wait({tile_b(0, 0, Location::L3, 0)});
    graph.add_fire(Operation::PUSH_TO_L2, {}, {});
    graph.add_fire(Operation::SEND_EAST, {}, {});
    graph.add_produce({});

    REQUIRE(graph.count_nodes_by_type(FlowNodeType::WAIT) == 2);
    REQUIRE(graph.count_nodes_by_type(FlowNodeType::FIRE) == 2);
    REQUIRE(graph.count_nodes_by_type(FlowNodeType::PRODUCE) == 1);

    REQUIRE(graph.count_nodes_by_operation(Operation::PUSH_TO_L2) == 1);
    REQUIRE(graph.count_nodes_by_operation(Operation::SEND_EAST) == 1);
}

// ============================================================================
// C-Stationary BlockMover Flow Graph Example
// ============================================================================

TEST_CASE("C-Stationary BlockMover flow graph for L3[1,1]", "[dataflow][graph][c_stationary]") {
    // Build the flow graph for BlockMover at L3[1,1] in a 4x4 mesh
    // For a single k iteration of C-stationary matmul

    OperandFlowGraph graph;
    graph.name = "C-Stationary BlockMover L3[1,1]";
    graph.level = ExecutionLevel::BLOCK_MOVER;
    graph.node_id = 5;  // L3[1,1] = 4*1 + 1 = 5
    graph.mesh_row = 1;
    graph.mesh_col = 1;
    graph.m_tiles = 4;
    graph.n_tiles = 4;
    graph.k_tiles = 4;

    const uint16_t i = 1;  // Row position
    const uint16_t j = 1;  // Column position
    const uint16_t k = 0;  // First k iteration

    // ========== A tile path ==========

    // Wait for A tile to arrive at L3
    auto wait_a = graph.add_wait(
        {tile_a(i, k, Location::L3, graph.node_id)},
        "wait_A[1,0]@L3"
    );

    // Wait for L2 buffer available
    auto wait_l2_a = graph.add_wait(
        {make_buffer_token(Location::L2, 0)},
        "wait_L2[0]_avail"
    );

    // Join: A ready AND L2 available
    auto join_a = graph.add_join(
        {tile_a(i, k, Location::L3, graph.node_id), make_buffer_token(Location::L2, 0)},
        "join_A_ready_L2_avail"
    );

    // Fire: Push A to L2
    auto push_a = graph.add_fire(
        Operation::PUSH_TO_L2,
        {tile_a(i, k, Location::L3, graph.node_id)},
        {tile_a(i, k, Location::L2, 0)},
        "push_A[1,0]_to_L2"
    );

    // Fire: Forward A east to L3[1,2] (since j < 3)
    auto send_a_east = graph.add_fire(
        Operation::SEND_EAST,
        {tile_a(i, k, Location::L3, graph.node_id)},
        {tile_a(i, k, Location::L3, 6)},  // L3[1,2] = 4*1 + 2 = 6
        "send_A[1,0]_east"
    );

    // ========== B tile path ==========

    // Wait for B tile to arrive at L3
    auto wait_b = graph.add_wait(
        {tile_b(k, j, Location::L3, graph.node_id)},
        "wait_B[0,1]@L3"
    );

    // Wait for L2 buffer available
    auto wait_l2_b = graph.add_wait(
        {make_buffer_token(Location::L2, 1)},
        "wait_L2[1]_avail"
    );

    // Join: B ready AND L2 available
    auto join_b = graph.add_join(
        {tile_b(k, j, Location::L3, graph.node_id), make_buffer_token(Location::L2, 1)},
        "join_B_ready_L2_avail"
    );

    // Fire: Push B to L2
    auto push_b = graph.add_fire(
        Operation::PUSH_TO_L2,
        {tile_b(k, j, Location::L3, graph.node_id)},
        {tile_b(k, j, Location::L2, 1)},
        "push_B[0,1]_to_L2"
    );

    // Fire: Forward B south to L3[2,1] (since i < 3)
    auto send_b_south = graph.add_fire(
        Operation::SEND_SOUTH,
        {tile_b(k, j, Location::L3, graph.node_id)},
        {tile_b(k, j, Location::L3, 9)},  // L3[2,1] = 4*2 + 1 = 9
        "send_B[0,1]_south"
    );

    // ========== Connect edges ==========

    // A path
    graph.add_edge(wait_a, join_a);
    graph.add_edge(wait_l2_a, join_a);
    graph.add_edge(join_a, push_a);
    graph.add_edge(push_a, send_a_east);

    // B path
    graph.add_edge(wait_b, join_b);
    graph.add_edge(wait_l2_b, join_b);
    graph.add_edge(join_b, push_b);
    graph.add_edge(push_b, send_b_south);

    // ========== Validate the graph ==========

    auto result = graph.validate();
    REQUIRE(result.valid);

    // Check graph structure
    // A path: wait_a, wait_l2_a, join_a, push_a, send_a_east (5 nodes)
    // B path: wait_b, wait_l2_b, join_b, push_b, send_b_south (5 nodes)
    REQUIRE(graph.num_nodes() == 10);
    REQUIRE(graph.num_edges() == 8);

    // Check entry/exit nodes
    auto entries = graph.get_entry_nodes();
    REQUIRE(entries.size() == 4);  // wait_a, wait_l2_a, wait_b, wait_l2_b

    auto exits = graph.get_exit_nodes();
    REQUIRE(exits.size() == 2);  // send_a_east, send_b_south

    // Check node types
    REQUIRE(graph.count_nodes_by_type(FlowNodeType::WAIT) == 4);
    REQUIRE(graph.count_nodes_by_type(FlowNodeType::JOIN) == 2);
    REQUIRE(graph.count_nodes_by_type(FlowNodeType::FIRE) == 4);

    // Check operations
    REQUIRE(graph.count_nodes_by_operation(Operation::PUSH_TO_L2) == 2);
    REQUIRE(graph.count_nodes_by_operation(Operation::SEND_EAST) == 1);
    REQUIRE(graph.count_nodes_by_operation(Operation::SEND_SOUTH) == 1);

    // Verify the graph can be serialized
    std::string s = graph.to_string();
    REQUIRE(s.find("C-Stationary BlockMover L3[1,1]") != std::string::npos);
    REQUIRE(s.find("BLOCK_MOVER") != std::string::npos);
}

// ============================================================================
// Coordination Event Tests
// ============================================================================

TEST_CASE("TileReadyEvent", "[dataflow][events]") {
    TileReadyEvent event;
    event.operand = tile_a(1, 0, Location::L3, 5);
    event.timestamp = 100;

    std::string s = event.to_string();
    REQUIRE(s.find("TILE_READY") != std::string::npos);
    REQUIRE(s.find("@100") != std::string::npos);
}

TEST_CASE("BufferAvailableEvent", "[dataflow][events]") {
    BufferAvailableEvent event;
    event.location = Location::L2;
    event.node_id = 0;
    event.bank_id = 1;
    event.capacity = 2;
    event.timestamp = 50;

    std::string s = event.to_string();
    REQUIRE(s.find("BUFFER_AVAILABLE") != std::string::npos);
    REQUIRE(s.find("L2") != std::string::npos);
}

TEST_CASE("OperandToken", "[dataflow][events]") {
    OperandToken token;
    token.operand = tile_a(0, 0, Location::L3, 0);
    token.ready = false;

    REQUIRE_FALSE(token.ready);
    REQUIRE_FALSE(token.consumed);

    token.ready = true;
    token.ready_cycle = 100;

    REQUIRE(token.ready);
    REQUIRE(token.ready_cycle == 100);
}
