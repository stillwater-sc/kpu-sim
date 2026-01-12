# Dataflow Tests
# Tests for Operand Flow Graph and related data structures

# OperandFlowGraph data structure tests
kpu_add_component_test(operand_flow_graph_test "dataflow"
    test_operand_flow_graph.cpp
)

# Flow Graph Executor tests
kpu_add_component_test(flow_graph_executor_test "dataflow"
    test_flow_graph_executors.cpp
)

# C-Stationary Matmul integration test
kpu_add_component_test(c_stationary_matmul_test "dataflow"
    test_c_stationary_matmul.cpp
)
