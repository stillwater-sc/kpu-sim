#!/usr/bin/env python3
"""
Graph Validation Suite
Verifies graph construction, validation, and analysis

Run from repository root:
    python3 examples/python/test_graph_validation.py
"""
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '../../build/src/bindings/python/Release')
sys.path.insert(0, build_dir)

import stillwater_kpu as kpu

def test_graph_construction():
    """Test basic graph construction"""
    graph = kpu.KernelGraph("test_graph")

    assert graph.empty()
    assert graph.num_nodes() == 0
    assert graph.num_edges() == 0

    # Add nodes
    n1 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "node1")
    n2 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "node2")

    assert not graph.empty()
    assert graph.num_nodes() == 2
    assert graph.has_node(n1)
    assert graph.has_node(n2)

    # Add edge
    graph.add_edge(n1, n2, "C", "A")
    assert graph.num_edges() == 1

    # Check incoming/outgoing edges
    outgoing = graph.outgoing_edges(n1)
    assert len(outgoing) == 1

    incoming = graph.incoming_edges(n2)
    assert len(incoming) == 1

    print("  PASS: Graph construction")
    return True

def test_graph_naming():
    """Test graph and node naming"""
    graph = kpu.KernelGraph("my_network")
    assert graph.name == "my_network"

    # Change name
    graph.name = "renamed_network"
    assert graph.name == "renamed_network"

    # Add named nodes
    n1 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "layer1")
    kernel = graph.get_kernel(n1)
    # Node name is stored in the graph, kernel name is separate

    print("  PASS: Graph naming")
    return True

def test_cycle_detection():
    """Test that cycles are detected"""
    graph = kpu.KernelGraph("cycle_test")

    n1 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "a")
    n2 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "b")
    n3 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "c")

    graph.add_edge(n1, n2, "C", "A")
    graph.add_edge(n2, n3, "C", "A")

    # This would create a cycle: c -> a
    would_cycle = graph.would_create_cycle(n3, n1)
    assert would_cycle, "Cycle should be detected"

    # This would not create a cycle
    would_not_cycle = graph.would_create_cycle(n1, n3)
    assert not would_not_cycle, "No cycle here"

    print("  PASS: Cycle detection")
    return True

def test_topological_sort():
    """Test execution order (topological sort)"""
    graph = kpu.KernelGraph("topo_test")

    # Create chain: a -> b -> c
    a = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "a")
    b = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "b")
    c = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "c")

    graph.add_edge(a, b, "C", "A")
    graph.add_edge(b, c, "C", "A")

    order = graph.get_execution_order()

    # a must come before b, b must come before c
    assert order.index(a) < order.index(b), "a should come before b"
    assert order.index(b) < order.index(c), "b should come before c"

    print("  PASS: Topological sort")
    return True

def test_parallel_levels():
    """Test parallel execution level detection"""
    graph = kpu.KernelGraph("parallel_test")

    # Diamond pattern:
    #       top
    #       / \
    #    left  right
    #       \ /
    #      bottom
    top = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "top")
    left = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "left")
    right = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "right")
    bottom = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "bottom")

    graph.add_edge(top, left, "C", "A")
    graph.add_edge(top, right, "C", "A")
    graph.add_edge(left, bottom, "C", "A")
    graph.add_edge(right, bottom, "C", "B")

    levels = graph.get_execution_levels()

    # Should have 3 levels: [top], [left, right], [bottom]
    assert len(levels) == 3, f"Expected 3 levels, got {len(levels)}"
    assert len(levels[0]) == 1, "Level 0 should have 1 node (top)"
    assert len(levels[1]) == 2, "Level 1 should have 2 nodes (left, right)"
    assert len(levels[2]) == 1, "Level 2 should have 1 node (bottom)"

    # Verify level 1 contains left and right
    assert left in levels[1] and right in levels[1]

    print("  PASS: Parallel level detection")
    return True

def test_critical_path():
    """Test critical path detection"""
    graph = kpu.KernelGraph("critical_test")

    # Chain with varying compute
    n1 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "small1")
    n2 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "small2")
    n3 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "small3")

    graph.add_edge(n1, n2, "C", "A")
    graph.add_edge(n2, n3, "C", "A")

    critical = graph.get_critical_path()

    # All nodes should be on critical path for a chain
    assert len(critical) == 3
    assert n1 in critical and n2 in critical and n3 in critical

    print("  PASS: Critical path detection")
    return True

def test_graph_validation():
    """Test graph validation"""
    # Valid graph with nodes and edges
    valid_graph = kpu.KernelGraph("valid")
    n1 = valid_graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "a")
    n2 = valid_graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "b")
    valid_graph.add_edge(n1, n2, "C", "A")

    is_valid, error = valid_graph.validate()
    assert is_valid, f"Valid graph should pass: {error}"

    # Single node graph should be valid
    single_graph = kpu.KernelGraph("single")
    single_graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "only")
    is_valid, error = single_graph.validate()
    assert is_valid, f"Single node graph should be valid: {error}"

    # Empty graph returns an error (expected behavior)
    empty_graph = kpu.KernelGraph("empty")
    is_valid, error = empty_graph.validate()
    assert not is_valid, "Empty graph should be invalid"
    assert "empty" in error.lower(), f"Error should mention empty: {error}"

    print("  PASS: Graph validation")
    return True

def test_input_output_nodes():
    """Test identification of input and output nodes"""
    graph = kpu.KernelGraph("io_test")

    # Create graph: input -> middle -> output
    input_node = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "input")
    middle = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "middle")
    output_node = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "output")

    graph.add_edge(input_node, middle, "C", "A")
    graph.add_edge(middle, output_node, "C", "A")

    inputs = graph.input_nodes()
    outputs = graph.output_nodes()

    assert len(inputs) == 1 and input_node in inputs
    assert len(outputs) == 1 and output_node in outputs

    print("  PASS: Input/output node detection")
    return True

def test_graph_statistics():
    """Test graph statistics computation"""
    graph = kpu.KernelGraph("stats_test")

    # Two-layer network
    fc1 = graph.add_kernel(
        kpu.Kernel.create_mlp(64, 256, 128, kpu.ActivationType.RELU, True),
        "fc1"
    )
    fc2 = graph.add_kernel(
        kpu.Kernel.create_mlp(64, 64, 256, kpu.ActivationType.NONE, True),
        "fc2"
    )
    graph.add_edge(fc1, fc2, "C", "A")

    stats = graph.compute_stats()

    assert stats.num_nodes == 2
    assert stats.num_edges == 1
    assert stats.max_depth == 1
    assert stats.total_flops > 0
    assert stats.total_instructions > 0
    assert stats.total_input_bytes > 0
    assert stats.total_output_bytes > 0
    assert stats.avg_arithmetic_intensity > 0

    print("  PASS: Graph statistics")
    return True

def test_fusion_detection():
    """Test fusible pair detection"""
    graph = kpu.KernelGraph("fusion_test")

    # Chain of layers (producer-consumer pairs are fusible)
    n1 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "layer1")
    n2 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "layer2")
    n3 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "layer3")

    graph.add_edge(n1, n2, "C", "A")
    graph.add_edge(n2, n3, "C", "A")

    fusible = graph.find_fusible_pairs()

    # Should find fusible pairs
    assert len(fusible) >= 1, "Should find at least one fusible pair"

    print("  PASS: Fusion detection")
    return True

def test_dot_export():
    """Test DOT graph export"""
    graph = kpu.KernelGraph("dot_test")

    n1 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "layer1")
    n2 = graph.add_kernel(kpu.Kernel.create_matmul(64, 64, 64), "layer2")
    graph.add_edge(n1, n2, "C", "A")

    dot = graph.to_dot(show_tensor_sizes=True)

    # Should contain DOT syntax
    assert "digraph" in dot
    assert "layer1" in dot
    assert "layer2" in dot
    assert "->" in dot

    print("  PASS: DOT export")
    return True

def test_graph_compilation():
    """Test graph compilation"""
    graph = kpu.KernelGraph("compile_test")

    fc1 = graph.add_kernel(
        kpu.Kernel.create_mlp(64, 128, 64, kpu.ActivationType.RELU, True),
        "fc1"
    )
    fc2 = graph.add_kernel(
        kpu.Kernel.create_mlp(64, 64, 128, kpu.ActivationType.NONE, True),
        "fc2"
    )
    graph.add_edge(fc1, fc2, "C", "A")

    result = graph.compile()

    assert result.success, f"Compilation failed: {result.error_message}"
    assert len(result.execution_order) == 2
    assert result.workspace_required > 0

    print("  PASS: Graph compilation")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("  Graph Validation Suite")
    print("=" * 60)

    tests = [
        ("Graph Construction", test_graph_construction),
        ("Graph Naming", test_graph_naming),
        ("Cycle Detection", test_cycle_detection),
        ("Topological Sort", test_topological_sort),
        ("Parallel Levels", test_parallel_levels),
        ("Critical Path", test_critical_path),
        ("Graph Validation", test_graph_validation),
        ("Input/Output Nodes", test_input_output_nodes),
        ("Graph Statistics", test_graph_statistics),
        ("Fusion Detection", test_fusion_detection),
        ("DOT Export", test_dot_export),
        ("Graph Compilation", test_graph_compilation),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
