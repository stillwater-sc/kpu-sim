#!/usr/bin/env python3
"""
Test script for the KPU Simulator Python bindings (C++ native module)

These tests require the C++ native bindings to be built and installed.
If the native module is not available, tests will be skipped.

To build the native module:
    cmake --preset release && cmake --build --preset release
    pip install -e .
"""

import numpy as np
import time
import sys
import pytest

# Try to import the native C++ bindings module
try:
    import stillwater_kpu as kpu
    NATIVE_AVAILABLE = True
except ImportError as e:
    NATIVE_AVAILABLE = False
    kpu = None

# Skip all tests in this module if native bindings are not available
pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="stillwater_kpu native bindings not available (build with cmake first)"
)

def test_basic_functionality():
    """Test basic simulator functionality with new architecture"""
    print("\n=== Testing Basic Functionality ===")
    
    # Create simulator configuration
    config = kpu.SimulatorConfig()
    config.memory_bank_count = 2
    config.memory_bank_capacity_mb = 128
    config.scratchpad_count = 4
    config.scratchpad_capacity_kb = 64
    config.compute_tile_count = 2
    config.dma_engine_count = 4
    
    # Create simulator
    simulator = kpu.KPUSimulator(config)
    print("✓ Simulator created successfully")
    
    # Test configuration queries
    print(f" Memory banks: {simulator.get_memory_bank_count()}")
    # Test capacity queries
    for i in range(simulator.get_memory_bank_count()):
        capacity = simulator.get_memory_bank_capacity(i) // (1024*1024)
        print(f"  Memory bank[{i}] capacity: {capacity} MB")
    print(f" Scratchpads: {simulator.get_scratchpad_count()}")
    for i in range(simulator.get_scratchpad_count()):
        capacity = simulator.get_scratchpad_capacity(i) // 1024
        print(f"  Scratchpad[{i}] capacity: {capacity} KB")    
    print(f" Compute tiles: {simulator.get_compute_tile_count()}")
    print(f" DMA engines: {simulator.get_dma_engine_count()}")
      
    # Test reset
    simulator.reset()
    print("✓ Simulator reset successful")
    
    return True

def test_memory_operations():
    """Test memory operations with new delegation API"""
    print("\n=== Testing Memory Operations ===")
    
    config = kpu.SimulatorConfig()
    config.memory_bank_count = 2
    config.memory_bank_capacity_mb = 128
    config.scratchpad_count = 2
    config.scratchpad_capacity_kb = 64
    config.compute_tile_count = 1
    config.dma_engine_count = 1
    simulator = kpu.KPUSimulator(config)
    
    # Test memory bank operations
    test_data = [1.0, 2.0, 3.0, 4.0]
    simulator.write_memory_bank(0, 0, test_data)
    read_data = simulator.read_memory_bank(0, 0, len(test_data))
    
    if np.allclose(test_data, read_data):
        print(" Memory bank read/write operations work")
    else:
        print(" Memory bank operations failed")
        return False
    
    # Test scratchpad operations
    simulator.write_scratchpad(0, 0, test_data)
    read_data = simulator.read_scratchpad(0, 0, len(test_data))
    
    if np.allclose(test_data, read_data):
        print(" Scratchpad read/write operations work")
    else:
        print(" Scratchpad operations failed")
        return False
    
    return True

def test_numpy_integration():
    """Test NumPy array integration with new API"""
    print("\n=== Testing NumPy Integration ===")
    
    # Create test matrices using NumPy
    np.random.seed(42)  # For reproducible results
    A = np.random.randn(4, 6).astype(np.float32)
    B = np.random.randn(6, 3).astype(np.float32)
    
    print(f" Created matrices A{A.shape} and B{B.shape}")
    
    # Compute expected result using NumPy
    expected_C = A @ B
    print(f" NumPy reference result computed: C{expected_C.shape}")
    
    # Test with KPU simulator using new API
    config = kpu.SimulatorConfig()
    config.memory_bank_count = 1
    config.memory_bank_capacity_mb = 64
    config.scratchpad_count = 1
    config.scratchpad_capacity_kb = 64
    config.compute_tile_count = 1
    config.dma_engine_count = 1
    
    try:
        simulator = kpu.KPUSimulator(config)
        start_time = time.time()
        result_C = simulator.run_numpy_matmul(A, B, 0, 0, 0)  # bank_id, pad_id, tile_id
        end_time = time.time()
        
        print(f" KPU simulation completed in {(end_time - start_time)*1000:.2f} ms")
        print(f" Result shape: {result_C.shape}")
        
        # Verify results match
        max_diff = np.max(np.abs(result_C - expected_C))
        print(f" Maximum difference from NumPy: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print(" Results match within tolerance!")
            return True
        else:
            print(" Results don't match!")
            return False
            
    except Exception as e:
        print(f" NumPy integration test failed: {e}")
        return False

def test_multi_bank_configuration():
    """Test multi-bank simulator configuration"""
    print("\n=== Testing Multi-Bank Configuration ===")
    
    # Generate multi-bank configuration
    config = kpu.generate_multi_bank_config(4, 2)  # 4 banks, 2 tiles
    simulator = kpu.KPUSimulator(config)
    
    print(f" Created {simulator.get_memory_bank_count()}-bank, {simulator.get_compute_tile_count()}-tile simulator")
    
    # Test the direct numpy path with an 8x8 problem on this multi-bank
    # config. run_distributed_matmul_test was removed along with the
    # broken KPUSimulator::run_matmul_test it relied on - see the
    # CHANGELOG entry for the removal PR.
    try:
        import numpy as np
        A = np.random.rand(8, 8).astype(np.float32)
        B = np.random.rand(8, 8).astype(np.float32)
        C = simulator.run_numpy_matmul(A, B, 0, 0, 0)
        expected = A @ B
        if np.allclose(C, expected, atol=1e-4):
            print(" Multi-bank numpy_matmul PASSED!")
            simulator.print_component_status()
            return True
        else:
            print("✗ Multi-bank numpy_matmul produced wrong result")
            return False
    except Exception as e:
        print(f"✗ Multi-bank test failed: {e}")
        return False

def test_step_by_step_simulation():
    """Test manual step-by-step simulation control"""
    print("\n=== Testing Step-by-Step Simulation ===")
    
    config = kpu.SimulatorConfig()
    config.memory_bank_count = 1
    config.memory_bank_capacity_mb = 64
    config.scratchpad_count = 1 
    config.scratchpad_capacity_kb = 64
    config.compute_tile_count = 1
    config.dma_engine_count = 1
    simulator = kpu.KPUSimulator(config)
    
    # Create small test matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    expected_C = A @ B
    
    print("Step-by-step simulation test:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"Expected C = A @ B = \n{expected_C}")
    
    # Reset and get initial state
    simulator.reset()
    initial_cycle = simulator.get_current_cycle()
    print(f" Initial cycle: {initial_cycle}")
    
    try:
        # Test the simulation
        result_C = simulator.run_numpy_matmul(A, B, 0, 0, 0)
        final_cycle = simulator.get_current_cycle()
        
        print(f" Simulation result:\n{result_C}")
        print(f" Final cycle: {final_cycle}")
        print(f" Cycles elapsed: {final_cycle - initial_cycle}")
        
        # Verify correctness
        if np.allclose(result_C, expected_C, rtol=1e-5):
            print(" Step-by-step simulation test PASSED!")
            return True
        else:
            print(" Step-by-step simulation test FAILED!")
            print(f"Expected:\n{expected_C}")
            print(f"Got:\n{result_C}")
            return False
            
    except Exception as e:
        print(f" Step-by-step simulation failed: {e}")
        return False

def test_performance_scaling():
    """Test performance with different matrix sizes and configurations"""
    print("\n=== Testing Performance Scaling ===")
    
    sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]
    
    configs = [
        ("Single Bank", kpu.SimulatorConfig()),
        ("Multi Bank", kpu.generate_multi_bank_config(2, 1))
    ]
    
    print("Config       | Size    | Sim Time (ms) | Cycles | Ops/Cycle")
    print("-" * 60)
    
    for config_name, config in configs:
        for m, n, k in sizes:
            try:
                simulator = kpu.KPUSimulator(config)
                A = np.random.randn(m, k).astype(np.float32)
                B = np.random.randn(k, n).astype(np.float32)
                
                start_time = time.time()
                result_C = simulator.run_numpy_matmul(A, B, 0, 0, 0)
                end_time = time.time()
                
                sim_time_ms = (end_time - start_time) * 1000
                cycles = simulator.get_current_cycle()
                ops = 2 * m * n * k  # 2 ops per MAC
                ops_per_cycle = ops / cycles if cycles > 0 else 0
                
                print(f"{config_name:12} | {m:2}x{n:2}x{k:2} | {sim_time_ms:8.2f}    | {cycles:6} | {ops_per_cycle:8.2f}")
                
            except Exception as e:
                print(f"{config_name:12} | {m:2}x{n:2}x{k:2} | ERROR: {e}")
    
    return True

def main():
    """Run all tests"""
    print("KPU Simulator Python Bindings Test Suite")
    print("=" * 70)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Memory Operations", test_memory_operations),
        ("NumPy Integration", test_numpy_integration),
        ("Multi-Bank Configuration", test_multi_bank_configuration),
        ("Step-by-Step Simulation", test_step_by_step_simulation),
        ("Performance Scaling", test_performance_scaling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KPU Simulator Python bindings are working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())