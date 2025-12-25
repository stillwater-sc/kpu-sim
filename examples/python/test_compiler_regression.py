#!/usr/bin/env python3
"""
Compiler Regression Suite
Verifies compiler output consistency and correctness

Run from repository root:
    python3 examples/python/test_compiler_regression.py
"""
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '../../build/src/bindings/python/Release')
sys.path.insert(0, build_dir)

import stillwater_kpu as kpu

def test_auto_tiling_consistency():
    """Test that auto-tiling produces consistent results"""
    compiler = kpu.KernelCompiler()

    # Compile same kernel twice
    k1 = compiler.compile_matmul(512, 512, 512)
    stats1 = compiler.last_stats()

    k2 = compiler.compile_matmul(512, 512, 512)
    stats2 = compiler.last_stats()

    # Results should be identical
    assert stats1.selected_Ti == stats2.selected_Ti, "Ti should be consistent"
    assert stats1.selected_Tj == stats2.selected_Tj, "Tj should be consistent"
    assert stats1.selected_Tk == stats2.selected_Tk, "Tk should be consistent"
    assert stats1.total_tiles == stats2.total_tiles, "Total tiles should be consistent"
    assert stats1.instruction_count == stats2.instruction_count, "Instruction count should be consistent"

    print("  PASS: Auto-tiling consistency")
    return True

def test_tile_bounds():
    """Test that tile sizes respect problem bounds"""
    compiler = kpu.KernelCompiler()

    test_cases = [
        (64, 64, 64),
        (100, 100, 100),  # Non-power-of-2
        (1024, 512, 256),
        (32, 128, 64),
        (17, 23, 31),     # Prime numbers
    ]

    for M, N, K in test_cases:
        kernel = compiler.compile_matmul(M, N, K)
        stats = compiler.last_stats()

        # Tiles should not exceed problem dimensions
        assert stats.selected_Ti <= M, f"Ti {stats.selected_Ti} > M {M}"
        assert stats.selected_Tj <= N, f"Tj {stats.selected_Tj} > N {N}"
        assert stats.selected_Tk <= K, f"Tk {stats.selected_Tk} > K {K}"

        # Tiles should be positive
        assert stats.selected_Ti > 0, "Ti should be positive"
        assert stats.selected_Tj > 0, "Tj should be positive"
        assert stats.selected_Tk > 0, "Tk should be positive"

        print(f"  PASS: {M}x{N}x{K} -> tiles {stats.selected_Ti}x{stats.selected_Tj}x{stats.selected_Tk}")

    return True

def test_explicit_tiles():
    """Test compilation with explicit tile sizes"""
    compiler = kpu.KernelCompiler()

    Ti, Tj, Tk = 32, 64, 128
    kernel = compiler.compile_matmul_tiled(256, 256, 256, Ti, Tj, Tk)
    stats = compiler.last_stats()

    assert stats.selected_Ti == Ti, f"Ti mismatch: {stats.selected_Ti} != {Ti}"
    assert stats.selected_Tj == Tj, f"Tj mismatch: {stats.selected_Tj} != {Tj}"
    assert stats.selected_Tk == Tk, f"Tk mismatch: {stats.selected_Tk} != {Tk}"

    # Verify tile counts
    expected_m_tiles = (256 + Ti - 1) // Ti
    expected_n_tiles = (256 + Tj - 1) // Tj
    expected_k_tiles = (256 + Tk - 1) // Tk

    assert stats.num_m_tiles == expected_m_tiles, f"M tiles: {stats.num_m_tiles} != {expected_m_tiles}"
    assert stats.num_n_tiles == expected_n_tiles, f"N tiles: {stats.num_n_tiles} != {expected_n_tiles}"
    assert stats.num_k_tiles == expected_k_tiles, f"K tiles: {stats.num_k_tiles} != {expected_k_tiles}"

    print("  PASS: Explicit tiles")
    return True

def test_instruction_count_scaling():
    """Test that instruction count scales reasonably with problem size"""
    compiler = kpu.KernelCompiler()

    # Fix tile sizes to isolate scaling
    opts = kpu.CompileOptions.with_tiles(64, 64, 64)

    sizes = [128, 256, 512]
    instruction_counts = []

    for size in sizes:
        kernel = compiler.compile_matmul(size, size, size, opts)
        instruction_counts.append(compiler.last_stats().instruction_count)

    # Instruction count should increase with problem size
    for i in range(1, len(instruction_counts)):
        assert instruction_counts[i] > instruction_counts[i-1], \
            f"Instructions should increase: {instruction_counts}"

    print(f"  PASS: Scaling {sizes} -> {instruction_counts}")
    return True

def test_flops_calculation():
    """Test that FLOPs are calculated correctly"""
    compiler = kpu.KernelCompiler()

    test_cases = [
        (64, 64, 64),
        (128, 256, 512),
        (1024, 1024, 1024),
    ]

    for M, N, K in test_cases:
        kernel = compiler.compile_matmul(M, N, K)
        expected_flops = 2 * M * N * K  # matmul: 2*M*N*K FLOPs

        assert kernel.total_flops() == expected_flops, \
            f"FLOPs mismatch for {M}x{N}x{K}: {kernel.total_flops()} != {expected_flops}"

    print("  PASS: FLOPs calculation")
    return True

def test_arithmetic_intensity():
    """Test arithmetic intensity calculation"""
    compiler = kpu.KernelCompiler()

    # Larger tiles should give higher arithmetic intensity
    small_opts = kpu.CompileOptions.with_tiles(16, 16, 16)
    large_opts = kpu.CompileOptions.with_tiles(64, 64, 64)

    compiler.compile_matmul(256, 256, 256, small_opts)
    small_ai = compiler.last_stats().estimated_arithmetic_intensity

    compiler.compile_matmul(256, 256, 256, large_opts)
    large_ai = compiler.last_stats().estimated_arithmetic_intensity

    # Larger tiles should have equal or higher AI (better data reuse)
    assert large_ai >= small_ai, \
        f"Larger tiles should have higher AI: {large_ai} < {small_ai}"

    print(f"  PASS: AI small={small_ai:.2f}, large={large_ai:.2f}")
    return True

def test_operation_counts():
    """Test that operation counts are reasonable"""
    compiler = kpu.KernelCompiler()

    kernel = compiler.compile_matmul(256, 256, 256)
    stats = compiler.last_stats()

    # All operation counts should be positive
    assert stats.dma_ops > 0, "Should have DMA ops"
    assert stats.block_mover_ops > 0, "Should have block mover ops"
    assert stats.streamer_ops > 0, "Should have streamer ops"
    assert stats.instruction_count > 0, "Should have instructions"

    # Instruction count should be sum of all ops (approximately)
    # Note: may include barriers and other overhead
    total_ops = stats.dma_ops + stats.block_mover_ops + stats.streamer_ops
    assert stats.instruction_count >= total_ops, \
        "Instruction count should include all ops"

    print(f"  PASS: DMA={stats.dma_ops}, BM={stats.block_mover_ops}, STR={stats.streamer_ops}")
    return True

def test_compile_options():
    """Test various compile options"""
    compiler = kpu.KernelCompiler()

    # Default options
    opts1 = kpu.CompileOptions.defaults()
    assert opts1.is_auto_tiling(), "Defaults should use auto-tiling"

    # Explicit tiles
    opts2 = kpu.CompileOptions.with_tiles(32, 32, 32)
    assert not opts2.is_auto_tiling(), "Explicit tiles should disable auto-tiling"
    assert opts2.Ti == 32
    assert opts2.Tj == 32
    assert opts2.Tk == 32

    # Inference options
    opts3 = kpu.CompileOptions.for_inference()
    # Should compile successfully
    kernel = compiler.compile_matmul(256, 256, 256, opts3)
    assert kernel.is_valid()

    print("  PASS: Compile options")
    return True

def test_mlp_compilation():
    """Test MLP kernel compilation"""
    compiler = kpu.KernelCompiler()

    activations = [
        kpu.ActivationType.RELU,
        kpu.ActivationType.GELU,
        kpu.ActivationType.SIGMOID,
    ]

    for act in activations:
        kernel = compiler.compile_mlp(64, 128, 256, act, has_bias=True)

        assert kernel.is_valid(), f"MLP with {act} should be valid"
        assert kernel.op_type() == kpu.KernelOpType.MLP
        assert kernel.activation() == act
        assert kernel.has_bias() == True

        stats = compiler.last_stats()
        assert stats.instruction_count > 0

        print(f"  PASS: MLP with {act}")

    return True

def test_large_problem():
    """Test compilation of large problems"""
    compiler = kpu.KernelCompiler()

    # Large transformer layer dimensions
    kernel = compiler.compile_matmul(4096, 4096, 4096)

    assert kernel.is_valid()
    stats = compiler.last_stats()

    # Should have many tiles
    assert stats.total_tiles > 100, f"Large problem should have many tiles: {stats.total_tiles}"

    # Check compile time is reasonable (< 1 second)
    assert stats.compile_time_us < 1_000_000, f"Compile time too long: {stats.compile_time_us} us"

    print(f"  PASS: 4096x4096x4096 in {stats.compile_time_us/1000:.1f}ms, {stats.total_tiles} tiles")
    return True

def test_small_problem():
    """Test compilation of small problems (edge cases)"""
    compiler = kpu.KernelCompiler()

    test_cases = [
        (16, 16, 16),  # Smaller than typical systolic array
        (8, 8, 8),
        (1, 256, 256),  # Batch size 1
        (256, 1, 256),  # Single output column
    ]

    for M, N, K in test_cases:
        kernel = compiler.compile_matmul(M, N, K)
        assert kernel.is_valid(), f"Small problem {M}x{N}x{K} should compile"
        assert compiler.last_succeeded()

        print(f"  PASS: {M}x{N}x{K}")

    return True

def test_compile_error_handling():
    """Test compiler error handling"""
    compiler = kpu.KernelCompiler()

    # Valid compilation should succeed
    kernel = compiler.compile_matmul(64, 64, 64)
    assert compiler.last_succeeded()
    assert compiler.last_error() == ""

    print("  PASS: Error handling")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("  Compiler Regression Suite")
    print("=" * 60)

    tests = [
        ("Auto-tiling Consistency", test_auto_tiling_consistency),
        ("Tile Bounds", test_tile_bounds),
        ("Explicit Tiles", test_explicit_tiles),
        ("Instruction Count Scaling", test_instruction_count_scaling),
        ("FLOPs Calculation", test_flops_calculation),
        ("Arithmetic Intensity", test_arithmetic_intensity),
        ("Operation Counts", test_operation_counts),
        ("Compile Options", test_compile_options),
        ("MLP Compilation", test_mlp_compilation),
        ("Large Problem", test_large_problem),
        ("Small Problem", test_small_problem),
        ("Error Handling", test_compile_error_handling),
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
