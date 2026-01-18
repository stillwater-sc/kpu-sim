# ============================================================================
# tools/benchmark/CMakeLists.txt
# KPU Benchmark Tools - v0.3.1
# ============================================================================

# KPU Benchmark Suite CLI Tool
# Provides comprehensive benchmarking capabilities:
#   - Matmul sweeps (standard, extended, quick)
#   - Tile sensitivity analysis
#   - Roofline analysis
#   - JSON/CSV output for integration
kpu_add_tool(benchmark
    FOLDER "Benchmark"
    DESCRIPTION "KPU benchmark suite CLI tool"
    SOURCES
        kpu-benchmark/benchmark.cpp
    LIBS
        kpu_benchmark
)

# NoC Benchmark Tool
kpu_add_tool(noc-bench
    FOLDER "Benchmark"
    DESCRIPTION "NoC characterization and benchmarking tool"
    SOURCES
        kpu-noc-bench/main.cpp
    LIBS
        kpu_benchmark
        kpu_noc
)
