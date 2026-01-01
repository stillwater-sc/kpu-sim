# ============================================================================
# tools/benchmark/CMakeLists.txt
# KPU Benchmark Tools
# ============================================================================

# KPU Benchmark Suite
kpu_add_tool(benchmark
    FOLDER "Benchmark"
    DESCRIPTION "KPU benchmark suite"
    SOURCES
        kpu-benchmark/benchmark.cpp
        kpu-benchmark/matrix_benchmarks.cpp
        kpu-benchmark/memory_benchmarks.cpp
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
