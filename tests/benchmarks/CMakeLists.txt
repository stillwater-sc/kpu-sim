# tests/benchmarks/CMakeLists.txt
# Benchmark tests for KPU simulator

# Matmul benchmarks
add_executable(benchmark_matmul test_matmul_benchmarks.cpp)
target_link_libraries(benchmark_matmul PRIVATE
    Catch2::Catch2WithMain
    StillwaterKPU::Benchmark
)
kpu_set_target_options(benchmark_matmul)
set_target_properties(benchmark_matmul PROPERTIES FOLDER "tests/benchmarks")

# MLP and activation benchmarks
add_executable(benchmark_mlp test_mlp_benchmarks.cpp)
target_link_libraries(benchmark_mlp PRIVATE
    Catch2::Catch2WithMain
    StillwaterKPU::Benchmark
)
kpu_set_target_options(benchmark_mlp)
set_target_properties(benchmark_mlp PROPERTIES FOLDER "tests/benchmarks")

# Graph benchmarks
add_executable(benchmark_graph test_graph_benchmarks.cpp)
target_link_libraries(benchmark_graph PRIVATE
    Catch2::Catch2WithMain
    StillwaterKPU::Benchmark
)
kpu_set_target_options(benchmark_graph)
set_target_properties(benchmark_graph PROPERTIES FOLDER "tests/benchmarks")

# Register with CTest
add_test(NAME benchmark_matmul COMMAND benchmark_matmul)
set_tests_properties(benchmark_matmul PROPERTIES LABELS "benchmark")

add_test(NAME benchmark_mlp COMMAND benchmark_mlp)
set_tests_properties(benchmark_mlp PROPERTIES LABELS "benchmark")

add_test(NAME benchmark_graph COMMAND benchmark_graph)
set_tests_properties(benchmark_graph PROPERTIES LABELS "benchmark")

# Efficiency diagnostic
add_executable(efficiency_diagnostic test_efficiency_diagnostic.cpp)
target_link_libraries(efficiency_diagnostic PRIVATE
    Catch2::Catch2WithMain
    kpu_simulator
    kpu_benchmark
    kpu_compiler
    kpu_simulator  # Repeated to resolve circular dependency with LTO
)
kpu_set_target_options(efficiency_diagnostic)
set_target_properties(efficiency_diagnostic PROPERTIES FOLDER "tests/benchmarks")

# Combined benchmark runner
add_executable(run_all_benchmarks
    test_matmul_benchmarks.cpp
    test_mlp_benchmarks.cpp
    test_graph_benchmarks.cpp
)
target_link_libraries(run_all_benchmarks PRIVATE
    Catch2::Catch2WithMain
    StillwaterKPU::Benchmark
)
kpu_set_target_options(run_all_benchmarks)
set_target_properties(run_all_benchmarks PROPERTIES FOLDER "tests/benchmarks")
