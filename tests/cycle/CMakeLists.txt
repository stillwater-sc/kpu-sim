# tests/cycle/CMakeLists.txt
# The cycle core (docs/plans/cycle-based-kpu-prototype.md, epic #532)

# Step 0a (#533): time, links, the scheduler, and the test structure
add_executable(test_cycle_core test_cycle_core.cpp)
target_link_libraries(test_cycle_core PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_cycle_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
kpu_set_target_options(test_cycle_core)
set_target_properties(test_cycle_core PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME test_cycle_core COMMAND test_cycle_core)
set_tests_properties(test_cycle_core PROPERTIES LABELS "cycle;core;v0.10")

# Step 0b (#534): operational-analysis counters on every component and link
add_executable(test_cycle_counters test_cycle_counters.cpp)
target_link_libraries(test_cycle_counters PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_cycle_counters PRIVATE ${CMAKE_SOURCE_DIR}/include)
kpu_set_target_options(test_cycle_counters)
set_target_properties(test_cycle_counters PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME test_cycle_counters COMMAND test_cycle_counters)
set_tests_properties(test_cycle_counters PROPERTIES LABELS "cycle;counters;v0.10")

# Step 0b (#534): the simulator's own throughput. The full table: build/tests/cycle/bench_cycle_core.
# CTest runs the quick sizes as a smoke test, so the benchmark keeps building and running.
add_executable(bench_cycle_core bench_cycle_core.cpp)
target_include_directories(bench_cycle_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
kpu_set_target_options(bench_cycle_core)
set_target_properties(bench_cycle_core PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME bench_cycle_core_quick COMMAND bench_cycle_core --quick)
set_tests_properties(bench_cycle_core_quick PROPERTIES LABELS "cycle;benchmark;v0.10")

# Step 0c (#535): the deadlock detector, invariant violations, the state dump, the trace
add_executable(test_cycle_debug test_cycle_debug.cpp)
target_link_libraries(test_cycle_debug PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_cycle_debug PRIVATE ${CMAKE_SOURCE_DIR}/include)
kpu_set_target_options(test_cycle_debug)
set_target_properties(test_cycle_debug PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME test_cycle_debug COMMAND test_cycle_debug)
set_tests_properties(test_cycle_debug PROPERTIES LABELS "cycle;debug;v0.10")

# Step 0d (#536): clock-domain crossings, and each domain's clock track in the trace
add_executable(test_cycle_crossing test_cycle_crossing.cpp)
target_link_libraries(test_cycle_crossing PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_cycle_crossing PRIVATE ${CMAKE_SOURCE_DIR}/include)
kpu_set_target_options(test_cycle_crossing)
set_target_properties(test_cycle_crossing PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME test_cycle_crossing COMMAND test_cycle_crossing)
set_tests_properties(test_cycle_crossing PROPERTIES LABELS "cycle;crossing;v0.10")

# Step 1a (#537): the LPDDR5 controller as a cycle-core component, against the bare controller
add_executable(test_cycle_lpddr5 test_cycle_lpddr5.cpp)
target_link_libraries(test_cycle_lpddr5 PRIVATE Catch2::Catch2WithMain kpu_memory_components kpu_trace)
target_include_directories(test_cycle_lpddr5 PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common)
kpu_set_target_options(test_cycle_lpddr5)
set_target_properties(test_cycle_lpddr5 PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME test_cycle_lpddr5 COMMAND test_cycle_lpddr5)
set_tests_properties(test_cycle_lpddr5 PROPERTIES LABELS "cycle;lpddr5;memory;v0.10")

# Step 1b (#538): the L3 tile and the DMA engine on a direct link, with DRAM behind them.
add_executable(test_cycle_l3_dma test_cycle_l3_dma.cpp)
target_link_libraries(test_cycle_l3_dma PRIVATE Catch2::Catch2WithMain kpu_memory_components kpu_trace)
target_include_directories(test_cycle_l3_dma PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common)
kpu_set_target_options(test_cycle_l3_dma)
set_target_properties(test_cycle_l3_dma PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME test_cycle_l3_dma COMMAND test_cycle_l3_dma)
set_tests_properties(test_cycle_l3_dma PROPERTIES LABELS "cycle;l3;dma;memory;v0.10")

# Step 1c (#539): the NoC hubs and ports on the core, built from a deployment's layout.
add_executable(test_cycle_noc test_cycle_noc.cpp)
target_link_libraries(test_cycle_noc PRIVATE Catch2::Catch2WithMain kpu_memory_components kpu_trace kpu_program)
target_include_directories(test_cycle_noc PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common)
target_compile_definitions(test_cycle_noc PRIVATE KPU_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
kpu_set_target_options(test_cycle_noc)
set_target_properties(test_cycle_noc PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME test_cycle_noc COMMAND test_cycle_noc)
set_tests_properties(test_cycle_noc PROPERTIES LABELS "cycle;noc;dma;memory;v0.10")

# Step 1d (#540): a real matmul program's L3 traffic, the hypothesized C from L-B.
add_executable(test_cycle_csp_l3 test_cycle_csp_l3.cpp)
target_link_libraries(test_cycle_csp_l3 PRIVATE Catch2::Catch2WithMain kpu_simulator kpu_memory_components kpu_trace
                      kpu_program)
target_include_directories(test_cycle_csp_l3 PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common)
target_compile_definitions(test_cycle_csp_l3 PRIVATE KPU_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
kpu_set_target_options(test_cycle_csp_l3)
set_target_properties(test_cycle_csp_l3 PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME test_cycle_csp_l3 COMMAND test_cycle_csp_l3)
set_tests_properties(test_cycle_csp_l3 PROPERTIES LABELS "cycle;csp;l3;dma;memory;v0.10")

# Step 0c (#535): the test structure's trace and counters files, validated against each other.
# Open build/tests/cycle/cycle_trace_demo.json in https://ui.perfetto.dev
add_executable(cycle_trace_demo cycle_trace_demo.cpp)
target_include_directories(cycle_trace_demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
kpu_set_target_options(cycle_trace_demo)
set_target_properties(cycle_trace_demo PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME cycle_trace_demo
    COMMAND cycle_trace_demo ${CMAKE_CURRENT_BINARY_DIR}/cycle_trace_demo.json
                             ${CMAKE_CURRENT_BINARY_DIR}/cycle_trace_demo.counters.json)
set_tests_properties(cycle_trace_demo PROPERTIES LABELS "cycle;trace;v0.10" FIXTURES_SETUP cycle_trace)
find_program(KPU_PYTHON3 NAMES python3 python)
if(KPU_PYTHON3)
    add_test(NAME cycle_trace_check
        COMMAND ${KPU_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/check_trace.py
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_trace_demo.json
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_trace_demo.counters.json)
    set_tests_properties(cycle_trace_check PROPERTIES LABELS "cycle;trace;v0.10" FIXTURES_REQUIRED cycle_trace)
    # The checker rejects what it should: corrupted copies of the demo's files.
    add_test(NAME cycle_trace_check_selftest
        COMMAND ${KPU_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/test_check_trace.py
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_trace_demo.json
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_trace_demo.counters.json)
    set_tests_properties(cycle_trace_check_selftest PROPERTIES LABELS "cycle;trace;v0.10" FIXTURES_REQUIRED cycle_trace)
endif()

# Step 1a (#537): one LPDDR5 channel on the core. Its controller trace answers to the LPDDR5 trace
# validator; its core trace and counters to check_trace.py.
add_executable(cycle_dram_demo cycle_dram_demo.cpp)
target_link_libraries(cycle_dram_demo PRIVATE kpu_memory_components kpu_trace)
target_include_directories(cycle_dram_demo PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common)
kpu_set_target_options(cycle_dram_demo)
set_target_properties(cycle_dram_demo PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME cycle_dram_demo
    COMMAND cycle_dram_demo ${CMAKE_CURRENT_BINARY_DIR}/cycle_dram_demo.lpddr5.json
                            ${CMAKE_CURRENT_BINARY_DIR}/cycle_dram_demo.json
                            ${CMAKE_CURRENT_BINARY_DIR}/cycle_dram_demo.counters.json)
set_tests_properties(cycle_dram_demo PROPERTIES LABELS "cycle;lpddr5;trace;v0.10" FIXTURES_SETUP cycle_dram)
if(KPU_PYTHON3)
    add_test(NAME cycle_dram_trace_validator
        COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common/trace_validator.py
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_dram_demo.lpddr5.json)
    set_tests_properties(cycle_dram_trace_validator PROPERTIES LABELS "cycle;lpddr5;trace;v0.10" FIXTURES_REQUIRED cycle_dram)
    add_test(NAME cycle_dram_trace_check
        COMMAND ${KPU_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/check_trace.py
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_dram_demo.json
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_dram_demo.counters.json)
    set_tests_properties(cycle_dram_trace_check PROPERTIES LABELS "cycle;lpddr5;trace;v0.10" FIXTURES_REQUIRED cycle_dram)
    # The validator's INV-006 on synthetic traces: a refresh with txn_id 0 passes, any other fails.
    add_test(NAME lpddr5_trace_validator_selftest
        COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common/test_trace_validator.py)
    set_tests_properties(lpddr5_trace_validator_selftest PROPERTIES LABELS "cycle;lpddr5;trace;v0.10")
endif()

# Step 1b (#538): rung 2 through the L3 tile and the DMA engine. The controller trace answers to
# the LPDDR5 trace validator; the core trace and counters to check_trace.py.
add_executable(cycle_l3_dma_demo cycle_l3_dma_demo.cpp)
target_link_libraries(cycle_l3_dma_demo PRIVATE kpu_memory_components kpu_trace)
target_include_directories(cycle_l3_dma_demo PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common)
kpu_set_target_options(cycle_l3_dma_demo)
set_target_properties(cycle_l3_dma_demo PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME cycle_l3_dma_demo
    COMMAND cycle_l3_dma_demo ${CMAKE_CURRENT_BINARY_DIR}/cycle_l3_dma_demo.lpddr5.json
                              ${CMAKE_CURRENT_BINARY_DIR}/cycle_l3_dma_demo.json
                              ${CMAKE_CURRENT_BINARY_DIR}/cycle_l3_dma_demo.counters.json)
set_tests_properties(cycle_l3_dma_demo PROPERTIES LABELS "cycle;l3;dma;trace;v0.10" FIXTURES_SETUP cycle_l3_dma)
if(KPU_PYTHON3)
    add_test(NAME cycle_l3_dma_trace_validator
        COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common/trace_validator.py
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_l3_dma_demo.lpddr5.json)
    set_tests_properties(cycle_l3_dma_trace_validator PROPERTIES LABELS "cycle;l3;dma;trace;v0.10"
                         FIXTURES_REQUIRED cycle_l3_dma)
    add_test(NAME cycle_l3_dma_trace_check
        COMMAND ${KPU_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/check_trace.py
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_l3_dma_demo.json
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_l3_dma_demo.counters.json)
    set_tests_properties(cycle_l3_dma_trace_check PROPERTIES LABELS "cycle;l3;dma;trace;v0.10"
                         FIXTURES_REQUIRED cycle_l3_dma)
endif()

# Step 1d (#540): rung 4, an int8 64^3 matmul's L3 traffic. The controller trace answers to the
# LPDDR5 trace validator; the core trace and counters to check_trace.py.
add_executable(cycle_csp_l3_demo cycle_csp_l3_demo.cpp)
target_link_libraries(cycle_csp_l3_demo PRIVATE kpu_simulator kpu_memory_components kpu_trace kpu_program)
target_include_directories(cycle_csp_l3_demo PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common)
kpu_set_target_options(cycle_csp_l3_demo)
set_target_properties(cycle_csp_l3_demo PROPERTIES FOLDER "Tests/Cycle")
add_test(NAME cycle_csp_l3_demo
    COMMAND cycle_csp_l3_demo ${CMAKE_SOURCE_DIR}
                              ${CMAKE_CURRENT_BINARY_DIR}/cycle_csp_l3_demo.lpddr5.json
                              ${CMAKE_CURRENT_BINARY_DIR}/cycle_csp_l3_demo.json
                              ${CMAKE_CURRENT_BINARY_DIR}/cycle_csp_l3_demo.counters.json)
set_tests_properties(cycle_csp_l3_demo PROPERTIES LABELS "cycle;csp;l3;dma;trace;v0.10" FIXTURES_SETUP cycle_csp_l3)
if(KPU_PYTHON3)
    add_test(NAME cycle_csp_l3_trace_validator
        COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/patterns/memory/lpddr5/common/trace_validator.py
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_csp_l3_demo.lpddr5.json)
    set_tests_properties(cycle_csp_l3_trace_validator PROPERTIES LABELS "cycle;csp;l3;dma;trace;v0.10"
                         FIXTURES_REQUIRED cycle_csp_l3)
    add_test(NAME cycle_csp_l3_trace_check
        COMMAND ${KPU_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/check_trace.py
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_csp_l3_demo.json
                ${CMAKE_CURRENT_BINARY_DIR}/cycle_csp_l3_demo.counters.json)
    set_tests_properties(cycle_csp_l3_trace_check PROPERTIES LABELS "cycle;csp;l3;dma;trace;v0.10"
                         FIXTURES_REQUIRED cycle_csp_l3)
endif()
