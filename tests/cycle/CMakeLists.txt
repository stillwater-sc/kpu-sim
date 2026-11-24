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
endif()
