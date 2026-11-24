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
