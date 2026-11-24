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
