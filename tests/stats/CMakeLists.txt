# tests/stats/CMakeLists.txt
# Statistics Collection Framework tests (v0.3.4)

# Stats test suite
add_executable(stats_tests test_stats.cpp)
target_link_libraries(stats_tests PRIVATE
    Catch2::Catch2WithMain
    kpu_simulator
)
kpu_set_target_options(stats_tests)
set_target_properties(stats_tests PROPERTIES FOLDER "Tests/Stats")

# Register with CTest
add_test(NAME stats_tests COMMAND stats_tests)
set_tests_properties(stats_tests PROPERTIES LABELS "stats")
