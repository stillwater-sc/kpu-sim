# ============================================================================
# tests/dsl/CMakeLists.txt
# Tests for Schedule DSL and kernel schedules

add_executable(test_schedule_dsl
    test_schedule_dsl.cpp
)

target_include_directories(test_schedule_dsl PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_schedule_dsl
    PRIVATE
    kpu_schedules
    kpu_dsl
    kpu_isa
    kpu_components
    kpu_trace
)

target_compile_features(test_schedule_dsl PUBLIC cxx_std_20)

add_test(NAME test_schedule_dsl COMMAND test_schedule_dsl)
set_tests_properties(test_schedule_dsl PROPERTIES LABELS dsl)
set_target_properties(test_schedule_dsl PROPERTIES FOLDER "Tests/DSL")
