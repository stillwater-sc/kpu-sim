# ============================================================================
# tests/isa/CMakeLists.txt
# Tests for Data Movement ISA

# Test for Data Movement ISA and program execution
add_executable(test_data_movement_isa
    test_data_movement_isa.cpp
)

target_include_directories(test_data_movement_isa PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_data_movement_isa
    PRIVATE
    kpu_isa
    kpu_components
    kpu_trace
)

target_compile_features(test_data_movement_isa PUBLIC cxx_std_20)

# Add to CTest
add_test(NAME test_data_movement_isa COMMAND test_data_movement_isa)
set_tests_properties(test_data_movement_isa PROPERTIES LABELS isa)

# Set folder organization
set_target_properties(test_data_movement_isa PROPERTIES FOLDER "Tests/ISA")

# Serialization tests
add_executable(serialization_test
    test_serialization.cpp
)

target_include_directories(serialization_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(serialization_test
    PRIVATE
    kpu_simulator
    Catch2::Catch2WithMain
)

target_compile_features(serialization_test PUBLIC cxx_std_20)

add_test(NAME serialization_test COMMAND serialization_test)
set_tests_properties(serialization_test PROPERTIES LABELS isa)

set_target_properties(serialization_test PROPERTIES FOLDER "Tests/ISA")

# Behavioral Program Executor: end-to-end DSL → execute → verify
add_executable(test_behavioral_program_executor
    test_behavioral_program_executor.cpp
)

target_include_directories(test_behavioral_program_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_behavioral_program_executor
    PRIVATE
    kpu_schedules
    kpu_dsl
    kpu_isa
    kpu_components
    kpu_trace
)

target_compile_features(test_behavioral_program_executor PUBLIC cxx_std_20)

add_test(NAME test_behavioral_program_executor COMMAND test_behavioral_program_executor)
set_tests_properties(test_behavioral_program_executor PROPERTIES LABELS "isa;behavioral")

set_target_properties(test_behavioral_program_executor PROPERTIES FOLDER "Tests/ISA")

# Transactional Program Executor: behavioral + timing overlay
add_executable(test_transactional_program_executor
    test_transactional_program_executor.cpp
)

target_include_directories(test_transactional_program_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_transactional_program_executor
    PRIVATE
    kpu_schedules
    kpu_dsl
    kpu_isa
    kpu_components
    kpu_trace
)

target_compile_features(test_transactional_program_executor PUBLIC cxx_std_20)

add_test(NAME test_transactional_program_executor COMMAND test_transactional_program_executor)
set_tests_properties(test_transactional_program_executor PROPERTIES LABELS "isa;transactional")

set_target_properties(test_transactional_program_executor PROPERTIES FOLDER "Tests/ISA")
