# ============================================================================
# examples/program/CMakeLists.txt
# Program-stack examples (#503): one CSP program at every level, checked against L-B.

add_executable(program_run_levels run_levels.cpp)
target_link_libraries(program_run_levels PRIVATE StillwaterKPU::Simulator)
target_compile_definitions(program_run_levels PRIVATE KPU_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
kpu_set_target_options(program_run_levels)
set_target_properties(program_run_levels PROPERTIES FOLDER "Examples/Program")

# KPU_BUILD_TESTS, not BUILD_TESTING (empty in this project). Each fails on a value mismatch.
if(KPU_BUILD_TESTS)
    # A corpus matmul at L-B, L-T1 and L-CA.
    add_test(NAME example_program_run_levels COMMAND program_run_levels)
    # A program that stages A's row panel in L2: the walker's L3 and L2 slots, executed.
    add_test(NAME example_program_placement COMMAND program_run_levels
        --program ${CMAKE_CURRENT_SOURCE_DIR}/matmul_48_stage_a.csp --placement)
    set_tests_properties(example_program_run_levels example_program_placement PROPERTIES
        LABELS "program;example" TIMEOUT 120)
endif()
