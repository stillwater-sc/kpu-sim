# ============================================================================
# examples/schedule/CMakeLists.txt
# CSP Schedule Generation Examples
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
# ============================================================================

# CSP Schedule Demonstration
add_executable(csp_schedule_demo csp_schedule_demo.cpp)
target_link_libraries(csp_schedule_demo PRIVATE kpu_simulator)
set_target_properties(csp_schedule_demo PROPERTIES
    FOLDER "Examples/Schedule"
)

# Matmul performance test for CSP process concurrency optimization
add_executable(run_matmul run_matmul.cpp)
target_link_libraries(run_matmul PRIVATE kpu_simulator)
set_target_properties(run_matmul PROPERTIES
    FOLDER "Examples/Schedule"
)

# Unified functional + CSP timing MLP vertical slice
add_executable(unified_xor_mlp unified_xor_mlp.cpp)
target_link_libraries(unified_xor_mlp PRIVATE kpu_simulator)
set_target_properties(unified_xor_mlp PROPERTIES
    FOLDER "Examples/Schedule"
)
if(KPU_BUILD_TESTS)
    add_test(NAME unified_xor_mlp COMMAND unified_xor_mlp)
    set_tests_properties(unified_xor_mlp PROPERTIES
        LABELS "behavioral;transactional;timing;mlp;v0.9"
    )
endif()

# CSP Pipeline Educational Demo - shows credit/debit flow and TagCAM operations
add_executable(csp_pipeline_demo csp_pipeline_demo.cpp)
target_link_libraries(csp_pipeline_demo PRIVATE kpu_simulator)
set_target_properties(csp_pipeline_demo PROPERTIES
    FOLDER "Examples/Schedule"
)

# DEPRECATED: Example of incorrect timing model usage
# This demonstrates what NOT to do - kept for educational purposes
add_executable(run_matmul_DEPRECATED run_matmul_DEPRECATED.cpp)
target_link_libraries(run_matmul_DEPRECATED PRIVATE
    kpu_isa
    kpu_dsl
    kpu_schedules
    kpu_memory_components
)
set_target_properties(run_matmul_DEPRECATED PROPERTIES
    FOLDER "Examples/Schedule/Deprecated"
)

# Online softmax simulator with tile-state tracker (issue #165)
add_executable(softmax_simulator softmax_simulator.cpp)
target_link_libraries(softmax_simulator PRIVATE kpu_simulator)
set_target_properties(softmax_simulator PROPERTIES FOLDER "Examples/Schedule")
if(KPU_BUILD_TESTS)
    # Smoke test: runs the simulator and expects exit 0 (each row sums to 1)
    add_test(NAME softmax_simulator COMMAND softmax_simulator)
    set_tests_properties(softmax_simulator PROPERTIES
        LABELS "timing;softmax;tracker;example;v0.9"
    )
endif()

# Matmul simulator with tile-state tracker (companion to softmax_simulator)
add_executable(matmul_simulator matmul_simulator.cpp)
target_link_libraries(matmul_simulator PRIVATE kpu_simulator)
set_target_properties(matmul_simulator PROPERTIES FOLDER "Examples/Schedule")
if(KPU_BUILD_TESTS)
    add_test(NAME matmul_simulator COMMAND matmul_simulator)
    set_tests_properties(matmul_simulator PROPERTIES
        LABELS "timing;matmul;tracker;example;v0.9"
    )
endif()

# LayerNorm simulator with tile-state tracker (companion to softmax_simulator)
add_executable(layernorm_simulator layernorm_simulator.cpp)
target_link_libraries(layernorm_simulator PRIVATE kpu_simulator)
set_target_properties(layernorm_simulator PROPERTIES FOLDER "Examples/Schedule")
if(KPU_BUILD_TESTS)
    add_test(NAME layernorm_simulator COMMAND layernorm_simulator)
    set_tests_properties(layernorm_simulator PROPERTIES
        LABELS "timing;layernorm;tracker;example;v0.9"
    )
endif()
