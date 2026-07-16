# ============================================================================
# examples/milestones/CMakeLists.txt
# DNN milestone demos (plan: docs/plans/csp_pattern_coverage_roadmap.md §6)
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
# ============================================================================

# M1: MLP baseline demo + benchmark (issue #129)
add_executable(m1_mlp_baseline m1_mlp_baseline.cpp)
target_link_libraries(m1_mlp_baseline PRIVATE kpu_simulator)
set_target_properties(m1_mlp_baseline PROPERTIES
    FOLDER "Examples/Milestones"
)
if(KPU_BUILD_TESTS)
    add_test(NAME m1_mlp_baseline COMMAND m1_mlp_baseline)
    set_tests_properties(m1_mlp_baseline PROPERTIES
        LABELS "timing;mlp;milestone;v0.9"
    )
endif()

add_executable(m2_resnet m2_resnet.cpp)
target_link_libraries(m2_resnet PRIVATE kpu_simulator)
set_target_properties(m2_resnet PROPERTIES
    FOLDER "Examples/Milestones"
)
if(KPU_BUILD_TESTS)
    add_test(NAME m2_resnet COMMAND m2_resnet)
    set_tests_properties(m2_resnet PROPERTIES
        LABELS "timing;resnet;milestone;v0.9"
    )
endif()
