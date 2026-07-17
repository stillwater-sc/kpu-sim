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

# M3: MobileNetV2 demo + benchmark (issue #131)
add_executable(m3_mobilenet m3_mobilenet.cpp)
target_link_libraries(m3_mobilenet PRIVATE kpu_simulator)
set_target_properties(m3_mobilenet PROPERTIES
    FOLDER "Examples/Milestones"
)
if(KPU_BUILD_TESTS)
    add_test(NAME m3_mobilenet COMMAND m3_mobilenet)
    set_tests_properties(m3_mobilenet PROPERTIES
        LABELS "timing;mobilenet;milestone;v0.9"
    )
endif()

# M3: EfficientNet-B0 demo + benchmark (issue #131)
add_executable(m3_efficientnet m3_efficientnet.cpp)
target_link_libraries(m3_efficientnet PRIVATE kpu_simulator)
set_target_properties(m3_efficientnet PROPERTIES
    FOLDER "Examples/Milestones"
)
if(KPU_BUILD_TESTS)
    add_test(NAME m3_efficientnet COMMAND m3_efficientnet)
    set_tests_properties(m3_efficientnet PROPERTIES
        LABELS "timing;efficientnet;milestone;v0.9"
    )
endif()
