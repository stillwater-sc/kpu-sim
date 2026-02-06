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
