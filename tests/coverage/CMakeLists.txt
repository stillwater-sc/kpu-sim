# ============================================================================
# tests/coverage/CMakeLists.txt
# CSP pattern-coverage matrix (issue #93)
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
# ============================================================================

kpu_add_component_test(test_pattern_coverage "coverage"
    test_pattern_coverage.cpp
)

target_compile_definitions(test_pattern_coverage PRIVATE
    COVERAGE_MANIFEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/pattern_coverage.json"
)

set_tests_properties(test_pattern_coverage PROPERTIES
    LABELS "coverage;csp-patterns;v0.9"
)
