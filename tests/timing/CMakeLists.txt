# ============================================================================
# tests/timing/CMakeLists.txt
# Tests for v0.9.0 Concurrent Timing Model Foundation Classes
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
# ============================================================================

# CreditPool tests
kpu_add_component_test(test_credit_pool "timing"
    test_credit_pool.cpp
)

set_tests_properties(test_credit_pool PROPERTIES
    LABELS "timing;credit_pool;v0.9"
)

# TagCAM tests
kpu_add_component_test(test_tag_cam "timing"
    test_tag_cam.cpp
)

set_tests_properties(test_tag_cam PROPERTIES
    LABELS "timing;tag_cam;v0.9"
)

# WorkQueue tests
kpu_add_component_test(test_work_queue "timing"
    test_work_queue.cpp
)

set_tests_properties(test_work_queue PROPERTIES
    LABELS "timing;work_queue;v0.9"
)

# DMAEngineProcess tests
kpu_add_component_test(test_dma_engine_process "timing"
    test_dma_engine_process.cpp
)

set_tests_properties(test_dma_engine_process PROPERTIES
    LABELS "timing;dma_engine_process;v0.9"
)

# BlockMoverProcess tests
kpu_add_component_test(test_block_mover_process "timing"
    test_block_mover_process.cpp
)

set_tests_properties(test_block_mover_process PROPERTIES
    LABELS "timing;block_mover_process;v0.9"
)

# StreamerProcess tests
kpu_add_component_test(test_streamer_process "timing"
    test_streamer_process.cpp
)

set_tests_properties(test_streamer_process PROPERTIES
    LABELS "timing;streamer_process;v0.9"
)

# Component integration tests
kpu_add_component_test(test_component_integration "timing"
    test_component_integration.cpp
)

set_tests_properties(test_component_integration PROPERTIES
    LABELS "timing;integration;v0.9"
)
