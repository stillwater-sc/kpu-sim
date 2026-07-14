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

# DMAEngineProcess tests (uses MemoryControllerProcess for DRAM access)
# DMA Engine is the ISA-driven CSP process; MC models DRAM contention
kpu_add_component_test(test_dma_engine_process "timing"
    test_dma_engine_process.cpp
)

set_tests_properties(test_dma_engine_process PROPERTIES
    LABELS "timing;dma_engine_process;v0.9"
)

# MemoryControllerProcess tests (replaces DMAEngineProcess)
kpu_add_component_test(test_memory_controller "timing"
    test_memory_controller.cpp
)

set_tests_properties(test_memory_controller PROPERTIES
    LABELS "timing;memory_controller;v0.9"
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

# ConcurrentTimingExecutor tests
kpu_add_component_test(test_concurrent_timing_executor "timing"
    test_concurrent_timing_executor.cpp
)

set_tests_properties(test_concurrent_timing_executor PROPERTIES
    LABELS "timing;executor;v0.9"
)

# ============================================================================
# Phase 4: CSP Validation and Schedule Generation Tests
# ============================================================================

# Schedule Generator tests
kpu_add_component_test(test_schedule_generators "timing"
    test_schedule_generators.cpp
)

set_tests_properties(test_schedule_generators PROPERTIES
    LABELS "timing;schedule;generators;v0.9"
)

# Livelock Scenarios tests
kpu_add_component_test(test_livelock_scenarios "timing"
    test_livelock_scenarios.cpp
)

set_tests_properties(test_livelock_scenarios PROPERTIES
    LABELS "timing;livelock;v0.9"
)

# Schedule Validation tests
kpu_add_component_test(test_schedule_validation "timing"
    test_schedule_validation.cpp
)

set_tests_properties(test_schedule_validation PROPERTIES
    LABELS "timing;schedule;validation;v0.9"
)

# Multi-tile execution regression (issues #61/#63/#64):
# executes generated schedules end-to-end across the strategy x size matrix
kpu_add_component_test(test_multi_tile_execution "timing"
    test_multi_tile_execution.cpp
)

set_tests_properties(test_multi_tile_execution PROPERTIES
    LABELS "timing;executor;regression;v0.9"
)

# Functional elementwise execution with host oracle (issue #102, epic E2):
# value-producing elementwise/broadcast on the CSP data path
kpu_add_component_test(test_functional_elementwise "timing"
    test_functional_elementwise.cpp
)

set_tests_properties(test_functional_elementwise PROPERTIES
    LABELS "timing;functional;elementwise;v0.9"
)

# Elementwise regression matrix (issue #103, epic E2): form x size x envelope
# with credit/stall invariants and characterization
kpu_add_component_test(test_elementwise_regression "timing"
    test_elementwise_regression.cpp
)

set_tests_properties(test_elementwise_regression PROPERTIES
    LABELS "timing;regression;elementwise;v0.9"
)

# Online reduction schedule generator (issue #106, epic E3): FULL_REDUCE /
# ROW_STATS / ROW_NORMALIZE with envelope-derived realization
kpu_add_component_test(test_reduction_generator "timing"
    test_reduction_generator.cpp
)

set_tests_properties(test_reduction_generator PROPERTIES
    LABELS "timing;schedule;reduction;v0.9"
)

# Functional reduction with host oracle (issue #107, epic E3): value-producing
# FULL_REDUCE / ROW_STATS on the CSP data path
kpu_add_component_test(test_functional_reduction "timing"
    test_functional_reduction.cpp
)

set_tests_properties(test_functional_reduction PROPERTIES
    LABELS "timing;functional;reduction;v0.9"
)

# Reduction regression matrix (issue #108, epic E3): op x stream-length x
# envelope with credit/stall invariants and characterization
kpu_add_component_test(test_reduction_regression "timing"
    test_reduction_regression.cpp
)

set_tests_properties(test_reduction_regression PROPERTIES
    LABELS "timing;regression;reduction;v0.9"
)

# Schedule-tier resident-dependency mechanism (issue #155, epic E8):
# compute-resident inputs (online softmax / norm apply hand-off)
kpu_add_component_test(test_resident_dependency "timing"
    test_resident_dependency.cpp
)

set_tests_properties(test_resident_dependency PROPERTIES
    LABELS "timing;executor;resident;v0.9"
)
