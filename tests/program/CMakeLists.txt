# ============================================================================
# tests/program/CMakeLists.txt
# L0 portable-program layer (tile/stream program) tests
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
# ============================================================================

# TileProgram (L0 tile-sequence) + functional reference
kpu_add_component_test(test_tile_program "program"
    test_tile_program.cpp
)

set_tests_properties(test_tile_program PROPERTIES
    LABELS "program;tile_program;v0.9"
)

# TRANSACTIONAL tier executor (event engine + firing rule)
kpu_add_component_test(test_tile_transaction_executor "program"
    test_tile_transaction_executor.cpp
)

set_tests_properties(test_tile_transaction_executor PROPERTIES
    LABELS "program;transactional;executor;v0.9"
)

# The driver seam: one function maps a level to an interpreter (#285)
kpu_add_component_test(test_driver_run_at "program"
    test_driver_run_at.cpp
)

set_tests_properties(test_driver_run_at PROPERTIES
    LABELS "program;driver;v0.9"
)

# L0 dependency model (tile + pivot-slot hazards, diagnosis)
kpu_add_component_test(test_tile_dependencies "program"
    test_tile_dependencies.cpp
)

set_tests_properties(test_tile_dependencies PROPERTIES
    LABELS "program;deps;v0.9"
)

# TileProgram characterization (tile-DAG concurrency + modeled metrics)
kpu_add_component_test(test_tile_characterize "program"
    test_tile_characterize.cpp
)

set_tests_properties(test_tile_characterize PROPERTIES
    LABELS "program;characterize;v0.9"
)

# L1 stream signatures (spatial/temporal layer over L0)
kpu_add_component_test(test_stream_signature "program"
    test_stream_signature.cpp
)

set_tests_properties(test_stream_signature PROPERTIES
    LABELS "program;stream;v0.9"
)
