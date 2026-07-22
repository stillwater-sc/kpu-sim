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
