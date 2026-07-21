# ============================================================================
# examples/characterize/CMakeLists.txt
# Tile-program characterization / design-of-experiments harness
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
# ============================================================================

add_executable(tile_characterize tile_characterize.cpp)
target_link_libraries(tile_characterize PRIVATE kpu_simulator)
set_target_properties(tile_characterize PROPERTIES FOLDER "Examples/Characterize")

# Smoke test: a tiny sweep must run and functionally validate (max_err small).
if(BUILD_TESTING)
    add_test(NAME tile_characterize_smoke
        COMMAND tile_characterize --algo lu --sizes 32 --tiles 8,16 --compute-tiles 1,4
    )
    set_tests_properties(tile_characterize_smoke PROPERTIES LABELS "program;characterize")
endif()
