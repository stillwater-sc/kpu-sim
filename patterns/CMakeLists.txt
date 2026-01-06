# patterns/CMakeLists.txt
#
# Multi-Component Pattern Test Suite
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

# ============================================================================
# Common Pattern Infrastructure
# ============================================================================

# Top-level pattern common headers (cross-component)
add_library(pattern_common INTERFACE)
target_include_directories(pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# ============================================================================
# Memory Patterns
# ============================================================================

# LPDDR5-specific common headers
add_library(lpddr5_pattern_common INTERFACE)
target_include_directories(lpddr5_pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/lpddr5/common
    ${PROJECT_SOURCE_DIR}/include
)

# Helper function to add LPDDR5 pattern
function(add_lpddr5_pattern NAME SOURCE_DIR)
    add_executable(lpddr5_${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/memory/lpddr5/${SOURCE_DIR}/main.cpp
    )
    target_link_libraries(lpddr5_${NAME} PRIVATE
        pattern_common
        lpddr5_pattern_common
        kpu_memory_components
        kpu_trace
    )
    set_target_properties(lpddr5_${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/patterns/memory/lpddr5
    )
endfunction()

# -----------------------------------------------------------------------------
# Level 1: Single Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(page_hits single-bank/page-hits)
add_lpddr5_pattern(page_conflicts single-bank/page-conflicts)
add_lpddr5_pattern(mixed_rw single-bank/mixed-rw)

# -----------------------------------------------------------------------------
# Level 2: Two Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(same_group two-bank/same-group)
add_lpddr5_pattern(diff_groups two-bank/diff-groups)

# -----------------------------------------------------------------------------
# Level 3: Three Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(three_mixed_groups three-bank/mixed-groups)
add_lpddr5_pattern(three_same_group three-bank/same-group)

# -----------------------------------------------------------------------------
# Level 4: Four Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(full_group four-bank/full-group)
add_lpddr5_pattern(across_groups four-bank/across-groups)
add_lpddr5_pattern(page_hit_burst four-bank/page-hit-burst)

# -----------------------------------------------------------------------------
# Level 5: Dual Channel Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(independent dual-channel/independent)
add_lpddr5_pattern(interleaved dual-channel/interleaved)

# -----------------------------------------------------------------------------
# Level 6: Complex Patterns (TODO)
# -----------------------------------------------------------------------------

# add_lpddr5_pattern(strided complex/strided)
# add_lpddr5_pattern(random complex/random)
# add_lpddr5_pattern(tile_load complex/tile-load)

# ============================================================================
# NoC Patterns (TODO)
# ============================================================================

# add_subdirectory(noc)

# ============================================================================
# Compute Patterns (TODO)
# ============================================================================

# add_subdirectory(compute)

# ============================================================================
# Connection Patterns (TODO)
# ============================================================================

# add_subdirectory(connection)
