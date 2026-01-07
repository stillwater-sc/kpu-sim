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
set_target_properties(pattern_common PROPERTIES FOLDER "Patterns/Common")

# ============================================================================
# Memory Patterns
# ============================================================================

# LPDDR5-specific common headers
add_library(lpddr5_pattern_common INTERFACE)
target_include_directories(lpddr5_pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/lpddr5/common
    ${PROJECT_SOURCE_DIR}/include
)
set_target_properties(lpddr5_pattern_common PROPERTIES FOLDER "Patterns/Memory/LPDDR5/Common")

# Helper function to add LPDDR5 pattern
# Usage: add_lpddr5_pattern(NAME SOURCE_DIR FOLDER_LEVEL)
#   NAME         - Pattern name (becomes lpddr5_${NAME})
#   SOURCE_DIR   - Source directory relative to memory/lpddr5/
#   FOLDER_LEVEL - MSVC folder level (e.g., "Level 1 - Single Bank")
function(add_lpddr5_pattern NAME SOURCE_DIR FOLDER_LEVEL)
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
        FOLDER "Patterns/Memory/LPDDR5/${FOLDER_LEVEL}"
    )
endfunction()

# -----------------------------------------------------------------------------
# Level 1: Single Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(page_hits      single-bank/page-hits      "Level 1 - Single Bank")
add_lpddr5_pattern(page_conflicts single-bank/page-conflicts "Level 1 - Single Bank")
add_lpddr5_pattern(mixed_rw       single-bank/mixed-rw       "Level 1 - Single Bank")

# -----------------------------------------------------------------------------
# Level 2: Two Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(same_group  two-bank/same-group  "Level 2 - Two Bank")
add_lpddr5_pattern(diff_groups two-bank/diff-groups "Level 2 - Two Bank")

# -----------------------------------------------------------------------------
# Level 3: Three Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(three_mixed_groups three-bank/mixed-groups "Level 3 - Three Bank")
add_lpddr5_pattern(three_same_group   three-bank/same-group   "Level 3 - Three Bank")

# -----------------------------------------------------------------------------
# Level 4: Four Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(full_group     four-bank/full-group     "Level 4 - Four Bank")
add_lpddr5_pattern(across_groups  four-bank/across-groups  "Level 4 - Four Bank")
add_lpddr5_pattern(page_hit_burst four-bank/page-hit-burst "Level 4 - Four Bank")

# -----------------------------------------------------------------------------
# Level 5: Dual Channel Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(independent dual-channel/independent "Level 5 - Dual Channel")
add_lpddr5_pattern(interleaved dual-channel/interleaved "Level 5 - Dual Channel")

# -----------------------------------------------------------------------------
# Level 6: Complex Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(strided    complex/strided    "Level 6 - Complex")
add_lpddr5_pattern(random     complex/random     "Level 6 - Complex")
add_lpddr5_pattern(tile_load  complex/tile-load  "Level 6 - Complex")

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
