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
# Usage: add_lpddr5_pattern(NAME SOURCE_FILE FOLDER_LEVEL)
#   NAME         - Pattern name (becomes lpddr5_${NAME})
#   SOURCE_FILE  - Source file relative to memory/lpddr5/
#   FOLDER_LEVEL - MSVC folder level (e.g., "Level 1 - Single Bank")
function(add_lpddr5_pattern NAME SOURCE_FILE FOLDER_LEVEL)
    add_executable(lpddr5_${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/memory/lpddr5/${SOURCE_FILE}
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

add_lpddr5_pattern(page_hits      single-bank/page_hits.cpp      "Level 1 - Single Bank")
add_lpddr5_pattern(page_conflicts single-bank/page_conflicts.cpp "Level 1 - Single Bank")
add_lpddr5_pattern(mixed_rw       single-bank/mixed_rw.cpp       "Level 1 - Single Bank")

# -----------------------------------------------------------------------------
# Level 2: Two Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(same_group  two-bank/same_group.cpp  "Level 2 - Two Bank")
add_lpddr5_pattern(diff_groups two-bank/diff_groups.cpp "Level 2 - Two Bank")

# -----------------------------------------------------------------------------
# Level 3: Three Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(three_mixed_groups three-bank/mixed_groups.cpp "Level 3 - Three Bank")
add_lpddr5_pattern(three_same_group   three-bank/same_group.cpp   "Level 3 - Three Bank")

# -----------------------------------------------------------------------------
# Level 4: Four Bank Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(full_group     four-bank/full_group.cpp     "Level 4 - Four Bank")
add_lpddr5_pattern(across_groups  four-bank/across_groups.cpp  "Level 4 - Four Bank")
add_lpddr5_pattern(page_hit_burst four-bank/page_hit_burst.cpp "Level 4 - Four Bank")

# -----------------------------------------------------------------------------
# Level 5: Dual Channel Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(independent dual-channel/independent.cpp "Level 5 - Dual Channel")
add_lpddr5_pattern(interleaved dual-channel/interleaved.cpp "Level 5 - Dual Channel")

# -----------------------------------------------------------------------------
# Level 6: Complex Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(strided    complex/strided.cpp    "Level 6 - Complex")
add_lpddr5_pattern(random     complex/random.cpp     "Level 6 - Complex")
add_lpddr5_pattern(tile_load  complex/tile_load.cpp  "Level 6 - Complex")

# ============================================================================
# GDDR6 Memory Patterns
# ============================================================================

# GDDR6-specific common headers
add_library(gddr6_pattern_common INTERFACE)
target_include_directories(gddr6_pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/gddr6/common
    ${PROJECT_SOURCE_DIR}/include
)
set_target_properties(gddr6_pattern_common PROPERTIES FOLDER "Patterns/Memory/GDDR6/Common")

# Helper function to add GDDR6 pattern
# Usage: add_gddr6_pattern(NAME SOURCE_FILE FOLDER_LEVEL)
function(add_gddr6_pattern NAME SOURCE_FILE FOLDER_LEVEL)
    add_executable(gddr6_${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/memory/gddr6/${SOURCE_FILE}
    )
    target_link_libraries(gddr6_${NAME} PRIVATE
        pattern_common
        gddr6_pattern_common
        kpu_memory_components
        kpu_trace
    )
    set_target_properties(gddr6_${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/patterns/memory/gddr6
        FOLDER "Patterns/Memory/GDDR6/${FOLDER_LEVEL}"
    )
endfunction()

# -----------------------------------------------------------------------------
# Level 1: Single Bank Patterns
# -----------------------------------------------------------------------------

add_gddr6_pattern(page_hits      single-bank/page_hits.cpp      "Level 1 - Single Bank")
add_gddr6_pattern(page_conflicts single-bank/page_conflicts.cpp "Level 1 - Single Bank")
add_gddr6_pattern(mixed_rw       single-bank/mixed_rw.cpp       "Level 1 - Single Bank")

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
