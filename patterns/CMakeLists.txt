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
add_lpddr5_pattern(stream     complex/stream.cpp     "Level 6 - Complex")
add_lpddr5_pattern(multi_dma  complex/multi_dma.cpp  "Level 6 - Complex")

# -----------------------------------------------------------------------------
# Level 7: Bandwidth Patterns
# -----------------------------------------------------------------------------

add_lpddr5_pattern(max_bandwidth bandwidth/max_bandwidth.cpp "Level 7 - Bandwidth")
add_lpddr5_pattern(page_burst    bandwidth/page_burst.cpp    "Level 7 - Bandwidth")

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

# -----------------------------------------------------------------------------
# Level 2: Two Bank Patterns
# -----------------------------------------------------------------------------

add_gddr6_pattern(two_same_group  two-bank/same_group.cpp  "Level 2 - Two Bank")
add_gddr6_pattern(two_diff_groups two-bank/diff_groups.cpp "Level 2 - Two Bank")

# -----------------------------------------------------------------------------
# Level 3: Three Bank Patterns
# -----------------------------------------------------------------------------

add_gddr6_pattern(three_same_group   three-bank/same_group.cpp   "Level 3 - Three Bank")
add_gddr6_pattern(three_mixed_groups three-bank/mixed_groups.cpp "Level 3 - Three Bank")

# -----------------------------------------------------------------------------
# Level 4: Four Bank Patterns
# -----------------------------------------------------------------------------

add_gddr6_pattern(full_group     four-bank/full_group.cpp     "Level 4 - Four Bank")
add_gddr6_pattern(across_groups  four-bank/across_groups.cpp  "Level 4 - Four Bank")
add_gddr6_pattern(page_hit_burst four-bank/page_hit_burst.cpp "Level 4 - Four Bank")

# -----------------------------------------------------------------------------
# Level 5: Dual Channel Patterns
# -----------------------------------------------------------------------------

add_gddr6_pattern(dual_independent dual-channel/independent.cpp "Level 5 - Dual Channel")
add_gddr6_pattern(dual_interleaved dual-channel/interleaved.cpp "Level 5 - Dual Channel")

# -----------------------------------------------------------------------------
# Level 6: Complex Patterns
# -----------------------------------------------------------------------------

add_gddr6_pattern(strided    complex/strided.cpp    "Level 6 - Complex")
add_gddr6_pattern(random     complex/random.cpp     "Level 6 - Complex")
add_gddr6_pattern(tile_load  complex/tile_load.cpp  "Level 6 - Complex")
add_gddr6_pattern(stream     complex/stream.cpp     "Level 6 - Complex")
add_gddr6_pattern(multi_dma  complex/multi_dma.cpp  "Level 6 - Complex")

# -----------------------------------------------------------------------------
# Level 7: Bandwidth Patterns
# -----------------------------------------------------------------------------

add_gddr6_pattern(eight_bank_bandwidth bandwidth/eight_bank_bandwidth.cpp "Level 7 - Bandwidth")
add_gddr6_pattern(max_bandwidth        bandwidth/max_bandwidth.cpp        "Level 7 - Bandwidth")
add_gddr6_pattern(page_burst           bandwidth/page_burst.cpp           "Level 7 - Bandwidth")

# ============================================================================
# HBM2 Memory Patterns
# ============================================================================

# HBM2-specific common headers
add_library(hbm2_pattern_common INTERFACE)
target_include_directories(hbm2_pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/hbm2/common
    ${PROJECT_SOURCE_DIR}/include
)
set_target_properties(hbm2_pattern_common PROPERTIES FOLDER "Patterns/Memory/HBM2/Common")

# Helper function to add HBM2 pattern
# Usage: add_hbm2_pattern(NAME SOURCE_FILE FOLDER_LEVEL)
function(add_hbm2_pattern NAME SOURCE_FILE FOLDER_LEVEL)
    add_executable(hbm2_${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/memory/hbm2/${SOURCE_FILE}
    )
    target_link_libraries(hbm2_${NAME} PRIVATE
        pattern_common
        hbm2_pattern_common
        kpu_memory_components
        kpu_trace
    )
    set_target_properties(hbm2_${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/patterns/memory/hbm2
        FOLDER "Patterns/Memory/HBM2/${FOLDER_LEVEL}"
    )
endfunction()

# -----------------------------------------------------------------------------
# Level 1: Single Bank Patterns
# -----------------------------------------------------------------------------

add_hbm2_pattern(page_hits      single-bank/hbm2_page_hits.cpp      "Level 1 - Single Bank")
add_hbm2_pattern(page_conflicts single-bank/hbm2_page_conflicts.cpp "Level 1 - Single Bank")
add_hbm2_pattern(mixed_rw       single-bank/hbm2_mixed_rw.cpp       "Level 1 - Single Bank")

# -----------------------------------------------------------------------------
# Level 2: Two Bank Patterns
# -----------------------------------------------------------------------------

add_hbm2_pattern(same_group  two-bank/hbm2_same_group.cpp  "Level 2 - Two Bank")
add_hbm2_pattern(diff_groups two-bank/hbm2_diff_groups.cpp "Level 2 - Two Bank")

# -----------------------------------------------------------------------------
# Level 3: Pseudo-Channel Patterns
# -----------------------------------------------------------------------------

add_hbm2_pattern(dual_pc pseudo-channel/hbm2_dual_pc.cpp "Level 3 - Pseudo Channel")

# -----------------------------------------------------------------------------
# Level 4: Multi-Channel Patterns
# -----------------------------------------------------------------------------

add_hbm2_pattern(four_channel  multi-channel/hbm2_four_channel.cpp  "Level 4 - Multi Channel")
add_hbm2_pattern(eight_channel multi-channel/hbm2_eight_channel.cpp "Level 4 - Multi Channel")

# -----------------------------------------------------------------------------
# Level 5: Bandwidth Patterns
# -----------------------------------------------------------------------------

add_hbm2_pattern(max_bandwidth bandwidth/hbm2_max_bandwidth.cpp "Level 5 - Bandwidth")

# -----------------------------------------------------------------------------
# Level 6: Full Bank Patterns (16 banks per PC, 256 total)
# -----------------------------------------------------------------------------

add_hbm2_pattern(all_banks_parallel    full-bank/hbm2_all_banks_parallel.cpp    "Level 6 - Full Bank")
add_hbm2_pattern(bank_group_stress     full-bank/hbm2_bank_group_stress.cpp     "Level 6 - Full Bank")
add_hbm2_pattern(full_stack_saturation full-bank/hbm2_full_stack_saturation.cpp "Level 6 - Full Bank")
add_hbm2_pattern(bank_distribution     full-bank/hbm2_bank_distribution.cpp     "Level 6 - Full Bank")

# ============================================================================
# HBM3 Memory Patterns
# ============================================================================

# HBM3-specific common headers
add_library(hbm3_pattern_common INTERFACE)
target_include_directories(hbm3_pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/hbm3/common
    ${PROJECT_SOURCE_DIR}/include
)
set_target_properties(hbm3_pattern_common PROPERTIES FOLDER "Patterns/Memory/HBM3/Common")

# Helper function to add HBM3 pattern
# Usage: add_hbm3_pattern(NAME SOURCE_FILE FOLDER_LEVEL)
function(add_hbm3_pattern NAME SOURCE_FILE FOLDER_LEVEL)
    add_executable(hbm3_${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/memory/hbm3/${SOURCE_FILE}
    )
    target_link_libraries(hbm3_${NAME} PRIVATE
        pattern_common
        hbm3_pattern_common
        kpu_memory_components
        kpu_trace
    )
    set_target_properties(hbm3_${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/patterns/memory/hbm3
        FOLDER "Patterns/Memory/HBM3/${FOLDER_LEVEL}"
    )
endfunction()

# -----------------------------------------------------------------------------
# Level 1: Single Bank Patterns
# -----------------------------------------------------------------------------

add_hbm3_pattern(page_hits      single-bank/hbm3_page_hits.cpp      "Level 1 - Single Bank")
add_hbm3_pattern(page_conflicts single-bank/hbm3_page_conflicts.cpp "Level 1 - Single Bank")
add_hbm3_pattern(mixed_rw       single-bank/hbm3_mixed_rw.cpp       "Level 1 - Single Bank")

# -----------------------------------------------------------------------------
# Level 2: Two Bank Patterns
# -----------------------------------------------------------------------------

add_hbm3_pattern(same_group  two-bank/hbm3_same_group.cpp  "Level 2 - Two Bank")
add_hbm3_pattern(diff_groups two-bank/hbm3_diff_groups.cpp "Level 2 - Two Bank")

# -----------------------------------------------------------------------------
# Level 3: Pseudo-Channel Patterns
# -----------------------------------------------------------------------------

add_hbm3_pattern(dual_pc pseudo-channel/hbm3_dual_pc.cpp "Level 3 - Pseudo Channel")

# -----------------------------------------------------------------------------
# Level 4: Multi-Channel Patterns
# -----------------------------------------------------------------------------

add_hbm3_pattern(eight_channel   multi-channel/hbm3_eight_channel.cpp   "Level 4 - Multi Channel")
add_hbm3_pattern(sixteen_channel multi-channel/hbm3_sixteen_channel.cpp "Level 4 - Multi Channel")

# -----------------------------------------------------------------------------
# Level 5: Bandwidth Patterns
# -----------------------------------------------------------------------------

add_hbm3_pattern(max_bandwidth bandwidth/hbm3_max_bandwidth.cpp "Level 5 - Bandwidth")

# -----------------------------------------------------------------------------
# Level 6: Full Bank Patterns (16 banks per PC, 512 total)
# -----------------------------------------------------------------------------

add_hbm3_pattern(all_banks_parallel    full-bank/hbm3_all_banks_parallel.cpp    "Level 6 - Full Bank")
add_hbm3_pattern(bank_group_stress     full-bank/hbm3_bank_group_stress.cpp     "Level 6 - Full Bank")
add_hbm3_pattern(full_stack_saturation full-bank/hbm3_full_stack_saturation.cpp "Level 6 - Full Bank")
add_hbm3_pattern(bank_distribution     full-bank/hbm3_bank_distribution.cpp     "Level 6 - Full Bank")

# ============================================================================
# HBM2E Memory Patterns (3.2-3.6 Gbps variants)
# ============================================================================

# HBM2E-specific common headers
add_library(hbm2e_pattern_common INTERFACE)
target_include_directories(hbm2e_pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/hbm2e/common
)
target_link_libraries(hbm2e_pattern_common INTERFACE kpu_memory_components kpu_trace)
set_target_properties(hbm2e_pattern_common PROPERTIES FOLDER "Patterns/Memory/HBM2E/Common")

# Helper function to add HBM2E pattern
# Usage: add_hbm2e_pattern(NAME SOURCE_FILE FOLDER_LEVEL)
function(add_hbm2e_pattern NAME SOURCE_FILE FOLDER_LEVEL)
    set(TARGET_NAME "hbm2e_${NAME}")
    add_executable(${TARGET_NAME} memory/hbm2e/${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE hbm2e_pattern_common)
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/patterns/memory/hbm2e"
        FOLDER "Patterns/Memory/HBM2E/${FOLDER_LEVEL}"
    )
endfunction()

# -----------------------------------------------------------------------------
# HBM2E-3600 Patterns (3.6 Gbps @ 1.8 GHz, 460.8 GB/s peak)
# -----------------------------------------------------------------------------

add_hbm2e_pattern(3600_page_hits      single-bank/hbm2e_3600_page_hits.cpp      "HBM2E-3600")
add_hbm2e_pattern(3600_page_conflicts single-bank/hbm2e_3600_page_conflicts.cpp "HBM2E-3600")
add_hbm2e_pattern(3600_max_bandwidth  bandwidth/hbm2e_3600_max_bandwidth.cpp    "HBM2E-3600")

# -----------------------------------------------------------------------------
# HBM2E-3200 Patterns (3.2 Gbps @ 1.6 GHz, 409.6 GB/s peak)
# -----------------------------------------------------------------------------

add_hbm2e_pattern(3200_page_hits      single-bank/hbm2e_3200_page_hits.cpp      "HBM2E-3200")

# ============================================================================
# HBM3E Memory Patterns (8.4-9.6 Gbps variants)
# ============================================================================

# HBM3E-specific common headers
add_library(hbm3e_pattern_common INTERFACE)
target_include_directories(hbm3e_pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/hbm3e/common
)
target_link_libraries(hbm3e_pattern_common INTERFACE kpu_memory_components kpu_trace)
set_target_properties(hbm3e_pattern_common PROPERTIES FOLDER "Patterns/Memory/HBM3E/Common")

# Helper function to add HBM3E pattern
# Usage: add_hbm3e_pattern(NAME SOURCE_FILE FOLDER_LEVEL)
function(add_hbm3e_pattern NAME SOURCE_FILE FOLDER_LEVEL)
    set(TARGET_NAME "hbm3e_${NAME}")
    add_executable(${TARGET_NAME} memory/hbm3e/${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE hbm3e_pattern_common)
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/patterns/memory/hbm3e"
        FOLDER "Patterns/Memory/HBM3E/${FOLDER_LEVEL}"
    )
endfunction()

# -----------------------------------------------------------------------------
# HBM3E-9600 Patterns (9.6 Gbps @ 4.8 GHz, 1.229 TB/s peak)
# -----------------------------------------------------------------------------

add_hbm3e_pattern(9600_page_hits      single-bank/hbm3e_9600_page_hits.cpp      "HBM3E-9600")
add_hbm3e_pattern(9600_page_conflicts single-bank/hbm3e_9600_page_conflicts.cpp "HBM3E-9600")
add_hbm3e_pattern(9600_max_bandwidth  bandwidth/hbm3e_9600_max_bandwidth.cpp    "HBM3E-9600")

# -----------------------------------------------------------------------------
# HBM3E-8400 Patterns (8.4 Gbps @ 4.2 GHz, 1.075 TB/s peak)
# -----------------------------------------------------------------------------

add_hbm3e_pattern(8400_page_hits      single-bank/hbm3e_8400_page_hits.cpp      "HBM3E-8400")

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
