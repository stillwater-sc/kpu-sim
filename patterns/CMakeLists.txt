# patterns/CMakeLists.txt
#
# LPDDR5 Memory Controller Pattern Test Suite
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

# Pattern common headers (header-only)
add_library(pattern_common INTERFACE)
target_include_directories(pattern_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Pattern 1: LPDDR5 Bank Access Patterns
# Tests single bank, multi-bank, page hits, conflicts, and turnarounds
add_executable(single_dma_single_bank
    pattern01_single_dma_single_bank/main.cpp
)
target_link_libraries(single_dma_single_bank PRIVATE
    pattern_common
    kpu_memory_components
    kpu_trace
)
target_include_directories(single_dma_single_bank PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Set output directory
set_target_properties(single_dma_single_bank PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/patterns
)

# Future patterns will be added as they are implemented:
# Level 1: Single Bank
# add_subdirectory(level1_single_bank/pattern02_page_conflict)
# add_subdirectory(level1_single_bank/pattern03_mixed_rw)

# Level 2: Two Banks
# add_subdirectory(level2_two_banks/pattern04_same_group)
# add_subdirectory(level2_two_banks/pattern05_diff_groups)

# Level 3: Three Banks
# add_subdirectory(level3_three_banks/pattern06_mixed_groups)
# add_subdirectory(level3_three_banks/pattern07_same_group)

# Level 4: Four Banks
# add_subdirectory(level4_four_banks/pattern08_full_group)
# add_subdirectory(level4_four_banks/pattern09_across_groups)
# add_subdirectory(level4_four_banks/pattern10_page_hit_burst)

# Level 5: Dual Channel
# add_subdirectory(level5_dual_channel/pattern11_independent)
# add_subdirectory(level5_dual_channel/pattern12_interleaved)

# Level 6: Complex Patterns
# add_subdirectory(level6_complex/pattern13_strided)
# add_subdirectory(level6_complex/pattern14_random)
# add_subdirectory(level6_complex/pattern15_tile_load)
