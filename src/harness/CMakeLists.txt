# ============================================================================
# src/harness/CMakeLists.txt
# KPU Component Test Harness Library
#
# Provides testing infrastructure for:
# - Individual component harnesses (DMA, BlockMover, Streamer)
# - Full pipeline integration harness
# - Statistics collection and analysis
# - Schedule validation
# ============================================================================

# Harness library sources
set(HARNESS_SOURCES
    dma_harness.cpp
    block_mover_harness.cpp
    streamer_harness.cpp
    pipeline_harness.cpp
)

# Create the harness library
add_library(kpu_harness STATIC ${HARNESS_SOURCES})

# Set include directories
target_include_directories(kpu_harness PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Link dependencies
target_link_libraries(kpu_harness PUBLIC
    kpu_behavioral
)

# Set target properties
set_target_properties(kpu_harness PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Compiler options
if(MSVC)
    target_compile_options(kpu_harness PRIVATE /W3)
else()
    target_compile_options(kpu_harness PRIVATE -Wall -Wextra)
endif()

# Install
install(TARGETS kpu_harness
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
