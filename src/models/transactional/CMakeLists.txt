# ============================================================================
# Transactional Simulation Components
# Statistical simulation models for the KPU
# ============================================================================

set(TRANSACTIONAL_SOURCES
    # Memory models
    memory/memory_controller.cpp
    memory/l3_tile.cpp
    # Compute models
    compute/compute_fabric.cpp
    # Data movement models
    datamovement/dma_engine.cpp
)

set(TRANSACTIONAL_HEADERS
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/transactional/memory/memory_controller.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/transactional/memory/l3_tile.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/transactional/compute/compute_fabric.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/transactional/datamovement/dma_engine.hpp
)

add_library(kpu_transactional STATIC
    ${TRANSACTIONAL_SOURCES}
    ${TRANSACTIONAL_HEADERS}
)

target_include_directories(kpu_transactional PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Link dependencies needed by transactional models
target_link_libraries(kpu_transactional
    PUBLIC
        kpu_trace
    PRIVATE
        $<TARGET_NAME_IF_EXISTS:spdlog::spdlog>
)

kpu_set_target_options(kpu_transactional)

set_target_properties(kpu_transactional PROPERTIES
    FOLDER "Simulation/Transactional"
)

# Create alias for consistent naming
add_library(StillwaterKPU::Transactional ALIAS kpu_transactional)
