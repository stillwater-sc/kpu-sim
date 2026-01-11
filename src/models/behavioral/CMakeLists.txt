# ============================================================================
# Behavioral Simulation Components
# Functional simulation models for the KPU
# ============================================================================

set(BEHAVIORAL_SOURCES
    # Memory models
    memory/memory_model.cpp
    memory/memory_controller.cpp
    memory/l3_tile.cpp
    # Compute models
    compute/compute_fabric.cpp
    compute/vector_engine.cpp
    # Data movement models
    datamovement/dma_engine.cpp
    datamovement/block_mover.cpp
    # NoC models
    noc/noc.cpp
    # Orchestration and execution
    orchestrator.cpp
    mlp_executor.cpp
)

set(BEHAVIORAL_HEADERS
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/behavioral/memory_model.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/behavioral/block_mover.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/behavioral/vector_engine.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/behavioral/orchestrator.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/behavioral/mlp_executor.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/components/memory/behavioral_memory_controller.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/components/memory/behavioral_l3_tile.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/components/compute/behavioral_compute_fabric.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/components/dma/behavioral_dma_engine.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/noc/behavioral_noc.hpp
)

add_library(kpu_behavioral STATIC
    ${BEHAVIORAL_SOURCES}
    ${BEHAVIORAL_HEADERS}
)

target_include_directories(kpu_behavioral PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Link dependencies needed by behavioral models
target_link_libraries(kpu_behavioral
    PUBLIC
        kpu_trace
        kpu_compute_components  # For SFU (activation functions)
    PRIVATE
        $<TARGET_NAME_IF_EXISTS:spdlog::spdlog>
)

kpu_set_target_options(kpu_behavioral)

set_target_properties(kpu_behavioral PROPERTIES
    FOLDER "Simulation/Behavioral"
)

# Create alias for consistent naming
add_library(StillwaterKPU::Behavioral ALIAS kpu_behavioral)

# Keep old name for backward compatibility
add_library(kpu_behavioral_components ALIAS kpu_behavioral)
add_library(StillwaterKPU::BehavioralComponents ALIAS kpu_behavioral)
