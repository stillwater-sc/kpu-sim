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
    tiled_matmul_program.cpp
)

set(BEHAVIORAL_HEADERS
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/memory/memory_model.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/memory/memory_controller.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/memory/l3_tile.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/compute/compute_fabric.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/compute/vector_engine.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/datamovement/block_mover.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/datamovement/dma_engine.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/noc/noc.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/orchestrator.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/models/behavioral/mlp_executor.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/behavioral/tiled_matmul_program.hpp
)

add_library(kpu_behavioral STATIC
    ${BEHAVIORAL_SOURCES}
    ${BEHAVIORAL_HEADERS}
)

target_include_directories(kpu_behavioral PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Add Universal library include path for quantization type dispatch
# Universal provides software emulation of fp16, bf16, fp8 variants, and fp4
if(UNIVERSAL_INCLUDE_DIR)
    target_include_directories(kpu_behavioral PRIVATE ${UNIVERSAL_INCLUDE_DIR})
endif()

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
