# ============================================================================
# Behavioral Simulation Components
# Functional simulation models for the KPU
# ============================================================================

set(BEHAVIORAL_SOURCES
    memory_model.cpp
)

set(BEHAVIORAL_HEADERS
    ${CMAKE_SOURCE_DIR}/include/sw/kpu/behavioral/memory_model.hpp
)

add_library(kpu_behavioral_components STATIC
    ${BEHAVIORAL_SOURCES}
    ${BEHAVIORAL_HEADERS}
)

target_include_directories(kpu_behavioral_components PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

kpu_set_target_options(kpu_behavioral_components)

set_target_properties(kpu_behavioral_components PROPERTIES
    FOLDER "Components/Behavioral"
)

# Create alias for consistent naming
add_library(StillwaterKPU::BehavioralComponents ALIAS kpu_behavioral_components)
