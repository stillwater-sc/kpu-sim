# Runtime Library
# Host-side orchestration for kernel execution

set(RUNTIME_SOURCES
    runtime.cpp
    executor.cpp
    graph.cpp
)

set(RUNTIME_HEADERS
    ${CMAKE_SOURCE_DIR}/include/sw/runtime/runtime.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/runtime/executor.hpp
    ${CMAKE_SOURCE_DIR}/include/sw/runtime/graph.hpp
)

# Behavioral simulation module
set(BEHAVIORAL_SOURCES
    behavioral/memory_model.cpp
)

set(BEHAVIORAL_HEADERS
    ${CMAKE_SOURCE_DIR}/include/sw/runtime/behavioral/memory_model.hpp
)

add_library(kpu_runtime STATIC
    ${RUNTIME_SOURCES}
    ${RUNTIME_HEADERS}
    ${BEHAVIORAL_SOURCES}
    ${BEHAVIORAL_HEADERS}
)

target_include_directories(kpu_runtime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(kpu_runtime PUBLIC
    StillwaterKPU::Simulator
    kpu_isa
    kpu_compiler
)

kpu_set_target_options(kpu_runtime)

# Create alias for consistent naming
add_library(StillwaterKPU::Runtime ALIAS kpu_runtime)

set_target_properties(kpu_runtime PROPERTIES
    FOLDER "Libraries"
)
