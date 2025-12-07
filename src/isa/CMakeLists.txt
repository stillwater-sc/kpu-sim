# ============================================================================
# src/isa/CMakeLists.txt
# Data Movement ISA for Domain Flow Architecture

add_library(kpu_isa STATIC
    data_movement_isa.cpp
    program_executor.cpp
    concurrent_executor.cpp
    tile_layout.cpp
    tile_cache.cpp
)

target_include_directories(kpu_isa PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(kpu_isa PUBLIC
    kpu_components
    kpu_trace
)

target_compile_features(kpu_isa PUBLIC cxx_std_20)
