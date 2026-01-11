# ============================================================================
# src/config/CMakeLists.txt
# Configuration parsing library for multi-fidelity simulation
# ============================================================================

add_library(kpu_config STATIC
    simulator_config_parser.cpp
)

target_include_directories(kpu_config PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(kpu_config PUBLIC
    kpu_memory_components
    nlohmann_json::nlohmann_json
)

# Compiler options
target_compile_features(kpu_config PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kpu_config PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()
