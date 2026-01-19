# src/dfx/CMakeLists.txt
# DFX parser library for Python kpu package format

add_library(kpu_dfx STATIC
    dfx_parser.cpp
)

target_include_directories(kpu_dfx
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kpu_dfx
    PUBLIC
        nlohmann_json::nlohmann_json
)

# Set C++17 standard
target_compile_features(kpu_dfx PUBLIC cxx_std_17)

# Add alias for consistent naming
add_library(kpu::dfx ALIAS kpu_dfx)
