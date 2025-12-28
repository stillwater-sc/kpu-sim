# ============================================================================
# examples/analysis/CMakeLists.txt
# Analysis tools and examples

add_executable(example_matmul_schedule_analysis
    matmul_schedule_analysis.cpp
)

target_include_directories(example_matmul_schedule_analysis PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Header-only, no library dependencies needed
