# src/benchmark/CMakeLists.txt
# Benchmark library for KPU simulator

add_library(kpu_benchmark STATIC
    benchmark.cpp
)

target_include_directories(kpu_benchmark PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kpu_benchmark PUBLIC
    kpu_simulator
    kpu_isa
)

add_library(StillwaterKPU::Benchmark ALIAS kpu_benchmark)

kpu_set_target_options(kpu_benchmark)

set_target_properties(kpu_benchmark PROPERTIES
    FOLDER "src"
)
