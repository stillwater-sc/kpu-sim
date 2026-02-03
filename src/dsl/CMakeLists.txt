# ============================================================================
# src/dsl/CMakeLists.txt
# Schedule DSL for KPU system-level schedules

add_library(kpu_dsl STATIC
    schedule.cpp
    schedule_compiler.cpp
)

target_include_directories(kpu_dsl PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(kpu_dsl PUBLIC
    kpu_isa
)

target_compile_features(kpu_dsl PUBLIC cxx_std_20)
