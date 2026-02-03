# ============================================================================
# src/schedules/CMakeLists.txt
# Prototypical kernel schedules (matmul, conv2d, softmax)

add_library(kpu_schedules STATIC
    matmul_schedule.cpp
    conv2d_schedule.cpp
    softmax_schedule.cpp
)

target_include_directories(kpu_schedules PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(kpu_schedules PUBLIC
    kpu_dsl
)

target_compile_features(kpu_schedules PUBLIC cxx_std_20)
