# KPU Model Runner Tool

add_executable(kpu-runner kpu_runner.cpp)

target_link_libraries(kpu-runner
    PRIVATE
        kpu_simulator
        kpu_runtime
        kpu_compiler
        kpu_isa
)

target_include_directories(kpu-runner
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

# Install
install(TARGETS kpu-runner RUNTIME DESTINATION bin)
