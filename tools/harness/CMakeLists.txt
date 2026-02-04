# ============================================================================
# tools/harness/CMakeLists.txt
# KPU Harness CLI Tools
# ============================================================================

# Schedule runner CLI
add_executable(schedule-runner schedule_runner.cpp)

target_include_directories(schedule-runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(schedule-runner PRIVATE
    kpu_harness
    kpu_behavioral
)

set_target_properties(schedule-runner PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(MSVC)
    target_compile_options(schedule-runner PRIVATE /W3)
else()
    target_compile_options(schedule-runner PRIVATE -Wall -Wextra)
endif()

# Install
install(TARGETS schedule-runner
    RUNTIME DESTINATION bin
)
