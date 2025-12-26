# ============================================================================
# tools/CMakeLists.txt
# Development tools and utilities
#
# Tool categories:
#   - compiler/     : Compilation toolchain (kpu-kernel-compiler)
#   - runtime/      : Runtime/loader tools (kpu-loader, kpu-driver)
#   - analysis/     : Analysis and profiling (kpu-profiler, kpu-trace-analyzer)
#   - development/  : Development tools (kpu-assembler, kpu-debugger)
#   - configuration/: Configuration tools (kpu-config)
#   - benchmark/    : Benchmarking tools (kpu-benchmark)
# ============================================================================

# Helper function to create tool executables
function(kpu_add_tool name)
    set(options "")
    set(oneValueArgs FOLDER DESCRIPTION)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(TOOL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${name} ${TOOL_SOURCES})
    add_executable(StillwaterKPU::${name} ALIAS ${name})

    target_link_libraries(${name} PRIVATE
        StillwaterKPU::Simulator
        ${TOOL_LIBS}
    )

    kpu_set_target_options(${name})

    set_target_properties(${name} PROPERTIES
        OUTPUT_NAME kpu-${name}
        FOLDER "Tools/${TOOL_FOLDER}"
    )

    install(TARGETS ${name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT Tools
    )
endfunction()

# ============================================================================
# Compiler Tools
# ============================================================================
add_subdirectory(compiler)

# ============================================================================
# Runtime Tools
# ============================================================================
add_subdirectory(runtime)

# ============================================================================
# Model Runner Tools
# ============================================================================
add_subdirectory(runner)

# ============================================================================
# Analysis Tools
# ============================================================================
add_subdirectory(analysis)

# ============================================================================
# Development Tools
# ============================================================================
add_subdirectory(development)

# ============================================================================
# Configuration Tools
# ============================================================================
add_subdirectory(configuration)

# ============================================================================
# Benchmark Tools
# ============================================================================
add_subdirectory(benchmark)

# ============================================================================
# Legacy C++ tools (deprecated - to be migrated)
# ============================================================================
# add_subdirectory(cpp)

# ============================================================================
# Python tools
# ============================================================================
if(KPU_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
