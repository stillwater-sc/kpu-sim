# ============================================================================
# examples/CMakeLists.txt
# Example applications

# Add existing example directories
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/basic)
    add_subdirectory(basic)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compiler)
    add_subdirectory(compiler)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/advanced)
    add_subdirectory(advanced)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorials)
    add_subdirectory(tutorials)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kernel)
    add_subdirectory(kernel)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
    add_subdirectory(runtime)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/blas)
    add_subdirectory(blas)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/analysis)
    add_subdirectory(analysis)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/c)
    add_subdirectory(c)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mlp)
    add_subdirectory(mlp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/behavioral)
    add_subdirectory(behavioral)
endif()

# ============================================================================
# Standalone matmul runner (behavioral + transactional comparison)
# ============================================================================
add_executable(run_matmul run_matmul.cpp)
target_link_libraries(run_matmul PRIVATE
    kpu_isa
    kpu_dsl
    kpu_schedules
    kpu_memory_components
)

if(KPU_BUILD_PYTHON_BINDINGS)
    # Python examples are handled by the Python package
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python)
        install(DIRECTORY python/
            DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/python
            COMPONENT Examples
        )
    endif()
endif()

