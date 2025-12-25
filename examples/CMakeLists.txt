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

if(KPU_BUILD_PYTHON_BINDINGS)
    # Python examples are handled by the Python package
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python)
        install(DIRECTORY python/
            DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/python
            COMPONENT Examples
        )
    endif()
endif()

