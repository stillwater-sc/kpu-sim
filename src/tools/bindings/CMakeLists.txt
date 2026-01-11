# ============================================================================
# src/bindings/CMakeLists.txt
# Language bindings

add_subdirectory(c)

if(KPU_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

