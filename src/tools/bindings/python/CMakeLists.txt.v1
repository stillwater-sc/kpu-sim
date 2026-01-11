# ============================================================================
# src/bindings/python/CMakeLists.txt
# Python bindings using pybind11

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(stillwater_kpu_native kpu_bindings.cpp)

target_link_libraries(stillwater_kpu_native PRIVATE
    StillwaterKPU::Simulator
    pybind11::headers
)

kpu_set_target_options(stillwater_kpu_native)

# Compiler-specific options for pybind11
target_compile_definitions(stillwater_kpu_native PRIVATE 
    VERSION_INFO="${PROJECT_VERSION}"
)

set_target_properties(stillwater_kpu_native PROPERTIES
    FOLDER "Bindings"
)

# Install Python module
install(TARGETS stillwater_kpu_native
    DESTINATION .
    COMPONENT Python
)

# Install Python package files
install(FILES __init__.py
    DESTINATION .
    COMPONENT Python
)

