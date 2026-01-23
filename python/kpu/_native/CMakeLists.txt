# ============================================================================
# python/kpu/_native/CMakeLists.txt
# Native pybind11 bindings for the kpu Python package
#
# This module provides the native backend for @kpu.compile decorator,
# enabling DFX program execution with optimized C++ code.
# ============================================================================

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Check if pybind11 is available (should be fetched by Dependencies.cmake)
if(NOT TARGET pybind11::module)
    message(WARNING "pybind11 not found, skipping kpu._native module")
    return()
endif()

# Build the _native module
pybind11_add_module(_native kpu_native.cpp)

# Link against pybind11 headers and KPU simulator libraries
# Note: XUE headers are header-only (constexpr/inline), no library needed
target_link_libraries(_native PRIVATE
    pybind11::headers
    kpu_behavioral
    kpu_transactional
    kpu_dfx
)

# Include simulator headers
target_include_directories(_native PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set C++20 for consistency with main project
target_compile_features(_native PRIVATE cxx_std_20)

set_target_properties(_native PROPERTIES
    FOLDER "Python/kpu"
    # Output directly to the Python package directory for development
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    # For multi-config generators
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Version info
target_compile_definitions(_native PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)

# Install target
install(TARGETS _native
    LIBRARY DESTINATION lib/python/kpu/_native
    COMPONENT Python
)

message(STATUS "Building kpu._native Python bindings")
