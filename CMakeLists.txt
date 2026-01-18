cmake_minimum_required(VERSION 4.2)

# Suppress deprecation warnings from external dependencies globally
# This must be set before any external dependencies are processed
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# Set modern CMake policies
cmake_policy(SET CMP0148 NEW)  # Use FindPython instead of FindPythonInterp/FindPythonLibs
# Note: Deprecation warnings from external dependencies are suppressed globally above.

# Ignore improper ASM tool
cmake_policy(SET CMP0194 NEW)
# if build fails, there is an ASM dependency

project(kpu_simulator
    DESCRIPTION "Stillwater Knowledge Processing Unit Simulator"
    HOMEPAGE_URL "https://github.com/stillwater-sc/kpu-simulator"
    LANGUAGES CXX C
)

# Enable folder organization in IDEs (Visual Studio, etc.)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Position Independent Code by default for all targets
# This is required when static libraries are linked into shared libraries (like Python modules)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Version management - derives version from git tags
# Creates: KPU_VERSION_MAJOR, KPU_VERSION_MINOR, KPU_VERSION_PATCH, KPU_VERSION_STRING
# Also generates: build/generated/sw/kpu/version.hpp and build/generated/kpu/_version.py
include(Version)

include(CompilerOptions)

# Include the custom script to gather system info during configuration
include(SystemInfoGatherer)

# Project options - MUST come before Dependencies.cmake
option(KPU_BUILD_TESTS            "Build test suite"              ON)
option(KPU_BUILD_EXAMPLES         "Build examples"                ON)
option(KPU_BUILD_TOOLS            "Build development tools"       ON)
option(KPU_BUILD_PYTHON_BINDINGS  "Build Python bindings"         ON)
option(KPU_BUILD_BENCHMARKS       "Build benchmark suite"         ON)
option(KPU_BUILD_MODELS           "Build architecture models"     ON)
option(KPU_BUILD_DOCS             "Build documentation"           OFF)
option(KPU_ENABLE_OPENMP          "Enable OpenMP parallelization" ON)
option(KPU_ENABLE_CUDA            "Enable CUDA support"           OFF)
option(KPU_ENABLE_OPENCL          "Enable OpenCL support"         OFF)
option(KPU_ENABLE_PROFILING       "Enable profiling support"      OFF)
option(KPU_ENABLE_SANITIZERS      "Enable sanitizers"             OFF)
option(KPU_STATIC_ANALYSIS        "Enable static analysis"        OFF)

# Include dependencies AFTER options are defined
include(Dependencies)
include(DomainFlowIntegration)  # Domain flow IR integration
include(Documentation)
include(Testing)
include(Packaging)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable testing
enable_testing()
include(CTest)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Remove problematic default flags
    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE 
                     CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REGEX REPLACE "/RTC[^ ]*" "" ${flag_var} "${${flag_var}}")
        string(REGEX REPLACE "/W[0-4]" "" ${flag_var} "${${flag_var}}")
    endforeach()
    
    # Set our own flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Od /D_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /O2 /Zi /DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "/MD /O1 /DNDEBUG")
    
    # Define for DLL export/import
    add_definitions(-DBUILDING_KPU_SIMULATOR)
    
    # Suppress specific warnings
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/wd4251 /wd4275 /wd4996)
endif()

# Python bindings - check if pybind11 was fetched by Dependencies.cmake
if(KPU_BUILD_PYTHON_BINDINGS)
    # Check for various possible pybind11 target names
    set(PYBIND11_AVAILABLE FALSE)
    if(TARGET pybind11::module OR TARGET pybind11::pybind11 OR TARGET pybind11)
        set(PYBIND11_AVAILABLE TRUE)
    endif()
    
    if(PYBIND11_AVAILABLE AND EXISTS "${CMAKE_SOURCE_DIR}/src/tools/bindings/python/kpu_bindings.cpp")
        message(STATUS "Python bindings enabled - will be built by src/bindings/")
    else()
        if(NOT PYBIND11_AVAILABLE)
            message(STATUS "pybind11 not available - Python bindings disabled")
        else()
            message(STATUS "Python bindings source not found at src/tools/bindings/python/kpu_bindings.cpp")
            message(STATUS "Python bindings disabled")
        endif()
    endif()
else()
    message(STATUS "Python bindings disabled by user option")
endif()

# Function to set common target options
function(kpu_set_target_options target_name)
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W3)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# Note: kpu_add_test() function is defined in cmake/Testing.cmake

# Global include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add subdirectory builds - each manages its own sources and targets
add_subdirectory(src)

# Only build tests if enabled
if(KPU_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Only build models if enabled
if(KPU_BUILD_MODELS)
    add_subdirectory(models)
endif()

# Only build examples if enabled
if(KPU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Only build tools if enabled
if(KPU_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# Domain Flow Pattern tests (build if directory exists - optional/experimental)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patterns")
    add_subdirectory(patterns)
endif()

# High-level Python package (kpu) with native bindings
if(KPU_BUILD_PYTHON_BINDINGS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/kpu/_native/kpu_native.cpp")
    if(PYBIND11_AVAILABLE)
        message(STATUS "Building kpu Python package native bindings")
        add_subdirectory(python/kpu/_native)
    else()
        message(STATUS "pybind11 not available - kpu Python package will use pure Python mode")
    endif()
endif()

# Installation of headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

## DISABLE FOR NOW - Exporting targets causes dependency chain issues

# # Export targets for use by other projects
# install(EXPORT kpu_simulator_targets
#     FILE kpu_simulator_targets.cmake
#     NAMESPACE kpu::
#     DESTINATION lib/cmake/kpu_simulator
# )

# # Create config file for find_package support
# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#     kpu_simulator_config_version.cmake
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY AnyNewerVersion
# )

# configure_package_config_file(
#     cmake/kpu_simulator_config.cmake.in
#     kpu_simulator_config.cmake
#     INSTALL_DESTINATION lib/cmake/kpu_simulator
# )

# install(FILES
#     ${CMAKE_CURRENT_BINARY_DIR}/kpu_simulator_config.cmake
#     ${CMAKE_CURRENT_BINARY_DIR}/kpu_simulator_config_version.cmake
#     DESTINATION lib/cmake/kpu_simulator
# )

# Print comprehensive build summary
message(STATUS "")
message(STATUS "========== KPU Simulator Build Configuration ==========")
message(STATUS "Project          : ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "CMake version    : ${CMAKE_VERSION}")
message(STATUS "Source dir       : ${CMAKE_SOURCE_DIR}")
message(STATUS "Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard     : C++${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler         : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Generator        : ${CMAKE_GENERATOR}")
message(STATUS "System           : ${CMAKE_SYSTEM}")
message(STATUS "Processor        : ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "")
message(STATUS "Install prefix   : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Testing          : ${BUILD_TESTING}")
message(STATUS "Python bindings  : ${KPU_BUILD_PYTHON_BINDINGS}")
if(TARGET stillwater_kpu)
    message(STATUS "Python KPU       : ENABLED")
else()
    message(STATUS "Python KPU       : DISABLED")
endif()
if(MSVC)
    message(STATUS "MSVC Version     : ${MSVC_VERSION}")
endif()
message(STATUS "CXX Flags        : ${CMAKE_CXX_FLAGS}")
message(STATUS "Build Type Flags : ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

message(STATUS "======================================================")
message(STATUS "")

# Helpful build instructions
message(STATUS "To build:")
message(STATUS "  # Enable Python bindings (default)")
message(STATUS "  cmake -DKPU_BUILD_PYTHON_BINDINGS=ON ..")
message(STATUS "  # Or, if pybind11 is not available, disable Python bindings :")
message(STATUS " cmake -DKPU_BUILD_PYTHON_BINDINGS=OFF ..")
message(STATUS "")
# Re-enable deprecation warnings for project code after dependencies are processed
set(CMAKE_WARN_DEPRECATED ON CACHE BOOL "" FORCE)

message(STATUS "To run tests:")
message(STATUS "  .\\${CMAKE_BUILD_TYPE}\\kpu_test.exe")
message(STATUS "")
