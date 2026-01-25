# ============================================================================
# CMakeLists.txt - KPU Python Package Native Extension
#
# This CMakeLists.txt supports two build modes:
#
# 1. MONOREPO BUILD (from kpu-sim root)
#    - Parent CMakeLists.txt has already defined kpu_behavioral, etc.
#    - Just build the _native module and link
#
# 2. STANDALONE BUILD (pip install from python/ directory)
#    - Uses FetchContent to download kpu-sim if not available
#    - Builds all dependencies from source
#    - This is what PyPI wheel builds use
#
# Usage:
#   pip install .                    # Standalone build
#   pip install -e .                 # Editable install (standalone)
#   cmake --build --preset release   # Monorepo build (from kpu-sim root)
# ============================================================================

cmake_minimum_required(VERSION 3.20)

# Check if we're being built as part of the parent project
if(NOT DEFINED PROJECT_NAME OR PROJECT_NAME STREQUAL "")
    # Standalone build - define our own project
    project(kpu_python VERSION 0.8.0 LANGUAGES CXX)  # Native wheel infrastructure
    set(KPU_STANDALONE_BUILD ON)
else()
    set(KPU_STANDALONE_BUILD OFF)
endif()

# C++20 required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Find or fetch pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ============================================================================
# Standalone build: Fetch and build kpu-sim dependencies
# ============================================================================

if(KPU_STANDALONE_BUILD)
    message(STATUS "KPU Python: Standalone build mode")
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)

    # Check if we're in the kpu-sim monorepo
    set(KPU_SIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
    if(EXISTS "${KPU_SIM_ROOT}/include/sw/kpu/kpu_simulator.hpp")
        message(STATUS "KPU Python: Found kpu-sim in parent directory")
        set(KPU_INCLUDE_DIR "${KPU_SIM_ROOT}/include")
        set(KPU_SRC_DIR "${KPU_SIM_ROOT}/src")
        set(KPU_FROM_PARENT ON)
    else()
        message(STATUS "KPU Python: Fetching kpu-sim from GitHub (sources only)")
        FetchContent_Declare(
            kpu_sim
            GIT_REPOSITORY https://github.com/stillwater-sc/kpu-sim.git
            GIT_TAG main
            GIT_SHALLOW TRUE
        )
        # Only populate, don't run kpu_sim's CMakeLists.txt (it has packaging/binding deps)
        FetchContent_GetProperties(kpu_sim)
        if(NOT kpu_sim_POPULATED)
            FetchContent_Populate(kpu_sim)
        endif()
        set(KPU_INCLUDE_DIR "${kpu_sim_SOURCE_DIR}/include")
        set(KPU_SRC_DIR "${kpu_sim_SOURCE_DIR}/src")
        set(KPU_FROM_PARENT OFF)
    endif()

    # Fetch nlohmann/json (required for DFX parser)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)

    # Fetch Universal library (header-only, for quantization types)
    FetchContent_Declare(
        universal
        GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
        GIT_TAG v3.77
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(universal)
    if(NOT universal_POPULATED)
        FetchContent_Populate(universal)
    endif()
    set(UNIVERSAL_INCLUDE_DIR ${universal_SOURCE_DIR}/include)

    # Build kpu_behavioral library (minimal subset for Python bindings)
    add_library(kpu_behavioral_for_python STATIC
        ${KPU_SRC_DIR}/models/behavioral/compute/compute_fabric.cpp
    )
    target_include_directories(kpu_behavioral_for_python PUBLIC
        ${KPU_INCLUDE_DIR}
        ${UNIVERSAL_INCLUDE_DIR}
    )
    target_link_libraries(kpu_behavioral_for_python PUBLIC nlohmann_json::nlohmann_json)
    target_compile_features(kpu_behavioral_for_python PUBLIC cxx_std_20)

    # Build kpu_transactional library (minimal subset)
    add_library(kpu_transactional_for_python STATIC
        ${KPU_SRC_DIR}/models/transactional/compute/compute_fabric.cpp
    )
    target_include_directories(kpu_transactional_for_python PUBLIC
        ${KPU_INCLUDE_DIR}
        ${UNIVERSAL_INCLUDE_DIR}
    )
    target_compile_features(kpu_transactional_for_python PUBLIC cxx_std_20)
    target_link_libraries(kpu_transactional_for_python PUBLIC kpu_behavioral_for_python nlohmann_json::nlohmann_json)

    # Build kpu_dfx library (minimal subset)
    # Check if dfx sources exist
    if(EXISTS "${KPU_SRC_DIR}/dfx/dfx_executor.cpp")
        add_library(kpu_dfx_for_python STATIC
            ${KPU_SRC_DIR}/dfx/dfx_executor.cpp
        )
        target_include_directories(kpu_dfx_for_python PUBLIC
            ${KPU_INCLUDE_DIR}
            ${UNIVERSAL_INCLUDE_DIR}
        )
        target_compile_features(kpu_dfx_for_python PUBLIC cxx_std_20)
        target_link_libraries(kpu_dfx_for_python PUBLIC kpu_behavioral_for_python nlohmann_json::nlohmann_json)
        set(KPU_DFX_LIBRARY kpu_dfx_for_python)
    else()
        # DFX not available, create empty interface
        add_library(kpu_dfx_for_python INTERFACE)
        set(KPU_DFX_LIBRARY kpu_dfx_for_python)
    endif()

    # Store include dirs for _native module
    set(KPU_EXTRA_INCLUDES ${UNIVERSAL_INCLUDE_DIR})

    set(KPU_BEHAVIORAL_LIBRARY kpu_behavioral_for_python)
    set(KPU_TRANSACTIONAL_LIBRARY kpu_transactional_for_python)
else()
    # Monorepo build - use parent's libraries
    message(STATUS "KPU Python: Monorepo build mode")
    set(KPU_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
    set(KPU_BEHAVIORAL_LIBRARY kpu_behavioral)
    set(KPU_TRANSACTIONAL_LIBRARY kpu_transactional)
    set(KPU_DFX_LIBRARY kpu_dfx)
    set(KPU_EXTRA_INCLUDES "")
endif()

# ============================================================================
# Build the _native pybind11 module
# ============================================================================

pybind11_add_module(_native kpu/_native/kpu_native.cpp)

target_include_directories(_native PRIVATE
    ${KPU_INCLUDE_DIR}
    ${KPU_EXTRA_INCLUDES}
)

target_link_libraries(_native PRIVATE
    ${KPU_BEHAVIORAL_LIBRARY}
    ${KPU_TRANSACTIONAL_LIBRARY}
    ${KPU_DFX_LIBRARY}
)

target_compile_features(_native PRIVATE cxx_std_20)

# Version info
if(DEFINED KPU_VERSION)
    target_compile_definitions(_native PRIVATE VERSION_INFO="${KPU_VERSION}")
else()
    target_compile_definitions(_native PRIVATE VERSION_INFO="0.8.0")
endif()

# ============================================================================
# Installation (for scikit-build-core)
# ============================================================================

# Install the module - scikit-build-core handles the destination
install(TARGETS _native LIBRARY DESTINATION .)

message(STATUS "KPU Python: Native module will be installed to kpu/_native/")
