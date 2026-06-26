# cmake/Dependencies.cmake
# External dependency management

include(FetchContent)

# Set common FetchContent properties
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Function to add external dependency
function(kpu_add_dependency name)
    set(options REQUIRED OPTIONAL)
    set(oneValueArgs GIT_REPOSITORY GIT_TAG CMAKE_ARGS)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments(DEP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    if(NOT TARGET ${name})
        message(STATUS "Adding dependency: ${name}")
        
        FetchContent_Declare(${name}
            GIT_REPOSITORY ${DEP_GIT_REPOSITORY}
            GIT_TAG ${DEP_GIT_TAG}
            ${DEP_CMAKE_ARGS}
        )
        
        FetchContent_MakeAvailable(${name})
        
        # Set folder for IDE organization
        if(DEP_TARGETS)
            foreach(target ${DEP_TARGETS})
                if(TARGET ${target})
                    set_target_properties(${target} PROPERTIES
                        FOLDER "Third Party/${name}"
                    )
                endif()
            endforeach()
        endif()
    endif()
endfunction()

# Define common dependencies
if(KPU_BUILD_TESTS)
    kpu_add_dependency(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        TARGETS Catch2 Catch2WithMain
    )
endif()

kpu_add_dependency(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
    TARGETS spdlog spdlog_header_only
)
# Note: v1.15.3 bundles fmt v11, which no longer references
# stdext::checked_array_iterator. Earlier versions (1.12 bundled fmt v9)
# required _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING on MSVC, and
# eventually broke entirely on MSVC 14.51+ where the symbol was removed.

kpu_add_dependency(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3  # Latest stable version
    TARGETS nlohmann_json
)

kpu_add_dependency(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1
    TARGETS fmt fmt-header-only
)

if(KPU_BUILD_PYTHON_BINDINGS)
    # Suppress pybind11's FindPython policy warnings by temporarily setting policy
    if(POLICY CMP0148)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0148 OLD)  # Allow pybind11 to use old FindPython modules
    endif()

    kpu_add_dependency(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6  # Latest stable with improved CMake support
        TARGETS pybind11 pybind11_headers
    )

    # Restore policy
    if(POLICY CMP0148)
        cmake_policy(POP)
    endif()
endif()

# Universal library for arbitrary precision number types
# Provides: bfloat16, half, cfloat (for FP8/FP4 variants), integer
# This is a header-only library - we only need the include directory
# SOURCE_SUBDIR points to non-existent dir to skip CMakeLists.txt processing

FetchContent_Declare(universal
    GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
    GIT_TAG v3.91
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR _skip_cmake  # Non-existent subdir prevents CMakeLists.txt processing
)

FetchContent_MakeAvailable(universal)

# Set include path for targets that need Universal types
# Universal v3.91+ moved headers under include/sw/, so use that as the include root
# This allows old-style includes like <universal/number/cfloat/cfloat.hpp> to work
set(UNIVERSAL_INCLUDE_DIR ${universal_SOURCE_DIR}/include/sw CACHE PATH "Universal library include directory")
message(STATUS "Universal library: ${UNIVERSAL_INCLUDE_DIR}")

