# cmake/Version.cmake
# Git-based version management for KPU Simulator
#
# This module derives the project version from git tags.
# Usage: include(Version) after project() declaration
#
# Provides:
#   KPU_VERSION_MAJOR, KPU_VERSION_MINOR, KPU_VERSION_PATCH
#   KPU_VERSION_STRING (e.g., "0.3.0")
#   KPU_VERSION_FULL (e.g., "0.3.0-5-g1a2b3c4" for dev builds)
#   KPU_VERSION_IS_RELEASE (TRUE if exact tag match)

find_package(Git QUIET)

# Default version if git not available or no tags
set(KPU_VERSION_FALLBACK "0.4.0")

if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    # Get version from git describe
    # Format: v0.3.0 or v0.3.0-5-g1a2b3c4 (if commits after tag)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*" --always
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DESCRIBE_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_DESCRIBE_RESULT
    )

    if(GIT_DESCRIBE_RESULT EQUAL 0 AND GIT_DESCRIBE_OUTPUT MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)(-([0-9]+)-g([a-f0-9]+))?$")
        set(KPU_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(KPU_VERSION_MINOR ${CMAKE_MATCH_2})
        set(KPU_VERSION_PATCH ${CMAKE_MATCH_3})

        if(CMAKE_MATCH_5)
            # Development version: commits after tag
            set(KPU_VERSION_DEV_COMMITS ${CMAKE_MATCH_5})
            set(KPU_VERSION_GIT_HASH ${CMAKE_MATCH_6})
            set(KPU_VERSION_IS_RELEASE FALSE)
            set(KPU_VERSION_IS_RELEASE_CPP "false")
            set(KPU_VERSION_IS_RELEASE_PY "False")
            set(KPU_VERSION_FULL "${KPU_VERSION_MAJOR}.${KPU_VERSION_MINOR}.${KPU_VERSION_PATCH}-dev${KPU_VERSION_DEV_COMMITS}+${KPU_VERSION_GIT_HASH}")
        else()
            # Exact tag match - release version
            set(KPU_VERSION_IS_RELEASE TRUE)
            set(KPU_VERSION_IS_RELEASE_CPP "true")
            set(KPU_VERSION_IS_RELEASE_PY "True")
            set(KPU_VERSION_FULL "${KPU_VERSION_MAJOR}.${KPU_VERSION_MINOR}.${KPU_VERSION_PATCH}")
        endif()

        set(KPU_VERSION_STRING "${KPU_VERSION_MAJOR}.${KPU_VERSION_MINOR}.${KPU_VERSION_PATCH}")

        message(STATUS "KPU Version: ${KPU_VERSION_FULL} (from git)")
    else()
        # No valid tag found, use fallback
        message(STATUS "No valid git tag found, using fallback version")
        string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ ${KPU_VERSION_FALLBACK})
        set(KPU_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(KPU_VERSION_MINOR ${CMAKE_MATCH_2})
        set(KPU_VERSION_PATCH ${CMAKE_MATCH_3})
        set(KPU_VERSION_STRING ${KPU_VERSION_FALLBACK})
        set(KPU_VERSION_FULL "${KPU_VERSION_FALLBACK}-dev")
        set(KPU_VERSION_IS_RELEASE FALSE)
        set(KPU_VERSION_IS_RELEASE_CPP "false")
        set(KPU_VERSION_IS_RELEASE_PY "False")

        message(STATUS "KPU Version: ${KPU_VERSION_FULL} (fallback)")
    endif()
else()
    # Git not available, use fallback
    message(STATUS "Git not found, using fallback version")
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ ${KPU_VERSION_FALLBACK})
    set(KPU_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(KPU_VERSION_MINOR ${CMAKE_MATCH_2})
    set(KPU_VERSION_PATCH ${CMAKE_MATCH_3})
    set(KPU_VERSION_STRING ${KPU_VERSION_FALLBACK})
    set(KPU_VERSION_FULL "${KPU_VERSION_FALLBACK}-dev")
    set(KPU_VERSION_IS_RELEASE FALSE)
    set(KPU_VERSION_IS_RELEASE_CPP "false")
    set(KPU_VERSION_IS_RELEASE_PY "False")

    message(STATUS "KPU Version: ${KPU_VERSION_FULL} (fallback)")
endif()

# Generate version header for C++ code
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/version.hpp.in"
    "${CMAKE_BINARY_DIR}/generated/sw/kpu/version.hpp"
    @ONLY
)

# Generate version file for Python
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/_version.py.in"
    "${CMAKE_BINARY_DIR}/generated/kpu/_version.py"
    @ONLY
)

# Make generated headers available
include_directories(${CMAKE_BINARY_DIR}/generated)

# Export version info for parent CMakeLists.txt
set(PROJECT_VERSION ${KPU_VERSION_STRING})
set(PROJECT_VERSION_MAJOR ${KPU_VERSION_MAJOR})
set(PROJECT_VERSION_MINOR ${KPU_VERSION_MINOR})
set(PROJECT_VERSION_PATCH ${KPU_VERSION_PATCH})
