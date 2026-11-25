# ============================================================================
# tools/kpu-run/expect_file.cmake
# Assert that a file exists, or with -DABSENT=1 that it does not.
#
# Usage: cmake -DFILE=<path> [-DABSENT=1] -P expect_file.cmake
# ============================================================================
if(NOT DEFINED FILE)
    message(FATAL_ERROR "expect_file: FILE is required")
endif()
if(ABSENT)
    if(EXISTS "${FILE}")
        message(FATAL_ERROR "expect_file: ${FILE} exists, and must not")
    endif()
elseif(NOT EXISTS "${FILE}")
    message(FATAL_ERROR "expect_file: ${FILE} is missing")
endif()
