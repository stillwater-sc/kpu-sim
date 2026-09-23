# ============================================================================
# tools/kpu-run/expect_exit.cmake
# Run a command and assert its EXIT CODE, plus optional required/forbidden output.
#
# Why this exists: CTest's PASS_REGULAR_EXPRESSION makes CTest IGNORE the exit
# code, so an output-matching test cannot tell a refusal from a successful run
# that merely mentions the same words. kpu-run prints "not implemented (#283)" in
# its normal header on every run, so matching that string alone would pass even if
# the tool silently fell back to a level that works -- the exact failure the test
# exists to catch.
#
# Usage:
#   cmake -DCMD=<exe> -DARGS=a;b -DEXPECT_CODE=2
#         [-DMUST_MATCH=regex] [-DMUST_NOT_MATCH=regex] -P expect_exit.cmake
# ============================================================================

if(NOT DEFINED CMD OR NOT DEFINED EXPECT_CODE)
    message(FATAL_ERROR "expect_exit: CMD and EXPECT_CODE are required")
endif()

# ARGS arrives as a CMake LIST (the caller escapes separators with $<SEMICOLON>), so it
# expands straight into the command. separate_arguments(UNIX_COMMAND) would splice it into
# ONE argument -- which the tool then ignores, falls back to its defaults, and exits 0,
# making this wrapper report a pass for a command it never really ran.
execute_process(
    COMMAND "${CMD}" ${ARGS}
    RESULT_VARIABLE _code
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
)
set(_all "${_out}${_err}")

if(NOT _code EQUAL EXPECT_CODE)
    message(FATAL_ERROR
        "expect_exit: expected exit ${EXPECT_CODE}, got ${_code}\n--- output ---\n${_all}")
endif()

if(DEFINED MUST_MATCH AND NOT _all MATCHES "${MUST_MATCH}")
    message(FATAL_ERROR
        "expect_exit: output does not match '${MUST_MATCH}'\n--- output ---\n${_all}")
endif()

if(DEFINED MUST_NOT_MATCH AND _all MATCHES "${MUST_NOT_MATCH}")
    message(FATAL_ERROR
        "expect_exit: output matches the forbidden '${MUST_NOT_MATCH}', so the command "
        "did more than it should have\n--- output ---\n${_all}")
endif()
