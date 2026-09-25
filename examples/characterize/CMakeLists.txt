# ============================================================================
# examples/characterize/CMakeLists.txt
# Tile-program characterization / design-of-experiments harness
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
# ============================================================================

add_executable(tile_characterize tile_characterize.cpp)
target_link_libraries(tile_characterize PRIVATE kpu_simulator)
set_target_properties(tile_characterize PROPERTIES FOLDER "Examples/Characterize")

# Smoke test: a tiny sweep must run and functionally validate (max_err small).
# KPU_BUILD_TESTS, not BUILD_TESTING: the latter is empty in this project, so this test
# had never actually registered. A test that never runs is worse than no test, because it
# reads as coverage.
if(KPU_BUILD_TESTS)
    add_test(NAME tile_characterize_smoke
        COMMAND tile_characterize --algo lu --sizes 32 --tiles 8,16 --compute-tiles 1,4
    )
    set_tests_properties(tile_characterize_smoke PROPERTIES LABELS "program;characterize")

    # The machine as a spec (#282 increment 4): with --deploy, the sweep's machine axes come
    # FROM THE SPEC. Leaving them at their flag defaults swept "single, 1/4/16" for a spec
    # that said "checkerboard, 16" -- the machine described and the machine measured being
    # different, without a word. The assertion is that the spec's own machine appears.
    add_test(NAME tile_characterize_deploy
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:tile_characterize>
            -DARGS=--sizes$<SEMICOLON>32$<SEMICOLON>--tiles$<SEMICOLON>16$<SEMICOLON>--deploy$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/deploy/adr_checkerboard16.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=checkerboard
            -DMUST_NOT_MATCH=single
            -P ${CMAKE_SOURCE_DIR}/tools/kpu-run/expect_exit.cmake
    )
    set_tests_properties(tile_characterize_deploy PROPERTIES LABELS "program;characterize")

    # Bad numeric input must exit 2 rather than abort. These were bare std::stod/std::stoul:
    # "abc" threw uncaught (SIGABRT, indistinguishable in CI from a crash in the model) and
    # "-2" wrapped to an enormous sweep count.
    add_test(NAME tile_characterize_rejects_bad_numbers
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:tile_characterize>
            -DARGS=--sizes$<SEMICOLON>-2
            -DEXPECT_CODE=2
            -DMUST_MATCH=not.a.non-negative.integer
            -P ${CMAKE_SOURCE_DIR}/tools/kpu-run/expect_exit.cmake
    )
    set_tests_properties(tile_characterize_rejects_bad_numbers PROPERTIES
        LABELS "program;characterize")
endif()
