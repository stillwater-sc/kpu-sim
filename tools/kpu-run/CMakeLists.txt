# ============================================================================
# tools/kpu-run/CMakeLists.txt
# kpu-run — execute a Domain Flow Program and compare the simulation models.
# ============================================================================

kpu_add_tool(run
    FOLDER "Driver"
    DESCRIPTION "Execute a Domain Flow Program at one or more levels and compare them"
    SOURCES
        kpu_run.cpp
)

# NOTE: the guard is KPU_BUILD_TESTS, not BUILD_TESTING. BUILD_TESTING is empty in this
# project (it never includes CTest), so a `if(BUILD_TESTING)` block silently registers
# nothing — which is how examples/characterize's smoke test went unregistered.
if(KPU_BUILD_TESTS)
    # Agreement between the levels is the point of the tool, so the smoke test is
    # the comparison itself: a non-zero exit means the models disagree.
    add_test(NAME kpu_run_matmul_compare
        COMMAND run --algo matmul --size 48 --tile 16 --level all
    )
    set_tests_properties(kpu_run_matmul_compare PROPERTIES LABELS "program;driver")

    # LU exercises the pivot permutation and swap count, which a value diff alone
    # would not catch.
    add_test(NAME kpu_run_lu_compare
        COMMAND run --algo lu --size 64 --tile 16 --level all
    )
    set_tests_properties(kpu_run_lu_compare PROPERTIES LABELS "program;driver")

    # A level with no interpreter must be refused, NOT silently substituted -- and the
    # assertion has to be able to tell those apart. PASS_REGULAR_EXPRESSION cannot: it
    # makes CTest ignore the exit code, and the string "not implemented (#283)" appears in
    # the tool's NORMAL header on every run, so a regression that fell back to behavioral
    # would still match. The wrapper asserts exit code 2 and forbids the run header.
    add_test(NAME kpu_run_refuses_unimplemented_level
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--level$<SEMICOLON>resource-transactional
            -DEXPECT_CODE=2
            -DMUST_MATCH=kpu-run:.resource-transactional:.*not.implemented..#283.
            -DMUST_NOT_MATCH=values.vs
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_refuses_unimplemented_level PROPERTIES
        LABELS "program;driver")

    # --timeline must write a trace the existing tooling can read, and must REFUSE when
    # only L-B was asked for, since L-B models no intervals to write.
    add_test(NAME kpu_run_timeline_written
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--timeline$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/kpu_run_timeline.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=events..one.per.hop
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_timeline_written PROPERTIES LABELS "program;driver")

    add_test(NAME kpu_run_timeline_refused_without_timing
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--level$<SEMICOLON>behavioral$<SEMICOLON>--timeline$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/should_not_exist.json
            -DEXPECT_CODE=2
            -DMUST_MATCH=needs.a.level.that.models.resources
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_timeline_refused_without_timing PROPERTIES
        LABELS "program;driver")

    # An unknown dataflow, and a dataflow on an algorithm that has no stream derivation,
    # must both be refused rather than silently ignored.
    add_test(NAME kpu_run_rejects_bad_streams
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--streams$<SEMICOLON>sideways
            -DEXPECT_CODE=2
            -DMUST_MATCH=unknown.--streams
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_rejects_bad_streams PROPERTIES LABELS "program;driver")

    add_test(NAME kpu_run_rejects_streams_for_lu
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--algo$<SEMICOLON>lu$<SEMICOLON>--streams$<SEMICOLON>os
            -DEXPECT_CODE=2
            -DMUST_MATCH=matmul.only
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_rejects_streams_for_lu PROPERTIES LABELS "program;driver")

    # An option present without its value must be an ERROR, not a silent absence: a
    # terminal --timeline previously read as "no timeline requested".
    add_test(NAME kpu_run_rejects_valueless_option
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--timeline
            -DEXPECT_CODE=2
            -DMUST_MATCH=--timeline:.missing.value
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_rejects_valueless_option PROPERTIES LABELS "program;driver")

    # --step must produce a readable transaction stream at each level, and the two levels
    # step by different mechanisms: L-T1 replays the run, L-B re-executes.
    add_test(NAME kpu_run_step_block_sequential
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--step$<SEMICOLON>--step-limit$<SEMICOLON>8
            -DEXPECT_CODE=0
            -DMUST_MATCH=hop-start.*dma:dram->l3
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_step_block_sequential PROPERTIES LABELS "program;driver")

    add_test(NAME kpu_run_step_behavioral
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--level$<SEMICOLON>behavioral$<SEMICOLON>--step$<SEMICOLON>--step-limit$<SEMICOLON>4
            -DEXPECT_CODE=0
            -DMUST_MATCH=re-executed.one.op.at.a.time
            -DMUST_NOT_MATCH=hop-start
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_step_behavioral PROPERTIES LABELS "program;driver")

    # Bad numeric input must also exit 2 rather than abort: an uncaught std::stoul throw
    # is SIGABRT, which CI cannot distinguish from a crash in the model.
    add_test(NAME kpu_run_rejects_bad_number
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--size$<SEMICOLON>abc
            -DEXPECT_CODE=2
            -DMUST_MATCH=--size
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_rejects_bad_number PROPERTIES
        LABELS "program;driver")
endif()
