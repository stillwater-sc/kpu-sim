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

    # A level with no interpreter must be refused, not silently substituted.
    add_test(NAME kpu_run_refuses_unimplemented_level
        COMMAND run --level resource-transactional
    )
    set_tests_properties(kpu_run_refuses_unimplemented_level PROPERTIES
        LABELS "program;driver"
        PASS_REGULAR_EXPRESSION "not implemented \\(#283\\)"
        WILL_FAIL FALSE)
endif()
