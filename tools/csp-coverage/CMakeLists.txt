# ============================================================================
# tools/csp-coverage/CMakeLists.txt
# kpu-csp-coverage -- the distributed CSP coverage report's measured block (#617;
# docs/plans/distributed-csp-programs.md §3.5).
# ============================================================================

kpu_add_tool(csp-coverage
    FOLDER "Driver"
    DESCRIPTION "Measure every pattern class's level-2 program on the DMM meshes"
    SOURCES
        kpu_csp_coverage.cpp
)

if(KPU_BUILD_TESTS)
    # The report is measured: its block is what the generators' programs measure now.
    add_test(NAME csp_coverage_report_current
        COMMAND $<TARGET_FILE:csp-coverage>
            --deploy-dir ${CMAKE_SOURCE_DIR}/tests/program/deploy
            --check ${CMAKE_SOURCE_DIR}/docs/analysis/distributed-csp-coverage.md
    )
    set_tests_properties(csp_coverage_report_current PROPERTIES LABELS "program;csp;distributed")
endif()
