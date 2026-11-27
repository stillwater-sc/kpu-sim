# ============================================================================
# tools/csp-check/CMakeLists.txt
# kpu-csp-check -- a CSP program against the machine it is written for (#607;
# docs/plans/distributed-csp-programs.md §4 step 4).
# ============================================================================

kpu_add_tool(csp-check
    FOLDER "Driver"
    DESCRIPTION "Check a level-2 CSP program against its deployment (D0-D4, and D5 with --run)"
    SOURCES
        kpu_csp_check.cpp
)

if(KPU_BUILD_TESTS)
    set(_expect ${CMAKE_SOURCE_DIR}/tools/kpu-run/expect_exit.cmake)
    set(_deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy)
    set(_dcsp ${CMAKE_SOURCE_DIR}/tests/program/csp/distributed)

    # Each sample passes D0-D4 on its mesh, and runs (D5).
    foreach(_case "vector_to_node3;kpu_dmm_2x2" "gemm_one_tile_node1;kpu_dmm_2x2" "gemm_ksplit_node0;kpu_dmm_1x1")
        list(GET _case 0 _prog)
        list(GET _case 1 _dep)
        add_test(NAME csp_check_${_prog}
            COMMAND ${CMAKE_COMMAND}
                -DCMD=$<TARGET_FILE:csp-check>
                -DARGS=--distributed$<SEMICOLON>${_dcsp}/${_prog}.dcsp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/${_dep}.json$<SEMICOLON>--run
                -DEXPECT_CODE=0
                -DMUST_MATCH=D0-D4.pass.*D5.pass
                -P ${_expect}
        )
    endforeach()
    # A program on the wrong mesh is a violation (exit 1), not a crash.
    add_test(NAME csp_check_wrong_mesh
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-check>
            -DARGS=--distributed$<SEMICOLON>${_dcsp}/vector_to_node3.dcsp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_dmm_4x4.json
            -DEXPECT_CODE=1
            -DMUST_MATCH=D0.*written.for.a.2x2.mesh
            -P ${_expect}
    )
    # A checkerboard is not a mesh.
    add_test(NAME csp_check_not_a_mesh
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-check>
            -DARGS=--distributed$<SEMICOLON>${_dcsp}/vector_to_node3.dcsp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_t4.json
            -DEXPECT_CODE=1
            -DMUST_MATCH=is.not.a.dmm_mesh
            -P ${_expect}
    )
    set_tests_properties(csp_check_vector_to_node3 csp_check_gemm_one_tile_node1 csp_check_gemm_ksplit_node0
        csp_check_wrong_mesh csp_check_not_a_mesh PROPERTIES LABELS "program;csp;distributed")
endif()
