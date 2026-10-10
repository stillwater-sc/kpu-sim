# ============================================================================
# tools/csp-gen/CMakeLists.txt
# kpu-csp-gen -- write CSP programs (docs/plans/kpu-run-csp-programs.md step 1). The simulator,
# kpu-run, executes them; producing them is this tool's job.
# ============================================================================

kpu_add_tool(csp-gen
    FOLDER "Driver"
    DESCRIPTION "Write an operator's canonical schedule as a CSP program"
    SOURCES
        csp_gen.cpp
)

if(KPU_BUILD_TESTS)
    set(_expect ${CMAKE_SOURCE_DIR}/tools/kpu-run/expect_exit.cmake)
    set(_deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy)

    # A program is written, with its loops.
    add_test(NAME csp_gen_matmul
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>256$<SEMICOLON>--tile$<SEMICOLON>32$<SEMICOLON>--l3$<SEMICOLON>128
            -DEXPECT_CODE=0
            -DMUST_MATCH=program.matmul.machine.flat.l3.=.128.*for.j.in.0..8
            -P ${_expect}
    )
    # The machine's L3 from a deployment.
    add_test(NAME csp_gen_lu_from_target
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>lu$<SEMICOLON>--size$<SEMICOLON>64$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_s1.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=program.lu.machine.flat.l3.=.128.*call.getrf
            -P ${_expect}
    )
    # A schedule that does not fit is refused with its slot count, and nothing is written.
    add_test(NAME csp_gen_refuses_small_l3
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>256$<SEMICOLON>--tile$<SEMICOLON>32$<SEMICOLON>--l3$<SEMICOLON>16
            -DEXPECT_CODE=1
            -DMUST_MATCH=needs.17.L3.slots.*the.L3.holds.16
            -DMUST_NOT_MATCH=program.matmul
            -P ${_expect}
    )
    # A stage the target cannot run: S1 declares no vector units.
    add_test(NAME csp_gen_refuses_misplaced_stage
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>linear$<SEMICOLON>--size$<SEMICOLON>128$<SEMICOLON>--tile$<SEMICOLON>32$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_s1.json$<SEMICOLON>--act$<SEMICOLON>atan$<SEMICOLON>--place$<SEMICOLON>bm.egress
            -DEXPECT_CODE=1
            -DMUST_MATCH=add.@.bm.egress:.the.target.s.BlockMover.has.no.vector.unit
            -P ${_expect}
    )
    add_test(NAME csp_gen_rejects_bad_option
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>64$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--l3$<SEMICOLON>64$<SEMICOLON>--act$<SEMICOLON>relu
            -DEXPECT_CODE=2
            -DMUST_MATCH=--act.and.--place.are.the.linear.operator.s
            -P ${_expect}
    )
    # An option followed by another option has no value: refused, nothing written.
    add_test(NAME csp_gen_rejects_missing_value
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>64$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--l3$<SEMICOLON>64$<SEMICOLON>-o$<SEMICOLON>--help
            -DEXPECT_CODE=2
            -DMUST_MATCH=-o:.missing.value
            -P ${_expect}
    )
    set_tests_properties(csp_gen_matmul csp_gen_lu_from_target csp_gen_refuses_small_l3
        csp_gen_refuses_misplaced_stage csp_gen_rejects_bad_option csp_gen_rejects_missing_value
        PROPERTIES LABELS "program;csp;gen;v0.9")
endif()
