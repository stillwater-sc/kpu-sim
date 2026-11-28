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
    # A level-2 matmul for a mesh (#608): the P1 schedule, as dcsp text.
    add_test(NAME csp_gen_distributed_matmul
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_dmm_2x2.json$<SEMICOLON>--distributed
            -DEXPECT_CODE=0
            -DMUST_MATCH=program.matmul.machine.mesh.2x2.*tensor.A.32,32..int8.*tensor.C.32,32..int32.*channel.a0to1.:.bm.E.0.
            -P ${_expect}
    )
    add_test(NAME csp_gen_distributed_needs_a_mesh
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--distributed
            -DEXPECT_CODE=1
            -DMUST_MATCH=not.a.mesh
            -P ${_expect}
    )
    # P5 (#609): a broadcast, to every node of the mesh.
    add_test(NAME csp_gen_distributed_broadcast
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>broadcast$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_dmm_2x2.json$<SEMICOLON>--distributed
            -DEXPECT_CODE=0
            -DMUST_MATCH=program.broadcast.machine.mesh.2x2.*vector.b.32..int32.*channel.fx3.:.bm.x.3.
            -P ${_expect}
    )
    # P6 (#610): Z = X + Y, both operands aligned.
    add_test(NAME csp_gen_distributed_add
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>add$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_dmm_2x2.json$<SEMICOLON>--distributed
            -DEXPECT_CODE=0
            -DMUST_MATCH=program.add.machine.mesh.2x2.*call.add.l2.0.:0,.l2.0.:2..to.Z.0,0.
            -P ${_expect}
    )
    # P3 (#611): the row sums, chained west and broadcast back.
    add_test(NAME csp_gen_distributed_rowsum
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>rowsum$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_dmm_2x2.json$<SEMICOLON>--distributed
            -DEXPECT_CODE=0
            -DMUST_MATCH=program.rowsum.machine.mesh.2x2.*vector.s.32..int32.*channel.p1to0.:.bm.W.1.
            -P ${_expect}
    )
    # P2 (#612): a 3 x 3 box filter, a halo exchange with four neighbours.
    add_test(NAME csp_gen_distributed_box3x3
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>box3x3$<SEMICOLON>--size$<SEMICOLON>16$<SEMICOLON>--tile$<SEMICOLON>8$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_dmm_2x2.json$<SEMICOLON>--distributed
            -DEXPECT_CODE=0
            -DMUST_MATCH=program.box3x3.machine.mesh.2x2.*call.box3h_e.*call.box3v_s
            -P ${_expect}
    )
    # box3x3's tile is the mesh's: another --tile is refused, not ignored (#643's review).
    add_test(NAME csp_gen_distributed_box3x3_tile
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--algo$<SEMICOLON>box3x3$<SEMICOLON>--size$<SEMICOLON>16$<SEMICOLON>--tile$<SEMICOLON>4$<SEMICOLON>--target$<SEMICOLON>${_deploy}/kpu_dmm_2x2.json$<SEMICOLON>--distributed
            -DEXPECT_CODE=1
            -DMUST_MATCH=one.tile.per.node:.on.this.2x2.mesh.a.16.x.16.X.is.in.8.x.8.tiles,.and.--tile.is.4
            -P ${_expect}
    )
    # The help says what --distributed's dtype default is (#638's review).
    add_test(NAME csp_gen_distributed_dtype_documented
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:csp-gen>
            -DARGS=--help
            -DEXPECT_CODE=0
            -DMUST_MATCH=default.fp32..with.--distributed,.int8
            -P ${_expect}
    )
    set_tests_properties(csp_gen_distributed_matmul csp_gen_distributed_needs_a_mesh csp_gen_distributed_dtype_documented
        csp_gen_distributed_broadcast csp_gen_distributed_add csp_gen_distributed_rowsum csp_gen_distributed_box3x3
        csp_gen_distributed_box3x3_tile
        PROPERTIES LABELS "program;csp;distributed;gen")
endif()
