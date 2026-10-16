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
    # kpu-run is the simulator: it runs a program and does not write one
    # (kpu-run-csp-programs step 5). The derivation flags are refused by name, with where their
    # job went, and a run needs a program. The comparison between levels is the CSP tests'
    # below, on programs kpu-csp-gen writes.
    foreach(_flag algo size tile act)
        add_test(NAME kpu_run_refuses_${_flag}
            COMMAND ${CMAKE_COMMAND}
                -DCMD=$<TARGET_FILE:run>
                -DARGS=--${_flag}$<SEMICOLON>x
                -DEXPECT_CODE=2
                -DMUST_MATCH=--${_flag}.left.kpu-run.*kpu-csp-gen.*kpu-run.--program
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
        )
        set_tests_properties(kpu_run_refuses_${_flag} PROPERTIES LABELS "program;driver")
    endforeach()
    add_test(NAME kpu_run_needs_a_program
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--level$<SEMICOLON>behavioral
            -DEXPECT_CODE=2
            -DMUST_MATCH=--program.is.required
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_needs_a_program PROPERTIES LABELS "program;driver")

    # A level with no interpreter must be refused, NOT silently substituted -- and the
    # assertion has to be able to tell those apart. PASS_REGULAR_EXPRESSION cannot: it
    # makes CTest ignore the exit code, and the string "not implemented (#283)" appears in
    # the tool's NORMAL header on every run, so a regression that fell back to behavioral
    # would still match. The wrapper asserts exit code 2 and forbids the run header.
    add_test(NAME kpu_run_refuses_unimplemented_level
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.csp$<SEMICOLON>--level$<SEMICOLON>resource-transactional
            -DEXPECT_CODE=2
            -DMUST_MATCH=kpu-run:.resource-transactional:.*not.implemented..#283.
            -DMUST_NOT_MATCH=values.vs
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_refuses_unimplemented_level PROPERTIES
        LABELS "program;driver")

    # The tile-flow checker (#286 step 3) over real bundles, and its self-test: every
    # invariant must fail on a bundle that breaks it. Skipped without a Python interpreter.
    find_program(KPU_PYTHON3 NAMES python3 python)
    if(KPU_PYTHON3)
        set(_tflow_check ${CMAKE_SOURCE_DIR}/tools/trace/tflow_check.py)
        # Version 3, an L0 run's record, is still read: the checker, its self-test and the
        # viewer run on a v3 bundle kept as a fixture (tests/program/tflow/README.md), since
        # the L0 L-T1 executor that wrote it retired (kpu-run-csp-programs step 4e). It is the
        # #279 program: two readers of one tile, the higher-indexed one finishing first, clean
        # at L3 = 5, and the checker's TF9 is what fails it under the old release rule.
        set(_v3_bundle ${CMAKE_SOURCE_DIR}/tests/program/tflow/shared_reader_v3.tflow)
        add_test(NAME tflow_check_shared_reader
            COMMAND ${KPU_PYTHON3} ${_tflow_check} ${_v3_bundle})
        add_test(NAME tflow_check_selftest
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/trace/test_tflow_check.py ${_v3_bundle})
        # The viewer (#286 step 4): pack a real bundle, refuse a non-bundle, parse the script.
        add_test(NAME tileflow_viewer_smoke
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/tileflow/test_viewer.py
                    ${_v3_bundle})

        # The derived binding (bind.py) at scale: a T64 run of a program kpu-csp-gen writes,
        # and its floorplan (written by kpu_floorplan_t64 in tools/floorplan).
        add_test(NAME csp_gen_t64_matmul
            COMMAND $<TARGET_FILE:csp-gen> --algo matmul --size 128 --tile 32
                    --target ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_t64.json
                    -o ${CMAKE_CURRENT_BINARY_DIR}/t64_matmul.csp)
        add_test(NAME kpu_run_csp_tflow_t64
            COMMAND run --program ${CMAKE_CURRENT_BINARY_DIR}/t64_matmul.csp
                    --deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_t64.json
                    --level block-sequential --tflow ${CMAKE_CURRENT_BINARY_DIR}/csp_t64.tflow)
        set_tests_properties(kpu_run_csp_tflow_t64 PROPERTIES DEPENDS csp_gen_t64_matmul)
        add_test(NAME tileflow_bind_t64
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/tileflow/test_bind.py
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_t64.tflow
                    ${CMAKE_BINARY_DIR}/tools/floorplan/t64_floorplan.json)
        set_tests_properties(tileflow_bind_t64 PROPERTIES DEPENDS "kpu_run_csp_tflow_t64;kpu_floorplan_t64")
        # ---- version 4: a CSP program's record (docs/plans/kpu-run-csp-programs.md step 4b) ----
        # The T4 reference run, from a program kpu-csp-gen writes: 512^3 in 64x64 tiles.
        add_test(NAME csp_gen_t4_matmul
            COMMAND $<TARGET_FILE:csp-gen> --algo matmul --size 512 --tile 64
                    --target ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_t4.json
                    -o ${CMAKE_CURRENT_BINARY_DIR}/t4_matmul.csp)
        add_test(NAME kpu_run_csp_tflow_t4
            COMMAND run --program ${CMAKE_CURRENT_BINARY_DIR}/t4_matmul.csp
                    --deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_t4.json
                    --level block-sequential --tflow ${CMAKE_CURRENT_BINARY_DIR}/csp_t4.tflow)
        set_tests_properties(kpu_run_csp_tflow_t4 PROPERTIES DEPENDS csp_gen_t4_matmul)
        add_test(NAME kpu_run_csp_tflow_linear
            COMMAND run --program ${CMAKE_SOURCE_DIR}/tests/program/corpus/linear_64_t16_relu.csp
                    --inputs ${CMAKE_SOURCE_DIR}/tests/program/corpus/linear_64_t16_relu.l0
                    --deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_s1.json
                    --tflow ${CMAKE_CURRENT_BINARY_DIR}/csp_linear.tflow)
        add_test(NAME tflow_check_csp_t4
            COMMAND ${KPU_PYTHON3} ${_tflow_check} ${CMAKE_CURRENT_BINARY_DIR}/csp_t4.tflow)
        set_tests_properties(tflow_check_csp_t4 PROPERTIES DEPENDS kpu_run_csp_tflow_t4)
        add_test(NAME tflow_check_csp_linear
            COMMAND ${KPU_PYTHON3} ${_tflow_check} ${CMAKE_CURRENT_BINARY_DIR}/csp_linear.tflow)
        set_tests_properties(tflow_check_csp_linear PROPERTIES DEPENDS kpu_run_csp_tflow_linear)
        add_test(NAME tflow_check_v4_selftest
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/trace/test_tflow_check_v4.py
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_linear.tflow)
        set_tests_properties(tflow_check_v4_selftest PROPERTIES DEPENDS kpu_run_csp_tflow_linear)
        add_test(NAME tileflow_viewer_csp_smoke
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/tileflow/test_viewer.py
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_linear.tflow)
        set_tests_properties(tileflow_viewer_csp_smoke PROPERTIES DEPENDS kpu_run_csp_tflow_linear)
        add_test(NAME tileflow_bind_csp_t4
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/tileflow/test_bind.py
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_t4.tflow
                    ${CMAKE_BINARY_DIR}/tools/floorplan/t4_floorplan.json)
        set_tests_properties(tileflow_bind_csp_t4 PROPERTIES DEPENDS "kpu_run_csp_tflow_t4;kpu_floorplan_t4")
        # Residency across operators (step 4d): an inherited slot is Seeded, a retained one Held,
        # and both records keep all nine invariants.
        foreach(_chain matmul_relu_retain_32_t16 bias_inherit_32_t16)
            add_test(NAME kpu_run_csp_tflow_${_chain}
                COMMAND run --program ${CMAKE_SOURCE_DIR}/tests/program/csp/${_chain}.csp
                        --deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_s1.json
                        --level block-sequential --tflow ${CMAKE_CURRENT_BINARY_DIR}/csp_${_chain}.tflow)
            add_test(NAME tflow_check_csp_${_chain}
                COMMAND ${KPU_PYTHON3} ${_tflow_check} ${CMAKE_CURRENT_BINARY_DIR}/csp_${_chain}.tflow)
            set_tests_properties(tflow_check_csp_${_chain} PROPERTIES DEPENDS kpu_run_csp_tflow_${_chain})
            set_tests_properties(kpu_run_csp_tflow_${_chain} tflow_check_csp_${_chain}
                PROPERTIES LABELS "program;tileflow;csp")
        endforeach()
        set_tests_properties(csp_gen_t4_matmul kpu_run_csp_tflow_t4 kpu_run_csp_tflow_linear
            tflow_check_csp_t4 tflow_check_csp_linear tflow_check_v4_selftest
            tileflow_viewer_csp_smoke tileflow_bind_csp_t4
            PROPERTIES LABELS "program;tileflow;csp")

        set_tests_properties(tflow_check_shared_reader tflow_check_selftest tileflow_viewer_smoke
            csp_gen_t64_matmul kpu_run_csp_tflow_t64 tileflow_bind_t64
            PROPERTIES LABELS "program;tileflow")
    endif()

    # The options that fed the L0 L-T1 executor are refused by name for an L0 program, with
    # where their job went (kpu-run-csp-programs step 4e), never ignored.
    foreach(_retired tflow timeline streams)
        add_test(NAME kpu_run_l0_refuses_${_retired}
            COMMAND ${CMAKE_COMMAND}
                -DCMD=$<TARGET_FILE:run>
                -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--${_retired}$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/should_not_exist_${_retired}
                -DEXPECT_CODE=2
                -DMUST_MATCH=--${_retired}.times.an.L0.program.s.L-T1.run,.which.retired.*kpu-csp-gen
                -DMUST_NOT_MATCH=values.vs
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
        )
        set_tests_properties(kpu_run_l0_refuses_${_retired} PROPERTIES LABELS "program;driver")
    endforeach()

    # An option present without its value must be an ERROR, not a silent absence: a
    # terminal --timeline once read as "no timeline requested".
    add_test(NAME kpu_run_rejects_valueless_option
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--emit-l0
            -DEXPECT_CODE=2
            -DMUST_MATCH=--emit-l0:.missing.value
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_rejects_valueless_option PROPERTIES LABELS "program;driver")

    # --step on an L0 program re-executes it at L-B, one op per step. (A CSP program's L-T1
    # replay is kpu_run_csp_step_lt1, below.)
    add_test(NAME kpu_run_step_behavioral
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--level$<SEMICOLON>behavioral$<SEMICOLON>--step$<SEMICOLON>--step-limit$<SEMICOLON>4
            -DEXPECT_CODE=0
            -DMUST_MATCH=re-executed.one.op.at.a.time
            -DMUST_NOT_MATCH=hop-start
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_step_behavioral PROPERTIES LABELS "program;driver")

    # --step-limit says "stop after n steps", and at L-B a step APPLIES an op, so stopping
    # late would execute more than it showed. The reported position makes that checkable.
    add_test(NAME kpu_run_step_limit_is_exact
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--step$<SEMICOLON>--step-limit$<SEMICOLON>4
            -DEXPECT_CODE=0
            -DMUST_MATCH=stopped.after.4.of
            -DMUST_NOT_MATCH=stopped.after.5.of
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_step_limit_is_exact PROPERTIES LABELS "program;driver")

    # --emit-l0-result must be written only AFTER the comparison agrees, so a run whose
    # levels disagreed cannot leave a file that looks like a golden expected output. An L0
    # program runs at L-B alone, so there is no verdict to wait for and it is written; the
    # ordering contract (OK before wrote) is kpu_run_csp_emit_result's.
    add_test(NAME kpu_run_emit_after_agreement
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--emit-l0$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/emit.l0$<SEMICOLON>--emit-l0-result$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/emit.result.l0
            -DEXPECT_CODE=0
            -DMUST_MATCH=wrote.*emit.l0.*L-B.ran.the.oracle.*wrote.*emit.result.l0
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_emit_after_agreement PROPERTIES LABELS "program;driver")

    # ---- a program from a FILE (#265 increment 5) --------------------------------
    # The point of the whole serialization effort: until this worked, every program the
    # simulator ran was one the simulator wrote. The corpus files are absolute paths
    # because these wrappers run with no WORKING_DIRECTORY of their own.
    set(_corpus ${CMAKE_SOURCE_DIR}/tests/program/corpus)
    set(_deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy)

    add_test(NAME kpu_run_program_from_file
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.l0
            -DEXPECT_CODE=0
            -DMUST_MATCH=values:.from.file.*behavioral.*L-B.ran.the.oracle
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_program_from_file PROPERTIES LABELS "program;driver")

    # A kernel carries no values, and running it anyway would have every level agree on
    # ZEROS -- a green result that proves nothing. Refused, and the refusal names the way out.
    add_test(NAME kpu_run_program_refuses_valueless
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_32x32x32_t16_kernel.l0
            -DEXPECT_CODE=2
            -DMUST_MATCH=carries.no.values.*--fill-inputs
            -DMUST_NOT_MATCH=values.vs
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_program_refuses_valueless PROPERTIES
        LABELS "program;driver")

    add_test(NAME kpu_run_program_fill_inputs
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_32x32x32_t16_kernel.l0$<SEMICOLON>--fill-inputs
            -DEXPECT_CODE=0
            -DMUST_MATCH=values:.synthesized.*behavioral.*L-B.ran.the.oracle
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_program_fill_inputs PROPERTIES LABELS "program;driver")

    # ...and the other direction: a file that HAS values must not have them replaced, or
    # the run is no longer the one the file describes.
    add_test(NAME kpu_run_program_refuses_replacing_values
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.l0$<SEMICOLON>--fill-inputs
            -DEXPECT_CODE=2
            -DMUST_MATCH=carries.its.own.values
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_program_refuses_replacing_values PROPERTIES
        LABELS "program;driver")

    # A file AND a derivation spec describe two different programs. Picking one silently
    # would make the tool report the flags it was given and execute something else.
    add_test(NAME kpu_run_program_rejects_derivation_spec
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.l0$<SEMICOLON>--algo$<SEMICOLON>lu
            -DEXPECT_CODE=2
            -DMUST_MATCH=--algo.left.kpu-run
            -DMUST_NOT_MATCH=values.vs
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_program_rejects_derivation_spec PROPERTIES
        LABELS "program;driver")

    # A refusal from the READER has to surface as a diagnostic and exit 2, not as an
    # uncaught FormatError -- which would abort, and CI cannot tell an abort from a crash
    # in the model.
    add_test(NAME kpu_run_program_refuses_newer_reader
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/needs_a_newer_reader.l0
            -DEXPECT_CODE=2
            -DMUST_MATCH=requires.a.reader
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_program_refuses_newer_reader PROPERTIES
        LABELS "program;driver")

    # ---- a CSP program (docs/plans/kpu-run-csp-programs.md step 2) -----------------
    # The program is the input; L0 is its oracle. Fixtures are written by kpu-csp-gen.
    set(_csp ${CMAKE_SOURCE_DIR}/tests/program/csp)

    add_test(NAME kpu_run_csp_matmul
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=oracle...L-B.*L0.trace.*replayed.bit-exact.*cycle-accurate.*L-CA..makespan.*cycle-accurate.+within.bound.*OK:.every.level
            -DMUST_NOT_MATCH=FAILED
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # Unfused linear: C is written first (from the accumulator), so it starts at zero.
    add_test(NAME kpu_run_csp_linear_unfused
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_csp}/linear_64_t16_unfused_gelu.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json$<SEMICOLON>--window$<SEMICOLON>16
            -DEXPECT_CODE=0
            -DMUST_MATCH=cycle-accurate.+within.bound.*OK:.every.level
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # Through the platform (step 4a): every level's run id, and the agreed values written only
    # after the comparison passed.
    add_test(NAME kpu_run_csp_emit_result
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/linear_64_t16_relu.csp$<SEMICOLON>--inputs$<SEMICOLON>${_corpus}/linear_64_t16_relu.l0$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json$<SEMICOLON>--emit-l0-result$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/linear_csp.result.l0
            -DEXPECT_CODE=0
            -DMUST_MATCH=run.id.L-CA.prog:.*schedule:window=256.*OK:.every.level.*wrote
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # L-T1 from the program (step 3): the program's credits, timed per process.
    add_test(NAME kpu_run_csp_block_sequential
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/lu_64_t16.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json$<SEMICOLON>--level$<SEMICOLON>block-sequential
            -DEXPECT_CODE=0
            -DMUST_MATCH=L-T1..makespan.*L3....peak.*of.128.*block-sequential.+within.bound.*OK
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # Values from an L0 file: the corpus's matmul inputs.
    add_test(NAME kpu_run_csp_inputs_from_l0
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--inputs$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.l0$<SEMICOLON>--level$<SEMICOLON>behavioral
            -DEXPECT_CODE=0
            -DMUST_MATCH=values:.from.*matmul_48x48x48_t16.l0.*OK
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # `all` runs what can run and says why the rest did not.
    add_test(NAME kpu_run_csp_lu_skips_lca
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/lu_64_t16.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=L-CA..skipped:.L-CA.runs.matmul.*OK:.every.level
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # ...and a level named explicitly that cannot run is a usage error, with the reason.
    add_test(NAME kpu_run_csp_refuses_lca_on_t4
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--level$<SEMICOLON>cycle-accurate
            -DEXPECT_CODE=2
            -DMUST_MATCH=cycle-accurate:.*level.2.distributes
            -DMUST_NOT_MATCH=values.vs
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # Stepping and the timeline from the program (step 4c): L-T1 replays its records with
    # lane and L3-slot occupancy; L-B applies one action per step and stops at the limit, naming
    # the process, the move, the residency and the source position (domain-flow step 1); the
    # timeline is one event per leg. Each refuses a run that lacks the level it reads.
    add_test(NAME kpu_run_csp_step_lt1
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/linear_64_t16_relu.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json$<SEMICOLON>--level$<SEMICOLON>block-sequential$<SEMICOLON>--step$<SEMICOLON>--step-limit$<SEMICOLON>20
            -DEXPECT_CODE=0
            -DMUST_MATCH=stepping.L-T1:.*records.*@0.+action.0.LOAD.*dma.lane.0.*L3.[0-9]+/128.*RELEASE.*stopped.after.20.of.*OK
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    add_test(NAME kpu_run_csp_step_lb
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/linear_64_t16_relu.csp$<SEMICOLON>--level$<SEMICOLON>behavioral$<SEMICOLON>--step$<SEMICOLON>--step-limit$<SEMICOLON>12
            -DEXPECT_CODE=0
            -DMUST_MATCH=stepping.L-B:.*actions.*each.applied.*LOAD.B.0,0.*dma.dram->l3..r0.opens....line.9..j=0/4.*LOAD.A.0,0.*line.11..j=0/4.i=0/4.*FEED.A.0,0...str.l2->cf.*line.13..j=0/4.i=0/4.k=0/4.*fabric.1.*stopped.after.12.of
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    add_test(NAME kpu_run_csp_step_refuses_lca_only
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json$<SEMICOLON>--level$<SEMICOLON>cycle-accurate$<SEMICOLON>--step
            -DEXPECT_CODE=2
            -DMUST_MATCH=--step.walks.L-B.*or.L-T1.*neither.ran
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    add_test(NAME kpu_run_csp_timeline
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--level$<SEMICOLON>block-sequential$<SEMICOLON>--timeline$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/csp_matmul_timeline.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=wrote.*csp_matmul_timeline.json.*events,.one.per.leg,.from.L-T1
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    add_test(NAME kpu_run_csp_timeline_needs_lt1
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--level$<SEMICOLON>behavioral$<SEMICOLON>--timeline$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/should_not_exist_csp.json
            -DEXPECT_CODE=2
            -DMUST_MATCH=--timeline.records.the.L-T1.run
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # The domain-flow record (domain-flow-debugger step 2, #363): the oracle's own L-B run, one
    # step per action. A bundle for every CSP corpus program and the retain/inherit chains, each
    # checked against the plan's invariants; a cut bundle; the reference run (matmul 512^3 in
    # 64-tiles at L3 17, plan Q6); and the checker's self-test, which must fail every invariant.
    # Each bundle's writer is a CTest fixture of its checkers, so a checker run alone writes a fresh
    # bundle first, and is skipped if writing failed (DEPENDS only orders tests).
    if(KPU_PYTHON3)
        set(_dflow_check ${CMAKE_SOURCE_DIR}/tools/trace/dflow_check.py)
        set(_dflow_tests "")
        foreach(_p matmul_48x48x48_t16 linear_64_t16_relu lu_64_t16)
            # linear records every tile's values (step 4) and the L-T1 overlay (step 5), so the
            # self-test's bundle carries both; the others are L-B only (no overlay).
            set(_vals --level behavioral)
            if(_p STREQUAL "linear_64_t16_relu")
                set(_vals --level block-sequential --deploy ${_deploy}/kpu_s1.json --dflow-values all)
            endif()
            add_test(NAME kpu_run_csp_dflow_${_p}
                COMMAND run --program ${_corpus}/${_p}.csp
                        --dflow ${CMAKE_CURRENT_BINARY_DIR}/csp_${_p}.dflow ${_vals})
            add_test(NAME dflow_check_${_p}
                COMMAND ${KPU_PYTHON3} ${_dflow_check} ${CMAKE_CURRENT_BINARY_DIR}/csp_${_p}.dflow)
            set_tests_properties(kpu_run_csp_dflow_${_p} PROPERTIES FIXTURES_SETUP dflow_${_p})
            set_tests_properties(dflow_check_${_p} PROPERTIES FIXTURES_REQUIRED dflow_${_p})
            list(APPEND _dflow_tests kpu_run_csp_dflow_${_p} dflow_check_${_p})
        endforeach()
        foreach(_p matmul_relu_retain_32_t16 bias_inherit_32_t16)
            add_test(NAME kpu_run_csp_dflow_${_p}
                COMMAND run --program ${_csp}/${_p}.csp --deploy ${_deploy}/kpu_s1.json --level behavioral
                        --dflow ${CMAKE_CURRENT_BINARY_DIR}/csp_${_p}.dflow)
            add_test(NAME dflow_check_${_p}
                COMMAND ${KPU_PYTHON3} ${_dflow_check} ${CMAKE_CURRENT_BINARY_DIR}/csp_${_p}.dflow)
            set_tests_properties(kpu_run_csp_dflow_${_p} PROPERTIES FIXTURES_SETUP dflow_${_p})
            set_tests_properties(dflow_check_${_p} PROPERTIES FIXTURES_REQUIRED dflow_${_p})
            list(APPEND _dflow_tests kpu_run_csp_dflow_${_p} dflow_check_${_p})
        endforeach()
        add_test(NAME kpu_run_csp_dflow_cut
            COMMAND ${CMAKE_COMMAND}
                -DCMD=$<TARGET_FILE:run>
                -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--level$<SEMICOLON>behavioral$<SEMICOLON>--dflow$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/csp_cut.dflow$<SEMICOLON>--dflow-limit$<SEMICOLON>50
                -DEXPECT_CODE=0
                -DMUST_MATCH=wrote.*csp_cut.dflow.*50.of.243.steps.*steps.cut.at.--dflow-limit
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
        )
        add_test(NAME dflow_check_cut
            COMMAND ${KPU_PYTHON3} ${_dflow_check} ${CMAKE_CURRENT_BINARY_DIR}/csp_cut.dflow)
        set_tests_properties(kpu_run_csp_dflow_cut PROPERTIES FIXTURES_SETUP dflow_cut)
        set_tests_properties(dflow_check_cut PROPERTIES FIXTURES_REQUIRED dflow_cut)
        add_test(NAME csp_gen_dflow_reference
            COMMAND $<TARGET_FILE:csp-gen> --algo matmul --size 512 --tile 64 --l3 17
                    -o ${CMAKE_CURRENT_BINARY_DIR}/dflow_reference.csp)
        add_test(NAME kpu_run_csp_dflow_reference
            COMMAND run --program ${CMAKE_CURRENT_BINARY_DIR}/dflow_reference.csp
                    --deploy ${_deploy}/kpu_s1.json --level block-sequential
                    --dflow ${CMAKE_CURRENT_BINARY_DIR}/csp_reference.dflow
                    --dflow-values C[2,3],A[2,0],B[0,3])
        set_tests_properties(csp_gen_dflow_reference PROPERTIES FIXTURES_SETUP dflow_reference_csp)
        set_tests_properties(kpu_run_csp_dflow_reference PROPERTIES
            FIXTURES_REQUIRED dflow_reference_csp FIXTURES_SETUP dflow_reference)
        add_test(NAME dflow_check_reference
            COMMAND ${KPU_PYTHON3} ${_dflow_check} ${CMAKE_CURRENT_BINARY_DIR}/csp_reference.dflow)
        set_tests_properties(dflow_check_reference PROPERTIES FIXTURES_REQUIRED dflow_reference)
        add_test(NAME dflow_check_selftest
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/trace/test_dflow_check.py
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_linear_64_t16_relu.dflow)
        set_tests_properties(dflow_check_selftest PROPERTIES FIXTURES_REQUIRED dflow_linear_64_t16_relu)
        # The L-T1 overlay's slot waits (step 5): a stream that reuses its slots early.
        add_test(NAME kpu_run_csp_dflow_streamed
            COMMAND ${CMAKE_COMMAND}
                -DCMD=$<TARGET_FILE:run>
                -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/dflow/streamed_lowest.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json$<SEMICOLON>--level$<SEMICOLON>block-sequential$<SEMICOLON>--dflow$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/csp_streamed.dflow$<SEMICOLON>--dflow-values$<SEMICOLON>C[0,0]
                -DEXPECT_CODE=0
                -DMUST_MATCH=wrote.*csp_streamed.dflow.*L-T1.overlay.*slot-bound.14..704.cycles
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
        )
        set_tests_properties(kpu_run_csp_dflow_streamed PROPERTIES FIXTURES_SETUP dflow_streamed)
        add_test(NAME dflow_check_streamed
            COMMAND ${KPU_PYTHON3} ${_dflow_check} ${CMAKE_CURRENT_BINARY_DIR}/csp_streamed.dflow)
        set_tests_properties(dflow_check_streamed PROPERTIES FIXTURES_REQUIRED dflow_streamed)
        # A value selection that names no operand is refused, and writes nothing.
        add_test(NAME kpu_run_csp_dflow_values_refuses
            COMMAND ${CMAKE_COMMAND}
                -DCMD=$<TARGET_FILE:run>
                -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--level$<SEMICOLON>behavioral$<SEMICOLON>--dflow$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/should_not_exist.dflow$<SEMICOLON>--dflow-values$<SEMICOLON>Q[0,0]
                -DEXPECT_CODE=2
                -DMUST_MATCH=--dflow-values:.Q.0,0..names.no.declared.operand
                -DMUST_NOT_MATCH=wrote
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
        )
        # The viewer (step 3): pack each bundle, refuse a non-bundle, and run the page's data layer
        # in Node on the reference run, a retain chain, LU and a cut bundle.
        add_test(NAME domainflow_viewer
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/domainflow/test_viewer.py
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_reference.dflow
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_matmul_relu_retain_32_t16.dflow
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_lu_64_t16.dflow
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_cut.dflow
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_linear_64_t16_relu.dflow
                    ${CMAKE_CURRENT_BINARY_DIR}/csp_streamed.dflow)
        set_tests_properties(domainflow_viewer PROPERTIES FIXTURES_REQUIRED
            "dflow_reference;dflow_matmul_relu_retain_32_t16;dflow_lu_64_t16;dflow_cut;dflow_linear_64_t16_relu;dflow_streamed")
        set_tests_properties(${_dflow_tests} kpu_run_csp_dflow_cut dflow_check_cut csp_gen_dflow_reference
            kpu_run_csp_dflow_reference dflow_check_reference dflow_check_selftest domainflow_viewer
            kpu_run_csp_dflow_values_refuses kpu_run_csp_dflow_streamed dflow_check_streamed
            PROPERTIES LABELS "program;domainflow;csp")
    endif()
    # Residency across operators (step 4d): the first operator of a chain retains its result
    # (no stores), the second inherits it (no loads of it), each at every level.
    add_test(NAME kpu_run_csp_retain
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_csp}/matmul_relu_retain_32_t16.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=0.stores,.4.retained.*L-CA..makespan.*DRAM..12.loads,.0.stores.*cycle-accurate.+within.bound.*OK
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    add_test(NAME kpu_run_csp_inherit
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_csp}/bias_inherit_32_t16.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=4.inherited.*L-CA..makespan.*DRAM..2.loads,.4.stores.*cycle-accurate.+within.bound.*OK
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    add_test(NAME kpu_run_csp_rejects_algo
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--algo$<SEMICOLON>matmul
            -DEXPECT_CODE=2
            -DMUST_MATCH=--algo.left.kpu-run.*kpu-csp-gen
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    add_test(NAME kpu_run_csp_refuses_invalid_program
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_csp}/leaves_resident.csp
            -DEXPECT_CODE=2
            -DMUST_MATCH=still.resident.at.the.end.of.the.program
            -DMUST_NOT_MATCH=values.vs
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # A file is written only after the levels agree, so asking for one without comparing is
    # refused, not answered with exit 0 and no file.
    add_test(NAME kpu_run_csp_emit_needs_compare
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--no-compare$<SEMICOLON>--emit-l0$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/should_not_exist_trace.l0
            -DEXPECT_CODE=2
            -DMUST_MATCH=cannot.be.combined.with.--no-compare
            -DMUST_NOT_MATCH=run.id
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # L-B is the oracle at any size; only the trace is limited, so asking for one past the limit
    # is refused.
    add_test(NAME kpu_run_csp_emit_l0_needs_a_trace
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--level$<SEMICOLON>block-sequential$<SEMICOLON>--trace-limit$<SEMICOLON>1$<SEMICOLON>--emit-l0$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/should_not_exist.l0
            -DEXPECT_CODE=2
            -DMUST_MATCH=emit-l0:.the.program.has.no.L0.trace
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_csp_emit_needs_compare kpu_run_csp_emit_l0_needs_a_trace
        PROPERTIES LABELS "program;driver;csp")
    # An L0 file is a trace: L-CA runs CSP programs.
    add_test(NAME kpu_run_l0_refuses_lca
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.l0$<SEMICOLON>--level$<SEMICOLON>cycle-accurate
            -DEXPECT_CODE=2
            -DMUST_MATCH=an.L0.program.is.a.trace.*kpu-csp-gen
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_csp_matmul kpu_run_csp_linear_unfused kpu_run_csp_inputs_from_l0
        kpu_run_csp_block_sequential kpu_run_csp_emit_result
        kpu_run_csp_lu_skips_lca kpu_run_csp_refuses_lca_on_t4 kpu_run_csp_rejects_algo
        kpu_run_csp_refuses_invalid_program kpu_run_l0_refuses_lca
        kpu_run_csp_step_lt1 kpu_run_csp_step_lb kpu_run_csp_step_refuses_lca_only
        kpu_run_csp_timeline kpu_run_csp_timeline_needs_lt1 kpu_run_csp_retain kpu_run_csp_inherit
        PROPERTIES LABELS "program;driver;csp")

    # ---- the machine as a spec (#282 increment 4) ---------------------------------
    add_test(NAME kpu_run_deploy_from_spec
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/adr_checkerboard16.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=checkerboard.cf16.*run.id.*L-B.ran.the.oracle
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_deploy_from_spec PROPERTIES LABELS "program;driver")

    # The device flags left with --algo (step 5): the machine is a deployment spec, and an L0
    # program runs at L-B, which models none of it. Refused by name, never ignored.
    add_test(NAME kpu_run_refuses_device_flags
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/canonical_single.json$<SEMICOLON>--topology$<SEMICOLON>news
            -DEXPECT_CODE=2
            -DMUST_MATCH=--topology.left.kpu-run.*deployment.spec
            -DMUST_NOT_MATCH=run.id
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_refuses_device_flags PROPERTIES LABELS "program;driver")

    # A declared field the level does not model is REPORTED, compactly -- one line per level
    # rather than one per field, which is what made an 11-line report nobody would read.
    add_test(NAME kpu_run_reports_unmodelled_fields
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/adr_checkerboard16.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=L-B.ignores:.*l3.banks.*l1.vectors
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_reports_unmodelled_fields PROPERTIES LABELS "program;driver")

    # Multi-device execution is not implemented, and running device 0 while ignoring the rest
    # would be a deployment described and a machine run that are not the same machine.
    add_test(NAME kpu_run_refuses_multi_device
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/two_devices.json
            -DEXPECT_CODE=2
            -DMUST_MATCH=multi-device.execution.is.not.implemented
            -DMUST_NOT_MATCH=OK:.every.level
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_refuses_multi_device PROPERTIES LABELS "program;driver")

    add_test(NAME kpu_run_deploy_missing_file
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--deploy$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/no_such_deploy.json
            -DEXPECT_CODE=2
            -DMUST_MATCH=cannot.read
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_deploy_missing_file PROPERTIES LABELS "program;driver")

    add_test(NAME kpu_run_program_missing_file
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/no_such_program.l0
            -DEXPECT_CODE=2
            -DMUST_MATCH=cannot.read
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_program_missing_file PROPERTIES LABELS "program;driver")

    # Bad numeric input must also exit 2 rather than abort: an uncaught std::stoul throw
    # is SIGABRT, which CI cannot distinguish from a crash in the model.
    add_test(NAME kpu_run_rejects_bad_number
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/corpus/matmul_48x48x48_t16.l0$<SEMICOLON>--step-limit$<SEMICOLON>abc
            -DEXPECT_CODE=2
            -DMUST_MATCH=--step-limit
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_rejects_bad_number PROPERTIES
        LABELS "program;driver")
endif()
