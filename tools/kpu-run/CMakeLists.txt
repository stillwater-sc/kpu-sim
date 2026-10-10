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

    # --tflow writes the tile-flow record from the level that models resources.
    add_test(NAME kpu_run_tflow_written
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--tflow$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/kpu_run_matmul.tflow
            -DEXPECT_CODE=0
            -DMUST_MATCH=residencies,.*transits,.*computes.from
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_tflow_written PROPERTIES LABELS "program;driver;tileflow")

    # The T4 reference SKU: 512^3 in 64x64 tiles is 1600 ops -- small enough to read every
    # movement in the tile-flow viewer, large enough to fill most of the 128-slot L3.
    add_test(NAME kpu_run_tflow_t4
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--deploy$<SEMICOLON>${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_t4.json$<SEMICOLON>--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>512$<SEMICOLON>--tile$<SEMICOLON>64$<SEMICOLON>--tflow$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/kpu_run_t4.tflow
            -DEXPECT_CODE=0
            -DMUST_MATCH=residencies,.*transits,.*computes.from
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_tflow_t4 PROPERTIES LABELS "program;driver;tileflow;t4")

    # The tile-flow checker (#286 step 3) over real bundles, and its self-test: every
    # invariant must fail on a bundle that breaks it. Skipped without a Python interpreter.
    find_program(KPU_PYTHON3 NAMES python3 python)
    if(KPU_PYTHON3)
        set(_tflow_check ${CMAKE_SOURCE_DIR}/tools/trace/tflow_check.py)
        add_test(NAME tflow_check_matmul
            COMMAND ${KPU_PYTHON3} ${_tflow_check} ${CMAKE_CURRENT_BINARY_DIR}/kpu_run_matmul.tflow)
        set_tests_properties(tflow_check_matmul PROPERTIES DEPENDS kpu_run_tflow_written)
        # The T4 reference run (#286): the bundle the viewer is demonstrated on must be clean.
        add_test(NAME tflow_check_t4
            COMMAND ${KPU_PYTHON3} ${_tflow_check} ${CMAKE_CURRENT_BINARY_DIR}/kpu_run_t4.tflow)
        set_tests_properties(tflow_check_t4 PROPERTIES DEPENDS kpu_run_tflow_t4)

        # The #279 program: two readers of one tile, the higher-indexed one finishing first.
        # Clean at L3 = 5; the checker's TF9 is what fails it under the old release rule.
        add_test(NAME kpu_run_tflow_shared_reader
            COMMAND run --program ${CMAKE_SOURCE_DIR}/tests/program/tflow/shared_reader.l0
                    --fill-inputs --level block-sequential --compute-tiles 2 --l3-tiles 5
                    --tflow ${CMAKE_CURRENT_BINARY_DIR}/shared_reader.tflow)
        add_test(NAME tflow_check_shared_reader
            COMMAND ${KPU_PYTHON3} ${_tflow_check} ${CMAKE_CURRENT_BINARY_DIR}/shared_reader.tflow)
        set_tests_properties(tflow_check_shared_reader PROPERTIES DEPENDS kpu_run_tflow_shared_reader)
        add_test(NAME tflow_check_selftest
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/trace/test_tflow_check.py
                    ${CMAKE_CURRENT_BINARY_DIR}/shared_reader.tflow)
        set_tests_properties(tflow_check_selftest PROPERTIES DEPENDS kpu_run_tflow_shared_reader)
        # The viewer (#286 step 4): pack a real bundle, refuse a non-bundle, parse the script.
        add_test(NAME tileflow_viewer_smoke
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/tileflow/test_viewer.py
                    ${CMAKE_CURRENT_BINARY_DIR}/kpu_run_matmul.tflow)
        set_tests_properties(tileflow_viewer_smoke PROPERTIES DEPENDS kpu_run_tflow_written)

        # The derived binding on the T4 reference run (#286): the bundle and floorplan the
        # viewer is demonstrated on. The T64 one below stays as the scale check.
        add_test(NAME tileflow_bind_t4
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/tileflow/test_bind.py
                    ${CMAKE_CURRENT_BINARY_DIR}/kpu_run_t4.tflow
                    ${CMAKE_BINARY_DIR}/tools/floorplan/t4_floorplan.json)
        set_tests_properties(tileflow_bind_t4 PROPERTIES DEPENDS "kpu_run_tflow_t4;kpu_floorplan_t4")

        # The derived binding (bind.py) on a T64 run and its floorplan (written by
        # kpu_floorplan_t64 in tools/floorplan).
        add_test(NAME kpu_run_tflow_t64
            COMMAND run --algo matmul --size 128 --tile 32 --level block-sequential
                    --deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_t64.json
                    --tflow ${CMAKE_CURRENT_BINARY_DIR}/t64_matmul.tflow)
        add_test(NAME tileflow_bind_t64
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/tileflow/test_bind.py
                    ${CMAKE_CURRENT_BINARY_DIR}/t64_matmul.tflow
                    ${CMAKE_BINARY_DIR}/tools/floorplan/t64_floorplan.json)
        set_tests_properties(tileflow_bind_t64 PROPERTIES DEPENDS "kpu_run_tflow_t64;kpu_floorplan_t64")
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
        set_tests_properties(csp_gen_t4_matmul kpu_run_csp_tflow_t4 kpu_run_csp_tflow_linear
            tflow_check_csp_t4 tflow_check_csp_linear tflow_check_v4_selftest
            tileflow_viewer_csp_smoke tileflow_bind_csp_t4
            PROPERTIES LABELS "program;tileflow;csp")

        set_tests_properties(tflow_check_matmul tflow_check_t4 kpu_run_tflow_shared_reader
            tflow_check_shared_reader tflow_check_selftest tileflow_viewer_smoke
            kpu_run_tflow_t64 tileflow_bind_t64 tileflow_bind_t4
            PROPERTIES LABELS "program;tileflow")
    endif()

    add_test(NAME kpu_run_tflow_refused_without_timing
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--level$<SEMICOLON>behavioral$<SEMICOLON>--tflow$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/should_not_exist.tflow
            -DEXPECT_CODE=2
            -DMUST_MATCH=--tflow.needs.a.level.that.models.resources
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_tflow_refused_without_timing PROPERTIES
        LABELS "program;driver;tileflow")

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

    # --step-limit says "stop after n steps", and at L-B a step APPLIES an op, so stopping
    # late would execute more than it showed. The reported position makes that checkable.
    add_test(NAME kpu_run_step_limit_is_exact
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--level$<SEMICOLON>behavioral$<SEMICOLON>--step$<SEMICOLON>--step-limit$<SEMICOLON>4
            -DEXPECT_CODE=0
            -DMUST_MATCH=stopped.after.4.of
            -DMUST_NOT_MATCH=stopped.after.5.of
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_step_limit_is_exact PROPERTIES LABELS "program;driver")

    # --emit-l0-result must be written only AFTER the comparison agrees, so a run whose
    # levels disagreed cannot leave a file that looks like a golden expected output. The
    # ordering is the observable contract: OK before wrote.
    add_test(NAME kpu_run_emit_after_agreement
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--algo$<SEMICOLON>matmul$<SEMICOLON>--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--emit-l0$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/emit.l0$<SEMICOLON>--emit-l0-result$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/emit.result.l0
            -DEXPECT_CODE=0
            -DMUST_MATCH=OK:.every.level.computes.identical.values.*wrote
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
            -DMUST_MATCH=values:.from.file.*values.vs.L-B.*OK:.every.level
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
            -DMUST_MATCH=values:.synthesized.*OK:.every.level
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
            -DMUST_MATCH=--program.and.--algo.cannot.be.combined
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

    # A loaded program has no algo to gate --streams on, so the refusal comes from the
    # derivation -- which throws about the operand it could not find. The FLAG has to appear
    # in the diagnostic, or "no operand 'C'" is all the user sees.
    add_test(NAME kpu_run_program_names_the_flag_for_streams
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/lu_64_t16.l0$<SEMICOLON>--streams$<SEMICOLON>os
            -DEXPECT_CODE=2
            -DMUST_MATCH=--streams.output-stationary:.*no.matmul.stream.derivation
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_program_names_the_flag_for_streams PROPERTIES
        LABELS "program;driver")

    # ---- a CSP program (docs/plans/kpu-run-csp-programs.md step 2) -----------------
    # The program is the input; L0 is its oracle. Fixtures are written by kpu-csp-gen.
    set(_csp ${CMAKE_SOURCE_DIR}/tests/program/csp)

    add_test(NAME kpu_run_csp_matmul
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=oracle...L0.trace.*cycle-accurate.*L-CA..makespan.*cycle-accurate.+identical.*OK:.every.level
            -DMUST_NOT_MATCH=FAILED
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    # Unfused linear: C is written first (from the accumulator), so it starts at zero.
    add_test(NAME kpu_run_csp_linear_unfused
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_csp}/linear_64_t16_unfused_gelu.csp$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/kpu_s1.json$<SEMICOLON>--window$<SEMICOLON>16
            -DEXPECT_CODE=0
            -DMUST_MATCH=cycle-accurate.+identical.*OK:.every.level
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
            -DMUST_MATCH=L-T1..makespan.*L3....peak.*of.128.*block-sequential.+identical.*OK
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
    # lane and L3-slot occupancy; L-B applies one action per step and stops at the limit; the
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
            -DMUST_MATCH=stepping.L-B:.*actions.*each.applied.*FEED.A.0,0.*fabric.1.*stopped.after.12.of
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
    add_test(NAME kpu_run_csp_rejects_algo
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--program$<SEMICOLON>${_corpus}/matmul_48x48x48_t16.csp$<SEMICOLON>--algo$<SEMICOLON>matmul
            -DEXPECT_CODE=2
            -DMUST_MATCH=--algo.does.not.apply.to.a.CSP.program.*kpu-csp-gen
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
        kpu_run_csp_timeline kpu_run_csp_timeline_needs_lt1
        PROPERTIES LABELS "program;driver;csp")

    # ---- the machine as a spec (#282 increment 4) ---------------------------------
    add_test(NAME kpu_run_deploy_from_spec
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/adr_checkerboard16.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=checkerboard.cf16.*run.id.*OK:.every.level
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_deploy_from_spec PROPERTIES LABELS "program;driver")

    # A spec AND a device flag describe two machines. Picking one silently would make the
    # tool report the flags it was given and run something else.
    add_test(NAME kpu_run_deploy_rejects_device_flags
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/canonical_single.json$<SEMICOLON>--topology$<SEMICOLON>news
            -DEXPECT_CODE=2
            -DMUST_MATCH=--deploy.and.--topology.cannot.be.combined
            -DMUST_NOT_MATCH=values.vs
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_deploy_rejects_device_flags PROPERTIES LABELS "program;driver")

    # A declared field the level does not model is REPORTED, compactly -- one line per level
    # rather than one per field, which is what made an 11-line report nobody would read.
    add_test(NAME kpu_run_reports_unmodelled_fields
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/adr_checkerboard16.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=L-T1.ignores:.*l3.banks.*l1.vectors
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_reports_unmodelled_fields PROPERTIES LABELS "program;driver")

    # Multi-device execution is not implemented, and running device 0 while ignoring the rest
    # would be a deployment described and a machine run that are not the same machine.
    add_test(NAME kpu_run_refuses_multi_device
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--size$<SEMICOLON>32$<SEMICOLON>--tile$<SEMICOLON>16$<SEMICOLON>--deploy$<SEMICOLON>${_deploy}/two_devices.json
            -DEXPECT_CODE=2
            -DMUST_MATCH=multi-device.execution.is.not.implemented
            -DMUST_NOT_MATCH=OK:.every.level
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_refuses_multi_device PROPERTIES LABELS "program;driver")

    add_test(NAME kpu_run_deploy_missing_file
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:run>
            -DARGS=--deploy$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/no_such_deploy.json
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
            -DARGS=--size$<SEMICOLON>abc
            -DEXPECT_CODE=2
            -DMUST_MATCH=--size
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake
    )
    set_tests_properties(kpu_run_rejects_bad_number PROPERTIES
        LABELS "program;driver")
endif()
