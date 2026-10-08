# ============================================================================
# tools/kpu-memsim/CMakeLists.txt
# kpu-memsim -- the memory side of a deployment on its own, recorded as .mflow
# (docs/plans/memory-side-debugger.md step 3).
# ============================================================================

kpu_add_tool(memsim
    FOLDER "Driver"
    DESCRIPTION "Run the memory side (DRAM, controllers, DMA engines, buffers) with stubbed NoC ports; write .mflow"
    SOURCES
        kpu_memsim.cpp
)

if(KPU_BUILD_TESTS)
    set(_expect ${CMAKE_SOURCE_DIR}/tools/kpu-run/expect_exit.cmake)
    set(_deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy)
    set(_memside ${CMAKE_SOURCE_DIR}/tests/memside)

    # The T4 runs a mixed scenario to completion and records it.
    add_test(NAME kpu_memsim_t4
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:memsim>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--scenario$<SEMICOLON>${_memside}/t4_mixed.json$<SEMICOLON>--out$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t4_mixed.mflow
            -DEXPECT_CODE=0
            -DMUST_MATCH=DRAM.ceiling
            -P ${_expect}
    )
    # A scenario with an unknown key is refused by name, exit 2.
    add_test(NAME kpu_memsim_rejects_bad_scenario
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:memsim>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--scenario$<SEMICOLON>${_memside}/bad_scenario.json$<SEMICOLON>--out$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/bad.mflow
            -DEXPECT_CODE=2
            -DMUST_MATCH=unknown.key..bogus.
            -P ${_expect}
    )
    # A run cut short by max_cycles says so, exit 1 (and still writes what it saw).
    add_test(NAME kpu_memsim_unfinished
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:memsim>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--scenario$<SEMICOLON>${_memside}/t4_too_short.json$<SEMICOLON>--out$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/short.mflow
            -DEXPECT_CODE=1
            -DMUST_MATCH=did.not.finish
            -P ${_expect}
    )
    # Input that would wrap or be silently ignored is refused by name, exit 2.
    foreach(_case IN ITEMS
            "negative|bad_negative.json|not.a.non-negative.number"
            "trailing|bad_trailing.json|not.JSON"
            "huge|bad_huge.json|larger.than.4294967295"
            "zero_random|bad_zero_random.json|bytes.must.be.greater.than.zero")
        string(REPLACE "|" ";" _parts "${_case}")
        list(GET _parts 0 _name)
        list(GET _parts 1 _file)
        list(GET _parts 2 _match)
        add_test(NAME kpu_memsim_rejects_${_name}
            COMMAND ${CMAKE_COMMAND}
                -DCMD=$<TARGET_FILE:memsim>
                -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--scenario$<SEMICOLON>${_memside}/${_file}$<SEMICOLON>--out$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/rejected.mflow
                -DEXPECT_CODE=2
                -DMUST_MATCH=${_match}
                -P ${_expect}
        )
        set_tests_properties(kpu_memsim_rejects_${_name} PROPERTIES LABELS "memside;dram;tools")
    endforeach()
    add_test(NAME kpu_memsim_rejects_bad_device
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:memsim>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--device$<SEMICOLON>1x$<SEMICOLON>--scenario$<SEMICOLON>${_memside}/t4_mixed.json$<SEMICOLON>--out$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/rejected.mflow
            -DEXPECT_CODE=2
            -DMUST_MATCH=device.index
            -P ${_expect}
    )
    set_tests_properties(kpu_memsim_t4 kpu_memsim_rejects_bad_scenario kpu_memsim_unfinished
        kpu_memsim_rejects_bad_device PROPERTIES LABELS "memside;dram;tools")

    # The debugger (plan step 5): pack the T4 run, refuse a non-bundle, parse the script and
    # run its reader over the bundle.
    find_program(KPU_PYTHON3 NAMES python3 python)
    if(KPU_PYTHON3)
        add_test(NAME memflow_viewer_smoke
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/visualization/memflow/test_viewer.py
                    ${CMAKE_CURRENT_BINARY_DIR}/t4_mixed.mflow)
        set_tests_properties(memflow_viewer_smoke PROPERTIES DEPENDS kpu_memsim_t4
            LABELS "memside;dram;tools")

        # The checker (plan step 4): the T4 run holds every invariant, and each check fails on
        # a bundle that breaks it.
        add_test(NAME mflow_check_t4
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/trace/mflow_check.py
                    ${CMAKE_CURRENT_BINARY_DIR}/t4_mixed.mflow)
        add_test(NAME mflow_check_selftest
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/trace/test_mflow_check.py
                    ${CMAKE_CURRENT_BINARY_DIR}/t4_mixed.mflow)
        set_tests_properties(mflow_check_t4 mflow_check_selftest PROPERTIES DEPENDS kpu_memsim_t4
            LABELS "memside;dram;tools")
    endif()
endif()
