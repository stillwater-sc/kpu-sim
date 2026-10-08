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
endif()
