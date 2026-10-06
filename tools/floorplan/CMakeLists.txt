# ============================================================================
# tools/floorplan/CMakeLists.txt
# kpu-floorplan -- the physical shape of a deployment (#286 step 1).
# ============================================================================

kpu_add_tool(floorplan
    FOLDER "Driver"
    DESCRIPTION "Generate or validate a deployment's floorplan; write JSON and SVG"
    SOURCES
        kpu_floorplan.cpp
)

if(KPU_BUILD_TESTS)
    set(_expect ${CMAKE_SOURCE_DIR}/tools/kpu-run/expect_exit.cmake)
    set(_deploy ${CMAKE_SOURCE_DIR}/tests/program/deploy)

    # The T64 lays out, and the round trip through the written JSON validates.
    add_test(NAME kpu_floorplan_t64
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t64.json$<SEMICOLON>--json$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t64_floorplan.json$<SEMICOLON>--svg$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t64_floorplan.svg
            -DEXPECT_CODE=0
            -DMUST_MATCH=folded.2D.torus:.4.row.loops.x.4.column.loops,.60.wires,.16.fold-end.ports
            -P ${_expect}
    )
    add_test(NAME kpu_floorplan_t64_import
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t64.json$<SEMICOLON>--import$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t64_floorplan.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=generated:checkerboard
            -P ${_expect}
    )
    set_tests_properties(kpu_floorplan_t64_import PROPERTIES DEPENDS kpu_floorplan_t64)

    # The T4 reference SKU: a 2x2 board whose torus is one wire.
    add_test(NAME kpu_floorplan_t4
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--json$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t4_floorplan.json$<SEMICOLON>--svg$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t4_floorplan.svg
            -DEXPECT_CODE=0
            -DMUST_MATCH=2x2:.2.L3.tiles,.2.compute.tiles,.4.BlockMovers
            -P ${_expect}
    )
    add_test(NAME kpu_floorplan_t4_import
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--import$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t4_floorplan.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=generated:checkerboard
            -P ${_expect}
    )
    set_tests_properties(kpu_floorplan_t4_import PROPERTIES DEPENDS kpu_floorplan_t4)

    # The T16: the smallest board whose torus has four-hub rings in both dimensions.
    add_test(NAME kpu_floorplan_t16
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t16.json$<SEMICOLON>--json$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t16_floorplan.json$<SEMICOLON>--svg$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t16_floorplan.svg
            -DEXPECT_CODE=0
            -DMUST_MATCH=folded.2D.torus:.2.row.loops.x.2.column.loops,.12.wires,.8.fold-end.ports
            -P ${_expect}
    )
    add_test(NAME kpu_floorplan_t16_import
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t16.json$<SEMICOLON>--import$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t16_floorplan.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=generated:checkerboard
            -P ${_expect}
    )
    set_tests_properties(kpu_floorplan_t16_import PROPERTIES DEPENDS kpu_floorplan_t16)

    # A deployment that describes no layout is refused with the reason, exit 1.
    add_test(NAME kpu_floorplan_refuses_no_layout
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/adr_checkerboard16.json
            -DEXPECT_CODE=1
            -DMUST_MATCH=no.array.layout
            -P ${_expect}
    )
    set_tests_properties(kpu_floorplan_t64 kpu_floorplan_t64_import kpu_floorplan_t4 kpu_floorplan_t4_import
        kpu_floorplan_t16 kpu_floorplan_t16_import kpu_floorplan_refuses_no_layout
        PROPERTIES LABELS "program;platform;floorplan")
endif()
