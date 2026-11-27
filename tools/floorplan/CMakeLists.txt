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
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_t4.json$<SEMICOLON>--json$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t4_floorplan.json$<SEMICOLON>--svg$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t4_floorplan.svg$<SEMICOLON>--structure$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/t4_native.structure.json
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

    # A DMM mesh (#604; docs/plans/distributed-csp-programs.md): nodes of an L3 tile and its compute
    # tile, one-hop links, eight BlockMovers per node, and the periphery ring.
    add_test(NAME kpu_floorplan_dmm4x4
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_dmm_4x4.json$<SEMICOLON>--json$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/dmm4x4_floorplan.json$<SEMICOLON>--svg$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/dmm4x4_floorplan.svg$<SEMICOLON>--structure$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/dmm4x4_native.structure.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=ring.of.24.stops:.16.edge.ports.and.8.memory.controller
            -P ${_expect}
    )
    add_test(NAME kpu_floorplan_dmm4x4_import
        COMMAND ${CMAKE_COMMAND}
            -DCMD=$<TARGET_FILE:floorplan>
            -DARGS=--deploy$<SEMICOLON>${_deploy}/kpu_dmm_4x4.json$<SEMICOLON>--import$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/dmm4x4_floorplan.json
            -DEXPECT_CODE=0
            -DMUST_MATCH=generated:dmm_mesh
            -P ${_expect}
    )
    # A fixture, not DEPENDS: `ctest -R kpu_floorplan_dmm4x4_import` alone still writes what it reads.
    set_tests_properties(kpu_floorplan_dmm4x4 PROPERTIES FIXTURES_SETUP dmm4x4_floorplan)
    set_tests_properties(kpu_floorplan_dmm4x4_import PROPERTIES FIXTURES_REQUIRED dmm4x4_floorplan)

    # The shared structure record (#392; docs/plans/shared-structure-record.md): the model's T4
    # floorplan converted and checked; the checker's self-test, over the samples from both
    # producers (the model's, and the RTL generator's converted from its floorplan.json v1).
    find_program(_struct_py NAMES python3 python)
    if(_struct_py)
        set(_trace ${CMAKE_SOURCE_DIR}/tools/trace)
        # Fixtures, not DEPENDS: `ctest -R structure_check_t4` alone still writes what it reads.
        set_tests_properties(kpu_floorplan_t4 PROPERTIES FIXTURES_SETUP t4_floorplan)
        add_test(NAME structure_record_t4
            COMMAND ${_struct_py} ${_trace}/structure_from_floorplan.py ${CMAKE_CURRENT_BINARY_DIR}/t4_floorplan.json
                    -o ${CMAKE_CURRENT_BINARY_DIR}/t4.structure.json)
        set_tests_properties(structure_record_t4 PROPERTIES FIXTURES_REQUIRED t4_floorplan FIXTURES_SETUP t4_structure)
        add_test(NAME structure_check_t4
            COMMAND ${_struct_py} ${_trace}/structure_check.py ${CMAKE_CURRENT_BINARY_DIR}/t4.structure.json)
        set_tests_properties(structure_check_t4 PROPERTIES FIXTURES_REQUIRED t4_structure)
        # kpu-floorplan --structure writes the record natively: it checks clean, and it is the
        # converted record block for block (structure_diff.py; only the provenance differs).
        add_test(NAME structure_check_native_t4
            COMMAND ${_struct_py} ${_trace}/structure_check.py ${CMAKE_CURRENT_BINARY_DIR}/t4_native.structure.json)
        set_tests_properties(structure_check_native_t4 PROPERTIES FIXTURES_REQUIRED t4_floorplan)
        add_test(NAME structure_native_t4
            COMMAND ${_struct_py} ${_trace}/structure_diff.py ${CMAKE_CURRENT_BINARY_DIR}/t4_native.structure.json
                    ${CMAKE_CURRENT_BINARY_DIR}/t4.structure.json)
        set_tests_properties(structure_native_t4 PROPERTIES FIXTURES_REQUIRED "t4_floorplan;t4_structure")
        # The mesh's record: native and converted agree, and both check clean.
        add_test(NAME structure_record_dmm4x4
            COMMAND ${_struct_py} ${_trace}/structure_from_floorplan.py ${CMAKE_CURRENT_BINARY_DIR}/dmm4x4_floorplan.json
                    -o ${CMAKE_CURRENT_BINARY_DIR}/dmm4x4.structure.json)
        set_tests_properties(structure_record_dmm4x4 PROPERTIES FIXTURES_REQUIRED dmm4x4_floorplan FIXTURES_SETUP dmm4x4_structure)
        add_test(NAME structure_check_native_dmm4x4
            COMMAND ${_struct_py} ${_trace}/structure_check.py ${CMAKE_CURRENT_BINARY_DIR}/dmm4x4_native.structure.json)
        set_tests_properties(structure_check_native_dmm4x4 PROPERTIES FIXTURES_REQUIRED dmm4x4_floorplan)
        add_test(NAME structure_native_dmm4x4
            COMMAND ${_struct_py} ${_trace}/structure_diff.py ${CMAKE_CURRENT_BINARY_DIR}/dmm4x4_native.structure.json
                    ${CMAKE_CURRENT_BINARY_DIR}/dmm4x4.structure.json)
        set_tests_properties(structure_native_dmm4x4 PROPERTIES FIXTURES_REQUIRED "dmm4x4_floorplan;dmm4x4_structure")
        # The converter writes a die-local record, wherever the die sits.
        add_test(NAME structure_from_floorplan_die_local
            COMMAND ${_struct_py} ${_trace}/test_structure_from_floorplan.py ${CMAKE_CURRENT_BINARY_DIR}/dmm4x4_floorplan.json)
        set_tests_properties(structure_from_floorplan_die_local PROPERTIES FIXTURES_REQUIRED dmm4x4_floorplan)
        set_tests_properties(structure_record_dmm4x4 structure_check_native_dmm4x4 structure_native_dmm4x4
            structure_from_floorplan_die_local PROPERTIES LABELS "program;records;structure;mesh")
        add_test(NAME structure_check_selftest
            COMMAND ${_struct_py} ${_trace}/test_structure_check.py ${CMAKE_SOURCE_DIR}/tests/program/structure)
        # The floorplan view (#447): pack.py over every sample; with Playwright, the page itself.
        add_test(NAME floorplan_viewer_smoke
            COMMAND ${_struct_py} ${CMAKE_SOURCE_DIR}/tools/visualization/floorplan/test_viewer.py
                    ${CMAKE_SOURCE_DIR}/tests/program/structure)
        set_tests_properties(structure_record_t4 structure_check_t4 structure_check_native_t4 structure_native_t4
            structure_check_selftest floorplan_viewer_smoke PROPERTIES LABELS "program;records;structure")
    endif()

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
        kpu_floorplan_dmm4x4 kpu_floorplan_dmm4x4_import
        PROPERTIES LABELS "program;platform;floorplan")
endif()
