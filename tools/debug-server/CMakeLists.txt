# ============================================================================
# tools/debug-server/CMakeLists.txt
# The read-only debug server (docs/plans/cycle-mflow-and-debug-server.md step 7): Python, standard
# library only; nothing to build. Its tests start it on a free loopback port.
# ============================================================================
find_program(KPU_PYTHON3 NAMES python3 python)
if(KPU_PYTHON3)
    add_test(NAME kpu_debug_server
        COMMAND ${KPU_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/test_kpu_debug_server.py)
    set_tests_properties(kpu_debug_server PROPERTIES LABELS "tools;debug-server")
endif()
