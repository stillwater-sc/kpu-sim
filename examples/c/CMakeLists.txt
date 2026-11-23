# ============================================================================
# examples/c/CMakeLists.txt
# C API example applications
#
# These examples demonstrate how to use the KPU simulator from pure C code.

# hello_kpu - Basic simulator creation and configuration
add_executable(hello_kpu hello_kpu.c)
target_link_libraries(hello_kpu PRIVATE StillwaterKPU::C_API)
set_target_properties(hello_kpu PROPERTIES
    FOLDER "Examples/C"
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# hello_kpu exits non-zero when a step fails or the host <-> device round trip loses data,
# so it is a smoke test of the C API (#497). It checks no launch values: the runtime is
# timing only (#257).
if(KPU_BUILD_TESTS)
    add_test(NAME hello_kpu_c COMMAND hello_kpu)
    set_tests_properties(hello_kpu_c PROPERTIES TIMEOUT 60)   # it runs in well under a second
    if(WIN32)
        # The C API is a DLL, built in its own directory, and Windows looks for a DLL beside
        # the exe or on PATH. Without this the test hung on CI until its timeout. ENVIRONMENT
        # is a list, so PATH's own separators are escaped.
        string(REPLACE ";" "\\;" _kpu_c_system_path "$ENV{PATH}")
        set_tests_properties(hello_kpu_c PROPERTIES
            ENVIRONMENT "PATH=$<SHELL_PATH:$<TARGET_FILE_DIR:kpu_c_api>>\\;${_kpu_c_system_path}"
        )
    endif()
endif()

# runtime_demo - Full runtime API demonstration
add_executable(runtime_demo_c runtime_demo.c)
target_link_libraries(runtime_demo_c PRIVATE StillwaterKPU::C_API)
set_target_properties(runtime_demo_c PROPERTIES
    FOLDER "Examples/C"
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Install examples
install(TARGETS hello_kpu runtime_demo_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Examples
)

install(FILES
    hello_kpu.c
    runtime_demo.c
    DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/c
    COMPONENT Examples
)
