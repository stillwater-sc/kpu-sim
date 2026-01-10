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
