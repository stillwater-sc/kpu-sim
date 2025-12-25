# ============================================================================
# examples/kernel/CMakeLists.txt
# Kernel-specific demonstration examples
#
# This directory contains examples demonstrating different kernel types:
# - matmul_kernel: Matrix multiplication kernel
# - mlp_kernel: MLP (matmul + bias + activation) kernel
# - conv2d_kernel: 2D convolution kernel (future)

# Helper function to add kernel examples
function(kpu_add_kernel_example name)
    set(options "")
    set(oneValueArgs DESCRIPTION)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(EXAMPLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${name}_kernel ${EXAMPLE_SOURCES})

    target_link_libraries(${name}_kernel PRIVATE
        StillwaterKPU::Simulator
        kpu_isa
        ${EXAMPLE_LIBS}
    )

    kpu_set_target_options(${name}_kernel)

    set_target_properties(${name}_kernel PROPERTIES
        FOLDER "Examples/Kernel"
    )

    install(TARGETS ${name}_kernel
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/bin
        COMPONENT Examples
    )

    # Install source code for reference
    install(FILES ${EXAMPLE_SOURCES}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/kernel
        COMPONENT Examples
    )
endfunction()

# Matrix Multiplication Kernel Demo
kpu_add_kernel_example(matmul
    DESCRIPTION "Matrix multiplication kernel demonstration"
    SOURCES matmul_kernel.cpp
)

# MLP Kernel Demo (matmul + bias + activation)
kpu_add_kernel_example(mlp
    DESCRIPTION "MLP kernel with fused bias and activation"
    SOURCES mlp_kernel.cpp
)
