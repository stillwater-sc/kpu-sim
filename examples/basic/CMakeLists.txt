# ============================================================================
# examples/basic/CMakeLists.txt
# Basic examples

function(kpu_add_example name)
    set(options "")
    set(oneValueArgs FOLDER DESCRIPTION)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(EXAMPLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    add_executable(example_${name} ${EXAMPLE_SOURCES})
    
    target_link_libraries(example_${name} PRIVATE
        StillwaterKPU::Simulator
        ${EXAMPLE_LIBS}
    )
    
    kpu_set_target_options(example_${name})
    
    set_target_properties(example_${name} PROPERTIES
        FOLDER "Examples/${EXAMPLE_FOLDER}"
    )
    
    install(TARGETS example_${name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/bin
        COMPONENT Examples
    )
    
    # Install source code for reference
    install(FILES ${EXAMPLE_SOURCES}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/${EXAMPLE_FOLDER}
        COMPONENT Examples
    )
endfunction()

kpu_add_example(hello_kpu
    FOLDER "basic"
    DESCRIPTION "Simple first KPU program"
    SOURCES hello_kpu.cpp
)

kpu_add_example(data_movement_pipeline
    FOLDER "basic"
    DESCRIPTION "Full data movement pipeline (direct component API)"
    SOURCES data_movement_pipeline.cpp
)

kpu_add_example(data_movement_isa_matmul
    FOLDER "basic"
    DESCRIPTION "Matrix multiplication using Data Movement ISA"
    SOURCES data_movement_isa_matmul.cpp
)

# Link ISA matmul example against ISA library
target_link_libraries(example_data_movement_isa_matmul PRIVATE
    kpu_isa
)

kpu_add_example(matrix_multiply
    FOLDER "basic"
    DESCRIPTION "Basic matrix multiplication"
    SOURCES matrix_multiply.cpp
)

kpu_add_example(memory_management
    FOLDER "basic"
    DESCRIPTION "Memory usage patterns"
    SOURCES memory_management.cpp
)

kpu_add_example(system_config_demo
    FOLDER "basic"
    DESCRIPTION "JSON-based system configuration"
    SOURCES system_config_demo.cpp
)

# Link system config demo against system library
target_link_libraries(example_system_config_demo PRIVATE
    StillwaterKPU::System
)

kpu_add_example(config_formatter_demo
    FOLDER "basic"
    DESCRIPTION "Config formatting and memory map API demonstration"
    SOURCES config_formatter_demo.cpp
)

# Link config formatter demo against system library
target_link_libraries(example_config_formatter_demo PRIVATE
    StillwaterKPU::System
)

kpu_add_example(concurrent_execution_debug
    FOLDER "basic"
    DESCRIPTION "Debug tool for concurrent execution model"
    SOURCES concurrent_execution_debug.cpp
)

# Link concurrent execution debug against ISA library
target_link_libraries(example_concurrent_execution_debug PRIVATE
    kpu_isa
)

kpu_add_example(tile_layout_test
    FOLDER "basic"
    DESCRIPTION "Test and compare tile layout policies"
    SOURCES tile_layout_test.cpp
)

# Link tile layout test against ISA library
target_link_libraries(example_tile_layout_test PRIVATE
    kpu_isa
)

kpu_add_example(resource_api_demo
    FOLDER "basic"
    DESCRIPTION "Resource API demonstration: discovery, allocation, status, statistics"
    SOURCES resource_api_demo.cpp
)

kpu_add_example(kernel_demo
    FOLDER "basic"
    DESCRIPTION "Kernel abstraction and KernelCompiler demonstration"
    SOURCES kernel_demo.cpp
)

# Link kernel demo against ISA library (for ConcurrentExecutor)
target_link_libraries(example_kernel_demo PRIVATE
    kpu_isa
)
