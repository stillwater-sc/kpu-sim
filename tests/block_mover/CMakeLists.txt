# Block Mover Tests

# Basic BlockMover functionality tests
kpu_add_component_test(block_mover_basic_test "iddo"
    test_block_mover_basic.cpp
)

# BlockMover tracing tests
kpu_add_component_test(block_mover_tracing_test "iddo"
    test_block_mover_tracing.cpp
)

# Stateful BlockMover and Data Flow Graph tests
add_executable(stateful_block_mover_test
    test_stateful_block_mover.cpp
)

target_include_directories(stateful_block_mover_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(stateful_block_mover_test
    kpu_dataflow
    kpu_datamovement_components
    Catch2::Catch2WithMain
)

set_target_properties(stateful_block_mover_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_test(NAME StatefulBlockMoverTest COMMAND stateful_block_mover_test)