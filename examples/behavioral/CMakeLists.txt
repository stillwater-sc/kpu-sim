# ============================================================================
# examples/behavioral/CMakeLists.txt
# Behavioral simulation examples
# ============================================================================

# Matmul example - tests basic matrix multiplication
add_executable(matmul_behavioral matmul_behavioral.cpp)
target_link_libraries(matmul_behavioral PRIVATE kpu_behavioral)
set_target_properties(matmul_behavioral PROPERTIES
    FOLDER "Examples/Behavioral"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/behavioral
)

# XOR classifier example - tests MLP execution
add_executable(xor_behavioral xor_behavioral.cpp)
target_link_libraries(xor_behavioral PRIVATE kpu_behavioral)
set_target_properties(xor_behavioral PROPERTIES
    FOLDER "Examples/Behavioral"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/behavioral
)

# OFG trace demo - demonstrates multi-level tracing with Chrome Trace export
add_executable(ofg_trace_demo ofg_trace_demo.cpp)
target_link_libraries(ofg_trace_demo PRIVATE kpu_behavioral)
set_target_properties(ofg_trace_demo PROPERTIES
    FOLDER "Examples/Behavioral"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/behavioral
)

# Tiled matmul trace generator - parameterized tiled matmul with timing
add_executable(tiled_matmul_trace tiled_matmul_trace.cpp)
target_link_libraries(tiled_matmul_trace PRIVATE kpu_behavioral)
set_target_properties(tiled_matmul_trace PROPERTIES
    FOLDER "Examples/Behavioral"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/behavioral
)

# Add tests
add_test(NAME behavioral_matmul COMMAND matmul_behavioral)
add_test(NAME behavioral_xor COMMAND xor_behavioral)
