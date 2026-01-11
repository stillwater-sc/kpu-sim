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

# Add tests
add_test(NAME behavioral_matmul COMMAND matmul_behavioral)
add_test(NAME behavioral_xor COMMAND xor_behavioral)
