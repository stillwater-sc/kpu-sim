# tests/quantization/CMakeLists.txt
# Quantization infrastructure tests

# Quantization type tests
add_executable(quantization_tests test_quantization_types.cpp)
target_link_libraries(quantization_tests PRIVATE
    Catch2::Catch2WithMain
    kpu_simulator
)

# Add Universal library include path if available
if(UNIVERSAL_INCLUDE_DIR)
    target_include_directories(quantization_tests PRIVATE ${UNIVERSAL_INCLUDE_DIR})
endif()

kpu_set_target_options(quantization_tests)
set_target_properties(quantization_tests PROPERTIES FOLDER "Tests/Quantization")

# Register with CTest
add_test(NAME quantization_tests COMMAND quantization_tests)
set_tests_properties(quantization_tests PROPERTIES LABELS "quantization")
