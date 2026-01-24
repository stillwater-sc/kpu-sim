# tests/validation/CMakeLists.txt
# Analytical validation tests for KPU simulator (v0.3.5)
#
# These tests validate simulator behavior against known analytical solutions.

# Analytical validation tests
add_executable(test_analytical_validation test_analytical_validation.cpp)
target_link_libraries(test_analytical_validation PRIVATE
    Catch2::Catch2WithMain
    StillwaterKPU::Benchmark
)
kpu_set_target_options(test_analytical_validation)
set_target_properties(test_analytical_validation PROPERTIES FOLDER "Validation")

# Register with CTest
add_test(NAME analytical_validation COMMAND test_analytical_validation)
set_tests_properties(analytical_validation PROPERTIES
    LABELS "validation;analytical"
    TIMEOUT 120
)

# Quick validation subset (for CI) - tests FLOP counts and AI calculations
add_test(NAME analytical_validation_quick
    COMMAND test_analytical_validation "[flops],[ai]"
)
set_tests_properties(analytical_validation_quick PROPERTIES
    LABELS "validation;quick"
    TIMEOUT 30
)
