# tests/validation/CMakeLists.txt
# Validation tests for KPU simulator (v0.3.5, v0.3.6)
#
# These tests validate simulator behavior against known analytical solutions
# and XUE prediction accuracy.

# Analytical validation tests (v0.3.5)
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

# XUE prediction accuracy tests (v0.3.6)
add_executable(test_xue_prediction_accuracy test_xue_prediction_accuracy.cpp)
target_link_libraries(test_xue_prediction_accuracy PRIVATE
    Catch2::Catch2WithMain
    StillwaterKPU::Benchmark
)
kpu_set_target_options(test_xue_prediction_accuracy)
set_target_properties(test_xue_prediction_accuracy PROPERTIES FOLDER "Validation")

add_test(NAME xue_prediction_accuracy COMMAND test_xue_prediction_accuracy)
set_tests_properties(xue_prediction_accuracy PROPERTIES
    LABELS "validation;xue;accuracy"
    TIMEOUT 120
)

# Quick XUE validation (for CI)
add_test(NAME xue_accuracy_quick
    COMMAND test_xue_prediction_accuracy "[hardware],[classification]"
)
set_tests_properties(xue_accuracy_quick PROPERTIES
    LABELS "validation;xue;quick"
    TIMEOUT 30
)
