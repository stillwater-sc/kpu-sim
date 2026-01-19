# tests/xue/CMakeLists.txt
# XUE (eXecution Usage Events) tests

# XUE test suite
add_executable(xue_tests test_xue.cpp)
target_link_libraries(xue_tests PRIVATE
    Catch2::Catch2WithMain
    kpu_simulator
)
kpu_set_target_options(xue_tests)
set_target_properties(xue_tests PROPERTIES FOLDER "Tests/XUE")

# Register with CTest
add_test(NAME xue_tests COMMAND xue_tests)
set_tests_properties(xue_tests PROPERTIES LABELS "xue")
