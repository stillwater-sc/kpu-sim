# ============================================================================
# tests/harness/CMakeLists.txt
# Tests for KPU component harness infrastructure
# ============================================================================

# DMA harness tests
add_executable(test_dma_harness test_dma_harness.cpp)
target_link_libraries(test_dma_harness PRIVATE
    kpu_harness
    kpu_behavioral
    Catch2::Catch2WithMain
)
set_target_properties(test_dma_harness PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/Harness"
)
add_test(NAME dma_harness_tests COMMAND test_dma_harness)

# BlockMover harness tests
add_executable(test_block_mover_harness test_block_mover_harness.cpp)
target_link_libraries(test_block_mover_harness PRIVATE
    kpu_harness
    kpu_behavioral
    Catch2::Catch2WithMain
)
set_target_properties(test_block_mover_harness PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/Harness"
)
add_test(NAME block_mover_harness_tests COMMAND test_block_mover_harness)

# Streamer harness tests
add_executable(test_streamer_harness test_streamer_harness.cpp)
target_link_libraries(test_streamer_harness PRIVATE
    kpu_harness
    kpu_behavioral
    Catch2::Catch2WithMain
)
set_target_properties(test_streamer_harness PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/Harness"
)
add_test(NAME streamer_harness_tests COMMAND test_streamer_harness)

# Pipeline harness tests
add_executable(test_pipeline_harness test_pipeline_harness.cpp)
target_link_libraries(test_pipeline_harness PRIVATE
    kpu_harness
    kpu_behavioral
    Catch2::Catch2WithMain
)
set_target_properties(test_pipeline_harness PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/Harness"
)
add_test(NAME pipeline_harness_tests COMMAND test_pipeline_harness)

# Schedule validator tests
add_executable(test_schedule_validator test_schedule_validator.cpp)
target_link_libraries(test_schedule_validator PRIVATE
    kpu_harness
    kpu_behavioral
    Catch2::Catch2WithMain
)
set_target_properties(test_schedule_validator PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/Harness"
)
add_test(NAME schedule_validator_tests COMMAND test_schedule_validator)
