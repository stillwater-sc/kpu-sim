# ============================================================================
# tests/CMakeLists.txt
# Test suite configuration

# Test configuration

# Find or fetch a testing framework (e.g., Catch2)
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Helper function for KPU component tests - automatically links KPU libraries
function(kpu_add_component_test test_name test_category)
    # Create the test executable
    add_executable(${test_name} ${ARGN})
    
    # Link against our core library and Catch2
    target_link_libraries(${test_name}
        PRIVATE
        kpu_simulator       # Includes all component libraries
        Catch2::Catch2WithMain
    )
    
    # Set folder organization in Solution Explorer
    set_target_properties(${test_name} PROPERTIES FOLDER "Tests/${test_category}")
    
    # Add the test to CTest with category label
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS ${test_category})
    
    # Install test executable (optional)
    install(TARGETS ${test_name} 
        DESTINATION bin/tests
        OPTIONAL
    )
endfunction()

# Add test subdirectories - each manages its own tests
add_subdirectory(system)
add_subdirectory(driver)
add_subdirectory(memory)
add_subdirectory(trace)
add_subdirectory(dma)
add_subdirectory(block_mover)
add_subdirectory(streamer)
add_subdirectory(compute)
add_subdirectory(storage)
add_subdirectory(compiler)
add_subdirectory(isa)
add_subdirectory(runtime)
add_subdirectory(integration)
#add_subdirectory(benchmarks)

# Legacy test (in this directory)
kpu_add_component_test(legacy_test_main "legacy" test_main.cpp)

# Custom test targets for different categories
# These targets are defined here because they orchestrate across all subdirectories

# Usage: make test_memory, make test_dma, etc.
# Each target runs tests with the corresponding label
# ctest -L memory
# ctest -L "dma|memory"  # multiple labels
# ctest -LE "benchmarks"  # exclude label
add_custom_target(test_system 
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L system
    COMMENT "Running system orchestration tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_system PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_driver 
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L driver
    COMMENT "Running resource manager driver tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_driver PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_memory
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L memory
    COMMENT "Running memory tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_memory PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_trace
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L trace
    COMMENT "Running trace system tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_trace PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_dma
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L dma
    COMMENT "Running DMA tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_dma PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_block_mover
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L block_mover
    COMMENT "Running block mover tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_block_mover PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_streamer
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L streamer
    COMMENT "Running streamer tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_streamer PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_compute
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L compute
    COMMENT "Running compute engine tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_compute PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_iddo
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L iddo
    COMMENT "Running IDDO workflow tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_iddo PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_eddo
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L eddo
    COMMENT "Running EDDO workflow tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_eddo PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_fabric
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L fabric
    COMMENT "Running compute fabric tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_fabric PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_integration
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L integration
    COMMENT "Running integration tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_integration PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_runtime
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L runtime
    COMMENT "Running runtime tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_runtime PROPERTIES FOLDER "Tests/Runners")

#add_custom_target(test_benchmarks
#    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -L benchmark
#    COMMENT "Running benchmark tests"
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#    VERBATIM
#)
#set_target_properties(test_benchmarks PROPERTIES FOLDER "Tests/Runners")

add_custom_target(test_all_categories
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure
    COMMENT "Running all tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)
set_target_properties(test_all_categories PROPERTIES FOLDER "Tests/Runners")
