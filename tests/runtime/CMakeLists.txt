# Runtime Tests
# Tests for KPURuntime and GraphExecutor

# Helper function that also links runtime library
function(kpu_add_runtime_test test_name)
    add_executable(${test_name} ${ARGN})

    target_link_libraries(${test_name}
        PRIVATE
        kpu_runtime
        Catch2::Catch2WithMain
    )

    set_target_properties(${test_name} PROPERTIES FOLDER "Tests/runtime")

    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS runtime)

    install(TARGETS ${test_name}
        DESTINATION bin/tests
        OPTIONAL
    )
endfunction()

# Runtime tests
kpu_add_runtime_test(runtime_test
    test_runtime.cpp
)

# Executor tests
kpu_add_runtime_test(executor_test
    test_executor.cpp
)

# Graph tests (multi-layer execution)
kpu_add_runtime_test(graph_test
    test_graph.cpp
)

# Behavioral memory model tests
# Note: Links against kpu_behavioral_components directly since
# behavioral models are in sw::kpu::behavioral, not sw::runtime
add_executable(behavioral_memory_test test_behavioral_memory.cpp)
target_link_libraries(behavioral_memory_test
    PRIVATE
    kpu_behavioral_components
    Catch2::Catch2WithMain
)
set_target_properties(behavioral_memory_test PROPERTIES FOLDER "Tests/runtime")
add_test(NAME behavioral_memory_test COMMAND behavioral_memory_test)
set_tests_properties(behavioral_memory_test PROPERTIES LABELS "runtime;behavioral")
