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
kpu_add_runtime_test(behavioral_memory_test
    test_behavioral_memory.cpp
)
