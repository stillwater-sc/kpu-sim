# Runtime Examples
# Demonstrates the KPURuntime and GraphExecutor APIs

# Helper function for runtime examples
function(kpu_add_runtime_example example_name)
    add_executable(${example_name} ${ARGN})

    target_link_libraries(${example_name}
        PRIVATE
        kpu_runtime
    )

    set_target_properties(${example_name} PROPERTIES FOLDER "Examples/runtime")

    install(TARGETS ${example_name}
        DESTINATION bin/examples
        OPTIONAL
    )
endfunction()

# Runtime demo - demonstrates full Runtime API
kpu_add_runtime_example(example_runtime_demo
    runtime_demo.cpp
)
