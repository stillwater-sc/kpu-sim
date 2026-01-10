# MLP Examples
# Demonstrates multi-layer perceptron execution on KPU simulator

# Helper function for MLP examples
function(kpu_add_mlp_example example_name)
    add_executable(${example_name} ${ARGN})

    target_link_libraries(${example_name}
        PRIVATE
        kpu_runtime
    )

    set_target_properties(${example_name} PROPERTIES
        FOLDER "Examples/MLP"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/mlp"
    )

    install(TARGETS ${example_name}
        DESTINATION bin/examples
        OPTIONAL
    )
endfunction()

# XOR Classifier - Classic 2-layer MLP solving XOR problem
kpu_add_mlp_example(xor_classifier
    xor_classifier.cpp
)

# Sine Approximation - Function approximation with 3-layer MLP
kpu_add_mlp_example(sine_approximation
    sine_approximation.cpp
)

# MNIST Classifier - Digit recognition with 3-layer MLP
kpu_add_mlp_example(mnist_classifier
    mnist_classifier.cpp
)

# Install source files for reference
install(FILES
    xor_classifier.cpp
    sine_approximation.cpp
    mnist_classifier.cpp
    DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/mlp
    COMPONENT Examples
)
