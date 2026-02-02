# verification/kernels/CMakeLists.txt
# Build verification executables for kernel correctness testing

function(kpu_add_verification_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE kpu_behavioral)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${name} PROPERTIES
        FOLDER "Verification/Kernels"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/verification/kernels"
    )
    install(TARGETS ${name} DESTINATION bin/verification OPTIONAL)
endfunction()

kpu_add_verification_test(verify_elementwise
    class0_elementwise/verify_elementwise.cpp)

kpu_add_verification_test(verify_matmul
    class1_dense_linear/verify_matmul.cpp)

kpu_add_verification_test(verify_fused_ops
    class1_dense_linear/verify_fused_ops.cpp)

# Class 2: Spatial Convolution
kpu_add_verification_test(verify_conv2d
    class2_spatial_conv/verify_conv2d.cpp)

kpu_add_verification_test(verify_pooling
    class2_spatial_conv/verify_pooling.cpp)

# Class 3: Multi-Branch and Residual Architectures
kpu_add_verification_test(verify_residual
    class3_multi_branch/verify_residual.cpp)

# Class 4: Depthwise Separable and Grouped Convolution
kpu_add_verification_test(verify_depthwise_conv
    class4_depthwise/verify_depthwise_conv.cpp)
