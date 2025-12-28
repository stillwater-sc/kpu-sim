# ============================================================================
# examples/blas/CMakeLists.txt
# BLAS-level examples (distributed matrix operations)

kpu_add_example(big_matmul
    FOLDER "blas"
    DESCRIPTION "Distributed matmul on 4x4 checkerboard (16 compute tiles)"
    SOURCES big_matmul.cpp
)
