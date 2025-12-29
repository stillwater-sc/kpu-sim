# ============================================================================
# examples/blas/CMakeLists.txt
# BLAS-level examples (distributed matrix operations)

kpu_add_example(big_matmul
    FOLDER "blas"
    DESCRIPTION "Distributed matmul on 4x4 checkerboard (16 compute tiles)"
    SOURCES big_matmul.cpp
)

kpu_add_example(block_systolic_matmul
    FOLDER "blas"
    DESCRIPTION "Block systolic matmul using stateful BlockMovers and TileDataFlowGraph"
    SOURCES block_systolic_matmul.cpp
)

# Link block systolic matmul against dataflow and datamovement libraries
target_link_libraries(example_block_systolic_matmul PRIVATE
    kpu_dataflow
    kpu_datamovement_components
)
