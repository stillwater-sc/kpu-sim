# ============================================================================
# examples/dataflow/CMakeLists.txt
# Domain-flow / SURE demonstrations
# ============================================================================

# Fused batched-MLP SURE demo: builds the fused matmul+bias+activation operator
# as a single SURE using domain_flow's standalone primitives (option B3).
# Only meaningful when domain_flow is available.
add_executable(fused_mlp_sure_demo fused_mlp_sure_demo.cpp)

# Thin client of the reusable FusedMlpSure library (kpu_dataflow). The library's
# public header is domain_flow-free, so the demo needs no domain_flow setup.
target_link_libraries(fused_mlp_sure_demo PRIVATE kpu_dataflow)

set_target_properties(fused_mlp_sure_demo PROPERTIES FOLDER "Examples/Dataflow")

# Self-validating (exits non-zero on mismatch) — register as a CTest.
if(KPU_BUILD_TESTS)
    add_test(NAME fused_mlp_sure_demo COMMAND fused_mlp_sure_demo)
    set_tests_properties(fused_mlp_sure_demo PROPERTIES LABELS "dataflow;fusion")
endif()
