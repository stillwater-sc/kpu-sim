# ============================================================================
# tests/program/CMakeLists.txt
# L0 portable-program layer (tile/stream program) tests
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
# ============================================================================

# TileProgram (L0 tile-sequence) + functional reference
kpu_add_component_test(test_tile_program "program"
    test_tile_program.cpp
)

set_tests_properties(test_tile_program PROPERTIES
    LABELS "program;tile_program;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}     # reads tests/program/csp/*.csp
)

# Functional checks on tile contents: the format interpretation layer and the proximity bound
kpu_add_component_test(test_tile_check "program"
    test_tile_check.cpp
)

set_tests_properties(test_tile_check PROPERTIES
    LABELS "program;tile_program;v0.9"
)

# The CSP program's compiled form, run behaviorally (docs/plans/csp-program-tile-sequencing.md step 1)
kpu_add_component_test(test_csp_program "program"
    test_csp_program.cpp
)

set_tests_properties(test_csp_program PROPERTIES
    LABELS "program;csp;v0.9"
)

# The CSP language: written programs, the validator, print (docs/plans/csp-language.md step 1)
kpu_add_component_test(test_csp_language "program"
    test_csp_language.cpp
)

set_tests_properties(test_csp_language PROPERTIES
    LABELS "program;csp;lang;v0.9"
)

# L3 placement: every residency's address, allocated once by the walker (docs/plans/l3-placement-at-lb.md step 2)
kpu_add_component_test(test_csp_place "program"
    test_csp_place.cpp
)

set_tests_properties(test_csp_place PROPERTIES
    LABELS "program;csp;lang;placement;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# L-B executes placement: its L3 is addressed storage (docs/plans/l3-placement-at-lb.md step 3)
kpu_add_component_test(test_csp_lb_place "program"
    test_csp_lb_place.cpp
)

set_tests_properties(test_csp_lb_place PROPERTIES
    LABELS "program;csp;behavioral;placement;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The domain-flow record: the oracle's L-B run, one step per action (docs/plans/domain-flow-debugger.md step 2)
kpu_add_component_test(test_domain_flow_record "program"
    test_domain_flow_record.cpp
)

set_tests_properties(test_domain_flow_record PROPERTIES
    LABELS "program;csp;domainflow;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# One program on both sides: a CSP program as the KPU RTL's tile command stream (#397)
kpu_add_component_test(test_tile_commands "program"
    test_tile_commands.cpp
)

set_tests_properties(test_tile_commands PROPERTIES
    LABELS "program;csp;tilecmd;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The KPU RTL generator's machine as a deployment (#396): kpu_soc_2x2.json, one tile of the
# generator's kpu_2x2_systolic, read and run; and the converter keeps it in step with the vendored
# generator config.
kpu_add_component_test(test_kpu_soc_deployment "program"
    test_kpu_soc_deployment.cpp
)

set_tests_properties(test_kpu_soc_deployment PROPERTIES
    LABELS "program;deploy;kpu_soc;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

find_program(KPU_PYTHON3 NAMES python3 python)
if(KPU_PYTHON3)
    execute_process(COMMAND ${KPU_PYTHON3} -c "import yaml" RESULT_VARIABLE _kpu_no_yaml OUTPUT_QUIET ERROR_QUIET)
    if(_kpu_no_yaml EQUAL 0)
        add_test(NAME kpuarch_to_deployment_selftest
            COMMAND ${KPU_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/deploy/test_kpuarch_to_deployment.py
                    ${CMAKE_SOURCE_DIR}/tests/program/deploy/generator/kpu_2x2_systolic.full.yaml
                    ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_soc_2x2.json
                    ${CMAKE_SOURCE_DIR}/tests/program/deploy/kpu_soc_2x2.notes.txt)
        # Every deployment the converter writes, kpu-sim reads and runs (KPU_RUN).
        set_tests_properties(kpuarch_to_deployment_selftest PROPERTIES LABELS "program;deploy;kpu_soc"
            ENVIRONMENT "KPU_RUN=$<TARGET_FILE:run>")
    else()
        message(STATUS "kpuarch_to_deployment_selftest: skipped (python3 has no PyYAML)")
    endif()
endif()

# L-T1 executes placement: a credit is a slot (docs/plans/l3-placement-at-lb.md step 5)
kpu_add_component_test(test_csp_lt1_place "program"
    test_csp_lt1_place.cpp
)

set_tests_properties(test_csp_lt1_place PROPERTIES
    LABELS "program;csp;lt1;placement;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# L-CA executes placement: a credit is a slot (docs/plans/l3-placement-at-lb.md step 6, #399)
kpu_add_component_test(test_csp_lca_place "program"
    test_csp_lca_place.cpp
)

set_tests_properties(test_csp_lca_place PROPERTIES
    LABELS "program;csp;lca;placement;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# L-T1 times the fabric as a systolic pipeline (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 2)
kpu_add_component_test(test_fabric_pipeline "program"
    test_fabric_pipeline.cpp
)

set_tests_properties(test_fabric_pipeline PROPERTIES
    LABELS "program;csp;lt1;fabric;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# L2 staging: `stage ... in l2 { }` at L-B, in the trace and at every level
# (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 3)
kpu_add_component_test(test_csp_l2_staging "program"
    test_csp_l2_staging.cpp
)

set_tests_properties(test_csp_l2_staging PROPERTIES
    LABELS "program;csp;lang;l2;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# `store ... late`: an accumulator's store emitted after the next accumulator's calls
# (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 2b)
kpu_add_component_test(test_csp_store_late "program"
    test_csp_store_late.cpp
)

set_tests_properties(test_csp_store_late PROPERTIES
    LABELS "program;csp;lang;v0.9"
)

# An accumulator's init: bias, bias row, accumulate onto C, as one more edge stream (#406;
# docs/plans/accumulator-init-stream.md)
kpu_add_component_test(test_csp_acc_init "program"
    test_csp_acc_init.cpp
)

set_tests_properties(test_csp_acc_init PROPERTIES
    LABELS "program;csp;lang;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The typed value path: number systems from Universal, int8 x int8 -> int32 (#394;
# docs/plans/typed-value-path.md)
kpu_add_component_test(test_typed_values "program"
    test_typed_values.cpp
)

set_tests_properties(test_typed_values PROPERTIES
    LABELS "program;csp;numeric;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The structured CSP program: streamed and validated without unrolling (docs/plans/csp-language.md step 1c)
kpu_add_component_test(test_csp_structured "program"
    test_csp_structured.cpp
)

set_tests_properties(test_csp_structured PROPERTIES
    LABELS "program;csp;lang;v0.9"
)

# The linear operator: L0 epilogue ops, vector operands, tile contexts (docs/plans/csp-language.md step 2)
kpu_add_component_test(test_csp_linear "program"
    test_csp_linear.cpp
)

set_tests_properties(test_csp_linear PROPERTIES
    LABELS "program;csp;lang;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# csp-gen: the generator kpu-run's programs come from (docs/plans/kpu-run-csp-programs.md step 1)
kpu_add_component_test(test_csp_gen "program"
    test_csp_gen.cpp
)

set_tests_properties(test_csp_gen PROPERTIES
    LABELS "program;csp;gen;v0.9"
)

# Running a CSP program at each level (docs/plans/kpu-run-csp-programs.md step 2)
kpu_add_component_test(test_csp_run "program"
    test_csp_run.cpp
)

set_tests_properties(test_csp_run PROPERTIES
    LABELS "program;csp;run;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The CSP corpus through the platform, and the canonical text (kpu-run-csp-programs step 4a)
kpu_add_component_test(test_csp_corpus "program"
    test_csp_corpus.cpp
)

set_tests_properties(test_csp_corpus PROPERTIES
    LABELS "program;csp;corpus;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Stepping a CSP program, and its L-T1 timeline (kpu-run-csp-programs step 4c)
kpu_add_component_test(test_csp_step "program"
    test_csp_step.cpp
)

set_tests_properties(test_csp_step PROPERTIES
    LABELS "program;csp;step;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Residency across operators: inherit and retain (kpu-run-csp-programs step 4d)
kpu_add_component_test(test_csp_inherit_retain "program"
    test_csp_inherit_retain.cpp
)

set_tests_properties(test_csp_inherit_retain PROPERTIES
    LABELS "program;csp;inherit;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The L0 portable program as a file (#265)
kpu_add_component_test(test_l0_serialization "program"
    test_l0_serialization.cpp
)

set_tests_properties(test_l0_serialization PROPERTIES
    LABELS "program;serialize;v0.9"
)

# The L0 golden corpus: checked-in files that CI loads and executes (#265)
kpu_add_component_test(test_l0_corpus "program"
    test_l0_corpus.cpp
)

# Paths are relative to the project root, following tests/compiler's convention.
set_tests_properties(test_l0_corpus PROPERTIES
    LABELS "program;serialize;corpus;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The deciding orchestrator: descriptors, status, program-order acquisition (#305 inc. 2)
kpu_add_component_test(test_orchestrator "program"
    test_orchestrator.cpp
)

set_tests_properties(test_orchestrator PROPERTIES
    LABELS "program;orchestration;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}     # reads tests/program/csp/*.csp
)

# The call ABI as MMIO: rings, doorbell, status registers, the payload-free bus (#305 inc. 3)
kpu_add_component_test(test_orchestration_mmio "program"
    test_orchestration_mmio.cpp
)

set_tests_properties(test_orchestration_mmio PROPERTIES
    LABELS "program;orchestration;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}     # reads tests/program/csp/*.csp
)

# Reserve-then-launch: R1-R4, the freedom, and sufficiency (#305 inc. 3; CSP operators, 4d.2)
kpu_add_component_test(test_reservation "program"
    test_reservation.cpp
)

set_tests_properties(test_reservation PROPERTIES
    LABELS "program;orchestration;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}     # reads tests/program/csp/*.csp
)

# The tile-flow record and its .tflow bundle (#286 step 2)
kpu_add_component_test(test_tile_flow_record "program"
    test_tile_flow_record.cpp
)

set_tests_properties(test_tile_flow_record PROPERTIES
    LABELS "program;record;tileflow;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The NoC's hubs and port controllers in the deployment (docs/plans/noc-port-arbitration.md step 2)
kpu_add_component_test(test_noc_spec "program"
    test_noc_spec.cpp
)

set_tests_properties(test_noc_spec PROPERTIES
    LABELS "program;platform;noc;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The DRAM geometry and its address map (docs/plans/dram-bank-model.md step 1)
kpu_add_component_test(test_dram_address_map "program"
    test_dram_address_map.cpp
)

set_tests_properties(test_dram_address_map PROPERTIES
    LABELS "program;platform;dram;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The level-of-detail pyramid over a tile-flow record (#286 step 3)
kpu_add_component_test(test_tile_flow_lod "program"
    test_tile_flow_lod.cpp
)

set_tests_properties(test_tile_flow_lod PROPERTIES
    LABELS "program;record;tileflow;v0.9"
)

# The physical shape: array layout, folded-torus NoC, floorplan (#286 step 1)
kpu_add_component_test(test_floorplan "program"
    test_floorplan.cpp
)

set_tests_properties(test_floorplan PROPERTIES
    LABELS "program;platform;floorplan;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The KPU loadable container: the unit of deployment (#305 increment 1)
kpu_add_component_test(test_loadable "program"
    test_loadable.cpp
)

set_tests_properties(test_loadable PROPERTIES
    LABELS "program;loadable;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# A deployment is data: one machine description, JSON at its edge (#282)
kpu_add_component_test(test_deployment_spec "program"
    test_deployment_spec.cpp
)

# Paths are relative to the project root, following test_l0_corpus.
set_tests_properties(test_deployment_spec PROPERTIES
    LABELS "program;platform;deploy;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# The global naming map: every declared resource, addressable (#282 increment 3)
kpu_add_component_test(test_resource_map "program"
    test_resource_map.cpp
)

set_tests_properties(test_resource_map PROPERTIES
    LABELS "program;platform;names;v0.9"
)

# The platform owns the deployment and the state (#282 increment 2)
kpu_add_component_test(test_virtual_platform "program"
    test_virtual_platform.cpp
)

set_tests_properties(test_virtual_platform PROPERTIES
    LABELS "program;platform;v0.9"
)

# The driver seam: one function maps a level to an interpreter (#285)
kpu_add_component_test(test_driver_run_at "program"
    test_driver_run_at.cpp
)

set_tests_properties(test_driver_run_at PROPERTIES
    LABELS "program;driver;v0.9"
)

# L0 dependency model (tile + pivot-slot hazards, diagnosis)
kpu_add_component_test(test_tile_dependencies "program"
    test_tile_dependencies.cpp
)

set_tests_properties(test_tile_dependencies PROPERTIES
    LABELS "program;deps;v0.9"
)

# TileProgram characterization (tile-DAG concurrency + modeled metrics)
kpu_add_component_test(test_tile_characterize "program"
    test_tile_characterize.cpp
)

set_tests_properties(test_tile_characterize PROPERTIES
    LABELS "program;characterize;v0.9"
)

# L1 stream signatures (spatial/temporal layer over L0)
kpu_add_component_test(test_stream_signature "program"
    test_stream_signature.cpp
)

set_tests_properties(test_stream_signature PROPERTIES
    LABELS "program;stream;v0.9"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}     # reads tests/program/csp/*.csp
)
