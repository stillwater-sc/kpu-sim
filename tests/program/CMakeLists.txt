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
)

# TRANSACTIONAL tier executor (event engine + firing rule)
kpu_add_component_test(test_tile_transaction_executor "program"
    test_tile_transaction_executor.cpp
)

set_tests_properties(test_tile_transaction_executor PROPERTIES
    LABELS "program;transactional;executor;v0.9"
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
)

# The call ABI as MMIO: rings, doorbell, status registers, the payload-free bus (#305 inc. 3)
kpu_add_component_test(test_orchestration_mmio "program"
    test_orchestration_mmio.cpp
)

set_tests_properties(test_orchestration_mmio PROPERTIES
    LABELS "program;orchestration;v0.9"
)

# Reserve-then-launch: R1-R4, the freedom, the wedge, and sufficiency (#305 inc. 3)
kpu_add_component_test(test_reservation "program"
    test_reservation.cpp
)

set_tests_properties(test_reservation PROPERTIES
    LABELS "program;orchestration;v0.9"
)

# The tile-flow record and its .tflow bundle (#286 step 2)
kpu_add_component_test(test_tile_flow_record "program"
    test_tile_flow_record.cpp
)

set_tests_properties(test_tile_flow_record PROPERTIES
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
)
