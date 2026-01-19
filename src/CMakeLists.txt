# ============================================================================
# src/CMakeLists.txt
# KPU Simulator Source Organization
#
# Directory Structure:
#   dfx/       - DFX parser for Python kpu package integration
#   models/    - Multi-fidelity simulation models (behavioral/transactional/temporal)
#   software/  - Host-side software (runtime, compiler, ISA)
#   system/    - System-level integration
#   tools/     - Development tools (bindings, benchmarks, trace)
#   xue/       - XUE (eXecution Unit Events) observability framework
# ============================================================================

# DFX parser (Python kpu package format)
add_subdirectory(dfx)

# Simulation models (by fidelity level)
add_subdirectory(models)

# Host-side software
add_subdirectory(software)

# System integration
add_subdirectory(system)

# Development tools
add_subdirectory(tools)
