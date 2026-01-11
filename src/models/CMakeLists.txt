# ============================================================================
# Simulation Models
# Multi-fidelity hardware simulation models
#
#   behavioral/    - Functional simulation (fast, no timing)
#   transactional/ - Statistical simulation (approximate timing)
#   temporal/      - Cycle-accurate simulation (precise timing)
# ============================================================================

add_subdirectory(behavioral)
add_subdirectory(transactional)
add_subdirectory(temporal)
