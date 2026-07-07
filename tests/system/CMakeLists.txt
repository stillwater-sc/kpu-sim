# ============================================================================
# tests/system/CMakeLists.txt
# System orchestration tests

kpu_add_component_test(test_toplevel_simulator "system" test_toplevel_simulator.cpp)

# Link against the system library
target_link_libraries(test_toplevel_simulator
    PRIVATE
    StillwaterKPU::System
)

# System configuration tests
kpu_add_component_test(test_system_config "system" test_system_config.cpp)

target_link_libraries(test_system_config
    PRIVATE
    StillwaterKPU::System
)

# Config formatting tests
kpu_add_component_test(test_config_formatting "system" test_config_formatting.cpp)

target_link_libraries(test_config_formatting
    PRIVATE
    StillwaterKPU::System
)

# KPU SoC configuration report tests
kpu_add_component_test(test_config_report "system" test_config_report.cpp)