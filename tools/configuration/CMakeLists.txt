# ============================================================================
# tools/configuration/CMakeLists.txt
# KPU Configuration Tools
# ============================================================================

# KPU Config Tool (legacy - hardware configuration)
kpu_add_tool(config
    FOLDER "Configuration"
    DESCRIPTION "KPU configuration and validation tool"
    SOURCES
        kpu-config/kpu_config.cpp
    LIBS
        nlohmann_json::nlohmann_json
)

# KPU Fidelity Config Tool (multi-fidelity simulation framework)
add_executable(kpu-fidelity-config
    kpu-config.cpp
)

target_link_libraries(kpu-fidelity-config PRIVATE
    kpu_config
    nlohmann_json::nlohmann_json
)

target_include_directories(kpu-fidelity-config PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(kpu-fidelity-config PRIVATE cxx_std_17)

set_target_properties(kpu-fidelity-config PROPERTIES
    FOLDER "Tools/Configuration"
)

install(TARGETS kpu-fidelity-config
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Tools
)
