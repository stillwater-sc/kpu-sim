# ============================================================================
# tools/configuration/CMakeLists.txt
# KPU Configuration Tools
# ============================================================================

# KPU Config Tool (multi-fidelity simulation framework)
add_executable(kpu-config
    kpu-config.cpp
)

target_link_libraries(kpu-config PRIVATE
    kpu_config
    nlohmann_json::nlohmann_json
)

target_include_directories(kpu-config PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(kpu-config PRIVATE cxx_std_17)

set_target_properties(kpu-config PROPERTIES
    FOLDER "Tools/Configuration"
)

install(TARGETS kpu-config
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Tools
)
