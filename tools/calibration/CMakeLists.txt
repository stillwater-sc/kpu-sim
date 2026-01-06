# ============================================================================
# tools/calibration/CMakeLists.txt
# KPU Memory Controller Calibration Tools
# ============================================================================

# KPU Calibrate Tool - runs CA simulation and extracts calibration parameters
add_executable(kpu-calibrate
    kpu-calibrate.cpp
)

target_link_libraries(kpu-calibrate PRIVATE
    kpu_calibration
    kpu_memory_components
    nlohmann_json::nlohmann_json
)

target_include_directories(kpu-calibrate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(kpu-calibrate PRIVATE cxx_std_17)

set_target_properties(kpu-calibrate PROPERTIES
    FOLDER "Tools/Calibration"
)

install(TARGETS kpu-calibrate
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Tools
)

# KPU Validate Tool - cross-validates calibration across fidelity levels
add_executable(kpu-validate
    kpu-validate.cpp
)

target_link_libraries(kpu-validate PRIVATE
    kpu_calibration
    kpu_memory_components
    nlohmann_json::nlohmann_json
)

target_include_directories(kpu-validate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(kpu-validate PRIVATE cxx_std_17)

set_target_properties(kpu-validate PROPERTIES
    FOLDER "Tools/Calibration"
)

install(TARGETS kpu-validate
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Tools
)
