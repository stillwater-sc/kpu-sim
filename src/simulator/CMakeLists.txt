# ============================================================================
# src/simulator/CMakeLists.txt
# Main simulator implementation

set(KPU_SIMULATOR_SOURCES
    kpu_simulator.cpp
    resource_manager.cpp
    kernel.cpp
    kernel_serializer.cpp
    kernel_graph.cpp
    kpu_config_loader.cpp
#    device_manager.cpp
#    instruction_set.cpp
#    execution_engine.cpp
#    trace_logger.cpp
)

# Find required system packages
find_package(Threads REQUIRED)    # this will create the Threads::Threads target

add_library(kpu_simulator ${KPU_SIMULATOR_SOURCES})
add_library(StillwaterKPU::Simulator ALIAS kpu_simulator)

target_include_directories(kpu_simulator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# PUBLIC dependencies get exported and must be available to consumers of your library
# PRIVATE dependencies are only used internally and don't get exported
# External dependencies should typically be PRIVATE unless your public API directly exposes them

target_link_libraries(kpu_simulator
    PUBLIC
        kpu_components          # New component-based architecture
        kpu_compiler            # Kernel compilation (TileOptimizer, etc.)
        kpu_isa                 # ISA (DMProgram, OutputStationaryProgramBuilder)
    PRIVATE
        $<TARGET_NAME_IF_EXISTS:Threads::Threads>
        $<TARGET_NAME_IF_EXISTS:fmt::fmt>
        $<TARGET_NAME_IF_EXISTS:spdlog::spdlog>
        $<TARGET_NAME_IF_EXISTS:nlohmann_json::nlohmann_json>
)

kpu_set_target_options(kpu_simulator)

set_target_properties(kpu_simulator PROPERTIES
    OUTPUT_NAME stillwater_kpu_simulator
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    FOLDER "src/Core"
    POSITION_INDEPENDENT_CODE ON  # Enable PIC for shared library compatibility
)

# Installation
install(TARGETS kpu_simulator
    # EXPORT kpu_simulator_targets
    LIBRARY DESTINATION lib
        COMPONENT Runtime
    ARCHIVE DESTINATION lib
        COMPONENT Development
)