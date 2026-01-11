# ============================================================================
# src/system/CMakeLists.txt
# System-level integration and orchestration
# ============================================================================

# Add subsystem components
add_subdirectory(config)
add_subdirectory(calibration)
add_subdirectory(simulator)

# ============================================================================
# TopLevel system orchestration module
# ============================================================================

set(SYSTEM_SOURCES
    toplevel.cpp
    system_config.cpp
    config_loader.cpp
    config_formatter.cpp
    pcie_arbiter.cpp
)

# Find required system packages
find_package(Threads REQUIRED)

add_library(kpu_system ${SYSTEM_SOURCES})
add_library(StillwaterKPU::System ALIAS kpu_system)

# Enable Position Independent Code for shared library compatibility
set_target_properties(kpu_system PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(kpu_system
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kpu_system
    PUBLIC
        StillwaterKPU::Simulator
        $<TARGET_NAME_IF_EXISTS:nlohmann_json::nlohmann_json>
    PRIVATE
        $<TARGET_NAME_IF_EXISTS:Threads::Threads>
        $<TARGET_NAME_IF_EXISTS:fmt::fmt>
        $<TARGET_NAME_IF_EXISTS:spdlog::spdlog>
)

kpu_set_target_options(kpu_system)

set_target_properties(kpu_system PROPERTIES
    OUTPUT_NAME stillwater_kpu_system
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    FOLDER "src/Core"
)

# Installation
install(TARGETS kpu_system
    LIBRARY DESTINATION lib
        COMPONENT Runtime
    ARCHIVE DESTINATION lib
        COMPONENT Development
)