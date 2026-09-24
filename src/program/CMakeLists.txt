# ============================================================================
# src/program/CMakeLists.txt
# Compiled units of the L0 program layer.
#
# The program layer is otherwise header-only and standard-library only. This
# library exists so the JSON edge of a DeploymentSpec can use nlohmann WITHOUT
# handing that dependency to every consumer of a TileProgram — the declaration
# lives in include/sw/kpu/program/platform/deployment_json.hpp and nothing else
# sees <nlohmann/json.hpp>.
#
# Linked PUBLIC into kpu_simulator so every test and tool that already links
# StillwaterKPU::Simulator picks it up with no change to its link list.
# ============================================================================

add_library(kpu_program
    deployment_json.cpp
)
add_library(StillwaterKPU::Program ALIAS kpu_program)

target_include_directories(kpu_program
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(kpu_program
    PRIVATE
        $<TARGET_NAME_IF_EXISTS:nlohmann_json::nlohmann_json>
)

kpu_set_target_options(kpu_program)

set_target_properties(kpu_program PROPERTIES
    OUTPUT_NAME stillwater_kpu_program
    FOLDER "src/Core"
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS kpu_program
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
)
