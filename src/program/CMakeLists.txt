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

# ---- the KPU loadable container (#305) -------------------------------------------
# THE SCHEMA IS THE SOURCE OF TRUTH and the header is generated from it at build time,
# never checked in. A checked-in generated header drifts from its schema silently, and
# the only cheap guard against that is not having one.
#
# The generated header goes in a PRIVATE directory rather than the shared
# build/generated tree, because nothing outside this library may see
# <flatbuffers/...>: the public API in include/sw/kpu/loadable/loadable.hpp is
# declaration-only for the same reason deployment_json.hpp is.
set(KPU_LOADABLE_SCHEMA ${CMAKE_SOURCE_DIR}/schemas/kpu_loadable.fbs)
set(KPU_LOADABLE_GEN_DIR ${CMAKE_BINARY_DIR}/generated-private/kpu/loadable)
set(KPU_LOADABLE_GEN_HDR ${KPU_LOADABLE_GEN_DIR}/kpu_loadable_generated.h)

add_custom_command(
    OUTPUT ${KPU_LOADABLE_GEN_HDR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${KPU_LOADABLE_GEN_DIR}
    COMMAND $<TARGET_FILE:flatc> --cpp --scoped-enums
            -o ${KPU_LOADABLE_GEN_DIR} ${KPU_LOADABLE_SCHEMA}
    DEPENDS ${KPU_LOADABLE_SCHEMA} flatc
    COMMENT "flatc: kpu_loadable.fbs -> kpu_loadable_generated.h"
    VERBATIM
)
add_custom_target(kpu_loadable_schema DEPENDS ${KPU_LOADABLE_GEN_HDR})

add_library(kpu_program
    deployment_json.cpp
    loadable.cpp
    ${KPU_LOADABLE_GEN_HDR}
)
add_dependencies(kpu_program kpu_loadable_schema)
add_library(StillwaterKPU::Program ALIAS kpu_program)

target_include_directories(kpu_program
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_BINARY_DIR}/generated-private
)

target_link_libraries(kpu_program
    PRIVATE
        $<TARGET_NAME_IF_EXISTS:nlohmann_json::nlohmann_json>
        $<TARGET_NAME_IF_EXISTS:flatbuffers>
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
