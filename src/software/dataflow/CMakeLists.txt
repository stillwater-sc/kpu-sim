# Data Flow Library
# Tile Data Flow Graph and scheduling infrastructure

set(DATAFLOW_SOURCES
    tile_dataflow_graph.cpp
    block_mover_compiler.cpp
    fused_mlp_sure.cpp
)

add_library(kpu_dataflow ${DATAFLOW_SOURCES})

target_include_directories(kpu_dataflow
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kpu_dataflow
    PUBLIC
        kpu_datamovement_components
)

# fused_mlp_sure.cpp uses domain_flow as its polyhedral math kernel. Add the
# include as SYSTEM (so -Wall/-Wextra/-Wpedantic + CI -Werror do not trip on
# domain_flow's headers), mirroring kpu_compiler. The public header is
# domain_flow-free, so this stays PRIVATE to the library.
if(DOMAIN_FLOW_AVAILABLE)
    target_compile_definitions(kpu_dataflow PRIVATE KPU_HAS_DOMAIN_FLOW)
    if(DOMAIN_FLOW_INCLUDE_DIR)
        target_include_directories(kpu_dataflow SYSTEM PRIVATE ${DOMAIN_FLOW_INCLUDE_DIR})
    endif()
endif()

set_target_properties(kpu_dataflow PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "kpu_dataflow"
    FOLDER "src/DataFlow"
)

if(MSVC)
    target_compile_definitions(kpu_dataflow PRIVATE BUILDING_KPU_SIMULATOR)
    target_compile_options(kpu_dataflow PRIVATE /W4)
else()
    target_compile_options(kpu_dataflow PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS kpu_dataflow
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
