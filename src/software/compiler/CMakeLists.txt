# src/compiler/CMakeLists.txt
# KPU Compiler components for graph ingestion and schedule generation

# Compiler library
add_library(kpu_compiler
    graph_loader.cpp
    tile_optimizer.cpp
    schedule_generator.cpp
    l2_tile_scheduler.cpp
    l3_scheduler.cpp
    schedule_characterizer.cpp
    kernel_compiler.cpp
)

target_include_directories(kpu_compiler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(kpu_compiler
    PUBLIC
        nlohmann_json::nlohmann_json
        kpu_isa                     # For DMProgram, OutputStationaryProgramBuilder
)

# Add domain_flow support if available
if(DOMAIN_FLOW_AVAILABLE)
    target_compile_definitions(kpu_compiler PUBLIC KPU_HAS_DOMAIN_FLOW)
    if(DOMAIN_FLOW_INCLUDE_DIR)
        target_include_directories(kpu_compiler PUBLIC ${DOMAIN_FLOW_INCLUDE_DIR})
    endif()

    # Link domain_flow library
    # The library is typically named 'dfa' or 'domain_flow'
    if(DOMAIN_FLOW_ROOT)
        # If using local installation, find the library
        find_library(DFA_LIBRARY
            NAMES dfa libdfa
            PATHS "${DOMAIN_FLOW_ROOT}/build/lib" "${DOMAIN_FLOW_ROOT}/build"
            NO_DEFAULT_PATH
        )
        if(DFA_LIBRARY)
            target_link_libraries(kpu_compiler PUBLIC ${DFA_LIBRARY})
            message(STATUS "  Linked domain_flow library: ${DFA_LIBRARY}")
        else()
            message(WARNING "domain_flow library not found in ${DOMAIN_FLOW_ROOT}/build")
        endif()
    elseif(TARGET dfa)
        # If built via FetchContent, link the target
        target_link_libraries(kpu_compiler PUBLIC dfa)
    endif()
endif()

# Add JSON support flag
target_compile_definitions(kpu_compiler PRIVATE KPU_HAS_JSON)

# Set target properties
set_target_properties(kpu_compiler PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Compiler"
)

# Installation rules
install(TARGETS kpu_compiler
    EXPORT kpu_compiler_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/sw/compiler
    DESTINATION include/sw
    FILES_MATCHING PATTERN "*.hpp"
)
