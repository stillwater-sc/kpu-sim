# Memory Component Library
# Hierarchical memory system components

# Memory component sources
set(MEMORY_COMPONENT_SOURCES
    external_memory.cpp
    memory_map.cpp
    sparse_memory.cpp
    scratchpad.cpp
    page_buffer.cpp
    l3_tile.cpp
    l2_bank.cpp
    l1_buffer.cpp
    cache.cpp
    storage_scheduler.cpp
    memory_orchestrator_integration.cpp
    address_decoder.cpp
)

# Create the memory components library
add_library(kpu_memory_components ${MEMORY_COMPONENT_SOURCES})

# Set up include directories
target_include_directories(kpu_memory_components
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies if needed
target_link_libraries(kpu_memory_components
    PUBLIC
        # Add any dependencies here
)

# Set target properties
set_target_properties(kpu_memory_components PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "kpu_memory_components"
    FOLDER "src/Components/Memory"
)

# Add compiler-specific options
if(MSVC)
    target_compile_definitions(kpu_memory_components PRIVATE BUILDING_KPU_SIMULATOR)
    target_compile_options(kpu_memory_components PRIVATE /W4)
else()
    target_compile_options(kpu_memory_components PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Export for installation
install(TARGETS kpu_memory_components
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)