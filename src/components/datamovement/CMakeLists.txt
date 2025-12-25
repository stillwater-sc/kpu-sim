# Data Movement Component Library

# Data movement component sources
set(DATAMOVEMENT_COMPONENT_SOURCES
    dma_engine.cpp
    block_mover.cpp
    streamer.cpp
    vector_engine.cpp
)

# Create the data_movement components library
add_library(kpu_datamovement_components ${DATAMOVEMENT_COMPONENT_SOURCES})

# Set up include directories
target_include_directories(kpu_datamovement_components
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(kpu_datamovement_components
    PUBLIC
        kpu_memory_components  # DATAMOVEMENT components depend on memory components (includes AddressDecoder)
        kpu_compute_components # VectorEngine depends on SFU from compute components
        kpu_trace             # For cycle-accurate tracing
)

# Set target properties
set_target_properties(kpu_datamovement_components PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "kpu_datamovement_components"
    FOLDER "src/Components/Data Movement"
)

# Add compiler-specific options
if(MSVC)
    target_compile_definitions(kpu_datamovement_components PRIVATE BUILDING_KPU_SIMULATOR)
    target_compile_options(kpu_datamovement_components PRIVATE /W4)
else()
    target_compile_options(kpu_datamovement_components PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Export for installation
install(TARGETS kpu_datamovement_components
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
