# Compute Component Library
# Compute fabric and matrix processing components

# Compute component sources
set(COMPUTE_COMPONENT_SOURCES
    compute_fabric.cpp
    systolic_array.cpp
    sfu.cpp
)

# Create the compute components library
add_library(kpu_compute_components ${COMPUTE_COMPONENT_SOURCES})

# Set up include directories
target_include_directories(kpu_compute_components
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(kpu_compute_components
    PUBLIC
        kpu_memory_components  # Compute components depend on memory components
)

# Set target properties
set_target_properties(kpu_compute_components PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "kpu_compute_components"
    FOLDER "src/Components/Compute"
)

# Add compiler-specific options
if(MSVC)
    target_compile_definitions(kpu_compute_components PRIVATE BUILDING_KPU_SIMULATOR)
    target_compile_options(kpu_compute_components PRIVATE /W4)
else()
    target_compile_options(kpu_compute_components PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Export for installation
install(TARGETS kpu_compute_components
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)