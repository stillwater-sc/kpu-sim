# Components Library
# Hardware component organization for the KPU simulator

# Add component subdirectories in dependency order
add_subdirectory(memory)        # Base memory components
add_subdirectory(datamovement)  # Depends on memory components
add_subdirectory(compute)       # Depends on memory components
add_subdirectory(behavioral)    # Behavioral/functional simulation models
# add_subdirectory(control)     # Future: control and management components

# Create a combined components library for convenience
add_library(kpu_components INTERFACE)

# Link all component libraries
target_link_libraries(kpu_components
    INTERFACE
        kpu_memory_components
        kpu_datamovement_components
        kpu_compute_components
        kpu_behavioral_components
)

# Set target properties
set_target_properties(kpu_components PROPERTIES
    FOLDER "Components"
)

# Export for installation
install(TARGETS kpu_components
    EXPORT kpu_components_targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
