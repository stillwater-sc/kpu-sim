# ============================================================================
# src/CMakeLists.txt
# Core simulator engine

# Add trace infrastructure first (needed by components)
add_subdirectory(trace)

# Add component-based architecture
add_subdirectory(components)

# Add main simulator and system components
add_subdirectory(system)
add_subdirectory(simulator)
add_subdirectory(driver)

# Add compiler components (graph loading, schedule generation)
add_subdirectory(compiler)

# Add ISA components (Data Movement ISA for Domain Flow Architecture)
add_subdirectory(isa)

# Add runtime library (host-side orchestration for kernel execution)
add_subdirectory(runtime)

# Keep old memory directory for backward compatibility during transition
# add_subdirectory(memory)  # Disabled - all components moved to src/components/

if(KPU_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
