# Memory module - manages its own sources and targets

set(MEMORY_SOURCES
    memory_manager.cpp
)

# Create memory library
add_library(driver STATIC ${MEMORY_SOURCES})

# Create the alias that other parts expect
add_library(StillwaterKPU::Driver ALIAS driver)

target_include_directories(driver 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Apply common target options
kpu_set_target_options(driver)

set_target_properties(driver PROPERTIES
    EXPORT_NAME Driver
    FOLDER "src/Core"
    POSITION_INDEPENDENT_CODE ON  # Enable PIC for shared library compatibility
)

# Don't export for now to avoid dependency chain issues
install(TARGETS driver
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
