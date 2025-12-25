# ============================================================================
# src/bindings/python/CMakeLists.txt
# Python bindings using pybind11

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# TopLevelSimulator bindings - clean and focused
pybind11_add_module(stillwater_toplevel stillwater_toplevel_bindings.cpp)

target_link_libraries(stillwater_toplevel PRIVATE
    StillwaterKPU::System
    pybind11::headers
)

kpu_set_target_options(stillwater_toplevel)

# Compiler-specific options for pybind11
target_compile_definitions(stillwater_toplevel PRIVATE 
    VERSION_INFO="${PROJECT_VERSION}"
)

set_target_properties(stillwater_toplevel PROPERTIES
    FOLDER "src/Bindings"
    # Put the Python module in the build directory, not source directory
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    # On Windows, also ensure it goes to the right Debug/Release subdirectory
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/Debug"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release"
)

# Copy __init__.py to build directories so Python can import the module
# Use custom command to ensure it's always up to date
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
    COMMENT "Copying __init__.py to build directory"
)

# For multi-config generators, also copy to Debug/Release subdirectories
if(CMAKE_CONFIGURATION_TYPES)
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${config}/__init__.py"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${config}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
                "${CMAKE_CURRENT_BINARY_DIR}/${config}/__init__.py"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
            COMMENT "Copying __init__.py to ${config} directory"
        )
    endforeach()
    
    # Add custom target to trigger the copies
    add_custom_target(copy_init_py ALL
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
                "${CMAKE_CURRENT_BINARY_DIR}/Debug/__init__.py"
                "${CMAKE_CURRENT_BINARY_DIR}/Release/__init__.py"
    )
    
    # Create properly named copies of the .pyd files for Python import
    add_custom_command(TARGET stillwater_toplevel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:stillwater_toplevel>"
            "$<TARGET_FILE_DIR:stillwater_toplevel>/stillwater_toplevel.pyd"
        COMMENT "Creating importable copy of Python module"
    )
    
    # Make stillwater_toplevel depend on the copy target
    add_dependencies(stillwater_toplevel copy_init_py)
else()
    # Single-config generator
    add_custom_target(copy_init_py ALL
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
    )
    add_dependencies(stillwater_toplevel copy_init_py)
    
    # Create properly named copy for single-config too
    add_custom_command(TARGET stillwater_toplevel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:stillwater_toplevel>"
            "$<TARGET_FILE_DIR:stillwater_toplevel>/stillwater_toplevel.pyd"
        COMMENT "Creating importable copy of Python module"
    )
endif()

# KPU simulator bindings
pybind11_add_module(stillwater_kpu kpu_bindings.cpp)
target_link_libraries(stillwater_kpu PRIVATE
    kpu_simulator
    kpu_isa
    kpu_runtime
    pybind11::headers
)
kpu_set_target_options(stillwater_kpu)
target_compile_definitions(stillwater_kpu PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)
set_target_properties(stillwater_kpu PROPERTIES
    FOLDER "src/Bindings"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/Debug"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release"
)

# Create properly named copies of the .pyd files for Python import
if(CMAKE_CONFIGURATION_TYPES)
    add_custom_command(TARGET stillwater_kpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:stillwater_kpu>"
            "$<TARGET_FILE_DIR:stillwater_kpu>/stillwater_kpu.pyd"
        COMMENT "Creating importable copy of KPU Python module"
    )
else()
    add_custom_command(TARGET stillwater_kpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:stillwater_kpu>"
            "$<TARGET_FILE_DIR:stillwater_kpu>/stillwater_kpu.pyd"
        COMMENT "Creating importable copy of KPU Python module"
    )
endif()

# Install Python modules
install(TARGETS stillwater_toplevel stillwater_kpu
    DESTINATION .
    COMPONENT Python
)

# Install Python package files
install(FILES __init__.py
    DESTINATION .
    COMPONENT Python
)