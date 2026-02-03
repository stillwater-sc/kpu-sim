# ============================================================================
# tools/runtime/CMakeLists.txt
# KPU Runtime Tools
# ============================================================================

# KPU Loader
# Loads KPU object files and programs the KPU simulator
kpu_add_tool(loader
    FOLDER "Runtime"
    DESCRIPTION "KPU loader - loads .kpubin binary files and executes on simulator"
    SOURCES
        kpu-loader/main.cpp
        kpu-loader/object_reader.cpp
        kpu-loader/schedule_binder.cpp
    LIBS
        kpu_isa
        kpu_schedules
        kpu_dsl
        kpu_components
        kpu_trace
        nlohmann_json::nlohmann_json
)

target_include_directories(loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kpu-loader
)
