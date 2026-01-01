# DFG Toolchain - Standalone tools for Data Flow Graph generation, scheduling, and compilation
#
# Tools:
#   kpu-dfg-gen     - Generate DFG from specification or template
#   kpu-dfg-sched   - Schedule DFG using various algorithms
#   kpu-dfg-compile - Compile scheduled DFG to BlockMover programs
#   kpu-dfg-viz     - Export DFG/schedule to visualization formats
#   kpu-dfg-analyze - Analyze and validate DFG/schedule/programs

# Common library for JSON serialization and spec parsing
add_library(kpu_dfg_tools_common STATIC
    common/dfg_json.cpp
    common/schedule_json.cpp
    common/compiled_json.cpp
)

target_include_directories(kpu_dfg_tools_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kpu_dfg_tools_common PUBLIC
    kpu_dataflow
    nlohmann_json::nlohmann_json
)

# kpu-dfg-gen - DFG Generator
kpu_add_tool(dfg-gen
    FOLDER "DFG"
    DESCRIPTION "Generate DFG from specification or template"
    SOURCES kpu-dfg-gen/main.cpp
    LIBS kpu_dfg_tools_common
)

# kpu-dfg-sched - DFG Scheduler
kpu_add_tool(dfg-sched
    FOLDER "DFG"
    DESCRIPTION "Schedule DFG using various algorithms"
    SOURCES kpu-dfg-sched/main.cpp
    LIBS kpu_dfg_tools_common
)

# kpu-dfg-compile - BlockMover Compiler
kpu_add_tool(dfg-compile
    FOLDER "DFG"
    DESCRIPTION "Compile scheduled DFG to BlockMover programs"
    SOURCES kpu-dfg-compile/main.cpp
    LIBS kpu_dfg_tools_common
)

# kpu-dfg-viz - Visualization Export
kpu_add_tool(dfg-viz
    FOLDER "DFG"
    DESCRIPTION "Export DFG/schedule to visualization formats"
    SOURCES kpu-dfg-viz/main.cpp
    LIBS kpu_dfg_tools_common
)

# kpu-dfg-analyze - Analysis Tool
kpu_add_tool(dfg-analyze
    FOLDER "DFG"
    DESCRIPTION "Analyze and validate DFG/schedule/programs"
    SOURCES kpu-dfg-analyze/main.cpp
    LIBS kpu_dfg_tools_common
)
