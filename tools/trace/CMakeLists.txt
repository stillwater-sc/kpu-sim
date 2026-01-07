# ============================================================================
# tools/trace/CMakeLists.txt
# KPU Trace Analysis Tools
#
# Tools for analyzing and visualizing memory controller traces.
# ============================================================================

# Trace Summary Tool - CLI trace analysis
kpu_add_tool(trace-summary
    FOLDER "Trace"
    DESCRIPTION "LPDDR5 trace analysis and summary tool"
    SOURCES
        trace-summary/trace_summary.cpp
    LIBS
        nlohmann_json::nlohmann_json
)
