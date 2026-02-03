# ============================================================================
# tools/development/CMakeLists.txt
# KPU Development Tools
# ============================================================================

# KPU Assembler
kpu_add_tool(assembler
    FOLDER "Development"
    DESCRIPTION "KPU assembly language assembler"
    SOURCES
        kpu-assembler/assembler.cpp
    LIBS
        kpu_isa
)

# KPU Debugger
kpu_add_tool(debugger
    FOLDER "Development"
    DESCRIPTION "Interactive KPU debugger"
    SOURCES
        kpu-debugger/debugger.cpp
)

# KPU Disassembler
kpu_add_tool(disassembler
    FOLDER "Development"
    DESCRIPTION "KPU object file disassembler"
    SOURCES
        kpu-disassembler/disassembler.cpp
    LIBS
        nlohmann_json::nlohmann_json
)
