# NoC (Network-on-Chip) component library

# NoC component sources (cycle-accurate/temporal implementations)
add_library(kpu_noc STATIC
    noc.cpp
    wormhole_router.cpp
    dataflow_noc.cpp
    noc_adapters.cpp
    # Behavioral NoC is in src/behavioral/noc/
)

target_include_directories(kpu_noc PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(kpu_noc PUBLIC
    kpu_trace
)

# Compiler options
target_compile_features(kpu_noc PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kpu_noc PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()
