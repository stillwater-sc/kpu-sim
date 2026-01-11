# Trace module for KPU simulator
# Provides cycle-accurate tracing and logging capabilities

set(TRACE_SOURCES
    trace_entry.cpp
)

set(TRACE_HEADERS
    ${PROJECT_SOURCE_DIR}/include/sw/trace/trace_entry.hpp
    ${PROJECT_SOURCE_DIR}/include/sw/trace/trace_logger.hpp
    ${PROJECT_SOURCE_DIR}/include/sw/trace/trace_exporter.hpp
    ${PROJECT_SOURCE_DIR}/include/sw/trace/resource_tracker.hpp
    ${PROJECT_SOURCE_DIR}/include/sw/trace/concurrency_analyzer.hpp
)

# Create trace library
add_library(kpu_trace STATIC ${TRACE_SOURCES} ${TRACE_HEADERS})

target_include_directories(kpu_trace
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(kpu_trace PUBLIC cxx_std_20)

# Install targets
install(TARGETS kpu_trace
    EXPORT KPUSimulatorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${TRACE_HEADERS}
    DESTINATION include/sw/trace
)
