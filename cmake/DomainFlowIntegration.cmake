# cmake/DomainFlowIntegration.cmake
# Integration with branes-ai/domain_flow for IR and graph representation

option(KPU_USE_DOMAIN_FLOW "Enable domain_flow IR integration" ON)
option(KPU_DOMAIN_FLOW_LOCAL_PATH "Path to local domain_flow installation" "")

if(KPU_USE_DOMAIN_FLOW)
    message(STATUS "Configuring domain_flow integration")

    # Option 1: Use local installation if provided
    if(KPU_DOMAIN_FLOW_LOCAL_PATH)
        message(STATUS "Using local domain_flow at: ${KPU_DOMAIN_FLOW_LOCAL_PATH}")
        set(DOMAIN_FLOW_ROOT "${KPU_DOMAIN_FLOW_LOCAL_PATH}")

        # Add include directories from local installation
        if(EXISTS "${DOMAIN_FLOW_ROOT}/include")
            set(DOMAIN_FLOW_INCLUDE_DIR "${DOMAIN_FLOW_ROOT}/include")
            include_directories(${DOMAIN_FLOW_INCLUDE_DIR})
            message(STATUS "  Added domain_flow includes: ${DOMAIN_FLOW_INCLUDE_DIR}")
        else()
            message(WARNING "domain_flow include directory not found at ${DOMAIN_FLOW_ROOT}/include")
        endif()

        # Try to find built libraries
        if(EXISTS "${DOMAIN_FLOW_ROOT}/build")
            set(DOMAIN_FLOW_LIBRARY_DIR "${DOMAIN_FLOW_ROOT}/build/lib")
            if(NOT EXISTS "${DOMAIN_FLOW_LIBRARY_DIR}")
                set(DOMAIN_FLOW_LIBRARY_DIR "${DOMAIN_FLOW_ROOT}/build")
            endif()
            link_directories("${DOMAIN_FLOW_LIBRARY_DIR}")
            message(STATUS "  Added domain_flow library dir: ${DOMAIN_FLOW_LIBRARY_DIR}")
        endif()

    # Option 2: Use FetchContent (requires CMake 3.28+ due to domain_flow)
    else()
        message(STATUS "Fetching domain_flow from GitHub")

        # Check CMake version requirement
        if(CMAKE_VERSION VERSION_LESS "3.28")
            message(WARNING "domain_flow requires CMake 3.28+, current version is ${CMAKE_VERSION}")
            message(WARNING "Some domain_flow features may not work correctly")
        endif()

        include(FetchContent)

        # Save BUILD_TESTING state before fetching domain_flow
        # (domain_flow respects BUILD_TESTING, but we want our own tests enabled)
        set(_KPU_SAVED_BUILD_TESTING ${BUILD_TESTING})
        set(BUILD_TESTING OFF CACHE BOOL "Disable domain_flow tests" FORCE)

        FetchContent_Declare(domain_flow
            GIT_REPOSITORY https://github.com/branes-ai/domain_flow.git
            GIT_TAG main  # Use specific tag/commit for production
            GIT_SHALLOW ON  # Faster checkout
            # Prevent domain_flow from building unnecessary components
            CMAKE_ARGS
                -DBUILD_TESTING=OFF
                -DBUILD_EXAMPLES=OFF
        )

        # Set this before MakeAvailable to prevent policy warnings
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # kpu-sim consumes domain_flow ONLY as a header-only library (the dfa IR +
        # polyhedral math). Force its optional tool subprojects OFF so the fetched
        # dependency never drags in heavy external packages or builds executables we
        # do not use. Each defaults OFF in domain_flow, but a value inherited from a
        # parent preset/cache can turn it back on; FORCE here is authoritative.
        # (CMAKE_ARGS in FetchContent_Declare is ignored for the add_subdirectory
        #  model, so these must be cache-set here, like BUILD_TESTING above.)
        #   MATPLOT_TOOLS -> Matplot++ | VISUALIZATION -> CGAL/Qt6 | MLIR_TOOLS -> LLVM/MLIR
        # This also fixes the Windows configure break (missing Matplot++ / CGAL).
        set(DOMAINFLOW_MATPLOT_TOOLS  OFF CACHE BOOL "kpu-sim: no domain_flow plot tools"    FORCE)
        set(DOMAINFLOW_VISUALIZATION  OFF CACHE BOOL "kpu-sim: no domain_flow viz tools"     FORCE)
        set(DOMAINFLOW_MLIR_TOOLS     OFF CACHE BOOL "kpu-sim: no domain_flow MLIR tools"    FORCE)
        set(DOMAINFLOW_DSE            OFF CACHE BOOL "kpu-sim: no domain_flow DSE tools"     FORCE)
        set(DOMAINFLOW_DATABASE_TOOLS OFF CACHE BOOL "kpu-sim: no domain_flow DB tools"      FORCE)
        set(DOMAINFLOW_TOOLS          OFF CACHE BOOL "kpu-sim: no domain_flow dfg/rdg tools" FORCE)

        FetchContent_MakeAvailable(domain_flow)

        # Restore BUILD_TESTING to enable kpu-sim's own tests
        set(BUILD_TESTING ${_KPU_SAVED_BUILD_TESTING} CACHE BOOL "Enable testing" FORCE)
        unset(_KPU_SAVED_BUILD_TESTING)

        # Set include directory
        set(DOMAIN_FLOW_INCLUDE_DIR "${domain_flow_SOURCE_DIR}/include")
        include_directories(${DOMAIN_FLOW_INCLUDE_DIR})
        message(STATUS "  domain_flow source: ${domain_flow_SOURCE_DIR}")
        message(STATUS "  domain_flow includes: ${DOMAIN_FLOW_INCLUDE_DIR}")

        # Organize in IDE folder structure
        if(TARGET domain_flow)
            set_target_properties(domain_flow PROPERTIES FOLDER "External/domain_flow")
        endif()
    endif()

    # Define preprocessor flag for conditional compilation
    add_compile_definitions(KPU_HAS_DOMAIN_FLOW)

    # Export variables for use in subdirectories
    set(DOMAIN_FLOW_AVAILABLE TRUE CACHE BOOL "domain_flow is available" FORCE)

else()
    message(STATUS "domain_flow integration disabled")
    set(DOMAIN_FLOW_AVAILABLE FALSE CACHE BOOL "domain_flow is available" FORCE)
endif()
