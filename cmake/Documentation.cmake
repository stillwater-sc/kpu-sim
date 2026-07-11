# cmake/Documentation.cmake
# Documentation generation with Doxygen

if(KPU_BUILD_DOCS)
    # Degrade gracefully: a missing docs toolchain should not abort the
    # configure of a code build (the "full" preset enables docs by default).
    # dot (graphviz) is optional - diagrams are enabled only when present.
    find_package(Doxygen OPTIONAL_COMPONENTS dot)

    if(NOT DOXYGEN_FOUND)
        message(WARNING
            "KPU_BUILD_DOCS=ON but Doxygen was not found - the 'docs' target "
            "will not be available. Install doxygen (and graphviz for "
            "diagrams): sudo apt-get install doxygen graphviz")
    endif()

    if(DOXYGEN_FOUND)
        # Doxygen configuration
        set(DOXYGEN_PROJECT_NAME "Stillwater KPU Simulator")
        set(DOXYGEN_PROJECT_NUMBER ${PROJECT_VERSION})
        set(DOXYGEN_PROJECT_BRIEF "Knowledge Processing Unit Simulator")
        set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
        set(DOXYGEN_FILE_PATTERNS "*.hpp *.cpp *.h *.c *.md")
        set(DOXYGEN_RECURSIVE YES)
        set(DOXYGEN_EXCLUDE_PATTERNS "*/third_party/*" "*/build/*")
        set(DOXYGEN_GENERATE_HTML YES)
        set(DOXYGEN_GENERATE_LATEX NO)
        set(DOXYGEN_HTML_OUTPUT html)
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_SOURCE_DIR}/README.md)
        set(DOXYGEN_EXTRACT_ALL YES)
        set(DOXYGEN_EXTRACT_PRIVATE YES)
        set(DOXYGEN_EXTRACT_STATIC YES)
        set(DOXYGEN_SOURCE_BROWSER YES)
        set(DOXYGEN_INLINE_SOURCES YES)
        set(DOXYGEN_STRIP_CODE_COMMENTS NO)
        set(DOXYGEN_REFERENCED_BY_RELATION YES)
        set(DOXYGEN_REFERENCES_RELATION YES)
        set(DOXYGEN_CALL_GRAPH YES)
        set(DOXYGEN_CALLER_GRAPH YES)
        if(TARGET Doxygen::dot)
            set(DOXYGEN_HAVE_DOT YES)
        else()
            set(DOXYGEN_HAVE_DOT NO)
        endif()
        set(DOXYGEN_DOT_NUM_THREADS 0)
        set(DOXYGEN_UML_LOOK YES)
        set(DOXYGEN_TEMPLATE_RELATIONS YES)
        set(DOXYGEN_INCLUDE_GRAPH YES)
        set(DOXYGEN_INCLUDED_BY_GRAPH YES)
        set(DOXYGEN_CLASS_DIAGRAMS YES)
        set(DOXYGEN_COLLABORATION_GRAPH YES)
        set(DOXYGEN_GROUP_GRAPHS YES)
        set(DOXYGEN_DIRECTORY_GRAPH YES)
        
        # Create Doxygen target
        doxygen_add_docs(docs
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/components
            ${CMAKE_SOURCE_DIR}/README.md
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
        )
        
        # Install documentation. GNUInstallDirs defines CMAKE_INSTALL_DOCDIR;
        # without it the destination expands empty and install() errors out.
        # (Packaging.cmake also includes it, but runs after this module.)
        include(GNUInstallDirs)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/docs/html/
            DESTINATION ${CMAKE_INSTALL_DOCDIR}
            OPTIONAL
        )
    endif()
endif()
