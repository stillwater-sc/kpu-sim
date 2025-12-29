# NoC tests

add_executable(noc_test
    noc_test.cpp
)

target_link_libraries(noc_test PRIVATE
    kpu_noc
    Catch2::Catch2WithMain
)

target_include_directories(noc_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(noc_test PROPERTIES FOLDER "Tests/noc")

add_test(NAME NoCTest COMMAND noc_test)
set_tests_properties(NoCTest PROPERTIES LABELS noc)
