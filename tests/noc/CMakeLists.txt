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

# Wormhole router tests (correct bandwidth modeling)
add_executable(wormhole_router_test
    wormhole_router_test.cpp
)

target_link_libraries(wormhole_router_test PRIVATE
    kpu_noc
    Catch2::Catch2WithMain
)

target_include_directories(wormhole_router_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(wormhole_router_test PROPERTIES FOLDER "Tests/noc")

add_test(NAME WormholeRouterTest COMMAND wormhole_router_test)
set_tests_properties(WormholeRouterTest PROPERTIES LABELS "noc;wormhole")

# Dataflow NoC tests (unified dataflow approach)
add_executable(dataflow_noc_test
    dataflow_noc_test.cpp
)

target_link_libraries(dataflow_noc_test PRIVATE
    kpu_noc
    Catch2::Catch2WithMain
)

target_include_directories(dataflow_noc_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(dataflow_noc_test PROPERTIES FOLDER "Tests/noc")

add_test(NAME DataflowNoCTest COMMAND dataflow_noc_test)
set_tests_properties(DataflowNoCTest PROPERTIES LABELS "noc;dataflow")

# NoC interface tests (unified INoC interface and adapters)
add_executable(noc_interface_test
    noc_interface_test.cpp
)

target_link_libraries(noc_interface_test PRIVATE
    kpu_noc
    Catch2::Catch2WithMain
)

target_include_directories(noc_interface_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(noc_interface_test PROPERTIES FOLDER "Tests/noc")

add_test(NAME NoCInterfaceTest COMMAND noc_interface_test)
set_tests_properties(NoCInterfaceTest PROPERTIES LABELS "noc;interface")
