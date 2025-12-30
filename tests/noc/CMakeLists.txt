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
