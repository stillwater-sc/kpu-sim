# tests/driver/CMakeLists.txt
# Driver Tests: the driver contains the resource manager code to manage memory, device contexts, and resources

# Memory allocation and management tests
kpu_add_component_test(memory_allocation_test "driver"
    test_memory_allocation.cpp
    test_memory_pool.cpp
    test_memory_alignment.cpp
)

# Data types and resource API tests
kpu_add_component_test(data_types_test "driver"
    test_data_types.cpp
)

kpu_add_component_test(resource_api_test "driver"
    test_resource_api.cpp
)

# Comprehensive test for all resource types
kpu_add_component_test(all_resources_test "driver"
    test_all_resources.cpp
)

# Memory safety and bounds checking tests  
# kpu_add_component_test(memory_safety_test "driver"
#     test_memory_bounds.cpp
#     test_memory_leaks.cpp
# )

# Memory performance tests
# kpu_add_component_test(memory_performance_test "driver"
#     test_memory_bandwidth.cpp
#     test_memory_latency.cpp
# )
