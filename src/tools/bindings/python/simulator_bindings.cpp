#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sw/system/simulator.hpp>

namespace py = pybind11;
using namespace sw::kpu;

PYBIND11_MODULE(stillwater_kpu, m) {
    m.doc() = "Stillwater KPU Simulator Python bindings";
    
    // KpuSimulator class
    py::class_<KpuSimulator>(m, "KpuSimulator")
        .def(py::init<>())
        .def("initialize", &KpuSimulator::initialize, "Initialize the simulator")
        .def("shutdown", &KpuSimulator::shutdown, "Shutdown the simulator")
        .def("is_initialized", &KpuSimulator::is_initialized, "Check if simulator is initialized")
        .def("run_self_test", &KpuSimulator::run_self_test, "Run basic self test")
        .def("get_memory_manager", &KpuSimulator::get_memory_manager, 
             py::return_value_policy::reference, "Get memory manager instance");
    
    // MemoryManager class
    py::class_<MemoryManager>(m, "MemoryManager")
        .def("allocate", [](MemoryManager& self, size_t size) -> py::int_ {
            return reinterpret_cast<uintptr_t>(self.allocate(size));
        }, "Allocate memory and return address as integer")
        .def("deallocate", [](MemoryManager& self, py::int_ addr) {
            self.deallocate(reinterpret_cast<void*>(static_cast<uintptr_t>(addr)));
        }, "Deallocate memory from address")
        .def("is_valid_address", [](const MemoryManager& self, py::int_ addr) -> bool {
            return self.is_valid_address(reinterpret_cast<void*>(static_cast<uintptr_t>(addr)));
        }, "Check if address is valid")
        .def("get_allocation_count", &MemoryManager::get_allocation_count, "Get allocation count")
        .def("get_allocated_bytes", &MemoryManager::get_allocated_bytes, "Get allocated bytes")
        .def("get_peak_allocated_bytes", &MemoryManager::get_peak_allocated_bytes, "Get peak allocated bytes");
    
    // MemoryPool class
    py::class_<MemoryPool>(m, "MemoryPool")
        .def(py::init<size_t, size_t>(), "Create memory pool with pool_size and block_size")
        .def("allocate", [](MemoryPool& self) -> py::int_ {
            void* ptr = self.allocate();
            return ptr ? reinterpret_cast<uintptr_t>(ptr) : 0;
        }, "Allocate a block")
        .def("deallocate", [](MemoryPool& self, py::int_ addr) {
            self.deallocate(reinterpret_cast<void*>(static_cast<uintptr_t>(addr)));
        }, "Deallocate a block")
        .def("get_block_size", &MemoryPool::get_block_size, "Get block size")
        .def("get_total_blocks", &MemoryPool::get_total_blocks, "Get total blocks")
        .def("get_used_blocks", &MemoryPool::get_used_blocks, "Get used blocks")
        .def("get_available_blocks", &MemoryPool::get_available_blocks, "Get available blocks")
        .def("is_full", &MemoryPool::is_full, "Check if pool is full")
        .def("is_empty", &MemoryPool::is_empty, "Check if pool is empty");

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
}