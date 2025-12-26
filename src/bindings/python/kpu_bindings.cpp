#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

// Core simulator
#include "sw/kpu/kpu_simulator.hpp"

// Kernel abstraction layer
#include "sw/kpu/data_types.hpp"
#include "sw/kpu/kernel.hpp"
#include "sw/compiler/kernel_compiler.hpp"

// Kernel graph
#include "sw/kpu/kernel_graph.hpp"

// Serialization
#include "sw/kpu/isa/program_serializer.hpp"
#include "sw/kpu/kernel_serializer.hpp"

// Executor
#include "sw/kpu/isa/concurrent_executor.hpp"

// Runtime
#include "sw/runtime/runtime.hpp"
#include "sw/runtime/executor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(stillwater_kpu, m) {
    m.doc() = "Stillwater KPU Simulator - High-performance C++ KPU simulator with Python bindings";
    
    // Version information
    m.attr("__version__") = PYBIND11_STRINGIFY(VERSION_INFO);
    
    // Basic types
    py::class_<sw::kpu::ExternalMemory>(m, "ExternalMemory")
        .def("get_capacity", &sw::kpu::ExternalMemory::get_capacity)
        .def("get_bandwidth", &sw::kpu::ExternalMemory::get_bandwidth)
        .def("is_ready", &sw::kpu::ExternalMemory::is_ready)
        .def("reset", &sw::kpu::ExternalMemory::reset)
        .def("get_last_access_cycle", &sw::kpu::ExternalMemory::get_last_access_cycle);
    
    py::class_<sw::kpu::L1Buffer>(m, "L1Buffer")
        .def("get_capacity", &sw::kpu::L1Buffer::get_capacity)
        .def("is_ready", &sw::kpu::L1Buffer::is_ready)
        .def("reset", &sw::kpu::L1Buffer::reset);

    py::enum_<sw::kpu::DMAEngine::MemoryType>(m, "MemoryType")
        .value("HOST_MEMORY", sw::kpu::DMAEngine::MemoryType::HOST_MEMORY)
        .value("KPU_MEMORY", sw::kpu::DMAEngine::MemoryType::KPU_MEMORY)
        .value("L3_TILE", sw::kpu::DMAEngine::MemoryType::L3_TILE);
    
    py::class_<sw::kpu::DMAEngine>(m, "DMAEngine")
        .def("is_busy", &sw::kpu::DMAEngine::is_busy)
        .def("reset", &sw::kpu::DMAEngine::reset)
        .def("get_engine_id", &sw::kpu::DMAEngine::get_engine_id)
        .def("get_queue_size", &sw::kpu::DMAEngine::get_queue_size);
    
    py::class_<sw::kpu::ComputeFabric>(m, "ComputeFabric")
        .def("is_busy", &sw::kpu::ComputeFabric::is_busy)
        .def("reset", &sw::kpu::ComputeFabric::reset)
        .def("get_tile_id", &sw::kpu::ComputeFabric::get_tile_id);
    
    py::class_<sw::kpu::KPUSimulator::Config>(m, "SimulatorConfig")
        .def(py::init<>())
        // Host memory configuration
        .def_readwrite("host_memory_region_count", &sw::kpu::KPUSimulator::Config::host_memory_region_count)
        .def_readwrite("host_memory_region_capacity_mb", &sw::kpu::KPUSimulator::Config::host_memory_region_capacity_mb)
        .def_readwrite("host_memory_bandwidth_gbps", &sw::kpu::KPUSimulator::Config::host_memory_bandwidth_gbps)
        // External memory configuration
        .def_readwrite("memory_bank_count", &sw::kpu::KPUSimulator::Config::memory_bank_count)
        .def_readwrite("memory_bank_capacity_mb", &sw::kpu::KPUSimulator::Config::memory_bank_capacity_mb)
        .def_readwrite("memory_bandwidth_gbps", &sw::kpu::KPUSimulator::Config::memory_bandwidth_gbps)
        // On-chip memory hierarchy
        .def_readwrite("l3_tile_count", &sw::kpu::KPUSimulator::Config::l3_tile_count)
        .def_readwrite("l3_tile_capacity_kb", &sw::kpu::KPUSimulator::Config::l3_tile_capacity_kb)
        .def_readwrite("l2_bank_count", &sw::kpu::KPUSimulator::Config::l2_bank_count)
        .def_readwrite("l2_bank_capacity_kb", &sw::kpu::KPUSimulator::Config::l2_bank_capacity_kb)
        .def_readwrite("l1_buffer_count", &sw::kpu::KPUSimulator::Config::l1_buffer_count)
        .def_readwrite("l1_buffer_capacity_kb", &sw::kpu::KPUSimulator::Config::l1_buffer_capacity_kb)
        // Compute resources
        .def_readwrite("compute_tile_count", &sw::kpu::KPUSimulator::Config::compute_tile_count)
        // Data movement engines
        .def_readwrite("dma_engine_count", &sw::kpu::KPUSimulator::Config::dma_engine_count)
        .def_readwrite("block_mover_count", &sw::kpu::KPUSimulator::Config::block_mover_count)
        .def_readwrite("streamer_count", &sw::kpu::KPUSimulator::Config::streamer_count)
        // Processor array configuration
        .def_readwrite("processor_array_rows", &sw::kpu::KPUSimulator::Config::processor_array_rows)
        .def_readwrite("processor_array_cols", &sw::kpu::KPUSimulator::Config::processor_array_cols)
        .def_readwrite("use_systolic_array_mode", &sw::kpu::KPUSimulator::Config::use_systolic_array_mode)
        // Programmable memory map base addresses
        .def_readwrite("host_memory_base", &sw::kpu::KPUSimulator::Config::host_memory_base)
        .def_readwrite("external_memory_base", &sw::kpu::KPUSimulator::Config::external_memory_base)
        .def_readwrite("l3_tile_base", &sw::kpu::KPUSimulator::Config::l3_tile_base)
        .def_readwrite("l2_bank_base", &sw::kpu::KPUSimulator::Config::l2_bank_base)
        .def_readwrite("l1_buffer_base", &sw::kpu::KPUSimulator::Config::l1_buffer_base);
    
    py::class_<sw::kpu::KPUSimulator::MatMulTest>(m, "MatMulTest")
        .def(py::init<>())
        .def_readwrite("m", &sw::kpu::KPUSimulator::MatMulTest::m)
        .def_readwrite("n", &sw::kpu::KPUSimulator::MatMulTest::n)
        .def_readwrite("k", &sw::kpu::KPUSimulator::MatMulTest::k)
        .def_readwrite("matrix_a", &sw::kpu::KPUSimulator::MatMulTest::matrix_a)
        .def_readwrite("matrix_b", &sw::kpu::KPUSimulator::MatMulTest::matrix_b)
        .def_readwrite("expected_c", &sw::kpu::KPUSimulator::MatMulTest::expected_c);
    
    py::class_<sw::kpu::KPUSimulator>(m, "KPUSimulator")
        .def(py::init<const sw::kpu::KPUSimulator::Config&>(), py::arg("config") = sw::kpu::KPUSimulator::Config{})
        
        // Memory operations - clean delegation API
        .def("read_host_memory", [](sw::kpu::KPUSimulator& self, size_t region_id, sw::kpu::Address addr, size_t count) {
            std::vector<float> data(count);
            self.read_host_memory(region_id, addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_host_memory", [](sw::kpu::KPUSimulator& self, size_t region_id, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write_host_memory(region_id, addr, data.data(), data.size() * sizeof(float));
        })
        .def("read_memory_bank", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, size_t count) {
            std::vector<float> data(count);
            self.read_memory_bank(bank_id, addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_memory_bank", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write_memory_bank(bank_id, addr, data.data(), data.size() * sizeof(float));
        })
        .def("read_l3_tile", [](sw::kpu::KPUSimulator& self, size_t tile_id, sw::kpu::Address addr, size_t count) {
            std::vector<float> data(count);
            self.read_l3_tile(tile_id, addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_l3_tile", [](sw::kpu::KPUSimulator& self, size_t tile_id, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write_l3_tile(tile_id, addr, data.data(), data.size() * sizeof(float));
        })
        .def("read_l2_bank", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, size_t count) {
            std::vector<float> data(count);
            self.read_l2_bank(bank_id, addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_l2_bank", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write_l2_bank(bank_id, addr, data.data(), data.size() * sizeof(float));
        })
        .def("read_l1_buffer", [](sw::kpu::KPUSimulator& self, size_t buffer_id, sw::kpu::Address addr, size_t count) {
            std::vector<float> data(count);
            self.read_l1_buffer(buffer_id, addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_l1_buffer", [](sw::kpu::KPUSimulator& self, size_t buffer_id, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write_l1_buffer(buffer_id, addr, data.data(), data.size() * sizeof(float));
        })
        
        // NumPy array support
        .def("read_memory_bank_numpy", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, const std::vector<size_t>& shape) {
            size_t total_elements = 1;
            for (auto dim : shape) total_elements *= dim;
            
            auto result = py::array_t<float>(shape);
            self.read_memory_bank(bank_id, addr, result.mutable_unchecked().mutable_data(0), total_elements * sizeof(float));
            return result;
        })
        .def("write_memory_bank_numpy", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, py::array_t<float> array) {
            py::buffer_info buf = array.request();
            self.write_memory_bank(bank_id, addr, buf.ptr, buf.size * sizeof(float));
        })
        .def("read_l1_buffer_numpy", [](sw::kpu::KPUSimulator& self, size_t buffer_id, sw::kpu::Address addr, const std::vector<size_t>& shape) {
            size_t total_elements = 1;
            for (auto dim : shape) total_elements *= dim;

            auto result = py::array_t<float>(shape);
            self.read_l1_buffer(buffer_id, addr, result.mutable_unchecked().mutable_data(0), total_elements * sizeof(float));
            return result;
        })
        .def("write_l1_buffer_numpy", [](sw::kpu::KPUSimulator& self, size_t buffer_id, sw::kpu::Address addr, py::array_t<float> array) {
            py::buffer_info buf = array.request();
            self.write_l1_buffer(buffer_id, addr, buf.ptr, buf.size * sizeof(float));
        })
        
        // DMA operations - Primary address-based API
        .def("start_dma_transfer", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address src_addr, sw::kpu::Address dst_addr,
                                     sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.start_dma_transfer(dma_id, src_addr, dst_addr, size);
            } else {
                self.start_dma_transfer(dma_id, src_addr, dst_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("src_addr"), py::arg("dst_addr"), py::arg("size"), py::arg("callback") = py::none(),
        "Primary DMA API - transfer between any two global addresses. Address decoder automatically routes based on address ranges.")
        .def("is_dma_busy", &sw::kpu::KPUSimulator::is_dma_busy)

        // DMA Convenience Helpers - All DMA Patterns
        // Pattern (a): Host ↔ External
        .def("dma_host_to_external", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address host_addr,
                                        sw::kpu::Address external_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_host_to_external(dma_id, host_addr, external_addr, size);
            } else {
                self.dma_host_to_external(dma_id, host_addr, external_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("host_addr"), py::arg("external_addr"), py::arg("size"), py::arg("callback") = py::none())
        .def("dma_external_to_host", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address external_addr,
                                        sw::kpu::Address host_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_external_to_host(dma_id, external_addr, host_addr, size);
            } else {
                self.dma_external_to_host(dma_id, external_addr, host_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("external_addr"), py::arg("host_addr"), py::arg("size"), py::arg("callback") = py::none())

        // Pattern (b): Host ↔ L3
        .def("dma_host_to_l3", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address host_addr,
                                  sw::kpu::Address l3_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_host_to_l3(dma_id, host_addr, l3_addr, size);
            } else {
                self.dma_host_to_l3(dma_id, host_addr, l3_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("host_addr"), py::arg("l3_addr"), py::arg("size"), py::arg("callback") = py::none())
        .def("dma_l3_to_host", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address l3_addr,
                                  sw::kpu::Address host_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_l3_to_host(dma_id, l3_addr, host_addr, size);
            } else {
                self.dma_l3_to_host(dma_id, l3_addr, host_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("l3_addr"), py::arg("host_addr"), py::arg("size"), py::arg("callback") = py::none())

        // Pattern (c): External ↔ L3
        .def("dma_external_to_l3", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address external_addr,
                                      sw::kpu::Address l3_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_external_to_l3(dma_id, external_addr, l3_addr, size);
            } else {
                self.dma_external_to_l3(dma_id, external_addr, l3_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("external_addr"), py::arg("l3_addr"), py::arg("size"), py::arg("callback") = py::none())
        .def("dma_l3_to_external", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address l3_addr,
                                      sw::kpu::Address external_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_l3_to_external(dma_id, l3_addr, external_addr, size);
            } else {
                self.dma_l3_to_external(dma_id, l3_addr, external_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("l3_addr"), py::arg("external_addr"), py::arg("size"), py::arg("callback") = py::none())

        // Pattern (d): Host ↔ Scratchpad
        .def("dma_host_to_scratchpad", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address host_addr,
                                          sw::kpu::Address scratchpad_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_host_to_scratchpad(dma_id, host_addr, scratchpad_addr, size);
            } else {
                self.dma_host_to_scratchpad(dma_id, host_addr, scratchpad_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("host_addr"), py::arg("scratchpad_addr"), py::arg("size"), py::arg("callback") = py::none())
        .def("dma_scratchpad_to_host", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address scratchpad_addr,
                                          sw::kpu::Address host_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_scratchpad_to_host(dma_id, scratchpad_addr, host_addr, size);
            } else {
                self.dma_scratchpad_to_host(dma_id, scratchpad_addr, host_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("scratchpad_addr"), py::arg("host_addr"), py::arg("size"), py::arg("callback") = py::none())

        // Pattern (e): External ↔ Scratchpad
        .def("dma_external_to_scratchpad", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address external_addr,
                                              sw::kpu::Address scratchpad_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_external_to_scratchpad(dma_id, external_addr, scratchpad_addr, size);
            } else {
                self.dma_external_to_scratchpad(dma_id, external_addr, scratchpad_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("external_addr"), py::arg("scratchpad_addr"), py::arg("size"), py::arg("callback") = py::none())
        .def("dma_scratchpad_to_external", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address scratchpad_addr,
                                              sw::kpu::Address external_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_scratchpad_to_external(dma_id, scratchpad_addr, external_addr, size);
            } else {
                self.dma_scratchpad_to_external(dma_id, scratchpad_addr, external_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("scratchpad_addr"), py::arg("external_addr"), py::arg("size"), py::arg("callback") = py::none())

        // Pattern (f): Scratchpad ↔ Scratchpad (data reshuffling)
        .def("dma_scratchpad_to_scratchpad", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address src_scratchpad_addr,
                                                 sw::kpu::Address dst_scratchpad_addr, sw::kpu::Size size, py::object callback) {
            if (callback.is_none()) {
                self.dma_scratchpad_to_scratchpad(dma_id, src_scratchpad_addr, dst_scratchpad_addr, size);
            } else {
                self.dma_scratchpad_to_scratchpad(dma_id, src_scratchpad_addr, dst_scratchpad_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("src_scratchpad_addr"), py::arg("dst_scratchpad_addr"), py::arg("size"), py::arg("callback") = py::none())
        
        // Compute operations
        .def("start_matmul", [](sw::kpu::KPUSimulator& self, size_t tile_id, size_t scratchpad_id, sw::kpu::Size m, sw::kpu::Size n, sw::kpu::Size k,
                               sw::kpu::Address a_addr, sw::kpu::Address b_addr, sw::kpu::Address c_addr, py::object callback) {
            if (callback.is_none()) {
                self.start_matmul(tile_id, scratchpad_id, m, n, k, a_addr, b_addr, c_addr);
            } else {
                self.start_matmul(tile_id, scratchpad_id, m, n, k, a_addr, b_addr, c_addr, [callback]() { callback(); });
            }
        }, py::arg("tile_id"), py::arg("scratchpad_id"), py::arg("m"), py::arg("n"), py::arg("k"), 
           py::arg("a_addr"), py::arg("b_addr"), py::arg("c_addr"), py::arg("callback") = py::none())
        .def("is_compute_busy", &sw::kpu::KPUSimulator::is_compute_busy)
        
        // Simulation control
        .def("reset", &sw::kpu::KPUSimulator::reset)
        .def("step", &sw::kpu::KPUSimulator::step)
        .def("run_until_idle", &sw::kpu::KPUSimulator::run_until_idle)
        
        // Configuration queries
        .def("get_host_memory_region_count", &sw::kpu::KPUSimulator::get_host_memory_region_count)
        .def("get_memory_bank_count", &sw::kpu::KPUSimulator::get_memory_bank_count)
        .def("get_l3_tile_count", &sw::kpu::KPUSimulator::get_l3_tile_count)
        .def("get_l2_bank_count", &sw::kpu::KPUSimulator::get_l2_bank_count)
        .def("get_l1_buffer_count", &sw::kpu::KPUSimulator::get_l1_buffer_count)
        .def("get_compute_tile_count", &sw::kpu::KPUSimulator::get_compute_tile_count)
        .def("get_dma_engine_count", &sw::kpu::KPUSimulator::get_dma_engine_count)
        .def("get_block_mover_count", &sw::kpu::KPUSimulator::get_block_mover_count)
        .def("get_streamer_count", &sw::kpu::KPUSimulator::get_streamer_count)
        .def("get_host_memory_region_capacity", &sw::kpu::KPUSimulator::get_host_memory_region_capacity)
        .def("get_memory_bank_capacity", &sw::kpu::KPUSimulator::get_memory_bank_capacity)
        .def("get_l3_tile_capacity", &sw::kpu::KPUSimulator::get_l3_tile_capacity)
        .def("get_l2_bank_capacity", &sw::kpu::KPUSimulator::get_l2_bank_capacity)
        .def("get_l1_buffer_capacity", &sw::kpu::KPUSimulator::get_l1_buffer_capacity)

        // Address computation helpers for unified address space
        .def("get_host_memory_region_base", &sw::kpu::KPUSimulator::get_host_memory_region_base,
             "Get the base address of a host memory region in the unified address space.\n\n"
             "Example:\n"
             "  host_addr = sim.get_host_memory_region_base(0) + offset\n"
             "  ext_addr = sim.get_external_bank_base(0) + offset\n"
             "  sim.dma_host_to_external(0, host_addr, ext_addr, size)")
        .def("get_external_bank_base", &sw::kpu::KPUSimulator::get_external_bank_base,
             "Get the base address of an external memory bank in the unified address space")
        .def("get_l3_tile_base", &sw::kpu::KPUSimulator::get_l3_tile_base,
             "Get the base address of an L3 tile in the unified address space")
        .def("get_l2_bank_base", &sw::kpu::KPUSimulator::get_l2_bank_base,
             "Get the base address of an L2 bank in the unified address space")
        .def("get_l1_buffer_base", &sw::kpu::KPUSimulator::get_l1_buffer_base,
             "Get the base address of an L1 buffer in the unified address space")
        
        // High-level operations
        .def("run_matmul_test", &sw::kpu::KPUSimulator::run_matmul_test,
             py::arg("test"), py::arg("memory_bank_id") = 0, py::arg("scratchpad_id") = 0, py::arg("compute_tile_id") = 0)
        
        // Statistics and monitoring
        .def("get_current_cycle", &sw::kpu::KPUSimulator::get_current_cycle)
        .def("get_elapsed_time_ms", &sw::kpu::KPUSimulator::get_elapsed_time_ms)
        .def("print_stats", &sw::kpu::KPUSimulator::print_stats)
        .def("print_component_status", &sw::kpu::KPUSimulator::print_component_status)
        .def("is_host_memory_region_ready", &sw::kpu::KPUSimulator::is_host_memory_region_ready)
        .def("is_memory_bank_ready", &sw::kpu::KPUSimulator::is_memory_bank_ready)
        .def("is_l3_tile_ready", &sw::kpu::KPUSimulator::is_l3_tile_ready)
        .def("is_l2_bank_ready", &sw::kpu::KPUSimulator::is_l2_bank_ready)
        .def("is_l1_buffer_ready", &sw::kpu::KPUSimulator::is_l1_buffer_ready)

        // Systolic array information
        .def("is_using_systolic_arrays", &sw::kpu::KPUSimulator::is_using_systolic_arrays)
        .def("get_systolic_array_rows", &sw::kpu::KPUSimulator::get_systolic_array_rows, py::arg("tile_id") = 0)
        .def("get_systolic_array_cols", &sw::kpu::KPUSimulator::get_systolic_array_cols, py::arg("tile_id") = 0)
        .def("get_systolic_array_total_pes", &sw::kpu::KPUSimulator::get_systolic_array_total_pes, py::arg("tile_id") = 0)
        
        // Convenient numpy matrix multiplication
        .def("run_numpy_matmul", [](sw::kpu::KPUSimulator& self, py::array_t<float> a, py::array_t<float> b,
                                   size_t memory_bank_id, size_t scratchpad_id, size_t compute_tile_id) {
            py::buffer_info a_buf = a.request();
            py::buffer_info b_buf = b.request();
            
            if (a_buf.ndim != 2 || b_buf.ndim != 2) {
                throw std::runtime_error("Input arrays must be 2-dimensional");
            }
            
            sw::kpu::Size m = a_buf.shape[0];
            sw::kpu::Size k = a_buf.shape[1];
            sw::kpu::Size n = b_buf.shape[1];
            
            if (k != static_cast<sw::kpu::Size>(b_buf.shape[0])) {
                throw std::runtime_error("Matrix dimensions don't match for multiplication");
            }
            
            // Create test structure
            sw::kpu::KPUSimulator::MatMulTest test;
            test.m = m;
            test.n = n;
            test.k = k;
            
            // Copy data from numpy arrays
            test.matrix_a.assign(static_cast<float*>(a_buf.ptr), 
                               static_cast<float*>(a_buf.ptr) + a_buf.size);
            test.matrix_b.assign(static_cast<float*>(b_buf.ptr), 
                               static_cast<float*>(b_buf.ptr) + b_buf.size);
            
            // Compute expected result
            test.expected_c.resize(m * n);
            for (sw::kpu::Size i = 0; i < m; ++i) {
                for (sw::kpu::Size j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (sw::kpu::Size ki = 0; ki < k; ++ki) {
                        sum += test.matrix_a[i * k + ki] * test.matrix_b[ki * n + j];
                    }
                    test.expected_c[i * n + j] = sum;
                }
            }
            
            // Run simulation
            bool success = self.run_matmul_test(test, memory_bank_id, scratchpad_id, compute_tile_id);
            
            if (!success) {
                throw std::runtime_error("Matrix multiplication simulation failed");
            }
            
            // Return result as numpy array
            auto result = py::array_t<float>({m, n});
            py::buffer_info result_buf = result.request();
            std::copy(test.expected_c.begin(), test.expected_c.end(), static_cast<float*>(result_buf.ptr));
            
            return result;
        }, py::arg("a"), py::arg("b"), py::arg("memory_bank_id") = 0, py::arg("scratchpad_id") = 0, py::arg("compute_tile_id") = 0);
    
    // Test utilities
    m.def("generate_simple_matmul_test", &sw::kpu::test_utils::generate_simple_matmul_test,
          py::arg("m") = 4, py::arg("n") = 4, py::arg("k") = 4);
    
    m.def("generate_random_matrix", &sw::kpu::test_utils::generate_random_matrix,
          py::arg("rows"), py::arg("cols"), py::arg("min_val") = -1.0f, py::arg("max_val") = 1.0f);
    
    m.def("verify_matmul_result", &sw::kpu::test_utils::verify_matmul_result,
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("m"), py::arg("n"), py::arg("k"), 
          py::arg("tolerance") = 1e-5f);
    
    m.def("generate_multi_bank_config", &sw::kpu::test_utils::generate_multi_bank_config,
          py::arg("num_banks") = 4, py::arg("num_tiles") = 2);
    
    m.def("run_distributed_matmul_test", &sw::kpu::test_utils::run_distributed_matmul_test,
          py::arg("sim"), py::arg("matrix_size") = 8);
    
    m.def("generate_simple_matmul_test", &sw::kpu::test_utils::generate_simple_matmul_test,
          py::arg("m") = 4, py::arg("n") = 4, py::arg("k") = 4);
    
    m.def("generate_random_matrix", &sw::kpu::test_utils::generate_random_matrix,
          py::arg("rows"), py::arg("cols"), py::arg("min_val") = -1.0f, py::arg("max_val") = 1.0f);
    
    m.def("verify_matmul_result", &sw::kpu::test_utils::verify_matmul_result,
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("m"), py::arg("n"), py::arg("k"), 
          py::arg("tolerance") = 1e-5f);
    
    // Convenience function for numpy integration
    //m.def("numpy_matmul", [](py::array_t<float> a, py::array_t<float> b, const sw::kpu::KPUSimulator::Config& config) {
    //    sw::kpu::KPUSimulator simulator(config);
    //    return simulator.run_numpy_matmul(a, b);
    //}, py::arg("a"), py::arg("b"), py::arg("config") = sw::kpu::KPUSimulator::Config{});

    // =========================================================================
    // Phase 2-6: Kernel Abstraction, Runtime, and Graph APIs
    // =========================================================================

    // DataType enum
    py::enum_<sw::kpu::DataType>(m, "DataType")
        .value("FLOAT32", sw::kpu::DataType::FLOAT32)
        .value("FLOAT16", sw::kpu::DataType::FLOAT16)
        .value("BFLOAT16", sw::kpu::DataType::BFLOAT16)
        .value("INT32", sw::kpu::DataType::INT32)
        .value("INT8", sw::kpu::DataType::INT8)
        .value("UINT8", sw::kpu::DataType::UINT8)
        .value("INT4", sw::kpu::DataType::INT4)
        .export_values();

    m.def("dtype_size", &sw::kpu::dtype_size, py::arg("dtype"),
          "Get size in bytes for a data type");
    m.def("dtype_name", &sw::kpu::dtype_name, py::arg("dtype"),
          "Get string name for a data type");

    // ActivationType enum
    py::enum_<sw::kpu::ActivationType>(m, "ActivationType")
        .value("NONE", sw::kpu::ActivationType::NONE)
        .value("RELU", sw::kpu::ActivationType::RELU)
        .value("GELU", sw::kpu::ActivationType::GELU)
        .value("SIGMOID", sw::kpu::ActivationType::SIGMOID)
        .value("TANH", sw::kpu::ActivationType::TANH)
        .value("SILU", sw::kpu::ActivationType::SILU)
        .value("LEAKY_RELU", sw::kpu::ActivationType::LEAKY_RELU)
        .export_values();

    // KernelOpType enum
    py::enum_<sw::kpu::KernelOpType>(m, "KernelOpType")
        .value("MATMUL", sw::kpu::KernelOpType::MATMUL)
        .value("BATCH_MATMUL", sw::kpu::KernelOpType::BATCH_MATMUL)
        .value("CONV2D", sw::kpu::KernelOpType::CONV2D)
        .value("ELEMENTWISE", sw::kpu::KernelOpType::ELEMENTWISE)
        .value("MLP", sw::kpu::KernelOpType::MLP)
        .value("CUSTOM", sw::kpu::KernelOpType::CUSTOM)
        .export_values();

    // KernelArgument
    py::class_<sw::kpu::KernelArgument>(m, "KernelArgument")
        .def(py::init<>())
        .def(py::init<const std::string&, sw::kpu::DataType, std::vector<sw::kpu::Size>, bool>(),
             py::arg("name"), py::arg("dtype"), py::arg("shape"), py::arg("is_output") = false)
        .def_readwrite("name", &sw::kpu::KernelArgument::name)
        .def_readwrite("dtype", &sw::kpu::KernelArgument::dtype)
        .def_readwrite("shape", &sw::kpu::KernelArgument::shape)
        .def_readwrite("is_output", &sw::kpu::KernelArgument::is_output)
        .def_readwrite("size_bytes", &sw::kpu::KernelArgument::size_bytes)
        .def("compute_size", &sw::kpu::KernelArgument::compute_size);

    // Kernel class
    py::class_<sw::kpu::Kernel>(m, "Kernel")
        .def(py::init<>())
        // Factory methods
        .def_static("create_matmul", &sw::kpu::Kernel::create_matmul,
                    py::arg("M"), py::arg("N"), py::arg("K"),
                    py::arg("dtype") = sw::kpu::DataType::FLOAT32,
                    "Create a matrix multiplication kernel")
        .def_static("create_mlp", &sw::kpu::Kernel::create_mlp,
                    py::arg("M"), py::arg("N"), py::arg("K"),
                    py::arg("activation"), py::arg("has_bias") = true,
                    py::arg("dtype") = sw::kpu::DataType::FLOAT32,
                    "Create an MLP kernel with activation and optional bias")
        // Metadata
        .def("is_valid", &sw::kpu::Kernel::is_valid)
        .def("name", &sw::kpu::Kernel::name)
        .def("op_type", &sw::kpu::Kernel::op_type)
        .def("dtype", &sw::kpu::Kernel::dtype)
        .def("arguments", &sw::kpu::Kernel::arguments)
        .def("total_input_bytes", &sw::kpu::Kernel::total_input_bytes)
        .def("total_output_bytes", &sw::kpu::Kernel::total_output_bytes)
        // Dimensions
        .def("M", &sw::kpu::Kernel::M)
        .def("N", &sw::kpu::Kernel::N)
        .def("K", &sw::kpu::Kernel::K)
        .def("Ti", &sw::kpu::Kernel::Ti)
        .def("Tj", &sw::kpu::Kernel::Tj)
        .def("Tk", &sw::kpu::Kernel::Tk)
        // MLP specific
        .def("activation", &sw::kpu::Kernel::activation)
        .def("has_bias", &sw::kpu::Kernel::has_bias)
        // Statistics
        .def("instruction_count", &sw::kpu::Kernel::instruction_count)
        .def("total_flops", &sw::kpu::Kernel::total_flops)
        .def("arithmetic_intensity", &sw::kpu::Kernel::arithmetic_intensity)
        .def("summary", &sw::kpu::Kernel::summary)
        .def("validate", [](const sw::kpu::Kernel& self) {
            std::string error;
            bool valid = self.validate(error);
            return py::make_tuple(valid, error);
        });

    // CompileOptions
    py::class_<sw::kpu::compiler::CompileOptions>(m, "CompileOptions")
        .def(py::init<>())
        .def_readwrite("Ti", &sw::kpu::compiler::CompileOptions::Ti)
        .def_readwrite("Tj", &sw::kpu::compiler::CompileOptions::Tj)
        .def_readwrite("Tk", &sw::kpu::compiler::CompileOptions::Tk)
        .def_readwrite("double_buffer", &sw::kpu::compiler::CompileOptions::double_buffer)
        .def_readwrite("systolic_size", &sw::kpu::compiler::CompileOptions::systolic_size)
        .def_readwrite("dtype", &sw::kpu::compiler::CompileOptions::dtype)
        .def_static("defaults", &sw::kpu::compiler::CompileOptions::defaults)
        .def_static("with_tiles", &sw::kpu::compiler::CompileOptions::with_tiles,
                    py::arg("ti"), py::arg("tj"), py::arg("tk"))
        .def_static("for_inference", &sw::kpu::compiler::CompileOptions::for_inference)
        .def("is_auto_tiling", &sw::kpu::compiler::CompileOptions::is_auto_tiling);

    // DataflowStrategy enum
    py::enum_<sw::kpu::compiler::DataflowStrategy>(m, "DataflowStrategy")
        .value("OUTPUT_STATIONARY", sw::kpu::compiler::DataflowStrategy::OUTPUT_STATIONARY)
        .value("WEIGHT_STATIONARY", sw::kpu::compiler::DataflowStrategy::WEIGHT_STATIONARY)
        .value("INPUT_STATIONARY", sw::kpu::compiler::DataflowStrategy::INPUT_STATIONARY)
        .value("AUTO", sw::kpu::compiler::DataflowStrategy::AUTO)
        .export_values();

    // CompilationStats
    py::class_<sw::kpu::compiler::CompilationStats>(m, "CompilationStats")
        .def(py::init<>())
        .def_readonly("compile_time_us", &sw::kpu::compiler::CompilationStats::compile_time_us)
        .def_readonly("used_auto_tiling", &sw::kpu::compiler::CompilationStats::used_auto_tiling)
        .def_readonly("selected_Ti", &sw::kpu::compiler::CompilationStats::selected_Ti)
        .def_readonly("selected_Tj", &sw::kpu::compiler::CompilationStats::selected_Tj)
        .def_readonly("selected_Tk", &sw::kpu::compiler::CompilationStats::selected_Tk)
        .def_readonly("instruction_count", &sw::kpu::compiler::CompilationStats::instruction_count)
        .def_readonly("dma_ops", &sw::kpu::compiler::CompilationStats::dma_ops)
        .def_readonly("block_mover_ops", &sw::kpu::compiler::CompilationStats::block_mover_ops)
        .def_readonly("streamer_ops", &sw::kpu::compiler::CompilationStats::streamer_ops)
        .def_readonly("estimated_external_bytes", &sw::kpu::compiler::CompilationStats::estimated_external_bytes)
        .def_readonly("estimated_arithmetic_intensity", &sw::kpu::compiler::CompilationStats::estimated_arithmetic_intensity)
        .def_readonly("num_m_tiles", &sw::kpu::compiler::CompilationStats::num_m_tiles)
        .def_readonly("num_n_tiles", &sw::kpu::compiler::CompilationStats::num_n_tiles)
        .def_readonly("num_k_tiles", &sw::kpu::compiler::CompilationStats::num_k_tiles)
        .def_readonly("total_tiles", &sw::kpu::compiler::CompilationStats::total_tiles)
        .def("summary", &sw::kpu::compiler::CompilationStats::summary);

    // KernelCompiler
    py::class_<sw::kpu::compiler::KernelCompiler>(m, "KernelCompiler")
        .def(py::init<>())
        // compile_matmul with options (auto-optimization)
        .def("compile_matmul",
             py::overload_cast<sw::kpu::Size, sw::kpu::Size, sw::kpu::Size,
                               const sw::kpu::compiler::CompileOptions&>(
                 &sw::kpu::compiler::KernelCompiler::compile_matmul),
             py::arg("M"), py::arg("N"), py::arg("K"),
             py::arg("options") = sw::kpu::compiler::CompileOptions::defaults(),
             "Compile a matrix multiplication kernel with automatic optimization")
        // compile_matmul with explicit tile sizes
        .def("compile_matmul_tiled",
             py::overload_cast<sw::kpu::Size, sw::kpu::Size, sw::kpu::Size,
                               sw::kpu::Size, sw::kpu::Size, sw::kpu::Size>(
                 &sw::kpu::compiler::KernelCompiler::compile_matmul),
             py::arg("M"), py::arg("N"), py::arg("K"),
             py::arg("Ti"), py::arg("Tj"), py::arg("Tk"),
             "Compile a matrix multiplication kernel with explicit tile sizes")
        // compile_mlp
        .def("compile_mlp", &sw::kpu::compiler::KernelCompiler::compile_mlp,
             py::arg("M"), py::arg("N"), py::arg("K"),
             py::arg("activation"),
             py::arg("has_bias") = true,
             py::arg("dtype") = sw::kpu::DataType::FLOAT32,
             py::arg("options") = sw::kpu::compiler::CompileOptions::defaults(),
             "Compile an MLP kernel with activation and bias")
        // Statistics
        .def("last_stats", &sw::kpu::compiler::KernelCompiler::last_stats,
             py::return_value_policy::reference,
             "Get statistics from the last compilation")
        .def("last_succeeded", &sw::kpu::compiler::KernelCompiler::last_succeeded)
        .def("last_error", &sw::kpu::compiler::KernelCompiler::last_error);

    // =========================================================================
    // Kernel Graph
    // =========================================================================

    // FusionStrategy enum
    py::enum_<sw::kpu::FusionStrategy>(m, "FusionStrategy")
        .value("NONE", sw::kpu::FusionStrategy::NONE)
        .value("PRODUCER_CONSUMER", sw::kpu::FusionStrategy::PRODUCER_CONSUMER)
        .value("HORIZONTAL", sw::kpu::FusionStrategy::HORIZONTAL)
        .value("PIPELINE", sw::kpu::FusionStrategy::PIPELINE)
        .export_values();

    // KernelEdge
    py::class_<sw::kpu::KernelEdge>(m, "KernelEdge")
        .def(py::init<>())
        .def_readonly("from_node", &sw::kpu::KernelEdge::from_node)
        .def_readonly("to_node", &sw::kpu::KernelEdge::to_node)
        .def_readonly("output_name", &sw::kpu::KernelEdge::output_name)
        .def_readonly("input_name", &sw::kpu::KernelEdge::input_name)
        .def_readonly("tensor_size_bytes", &sw::kpu::KernelEdge::tensor_size_bytes);

    // KernelGraphStats
    py::class_<sw::kpu::KernelGraphStats>(m, "KernelGraphStats")
        .def(py::init<>())
        .def_readonly("num_nodes", &sw::kpu::KernelGraphStats::num_nodes)
        .def_readonly("num_edges", &sw::kpu::KernelGraphStats::num_edges)
        .def_readonly("num_input_nodes", &sw::kpu::KernelGraphStats::num_input_nodes)
        .def_readonly("num_output_nodes", &sw::kpu::KernelGraphStats::num_output_nodes)
        .def_readonly("max_depth", &sw::kpu::KernelGraphStats::max_depth)
        .def_readonly("total_instructions", &sw::kpu::KernelGraphStats::total_instructions)
        .def_readonly("total_flops", &sw::kpu::KernelGraphStats::total_flops)
        .def_readonly("total_input_bytes", &sw::kpu::KernelGraphStats::total_input_bytes)
        .def_readonly("total_output_bytes", &sw::kpu::KernelGraphStats::total_output_bytes)
        .def_readonly("intermediate_bytes", &sw::kpu::KernelGraphStats::intermediate_bytes)
        .def_readonly("avg_arithmetic_intensity", &sw::kpu::KernelGraphStats::avg_arithmetic_intensity);

    // KernelGraphCompileOptions
    py::class_<sw::kpu::KernelGraphCompileOptions>(m, "KernelGraphCompileOptions")
        .def(py::init<>())
        .def_readwrite("fusion_strategy", &sw::kpu::KernelGraphCompileOptions::fusion_strategy)
        .def_readwrite("enable_double_buffering", &sw::kpu::KernelGraphCompileOptions::enable_double_buffering)
        .def_readwrite("optimize_memory_allocation", &sw::kpu::KernelGraphCompileOptions::optimize_memory_allocation)
        .def_readwrite("insert_global_barriers", &sw::kpu::KernelGraphCompileOptions::insert_global_barriers)
        .def_readwrite("workspace_limit", &sw::kpu::KernelGraphCompileOptions::workspace_limit);

    // KernelGraphCompileResult
    py::class_<sw::kpu::KernelGraphCompileResult>(m, "KernelGraphCompileResult")
        .def(py::init<>())
        .def_readonly("execution_order", &sw::kpu::KernelGraphCompileResult::execution_order)
        .def_readonly("fused_pairs", &sw::kpu::KernelGraphCompileResult::fused_pairs)
        .def_readonly("workspace_required", &sw::kpu::KernelGraphCompileResult::workspace_required)
        .def_readonly("success", &sw::kpu::KernelGraphCompileResult::success)
        .def_readonly("error_message", &sw::kpu::KernelGraphCompileResult::error_message)
        .def_property_readonly("program", [](const sw::kpu::KernelGraphCompileResult& self) {
            return self.program;
        });

    // KernelGraph
    py::class_<sw::kpu::KernelGraph>(m, "KernelGraph")
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("name"))
        // Node management
        .def("add_kernel", [](sw::kpu::KernelGraph& self, sw::kpu::Kernel kernel, const std::string& name) {
            return self.add_kernel(std::move(kernel), name);
        }, py::arg("kernel"), py::arg("name") = "")
        .def("get_kernel", py::overload_cast<size_t>(&sw::kpu::KernelGraph::get_kernel, py::const_),
             py::return_value_policy::reference)
        .def("has_node", &sw::kpu::KernelGraph::has_node)
        .def("num_nodes", &sw::kpu::KernelGraph::num_nodes)
        .def("node_ids", &sw::kpu::KernelGraph::node_ids)
        // Edge management
        .def("add_edge", &sw::kpu::KernelGraph::add_edge,
             py::arg("from_node"), py::arg("to_node"),
             py::arg("output_name") = "C", py::arg("input_name") = "A")
        .def("get_edge", &sw::kpu::KernelGraph::get_edge, py::return_value_policy::reference)
        .def("num_edges", &sw::kpu::KernelGraph::num_edges)
        .def("outgoing_edges", &sw::kpu::KernelGraph::outgoing_edges)
        .def("incoming_edges", &sw::kpu::KernelGraph::incoming_edges)
        .def("would_create_cycle", &sw::kpu::KernelGraph::would_create_cycle)
        // Graph properties
        .def_property("name", &sw::kpu::KernelGraph::name, &sw::kpu::KernelGraph::set_name)
        .def("empty", &sw::kpu::KernelGraph::empty)
        .def("validate", [](const sw::kpu::KernelGraph& self) {
            std::string error;
            bool valid = self.validate(error);
            return py::make_tuple(valid, error);
        })
        .def("input_nodes", &sw::kpu::KernelGraph::input_nodes)
        .def("output_nodes", &sw::kpu::KernelGraph::output_nodes)
        .def("compute_stats", &sw::kpu::KernelGraph::compute_stats)
        // Execution order
        .def("get_execution_order", &sw::kpu::KernelGraph::get_execution_order)
        .def("get_execution_levels", &sw::kpu::KernelGraph::get_execution_levels)
        .def("get_critical_path", &sw::kpu::KernelGraph::get_critical_path)
        // Fusion
        .def("find_fusible_pairs", &sw::kpu::KernelGraph::find_fusible_pairs)
        .def("can_fuse", &sw::kpu::KernelGraph::can_fuse)
        .def("mark_for_fusion", &sw::kpu::KernelGraph::mark_for_fusion)
        .def("clear_fusion_marks", &sw::kpu::KernelGraph::clear_fusion_marks)
        // Compilation
        .def("compile", &sw::kpu::KernelGraph::compile,
             py::arg("options") = sw::kpu::KernelGraphCompileOptions{})
        .def("compile_sequential", &sw::kpu::KernelGraph::compile_sequential)
        // Visualization
        .def("summary", &sw::kpu::KernelGraph::summary)
        .def("to_dot", &sw::kpu::KernelGraph::to_dot, py::arg("show_tensor_sizes") = true);

    // =========================================================================
    // Serialization
    // =========================================================================

    // ProgramSerializer
    py::class_<sw::kpu::isa::ProgramSerializer>(m, "ProgramSerializer")
        .def(py::init<>())
        .def("serialize", &sw::kpu::isa::ProgramSerializer::serialize)
        .def("deserialize", &sw::kpu::isa::ProgramSerializer::deserialize)
        .def("save", &sw::kpu::isa::ProgramSerializer::save)
        .def("load", &sw::kpu::isa::ProgramSerializer::load)
        .def("to_json", &sw::kpu::isa::ProgramSerializer::to_json,
             py::arg("program"), py::arg("pretty") = true)
        .def("from_json", &sw::kpu::isa::ProgramSerializer::from_json)
        .def("save_json", &sw::kpu::isa::ProgramSerializer::save_json,
             py::arg("program"), py::arg("path"), py::arg("pretty") = true)
        .def("load_json", &sw::kpu::isa::ProgramSerializer::load_json)
        .def("validate", &sw::kpu::isa::ProgramSerializer::validate)
        .def_static("detect_format", &sw::kpu::isa::ProgramSerializer::detect_format);

    // KernelSerializer
    py::class_<sw::kpu::KernelSerializer>(m, "KernelSerializer")
        .def(py::init<>())
        .def("serialize", &sw::kpu::KernelSerializer::serialize)
        .def("deserialize", &sw::kpu::KernelSerializer::deserialize)
        .def("save", &sw::kpu::KernelSerializer::save)
        .def("load", &sw::kpu::KernelSerializer::load)
        .def("to_json", &sw::kpu::KernelSerializer::to_json,
             py::arg("kernel"), py::arg("pretty") = true)
        .def("from_json", &sw::kpu::KernelSerializer::from_json)
        .def("save_json", &sw::kpu::KernelSerializer::save_json,
             py::arg("kernel"), py::arg("path"), py::arg("pretty") = true)
        .def("load_json", &sw::kpu::KernelSerializer::load_json)
        .def("save_auto", &sw::kpu::KernelSerializer::save_auto)
        .def("load_auto", &sw::kpu::KernelSerializer::load_auto)
        .def("validate", &sw::kpu::KernelSerializer::validate)
        .def_static("detect_format", &sw::kpu::KernelSerializer::detect_format);

    // =========================================================================
    // Executor
    // =========================================================================

    // ResourceConfig - Hardware resource configuration for concurrent execution
    py::class_<sw::kpu::isa::ResourceConfig>(m, "ResourceConfig")
        .def(py::init<>())
        // Resource counts
        .def_readwrite("num_memory_channels", &sw::kpu::isa::ResourceConfig::num_memory_channels)
        .def_readwrite("num_block_movers", &sw::kpu::isa::ResourceConfig::num_block_movers)
        .def_readwrite("num_streamers", &sw::kpu::isa::ResourceConfig::num_streamers)
        // Clock frequencies (MHz)
        .def_readwrite("dma_clock_mhz", &sw::kpu::isa::ResourceConfig::dma_clock_mhz)
        .def_readwrite("block_mover_clock_mhz", &sw::kpu::isa::ResourceConfig::block_mover_clock_mhz)
        .def_readwrite("streamer_clock_mhz", &sw::kpu::isa::ResourceConfig::streamer_clock_mhz)
        .def_readwrite("compute_clock_mhz", &sw::kpu::isa::ResourceConfig::compute_clock_mhz)
        // Bus widths (bytes per cycle)
        .def_readwrite("dma_bus_width_bytes", &sw::kpu::isa::ResourceConfig::dma_bus_width_bytes)
        .def_readwrite("block_mover_bus_width_bytes", &sw::kpu::isa::ResourceConfig::block_mover_bus_width_bytes)
        .def_readwrite("streamer_bus_width_bytes", &sw::kpu::isa::ResourceConfig::streamer_bus_width_bytes)
        // Bandwidths (GB/s)
        .def_readwrite("dma_bandwidth_gb_s", &sw::kpu::isa::ResourceConfig::dma_bandwidth_gb_s)
        .def_readwrite("block_mover_bandwidth_gb_s", &sw::kpu::isa::ResourceConfig::block_mover_bandwidth_gb_s)
        .def_readwrite("streamer_bandwidth_gb_s", &sw::kpu::isa::ResourceConfig::streamer_bandwidth_gb_s)
        // Compute fabric
        .def_readwrite("systolic_size", &sw::kpu::isa::ResourceConfig::systolic_size)
        .def_readwrite("compute_throughput_gflops", &sw::kpu::isa::ResourceConfig::compute_throughput_gflops);

    // UtilizationStats - Executor utilization statistics
    py::class_<sw::kpu::isa::ConcurrentExecutor::UtilizationStats>(m, "UtilizationStats")
        .def(py::init<>())
        .def_readonly("dma_utilization", &sw::kpu::isa::ConcurrentExecutor::UtilizationStats::dma_utilization)
        .def_readonly("block_mover_utilization", &sw::kpu::isa::ConcurrentExecutor::UtilizationStats::block_mover_utilization)
        .def_readonly("streamer_utilization", &sw::kpu::isa::ConcurrentExecutor::UtilizationStats::streamer_utilization)
        .def_readonly("compute_utilization", &sw::kpu::isa::ConcurrentExecutor::UtilizationStats::compute_utilization)
        .def_readonly("total_cycles", &sw::kpu::isa::ConcurrentExecutor::UtilizationStats::total_cycles)
        .def_readonly("makespan", &sw::kpu::isa::ConcurrentExecutor::UtilizationStats::makespan);

    // ConcurrentExecutor
    py::class_<sw::kpu::isa::ConcurrentExecutor>(m, "ConcurrentExecutor")
        .def(py::init<const sw::kpu::isa::ResourceConfig&>(), py::arg("config"))
        .def("execute", &sw::kpu::isa::ConcurrentExecutor::execute,
             py::arg("program"),
             "Execute a DMProgram and return the cycle count")
        .def("get_utilization", &sw::kpu::isa::ConcurrentExecutor::get_utilization,
             "Get resource utilization statistics from last execution")
        .def("generate_timeline", &sw::kpu::isa::ConcurrentExecutor::generate_timeline,
             py::arg("width") = 80,
             "Generate ASCII timeline visualization")
        .def("generate_cycle_report", &sw::kpu::isa::ConcurrentExecutor::generate_cycle_report,
             "Generate detailed cycle-by-cycle report");

    // =========================================================================
    // Runtime
    // =========================================================================

    // Stream handle
    py::class_<sw::runtime::Stream>(m, "Stream")
        .def(py::init<>())
        .def_readonly("id", &sw::runtime::Stream::id)
        .def_readonly("valid", &sw::runtime::Stream::valid);

    // Event handle
    py::class_<sw::runtime::Event>(m, "Event")
        .def(py::init<>())
        .def_readonly("id", &sw::runtime::Event::id)
        .def_readonly("valid", &sw::runtime::Event::valid);

    // LaunchResult
    py::class_<sw::runtime::LaunchResult>(m, "LaunchResult")
        .def(py::init<>())
        .def_readonly("success", &sw::runtime::LaunchResult::success)
        .def_readonly("cycles", &sw::runtime::LaunchResult::cycles)
        .def_readonly("error", &sw::runtime::LaunchResult::error);

    // MemcpyKind enum
    py::enum_<sw::runtime::MemcpyKind>(m, "MemcpyKind")
        .value("HostToDevice", sw::runtime::MemcpyKind::HostToDevice)
        .value("DeviceToHost", sw::runtime::MemcpyKind::DeviceToHost)
        .value("DeviceToDevice", sw::runtime::MemcpyKind::DeviceToDevice)
        .export_values();

    // KPURuntime - CUDA-like runtime for KPU
    py::class_<sw::runtime::KPURuntime>(m, "Runtime")
        .def(py::init<sw::kpu::KPUSimulator*>(), py::arg("simulator"),
             py::keep_alive<1, 2>())  // Keep simulator alive while runtime exists
        // Memory management
        .def("malloc", &sw::runtime::KPURuntime::malloc,
             py::arg("size"), py::arg("alignment") = 64,
             "Allocate device memory")
        .def("free", &sw::runtime::KPURuntime::free,
             "Free device memory")
        .def("memcpy_h2d", [](sw::runtime::KPURuntime& self, sw::kpu::Address dst,
                              py::array_t<float> src) {
            py::buffer_info buf = src.request();
            self.memcpy_h2d(dst, buf.ptr, buf.size * sizeof(float));
        }, py::arg("dst"), py::arg("src"),
           "Copy from host (numpy array) to device")
        .def("memcpy_d2h", [](sw::runtime::KPURuntime& self, py::array_t<float> dst,
                              sw::kpu::Address src) {
            py::buffer_info buf = dst.request();
            self.memcpy_d2h(buf.ptr, src, buf.size * sizeof(float));
        }, py::arg("dst"), py::arg("src"),
           "Copy from device to host (numpy array)")
        .def("memcpy_d2d", &sw::runtime::KPURuntime::memcpy_d2d,
             py::arg("dst"), py::arg("src"), py::arg("size"),
             "Copy within device memory")
        .def("memset", &sw::runtime::KPURuntime::memset,
             py::arg("ptr"), py::arg("value"), py::arg("size"),
             "Set device memory to a value")
        // Kernel execution
        .def("launch", &sw::runtime::KPURuntime::launch,
             py::arg("kernel"), py::arg("args"),
             "Launch a kernel synchronously")
        .def("synchronize", &sw::runtime::KPURuntime::synchronize,
             "Wait for all operations to complete")
        // Streams
        .def("create_stream", &sw::runtime::KPURuntime::create_stream)
        .def("destroy_stream", &sw::runtime::KPURuntime::destroy_stream)
        .def("stream_synchronize", &sw::runtime::KPURuntime::stream_synchronize)
        .def("default_stream", &sw::runtime::KPURuntime::default_stream)
        // Events
        .def("create_event", &sw::runtime::KPURuntime::create_event)
        .def("destroy_event", &sw::runtime::KPURuntime::destroy_event)
        .def("record_event", &sw::runtime::KPURuntime::record_event)
        .def("wait_event", &sw::runtime::KPURuntime::wait_event)
        .def("elapsed_time", &sw::runtime::KPURuntime::elapsed_time)
        // Device information
        .def("get_total_memory", &sw::runtime::KPURuntime::get_total_memory)
        .def("get_free_memory", &sw::runtime::KPURuntime::get_free_memory)
        .def("get_total_cycles", &sw::runtime::KPURuntime::get_total_cycles)
        .def("get_launch_count", &sw::runtime::KPURuntime::get_launch_count)
        .def("print_stats", &sw::runtime::KPURuntime::print_stats);

    // =========================================================================
    // GraphExecutor - High-level execution API
    // =========================================================================

    // TensorBinding
    py::class_<sw::runtime::TensorBinding>(m, "TensorBinding")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::vector<sw::kpu::Size>&, sw::kpu::DataType>(),
             py::arg("name"), py::arg("shape"), py::arg("dtype") = sw::kpu::DataType::FLOAT32)
        .def_readwrite("name", &sw::runtime::TensorBinding::name)
        .def_readwrite("shape", &sw::runtime::TensorBinding::shape)
        .def_readwrite("dtype", &sw::runtime::TensorBinding::dtype)
        .def_readonly("device_address", &sw::runtime::TensorBinding::device_address)
        .def_readonly("size_bytes", &sw::runtime::TensorBinding::size_bytes);

    // ExecutionResult
    py::class_<sw::runtime::ExecutionResult>(m, "ExecutionResult")
        .def(py::init<>())
        .def_readonly("success", &sw::runtime::ExecutionResult::success)
        .def_readonly("cycles", &sw::runtime::ExecutionResult::cycles)
        .def_readonly("time_ms", &sw::runtime::ExecutionResult::time_ms)
        .def_readonly("error", &sw::runtime::ExecutionResult::error);

    // GraphExecutor
    py::class_<sw::runtime::GraphExecutor>(m, "GraphExecutor")
        .def(py::init<sw::runtime::KPURuntime*>(), py::arg("runtime"),
             py::keep_alive<1, 2>())
        // Kernel setup
        .def("set_kernel", &sw::runtime::GraphExecutor::set_kernel,
             py::arg("kernel"),
             "Set the kernel to execute")
        .def("create_matmul", &sw::runtime::GraphExecutor::create_matmul,
             py::arg("M"), py::arg("N"), py::arg("K"),
             py::arg("dtype") = sw::kpu::DataType::FLOAT32,
             "Create and set a matmul kernel")
        .def("create_mlp", &sw::runtime::GraphExecutor::create_mlp,
             py::arg("M"), py::arg("N"), py::arg("K"),
             py::arg("activation"),
             py::arg("has_bias") = true,
             py::arg("dtype") = sw::kpu::DataType::FLOAT32,
             "Create and set an MLP kernel")
        .def("has_kernel", &sw::runtime::GraphExecutor::has_kernel)
        // Input/Output binding with numpy support
        .def("set_input", [](sw::runtime::GraphExecutor& self, const std::string& name,
                            py::array_t<float> data) {
            py::buffer_info buf = data.request();
            std::vector<sw::kpu::Size> shape;
            for (auto dim : buf.shape) {
                shape.push_back(static_cast<sw::kpu::Size>(dim));
            }
            self.set_input(name, buf.ptr, shape);
        }, py::arg("name"), py::arg("data"),
           "Set input tensor from numpy array")
        .def("get_output", [](sw::runtime::GraphExecutor& self, const std::string& name,
                             py::array_t<float> data) {
            py::buffer_info buf = data.request();
            self.get_output(name, buf.ptr, buf.size * sizeof(float));
        }, py::arg("name"), py::arg("data"),
           "Get output tensor to numpy array")
        .def("get_binding", &sw::runtime::GraphExecutor::get_binding,
             py::return_value_policy::reference)
        // Execution
        .def("execute", &sw::runtime::GraphExecutor::execute)
        .def("last_result", &sw::runtime::GraphExecutor::last_result)
        .def("get_last_execution_time_ms", &sw::runtime::GraphExecutor::get_last_execution_time_ms)
        .def("get_last_execution_cycles", &sw::runtime::GraphExecutor::get_last_execution_cycles)
        // Cleanup
        .def("release", &sw::runtime::GraphExecutor::release);
}