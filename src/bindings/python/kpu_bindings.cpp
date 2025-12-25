#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "sw/kpu/kpu_simulator.hpp"

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
    
    py::class_<sw::kpu::Scratchpad>(m, "Scratchpad")
        .def("get_capacity", &sw::kpu::Scratchpad::get_capacity)
        .def("is_ready", &sw::kpu::Scratchpad::is_ready)
        .def("reset", &sw::kpu::Scratchpad::reset);
    
    py::enum_<sw::kpu::DMAEngine::MemoryType>(m, "MemoryType")
        .value("HOST_MEMORY", sw::kpu::DMAEngine::MemoryType::HOST_MEMORY)
        .value("KPU_MEMORY", sw::kpu::DMAEngine::MemoryType::KPU_MEMORY)
        .value("L3_TILE", sw::kpu::DMAEngine::MemoryType::L3_TILE)
        .value("L2_BANK", sw::kpu::DMAEngine::MemoryType::L2_BANK)
        .value("PAGE_BUFFER", sw::kpu::DMAEngine::MemoryType::PAGE_BUFFER);
    
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
        .def_readwrite("scratchpad_count", &sw::kpu::KPUSimulator::Config::scratchpad_count)
        .def_readwrite("scratchpad_capacity_kb", &sw::kpu::KPUSimulator::Config::scratchpad_capacity_kb)
        // Compute resources
        .def_readwrite("compute_tile_count", &sw::kpu::KPUSimulator::Config::compute_tile_count)
        // Data movement engines
        .def_readwrite("dma_engine_count", &sw::kpu::KPUSimulator::Config::dma_engine_count)
        .def_readwrite("block_mover_count", &sw::kpu::KPUSimulator::Config::block_mover_count)
        .def_readwrite("streamer_count", &sw::kpu::KPUSimulator::Config::streamer_count)
        // Systolic array configuration
        .def_readwrite("systolic_array_rows", &sw::kpu::KPUSimulator::Config::systolic_array_rows)
        .def_readwrite("systolic_array_cols", &sw::kpu::KPUSimulator::Config::systolic_array_cols)
        .def_readwrite("use_systolic_arrays", &sw::kpu::KPUSimulator::Config::use_systolic_arrays)
        // Programmable memory map base addresses
        .def_readwrite("host_memory_base", &sw::kpu::KPUSimulator::Config::host_memory_base)
        .def_readwrite("external_memory_base", &sw::kpu::KPUSimulator::Config::external_memory_base)
        .def_readwrite("l3_tile_base", &sw::kpu::KPUSimulator::Config::l3_tile_base)
        .def_readwrite("l2_bank_base", &sw::kpu::KPUSimulator::Config::l2_bank_base)
        .def_readwrite("scratchpad_base", &sw::kpu::KPUSimulator::Config::scratchpad_base);
    
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
        .def("read_scratchpad", [](sw::kpu::KPUSimulator& self, size_t pad_id, sw::kpu::Address addr, size_t count) {
            std::vector<float> data(count);
            self.read_scratchpad(pad_id, addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_scratchpad", [](sw::kpu::KPUSimulator& self, size_t pad_id, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write_scratchpad(pad_id, addr, data.data(), data.size() * sizeof(float));
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
        .def("read_scratchpad_numpy", [](sw::kpu::KPUSimulator& self, size_t pad_id, sw::kpu::Address addr, const std::vector<size_t>& shape) {
            size_t total_elements = 1;
            for (auto dim : shape) total_elements *= dim;
            
            auto result = py::array_t<float>(shape);
            self.read_scratchpad(pad_id, addr, result.mutable_unchecked().mutable_data(0), total_elements * sizeof(float));
            return result;
        })
        .def("write_scratchpad_numpy", [](sw::kpu::KPUSimulator& self, size_t pad_id, sw::kpu::Address addr, py::array_t<float> array) {
            py::buffer_info buf = array.request();
            self.write_scratchpad(pad_id, addr, buf.ptr, buf.size * sizeof(float));
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
        .def("get_scratchpad_count", &sw::kpu::KPUSimulator::get_scratchpad_count)
        .def("get_compute_tile_count", &sw::kpu::KPUSimulator::get_compute_tile_count)
        .def("get_dma_engine_count", &sw::kpu::KPUSimulator::get_dma_engine_count)
        .def("get_block_mover_count", &sw::kpu::KPUSimulator::get_block_mover_count)
        .def("get_streamer_count", &sw::kpu::KPUSimulator::get_streamer_count)
        .def("get_host_memory_region_capacity", &sw::kpu::KPUSimulator::get_host_memory_region_capacity)
        .def("get_memory_bank_capacity", &sw::kpu::KPUSimulator::get_memory_bank_capacity)
        .def("get_l3_tile_capacity", &sw::kpu::KPUSimulator::get_l3_tile_capacity)
        .def("get_l2_bank_capacity", &sw::kpu::KPUSimulator::get_l2_bank_capacity)
        .def("get_scratchpad_capacity", &sw::kpu::KPUSimulator::get_scratchpad_capacity)

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
        .def("get_scratchpad_base", &sw::kpu::KPUSimulator::get_scratchpad_base,
             "Get the base address of a scratchpad in the unified address space")
        
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
        .def("is_scratchpad_ready", &sw::kpu::KPUSimulator::is_scratchpad_ready)

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
}