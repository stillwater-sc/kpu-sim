#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "sw/kpu/simulator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(stillwater_kpu, m) {
    m.doc() = "Stillwater KPU Simulator - High-performance C++ KPU simulator with Python bindings";
    
    // Version information
    m.attr("__version__") = PYBIND11_STRINGIFY(VERSION_INFO);
    
    // Basic types
    py::class_<sw::kpu::ExternalMemory>(m, "ExternalMemory")
        .def(py::init<sw::kpu::Size, sw::kpu::Size>(), 
             py::arg("capacity_mb") = 1024, py::arg("bandwidth_gbps") = 100)
        .def("get_capacity", &sw::kpu::ExternalMemory::get_capacity)
        .def("is_ready", &sw::kpu::ExternalMemory::is_ready)
        .def("reset", &sw::kpu::ExternalMemory::reset)
        .def("read_float_array", [](sw::kpu::ExternalMemory& self, sw::kpu::Address addr, sw::kpu::Size count) {
            std::vector<float> data(count);
            self.read(addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_float_array", [](sw::kpu::ExternalMemory& self, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write(addr, data.data(), data.size() * sizeof(float));
        })
        .def("read_numpy", [](sw::kpu::ExternalMemory& self, sw::kpu::Address addr, const std::vector<sw::kpu::Size>& shape) {
            sw::kpu::Size total_elements = 1;
            for (auto dim : shape) total_elements *= dim;
            
            auto result = py::array_t<float>(shape);
            self.read(addr, result.mutable_unchecked().mutable_data(0), total_elements * sizeof(float));
            return result;
        })
        .def("write_numpy", [](sw::kpu::ExternalMemory& self, sw::kpu::Address addr, py::array_t<float> array) {
            py::buffer_info buf = array.request();
            self.write(addr, buf.ptr, buf.size * sizeof(float));
        });
    
    py::class_<sw::kpu::Scratchpad>(m, "Scratchpad")
        .def(py::init<sw::kpu::Size>(), py::arg("capacity_kb") = 512)
        .def("get_capacity", &sw::kpu::Scratchpad::get_capacity)
        .def("is_ready", &sw::kpu::Scratchpad::is_ready)
        .def("reset", &sw::kpu::Scratchpad::reset)
        .def("read_float_array", [](sw::kpu::Scratchpad& self, sw::kpu::Address addr, sw::kpu::Size count) {
            std::vector<float> data(count);
            self.read(addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_float_array", [](sw::kpu::Scratchpad& self, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write(addr, data.data(), data.size() * sizeof(float));
        })
        .def("read_numpy", [](sw::kpu::Scratchpad& self, sw::kpu::Address addr, const std::vector<sw::kpu::Size>& shape) {
            sw::kpu::Size total_elements = 1;
            for (auto dim : shape) total_elements *= dim;
            
            auto result = py::array_t<float>(shape);
            self.read(addr, result.mutable_unchecked().mutable_data(0), total_elements * sizeof(float));
            return result;
        })
        .def("write_numpy", [](sw::kpu::Scratchpad& self, sw::kpu::Address addr, py::array_t<float> array) {
            py::buffer_info buf = array.request();
            self.write(addr, buf.ptr, buf.size * sizeof(float));
        });
    
    py::class_<sw::kpu::DMAEngine>(m, "DMAEngine")
        .def("enqueue_transfer", [](sw::kpu::DMAEngine& self, sw::kpu::Address src_addr, 
                                   sw::kpu::Address dst_addr, sw::kpu::Size size, py::function callback) {
            if (callback.is_none()) {
                self.enqueue_transfer(src_addr, dst_addr, size);
            } else {
                self.enqueue_transfer(src_addr, dst_addr, size, [callback]() { callback(); });
            }
        }, py::arg("src_addr"), py::arg("dst_addr"), py::arg("size"), py::arg("callback") = py::none())
        .def("is_busy", &sw::kpu::DMAEngine::is_busy)
        .def("reset", &sw::kpu::DMAEngine::reset);
    
    py::class_<sw::kpu::ComputeFabric::MatMulConfig>(m, "MatMulConfig")
        .def(py::init<>())
        .def_readwrite("m", &sw::kpu::ComputeFabric::MatMulConfig::m)
        .def_readwrite("n", &sw::kpu::ComputeFabric::MatMulConfig::n)
        .def_readwrite("k", &sw::kpu::ComputeFabric::MatMulConfig::k)
        .def_readwrite("a_addr", &sw::kpu::ComputeFabric::MatMulConfig::a_addr)
        .def_readwrite("b_addr", &sw::kpu::ComputeFabric::MatMulConfig::b_addr)
        .def_readwrite("c_addr", &sw::kpu::ComputeFabric::MatMulConfig::c_addr);
    
    py::class_<sw::kpu::ComputeFabric>(m, "ComputeFabric")
        .def("start_matmul", [](sw::kpu::ComputeFabric& self, const sw::kpu::ComputeFabric::MatMulConfig& config, py::function callback) {
            auto config_copy = config;
            if (!callback.is_none()) {
                config_copy.completion_callback = [callback]() { callback(); };
            }
            self.start_matmul(config_copy);
        }, py::arg("config"), py::arg("callback") = py::none())
        .def("is_busy", &sw::kpu::ComputeFabric::is_busy)
        .def("reset", &sw::kpu::ComputeFabric::reset);
    
    py::class_<sw::kpu::KPUSimulator::Config>(m, "SimulatorConfig")
        .def(py::init<>())
        .def_readwrite("external_memory_mb", &sw::kpu::KPUSimulator::Config::external_memory_mb)
        .def_readwrite("scratchpad_kb", &sw::kpu::KPUSimulator::Config::scratchpad_kb)
        .def_readwrite("memory_bandwidth_gbps", &sw::kpu::KPUSimulator::Config::memory_bandwidth_gbps);
    
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
        .def("get_external_memory", &sw::kpu::KPUSimulator::get_external_memory, 
             py::return_value_policy::reference_internal)
        .def("get_scratchpad", &sw::kpu::KPUSimulator::get_scratchpad,
             py::return_value_policy::reference_internal)
        .def("get_dma_ext_to_scratch", &sw::kpu::KPUSimulator::get_dma_ext_to_scratch,
             py::return_value_policy::reference_internal)
        .def("get_dma_scratch_to_ext", &sw::kpu::KPUSimulator::get_dma_scratch_to_ext,
             py::return_value_policy::reference_internal)
        .def("get_compute_fabric", &sw::kpu::KPUSimulator::get_compute_fabric,
             py::return_value_policy::reference_internal)
        .def("reset", &sw::kpu::KPUSimulator::reset)
        .def("step", &sw::kpu::KPUSimulator::step)
        .def("run_until_idle", &sw::kpu::KPUSimulator::run_until_idle)
        .def("run_matmul_test", &sw::kpu::KPUSimulator::run_matmul_test)
        .def("get_current_cycle", &sw::kpu::KPUSimulator::get_current_cycle)
        .def("get_elapsed_time_ms", &sw::kpu::KPUSimulator::get_elapsed_time_ms)
        .def("print_stats", &sw::kpu::KPUSimulator::print_stats)
        .def("run_numpy_matmul", [](sw::kpu::KPUSimulator& self, py::array_t<float> a, py::array_t<float> b) {
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
            bool success = self.run_matmul_test(test);
            
            if (!success) {
                throw std::runtime_error("Matrix multiplication simulation failed");
            }
            
            // Return result as numpy array
            auto result = py::array_t<float>({m, n});
            py::buffer_info result_buf = result.request();
            std::copy(test.expected_c.begin(), test.expected_c.end(), static_cast<float*>(result_buf.ptr));
            
            return result;
        });
    
    // Test utilities
    m.def("generate_simple_matmul_test", &sw::kpu::test_utils::generate_simple_matmul_test,
          py::arg("m") = 4, py::arg("n") = 4, py::arg("k") = 4);
    
    m.def("generate_random_matrix", &sw::kpu::test_utils::generate_random_matrix,
          py::arg("rows"), py::arg("cols"), py::arg("min_val") = -1.0f, py::arg("max_val") = 1.0f);
    
    m.def("verify_matmul_result", &sw::kpu::test_utils::verify_matmul_result,
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("m"), py::arg("n"), py::arg("k"), 
          py::arg("tolerance") = 1e-5f);
    
    // Convenience function for numpy integration
    m.def("numpy_matmul", [](py::array_t<float> a, py::array_t<float> b, const sw::kpu::KPUSimulator::Config& config) {
        sw::kpu::KPUSimulator simulator(config);
        return simulator.run_numpy_matmul(a, b);
    }, py::arg("a"), py::arg("b"), py::arg("config") = sw::kpu::KPUSimulator::Config{});
}