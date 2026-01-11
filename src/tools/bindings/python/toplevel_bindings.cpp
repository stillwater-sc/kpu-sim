#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sw/system/toplevel.hpp"

namespace py = pybind11;
using namespace sw::sim;

void bind_system_simulator(py::module_& m) {
    py::class_<SystemSimulator>(m, "SystemSimulator")
        .def(py::init<>(), "Create a new SystemSimulator instance")
        .def("initialize", &SystemSimulator::initialize, 
             "Initialize the simulator and all its components.\n\n"
             "Returns:\n"
             "    bool: True if initialization successful, False otherwise")
        .def("shutdown", &SystemSimulator::shutdown, 
             "Shutdown the simulator and clean up all resources")
        .def("is_initialized", &SystemSimulator::is_initialized, 
             "Check if the simulator is initialized and ready for operations.\n\n"
             "Returns:\n"
             "    bool: True if initialized, False otherwise")
        .def("run_self_test", &SystemSimulator::run_self_test,
             "Run built-in self test to verify simulator functionality.\n\n"
             "Returns:\n"
             "    bool: True if self test passed, False otherwise.\n\n"
             "Note:\n"
             "    Simulator must be initialized before running self test")
        .def("__enter__", [](SystemSimulator& self) -> SystemSimulator& {
            self.initialize();
            return self;
        }, "Context manager entry - initializes the simulator")
        .def("__exit__", [](SystemSimulator& self, py::object, py::object, py::object) {
            self.shutdown();
        }, "Context manager exit - shuts down the simulator");
}