# Stillwater KPU Project Structure

```
stillwater-kpu/
├── CMakeLists.txt                          # Root CMake configuration
├── README.md                               # Project overview and quick start
├── LICENSE                                 # Project license
├── CHANGELOG.md                            # Version history
├── .gitignore                             # Git ignore rules
├── .clang-format                          # Code formatting rules
├── .github/                               # GitHub Actions CI/CD
│   └── workflows/
│       ├── ci.yml                         # Continuous integration
│       ├── release.yml                    # Release automation
│       └── documentation.yml              # Doc generation
├── docs/                                  # Documentation
│   ├── architecture/                     # Architecture docs
│   ├── api/                              # API documentation
│   ├── tutorials/                        # User tutorials
│   ├── examples/                         # Code examples
│   └── performance/                      # Performance analysis
├── cmake/                                 # CMake utilities and find modules
│   ├── StillwaterKPUConfig.cmake.in      # Package config template
│   ├── FindOpenCL.cmake                  # Custom find modules
│   ├── FindCUDA.cmake                    
│   ├── CompilerOptions.cmake             # Compiler settings
│   ├── Dependencies.cmake                # External dependencies
│   ├── Documentation.cmake               # Doxygen integration
│   ├── Testing.cmake                     # Test configuration
│   └── Packaging.cmake                   # CPack configuration
├── include/                              # Public headers (install)
│   └── sw/                               # sw == Stillwater
│       └── kpu/                          # kpu == Knowledge Processing Unit
│           ├── kpu.hpp                   # Main public header
│           ├── memory/                   # Memory system headers
│           ├── compute/                  # Compute system headers
│           ├── fabric/                   # Fabric system headers
│           └── utilities/                # Utility headers
├── src/                                  # Core simulator engine
│   ├── CMakeLists.txt
│   ├── simulator/                        # Main simulator engine
│   │   ├── CMakeLists.txt
│   │   ├── kpu_simulator.cpp
│   │   ├── kpu_simulator.hpp
│   │   ├── device_manager.cpp
│   │   ├── device_manager.hpp
│   │   ├── instruction_set.cpp
│   │   ├── instruction_set.hpp
│   │   ├── execution_engine.cpp
│   │   ├── execution_engine.hpp
│   │   └── trace_logger.cpp
│   └── bindings/                         # Language bindings
│       ├── CMakeLists.txt
│       ├── c/                            # C API bindings
│       │   ├── CMakeLists.txt
│       │   ├── kpu_c_api.h
│       │   └── kpu_c_api.cpp
│       └── python/                       # Python bindings
│           ├── CMakeLists.txt
│           ├── pybind_module.cpp
│           └── __init__.py
├── components/                           # Hardware component libraries
│   ├── CMakeLists.txt
│   ├── memory/                           # Memory subsystem
│   │   ├── CMakeLists.txt
│   │   ├── include/stillwater/kpu/memory/
│   │   │   ├── memory_interface.hpp      # Abstract memory interface
│   │   │   ├── main_memory.hpp           # DRAM simulation
│   │   │   ├── scratchpad.hpp            # Fast scratchpad memory
│   │   │   ├── cache.hpp                 # Cache hierarchy
│   │   │   ├── memory_controller.hpp     # Memory controller
│   │   │   └── address_translation.hpp   # Virtual memory
│   │   ├── src/
│   │   │   ├── main_memory.cpp
│   │   │   ├── scratchpad.cpp
│   │   │   ├── cache.cpp
│   │   │   ├── memory_controller.cpp
│   │   │   └── address_translation.cpp
│   │   └── tests/
│   │       ├── test_main_memory.cpp
│   │       ├── test_scratchpad.cpp
│   │       └── test_cache.cpp
│   ├── compute/                          # Compute subsystem
│   │   ├── CMakeLists.txt
│   │   ├── include/stillwater/kpu/compute/
│   │   │   ├── compute_engine.hpp        # Abstract compute interface
│   │   │   ├── matrix_unit.hpp           # Matrix processing unit
│   │   │   ├── vector_unit.hpp           # Vector processing unit
│   │   │   ├── scalar_unit.hpp           # Scalar processing unit
│   │   │   ├── activation_unit.hpp       # Activation functions
│   │   │   ├── data_format.hpp           # Number format support
│   │   │   └── instruction_decoder.hpp   # Instruction decoding
│   │   ├── src/
│   │   │   ├── matrix_unit.cpp
│   │   │   ├── vector_unit.cpp
│   │   │   ├── scalar_unit.cpp
│   │   │   ├── activation_unit.cpp
│   │   │   └── instruction_decoder.cpp
│   │   └── tests/
│   │       ├── test_matrix_unit.cpp
│   │       ├── test_vector_unit.cpp
│   │       └── test_activation_unit.cpp
│   ├── fabric/                           # Interconnect and fabric
│   │   ├── CMakeLists.txt
│   │   ├── include/stillwater/kpu/fabric/
│   │   │   ├── fabric_interface.hpp      # Abstract fabric interface
│   │   │   ├── crossbar.hpp              # Crossbar switch
│   │   │   ├── mesh_noc.hpp              # Mesh network-on-chip
│   │   │   ├── ring_noc.hpp              # Ring network-on-chip
│   │   │   ├── packet.hpp                # Network packets
│   │   │   ├── router.hpp                # NOC router
│   │   │   └── flow_control.hpp          # Flow control mechanisms
│   │   ├── src/
│   │   │   ├── crossbar.cpp
│   │   │   ├── mesh_noc.cpp
│   │   │   ├── ring_noc.cpp
│   │   │   ├── packet.cpp
│   │   │   └── router.cpp
│   │   └── tests/
│   │       ├── test_crossbar.cpp
│   │       ├── test_mesh_noc.cpp
│   │       └── test_router.cpp
│   ├── dma/                              # DMA controllers
│   │   ├── CMakeLists.txt
│   │   ├── include/stillwater/kpu/dma/
│   │   │   ├── dma_engine.hpp            # DMA engine interface
│   │   │   ├── scatter_gather.hpp        # Scatter-gather operations
│   │   │   ├── streaming_dma.hpp         # Streaming DMA
│   │   │   └── dma_scheduler.hpp         # DMA request scheduling
│   │   ├── src/
│   │   │   ├── dma_engine.cpp
│   │   │   ├── scatter_gather.cpp
│   │   │   ├── streaming_dma.cpp
│   │   │   └── dma_scheduler.cpp
│   │   └── tests/
│   │       ├── test_dma_engine.cpp
│   │       └── test_streaming_dma.cpp
│   └── power/                            # Power modeling
│       ├── CMakeLists.txt
│       ├── include/stillwater/kpu/power/
│       │   ├── power_model.hpp           # Power modeling interface
│       │   ├── thermal_model.hpp         # Thermal simulation
│       │   └── energy_counter.hpp        # Energy accounting
│       ├── src/
│       │   ├── power_model.cpp
│       │   ├── thermal_model.cpp
│       │   └── energy_counter.cpp
│       └── tests/
│           └── test_power_model.cpp
├── tools/                                # Development tools and utilities
│   ├── CMakeLists.txt
│   ├── cpp/                              # C++ tools
│   │   ├── CMakeLists.txt
│   │   ├── kpu-config/                   # Configuration tool
│   │   │   ├── CMakeLists.txt
│   │   │   └── kpu_config.cpp
│   │   ├── kpu-profiler/                 # Performance profiler
│   │   │   ├── CMakeLists.txt
│   │   │   ├── profiler.cpp
│   │   │   └── profiler.hpp
│   │   ├── kpu-assembler/                # Assembly language tools
│   │   │   ├── CMakeLists.txt
│   │   │   ├── assembler.cpp
│   │   │   ├── parser.hpp
│   │   │   └── instruction_set.hpp
│   │   ├── kpu-debugger/                 # Interactive debugger
│   │   │   ├── CMakeLists.txt
│   │   │   ├── debugger.cpp
│   │   │   ├── breakpoint.hpp
│   │   │   └── symbol_table.hpp
│   │   └── kpu-benchmark/                # Benchmarking suite
│   │       ├── CMakeLists.txt
│   │       ├── benchmark.cpp
│   │       ├── matrix_benchmarks.cpp
│   │       └── memory_benchmarks.cpp
│   ├── python/                           # Python tools
│   │   ├── setup.py                      # Python package setup
│   │   ├── stillwater_kpu/               # Python package
│   │   │   ├── __init__.py
│   │   │   ├── simulator.py              # High-level Python API
│   │   │   ├── visualization/            # Visualization tools
│   │   │   │   ├── __init__.py
│   │   │   │   ├── performance_plot.py
│   │   │   │   ├── memory_heatmap.py
│   │   │   │   └── network_topology.py
│   │   │   ├── analysis/                 # Analysis tools
│   │   │   │   ├── __init__.py
│   │   │   │   ├── performance_analyzer.py
│   │   │   │   ├── power_analyzer.py
│   │   │   │   └── bottleneck_finder.py
│   │   │   ├── generators/               # Code generators
│   │   │   │   ├── __init__.py
│   │   │   │   ├── workload_generator.py
│   │   │   │   ├── test_generator.py
│   │   │   │   └── config_generator.py
│   │   │   └── utilities/                # Utility functions
│   │   │       ├── __init__.py
│   │   │       ├── data_converter.py
│   │   │       ├── file_parser.py
│   │   │       └── report_generator.py
│   │   └── scripts/                      # Standalone scripts
│   │       ├── run_benchmarks.py
│   │       ├── generate_report.py
│   │       ├── compare_configurations.py
│   │       └── batch_simulation.py
│   └── matlab/                           # MATLAB integration
│       ├── +stillwater/                  # MATLAB package
│       │   ├── KPUSimulator.m
│       │   └── plot_performance.m
│       └── examples/
│           ├── basic_simulation.m
│           └── performance_analysis.m
├── tests/                                # Integration and system tests
│   ├── CMakeLists.txt
│   ├── integration/                      # Integration tests
│   │   ├── CMakeLists.txt
│   │   ├── test_end_to_end.cpp
│   │   ├── test_multi_component.cpp
│   │   └── test_python_cpp_integration.cpp
│   ├── performance/                      # Performance tests
│   │   ├── CMakeLists.txt
│   │   ├── benchmark_matrix_ops.cpp
│   │   ├── benchmark_memory_hierarchy.cpp
│   │   └── stress_test.cpp
│   ├── regression/                       # Regression tests
│   │   ├── CMakeLists.txt
│   │   ├── golden_results/               # Reference outputs
│   │   └── test_regression.cpp
│   └── data/                             # Test data files
│       ├── matrices/
│       ├── configurations/
│       └── workloads/
├── examples/                             # Example applications
│   ├── CMakeLists.txt
│   ├── basic/                            # Basic usage examples
│   │   ├── CMakeLists.txt
│   │   ├── hello_kpu.cpp                 # Simple first program
│   │   ├── matrix_multiply.cpp           # Basic matrix operations
│   │   └── memory_management.cpp        # Memory usage patterns
│   ├── advanced/                         # Advanced examples
│   │   ├── CMakeLists.txt
│   │   ├── neural_network.cpp            # Neural network simulation
│   │   ├── convolution.cpp               # Convolution operations
│   │   ├── streaming_workload.cpp        # Streaming data processing
│   │   └── multi_kpu.cpp                 # Multi-device simulation
│   ├── python/                           # Python examples
│   │   ├── basic_usage.py
│   │   ├── neural_network.py
│   │   ├── performance_analysis.py
│   │   └── visualization_demo.py
│   └── tutorials/                        # Step-by-step tutorials
│       ├── 01_getting_started/
│       ├── 02_memory_hierarchy/
│       ├── 03_compute_operations/
│       ├── 04_performance_optimization/
│       └── 05_custom_components/
├── benchmarks/                           # Standard benchmarks
│   ├── CMakeLists.txt
│   ├── mlperf/                          # MLPerf benchmarks
│   ├── rodinia/                         # Rodinia benchmark suite
│   ├── custom/                          # Custom KPU benchmarks
│   └── reference_results/               # Baseline results
├── third_party/                         # External dependencies
│   ├── CMakeLists.txt
│   ├── catch2/                          # Testing framework
│   ├── spdlog/                          # Logging library
│   ├── nlohmann_json/                   # JSON library
│   ├── pybind11/                        # Python bindings
│   └── fmt/                             # Formatting library
├── CMakePresets.json                   # CMake preset configurations
├── scripts/                             # Development utility scripts
│   ├── format.sh                        # Code formatting (clang-format)
│   ├── format.ps1                       # Code formatting (Windows)
│   ├── analyze.sh                       # Static analysis runner
│   ├── analyze.ps1                      # Static analysis (Windows)
│   └── ci/                              # CI/CD helper scripts
│       ├── setup-deps.sh                # CI dependency setup
│       └── run-tests.sh                 # CI test execution
└── packaging/                           # Package configuration
    ├── CMakeLists.txt
    ├── debian/                          # Debian packages
    │   ├── control
    │   ├── rules
    │   └── changelog
    ├── rpm/                             # RPM packages
    │   └── stillwater-kpu.spec
    ├── docker/                          # Docker containers
    │   ├── Dockerfile.dev               # Development environment
    │   ├── Dockerfile.runtime           # Runtime environment
    │   └── docker-compose.yml           # Multi-container setup
    └── conda/                           # Conda packages
        ├── meta.yaml
        └── build.sh
```