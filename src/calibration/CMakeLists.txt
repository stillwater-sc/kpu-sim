# src/calibration/CMakeLists.txt
#
# Calibration storage library for multi-fidelity memory controller models
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

add_library(kpu_calibration STATIC
    calibration_storage.cpp
)

target_include_directories(kpu_calibration PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kpu_calibration PUBLIC
    nlohmann_json::nlohmann_json
)

# Require C++17 for std::filesystem and std::optional
target_compile_features(kpu_calibration PUBLIC cxx_std_17)
