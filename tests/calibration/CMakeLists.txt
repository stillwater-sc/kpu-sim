# tests/calibration/CMakeLists.txt
#
# Tests for calibration storage, extraction, and quality assessment
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

# Calibration storage test (standalone, only needs calibration library)
add_executable(calibration_storage_test
    calibration_storage_test.cpp
)
target_link_libraries(calibration_storage_test PRIVATE
    kpu_calibration
    Catch2::Catch2WithMain
)
add_test(NAME calibration_storage_test COMMAND calibration_storage_test)

# Calibration extraction test (needs full simulator for LPDDR5 Statistics)
add_executable(calibration_extraction_test
    calibration_extraction_test.cpp
)
target_link_libraries(calibration_extraction_test PRIVATE
    kpu_calibration
    kpu_memory_components
    Catch2::Catch2WithMain
)
add_test(NAME calibration_extraction_test COMMAND calibration_extraction_test)

# Calibration quality test (standalone, header-only quality assessment)
add_executable(calibration_quality_test
    calibration_quality_test.cpp
)
target_link_libraries(calibration_quality_test PRIVATE
    kpu_calibration
    Catch2::Catch2WithMain
)
add_test(NAME calibration_quality_test COMMAND calibration_quality_test)
