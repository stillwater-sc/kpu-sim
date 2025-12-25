#pragma once

#include <string>
#include <filesystem>

namespace sw::test {

/**
 * @brief Get the directory for test output files
 *
 * Creates and returns a path to a temporary directory for test artifacts.
 * The directory is created under the system temp directory with a
 * "kpu_sim_test_output" subdirectory.
 *
 * @return Path to the test output directory
 */
inline std::filesystem::path get_test_output_dir() {
    auto temp_dir = std::filesystem::temp_directory_path() / "kpu_sim_test_output";

    // Create directory if it doesn't exist
    if (!std::filesystem::exists(temp_dir)) {
        std::filesystem::create_directories(temp_dir);
    }

    return temp_dir;
}

/**
 * @brief Get a full path for a test output file
 *
 * @param filename The filename (without path)
 * @return Full path in the test output directory
 */
inline std::string get_test_output_path(const std::string& filename) {
    return (get_test_output_dir() / filename).string();
}

/**
 * @brief Clean up test output files
 *
 * Removes all files in the test output directory.
 * Useful for cleanup after tests or before a test run.
 */
inline void cleanup_test_outputs() {
    auto dir = get_test_output_dir();
    if (std::filesystem::exists(dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            std::filesystem::remove(entry.path());
        }
    }
}

} // namespace sw::test
