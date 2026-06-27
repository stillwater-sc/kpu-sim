// MNIST Digit Classifier Example
// Demonstrates a 3-layer MLP for handwritten digit recognition
//
// Network architecture: 784 -> 128 -> 64 -> 10
// - Input: 28x28 grayscale image flattened to 784 values
// - Hidden 1: 128 neurons with ReLU
// - Hidden 2: 64 neurons with ReLU
// - Output: 10 neurons (one per digit class)
//
// This example uses simplified "digit patterns" for demonstration.
// For real MNIST, you would load actual test images and trained weights.

#include <sw/runtime/graph.hpp>
#include <sw/kpu/kpu_simulator.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

using namespace sw::runtime;
using namespace sw::kpu;

// Network dimensions
constexpr Size INPUT_DIM = 784;   // 28x28
constexpr Size HIDDEN1_DIM = 128;
constexpr Size HIDDEN2_DIM = 64;
constexpr Size OUTPUT_DIM = 10;

// Generate pseudo-random weights with a specific seed for reproducibility
// In a real application, these would be loaded from a trained model
class WeightGenerator {
public:
    WeightGenerator(unsigned seed = 42) : rng_(seed), dist_(-0.1f, 0.1f) {}

    float next() { return dist_(rng_); }

    void fill(float* arr, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            arr[i] = next();
        }
    }

    // Xavier initialization scaled weights
    void fill_xavier(float* arr, size_t fan_in, size_t fan_out) {
        float scale = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> normal(0.0f, scale);
        for (size_t i = 0; i < fan_in * fan_out; ++i) {
            arr[i] = normal(rng_);
        }
    }

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
};

// Simple digit patterns (7x7 downsampled, then upscaled to 28x28)
// These are simplified representations of digits for testing
void create_digit_pattern(int digit, float* output) {
    // Initialize to zero
    std::fill(output, output + 784, 0.0f);

    // 7x7 patterns scaled to 28x28 (each "pixel" is 4x4)
    // We'll define simple patterns for digits 0-9

    auto set_block = [&output](int row, int col, float value) {
        // Set a 4x4 block in the 28x28 image
        for (int dr = 0; dr < 4; ++dr) {
            for (int dc = 0; dc < 4; ++dc) {
                int r = row * 4 + dr;
                int c = col * 4 + dc;
                if (r < 28 && c < 28) {
                    output[r * 28 + c] = value;
                }
            }
        }
    };

    switch (digit) {
        case 0:
            // Oval shape
            for (int i = 1; i < 6; ++i) {
                set_block(i, 1, 1.0f);  // Left edge
                set_block(i, 5, 1.0f);  // Right edge
            }
            for (int j = 2; j < 5; ++j) {
                set_block(0, j, 1.0f);  // Top
                set_block(6, j, 1.0f);  // Bottom
            }
            break;

        case 1:
            // Vertical line
            for (int i = 0; i < 7; ++i) {
                set_block(i, 3, 1.0f);
            }
            set_block(6, 2, 0.8f);
            set_block(6, 4, 0.8f);
            break;

        case 2:
            // 2 shape
            for (int j = 1; j < 6; ++j) set_block(0, j, 1.0f);  // Top
            set_block(1, 5, 1.0f);
            set_block(2, 4, 1.0f);
            set_block(3, 3, 1.0f);
            set_block(4, 2, 1.0f);
            set_block(5, 1, 1.0f);
            for (int j = 1; j < 6; ++j) set_block(6, j, 1.0f);  // Bottom
            break;

        case 3:
            // 3 shape
            for (int j = 1; j < 5; ++j) {
                set_block(0, j, 1.0f);  // Top
                set_block(3, j, 1.0f);  // Middle
                set_block(6, j, 1.0f);  // Bottom
            }
            for (int i = 1; i < 6; ++i) set_block(i, 5, 1.0f);  // Right
            break;

        case 4:
            // 4 shape
            for (int i = 0; i < 4; ++i) set_block(i, 1, 1.0f);  // Left upper
            for (int j = 1; j < 6; ++j) set_block(3, j, 1.0f);  // Middle
            for (int i = 0; i < 7; ++i) set_block(i, 5, 1.0f);  // Right
            break;

        case 5:
            // 5 shape
            for (int j = 1; j < 6; ++j) set_block(0, j, 1.0f);  // Top
            set_block(1, 1, 1.0f);
            set_block(2, 1, 1.0f);
            for (int j = 1; j < 5; ++j) set_block(3, j, 1.0f);  // Middle
            set_block(4, 5, 1.0f);
            set_block(5, 5, 1.0f);
            for (int j = 1; j < 6; ++j) set_block(6, j, 1.0f);  // Bottom
            break;

        case 6:
            // 6 shape
            for (int i = 1; i < 6; ++i) set_block(i, 1, 1.0f);  // Left
            for (int j = 2; j < 5; ++j) {
                set_block(0, j, 1.0f);  // Top
                set_block(3, j, 1.0f);  // Middle
                set_block(6, j, 1.0f);  // Bottom
            }
            for (int i = 4; i < 6; ++i) set_block(i, 5, 1.0f);  // Right bottom
            break;

        case 7:
            // 7 shape
            for (int j = 1; j < 6; ++j) set_block(0, j, 1.0f);  // Top
            for (int i = 1; i < 7; ++i) {
                int col = 5 - (i / 2);  // Slight diagonal
                set_block(i, col, 1.0f);
            }
            break;

        case 8:
            // 8 shape (two stacked circles)
            for (int i = 1; i < 6; ++i) {
                if (i != 3) {
                    set_block(i, 1, 1.0f);  // Left
                    set_block(i, 5, 1.0f);  // Right
                }
            }
            for (int j = 2; j < 5; ++j) {
                set_block(0, j, 1.0f);  // Top
                set_block(3, j, 1.0f);  // Middle
                set_block(6, j, 1.0f);  // Bottom
            }
            set_block(3, 1, 1.0f);
            set_block(3, 5, 1.0f);
            break;

        case 9:
            // 9 shape
            for (int i = 0; i < 4; ++i) set_block(i, 1, 1.0f);  // Left upper
            for (int i = 1; i < 7; ++i) set_block(i, 5, 1.0f);  // Right
            for (int j = 2; j < 5; ++j) {
                set_block(0, j, 1.0f);  // Top
                set_block(3, j, 1.0f);  // Middle
            }
            for (int j = 2; j < 5; ++j) set_block(6, j, 1.0f);  // Bottom
            break;
    }
}

// Print a 28x28 image as ASCII art
void print_digit(const float* image, int width = 28, int height = 28) {
    for (int r = 0; r < height; r += 2) {
        std::cout << "    ";
        for (int c = 0; c < width; ++c) {
            float val = image[r * width + c];
            if (r + 1 < height) {
                val = std::max(val, image[(r + 1) * width + c]);
            }
            if (val > 0.5f) std::cout << "##";
            else if (val > 0.2f) std::cout << "..";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
}

// Softmax for output interpretation
void softmax(const float* input, float* output, size_t n) {
    float max_val = *std::max_element(input, input + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (size_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

int main() {
    std::cout << "================================================\n";
    std::cout << "MNIST Digit Classifier on KPU Simulator\n";
    std::cout << "================================================\n\n";

    std::cout << "Network architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10\n";
    std::cout << "Task: Classify 28x28 grayscale digit images\n\n";

    // Configure KPU simulator
    KPUSimulator::Config config;
    config.memory_bank_count = 4;
    config.memory_bank_capacity_mb = 128;
    config.l3_tile_count = 8;
    config.l3_tile_capacity_kb = 256;
    config.l2_bank_count = 16;
    config.l2_bank_capacity_kb = 64;
    config.l1_buffer_count = 8;
    config.l1_buffer_capacity_kb = 64;
    config.processor_array_rows = 16;
    config.processor_array_cols = 16;
    config.use_systolic_array_mode = true;

    KPUSimulator simulator(config);
    KPURuntime runtime(&simulator);

    // Generate weights (in practice, these would be loaded from a trained model)
    WeightGenerator gen(42);

    std::vector<float> layer1_weights(INPUT_DIM * HIDDEN1_DIM);
    std::vector<float> layer1_bias(HIDDEN1_DIM, 0.0f);
    gen.fill_xavier(layer1_weights.data(), INPUT_DIM, HIDDEN1_DIM);

    std::vector<float> layer2_weights(HIDDEN1_DIM * HIDDEN2_DIM);
    std::vector<float> layer2_bias(HIDDEN2_DIM, 0.0f);
    gen.fill_xavier(layer2_weights.data(), HIDDEN1_DIM, HIDDEN2_DIM);

    std::vector<float> layer3_weights(HIDDEN2_DIM * OUTPUT_DIM);
    std::vector<float> layer3_bias(OUTPUT_DIM, 0.0f);
    gen.fill_xavier(layer3_weights.data(), HIDDEN2_DIM, OUTPUT_DIM);

    // Build computational graph
    const Size BATCH_SIZE = 10;  // Process all 10 test digits at once

    ComputeGraph graph("mnist_classifier");

    // Input: [10, 784]
    graph.add_input("input", {BATCH_SIZE, INPUT_DIM});

    // Layer 1: [10, 784] @ [784, 128] + [128] -> [10, 128]
    auto layer1 = Kernel::create_mlp(
        BATCH_SIZE, HIDDEN1_DIM, INPUT_DIM,
        ActivationType::RELU, true);
    graph.add_node("fc1", layer1, {"input"}, {"h1"});

    // Layer 2: [10, 128] @ [128, 64] + [64] -> [10, 64]
    auto layer2 = Kernel::create_mlp(
        BATCH_SIZE, HIDDEN2_DIM, HIDDEN1_DIM,
        ActivationType::RELU, true);
    graph.add_node("fc2", layer2, {"h1"}, {"h2"});

    // Layer 3: [10, 64] @ [64, 10] + [10] -> [10, 10]
    auto layer3 = Kernel::create_mlp(
        BATCH_SIZE, OUTPUT_DIM, HIDDEN2_DIM,
        ActivationType::NONE, true);  // No activation - logits for softmax
    graph.add_node("fc3", layer3, {"h2"}, {"logits"});

    graph.add_output("logits");
    graph.finalize();

    std::cout << graph.summary() << "\n";

    // Create runner and allocate
    GraphRunner runner(&runtime);
    runner.set_graph(graph);
    runner.allocate();

    std::cout << "Memory allocated: " << runner.total_allocated_bytes() / 1024 << " KB\n\n";

    // Generate test digit images
    std::cout << "Generating test digit patterns...\n\n";
    std::vector<float> input_images(BATCH_SIZE * INPUT_DIM);
    for (int digit = 0; digit < 10; ++digit) {
        create_digit_pattern(digit, &input_images[digit * INPUT_DIM]);
    }

    // Show a few example digits
    std::cout << "Sample digit patterns:\n\n";
    std::cout << "Digit 0:\n";
    print_digit(&input_images[0 * INPUT_DIM]);
    std::cout << "\nDigit 3:\n";
    print_digit(&input_images[3 * INPUT_DIM]);
    std::cout << "\nDigit 7:\n";
    print_digit(&input_images[7 * INPUT_DIM]);
    std::cout << "\n";

    // Set inputs
    runner.set_input("input", input_images.data(), input_images.size() * sizeof(float));

    // Set weights
    Address fc1_B = runner.get_tensor_address("fc1_B");
    Address fc1_bias = runner.get_tensor_address("fc1_bias");
    if (fc1_B) runtime.memcpy_h2d(fc1_B, layer1_weights.data(), layer1_weights.size() * sizeof(float));
    if (fc1_bias) runtime.memcpy_h2d(fc1_bias, layer1_bias.data(), layer1_bias.size() * sizeof(float));

    Address fc2_B = runner.get_tensor_address("fc2_B");
    Address fc2_bias = runner.get_tensor_address("fc2_bias");
    if (fc2_B) runtime.memcpy_h2d(fc2_B, layer2_weights.data(), layer2_weights.size() * sizeof(float));
    if (fc2_bias) runtime.memcpy_h2d(fc2_bias, layer2_bias.data(), layer2_bias.size() * sizeof(float));

    Address fc3_B = runner.get_tensor_address("fc3_B");
    Address fc3_bias = runner.get_tensor_address("fc3_bias");
    if (fc3_B) runtime.memcpy_h2d(fc3_B, layer3_weights.data(), layer3_weights.size() * sizeof(float));
    if (fc3_bias) runtime.memcpy_h2d(fc3_bias, layer3_bias.data(), layer3_bias.size() * sizeof(float));

    std::cout << "Running MNIST classifier on KPU...\n";

    // Execute
    auto result = runner.run();

    if (!result.success) {
        std::cerr << "Execution failed: " << result.error << "\n";
        return 1;
    }

    std::cout << "\n================================================\n";
    std::cout << "Execution Complete!\n";
    std::cout << "================================================\n\n";

    std::cout << "Performance metrics:\n";
    std::cout << "  Total cycles: " << result.total_cycles << "\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(4)
              << result.total_time_ms << " ms\n";
    std::cout << "  Throughput: " << std::setprecision(2)
              << (BATCH_SIZE / result.total_time_ms * 1000) << " images/sec\n\n";

    std::cout << "Per-layer breakdown:\n";
    for (const auto& profile : result.layer_profiles) {
        double pct = 100.0 * profile.cycles / result.total_cycles;
        std::cout << "  " << std::setw(8) << profile.name << ": "
                  << std::setw(10) << profile.cycles << " cycles ("
                  << std::setw(5) << std::setprecision(1) << pct << "%)\n";
    }
    std::cout << "\n";

    // Get logits output
    std::vector<float> logits(BATCH_SIZE * OUTPUT_DIM);
    runner.get_output("logits", logits.data());

    // Compute predictions
    std::cout << "Classification Results:\n";
    std::cout << "  Digit  Predicted  Confidence  Status\n";
    std::cout << "  -----  ---------  ----------  ------\n";

    int correct = 0;
    for (int digit = 0; digit < 10; ++digit) {
        // Apply softmax
        std::vector<float> probs(OUTPUT_DIM);
        softmax(&logits[digit * OUTPUT_DIM], probs.data(), OUTPUT_DIM);

        // Find prediction
        int predicted = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
        float confidence = probs[predicted];

        bool is_correct = (predicted == digit);
        if (is_correct) correct++;

        std::cout << "    " << digit << "        " << predicted
                  << "        " << std::setw(6) << std::setprecision(2)
                  << (confidence * 100) << "%     "
                  << (is_correct ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\nAccuracy: " << correct << "/10 (" << (correct * 10) << "%)\n\n";

    // Note about the demo
    std::cout << "================================================\n";
    std::cout << "Note: This demo uses randomly initialized weights.\n";
    std::cout << "With properly trained weights (e.g., from PyTorch),\n";
    std::cout << "accuracy would be ~98% on real MNIST test data.\n";
    std::cout << "\n";
    std::cout << "To use trained weights:\n";
    std::cout << "  1. Train model in PyTorch/TensorFlow\n";
    std::cout << "  2. Export weights to binary format\n";
    std::cout << "  3. Load weights before execution\n";
    std::cout << "================================================\n";

    return 0;
}
