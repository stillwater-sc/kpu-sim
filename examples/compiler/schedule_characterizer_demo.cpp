/**
 * @file schedule_characterizer_demo.cpp
 * @brief Demonstration of Pareto frontier characterization for scheduling strategies
 *
 * This tool evaluates weight-stationary, input-stationary, and output-stationary
 * dataflow strategies across realistic tensor workloads to identify the Pareto
 * frontier of energy-latency trade-offs.
 */

#include <sw/compiler/schedule_characterizer.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace sw::kpu::compiler;

void demo_small_characterization() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         DEMO 1: Small-Scale Characterization (100 workloads)       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Generate small set of workloads
    auto workloads = WorkloadGenerator::generate_ml_workloads(100, "real-world");

    std::cout << "Generated " << workloads.size() << " workloads\n";
    std::cout << "Sample workloads:\n";
    for (size_t i = 0; i < std::min(size_t(5), workloads.size()); ++i) {
        std::cout << "  " << workloads[i].to_string() << "\n";
    }
    std::cout << "\n";

    // Create characterizer
    ScheduleCharacterizer characterizer;

    // Characterize
    auto start = std::chrono::high_resolution_clock::now();
    auto frontier = characterizer.characterize_workloads(workloads);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nCharacterization time: " << duration.count() << " ms\n";

    // Print results
    characterizer.print_summary(frontier);

    // Export ALL evaluations (not just Pareto frontier) for better design space visualization
    characterizer.export_csv(frontier.all_evaluations, "pareto_frontier_small.csv");
}

void demo_network_layers() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        DEMO 2: Popular Network Layers Characterization             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Generate from popular networks
    auto workloads = WorkloadGenerator::generate_from_networks({
        "resnet50", "vgg16", "bert", "gpt2", "mobilenet"
    });

    std::cout << "Generated " << workloads.size() << " layer shapes from networks\n";
    std::cout << "All workloads:\n";
    for (const auto& shape : workloads) {
        std::cout << "  " << shape.to_string() << "\n";
    }
    std::cout << "\n";

    ScheduleCharacterizer characterizer;

    auto start = std::chrono::high_resolution_clock::now();
    auto frontier = characterizer.characterize_workloads(workloads);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nCharacterization time: " << duration.count() << " ms\n";

    characterizer.print_summary(frontier);
    characterizer.export_csv(frontier.all_evaluations, "pareto_frontier_networks.csv");
}

void demo_parameter_sweep() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           DEMO 3: Parameter Space Sweep (1000 points)             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Sweep M, N, K independently
    auto workloads = WorkloadGenerator::generate_sweep(
        64, 1024, 64,    // M: 64 to 1024, step 64
        64, 1024, 128,   // N: 64 to 1024, step 128
        64, 512, 64      // K: 64 to 512, step 64
    );

    std::cout << "Generated " << workloads.size() << " workloads from parameter sweep\n";
    std::cout << "M range: 64-1024 (step 64), N range: 64-1024 (step 128), K range: 64-512 (step 64)\n";
    std::cout << "\n";

    ScheduleCharacterizer characterizer;

    auto start = std::chrono::high_resolution_clock::now();
    auto frontier = characterizer.characterize_workloads(workloads);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nCharacterization time: " << duration.count() << " ms\n";

    characterizer.print_summary(frontier);
    characterizer.export_csv(frontier.all_evaluations, "pareto_frontier_sweep.csv");
}

void demo_large_scale_characterization() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      DEMO 4: Large-Scale Characterization (10,000 workloads)      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "WARNING: This will take several minutes...\n";
    std::cout << "Press Enter to continue or Ctrl+C to skip...";
    std::cin.get();

    // Generate large set
    auto workloads = WorkloadGenerator::generate_ml_workloads(10000, "real-world");

    std::cout << "Generated " << workloads.size() << " workloads\n\n";

    ScheduleCharacterizer characterizer;

    auto start = std::chrono::high_resolution_clock::now();
    auto frontier = characterizer.characterize_workloads(workloads);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "\nCharacterization time: " << duration.count() << " seconds\n";

    characterizer.print_summary(frontier);
    characterizer.export_csv(frontier.all_evaluations, "pareto_frontier_large.csv");
}

void demo_slowdown_analysis() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          DEMO 5: Slowdown Analysis (Non-Ideal Tensors)             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "Analyzing slowdown for non-multiples of systolic array size...\n\n";

    // Generate tensors that are NOT nice multiples of 16
    std::vector<TensorShape> workloads = {
        {17, 17, 17},      // Just off by 1
        {100, 100, 100},   // Not aligned at all
        {127, 127, 127},   // Prime-ish
        {255, 255, 255},   // Almost 256
        {1000, 1000, 1000} // Large unaligned
    };

    // Add some aligned ones for comparison
    workloads.push_back({16, 16, 16});
    workloads.push_back({128, 128, 128});
    workloads.push_back({256, 256, 256});
    workloads.push_back({1024, 1024, 1024});

    ScheduleCharacterizer characterizer;
    auto frontier = characterizer.characterize_workloads(workloads);

    characterizer.print_summary(frontier);

    // Detailed slowdown analysis
    std::cout << "\nSlowdown Analysis:\n";
    std::cout << "  ┌──────────────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "  │      Shape       │ Energy Slow. │ Latency Slow.│   Aligned?   │\n";
    std::cout << "  ├──────────────────┼──────────────┼──────────────┼──────────────┤\n";

    for (const auto& pt : frontier.points) {
        bool aligned = (pt.schedule->shape.M % 16 == 0) &&
                      (pt.schedule->shape.N % 16 == 0) &&
                      (pt.schedule->shape.K % 16 == 0);

        std::cout << "  │ " << std::setw(16) << pt.schedule->shape.to_string() << " │ "
                  << std::setw(11) << std::fixed << std::setprecision(2)
                  << pt.schedule->energy_slowdown << "× │ "
                  << std::setw(11) << pt.schedule->latency_slowdown << "× │ "
                  << std::setw(12) << (aligned ? "Yes" : "No") << " │\n";
    }

    std::cout << "  └──────────────────┴──────────────┴──────────────┴──────────────┘\n";
}

void demo_strategy_comparison() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        DEMO 6: Dataflow Strategy Comparison (Same Tensors)        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Single tensor, all strategies
    std::vector<TensorShape> workloads = {{512, 512, 512}};

    ScheduleCharacterizer characterizer;

    std::vector<DataflowStrategy> strategies = {
        DataflowStrategy::WEIGHT_STATIONARY,
        DataflowStrategy::INPUT_STATIONARY,
        DataflowStrategy::OUTPUT_STATIONARY
    };

    std::cout << "Evaluating 512×512×512 with all three dataflow strategies...\n\n";

    auto evaluations = characterizer.evaluate_all_strategies(workloads[0]);

    std::cout << "Strategy Comparison:\n";
    std::cout << "  ┌──────────────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "  │    Strategy      │  Energy (pJ) │ Latency (cyc)│   Reuse A/B  │\n";
    std::cout << "  ├──────────────────┼──────────────┼──────────────┼──────────────┤\n";

    for (const auto& eval : evaluations) {
        std::cout << "  │ " << std::setw(16)
                  << ScheduleEvaluation::dataflow_to_string(eval.strategy) << " │ "
                  << std::setw(12) << std::fixed << std::setprecision(1)
                  << eval.metrics.total_energy << " │ "
                  << std::setw(12) << eval.metrics.total_cycles << " │ "
                  << std::setw(6) << eval.metrics.reuse_A << "/"
                  << std::setw(6) << eval.metrics.reuse_B << " │\n";
    }

    std::cout << "  └──────────────────┴──────────────┴──────────────┴──────────────┘\n";

    std::cout << "\nNote: Currently all strategies use output-stationary implementation.\n";
    std::cout << "Full WS/IS implementations are planned for future work.\n";
}

int main(int argc, char** argv) {
    (void)argc;  // No command-line arguments currently used
    (void)argv;
    std::cout << "\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█            SCHEDULE CHARACTERIZATION & PARETO FRONTIER ANALYSIS          █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█  This tool evaluates scheduling strategies across realistic workloads    █\n";
    std::cout << "█  to identify the Pareto frontier of energy-latency trade-offs.           █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";

    try {
        demo_small_characterization();
        demo_network_layers();
        demo_slowdown_analysis();
        demo_strategy_comparison();
        demo_parameter_sweep();

        // Optional: large scale (commented out by default)
        // demo_large_scale_characterization();

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                   ALL CHARACTERIZATIONS COMPLETE                   ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        std::cout << "CSV files generated:\n";
        std::cout << "  - pareto_frontier_small.csv (100 workloads)\n";
        std::cout << "  - pareto_frontier_networks.csv (network layers)\n";
        std::cout << "  - pareto_frontier_sweep.csv (parameter sweep)\n";
        std::cout << "\n";
        std::cout << "Use Python/matplotlib or similar tools to visualize the Pareto frontier.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
