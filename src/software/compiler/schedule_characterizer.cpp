/**
 * @file schedule_characterizer.cpp
 * @brief Implementation of schedule characterization framework
 */

#include <sw/compiler/schedule_characterizer.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace sw::kpu::compiler {

// ============================================================================
// WorkloadGenerator Implementation
// ============================================================================

std::vector<TensorShape> WorkloadGenerator::generate_ml_workloads(
    size_t count,
    const std::string& distribution)
{
    std::vector<TensorShape> shapes;
    shapes.reserve(count);

    std::random_device rd;
    std::mt19937 gen(rd());

    if (distribution == "uniform") {
        // Uniform distribution across reasonable sizes
        std::uniform_int_distribution<Size> dist_log(4, 12);  // 2^4 to 2^12

        for (size_t i = 0; i < count; ++i) {
            Size M = 1 << dist_log(gen);
            Size N = 1 << dist_log(gen);
            Size K = 1 << dist_log(gen);
            shapes.emplace_back(M, N, K);
        }
    }
    else if (distribution == "power-law") {
        // Power-law distribution (more small tensors)
        std::exponential_distribution<> exp_dist(0.3);

        for (size_t i = 0; i < count; ++i) {
            Size M = std::min(Size(4096), Size(64 * (1 + std::abs(exp_dist(gen)))));
            Size N = std::min(Size(4096), Size(64 * (1 + std::abs(exp_dist(gen)))));
            Size K = std::min(Size(4096), Size(64 * (1 + std::abs(exp_dist(gen)))));
            shapes.emplace_back(M, N, K);
        }
    }
    else { // "real-world"
        // Mix of common ML layer patterns
        std::vector<Size> common_batch = {1, 4, 8, 16, 32, 64, 128, 256};
        std::vector<Size> common_features = {64, 128, 256, 512, 768, 1024, 2048, 4096};
        std::vector<Size> common_hidden = {256, 512, 768, 1024, 1536, 2048, 3072, 4096};

        std::uniform_int_distribution<> batch_dist(0, common_batch.size() - 1);
        std::uniform_int_distribution<> feat_dist(0, common_features.size() - 1);
        std::uniform_int_distribution<> hidden_dist(0, common_hidden.size() - 1);

        for (size_t i = 0; i < count; ++i) {
            Size M = common_batch[batch_dist(gen)];
            Size N = common_features[feat_dist(gen)];
            Size K = common_hidden[hidden_dist(gen)];
            shapes.emplace_back(M, N, K);
        }
    }

    return shapes;
}

std::vector<TensorShape> WorkloadGenerator::generate_sweep(
    Size m_min, Size m_max, Size m_step,
    Size n_min, Size n_max, Size n_step,
    Size k_min, Size k_max, Size k_step)
{
    std::vector<TensorShape> shapes;

    for (Size M = m_min; M <= m_max; M += m_step) {
        for (Size N = n_min; N <= n_max; N += n_step) {
            for (Size K = k_min; K <= k_max; K += k_step) {
                shapes.emplace_back(M, N, K);
            }
        }
    }

    return shapes;
}

std::vector<TensorShape> WorkloadGenerator::generate_from_networks(
    const std::vector<std::string>& networks)
{
    std::vector<TensorShape> shapes;

    for (const auto& network : networks) {
        if (network == "resnet50") {
            // ResNet-50 FC layer and representative conv layers
            shapes.emplace_back(1, 1000, 2048);    // FC
            shapes.emplace_back(64, 56*56, 64);    // conv1
            shapes.emplace_back(64, 56*56, 256);   // conv2_x
            shapes.emplace_back(128, 28*28, 512);  // conv3_x
            shapes.emplace_back(256, 14*14, 1024); // conv4_x
            shapes.emplace_back(512, 7*7, 2048);   // conv5_x
        }
        else if (network == "vgg16") {
            shapes.emplace_back(1, 4096, 4096);    // FC layers
            shapes.emplace_back(1, 1000, 4096);    // Final FC
            shapes.emplace_back(512, 14*14, 512);  // Conv layers
        }
        else if (network == "bert") {
            // BERT-base: 768 hidden, 12 heads
            shapes.emplace_back(128, 768, 768);    // Q, K, V projections
            shapes.emplace_back(128, 768, 3072);   // FFN expansion
            shapes.emplace_back(128, 3072, 768);   // FFN reduction
        }
        else if (network == "gpt2") {
            // GPT-2 medium: 1024 hidden
            shapes.emplace_back(1024, 1024, 1024); // Self-attention
            shapes.emplace_back(1024, 1024, 4096); // FFN expansion
            shapes.emplace_back(1024, 4096, 1024); // FFN reduction
        }
        else if (network == "mobilenet") {
            shapes.emplace_back(1, 1000, 1024);    // FC
            shapes.emplace_back(32, 112*112, 32);  // Depthwise separable
            shapes.emplace_back(64, 56*56, 128);
        }
    }

    return shapes;
}

// ============================================================================
// ScheduleCharacterizer Implementation
// ============================================================================

ScheduleCharacterizer::ScheduleCharacterizer(
    const TileOptimizer::MemoryHierarchy& mem,
    const EnergyModel& energy,
    const LatencyModel& latency)
    : memory_(mem)
    , energy_model_(energy)
    , latency_model_(latency)
    , optimizer_(mem)
    , scheduler_(mem)
{
}

IdealMetrics ScheduleCharacterizer::calculate_ideal(const TensorShape& shape) {
    IdealMetrics ideal;

    Size M = shape.M, N = shape.N, K = shape.K;

    // Ideal cycles: assuming perfect PE utilization
    Size systolic_dim = std::min(memory_.systolic_rows, memory_.systolic_cols);
    ideal.ideal_cycles = (M * N * K) / (systolic_dim * systolic_dim);

    // Ideal energy: only compute cost + minimum data movement
    Size total_macs = 2 * M * N * K;
    ideal.ideal_energy = total_macs * energy_model_.mac_pj;

    // Add minimum data movement (read A, B once, write C once)
    Size min_dram_bytes = (M * K + K * N + M * N) * memory_.element_size;
    ideal.ideal_energy += min_dram_bytes * energy_model_.dram_read_pj;

    // Peak throughput
    Size peak_ops_per_cycle = memory_.systolic_rows * memory_.systolic_cols * 2; // 2 for MAC
    ideal.peak_throughput = peak_ops_per_cycle * latency_model_.clock_freq_ghz * 1e9;

    return ideal;
}

double ScheduleCharacterizer::calculate_energy(
    const TensorShape& shape,
    const L2TileScheduler::L2Schedule& schedule)
{
    double energy = 0;

    // DRAM energy (L3 misses)
    energy += schedule.l3_misses * schedule.config.Ti * schedule.config.Tk *
              memory_.element_size * energy_model_.dram_read_pj;

    // L3 energy (L2 loads)
    energy += schedule.total_bytes_loaded * energy_model_.l3_read_pj;

    // L2 energy (all accesses - approximate from total loads)
    Size l2_accesses = schedule.total_bytes_loaded * 2; // Read + some writes
    energy += l2_accesses * energy_model_.l2_read_pj;

    // L1 energy (streaming to systolic array)
    Size l1_accesses = 2 * shape.M * shape.N * shape.K;  // Approximate
    energy += l1_accesses * energy_model_.l1_read_pj;

    // Compute energy (MACs)
    Size total_macs = 2 * shape.M * shape.N * shape.K;
    energy += total_macs * energy_model_.mac_pj;

    return energy;
}

Cycle ScheduleCharacterizer::calculate_latency(
    const TensorShape& shape,
    const L2TileScheduler::L2Schedule& schedule)
{
    Cycle latency = 0;

    // Compute cycles (systolic array)
    // Size num_c_tiles = schedule.num_tile_rows_C * schedule.num_tile_cols_C;  // Reserved for future use
    Size num_k_tiles = schedule.num_tile_cols_A;

    for (Size ti = 0; ti < schedule.num_tile_rows_C; ++ti) {
        for (Size tj = 0; tj < schedule.num_tile_cols_C; ++tj) {
            Size tile_m = std::min(schedule.config.Ti, shape.M - ti * schedule.config.Ti);
            Size tile_n = std::min(schedule.config.Tj, shape.N - tj * schedule.config.Tj);

            for (Size tk = 0; tk < num_k_tiles; ++tk) {
                Size tile_k = std::min(schedule.config.Tk, shape.K - tk * schedule.config.Tk);

                // Systolic array latency: k + max(m, n)
                Cycle tile_cycles = tile_k + std::max(tile_m, tile_n);
                latency += tile_cycles;
            }
        }
    }

    // Add data movement cycles (simplified: assume overlapped with compute)
    // In reality, use pipelining model
    Cycle dram_cycles = (schedule.l3_misses * schedule.config.Ti * schedule.config.Tk *
                        memory_.element_size) / (latency_model_.dram_bandwidth * 1e9 /
                        latency_model_.clock_freq_ghz / 1e9);
    latency += dram_cycles / 10;  // Assume 90% overlap

    return latency;
}

double ScheduleCharacterizer::calculate_utilization(
    const TensorShape& shape,
    const TileOptimizer::TileConfig& config)
{
    (void)shape;  // Reserved for shape-aware utilization calculation

    // Calculate how well tiles fill the systolic array
    // Size systolic_area = memory_.systolic_rows * memory_.systolic_cols;  // Reserved for future use

    // Average tile utilization
    double util_m = std::min(1.0, (double)config.Ti / memory_.systolic_rows);
    double util_n = std::min(1.0, (double)config.Tj / memory_.systolic_cols);

    return util_m * util_n;
}

ScheduleEvaluation ScheduleCharacterizer::evaluate_schedule(
    const TensorShape& shape,
    DataflowStrategy strategy)
{
    ScheduleEvaluation eval;
    eval.shape = shape;
    eval.strategy = strategy;

    // *** KEY CHANGE: Select tile optimizer based on strategy ***
    switch (strategy) {
        case DataflowStrategy::OUTPUT_STATIONARY:
            eval.tile_config = optimizer_.optimize(shape.M, shape.N, shape.K);
            break;

        case DataflowStrategy::WEIGHT_STATIONARY:
            eval.tile_config = optimizer_.optimize_weight_stationary(shape.M, shape.N, shape.K);
            break;

        case DataflowStrategy::INPUT_STATIONARY:
            eval.tile_config = optimizer_.optimize_input_stationary(shape.M, shape.N, shape.K);
            break;
    }

    // Generate L2 schedule
    // TODO: Implement strategy-specific schedulers (generate_schedule_ws, generate_schedule_is)
    // For now, all strategies use output-stationary scheduler
    auto schedule = scheduler_.generate_schedule(
        shape.M, shape.N, shape.K,
        eval.tile_config,
        L2TileScheduler::ReplacementPolicy::LRU,
        L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY
    );

    // Calculate performance metrics
    // TODO: Strategy-specific energy/latency calculations
    eval.metrics.total_energy = calculate_energy(shape, schedule);
    eval.metrics.total_cycles = calculate_latency(shape, schedule);

    eval.metrics.dram_accesses = schedule.total_bytes_loaded;
    eval.metrics.l2_accesses = schedule.total_bytes_loaded * 2; // Approximate

    eval.metrics.arithmetic_intensity = eval.tile_config.arithmetic_intensity;
    eval.metrics.utilization = calculate_utilization(shape, eval.tile_config);

    eval.metrics.reuse_A = eval.tile_config.reuse_A;
    eval.metrics.reuse_B = eval.tile_config.reuse_B;
    eval.metrics.reuse_C = eval.tile_config.reuse_C;

    // Calculate throughput metrics
    Size total_ops = 2 * shape.M * shape.N * shape.K;  // 2 ops per MAC
    if (eval.metrics.total_cycles > 0) {
        // Assume 1 GHz clock for GFLOP/s calculation
        double clock_ghz = 1.0;
        eval.metrics.throughput_gflops = (total_ops / 1e9) * clock_ghz / (eval.metrics.total_cycles / 1e9);
        eval.metrics.frequency_mhz = (1.0 / eval.metrics.total_cycles) * 1000.0;  // Normalized frequency
    }

    // Calculate total tile size (Ti × Tj × Tk) for Pareto analysis
    eval.metrics.total_tile_size = eval.tile_config.Ti * eval.tile_config.Tj * eval.tile_config.Tk;

    // Calculate ideal
    eval.ideal = calculate_ideal(shape);

    // Calculate slowdowns
    eval.energy_slowdown = eval.metrics.total_energy / eval.ideal.ideal_energy;
    eval.latency_slowdown = (double)eval.metrics.total_cycles / eval.ideal.ideal_cycles;

    return eval;
}

std::vector<ScheduleEvaluation> ScheduleCharacterizer::evaluate_all_strategies(
    const TensorShape& shape)
{
    std::vector<ScheduleEvaluation> results;

    // Evaluate each dataflow strategy
    for (auto strategy : {DataflowStrategy::WEIGHT_STATIONARY,
                         DataflowStrategy::INPUT_STATIONARY,
                         DataflowStrategy::OUTPUT_STATIONARY}) {
        results.push_back(evaluate_schedule(shape, strategy));
    }

    return results;
}

ParetoFrontier ScheduleCharacterizer::characterize_workloads(
    const std::vector<TensorShape>& workloads,
    const std::vector<DataflowStrategy>& strategies)
{
    std::vector<ScheduleEvaluation> all_evaluations;
    all_evaluations.reserve(workloads.size() * strategies.size());

    std::cout << "Characterizing " << workloads.size() << " workloads × "
              << strategies.size() << " strategies = "
              << (workloads.size() * strategies.size()) << " evaluations...\n";

    size_t count = 0;
    for (const auto& shape : workloads) {
        for (const auto& strategy : strategies) {
            all_evaluations.push_back(evaluate_schedule(shape, strategy));

            if (++count % 1000 == 0) {
                std::cout << "  Progress: " << count << " / "
                          << (workloads.size() * strategies.size()) << "\n";
            }
        }
    }

    // Export ALL evaluations first (before Pareto filtering)
    std::cout << "Exporting all " << all_evaluations.size() << " evaluations...\n";
    export_csv(all_evaluations, "all_evaluations.csv");

    std::cout << "Computing Pareto frontier...\n";
    return compute_pareto_frontier(all_evaluations);
}

ParetoFrontier ScheduleCharacterizer::compute_pareto_frontier(
    const std::vector<ScheduleEvaluation>& evaluations)
{
    ParetoFrontier frontier;
    frontier.total_schedules = evaluations.size();

    // Convert to Pareto points (Energy vs Tile Size - THE FUNDAMENTAL TRADEOFF!)
    std::vector<ParetoPoint> points;
    points.reserve(evaluations.size());

    for (size_t i = 0; i < evaluations.size(); ++i) {
        auto eval_ptr = std::make_shared<ScheduleEvaluation>(evaluations[i]);
        points.emplace_back(
            eval_ptr->metrics.total_energy,
            eval_ptr->metrics.total_tile_size,         // Tile size Ti×Tj×Tk (PROPER Pareto metric!)
            eval_ptr->tile_config.l2_footprint,        // L2 footprint (for reference)
            eval_ptr->metrics.throughput_gflops,       // Throughput (for reference)
            eval_ptr->metrics.total_cycles,            // Latency (for reference)
            eval_ptr
        );
    }

    // Find Pareto-optimal points
    for (size_t i = 0; i < points.size(); ++i) {
        bool is_dominated = false;

        for (size_t j = 0; j < points.size(); ++j) {
            if (i != j && dominates(points[j], points[i])) {
                is_dominated = true;
                points[i].schedule->dominated_by_count++;
            }
        }

        if (!is_dominated) {
            points[i].schedule->is_pareto_optimal = true;
            frontier.points.push_back(points[i]);
        }
    }

    // Sort frontier by energy
    std::sort(frontier.points.begin(), frontier.points.end());

    frontier.frontier_size = frontier.points.size();
    frontier.coverage_percentage = 100.0 * frontier.frontier_size / frontier.total_schedules;

    // Store all evaluations for export (user wants to see full design space)
    frontier.all_evaluations = evaluations;

    return frontier;
}

void ScheduleCharacterizer::print_summary(const ParetoFrontier& frontier) const {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                PARETO FRONTIER CHARACTERIZATION                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "Total Schedules Evaluated: " << frontier.total_schedules << "\n";
    std::cout << "Pareto-Optimal Schedules: " << frontier.frontier_size << "\n";
    std::cout << "Coverage: " << std::fixed << std::setprecision(2)
              << frontier.coverage_percentage << "%\n";
    std::cout << "\n";

    if (!frontier.points.empty()) {
        std::cout << "Pareto Frontier Points (sorted by energy):\n";
        std::cout << "  ┌──────────────┬──────────────┬──────────────────┬──────────┬────────────┐\n";
        std::cout << "  │  Energy (pJ) │ Latency (cyc)│      Shape       │ Strategy │  Slowdown  │\n";
        std::cout << "  ├──────────────┼──────────────┼──────────────────┼──────────┼────────────┤\n";

        for (size_t i = 0; i < std::min(frontier.points.size(), size_t(20)); ++i) {
            const auto& pt = frontier.points[i];
            std::cout << "  │ " << std::setw(12) << std::fixed << std::setprecision(1)
                      << pt.energy << " │ "
                      << std::setw(12) << pt.latency << " │ "
                      << std::setw(16) << pt.schedule->shape.to_string() << " │ "
                      << std::setw(8) << ScheduleEvaluation::dataflow_to_string(pt.schedule->strategy) << " │ "
                      << std::setw(4) << std::fixed << std::setprecision(2)
                      << pt.schedule->energy_slowdown << "×/"
                      << std::setw(4) << pt.schedule->latency_slowdown << "× │\n";
        }

        if (frontier.points.size() > 20) {
            std::cout << "  │     ...      │     ...      │      ...         │   ...    │    ...     │\n";
            std::cout << "  │              │              │  (+" << (frontier.points.size() - 20)
                      << " more)      │          │            │\n";
        }

        std::cout << "  └──────────────┴──────────────┴──────────────────┴──────────┴────────────┘\n";
    }
}

void ScheduleCharacterizer::export_csv(
    const std::vector<ScheduleEvaluation>& evaluations,
    const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing\n";
        return;
    }

    // CSV header
    file << "M,N,K,Strategy,Energy_pJ,Tile_Size,L2_Footprint_bytes,Latency_cycles,Throughput_GFLOPS,"
         << "Energy_Slowdown,Latency_Slowdown,"
         << "DRAM_bytes,AI,Utilization,"
         << "Reuse_A,Reuse_B,Reuse_C,Is_Pareto\n";

    for (const auto& eval : evaluations) {
        file << eval.shape.M << ","
             << eval.shape.N << ","
             << eval.shape.K << ","
             << ScheduleEvaluation::dataflow_to_string(eval.strategy) << ","
             << eval.metrics.total_energy << ","
             << eval.metrics.total_tile_size << ","
             << eval.tile_config.l2_footprint << ","
             << eval.metrics.total_cycles << ","
             << eval.metrics.throughput_gflops << ","
             << eval.energy_slowdown << ","
             << eval.latency_slowdown << ","
             << eval.metrics.dram_accesses << ","
             << eval.metrics.arithmetic_intensity << ","
             << eval.metrics.utilization << ","
             << eval.metrics.reuse_A << ","
             << eval.metrics.reuse_B << ","
             << eval.metrics.reuse_C << ","
             << (eval.is_pareto_optimal ? 1 : 0) << "\n";
    }

    file.close();
    std::cout << "Exported " << evaluations.size() << " evaluations to " << filename << "\n";
}

void ScheduleCharacterizer::export_pareto_csv(
    const ParetoFrontier& frontier,
    const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing\n";
        return;
    }

    // CSV header
    file << "Energy_pJ,Latency_cycles,M,N,K,Strategy\n";

    for (const auto& pt : frontier.points) {
        file << pt.energy << ","
             << pt.latency << ","
             << pt.schedule->shape.M << ","
             << pt.schedule->shape.N << ","
             << pt.schedule->shape.K << ","
             << ScheduleEvaluation::dataflow_to_string(pt.schedule->strategy) << "\n";
    }

    file.close();
    std::cout << "Exported Pareto frontier (" << frontier.points.size()
              << " points) to " << filename << "\n";
}

} // namespace sw::kpu::compiler
