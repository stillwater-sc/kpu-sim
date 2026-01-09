/**
 * @file dfx_generator.cpp
 * @brief Implementation of DFX generator
 */

#include "dfx_generator.hpp"
#include <iostream>
#include <algorithm>

namespace sw::kpu::compiler {

DFXGenerator::DFXGenerator(const DFXGeneratorOptions& options)
    : options_(options),
      tile_optimizer_(options.memory_hierarchy),
      stats_{}
{
}

dfx::Program DFXGenerator::generate_matmul(const MatrixOpInfo& op_info,
                                            const std::string& graph_name) {
    // Reset state
    next_op_id_ = 1;
    stats_ = GenerationStats{};

    dfx::Program program;
    program.name = graph_name;
    program.source_graph = graph_name;
    program.dataflow = options_.dataflow;

    // Add tensor descriptors
    add_tensor(program, op_info.tensor_a, {op_info.M, op_info.K},
               op_info.dtype, true, false);   // A is input (constant)
    add_tensor(program, op_info.tensor_b, {op_info.K, op_info.N},
               op_info.dtype, true, false);   // B is input (constant)
    add_tensor(program, op_info.tensor_c, {op_info.M, op_info.N},
               op_info.dtype, false, true);   // C is output

    // Optimize tile sizes
    TileOptimizer::TileConfig tile_config;

    switch (options_.dataflow) {
        case dfx::DataflowStrategy::OUTPUT_STATIONARY:
            tile_config = tile_optimizer_.optimize(
                op_info.M, op_info.N, op_info.K, options_.tile_strategy);
            break;
        case dfx::DataflowStrategy::WEIGHT_STATIONARY:
            tile_config = tile_optimizer_.optimize_weight_stationary(
                op_info.M, op_info.N, op_info.K);
            break;
        case dfx::DataflowStrategy::INPUT_STATIONARY:
            tile_config = tile_optimizer_.optimize_input_stationary(
                op_info.M, op_info.N, op_info.K);
            break;
    }

    if (!tile_config.valid) {
        // Fall back to simple tiling
        tile_config.Ti = std::min(op_info.M, size_t(16));
        tile_config.Tj = std::min(op_info.N, size_t(16));
        tile_config.Tk = std::min(op_info.K, size_t(16));
        tile_config.valid = true;
    }

    // Store tiling configuration
    program.tiling.Ti = tile_config.Ti;
    program.tiling.Tj = tile_config.Tj;
    program.tiling.Tk = tile_config.Tk;
    program.tiling.L1_Ki = tile_config.L1_Ki;
    program.tiling.num_tiles_m = ceil_div(op_info.M, tile_config.Ti);
    program.tiling.num_tiles_n = ceil_div(op_info.N, tile_config.Tj);
    program.tiling.num_tiles_k = ceil_div(op_info.K, tile_config.Tk);

    // Generate tile iteration loops
    generate_tile_loops(program, tile_config, op_info.M, op_info.N, op_info.K);

    // Generate schedule based on dataflow strategy
    switch (options_.dataflow) {
        case dfx::DataflowStrategy::OUTPUT_STATIONARY:
            generate_output_stationary_schedule(program, op_info, tile_config);
            break;
        case dfx::DataflowStrategy::WEIGHT_STATIONARY:
            generate_weight_stationary_schedule(program, op_info, tile_config);
            break;
        case dfx::DataflowStrategy::INPUT_STATIONARY:
            // TODO: Implement input stationary schedule
            generate_output_stationary_schedule(program, op_info, tile_config);
            break;
    }

    // Store performance hints
    program.hints.estimated_dram_bytes = tile_config.dram_accesses;
    program.hints.arithmetic_intensity = tile_config.arithmetic_intensity;
    program.hints.reuse_factor_a = tile_config.reuse_A;
    program.hints.reuse_factor_b = tile_config.reuse_B;

    // Calculate estimated compute cycles
    // 2 * M * N * K FLOPs, systolic array does 16*16*2 = 512 FLOPs/cycle
    size_t total_flops = 2 * op_info.M * op_info.N * op_info.K;
    size_t flops_per_cycle = options_.memory_hierarchy.systolic_rows *
                             options_.memory_hierarchy.systolic_cols * 2;
    program.hints.estimated_compute_cycles = total_flops / flops_per_cycle;

    // Update stats
    stats_.estimated_dram_bytes = tile_config.dram_accesses;
    stats_.estimated_compute_cycles = static_cast<double>(program.hints.estimated_compute_cycles);

    if (options_.verbose) {
        std::cout << "DFX Generation Summary:\n";
        std::cout << "  Matrix: " << op_info.M << "x" << op_info.N << "x" << op_info.K << "\n";
        std::cout << "  Tiles: " << tile_config.Ti << "x" << tile_config.Tj << "x" << tile_config.Tk << "\n";
        std::cout << "  Tile grid: " << program.tiling.num_tiles_m << "x"
                  << program.tiling.num_tiles_n << "x" << program.tiling.num_tiles_k << "\n";
        std::cout << "  Operations: " << program.operations.size() << "\n";
        std::cout << "  Data moves: " << stats_.num_data_moves << "\n";
        std::cout << "  Computes: " << stats_.num_computes << "\n";
    }

    return program;
}

dfx::Program DFXGenerator::generate_program(const ComputationalGraph& graph,
                                             const std::vector<MatrixOpInfo>& ops) {
    if (ops.empty()) {
        throw std::runtime_error("No operations to compile");
    }

    // For now, handle single-operation graphs
    // TODO: Handle multi-operator graphs with proper scheduling
    return generate_matmul(ops[0], graph.name);
}

void DFXGenerator::generate_tile_loops(dfx::Program& program,
                                        const TileOptimizer::TileConfig& tile_config,
                                        size_t M, size_t N, size_t K) {
    size_t num_m = ceil_div(M, tile_config.Ti);
    size_t num_n = ceil_div(N, tile_config.Tj);
    size_t num_k = ceil_div(K, tile_config.Tk);

    // Create tile loops based on dataflow strategy
    // Output stationary: ti -> tj -> tk (C tiles outer)
    // Weight stationary: tk -> tj -> ti (B tiles outer)

    dfx::TileLoop ti_loop;
    ti_loop.induction_var = "ti";
    ti_loop.start = 0;
    ti_loop.end = num_m;
    ti_loop.step = 1;

    dfx::TileLoop tj_loop;
    tj_loop.induction_var = "tj";
    tj_loop.start = 0;
    tj_loop.end = num_n;
    tj_loop.step = 1;

    dfx::TileLoop tk_loop;
    tk_loop.induction_var = "tk";
    tk_loop.start = 0;
    tk_loop.end = num_k;
    tk_loop.step = 1;

    switch (options_.dataflow) {
        case dfx::DataflowStrategy::OUTPUT_STATIONARY:
            program.tile_loops = {ti_loop, tj_loop, tk_loop};
            break;
        case dfx::DataflowStrategy::WEIGHT_STATIONARY:
            program.tile_loops = {tk_loop, tj_loop, ti_loop};
            break;
        case dfx::DataflowStrategy::INPUT_STATIONARY:
            program.tile_loops = {ti_loop, tk_loop, tj_loop};
            break;
    }
}

void DFXGenerator::generate_output_stationary_schedule(
    dfx::Program& program,
    const MatrixOpInfo& op_info,
    const TileOptimizer::TileConfig& config) {

    size_t num_tiles_m = program.tiling.num_tiles_m;
    size_t num_tiles_n = program.tiling.num_tiles_n;
    size_t num_tiles_k = program.tiling.num_tiles_k;

    // Output stationary loop order: for each C tile, iterate through K
    // for ti in 0..num_tiles_m:
    //   for tj in 0..num_tiles_n:
    //     for tk in 0..num_tiles_k:
    //       load A[ti, tk], B[tk, tj]
    //       compute C[ti, tj] += A[ti, tk] * B[tk, tj]
    //     store C[ti, tj]

    for (size_t ti = 0; ti < num_tiles_m; ++ti) {
        for (size_t tj = 0; tj < num_tiles_n; ++tj) {
            // Calculate actual tile shapes for this position
            size_t tile_m = std::min(config.Ti, op_info.M - ti * config.Ti);
            size_t tile_n = std::min(config.Tj, op_info.N - tj * config.Tj);

            std::vector<uint64_t> c_compute_deps;

            for (size_t tk = 0; tk < num_tiles_k; ++tk) {
                size_t tile_k = std::min(config.Tk, op_info.K - tk * config.Tk);

                // Load A tile
                uint64_t a_load = generate_tile_load(
                    program, op_info.tensor_a,
                    {ti, tk}, {tile_m, tile_k},
                    op_info.dtype, {});

                // Load B tile (can happen in parallel with A)
                uint64_t b_load = generate_tile_load(
                    program, op_info.tensor_b,
                    {tk, tj}, {tile_k, tile_n},
                    op_info.dtype, {});

                // Compute: C += A * B
                dfx::TileSpec a_tile, b_tile, c_tile;
                a_tile.tensor_name = op_info.tensor_a;
                a_tile.level = dfx::MemoryLevel::L1;
                a_tile.tile_indices = {ti, tk};
                a_tile.tile_shape = {tile_m, tile_k};

                b_tile.tensor_name = op_info.tensor_b;
                b_tile.level = dfx::MemoryLevel::L1;
                b_tile.tile_indices = {tk, tj};
                b_tile.tile_shape = {tile_k, tile_n};

                c_tile.tensor_name = op_info.tensor_c;
                c_tile.level = dfx::MemoryLevel::REGISTER;
                c_tile.tile_indices = {ti, tj};
                c_tile.tile_shape = {tile_m, tile_n};

                bool accumulate = (tk > 0);  // Accumulate for all but first K tile
                std::vector<uint64_t> compute_deps = {a_load, b_load};
                if (!c_compute_deps.empty()) {
                    compute_deps.push_back(c_compute_deps.back());
                }

                uint64_t compute = generate_matmul_compute(
                    program, a_tile, b_tile, c_tile, accumulate, compute_deps);

                c_compute_deps.push_back(compute);
            }

            // Store C tile after all K tiles processed
            generate_tile_store(
                program, op_info.tensor_c,
                {ti, tj}, {tile_m, tile_n},
                op_info.dtype, c_compute_deps);
        }
    }
}

void DFXGenerator::generate_weight_stationary_schedule(
    dfx::Program& program,
    const MatrixOpInfo& op_info,
    const TileOptimizer::TileConfig& config) {

    size_t num_tiles_m = program.tiling.num_tiles_m;
    size_t num_tiles_n = program.tiling.num_tiles_n;
    size_t num_tiles_k = program.tiling.num_tiles_k;

    // Weight stationary: B tiles stay resident longer
    // for tk in 0..num_tiles_k:
    //   for tj in 0..num_tiles_n:
    //     load B[tk, tj] (stays resident)
    //     for ti in 0..num_tiles_m:
    //       load A[ti, tk]
    //       compute C[ti, tj] += A[ti, tk] * B[tk, tj]
    //       store/accumulate C[ti, tj]

    // Track C tile accumulators
    std::vector<std::vector<uint64_t>> c_last_compute(num_tiles_m,
        std::vector<uint64_t>(num_tiles_n, 0));

    for (size_t tk = 0; tk < num_tiles_k; ++tk) {
        size_t tile_k = std::min(config.Tk, op_info.K - tk * config.Tk);

        for (size_t tj = 0; tj < num_tiles_n; ++tj) {
            size_t tile_n = std::min(config.Tj, op_info.N - tj * config.Tj);

            // Load B tile (will be reused across all M tiles)
            uint64_t b_load = generate_tile_load(
                program, op_info.tensor_b,
                {tk, tj}, {tile_k, tile_n},
                op_info.dtype, {});

            for (size_t ti = 0; ti < num_tiles_m; ++ti) {
                size_t tile_m = std::min(config.Ti, op_info.M - ti * config.Ti);

                // Load A tile
                uint64_t a_load = generate_tile_load(
                    program, op_info.tensor_a,
                    {ti, tk}, {tile_m, tile_k},
                    op_info.dtype, {});

                // Compute
                dfx::TileSpec a_tile, b_tile, c_tile;
                a_tile.tensor_name = op_info.tensor_a;
                a_tile.level = dfx::MemoryLevel::L1;
                a_tile.tile_indices = {ti, tk};
                a_tile.tile_shape = {tile_m, tile_k};

                b_tile.tensor_name = op_info.tensor_b;
                b_tile.level = dfx::MemoryLevel::L1;
                b_tile.tile_indices = {tk, tj};
                b_tile.tile_shape = {tile_k, tile_n};

                c_tile.tensor_name = op_info.tensor_c;
                c_tile.level = dfx::MemoryLevel::REGISTER;
                c_tile.tile_indices = {ti, tj};
                c_tile.tile_shape = {tile_m, tile_n};

                bool accumulate = (tk > 0);
                std::vector<uint64_t> compute_deps = {a_load, b_load};
                if (c_last_compute[ti][tj] != 0) {
                    compute_deps.push_back(c_last_compute[ti][tj]);
                }

                uint64_t compute = generate_matmul_compute(
                    program, a_tile, b_tile, c_tile, accumulate, compute_deps);

                c_last_compute[ti][tj] = compute;
            }
        }
    }

    // Store all C tiles
    for (size_t ti = 0; ti < num_tiles_m; ++ti) {
        for (size_t tj = 0; tj < num_tiles_n; ++tj) {
            size_t tile_m = std::min(config.Ti, op_info.M - ti * config.Ti);
            size_t tile_n = std::min(config.Tj, op_info.N - tj * config.Tj);

            if (c_last_compute[ti][tj] != 0) {
                generate_tile_store(
                    program, op_info.tensor_c,
                    {ti, tj}, {tile_m, tile_n},
                    op_info.dtype, {c_last_compute[ti][tj]});
            }
        }
    }
}

uint64_t DFXGenerator::generate_tile_load(dfx::Program& program,
                                           const std::string& tensor_name,
                                           const std::vector<size_t>& tile_idx,
                                           const std::vector<size_t>& tile_shape,
                                           dfx::DataType /*dtype*/,
                                           const std::vector<uint64_t>& depends_on) {
    // Generate load chain: EXTERNAL → L3 → L2 → L1

    // EXTERNAL → L3
    auto& ext_to_l3 = program.add_operation<dfx::DataMoveOp>();
    ext_to_l3.op_id = get_next_op_id();
    ext_to_l3.move_type = dfx::DataMoveType::LOAD;
    ext_to_l3.source.tensor_name = tensor_name;
    ext_to_l3.source.level = dfx::MemoryLevel::EXTERNAL;
    ext_to_l3.source.tile_indices = tile_idx;
    ext_to_l3.source.tile_shape = tile_shape;
    ext_to_l3.destination.tensor_name = tensor_name;
    ext_to_l3.destination.level = dfx::MemoryLevel::L3;
    ext_to_l3.destination.tile_indices = tile_idx;
    ext_to_l3.destination.tile_shape = tile_shape;
    ext_to_l3.depends_on = depends_on;
    ext_to_l3.label = tensor_name + " DRAM→L3 [" +
                      std::to_string(tile_idx[0]) + "," + std::to_string(tile_idx[1]) + "]";
    stats_.num_data_moves++;

    // L3 → L2
    auto& l3_to_l2 = program.add_operation<dfx::DataMoveOp>();
    l3_to_l2.op_id = get_next_op_id();
    l3_to_l2.move_type = dfx::DataMoveType::LOAD;
    l3_to_l2.source.tensor_name = tensor_name;
    l3_to_l2.source.level = dfx::MemoryLevel::L3;
    l3_to_l2.source.tile_indices = tile_idx;
    l3_to_l2.source.tile_shape = tile_shape;
    l3_to_l2.destination.tensor_name = tensor_name;
    l3_to_l2.destination.level = dfx::MemoryLevel::L2;
    l3_to_l2.destination.tile_indices = tile_idx;
    l3_to_l2.destination.tile_shape = tile_shape;
    l3_to_l2.depends_on = {ext_to_l3.op_id};
    l3_to_l2.label = tensor_name + " L3→L2";
    stats_.num_data_moves++;

    // L2 → L1
    auto& l2_to_l1 = program.add_operation<dfx::DataMoveOp>();
    l2_to_l1.op_id = get_next_op_id();
    l2_to_l1.move_type = dfx::DataMoveType::LOAD;
    l2_to_l1.source.tensor_name = tensor_name;
    l2_to_l1.source.level = dfx::MemoryLevel::L2;
    l2_to_l1.source.tile_indices = tile_idx;
    l2_to_l1.source.tile_shape = tile_shape;
    l2_to_l1.destination.tensor_name = tensor_name;
    l2_to_l1.destination.level = dfx::MemoryLevel::L1;
    l2_to_l1.destination.tile_indices = tile_idx;
    l2_to_l1.destination.tile_shape = tile_shape;
    l2_to_l1.depends_on = {l3_to_l2.op_id};
    l2_to_l1.label = tensor_name + " L2→L1";
    stats_.num_data_moves++;

    return l2_to_l1.op_id;
}

uint64_t DFXGenerator::generate_tile_store(dfx::Program& program,
                                            const std::string& tensor_name,
                                            const std::vector<size_t>& tile_idx,
                                            const std::vector<size_t>& tile_shape,
                                            dfx::DataType /*dtype*/,
                                            const std::vector<uint64_t>& depends_on) {
    // Generate store chain: REGISTER → L1 → L2 → L3 → EXTERNAL

    // REGISTER → L1
    auto& reg_to_l1 = program.add_operation<dfx::DataMoveOp>();
    reg_to_l1.op_id = get_next_op_id();
    reg_to_l1.move_type = dfx::DataMoveType::STORE;
    reg_to_l1.source.tensor_name = tensor_name;
    reg_to_l1.source.level = dfx::MemoryLevel::REGISTER;
    reg_to_l1.source.tile_indices = tile_idx;
    reg_to_l1.source.tile_shape = tile_shape;
    reg_to_l1.destination.tensor_name = tensor_name;
    reg_to_l1.destination.level = dfx::MemoryLevel::L1;
    reg_to_l1.destination.tile_indices = tile_idx;
    reg_to_l1.destination.tile_shape = tile_shape;
    reg_to_l1.depends_on = depends_on;
    reg_to_l1.label = tensor_name + " REG→L1";
    stats_.num_data_moves++;

    // L1 → L2
    auto& l1_to_l2 = program.add_operation<dfx::DataMoveOp>();
    l1_to_l2.op_id = get_next_op_id();
    l1_to_l2.move_type = dfx::DataMoveType::STORE;
    l1_to_l2.source.tensor_name = tensor_name;
    l1_to_l2.source.level = dfx::MemoryLevel::L1;
    l1_to_l2.source.tile_indices = tile_idx;
    l1_to_l2.source.tile_shape = tile_shape;
    l1_to_l2.destination.tensor_name = tensor_name;
    l1_to_l2.destination.level = dfx::MemoryLevel::L2;
    l1_to_l2.destination.tile_indices = tile_idx;
    l1_to_l2.destination.tile_shape = tile_shape;
    l1_to_l2.depends_on = {reg_to_l1.op_id};
    l1_to_l2.label = tensor_name + " L1→L2";
    stats_.num_data_moves++;

    // L2 → L3
    auto& l2_to_l3 = program.add_operation<dfx::DataMoveOp>();
    l2_to_l3.op_id = get_next_op_id();
    l2_to_l3.move_type = dfx::DataMoveType::STORE;
    l2_to_l3.source.tensor_name = tensor_name;
    l2_to_l3.source.level = dfx::MemoryLevel::L2;
    l2_to_l3.source.tile_indices = tile_idx;
    l2_to_l3.source.tile_shape = tile_shape;
    l2_to_l3.destination.tensor_name = tensor_name;
    l2_to_l3.destination.level = dfx::MemoryLevel::L3;
    l2_to_l3.destination.tile_indices = tile_idx;
    l2_to_l3.destination.tile_shape = tile_shape;
    l2_to_l3.depends_on = {l1_to_l2.op_id};
    l2_to_l3.label = tensor_name + " L2→L3";
    stats_.num_data_moves++;

    // L3 → EXTERNAL
    auto& l3_to_ext = program.add_operation<dfx::DataMoveOp>();
    l3_to_ext.op_id = get_next_op_id();
    l3_to_ext.move_type = dfx::DataMoveType::STORE;
    l3_to_ext.source.tensor_name = tensor_name;
    l3_to_ext.source.level = dfx::MemoryLevel::L3;
    l3_to_ext.source.tile_indices = tile_idx;
    l3_to_ext.source.tile_shape = tile_shape;
    l3_to_ext.destination.tensor_name = tensor_name;
    l3_to_ext.destination.level = dfx::MemoryLevel::EXTERNAL;
    l3_to_ext.destination.tile_indices = tile_idx;
    l3_to_ext.destination.tile_shape = tile_shape;
    l3_to_ext.depends_on = {l2_to_l3.op_id};
    l3_to_ext.label = tensor_name + " L3→DRAM";
    stats_.num_data_moves++;

    return l3_to_ext.op_id;
}

uint64_t DFXGenerator::generate_matmul_compute(dfx::Program& program,
                                                const dfx::TileSpec& a_tile,
                                                const dfx::TileSpec& b_tile,
                                                const dfx::TileSpec& c_tile,
                                                bool accumulate,
                                                const std::vector<uint64_t>& depends_on) {
    auto& compute = program.add_operation<dfx::ComputeOp>();
    compute.op_id = get_next_op_id();
    compute.compute_type = dfx::ComputeType::MATMUL_TILE;
    compute.inputs = {a_tile, b_tile};
    compute.output = c_tile;
    compute.accumulate = accumulate;
    compute.depends_on = depends_on;
    compute.label = "MATMUL C[" + std::to_string(c_tile.tile_indices[0]) + "," +
                    std::to_string(c_tile.tile_indices[1]) + "]" +
                    (accumulate ? " (acc)" : "");

    stats_.num_computes++;
    return compute.op_id;
}

void DFXGenerator::add_tensor(dfx::Program& program,
                               const std::string& name,
                               const std::vector<size_t>& shape,
                               dfx::DataType dtype,
                               bool is_constant,
                               bool is_output) {
    dfx::TensorDescriptor tensor;
    tensor.name = name;
    tensor.shape = shape;
    tensor.dtype = dtype;
    tensor.is_constant = is_constant;
    tensor.is_output = is_output;
    program.tensors.push_back(tensor);
}

} // namespace sw::kpu::compiler
