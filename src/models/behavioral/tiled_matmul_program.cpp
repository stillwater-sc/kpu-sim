// ============================================================================
// src/models/behavioral/tiled_matmul_program.cpp
// Parameterized Tiled Matrix Multiplication - Implementation
// Uses dataflow semantics (credits up, data down)
// See docs/kpu-execution-model.md for the execution model
// ============================================================================

#include <sw/kpu/behavioral/tiled_matmul_program.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace sw::kpu::behavioral {

std::string TiledMatmulProgram::to_json() const {
    std::ostringstream json;

    // Use JSON format compatible with the OFG visualization animation
    json << "{\n";

    // Metadata
    json << "  \"metadata\": {\n";
    json << "    \"generator\": \"kpu-sim TiledMatmulProgram\",\n";
    json << "    \"version\": \"3.0\",\n";
    json << "    \"execution_model\": \"dataflow\",\n";
    json << "    \"problem\": \"D[" << config_.M << "," << config_.N
         << "] = C + A[" << config_.M << "," << config_.K
         << "] * B[" << config_.K << "," << config_.N << "]\",\n";
    json << "    \"loop_order\": \"" << to_string(config_.loop_order) << "\",\n";
    json << "    \"config\": {\n";
    json << "      \"M\": " << config_.M << ",\n";
    json << "      \"N\": " << config_.N << ",\n";
    json << "      \"K\": " << config_.K << ",\n";
    json << "      \"tile_m\": " << config_.tile_m << ",\n";
    json << "      \"tile_n\": " << config_.tile_n << ",\n";
    json << "      \"tile_k\": " << config_.tile_k << ",\n";
    json << "      \"systolic_rows\": " << config_.systolic_rows << ",\n";
    json << "      \"systolic_cols\": " << config_.systolic_cols << ",\n";
    json << "      \"l3_buffers\": " << static_cast<int>(config_.num_l3_tiles) << ",\n";
    json << "      \"l2_banks\": " << static_cast<int>(config_.num_l2_banks) << ",\n";
    json << "      \"m_tiles\": " << config_.m_tiles() << ",\n";
    json << "      \"n_tiles\": " << config_.n_tiles() << ",\n";
    json << "      \"k_tiles\": " << config_.k_tiles() << "\n";
    json << "    }\n";
    json << "  },\n";

    // Tensors
    json << "  \"tensors\": [\n";
    json << "    {\"id\": \"A\", \"shape\": [" << config_.M << ", " << config_.K << "], \"type\": \"input\"},\n";
    json << "    {\"id\": \"B\", \"shape\": [" << config_.K << ", " << config_.N << "], \"type\": \"input\"},\n";
    json << "    {\"id\": \"C\", \"shape\": [" << config_.M << ", " << config_.N << "], \"type\": \"input\"},\n";
    json << "    {\"id\": \"D\", \"shape\": [" << config_.M << ", " << config_.N << "], \"type\": \"output\"}\n";
    json << "  ],\n";

    // Statistics
    json << "  \"stats\": {\n";
    json << "    \"total_cycles\": " << stats_.total_cycles << ",\n";
    json << "    \"compute_cycles\": " << stats_.compute_cycles << ",\n";
    json << "    \"dma_pushes\": " << stats_.dma_loads << ",\n";
    json << "    \"dma_pulls\": " << stats_.dma_stores << ",\n";
    json << "    \"dma_bytes\": " << stats_.dma_bytes << ",\n";
    json << "    \"bm_pushes\": " << stats_.bm_pushes << ",\n";
    json << "    \"bm_pulls\": " << stats_.bm_pulls << ",\n";
    json << "    \"bm_bytes\": " << stats_.bm_bytes << ",\n";
    json << "    \"str_feeds\": " << stats_.str_feeds << ",\n";
    json << "    \"str_drains\": " << stats_.str_drains << ",\n";
    json << "    \"computes\": " << stats_.matmuls << ",\n";
    json << "    \"flops\": " << stats_.flops << ",\n";
    json << "    \"compute_utilization\": " << std::fixed << std::setprecision(4)
         << stats_.compute_utilization() << "\n";
    json << "  },\n";

    // Events
    json << "  \"events\": [\n";

    for (size_t i = 0; i < trace_.size(); ++i) {
        const auto& e = trace_[i];

        json << "    {\n";
        json << "      \"cycle\": " << e.cycle << ",\n";
        json << "      \"level\": \"" << to_string(e.level) << "\",\n";
        json << "      \"type\": \"" << to_string(e.operation) << "\",\n";
        json << "      \"operand\": {\n";
        json << "        \"type\": \"" << to_string(e.operand) << "\",\n";
        json << "        \"coord\": [" << e.tile_i << ", " << e.tile_j << ", " << e.tile_k << "],\n";
        json << "        \"buffer_id\": " << static_cast<int>(e.buffer_id) << ",\n";
        json << "        \"src_buffer\": " << static_cast<int>(e.src_buffer) << ",\n";
        json << "        \"dst_buffer\": " << static_cast<int>(e.dst_buffer) << "\n";
        json << "      }";

        if (e.duration > 0) {
            json << ",\n      \"duration\": " << e.duration;
        }

        if (!e.name.empty()) {
            json << ",\n      \"name\": \"" << e.name << "\"";
        }

        // Buffer state for dataflow events
        if (e.operation == TraceOperation::DMA_PUSH ||
            e.operation == TraceOperation::TILE_READY ||
            e.operation == TraceOperation::BUFFER_AVAILABLE) {
            json << ",\n      \"l3_buffers_occupied\": " << static_cast<int>(e.l3_buffers_occupied);
            json << ",\n      \"l2_buffers_occupied\": " << static_cast<int>(e.l2_buffers_occupied);
        }

        // Loop state for compute and tile events
        if (e.operation == TraceOperation::DMA_PUSH ||
            e.operation == TraceOperation::TILE_READY ||
            e.operation == TraceOperation::BM_PUSH ||
            e.operation == TraceOperation::COMPUTE ||
            e.operation == TraceOperation::TILE_COMPLETE ||
            e.operation == TraceOperation::MATMUL) {
            json << ",\n      \"loop_state\": {"
                 << "\"i\": " << e.loop_state.outer
                 << ", \"j\": " << e.loop_state.middle
                 << ", \"k\": " << e.loop_state.inner << "}";
        }

        json << "\n    }";

        if (i + 1 < trace_.size()) {
            json << ",";
        }
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";

    return json.str();
}

bool TiledMatmulProgram::write_trace_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << to_json();
    return file.good();
}

} // namespace sw::kpu::behavioral
