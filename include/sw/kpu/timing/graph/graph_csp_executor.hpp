// ============================================================================
// include/sw/kpu/timing/graph/graph_csp_executor.hpp
// Graph->CSP bridge: execute a KernelGraph on the credit-based value path
// (M2-T2, #201). See docs/plans/m2_resnet_dfg.md.
//
// Walks a KernelGraph in topological order and runs each operator node on the
// schedule-generator value path (csp_op_runners.hpp), threading activations
// between nodes and applying the operator fusions as it lowers:
//   * conv+BN fold  - a CONV2D followed (as sole consumer) by a BATCHNORM folds
//     BN's scale/shift into the conv's weights+bias; the BN node is not executed.
//   * conv+ReLU fused epilogue - a CONV2D/folded-conv followed (as sole consumer)
//     by an ELEMENTWISE RELU applies ReLU in-compute; the ReLU node is not run.
// A standalone ELEMENTWISE (the residual ADD, and the block's final ReLU whose
// producer is the ADD) runs as its own CSP op.
//
// Nodes execute SEQUENTIALLY in topological order (the design's honest scope):
// the measured concurrency is the tile-level pipeline overlap within each op,
// not operator-branch overlap. get_execution_levels / find_fusible_pairs remain
// available for analysis/viz.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/csp_op_runners.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sw::kpu::timing::graph {

/// Per-node data the bridge needs beyond the KernelGraph Kernel config (the
/// graph carries shapes, not weights). Keyed by node id.
struct NodeData {
    std::vector<float> filter;   ///< CONV2D weights [Cout,Cin,Kh,Kw]
    std::vector<float> bias;     ///< optional CONV2D bias [Cout]
    // BATCHNORM raw inference params [C]:
    std::vector<float> gamma, beta, mean, var;
    float eps = 1e-5f;
    // MATMUL (FC) params: weight [fc_K, fc_N] row-major, optional bias [fc_N].
    // Dims are carried here because create_matmul exposes no config accessor.
    std::vector<float> fc_weight, fc_bias;
    Size fc_M = 0, fc_K = 0, fc_N = 0;
    bool fc_relu = false;
};

/**
 * @brief Execute a KernelGraph (a ResNet block / network) on the CSP value path.
 *
 * The graph is a single-source, single-sink operator DAG. `input` is the source
 * tensor (NCHW); `node_data` supplies per-node weights/params. Returns the sink
 * node's output tensor plus accumulated CSP stats.
 *
 * @param T tile size (must divide the GEMM M/N/K of every conv node)
 */
class GraphCspExecutor {
public:
    struct Result {
        std::vector<float> output;
        RunStats stats;
    };

    Result run(const KernelGraph& g, const std::vector<float>& input,
               const std::unordered_map<std::size_t, NodeData>& node_data, Size T) {
        using sw::kpu::KernelOpType;
        using sw::kpu::ElementwiseOp;

        Result result;
        std::unordered_map<std::size_t, std::vector<float>> out;  // node id -> tensor

        // --- fusion pre-pass ---------------------------------------------------
        // fold_bn[conv] = bn node folded into conv; consumed[node] = not run.
        std::unordered_map<std::size_t, std::size_t> fold_bn;
        std::unordered_map<std::size_t, std::size_t> bn_to_conv;  // bn -> conv it folds into
        std::unordered_map<std::size_t, std::size_t> fuse_relu;   // conv -> relu
        std::unordered_map<std::size_t, bool> consumed;

        // Pass 1: fold each BATCHNORM whose sole producer is a CONV2D (and which
        // is that conv's sole consumer) into the conv.
        for (auto id : g.get_execution_order()) {
            if (g.get_kernel(id).op_type() != KernelOpType::BATCHNORM) continue;
            std::optional<std::size_t> prod = sole_producer(g, id);
            if (prod && g.get_kernel(*prod).op_type() == KernelOpType::CONV2D &&
                sole_consumer(g, *prod) == id) {
                fold_bn[*prod] = id;
                bn_to_conv[id] = *prod;
                consumed[id] = true;
            }
        }
        // Pass 2: fuse a ReLU epilogue onto the conv it follows - directly
        // (conv->relu) or through a folded BN (conv->bn->relu). Never onto the ADD.
        for (auto id : g.get_execution_order()) {
            const auto& k = g.get_kernel(id);
            if (k.op_type() != KernelOpType::ELEMENTWISE ||
                k.elementwise_config().op != ElementwiseOp::RELU) continue;
            std::optional<std::size_t> prod = sole_producer(g, id);
            if (!prod || sole_consumer(g, *prod) != id) continue;
            std::size_t conv = static_cast<std::size_t>(-1);
            if (g.get_kernel(*prod).op_type() == KernelOpType::CONV2D) conv = *prod;
            else if (auto it = bn_to_conv.find(*prod); it != bn_to_conv.end()) conv = it->second;
            if (conv != static_cast<std::size_t>(-1)) {
                // Never fuse onto a depthwise conv: run_depthwise_conv applies
                // only ReLU6, so a fused plain-RELU could not be honored and
                // would be silently dropped - keep it as a standalone node.
                const auto& conv_cc = g.get_kernel(conv).conv2d_config();
                if (conv_cc.groups > 1 && conv_cc.groups == conv_cc.in_channels) continue;
                fuse_relu[conv] = id;
                consumed[id] = true;
            }
        }

        // --- topological execution --------------------------------------------
        std::size_t last = 0;
        for (auto id : g.get_execution_order()) {
            if (consumed.count(id)) continue;
            const auto& k = g.get_kernel(id);
            const NodeData* nd = data_or_null(node_data, id);

            switch (k.op_type()) {
                case KernelOpType::CONV2D: {
                    const auto& cc = k.conv2d_config();
                    if (!nd) throw std::runtime_error("GraphCspExecutor: missing NodeData for conv node");
                    Conv2DGeometry geom = conv_geom(cc);
                    // input: this conv's graph predecessor output, else external input.
                    const std::vector<float>& x = input_of(g, id, out, input);

                    // Optional folded BN.
                    schedule::BatchNormAffine affine;
                    const schedule::BatchNormAffine* bn = nullptr;
                    if (auto it = fold_bn.find(id); it != fold_bn.end()) {
                        const auto* bd = data_or_null(node_data, it->second);
                        if (!bd) throw std::runtime_error("GraphCspExecutor: missing NodeData for folded BN");
                        affine = schedule::bn_fold(bd->gamma, bd->beta, bd->mean, bd->var, bd->eps);
                        bn = &affine;
                    }
                    const bool relu = fuse_relu.count(id) > 0;
                    std::vector<float> y;
                    // Depthwise conv (groups == Cin, MobileNet): each output
                    // channel is the 2D conv of a single input channel with its
                    // own filter - delivered via the pooling-window unfold +
                    // per-channel filter reduce (run_depthwise_conv), not im2col
                    // GEMM. BN folds identically. Activations stay standalone
                    // nodes (never fused onto depthwise - see fusion Pass 2),
                    // so relu6 is always false here; run_depthwise_conv applies
                    // only ReLU6, so a fused plain-RELU must not reach it.
                    if (cc.groups > 1 && cc.groups == cc.in_channels) {
                        schedule::Pool2DGeometry pg;
                        pg.N = static_cast<Size>(cc.batch_size);
                        pg.C = static_cast<Size>(cc.in_channels);
                        pg.H = static_cast<Size>(cc.input_height);
                        pg.W = static_cast<Size>(cc.input_width);
                        pg.Kh = static_cast<Size>(cc.kernel_height);
                        pg.Kw = static_cast<Size>(cc.kernel_width);
                        pg.stride_h = static_cast<Size>(cc.stride_h);
                        pg.stride_w = static_cast<Size>(cc.stride_w);
                        pg.pad_h = static_cast<Size>(cc.padding_h);
                        pg.pad_w = static_cast<Size>(cc.padding_w);
                        y = run_depthwise_conv(x, pg, nd->filter, bn, nd->bias,
                                               /*relu6*/ false, T, result.stats);
                    } else {
                        y = run_conv2d_fused(x, nd->filter, geom, bn, nd->bias, relu, T, result.stats);
                    }
                    out[id] = y;
                    // Fused BN / ReLU nodes alias this output for their consumers.
                    if (auto it = fold_bn.find(id); it != fold_bn.end()) out[it->second] = y;
                    if (auto it = fuse_relu.find(id); it != fuse_relu.end()) out[it->second] = y;
                    last = alias_last(id, fold_bn, fuse_relu);
                    break;
                }
                case KernelOpType::ELEMENTWISE: {
                    const auto op = k.elementwise_config().op;
                    if (op == ElementwiseOp::ADD) {
                        // Residual join: gather the ADD's input-edge producers.
                        // Two edges = both branches (projection skip); one edge =
                        // main branch + the external input (identity skip).
                        auto ins = edge_inputs(g, id, out);
                        const std::vector<float>& a = ins.empty() ? input : *ins[0];
                        const std::vector<float>& b = ins.size() > 1 ? *ins[1] : input;
                        out[id] = run_elementwise(sw::kpu::isa::VEOp::ADD, a, b, result.stats);
                    } else if (op == ElementwiseOp::RELU) {
                        out[id] = run_relu(input_of(g, id, out, input), result.stats);
                    } else if (op == ElementwiseOp::RELU6) {
                        // MobileNetV2 activation clamp min(max(x,0),6), run as a
                        // standalone VE op (the design's permitted clamp form).
                        out[id] = run_relu6(input_of(g, id, out, input), result.stats);
                    } else {
                        throw std::runtime_error("GraphCspExecutor: unsupported elementwise op");
                    }
                    last = id;
                    break;
                }
                case KernelOpType::POOL2D: {
                    const auto& pc = k.pool2d_config();
                    if (pc.pool_type != sw::kpu::PoolType::GLOBAL_AVG)
                        throw std::runtime_error("GraphCspExecutor: only GLOBAL_AVG pooling supported");
                    schedule::Pool2DGeometry pg;
                    pg.N = static_cast<Size>(pc.batch_size);
                    pg.C = static_cast<Size>(pc.channels);
                    pg.H = static_cast<Size>(pc.input_height);
                    pg.W = static_cast<Size>(pc.input_width);
                    out[id] = run_global_avg_pool(input_of(g, id, out, input), pg, result.stats);
                    last = id;
                    break;
                }
                case KernelOpType::MATMUL: {
                    if (!nd) throw std::runtime_error("GraphCspExecutor: missing NodeData for matmul node");
                    out[id] = run_matmul(input_of(g, id, out, input), nd->fc_weight,
                                         nd->fc_M, nd->fc_N, nd->fc_K, nd->fc_bias,
                                         nd->fc_relu, T, result.stats);
                    last = id;
                    break;
                }
                default:
                    throw std::runtime_error("GraphCspExecutor: unsupported kernel op type");
            }
        }
        result.output = out.at(last);
        return result;
    }

private:
    static const NodeData* data_or_null(
        const std::unordered_map<std::size_t, NodeData>& m, std::size_t id) {
        auto it = m.find(id);
        return it == m.end() ? nullptr : &it->second;
    }

    // The output of this node's single internal producer, or the external input
    // if it has no incoming edge (a graph source).
    static const std::vector<float>& input_of(
        const KernelGraph& g, std::size_t id,
        const std::unordered_map<std::size_t, std::vector<float>>& out,
        const std::vector<float>& external) {
        const auto& node = g.get_node(id);
        for (std::size_t e : node.input_edges) {
            const auto& edge = g.get_edge(e);
            auto it = out.find(edge.from_node);
            if (it != out.end()) return it->second;
        }
        return external;
    }

    // All input-edge producer outputs, in edge order (only those already run).
    static std::vector<const std::vector<float>*> edge_inputs(
        const KernelGraph& g, std::size_t id,
        const std::unordered_map<std::size_t, std::vector<float>>& out) {
        std::vector<const std::vector<float>*> res;
        for (std::size_t e : g.get_node(id).input_edges) {
            auto it = out.find(g.get_edge(e).from_node);
            if (it != out.end()) res.push_back(&it->second);
        }
        return res;
    }

    static std::optional<std::size_t> sole_producer(const KernelGraph& g, std::size_t id) {
        const auto& node = g.get_node(id);
        if (node.input_edges.size() != 1) return std::nullopt;
        return g.get_edge(node.input_edges[0]).from_node;
    }
    static std::size_t sole_consumer(const KernelGraph& g, std::size_t id) {
        const auto& node = g.get_node(id);
        if (node.output_edges.size() != 1) return static_cast<std::size_t>(-1);
        return g.get_edge(node.output_edges[0]).to_node;
    }
    static std::size_t alias_last(std::size_t conv,
        const std::unordered_map<std::size_t, std::size_t>& fold_bn,
        const std::unordered_map<std::size_t, std::size_t>& fuse_relu) {
        std::size_t l = conv;
        if (auto it = fold_bn.find(conv); it != fold_bn.end()) l = it->second;
        if (auto it = fuse_relu.find(conv); it != fuse_relu.end()) l = it->second;
        return l;
    }

    static Conv2DGeometry conv_geom(const sw::kpu::Conv2DConfig& c) {
        // Conv2DConfig uses sw::kpu::Size (64-bit); Conv2DGeometry uses the
        // timing Size (32-bit) - cast to avoid MSVC C4267.
        Conv2DGeometry g;
        g.N = static_cast<Size>(c.batch_size);
        g.C_in = static_cast<Size>(c.in_channels);
        g.H_in = static_cast<Size>(c.input_height);
        g.W_in = static_cast<Size>(c.input_width);
        g.C_out = static_cast<Size>(c.out_channels);
        g.Kh = static_cast<Size>(c.kernel_height);
        g.Kw = static_cast<Size>(c.kernel_width);
        g.stride_h = static_cast<Size>(c.stride_h);
        g.stride_w = static_cast<Size>(c.stride_w);
        g.pad_h = static_cast<Size>(c.padding_h);
        g.pad_w = static_cast<Size>(c.padding_w);
        return g;
    }
};

} // namespace sw::kpu::timing::graph
