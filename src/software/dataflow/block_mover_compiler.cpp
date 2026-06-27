// ============================================================================
// src/dataflow/block_mover_compiler.cpp
// BlockMover Compiler Implementation
// ============================================================================

#include "sw/kpu/dataflow/block_mover_compiler.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace sw::kpu::dataflow {

// ============================================================================
// CompiledSchedule Implementation
// ============================================================================

std::string CompiledSchedule::to_string() const {
    std::ostringstream oss;
    oss << "CompiledSchedule {\n";
    oss << "  estimated_cycles: " << estimated_cycles << "\n";
    oss << "  compute_cycles: " << compute_cycles << "\n";
    oss << "  data_movement_cycles: " << data_movement_cycles << "\n";
    oss << "  stats:\n";
    oss << "    total_commands: " << stats.total_commands << "\n";
    oss << "    l3_transfers: " << stats.total_l3_transfers << "\n";
    oss << "    l2_transfers: " << stats.total_l2_transfers << "\n";
    oss << "    dma_ops: " << stats.total_dma_ops << "\n";
    oss << "    compute_ops: " << stats.total_compute_ops << "\n";
    oss << "    triggers: " << stats.total_triggers << "\n";
    oss << "    barriers: " << stats.total_barriers << "\n";

    for (uint8_t i = 0; i < 16; ++i) {
        if (!blockmover_programs[i].empty()) {
            oss << "  L3[" << static_cast<int>(i) << "]: "
                << blockmover_programs[i].size() << " commands\n";
        }
    }
    oss << "}\n";
    return oss.str();
}

void CompiledSchedule::print_summary() const {
    std::cout << "\n=== Compiled Schedule Summary ===\n";
    std::cout << "Estimated cycles: " << estimated_cycles << "\n";
    std::cout << "  Compute: " << compute_cycles << " ("
              << (100.0 * compute_cycles / estimated_cycles) << "%)\n";
    std::cout << "  Data movement: " << data_movement_cycles << " ("
              << (100.0 * data_movement_cycles / estimated_cycles) << "%)\n";
    std::cout << "\nCommands:\n";
    std::cout << "  Total: " << stats.total_commands << "\n";
    std::cout << "  DMA ops: " << stats.total_dma_ops << "\n";
    std::cout << "  L3 transfers: " << stats.total_l3_transfers << "\n";
    std::cout << "  L2 transfers: " << stats.total_l2_transfers << "\n";
    std::cout << "  Compute: " << stats.total_compute_ops << "\n";
    std::cout << "  Triggers: " << stats.total_triggers << "\n";
    std::cout << "  Barriers: " << stats.total_barriers << "\n";

    std::cout << "\nPer-L3 programs:\n";
    for (uint8_t i = 0; i < 16; ++i) {
        if (!blockmover_programs[i].empty()) {
            std::cout << "  L3[" << static_cast<int>(i) << "]: "
                      << blockmover_programs[i].size() << " commands\n";
        }
    }
    std::cout << "=================================\n";
}

// ============================================================================
// BlockMoverCompiler Implementation
// ============================================================================

BlockMoverCompiler::BlockMoverCompiler()
    : BlockMoverCompiler(Config{})
{
}

BlockMoverCompiler::BlockMoverCompiler(const Config& config)
    : config_(config)
{
    // Disable triggers for now - they cause deadlocks with limited channels
    config_.use_explicit_triggers = false;
}

CompiledSchedule BlockMoverCompiler::compile(const TileDataFlowGraph& dfg) {
    // Create a schedule first
    DFGScheduler scheduler;
    DFGSchedule schedule = scheduler.schedule(dfg);

    return compile(dfg, schedule);
}

CompiledSchedule BlockMoverCompiler::compile(const TileDataFlowGraph& dfg,
                                              const DFGSchedule& schedule) {
    CompiledSchedule result;

    // Store schedule for cycle-accurate timing
    schedule_ = &schedule;

    // Initialize programs
    for (uint8_t i = 0; i < 16; ++i) {
        result.blockmover_programs[i].l3_id = i;
        result.blockmover_programs[i].name = "L3_" + std::to_string(i);
    }

    // Reset trigger allocation
    trigger_alloc_ = TriggerAllocation{};

    // Phase 1: Assign triggers
    if (config_.use_explicit_triggers) {
        assign_triggers(dfg);
    }

    // Phase 2: Emit commands for each node in SCHEDULED order (by start_cycle)
    // Using topological order alone isn't sufficient for cycle-accurate scheduling
    // because it doesn't respect timing relationships between independent nodes.
    std::vector<size_t> order = dfg.topological_order();

    // Sort by scheduled start cycle while maintaining topological order as tie-breaker
    // This ensures WAIT_UNTIL_CYCLE commands are emitted in monotonically increasing order
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        auto sched_a = schedule.find_node(a);
        auto sched_b = schedule.find_node(b);
        int64_t start_a = sched_a ? sched_a->start_cycle : 0;
        int64_t start_b = sched_b ? sched_b->start_cycle : 0;
        if (start_a != start_b) {
            return start_a < start_b;
        }
        // Tie-breaker: lower node ID first
        return a < b;
    });

    // Emit all nodes in scheduled order
    for (size_t node_id : order) {
        emit_node(dfg.node(node_id), dfg, result);
    }

    // Phase 3: Insert barriers
    if (config_.emit_barriers) {
        insert_barriers(result);
    }

    // Phase 4: Assemble programs
    assemble_programs(result);

    // Compute statistics
    result.stats = CompiledSchedule::Stats{};
    for (const auto& prog : result.blockmover_programs) {
        result.stats.total_commands += prog.size();
        for (const auto& cmd : prog.commands) {
            switch (cmd.op) {
                case BlockMoverOp::PUSH_TO_L2:
                case BlockMoverOp::PULL_FROM_L2:
                    result.stats.total_l2_transfers++;
                    break;
                case BlockMoverOp::SEND_EAST:
                case BlockMoverOp::SEND_WEST:
                case BlockMoverOp::SEND_NORTH:
                case BlockMoverOp::SEND_SOUTH:
                case BlockMoverOp::SEND_TO:
                    result.stats.total_l3_transfers++;
                    break;
                case BlockMoverOp::EMIT_TRIGGER:
                    result.stats.total_triggers++;
                    break;
                case BlockMoverOp::BARRIER:
                    result.stats.total_barriers++;
                    break;
                default:
                    break;
            }
        }
    }

    // Use schedule for timing
    result.estimated_cycles = schedule.makespan;

    // Estimate compute vs data movement
    for (const auto& node : dfg.nodes()) {
        if (node.type == DFNodeType::MATMUL) {
            result.compute_cycles += node.duration;
            result.stats.total_compute_ops++;
        } else if (node.type == DFNodeType::DMA_LOAD ||
                   node.type == DFNodeType::DMA_STORE) {
            result.data_movement_cycles += node.duration;
            result.stats.total_dma_ops++;
        } else if (node.type == DFNodeType::L3_TRANSFER ||
                   node.type == DFNodeType::L3_TO_L2 ||
                   node.type == DFNodeType::L2_TO_L3) {
            result.data_movement_cycles += node.duration;
        }
    }

    last_stats_ = result.stats;

    // Clear schedule pointer to avoid dangling references
    schedule_ = nullptr;

    return result;
}

void BlockMoverCompiler::assign_triggers(const TileDataFlowGraph& dfg) {
    // Assign a trigger channel to each node that has successors
    // Successors will wait on this trigger
    for (const auto& node : dfg.nodes()) {
        if (!node.successors.empty()) {
            uint8_t trigger = allocate_trigger();
            trigger_alloc_.node_to_trigger[node.id] = trigger;
        }
    }
}

void BlockMoverCompiler::emit_node(const DFNode& node, const TileDataFlowGraph& dfg,
                                    CompiledSchedule& schedule) {
    BlockMoverProgram& prog = schedule.program(node.l3_id);

    // Insert WAIT_UNTIL_CYCLE for cycle-accurate scheduling
    // This ensures operations start at their scheduled times, providing
    // implicit synchronization without barriers for systolic schedules
    if (config_.emit_cycle_waits && schedule_) {
        auto sched_node = schedule_->find_node(node.id);
        if (sched_node && sched_node->start_cycle > 0) {
            prog.wait_until_cycle(static_cast<uint64_t>(sched_node->start_cycle));
        }
    }

    // Wait for predecessors if using explicit triggers
    if (config_.use_explicit_triggers && !node.predecessors.empty()) {
        uint16_t wait_mask = 0;
        for (size_t pred_id : node.predecessors) {
            auto it = trigger_alloc_.node_to_trigger.find(pred_id);
            if (it != trigger_alloc_.node_to_trigger.end()) {
                wait_mask |= (1 << it->second);
            }
        }
        if (wait_mask != 0) {
            prog.wait_trigger(wait_mask);
        }
    }

    // Emit the node-specific commands
    switch (node.type) {
        case DFNodeType::DMA_LOAD:
            emit_dma_load(node, prog);
            break;
        case DFNodeType::DMA_STORE:
            emit_dma_store(node, prog);
            break;
        case DFNodeType::L3_TRANSFER:
            emit_l3_transfer(node, schedule);
            break;
        case DFNodeType::L3_TO_L2:
            emit_l3_to_l2(node, prog);
            break;
        case DFNodeType::L2_TO_L3:
            emit_l2_to_l3(node, prog);
            break;
        case DFNodeType::MATMUL:
            emit_matmul(node, dfg, prog);
            break;
        case DFNodeType::BARRIER:
            emit_barrier(node, prog);
            break;
        default:
            // Other node types not yet implemented
            break;
    }

    // Emit trigger if this node has successors
    if (config_.use_explicit_triggers && !node.successors.empty()) {
        auto it = trigger_alloc_.node_to_trigger.find(node.id);
        if (it != trigger_alloc_.node_to_trigger.end()) {
            // Determine which L3s need this trigger
            uint16_t dest_mask = 0;
            for (size_t succ_id : node.successors) {
                const DFNode& succ = dfg.node(succ_id);
                dest_mask |= (1 << succ.l3_id);
            }
            prog.emit_trigger(it->second, dest_mask);
        }
    }
}

void BlockMoverCompiler::emit_dma_load(const DFNode& node, BlockMoverProgram& prog) {
    // For DMA loads, we model this as data arriving in L3
    // The actual DMA engine is separate, but we need to signal when data is ready

    // For now, emit a TRACE_MARKER to indicate DMA load
    BlockMoverCommand cmd;
    cmd.op = BlockMoverOp::TRACE_MARKER;
    cmd.trace_id = static_cast<uint32_t>(node.id);
    cmd.tile = node.tile;
    prog.append(cmd);

    // In a real implementation, the DMA engine would:
    // 1. Transfer data from External Memory to L3
    // 2. Send a trigger when complete

    // We simulate this with an implicit delay (handled by scheduler)
}

void BlockMoverCompiler::emit_dma_store(const DFNode& node, BlockMoverProgram& prog) {
    // Similar to DMA load, emit a marker
    BlockMoverCommand cmd;
    cmd.op = BlockMoverOp::TRACE_MARKER;
    cmd.trace_id = static_cast<uint32_t>(node.id);
    cmd.tile = node.tile;
    prog.append(cmd);
}

void BlockMoverCompiler::emit_l3_transfer(const DFNode& node, CompiledSchedule& schedule) {
    // Get source and destination programs
    BlockMoverProgram& src_prog = schedule.program(node.src_l3_id);
    BlockMoverProgram& dst_prog = schedule.program(node.dst_l3_id);

    // Skip if source == destination
    if (node.src_l3_id == node.dst_l3_id) {
        return;
    }

    // Emit RECEIVE on destination L3 first (it will block until tile arrives)
    // Add WAIT_UNTIL_CYCLE for the destination to ensure correct tile ordering
    // The tile arrives at end_cycle (start_cycle + propagation delay)
    if (config_.emit_cycle_waits && schedule_) {
        auto sched_node = schedule_->find_node(node.id);
        if (sched_node && sched_node->end_cycle > 0) {
            dst_prog.wait_until_cycle(static_cast<uint64_t>(sched_node->end_cycle));
        }
    }

    {
        // RECEIVE_FROM filters by source L3. This ensures we get the right packet
        // when multiple sources send to the same destination at the same cycle.
        // The WAIT_UNTIL_CYCLE ensures timing; RECEIVE_FROM ensures correctness.
        BlockMoverCommand recv_cmd;
        recv_cmd.op = BlockMoverOp::RECEIVE_FROM;
        recv_cmd.tile = node.tile;
        recv_cmd.src_l3_id = node.src_l3_id;
        dst_prog.append(recv_cmd);
    }

    // Emit SEND on source L3
    auto dir = get_direction(node.src_l3_id, node.dst_l3_id);

    if (dir && *dir != Direction::LOCAL) {
        BlockMoverCommand cmd;
        switch (*dir) {
            case Direction::NORTH: cmd.op = BlockMoverOp::SEND_NORTH; break;
            case Direction::SOUTH: cmd.op = BlockMoverOp::SEND_SOUTH; break;
            case Direction::EAST: cmd.op = BlockMoverOp::SEND_EAST; break;
            case Direction::WEST: cmd.op = BlockMoverOp::SEND_WEST; break;
            default: break;
        }
        cmd.tile = node.tile;
        src_prog.append(cmd);
    } else {
        // Non-neighbor transfer, use SEND_TO
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::SEND_TO;
        cmd.tile = node.tile;
        cmd.dest_l3_id = node.dst_l3_id;
        src_prog.append(cmd);
    }

    // Emit WAIT_DELIVERY after SEND to synchronize K iterations
    // This ensures A[m,k] is delivered before A[m,k+1] is sent
    if (config_.sync_sends) {
        BlockMoverCommand wait_cmd;
        wait_cmd.op = BlockMoverOp::WAIT_DELIVERY;
        src_prog.append(wait_cmd);
    }
}

void BlockMoverCompiler::emit_l3_to_l2(const DFNode& node, BlockMoverProgram& prog) {
    prog.push_to_l2(node.tile, node.l2_bank_id);
}

void BlockMoverCompiler::emit_l2_to_l3(const DFNode& node, BlockMoverProgram& prog) {
    BlockMoverCommand cmd;
    cmd.op = BlockMoverOp::PULL_FROM_L2;
    cmd.tile = node.tile;
    cmd.l2_bank_id = node.l2_bank_id;
    prog.append(cmd);
}

void BlockMoverCompiler::emit_matmul(const DFNode& node, const TileDataFlowGraph& dfg,
                                      BlockMoverProgram& prog) {
    (void)dfg;

    // For matmul, we need to:
    // 1. Push A tile to L2 (if not already there)
    // 2. Push B tile to L2
    // 3. Signal compute start (in real hardware, systolic array would compute)
    // 4. For simulation, we model compute as instant (no COMPUTE_DONE wait)
    // 5. Result is in L1, will be drained later

    // Create A tile descriptor
    TileDescriptor a_tile;
    a_tile.tensor = TensorId::A;
    a_tile.m_tile = node.tile.m_tile;
    a_tile.k_tile = node.tile.k_tile;
    a_tile.size = node.M * node.K * sizeof(float);

    // Create B tile descriptor
    TileDescriptor b_tile;
    b_tile.tensor = TensorId::B;
    b_tile.n_tile = node.tile.n_tile;
    b_tile.k_tile = node.tile.k_tile;
    b_tile.size = node.K * node.N * sizeof(float);

    // Push A and B to L2
    prog.push_to_l2(a_tile, node.l3_id);  // Using L3 ID as L2 bank for simplicity
    prog.push_to_l2(b_tile, node.l3_id);

    // In simulation, compute is modeled as instant
    // Real hardware would signal compute start and wait for completion
    // The timing is captured in the schedule's node.duration

    // Add a trace marker to indicate compute happened
    BlockMoverCommand cmd;
    cmd.op = BlockMoverOp::TRACE_MARKER;
    cmd.trace_id = static_cast<uint32_t>(node.id);
    cmd.tile = node.tile;
    prog.append(cmd);

    // Note: C tile draining is handled by a separate L2_TO_L3 or DMA_STORE node
}

void BlockMoverCompiler::emit_barrier(const DFNode& node, BlockMoverProgram& prog) {
    (void)node;
    prog.barrier();
}

void BlockMoverCompiler::insert_barriers(CompiledSchedule& schedule) {
    // Insert a barrier at the end of each program
    for (auto& prog : schedule.blockmover_programs) {
        if (!prog.empty()) {
            prog.barrier();
        }
    }
}

void BlockMoverCompiler::assemble_programs(CompiledSchedule& schedule) {
    // Sort commands within each program by cycle values
    // This ensures WAIT_UNTIL_CYCLE commands are in ascending order
    for (auto& prog : schedule.blockmover_programs) {
        if (prog.commands.empty()) continue;

        // Group commands into (WAIT_UNTIL_CYCLE, subsequent commands) pairs
        // Each group is executed at a specific cycle
        struct CommandGroup {
            uint64_t cycle = 0;
            std::vector<BlockMoverCommand> cmds;
        };
        std::vector<CommandGroup> groups;
        CommandGroup current_group;
        current_group.cycle = 0;

        for (const auto& cmd : prog.commands) {
            if (cmd.op == BlockMoverOp::WAIT_UNTIL_CYCLE) {
                // Start a new group if current group has commands
                if (!current_group.cmds.empty()) {
                    groups.push_back(std::move(current_group));
                    current_group = CommandGroup{};
                }
                current_group.cycle = cmd.target_cycle;
                current_group.cmds.push_back(cmd);
            } else {
                current_group.cmds.push_back(cmd);
            }
        }
        // Add the last group
        if (!current_group.cmds.empty()) {
            groups.push_back(std::move(current_group));
        }

        // Sort groups by cycle value, with secondary sort by command type
        // RECEIVE_FROM must come before SEND in the same cycle (data must arrive before forwarding)
        // Priority order for systolic efficiency:
        // 1. RECEIVE: Must receive data before using it
        // 2. SEND: Forward data ASAP to minimize latency for downstream L3s
        // 3. PUSH_TO_L2: Local L2 operations (reads from L3, no dependency on SEND)
        // 4. TRACE_MARKER: Just logging, lowest priority
        auto command_priority = [](const BlockMoverCommand& cmd) -> int {
            switch (cmd.op) {
                case BlockMoverOp::RECEIVE_FROM:
                    return 0;  // RECEIVE first (data must arrive before use)
                case BlockMoverOp::SEND_EAST:
                case BlockMoverOp::SEND_WEST:
                case BlockMoverOp::SEND_NORTH:
                case BlockMoverOp::SEND_SOUTH:
                case BlockMoverOp::SEND_TO:
                    return 1;  // SEND early to start tile moving through NoC
                case BlockMoverOp::PUSH_TO_L2:
                    return 2;  // Then L2 operations (can happen after or parallel to SEND)
                case BlockMoverOp::TRACE_MARKER:
                    return 3;  // Then markers
                default:
                    return 5;
            }
        };

        std::sort(groups.begin(), groups.end(),
                  [&command_priority](const CommandGroup& a, const CommandGroup& b) {
                      if (a.cycle != b.cycle) {
                          return a.cycle < b.cycle;
                      }
                      // Same cycle - sort by command type priority
                      // Find the primary command (first non-WAIT command)
                      int prio_a = 10, prio_b = 10;
                      for (const auto& cmd : a.cmds) {
                          if (cmd.op != BlockMoverOp::WAIT_UNTIL_CYCLE) {
                              prio_a = command_priority(cmd);
                              break;
                          }
                      }
                      for (const auto& cmd : b.cmds) {
                          if (cmd.op != BlockMoverOp::WAIT_UNTIL_CYCLE) {
                              prio_b = command_priority(cmd);
                              break;
                          }
                      }
                      return prio_a < prio_b;
                  });

        // Flatten back to command list
        prog.commands.clear();
        for (const auto& group : groups) {
            for (const auto& cmd : group.cmds) {
                prog.commands.push_back(cmd);
            }
        }

        prog.assemble();
    }
}

std::optional<Direction> BlockMoverCompiler::get_direction(uint8_t src_l3, uint8_t dst_l3) const {
    if (src_l3 == dst_l3) return Direction::LOCAL;

    auto [src_row, src_col] = get_mesh_position(src_l3);
    auto [dst_row, dst_col] = get_mesh_position(dst_l3);

    // Check if neighbors
    if (src_row == dst_row) {
        if (dst_col == src_col + 1) return Direction::EAST;
        if (dst_col + 1 == src_col) return Direction::WEST;
    }
    if (src_col == dst_col) {
        if (dst_row == src_row + 1) return Direction::SOUTH;
        if (dst_row + 1 == src_row) return Direction::NORTH;
    }

    return std::nullopt;  // Not neighbors
}

uint8_t BlockMoverCompiler::allocate_trigger() {
    uint8_t trigger = trigger_alloc_.next_trigger;
    trigger_alloc_.next_trigger = (trigger_alloc_.next_trigger + 1) % TriggerChannel::NUM_CHANNELS;
    return trigger;
}

uint8_t BlockMoverCompiler::get_trigger_for_node(size_t node_id) {
    auto it = trigger_alloc_.node_to_trigger.find(node_id);
    if (it != trigger_alloc_.node_to_trigger.end()) {
        return it->second;
    }
    // Allocate new trigger
    uint8_t trigger = allocate_trigger();
    trigger_alloc_.node_to_trigger[node_id] = trigger;
    return trigger;
}

std::pair<uint8_t, uint8_t> BlockMoverCompiler::get_mesh_position(uint8_t l3_id) const {
    // C integer promotion bumps uint8_t / uint8_t to int, then the pair
    // brace-init narrows it back - MSVC C4244 in <utility>. Cast at the
    // construction site so the narrowing is explicit.
    return {static_cast<uint8_t>(l3_id / config_.mesh_cols),
            static_cast<uint8_t>(l3_id % config_.mesh_cols)};
}

uint8_t BlockMoverCompiler::get_l3_id(uint8_t row, uint8_t col) const {
    return row * config_.mesh_cols + col;
}

// ============================================================================
// Convenience Functions
// ============================================================================

CompiledSchedule compile_matmul(
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles,
    const BlockMoverCompiler::Config& config)
{
    // Build DFG for matmul
    TileDataFlowGraph dfg = TileDataFlowGraph::build_matmul_output_stationary(
        M, N, K, m_tiles, n_tiles, k_tiles);

    // Compile to BlockMover programs
    BlockMoverCompiler compiler(config);
    return compiler.compile(dfg);
}

void print_compiled_schedule(const CompiledSchedule& schedule) {
    schedule.print_summary();

    std::cout << "\nDetailed programs:\n";
    for (uint8_t i = 0; i < 16; ++i) {
        const auto& prog = schedule.blockmover_programs[i];
        if (!prog.empty()) {
            std::cout << "\n" << prog.to_string() << "\n";
        }
    }
}

} // namespace sw::kpu::dataflow
