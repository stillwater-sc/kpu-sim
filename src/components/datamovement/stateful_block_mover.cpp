// ============================================================================
// src/components/datamovement/stateful_block_mover.cpp
// Stateful BlockMover Implementation
// ============================================================================

#include "sw/kpu/components/stateful_block_mover.hpp"
#include <sstream>
#include <iomanip>

namespace sw::kpu {

// ============================================================================
// StatefulBlockMover Implementation
// ============================================================================

StatefulBlockMover::StatefulBlockMover(const StatefulBlockMoverConfig& config)
    : config_(config)
    , state_(BlockMoverState::IDLE)
{
}

// ========== Program Loading ==========

void StatefulBlockMover::load_program(const BlockMoverProgram& program) {
    program_ = program.commands;
    pc_ = 0;
    state_ = program_.empty() ? BlockMoverState::IDLE : BlockMoverState::RUNNING;

    // Clear execution state
    while (!loop_stack_.empty()) loop_stack_.pop();
    triggers_received_.reset();
    trigger_wait_mask_ = 0;
    while (!pending_transfers_.empty()) pending_transfers_.pop();
    while (!receive_queue_.empty()) receive_queue_.pop();
    expected_receive_.reset();
}

void StatefulBlockMover::load_program(const std::vector<BlockMoverCommand>& commands) {
    BlockMoverProgram prog;
    prog.l3_id = config_.id;
    prog.commands = commands;
    load_program(prog);
}

void StatefulBlockMover::clear_program() {
    program_.clear();
    pc_ = 0;
    state_ = BlockMoverState::IDLE;
}

// ========== Execution Control ==========

bool StatefulBlockMover::step(uint64_t cycle) {
    if (state_ == BlockMoverState::IDLE || state_ == BlockMoverState::HALTED) {
        return false;
    }

    if (state_ == BlockMoverState::ERROR) {
        return false;
    }

    // Process any completed transfers
    process_completed_transfers(cycle);

    // Check if we can proceed from waiting state
    if (is_waiting()) {
        if (!check_wait_conditions(cycle)) {
            // Still waiting
            switch (state_) {
                case BlockMoverState::WAITING_TRIGGER:
                    stats_.cycles_waiting_trigger++;
                    break;
                case BlockMoverState::WAITING_TRANSFER:
                    stats_.cycles_waiting_transfer++;
                    break;
                case BlockMoverState::WAITING_RECEIVE:
                    stats_.cycles_waiting_receive++;
                    break;
                default:
                    break;
            }
            return true;  // Still executing (waiting)
        }
        state_ = BlockMoverState::RUNNING;
    }

    // Execute current command
    if (pc_ < program_.size()) {
        stats_.cycles_active++;

        if (execute_current()) {
            advance_pc();
        }
        // If execute_current returns false, we're blocked (waiting)
    }

    // Check if program complete
    if (state_ == BlockMoverState::RUNNING && pc_ >= program_.size()) {
        state_ = BlockMoverState::IDLE;
        return false;
    }

    return state_ != BlockMoverState::IDLE && state_ != BlockMoverState::HALTED;
}

void StatefulBlockMover::reset() {
    pc_ = 0;
    state_ = program_.empty() ? BlockMoverState::IDLE : BlockMoverState::RUNNING;

    while (!loop_stack_.empty()) loop_stack_.pop();
    triggers_received_.reset();
    trigger_wait_mask_ = 0;
    while (!pending_transfers_.empty()) pending_transfers_.pop();
    while (!receive_queue_.empty()) receive_queue_.pop();
    expected_receive_.reset();
}

void StatefulBlockMover::halt() {
    state_ = BlockMoverState::HALTED;
}

void StatefulBlockMover::resume() {
    if (state_ == BlockMoverState::HALTED) {
        state_ = BlockMoverState::RUNNING;
    }
}

// ========== External Events ==========

void StatefulBlockMover::receive_trigger(uint8_t channel) {
    if (channel < 16) {
        triggers_received_.set(channel);
        stats_.triggers_received++;
    }
}

void StatefulBlockMover::receive_packet(const L3TransferPacket& packet) {
    receive_queue_.push(packet);
}

void StatefulBlockMover::notify_l2_transfer_complete(const TileDescriptor& tile) {
    // Mark corresponding pending transfer as complete
    // This is a simplification - in reality we'd match by tile descriptor
    (void)tile;
}

// ========== State Query ==========

std::string StatefulBlockMover::to_string() const {
    std::ostringstream oss;
    oss << "BlockMover[" << static_cast<int>(config_.id) << "] "
        << sw::kpu::to_string(state_)
        << " PC=" << pc_ << "/" << program_.size();

    if (pending_transfers_.size() > 0) {
        oss << " pending=" << pending_transfers_.size();
    }
    if (receive_queue_.size() > 0) {
        oss << " rx_queue=" << receive_queue_.size();
    }

    return oss.str();
}

std::string StatefulBlockMover::dump_state() const {
    std::ostringstream oss;

    oss << "=== BlockMover[" << static_cast<int>(config_.id) << "] State ===\n";
    oss << "State: " << sw::kpu::to_string(state_) << "\n";
    oss << "PC: " << pc_ << " / " << program_.size() << "\n";

    if (pc_ < program_.size()) {
        oss << "Current: " << program_[pc_].to_string() << "\n";
    }

    oss << "Triggers received: 0x" << std::hex << triggers_received_.to_ulong() << std::dec << "\n";

    if (trigger_wait_mask_ != 0) {
        oss << "Waiting for triggers: 0x" << std::hex << trigger_wait_mask_ << std::dec << "\n";
    }

    oss << "Pending transfers: " << pending_transfers_.size() << "\n";
    oss << "Receive queue: " << receive_queue_.size() << "\n";
    oss << "Loop stack depth: " << loop_stack_.size() << "\n";

    oss << "\n--- Statistics ---\n";
    oss << "Commands executed: " << stats_.commands_executed << "\n";
    oss << "Cycles active: " << stats_.cycles_active << "\n";
    oss << "Cycles waiting (trigger): " << stats_.cycles_waiting_trigger << "\n";
    oss << "Cycles waiting (transfer): " << stats_.cycles_waiting_transfer << "\n";
    oss << "Cycles waiting (receive): " << stats_.cycles_waiting_receive << "\n";
    oss << "Utilization: " << std::fixed << std::setprecision(1)
        << (stats_.utilization() * 100) << "%\n";

    return oss.str();
}

// ========== Internal Execution ==========

bool StatefulBlockMover::execute_current() {
    if (pc_ >= program_.size()) {
        return false;
    }

    const BlockMoverCommand& cmd = program_[pc_];
    stats_.commands_executed++;

    switch (cmd.op) {
        case BlockMoverOp::NOP:
            return true;

        case BlockMoverOp::PUSH_TO_L2:
            return exec_push_to_l2(cmd, 0);  // cycle not needed for now

        case BlockMoverOp::PULL_FROM_L2:
            return exec_pull_from_l2(cmd, 0);

        case BlockMoverOp::SEND_EAST:
            return exec_send(cmd, Direction::EAST, 0);

        case BlockMoverOp::SEND_WEST:
            return exec_send(cmd, Direction::WEST, 0);

        case BlockMoverOp::SEND_NORTH:
            return exec_send(cmd, Direction::NORTH, 0);

        case BlockMoverOp::SEND_SOUTH:
            return exec_send(cmd, Direction::SOUTH, 0);

        case BlockMoverOp::SEND_TO:
            return exec_send_to(cmd, 0);

        case BlockMoverOp::RECEIVE:
        case BlockMoverOp::RECEIVE_FROM:
            return exec_receive(cmd, 0);

        case BlockMoverOp::WAIT_TRIGGER:
            return exec_wait_trigger(cmd);

        case BlockMoverOp::EMIT_TRIGGER:
            return exec_emit_trigger(cmd);

        case BlockMoverOp::BARRIER:
            return exec_barrier(cmd, 0);

        case BlockMoverOp::FENCE:
            // Memory fence - ensure ordering (no-op in simulation)
            return true;

        case BlockMoverOp::LOOP_START:
            return exec_loop_start(cmd);

        case BlockMoverOp::LOOP_END:
            return exec_loop_end(cmd);

        case BlockMoverOp::JUMP:
            return exec_jump(cmd);

        case BlockMoverOp::JUMP_IF_TRIGGER:
            if (triggers_received_.test(cmd.trigger_id)) {
                pc_ = static_cast<size_t>(static_cast<int>(pc_) + cmd.jump_offset);
                return false;  // Don't advance PC (we just set it)
            }
            return true;

        case BlockMoverOp::SET_L2_BANK:
            return exec_set_l2_bank(cmd);

        case BlockMoverOp::SET_BUFFER:
            // Configure buffer region (future use)
            return true;

        case BlockMoverOp::TRACE_MARKER:
            if (trace_callback_) {
                trace_callback_(cmd.trace_id, 0);
            }
            return true;

        case BlockMoverOp::HALT:
            state_ = BlockMoverState::HALTED;
            return false;

        default:
            state_ = BlockMoverState::ERROR;
            return false;
    }
}

void StatefulBlockMover::advance_pc() {
    pc_++;
}

bool StatefulBlockMover::check_wait_conditions(uint64_t cycle) {
    switch (state_) {
        case BlockMoverState::WAITING_TRIGGER: {
            // Check if all required triggers are set
            uint16_t received = static_cast<uint16_t>(triggers_received_.to_ulong());
            if ((received & trigger_wait_mask_) == trigger_wait_mask_) {
                // Clear the triggers we were waiting for
                triggers_received_ &= std::bitset<16>(~trigger_wait_mask_);
                trigger_wait_mask_ = 0;
                return true;
            }
            return false;
        }

        case BlockMoverState::WAITING_TRANSFER: {
            // Check if all pending transfers are complete
            while (!pending_transfers_.empty() &&
                   pending_transfers_.front().is_complete(cycle)) {
                pending_transfers_.pop();
            }
            return pending_transfers_.empty();
        }

        case BlockMoverState::WAITING_RECEIVE: {
            // Check if we have a packet in the receive queue
            if (!receive_queue_.empty()) {
                if (expected_receive_) {
                    // Check if it matches what we're expecting
                    const auto& packet = receive_queue_.front();
                    if (packet.tile == *expected_receive_) {
                        receive_queue_.pop();
                        expected_receive_.reset();
                        return true;
                    }
                } else {
                    // Accept any packet
                    receive_queue_.pop();
                    return true;
                }
            }
            return false;
        }

        default:
            return true;
    }
}

void StatefulBlockMover::process_completed_transfers(uint64_t cycle) {
    while (!pending_transfers_.empty() &&
           pending_transfers_.front().is_complete(cycle)) {
        pending_transfers_.pop();
    }
}

bool StatefulBlockMover::exec_push_to_l2(const BlockMoverCommand& cmd, uint64_t cycle) {
    // Create pending transfer
    PendingTransfer transfer;
    transfer.type = PendingTransfer::Type::L3_TO_L2;
    transfer.tile = cmd.tile;
    transfer.dest_id = cmd.l2_bank_id;
    transfer.bytes_total = cmd.tile.size;
    transfer.bytes_transferred = 0;
    transfer.start_cycle = cycle;
    transfer.complete_cycle = calc_transfer_complete_cycle(
        PendingTransfer::Type::L3_TO_L2, cmd.tile.size, cycle);

    pending_transfers_.push(transfer);

    stats_.l3_to_l2_transfers++;
    stats_.l3_to_l2_bytes += cmd.tile.size;

    return true;  // Command issued, can proceed
}

bool StatefulBlockMover::exec_pull_from_l2(const BlockMoverCommand& cmd, uint64_t cycle) {
    PendingTransfer transfer;
    transfer.type = PendingTransfer::Type::L2_TO_L3;
    transfer.tile = cmd.tile;
    transfer.dest_id = cmd.l2_bank_id;
    transfer.bytes_total = cmd.tile.size;
    transfer.start_cycle = cycle;
    transfer.complete_cycle = calc_transfer_complete_cycle(
        PendingTransfer::Type::L2_TO_L3, cmd.tile.size, cycle);

    pending_transfers_.push(transfer);

    return true;
}

bool StatefulBlockMover::exec_send(const BlockMoverCommand& cmd, Direction dir, uint64_t cycle) {
    int neighbor = config_.get_neighbor(dir);
    if (neighbor < 0) {
        // No neighbor in this direction - error or skip
        state_ = BlockMoverState::ERROR;
        return false;
    }

    PendingTransfer transfer;
    transfer.type = PendingTransfer::Type::L3_TO_L3;
    transfer.tile = cmd.tile;
    transfer.dest_id = static_cast<uint8_t>(neighbor);
    transfer.bytes_total = cmd.tile.size;
    transfer.start_cycle = cycle;
    transfer.complete_cycle = calc_transfer_complete_cycle(
        PendingTransfer::Type::L3_TO_L3, cmd.tile.size, cycle);

    pending_transfers_.push(transfer);

    // Notify via callback
    if (transfer_callback_) {
        transfer_callback_(cmd.tile, static_cast<uint8_t>(neighbor));
    }

    stats_.l3_to_l3_transfers++;
    stats_.l3_to_l3_bytes += cmd.tile.size;

    return true;
}

bool StatefulBlockMover::exec_send_to(const BlockMoverCommand& cmd, uint64_t cycle) {
    PendingTransfer transfer;
    transfer.type = PendingTransfer::Type::L3_TO_L3;
    transfer.tile = cmd.tile;
    transfer.dest_id = cmd.dest_l3_id;
    transfer.bytes_total = cmd.tile.size;
    transfer.start_cycle = cycle;
    transfer.complete_cycle = calc_transfer_complete_cycle(
        PendingTransfer::Type::L3_TO_L3, cmd.tile.size, cycle);

    pending_transfers_.push(transfer);

    if (transfer_callback_) {
        transfer_callback_(cmd.tile, cmd.dest_l3_id);
    }

    stats_.l3_to_l3_transfers++;
    stats_.l3_to_l3_bytes += cmd.tile.size;

    return true;
}

bool StatefulBlockMover::exec_receive(const BlockMoverCommand& cmd, uint64_t cycle) {
    (void)cycle;

    // Check if we already have a packet waiting
    if (!receive_queue_.empty()) {
        if (cmd.op == BlockMoverOp::RECEIVE_FROM) {
            // Check if it's from the expected source
            if (receive_queue_.front().src_l3_id == cmd.src_l3_id) {
                receive_queue_.pop();
                return true;
            }
        } else {
            // Accept any packet
            receive_queue_.pop();
            return true;
        }
    }

    // Need to wait for packet
    state_ = BlockMoverState::WAITING_RECEIVE;
    if (cmd.op == BlockMoverOp::RECEIVE_FROM) {
        // Remember what we're expecting
        expected_receive_ = cmd.tile;
    }
    return false;
}

bool StatefulBlockMover::exec_wait_trigger(const BlockMoverCommand& cmd) {
    uint16_t received = static_cast<uint16_t>(triggers_received_.to_ulong());

    // Check if already satisfied
    if ((received & cmd.trigger_wait_mask) == cmd.trigger_wait_mask) {
        // Clear the triggers
        triggers_received_ &= std::bitset<16>(~cmd.trigger_wait_mask);
        return true;
    }

    // Need to wait
    trigger_wait_mask_ = cmd.trigger_wait_mask;
    state_ = BlockMoverState::WAITING_TRIGGER;
    return false;
}

bool StatefulBlockMover::exec_emit_trigger(const BlockMoverCommand& cmd) {
    if (trigger_callback_) {
        trigger_callback_(cmd.trigger_id, cmd.trigger_dest_mask);
    }
    stats_.triggers_sent++;
    return true;
}

bool StatefulBlockMover::exec_barrier(const BlockMoverCommand& cmd, uint64_t cycle) {
    (void)cmd;

    // Check if all pending transfers are complete
    while (!pending_transfers_.empty() &&
           pending_transfers_.front().is_complete(cycle)) {
        pending_transfers_.pop();
    }

    if (pending_transfers_.empty()) {
        return true;
    }

    state_ = BlockMoverState::WAITING_TRANSFER;
    return false;
}

bool StatefulBlockMover::exec_loop_start(const BlockMoverCommand& cmd) {
    if (cmd.loop_count == 0) {
        // Skip to matching LOOP_END
        // For now, just skip one command (simplified)
        return true;
    }

    LoopContext ctx;
    ctx.start_pc = pc_;
    ctx.remaining = cmd.loop_count;
    loop_stack_.push(ctx);
    stats_.loops_executed++;

    return true;
}

bool StatefulBlockMover::exec_loop_end(const BlockMoverCommand& cmd) {
    if (loop_stack_.empty()) {
        // Unbalanced loop - error or just continue
        return true;
    }

    LoopContext& ctx = loop_stack_.top();
    ctx.remaining--;

    if (ctx.remaining > 0) {
        // Jump back to loop start
        pc_ = ctx.start_pc;
        return false;  // Don't advance PC
    } else {
        // Loop complete
        loop_stack_.pop();
        return true;
    }
}

bool StatefulBlockMover::exec_jump(const BlockMoverCommand& cmd) {
    int new_pc = static_cast<int>(pc_) + cmd.jump_offset;
    if (new_pc < 0 || new_pc >= static_cast<int>(program_.size())) {
        state_ = BlockMoverState::ERROR;
        return false;
    }
    pc_ = static_cast<size_t>(new_pc);
    return false;  // Don't advance PC (we just set it)
}

bool StatefulBlockMover::exec_set_l2_bank(const BlockMoverCommand& cmd) {
    current_l2_bank_ = cmd.l2_bank_id;
    return true;
}

uint64_t StatefulBlockMover::calc_transfer_complete_cycle(
    PendingTransfer::Type type, size_t bytes, uint64_t start) const
{
    uint32_t bandwidth = 0;
    uint32_t latency = 0;

    switch (type) {
        case PendingTransfer::Type::L3_TO_L2:
        case PendingTransfer::Type::L2_TO_L3:
            bandwidth = config_.l3_to_l2_bandwidth_bytes_per_cycle;
            latency = config_.l3_to_l2_latency_cycles;
            break;

        case PendingTransfer::Type::L3_TO_L3:
            bandwidth = config_.l3_to_l3_bandwidth_bytes_per_cycle;
            latency = config_.l3_to_l3_latency_cycles;
            break;
    }

    uint32_t transfer_cycles = (bytes + bandwidth - 1) / bandwidth;
    return start + latency + transfer_cycles;
}

// ============================================================================
// BlockMoverArray Implementation
// ============================================================================

BlockMoverArray::BlockMoverArray()
    : BlockMoverArray(Config{})
{
}

BlockMoverArray::BlockMoverArray(const Config& config)
    : config_(config)
    , interconnect_(L3InterconnectConfig{
          InterconnectTopology::MESH_2D,
          config.rows,
          config.cols,
          config.mover_config.l3_to_l3_bandwidth_bytes_per_cycle,
          config.mover_config.l3_to_l3_latency_cycles
      })
    , triggers_(config.rows * config.cols)
{
    size_t num_tiles = config.rows * config.cols;
    movers_.reserve(num_tiles);

    for (size_t id = 0; id < num_tiles; ++id) {
        StatefulBlockMoverConfig mover_config = config.mover_config;
        mover_config.configure_from_mesh(static_cast<uint8_t>(id), config.rows, config.cols);

        movers_.push_back(std::make_unique<StatefulBlockMover>(mover_config));
    }

    setup_callbacks();
}

void BlockMoverArray::setup_callbacks() {
    // Set up transfer callbacks to inject packets into interconnect
    for (size_t id = 0; id < movers_.size(); ++id) {
        movers_[id]->set_transfer_callback(
            [this, id](const TileDescriptor& tile, uint8_t dest_l3) {
                L3TransferPacket packet;
                packet.src_l3_id = static_cast<uint8_t>(id);
                packet.dst_l3_id = dest_l3;
                packet.tile = tile;
                // Note: actual data not copied in simulation
                interconnect_.inject_packet(packet, 0);  // cycle managed externally
            });

        movers_[id]->set_trigger_callback(
            [this, id](uint8_t channel, uint16_t dest_mask) {
                triggers_.send_trigger(static_cast<uint8_t>(id), channel, dest_mask);
            });
    }

    // Set up interconnect delivery callbacks
    for (size_t id = 0; id < movers_.size(); ++id) {
        interconnect_.set_delivery_callback(static_cast<uint8_t>(id),
            [this, id](const L3TransferPacket& packet) {
                movers_[id]->receive_packet(packet);
            });
    }
}

void BlockMoverArray::load_programs(const std::vector<BlockMoverProgram>& programs) {
    for (const auto& prog : programs) {
        if (prog.l3_id < movers_.size()) {
            movers_[prog.l3_id]->load_program(prog);
        }
    }
}

void BlockMoverArray::step(uint64_t cycle) {
    // Step all BlockMovers
    for (auto& mover : movers_) {
        mover->step(cycle);
    }

    // Step interconnect
    interconnect_.step(cycle);

    // Step trigger network
    triggers_.step(cycle);

    // Deliver triggers to BlockMovers
    for (size_t id = 0; id < movers_.size(); ++id) {
        for (uint8_t ch = 0; ch < 16; ++ch) {
            if (triggers_.is_trigger_pending(static_cast<uint8_t>(id), ch)) {
                movers_[id]->receive_trigger(ch);
                triggers_.clear_trigger(static_cast<uint8_t>(id), ch);
            }
        }
    }
}

void BlockMoverArray::reset() {
    for (auto& mover : movers_) {
        mover->reset();
    }
    interconnect_.reset_stats();
    triggers_.reset();
}

bool BlockMoverArray::all_idle() const {
    for (const auto& mover : movers_) {
        if (!mover->is_idle()) return false;
    }
    return interconnect_.is_idle();
}

bool BlockMoverArray::any_running() const {
    for (const auto& mover : movers_) {
        if (mover->is_running() || mover->is_waiting()) return true;
    }
    return false;
}

size_t BlockMoverArray::num_running() const {
    size_t count = 0;
    for (const auto& mover : movers_) {
        if (mover->is_running()) count++;
    }
    return count;
}

size_t BlockMoverArray::num_waiting() const {
    size_t count = 0;
    for (const auto& mover : movers_) {
        if (mover->is_waiting()) count++;
    }
    return count;
}

BlockMoverArray::AggregateStats BlockMoverArray::get_aggregate_stats() const {
    AggregateStats agg{};

    double total_util = 0;
    for (const auto& mover : movers_) {
        const auto& s = mover->stats();
        agg.total_commands += s.commands_executed;
        agg.total_l3_l3_transfers += s.l3_to_l3_transfers;
        agg.total_l3_l2_transfers += s.l3_to_l2_transfers;
        agg.total_bytes_moved += s.l3_to_l2_bytes + s.l3_to_l3_bytes;
        total_util += s.utilization();
    }

    agg.avg_utilization = movers_.empty() ? 0.0 : total_util / movers_.size();

    return agg;
}

} // namespace sw::kpu
