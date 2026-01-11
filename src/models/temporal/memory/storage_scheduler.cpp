#include <sw/kpu/components/storage_scheduler.hpp>
#include <sw/kpu/components/block_mover.hpp>
#include <sw/kpu/components/streamer.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sw::kpu {

// StorageScheduler Implementation
StorageScheduler::StorageScheduler(size_t scheduler_id, size_t num_banks, const BankConfig& default_config)
    : num_banks(num_banks), scheduler_id(scheduler_id), next_sequence_id(1) {

    bank_configs.resize(num_banks, default_config);
    bank_states.reserve(num_banks);

    // Initialize bank states
    for (size_t i = 0; i < num_banks; ++i) {
        auto bank_state = std::make_unique<BankState>();
        bank_state->capacity = default_config.bank_size_kb * 1024;
        bank_state->data.resize(bank_state->capacity, 0);
        bank_state->current_occupancy = 0;
        bank_state->is_reading = false;
        bank_state->is_writing = false;
        bank_state->current_operation = StorageOperation::BARRIER;
        bank_state->active_sequence_id = 0;
        bank_state->read_accesses = 0;
        bank_state->write_accesses = 0;
        bank_state->cache_hits = 0;
        bank_state->cache_misses = 0;

        bank_states.push_back(std::move(bank_state));
    }
}

StorageScheduler::StorageScheduler(const StorageScheduler& other)
    : num_banks(other.num_banks)
    , bank_configs(other.bank_configs)
    , scheduler_id(other.scheduler_id)
    , next_sequence_id(1) {

    // Deep copy bank states (manual copy due to atomic members)
    bank_states.reserve(num_banks);
    for (size_t i = 0; i < num_banks; ++i) {
        auto bank_state = std::make_unique<BankState>();
        const auto& other_bank = *other.bank_states[i];

        bank_state->data = other_bank.data;
        bank_state->capacity = other_bank.capacity;
        bank_state->current_occupancy = other_bank.current_occupancy;
        bank_state->is_reading = other_bank.is_reading.load();
        bank_state->is_writing = other_bank.is_writing.load();
        bank_state->current_operation = other_bank.current_operation;
        bank_state->active_sequence_id = other_bank.active_sequence_id;
        bank_state->read_accesses = other_bank.read_accesses.load();
        bank_state->write_accesses = other_bank.write_accesses.load();
        bank_state->cache_hits = other_bank.cache_hits.load();
        bank_state->cache_misses = other_bank.cache_misses.load();

        bank_states.push_back(std::move(bank_state));
    }

    // Note: We don't copy active commands or dependencies as they're runtime state
}

StorageScheduler& StorageScheduler::operator=(const StorageScheduler& other) {
    if (this != &other) {
        num_banks = other.num_banks;
        scheduler_id = other.scheduler_id;
        bank_configs = other.bank_configs;

        bank_states.clear();
        bank_states.reserve(num_banks);

        for (size_t i = 0; i < num_banks; ++i) {
            auto bank_state = std::make_unique<BankState>();
            const auto& other_bank = *other.bank_states[i];

            bank_state->data = other_bank.data;
            bank_state->capacity = other_bank.capacity;
            bank_state->current_occupancy = other_bank.current_occupancy;
            bank_state->is_reading = other_bank.is_reading.load();
            bank_state->is_writing = other_bank.is_writing.load();
            bank_state->current_operation = other_bank.current_operation;
            bank_state->active_sequence_id = other_bank.active_sequence_id;
            bank_state->read_accesses = other_bank.read_accesses.load();
            bank_state->write_accesses = other_bank.write_accesses.load();
            bank_state->cache_hits = other_bank.cache_hits.load();
            bank_state->cache_misses = other_bank.cache_misses.load();

            bank_states.push_back(std::move(bank_state));
        }

        // Clear runtime state
        std::lock_guard<std::mutex> cmd_lock(command_mutex);
        command_queue = std::queue<StorageCommand>();
        active_commands.clear();
        dependency_graph.clear();
    }
    return *this;
}

void StorageScheduler::configure_bank(size_t bank_id, const BankConfig& config) {
    if (bank_id >= num_banks) {
        throw std::out_of_range("Bank ID out of range");
    }

    std::lock_guard<std::mutex> lock(bank_mutex);
    bank_configs[bank_id] = config;

    // Resize bank storage if needed
    Size new_capacity = config.bank_size_kb * 1024;
    if (new_capacity != bank_states[bank_id]->capacity) {
        bank_states[bank_id]->data.resize(new_capacity, 0);
        bank_states[bank_id]->capacity = new_capacity;
        bank_states[bank_id]->current_occupancy = std::min(
            bank_states[bank_id]->current_occupancy, new_capacity);
    }
}

void StorageScheduler::register_block_mover(BlockMover* mover) {
    if (mover != nullptr) {
        block_movers.push_back(mover);
    }
}

void StorageScheduler::register_streamer(Streamer* streamer) {
    if (streamer != nullptr) {
        streamers.push_back(streamer);
    }
}

void StorageScheduler::direct_read(size_t bank_id, Address addr, void* data, Size size) {
    if (!validate_bank_access(bank_id, addr, size)) {
        throw std::out_of_range("Invalid bank read access");
    }

    std::lock_guard<std::mutex> lock(bank_mutex);
    auto& bank = *bank_states[bank_id];

    bank.is_reading = true;

    std::memcpy(data, &bank.data[addr], size);

    bank.read_accesses++;
    bank.is_reading = false;
}

void StorageScheduler::direct_write(size_t bank_id, Address addr, const void* data, Size size) {
    if (!validate_bank_access(bank_id, addr, size)) {
        throw std::out_of_range("Invalid bank write access");
    }

    std::lock_guard<std::mutex> lock(bank_mutex);
    auto& bank = *bank_states[bank_id];

    bank.is_writing = true;

    std::memcpy(&bank.data[addr], data, size);

    // Update occupancy tracking
    Size end_addr = addr + size;
    if (end_addr > bank.current_occupancy) {
        bank.current_occupancy = end_addr;
    }

    bank.write_accesses++;
    bank.is_writing = false;
}

bool StorageScheduler::is_ready(size_t bank_id) const {
    if (bank_id >= num_banks) return false;

    std::lock_guard<std::mutex> lock(bank_mutex);
    const auto& bank = *bank_states[bank_id];
    return !bank.is_reading && !bank.is_writing;
}

void StorageScheduler::schedule_operation(const StorageCommand& cmd) {
    std::lock_guard<std::mutex> lock(command_mutex);
    command_queue.push(cmd);
}

bool StorageScheduler::execute_pending_operations() {
    std::lock_guard<std::mutex> lock(command_mutex);

    if (command_queue.empty()) {
        return false;
    }

    // Try to execute ready commands
    bool executed_any = false;
    std::queue<StorageCommand> deferred_commands;

    while (!command_queue.empty()) {
        StorageCommand cmd = command_queue.front();
        command_queue.pop();

        if (can_execute_command(cmd)) {
            // Execute based on phase
            switch (cmd.operation) {
                case StorageOperation::FETCH_UPSTREAM:
                    execute_fetch_upstream_command(cmd);
                    break;
                case StorageOperation::FETCH_DOWNSTREAM:
                    execute_fetch_downstream_command(cmd);
                    break;
                case StorageOperation::WRITEBACK_UPSTREAM:
                    execute_writeback_upstream_command(cmd);
                    break;
                case StorageOperation::WRITEBACK_DOWNSTREAM:
                    execute_writeback_downstream_command(cmd);
                    break;
                case StorageOperation::YIELD:
                    execute_yield_command(cmd);
                    break;
                case StorageOperation::BARRIER:
                    execute_barrier_command(cmd);
                    break;
            }
            executed_any = true;
            complete_command(cmd);
        } else {
            // Defer command for later execution
            deferred_commands.push(cmd);
        }
    }

    // Put deferred commands back
    command_queue = std::move(deferred_commands);

    return executed_any;
}

bool StorageScheduler::can_execute_command(const StorageCommand& cmd) const {
    // Check if bank is available for this phase
    if (!is_bank_available(cmd.bank_id, cmd.operation)) {
        return false;
    }

    // Check dependencies
    for (size_t dep_id : cmd.dependencies) {
        if (active_commands.find(dep_id) != active_commands.end()) {
            return false; // Dependency still active
        }
    }

    return true;
}

void StorageScheduler::execute_fetch_upstream_command(const StorageCommand& cmd) {
    // Mark bank as transitioning to fetch upstream operation
    transition_bank_operation(cmd.bank_id, StorageOperation::FETCH_UPSTREAM, cmd.sequence_id);

    // Simulate fetch completion immediately for now
    // In a real implementation, this would be asynchronous
    // and complete when the data movement is done

    // For simulation purposes, mark as completed immediately
    // Don't add to active_commands since we're completing immediately
}

void StorageScheduler::execute_fetch_downstream_command(const StorageCommand& cmd) {
    // Mark bank as transitioning to fetch downstream operation
    transition_bank_operation(cmd.bank_id, StorageOperation::FETCH_DOWNSTREAM, cmd.sequence_id);

    // Simulate fetch completion immediately for now
    // In a real implementation, this would be asynchronous
    // and complete when the data movement is done

    // For simulation purposes, mark as completed immediately
    // Don't add to active_commands since we're completing immediately
}

void StorageScheduler::execute_writeback_downstream_command(const StorageCommand& cmd) {
    transition_bank_operation(cmd.bank_id, StorageOperation::WRITEBACK_DOWNSTREAM, cmd.sequence_id);

    // Simulate writeback completion immediately for testing
    // In real implementation, this would involve data streaming
    // Don't add to active_commands since we're completing immediately
}

void StorageScheduler::execute_yield_command(const StorageCommand& cmd) {
    transition_bank_operation(cmd.bank_id, StorageOperation::YIELD, cmd.sequence_id);

    // Yield allows external access to the bank data
    // In real implementation, this would allow external components
    // to access the bank data while the scheduler waits
    // Don't add to active_commands since we're completing immediately
}

void StorageScheduler::execute_writeback_upstream_command(const StorageCommand& cmd) {
    transition_bank_operation(cmd.bank_id, StorageOperation::WRITEBACK_UPSTREAM, cmd.sequence_id);

    // Simulate writeback completion immediately for testing
    // In real implementation, this would involve data streaming
    // Don't add to active_commands since we're completing immediately
}

void StorageScheduler::execute_barrier_command(const StorageCommand& cmd) {
    // Barrier commands ensure all previous operations complete
    // Reset bank to barrier state
    transition_bank_operation(cmd.bank_id, StorageOperation::BARRIER, cmd.sequence_id);

    // Barrier completes immediately - it's just a coordination point
}

void StorageScheduler::complete_command(const StorageCommand& cmd) {
    // Remove from active commands
    active_commands.erase(cmd.sequence_id);

    // Update dependencies
    update_dependencies(cmd.sequence_id);

    // Call completion callback if provided
    if (cmd.completion_callback) {
        cmd.completion_callback(cmd);
    }
}

void StorageScheduler::update_dependencies(size_t completed_sequence_id) {
    // Remove completed command from all dependency lists
    for (auto& [seq_id, dependents] : dependency_graph) {
        dependents.erase(
            std::remove(dependents.begin(), dependents.end(), completed_sequence_id),
            dependents.end());
    }
}

bool StorageScheduler::is_bank_available(size_t bank_id, StorageOperation operation) const {
    if (bank_id >= num_banks) return false;

    const auto& bank = *bank_states[bank_id];

    // More permissive availability check to prevent infinite loops
    switch (operation) {
        case StorageOperation::FETCH_UPSTREAM:
        case StorageOperation::FETCH_DOWNSTREAM:
            return !bank.is_writing;
        case StorageOperation::WRITEBACK_UPSTREAM:
        case StorageOperation::WRITEBACK_DOWNSTREAM:
            return !bank.is_reading;
        case StorageOperation::YIELD:
            // Allow yield on any bank that's not actively reading/writing
            return !bank.is_reading && !bank.is_writing;
        case StorageOperation::BARRIER:
            // Barrier can always proceed - it's a coordination phase
            return true;
    }
    return false;
}

void StorageScheduler::transition_bank_operation(size_t bank_id, StorageOperation new_operation, size_t sequence_id) {
    if (bank_id >= num_banks) return;

    std::lock_guard<std::mutex> lock(bank_mutex);
    auto& bank = *bank_states[bank_id];
    bank.current_operation = new_operation;
    bank.active_sequence_id = sequence_id;
}

bool StorageScheduler::validate_bank_access(size_t bank_id, Address addr, Size size) const {
    if (bank_id >= num_banks) return false;

    // Check if the access would exceed bank capacity
    return (addr + size) <= bank_states[bank_id]->capacity;
}

Address StorageScheduler::map_to_bank_address(size_t bank_id, Address global_addr) const {
    // Simple mapping - in practice this could be more sophisticated
    return global_addr % bank_states[bank_id]->capacity;
}

void StorageScheduler::schedule_double_buffer(size_t bank_a, size_t bank_b,
                                     Address src_addr, Size transfer_size) {
    StorageCommand prefetch_a{
        .operation = StorageOperation::FETCH_UPSTREAM,
        .bank_id = bank_a,
        .source_addr = src_addr,
        .dest_addr = 0,
        .transfer_size = transfer_size,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };

    StorageCommand prefetch_b{
        .operation = StorageOperation::FETCH_UPSTREAM,
        .bank_id = bank_b,
        .source_addr = src_addr + transfer_size,
        .dest_addr = 0,
        .transfer_size = transfer_size,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };

    schedule_operation(prefetch_a);
    schedule_operation(prefetch_b);
}

void StorageScheduler::schedule_pipeline_stage(size_t input_bank, size_t output_bank,
                                       const std::function<void()>& compute_func) {
    StorageCommand compute_cmd{
        .operation = StorageOperation::YIELD,
        .bank_id = input_bank,
        .source_addr = 0,
        .dest_addr = 0,
        .transfer_size = 0,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = [compute_func](const StorageCommand&) { compute_func(); }
    };

    StorageCommand writeback_cmd{
        .operation = StorageOperation::WRITEBACK_UPSTREAM,
        .bank_id = output_bank,
        .source_addr = 0,
        .dest_addr = 0,
        .transfer_size = 0,
        .sequence_id = next_sequence_id++,
        .dependencies = {compute_cmd.sequence_id},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };

    schedule_operation(compute_cmd);
    schedule_operation(writeback_cmd);
}

size_t StorageScheduler::get_pending_operations() const {
    std::lock_guard<std::mutex> lock(command_mutex);
    return command_queue.size();
}

bool StorageScheduler::is_busy() const {
    std::lock_guard<std::mutex> lock(command_mutex);
    return !command_queue.empty() || !active_commands.empty();
}

bool StorageScheduler::is_bank_busy(size_t bank_id) const {
    if (bank_id >= num_banks) return false;

    std::lock_guard<std::mutex> lock(bank_mutex);
    const auto& bank = *bank_states[bank_id];
    return bank.is_reading || bank.is_writing || bank.current_operation != StorageOperation::BARRIER;
}

StorageScheduler::StorageOperation StorageScheduler::get_bank_operation(size_t bank_id) const {
    if (bank_id >= num_banks) return StorageOperation::BARRIER;

    std::lock_guard<std::mutex> lock(bank_mutex);
    return bank_states[bank_id]->current_operation;
}

StorageScheduler::PerformanceMetrics StorageScheduler::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(bank_mutex);

    PerformanceMetrics metrics{};
    size_t busy_banks = 0;

    for (const auto& bank : bank_states) {
        metrics.total_read_accesses += bank->read_accesses;
        metrics.total_write_accesses += bank->write_accesses;
        metrics.total_cache_hits += bank->cache_hits;
        metrics.total_cache_misses += bank->cache_misses;

        if (bank->current_operation != StorageOperation::BARRIER) {
            busy_banks++;
        }
    }

    metrics.average_bank_utilization = static_cast<double>(busy_banks) / num_banks;
    metrics.completed_storage_operations = 0; // Would track this in real implementation

    return metrics;
}

Size StorageScheduler::get_bank_capacity(size_t bank_id) const {
    if (bank_id >= num_banks) return 0;
    return bank_states[bank_id]->capacity;
}

Size StorageScheduler::get_bank_occupancy(size_t bank_id) const {
    if (bank_id >= num_banks) return 0;
    return bank_states[bank_id]->current_occupancy;
}

void StorageScheduler::reset() {
    std::lock_guard<std::mutex> cmd_lock(command_mutex);
    std::lock_guard<std::mutex> bank_lock(bank_mutex);

    // Clear command queues
    command_queue = std::queue<StorageCommand>();
    active_commands.clear();
    dependency_graph.clear();

    // Reset bank states
    for (auto& bank : bank_states) {
        bank->current_occupancy = 0;
        bank->is_reading = false;
        bank->is_writing = false;
        bank->current_operation = StorageOperation::BARRIER;
        bank->active_sequence_id = 0;
        bank->read_accesses = 0;
        bank->write_accesses = 0;
        bank->cache_hits = 0;
        bank->cache_misses = 0;
        std::fill(bank->data.begin(), bank->data.end(), uint8_t(0));
    }
}

void StorageScheduler::flush_all_banks() {
    std::lock_guard<std::mutex> lock(bank_mutex);
    for (auto& bank : bank_states) {
        bank->current_occupancy = 0;
        std::fill(bank->data.begin(), bank->data.end(), uint8_t(0));
    }
}

void StorageScheduler::abort_pending_operations() {
    std::lock_guard<std::mutex> lock(command_mutex);
    command_queue = std::queue<StorageCommand>();
    active_commands.clear();
    dependency_graph.clear();
}

// StorageWorkflowBuilder Implementation
StorageWorkflowBuilder& StorageWorkflowBuilder::fetch_upstream(size_t bank_id, Address src_addr,
                                                 Address dest_addr, Size size) {
    StorageScheduler::StorageCommand cmd{
        .operation = StorageScheduler::StorageOperation::FETCH_UPSTREAM,
        .bank_id = bank_id,
        .source_addr = src_addr,
        .dest_addr = dest_addr,
        .transfer_size = size,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };
    commands.push_back(cmd);
    return *this;
}

StorageWorkflowBuilder& StorageWorkflowBuilder::yield(size_t bank_id,
                                                const std::function<void()>& yield_func) {
    StorageScheduler::StorageCommand cmd{
        .operation = StorageScheduler::StorageOperation::YIELD,
        .bank_id = bank_id,
        .source_addr = 0,
        .dest_addr = 0,
        .transfer_size = 0,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = [yield_func](const StorageScheduler::StorageCommand&) { yield_func(); }
    };
    commands.push_back(cmd);
    return *this;
}

StorageWorkflowBuilder& StorageWorkflowBuilder::writeback_upstream(size_t bank_id, Address src_addr,
                                                  Address dest_addr, Size size) {
    StorageScheduler::StorageCommand cmd{
        .operation = StorageScheduler::StorageOperation::WRITEBACK_UPSTREAM,
        .bank_id = bank_id,
        .source_addr = src_addr,
        .dest_addr = dest_addr,
        .transfer_size = size,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };
    commands.push_back(cmd);
    return *this;
}

StorageWorkflowBuilder& StorageWorkflowBuilder::fetch_downstream(size_t bank_id, Address src_addr,
                                                     Address dest_addr, Size size) {
    StorageScheduler::StorageCommand cmd{
        .operation = StorageScheduler::StorageOperation::FETCH_DOWNSTREAM,
        .bank_id = bank_id,
        .source_addr = src_addr,
        .dest_addr = dest_addr,
        .transfer_size = size,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };
    commands.push_back(cmd);
    return *this;
}

StorageWorkflowBuilder& StorageWorkflowBuilder::writeback_downstream(size_t bank_id, Address src_addr,
                                                      Address dest_addr, Size size) {
    StorageScheduler::StorageCommand cmd{
        .operation = StorageScheduler::StorageOperation::WRITEBACK_DOWNSTREAM,
        .bank_id = bank_id,
        .source_addr = src_addr,
        .dest_addr = dest_addr,
        .transfer_size = size,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };
    commands.push_back(cmd);
    return *this;
}

StorageWorkflowBuilder& StorageWorkflowBuilder::barrier() {
    StorageScheduler::StorageCommand cmd{
        .operation = StorageScheduler::StorageOperation::BARRIER,
        .bank_id = 0, // Barrier applies to all banks
        .source_addr = 0,
        .dest_addr = 0,
        .transfer_size = 0,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };
    commands.push_back(cmd);
    return *this;
}

StorageWorkflowBuilder& StorageWorkflowBuilder::depend_on(size_t dependency_sequence_id) {
    if (!commands.empty()) {
        commands.back().dependencies.push_back(dependency_sequence_id);
    }
    return *this;
}

std::vector<StorageScheduler::StorageCommand> StorageWorkflowBuilder::build() {
    return commands;
}

void StorageWorkflowBuilder::execute_on(StorageScheduler& scheduler) {
    for (const auto& cmd : commands) {
        scheduler.schedule_operation(cmd);
    }
}

} // namespace sw::kpu