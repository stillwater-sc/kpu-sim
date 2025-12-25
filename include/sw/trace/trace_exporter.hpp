#pragma once

#include <sw/trace/trace_logger.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

namespace sw::trace {

// Helper to convert payload to string for export
inline std::string payload_to_string(const PayloadData& payload) {
    std::ostringstream oss;

    if (std::holds_alternative<DMAPayload>(payload)) {
        const auto& dma = std::get<DMAPayload>(payload);
        oss << "DMA[src:" << to_string(dma.source.type) << ":" << dma.source.bank_id
            << "@0x" << std::hex << dma.source.address << std::dec
            << " dst:" << to_string(dma.destination.type) << ":" << dma.destination.bank_id
            << "@0x" << std::hex << dma.destination.address << std::dec
            << " size:" << dma.bytes_transferred
            << " bw:" << std::fixed << std::setprecision(2) << dma.bandwidth_gb_s << "GB/s]";
    }
    else if (std::holds_alternative<ComputePayload>(payload)) {
        const auto& comp = std::get<ComputePayload>(payload);
        oss << "Compute[" << comp.kernel_name << " ops:" << comp.num_operations;
        if (comp.m > 0 && comp.n > 0 && comp.k > 0) {
            oss << " dims:" << comp.m << "x" << comp.n << "x" << comp.k;
        }
        oss << "]";
    }
    else if (std::holds_alternative<ControlPayload>(payload)) {
        const auto& ctrl = std::get<ControlPayload>(payload);
        oss << "Control[" << ctrl.command << " param:" << ctrl.parameter << "]";
    }
    else if (std::holds_alternative<MemoryPayload>(payload)) {
        const auto& mem = std::get<MemoryPayload>(payload);
        oss << "Memory[" << to_string(mem.location.type) << ":" << mem.location.bank_id
            << "@0x" << std::hex << mem.location.address << std::dec
            << " size:" << mem.location.size
            << " hit:" << (mem.is_hit ? "Y" : "N")
            << " lat:" << mem.latency_cycles << "]";
    }
    else {
        oss << "NoPayload";
    }

    return oss.str();
}

// CSV Export
class KPU_API CSVExporter {
public:
    static bool export_traces(const std::string& filename, const std::vector<TraceEntry>& traces) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        // Write header
        file << "TransactionID,ComponentType,ComponentID,TransactionType,Status,"
             << "CycleIssue,CycleComplete,DurationCycles,"
             << "TimeIssueNs,TimeCompleteNs,DurationNs,"
             << "Payload,Description\n";

        // Write entries
        for (const auto& entry : traces) {
            file << entry.transaction_id << ","
                 << to_string(entry.component_type) << ","
                 << entry.component_id << ","
                 << to_string(entry.transaction_type) << ","
                 << to_string(entry.status) << ","
                 << entry.cycle_issue << ","
                 << entry.cycle_complete << ","
                 << entry.get_duration_cycles() << ","
                 << std::fixed << std::setprecision(3) << entry.get_issue_time_ns() << ","
                 << entry.get_complete_time_ns() << ","
                 << entry.get_duration_ns() << ","
                 << "\"" << payload_to_string(entry.payload) << "\","
                 << "\"" << entry.description << "\"\n";
        }

        file.close();
        return true;
    }
};

// JSON Export
class KPU_API JSONExporter {
public:
    static bool export_traces(const std::string& filename, const std::vector<TraceEntry>& traces) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "{\n";
        file << "  \"traces\": [\n";

        for (size_t i = 0; i < traces.size(); ++i) {
            const auto& entry = traces[i];

            file << "    {\n";
            file << "      \"transaction_id\": " << entry.transaction_id << ",\n";
            file << "      \"component_type\": \"" << to_string(entry.component_type) << "\",\n";
            file << "      \"component_id\": " << entry.component_id << ",\n";
            file << "      \"transaction_type\": \"" << to_string(entry.transaction_type) << "\",\n";
            file << "      \"status\": \"" << to_string(entry.status) << "\",\n";
            file << "      \"cycle_issue\": " << entry.cycle_issue << ",\n";
            file << "      \"cycle_complete\": " << entry.cycle_complete << ",\n";
            file << "      \"duration_cycles\": " << entry.get_duration_cycles() << ",\n";

            if (entry.clock_freq_ghz.has_value()) {
                file << "      \"clock_freq_ghz\": " << std::fixed << std::setprecision(3)
                     << entry.clock_freq_ghz.value() << ",\n";
                file << "      \"time_issue_ns\": " << entry.get_issue_time_ns() << ",\n";
                file << "      \"time_complete_ns\": " << entry.get_complete_time_ns() << ",\n";
                file << "      \"duration_ns\": " << entry.get_duration_ns() << ",\n";
            }

            file << "      \"payload\": \"" << payload_to_string(entry.payload) << "\",\n";
            file << "      \"description\": \"" << entry.description << "\"\n";
            file << "    }" << (i < traces.size() - 1 ? "," : "") << "\n";
        }

        file << "  ]\n";
        file << "}\n";

        file.close();
        return true;
    }
};

// Chrome Trace Event Format Export (for chrome://tracing visualization)
class KPU_API ChromeTraceExporter {
private:
    // Map ComponentType to display order (lower values appear first in viewer)
    // This reflects the physical pipeline order from host to compute
    static uint32_t get_display_pid(ComponentType type) {
        switch (type) {
            case ComponentType::HOST_MEMORY:        return 1;  // Host DDR
            case ComponentType::HOST_CPU:           return 2;  // Host CPU
            case ComponentType::PCIE_BUS:           return 3;  // PCIe interconnect
            case ComponentType::DMA_ENGINE:         return 4;  // PCIe bus master
            case ComponentType::KPU_MEMORY:         return 5;  // KPU main memory (GDDR6 banks)
            case ComponentType::L3_TILE:            return 6;  // L3 cache tiles
            case ComponentType::BLOCK_MOVER:        return 7;  // L3->L2 movement
            case ComponentType::L2_BANK:            return 8;  // L2 cache
            case ComponentType::STREAMER:           return 9;  // L2<->L1 movement
            case ComponentType::L1:                 return 10; // L1 streaming buffers
            case ComponentType::COMPUTE_FABRIC:     return 11; // Compute orchestrator
            case ComponentType::SYSTOLIC_ARRAY:     return 12; // Compute engine
            case ComponentType::PAGE_BUFFER:        return 15; // Memory controller page buffers
            case ComponentType::STORAGE_SCHEDULER:  return 20; // System services
            case ComponentType::MEMORY_ORCHESTRATOR: return 21;
            default: return 99;  // Unknown/other components
        }
    }

public:
    static bool export_traces(const std::string& filename, const std::vector<TraceEntry>& traces,
                             double default_freq_ghz = 1.0) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "[\n";

        // First, collect unique process and thread IDs to emit metadata
        std::map<uint32_t, std::string> process_names;
        std::map<std::pair<uint32_t, uint32_t>, std::string> thread_names;

        for (const auto& entry : traces) {
            uint32_t pid = get_display_pid(entry.component_type);
            uint32_t tid = entry.component_id;

            // Prefix with display order to force correct alphabetical sorting in Chrome viewer
            std::ostringstream process_name_stream;
            process_name_stream << std::setfill('0') << std::setw(2) << pid << "-" << to_string(entry.component_type);
            std::string process_name = process_name_stream.str();

            process_names[pid] = process_name;

            // Create thread name: "ComponentType #ID"
            std::ostringstream thread_name;
            thread_name << to_string(entry.component_type) << " #" << tid;
            thread_names[{pid, tid}] = thread_name.str();
        }

        // Emit process name metadata events
        bool first_event = true;
        for (const auto& [pid, name] : process_names) {
            if (!first_event) file << ",\n";
            file << "  {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": " << pid
                 << ", \"args\": {\"name\": \"" << name << "\"}}";
            first_event = false;
        }

        // Emit thread name metadata events
        for (const auto& [pid_tid, name] : thread_names) {
            file << ",\n";
            file << "  {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": " << pid_tid.first
                 << ", \"tid\": " << pid_tid.second
                 << ", \"args\": {\"name\": \"" << name << "\"}}";
        }

        // Now emit the actual trace events
        for (size_t i = 0; i < traces.size(); ++i) {
            const auto& entry = traces[i];
            file << ",\n";

            // Get frequency (use entry's freq if available, otherwise default)
            double freq = entry.clock_freq_ghz.value_or(default_freq_ghz);

            // Convert cycles to microseconds (chrome trace expects microseconds)
            double ts_us = static_cast<double>(entry.cycle_issue) / freq * 1000.0;
            double dur_us = static_cast<double>(entry.get_duration_cycles()) / freq * 1000.0;

            // Create process and thread names based on component
            std::string process_name = to_string(entry.component_type);
            uint32_t pid = get_display_pid(entry.component_type);
            uint32_t tid = entry.component_id;

            // Complete event (has duration)
            if (entry.cycle_complete > 0) {
                file << "  {\"name\": \"" << to_string(entry.transaction_type) << "\",";
                file << " \"cat\": \"" << process_name << "\",";
                file << " \"ph\": \"X\",";  // Complete event
                file << " \"ts\": " << std::fixed << std::setprecision(3) << ts_us << ",";
                file << " \"dur\": " << dur_us << ",";
                file << " \"pid\": " << pid << ",";
                file << " \"tid\": " << tid << ",";
                file << " \"args\": {";
                file << "\"txn_id\": " << entry.transaction_id << ",";
                file << "\"status\": \"" << to_string(entry.status) << "\",";
                file << "\"cycle_issue\": " << entry.cycle_issue << ",";
                file << "\"cycle_complete\": " << entry.cycle_complete << ",";
                file << "\"payload\": \"" << payload_to_string(entry.payload) << "\"";
                if (!entry.description.empty()) {
                    file << ",\"desc\": \"" << entry.description << "\"";
                }
                file << "}}";
            }
            // Instant event (no duration yet)
            else {
                file << "  {\"name\": \"" << to_string(entry.transaction_type) << "\",";
                file << " \"cat\": \"" << process_name << "\",";
                file << " \"ph\": \"i\",";  // Instant event
                file << " \"ts\": " << std::fixed << std::setprecision(3) << ts_us << ",";
                file << " \"pid\": " << pid << ",";
                file << " \"tid\": " << tid << ",";
                file << " \"s\": \"t\",";  // Thread scope
                file << " \"args\": {";
                file << "\"txn_id\": " << entry.transaction_id << ",";
                file << "\"status\": \"" << to_string(entry.status) << "\",";
                file << "\"cycle\": " << entry.cycle_issue;
                if (!entry.description.empty()) {
                    file << ",\"desc\": \"" << entry.description << "\"";
                }
                file << "}}";
            }
        }

        file << "\n]\n";

        file.close();
        return true;
    }
};

// Convenience function to export from logger
inline bool export_logger_traces(const std::string& filename,
                                 const std::string& format = "csv",
                                 TraceLogger& logger = TraceLogger::instance()) {
    auto traces = logger.get_all_traces();

    if (format == "csv") {
        return CSVExporter::export_traces(filename, traces);
    }
    else if (format == "json") {
        return JSONExporter::export_traces(filename, traces);
    }
    else if (format == "chrome" || format == "trace") {
        return ChromeTraceExporter::export_traces(filename, traces);
    }

    return false;
}

} // namespace sw::trace

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
