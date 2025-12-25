#include <sw/trace/trace_entry.hpp>

namespace sw::trace {

const char* to_string(ComponentType type) {
    switch (type) {
        case ComponentType::HOST_MEMORY: return "HOST_MEMORY";
        case ComponentType::HOST_CPU: return "HOST_CPU";
        case ComponentType::PCIE_BUS: return "PCIE_BUS";
        case ComponentType::DMA_ENGINE: return "DMA_ENGINE";
        case ComponentType::BLOCK_MOVER: return "BLOCK_MOVER";
        case ComponentType::STREAMER: return "STREAMER";
        case ComponentType::KPU_MEMORY: return "KPU_MEMORY";
        case ComponentType::L3_TILE: return "L3_TILE";
        case ComponentType::L2_BANK: return "L2_BANK";
        case ComponentType::L1: return "L1";
        case ComponentType::PAGE_BUFFER: return "PAGE_BUFFER";
        case ComponentType::COMPUTE_FABRIC: return "COMPUTE_FABRIC";
        case ComponentType::SYSTOLIC_ARRAY: return "SYSTOLIC_ARRAY";
        case ComponentType::STORAGE_SCHEDULER: return "STORAGE_SCHEDULER";
        case ComponentType::MEMORY_ORCHESTRATOR: return "MEMORY_ORCHESTRATOR";
        case ComponentType::UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

const char* to_string(TransactionType type) {
    switch (type) {
        case TransactionType::READ: return "READ";
        case TransactionType::WRITE: return "WRITE";
        case TransactionType::TRANSFER: return "TRANSFER";
        case TransactionType::COPY: return "COPY";
        case TransactionType::COMPUTE: return "COMPUTE";
        case TransactionType::MATMUL: return "MATMUL";
        case TransactionType::DOT_PRODUCT: return "DOT_PRODUCT";
        case TransactionType::CONFIGURE: return "CONFIGURE";
        case TransactionType::SYNC: return "SYNC";
        case TransactionType::FENCE: return "FENCE";
        case TransactionType::ALLOCATE: return "ALLOCATE";
        case TransactionType::DEALLOCATE: return "DEALLOCATE";
        case TransactionType::UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

const char* to_string(TransactionStatus status) {
    switch (status) {
        case TransactionStatus::ISSUED: return "ISSUED";
        case TransactionStatus::IN_PROGRESS: return "IN_PROGRESS";
        case TransactionStatus::COMPLETED: return "COMPLETED";
        case TransactionStatus::FAILED: return "FAILED";
        case TransactionStatus::CANCELLED: return "CANCELLED";
        default: return "INVALID";
    }
}

} // namespace sw::trace
