/**
 * @file data_movement_isa.hpp
 * @brief Data Movement ISA for Domain Flow Architecture
 *
 * In Domain Flow Architecture, the program IS the data movement schedule.
 * The compute fabric reacts to arriving data tokens - it doesn't need
 * explicit instructions. The intelligence is in orchestrating data movement
 * to create the optimal system-level schedule derived from SURE analysis.
 *
 * This ISA defines operations to configure and control:
 * - DMA Engines (External Memory ↔ L3)
 * - Block Movers (L3 ↔ L2 with transformations)
 * - Streamers (L2 ↔ L1 with systolic array feeding)
 *
 * Output-Stationary Schedule Pattern:
 * For C[M,N] = A[M,K] × B[K,N]:
 *   - C tiles stay in PE accumulators (no writeback until complete)
 *   - Loop order: for ti, for tj, for tk (output tiles outer, reduction inner)
 *   - A tiles reused across N dimension
 *   - B tiles reused across M dimension
 *
 * Note: This ISA is memory-technology agnostic. External memory may be
 * implemented as DDR4, GDDR6, HBM2/3/4, or any future memory technology.
 */

#pragma once

#include <sw/concepts.hpp>
#include <sw/kpu/components/sfu.hpp>  // For ActivationType
#include <cstdint>
#include <vector>
#include <string>
#include <variant>
#include <optional>
#include <set>

namespace sw::kpu::isa {

// ============================================================================
// Data Movement ISA Opcodes
// ============================================================================

/**
 * @brief Data Movement Operation Codes
 *
 * These opcodes configure the data movement datapath to execute
 * the system-level schedule. Each opcode maps to specific hardware
 * configuration of DMA, BlockMover, or Streamer.
 */
enum class DMOpcode : uint8_t {
    // DMA Operations (External Memory ↔ L3)
    DMA_LOAD_TILE,          // Load a tile from external memory to L3
    DMA_STORE_TILE,         // Store a tile from L3 to external memory
    DMA_PREFETCH_TILE,      // Prefetch a tile (non-blocking)

    // Block Mover Operations (L3 ↔ L2)
    BM_MOVE_TILE,           // Move tile L3 → L2 (identity)
    BM_TRANSPOSE_TILE,      // Move tile L3 → L2 with transpose
    BM_WRITEBACK_TILE,      // Move tile L2 → L3
    BM_RESHAPE_TILE,        // Move with block reshape

    // Streamer Operations (L2 ↔ L1)
    STR_FEED_ROWS,          // Stream rows to systolic array (A matrix)
    STR_FEED_COLS,          // Stream columns to systolic array (B matrix)
    STR_DRAIN_OUTPUT,       // Drain output from systolic array (C matrix)
    STR_BROADCAST_ROW,      // Broadcast row to all PE columns
    STR_BROADCAST_COL,      // Broadcast column to all PE rows

    // Synchronization Operations
    BARRIER,                // Wait for all pending operations
    WAIT_DMA,               // Wait for specific DMA completion
    WAIT_BM,                // Wait for specific BlockMover completion
    WAIT_STR,               // Wait for specific Streamer completion
    SIGNAL,                 // Signal completion token

    // Configuration Operations
    SET_TILE_SIZE,          // Configure tile dimensions
    SET_BUFFER,             // Configure double-buffer selection
    SET_STRIDE,             // Configure address stride patterns

    // Loop Control (for hardware loop support)
    LOOP_BEGIN,             // Start hardware loop
    LOOP_END,               // End hardware loop

    // NOP and special
    NOP,                    // No operation
    HALT                    // End of program
};

// ============================================================================
// Operand Types
// ============================================================================

/**
 * @brief Memory level specification (technology-agnostic)
 *
 * The memory hierarchy is defined by logical levels, not physical
 * implementation. External memory could be DDR4, GDDR6, HBM, etc.
 */
enum class MemLevel : uint8_t {
    EXTERNAL = 0,   // External memory (DDR4, GDDR6, HBM, etc.)
    L3 = 1,         // L3 tile cache
    L2 = 2,         // L2 bank cache
    L1 = 3,         // L1 streaming buffer
    PE_REG = 4      // PE accumulator registers
};

/**
 * @brief Matrix identifier
 */
enum class MatrixID : uint8_t {
    A = 0,          // Input matrix A
    B = 1,          // Input/Weight matrix B
    C = 2           // Output matrix C
};

/**
 * @brief Tile coordinate in the tiled iteration space
 */
struct TileCoord {
    uint16_t ti;    // M-dimension tile index
    uint16_t tj;    // N-dimension tile index
    uint16_t tk;    // K-dimension tile index (reduction)

    bool operator==(const TileCoord& other) const {
        return ti == other.ti && tj == other.tj && tk == other.tk;
    }
};

/**
 * @brief Buffer slot for double-buffering
 */
enum class BufferSlot : uint8_t {
    BUF_0 = 0,
    BUF_1 = 1,
    AUTO = 2        // Automatically alternate
};

/**
 * @brief Transform type for BlockMover
 */
enum class Transform : uint8_t {
    IDENTITY = 0,
    TRANSPOSE = 1,
    RESHAPE = 2,
    SHUFFLE = 3
};

// ============================================================================
// Instruction Operands
// ============================================================================

/**
 * @brief DMA operation operands
 */
struct DMAOperands {
    MatrixID matrix;            // Which matrix (A, B, or C)
    TileCoord tile;             // Which tile
    Address ext_mem_addr;       // Address in external memory
    uint8_t l3_tile_id;         // Which L3 tile
    Address l3_offset;          // Offset within L3 tile
    Size size_bytes;            // Transfer size
    BufferSlot buffer;          // Which buffer slot
};

/**
 * @brief Block Mover operation operands
 */
struct BlockMoverOperands {
    MatrixID matrix;            // Which matrix
    TileCoord tile;             // Which tile
    uint8_t src_l3_tile_id;     // Source L3 tile
    Address src_offset;         // Source offset
    uint8_t dst_l2_bank_id;     // Destination L2 bank
    Address dst_offset;         // Destination offset
    Size height;                // Block height (rows)
    Size width;                 // Block width (cols)
    Size element_size;          // Element size in bytes
    Transform transform;        // Transformation to apply
    BufferSlot buffer;          // Buffer slot
};

/**
 * @brief Streamer operation operands
 */
struct StreamerOperands {
    MatrixID matrix;            // Which matrix
    TileCoord tile;             // Which tile
    uint8_t l2_bank_id;         // L2 bank
    uint8_t l1_buffer_id;       // L1 buffer
    Address l2_addr;            // L2 address
    Address l1_addr;            // L1 address
    Size height;                // Matrix height
    Size width;                 // Matrix width
    Size fabric_size;           // Systolic array size
    BufferSlot buffer;          // Buffer slot

    // Vector Engine configuration (for STR_DRAIN_OUTPUT)
    // VE processes data inline during L1→L2 transfer
    bool ve_enabled = false;                        ///< Route through Vector Engine
    ActivationType ve_activation = ActivationType::NONE;  ///< Activation function
    bool ve_bias_enabled = false;                   ///< Apply bias addition
    Address ve_bias_addr = 0;                       ///< Bias vector address in L1
};

/**
 * @brief Synchronization operands
 */
struct SyncOperands {
    uint32_t wait_mask;         // Bitmask of operations to wait for
    uint32_t signal_id;         // Signal identifier
};

/**
 * @brief Loop control operands
 */
struct LoopOperands {
    uint16_t loop_count;        // Number of iterations
    uint8_t loop_id;            // Loop identifier (for nesting)
    uint16_t loop_stride;       // Tile index stride per iteration
};

/**
 * @brief Configuration operands
 */
struct ConfigOperands {
    Size Ti, Tj, Tk;            // Tile dimensions
    Size L1_Ki;                 // L1 streaming chunk
    uint8_t buffer_id;          // Buffer to configure
    Size stride_m, stride_n, stride_k;  // Address strides
};

// ============================================================================
// Data Movement Instruction
// ============================================================================

/**
 * @brief A single data movement instruction
 *
 * Instructions are the units of the Data Movement ISA. They encode
 * operations that configure and trigger the data movement hardware.
 */
struct DMInstruction {
    DMOpcode opcode;

    // Operands (variant for type safety)
    std::variant<
        std::monostate,         // For NOP, HALT, BARRIER
        DMAOperands,
        BlockMoverOperands,
        StreamerOperands,
        SyncOperands,
        LoopOperands,
        ConfigOperands
    > operands;

    // Timing hints (from SURE analysis)
    uint32_t earliest_cycle;    // Earliest valid issue cycle
    uint32_t deadline_cycle;    // Latest valid issue cycle (for pipelining)

    // Dependency tracking
    uint32_t instruction_id;    // Unique instruction ID
    std::vector<uint32_t> dependencies;  // IDs of instructions that must complete first

    // Debug info
    std::string label;          // Human-readable label (e.g., "Load A[0,0]")

    // Default constructor
    DMInstruction() : opcode(DMOpcode::NOP), earliest_cycle(0),
                     deadline_cycle(UINT32_MAX), instruction_id(0) {}

    // Convenience constructors
    static DMInstruction dma_load(MatrixID mat, TileCoord tile, Address ext_mem_addr,
                                  uint8_t l3_tile, Address l3_offset, Size bytes);

    static DMInstruction bm_move(MatrixID mat, TileCoord tile,
                                 uint8_t src_l3, Address src_off,
                                 uint8_t dst_l2, Address dst_off,
                                 Size height, Size width, Size elem_size,
                                 Transform xform = Transform::IDENTITY);

    static DMInstruction str_feed_rows(MatrixID mat, TileCoord tile,
                                       uint8_t l2_bank, uint8_t l1_buf,
                                       Address l2_addr, Address l1_addr,
                                       Size height, Size width, Size fabric_size);

    static DMInstruction str_feed_cols(MatrixID mat, TileCoord tile,
                                       uint8_t l2_bank, uint8_t l1_buf,
                                       Address l2_addr, Address l1_addr,
                                       Size height, Size width, Size fabric_size);

    static DMInstruction str_drain(TileCoord tile,
                                   uint8_t l2_bank, uint8_t l1_buf,
                                   Address l2_addr, Address l1_addr,
                                   Size height, Size width, Size fabric_size,
                                   bool ve_enabled = false,
                                   ActivationType ve_activation = ActivationType::NONE,
                                   bool ve_bias_enabled = false,
                                   Address ve_bias_addr = 0);

    static DMInstruction barrier();
    static DMInstruction wait(uint32_t op_mask);
    static DMInstruction signal(uint32_t signal_id);
    static DMInstruction halt();
};

// ============================================================================
// Data Movement Program
// ============================================================================

/**
 * @brief A complete data movement program
 *
 * This represents the system-level schedule for a kernel like matmul.
 * The program is derived from SURE analysis and encodes the optimal
 * data movement pattern for the chosen dataflow strategy.
 */
struct DMProgram {
    // Program metadata
    std::string name;           // e.g., "matmul_1024x1024x1024_os"
    uint32_t version;           // Program format version

    // Matrix dimensions
    Size M, N, K;

    // Tiling configuration
    Size Ti, Tj, Tk;            // Tile dimensions
    Size L1_Ki;                 // L1 streaming chunk

    // Dataflow strategy
    enum class Dataflow {
        OUTPUT_STATIONARY,      // C in PEs, A+B stream through
        WEIGHT_STATIONARY,      // B in PEs, A streams, C accumulates in L2
        INPUT_STATIONARY        // A in PEs, B streams, C accumulates in L2
    } dataflow;

    // Instruction stream
    std::vector<DMInstruction> instructions;

    // Memory layout (technology-agnostic)
    struct MemoryMap {
        // External memory addresses (set at load time)
        Address a_base;
        Address b_base;
        Address c_base;

        // L3 tile allocations
        struct L3Alloc {
            uint8_t tile_id;
            Address offset;
            Size size;
            MatrixID matrix;
            BufferSlot buffer;
        };
        std::vector<L3Alloc> l3_allocations;

        // L2 bank allocations
        struct L2Alloc {
            uint8_t bank_id;
            Address offset;
            Size size;
            MatrixID matrix;
            BufferSlot buffer;
        };
        std::vector<L2Alloc> l2_allocations;
    } memory_map;

    // Performance estimates (from SURE analysis)
    struct Estimates {
        uint64_t total_cycles;
        uint64_t external_mem_bytes;    // Technology-agnostic
        uint64_t l3_bytes;
        uint64_t l2_bytes;
        double arithmetic_intensity;
        double estimated_gflops;
    } estimates;

    // Program statistics
    size_t num_dma_ops() const;
    size_t num_bm_ops() const;
    size_t num_str_ops() const;
    size_t num_sync_ops() const;
};

// ============================================================================
// Program Builder for Output-Stationary MatMul
// ============================================================================

/**
 * @brief Builds output-stationary data movement programs
 *
 * Output-stationary is optimal when:
 * - K is large (many accumulations per output)
 * - M and N are balanced (good reuse of both A and B)
 * - Avoiding C writeback during accumulation
 */
class OutputStationaryProgramBuilder {
public:
    struct Config {
        Size M, N, K;               // Matrix dimensions
        Size Ti, Tj, Tk;            // Tile sizes
        Size L1_Ki;                 // L1 streaming chunk
        Size systolic_size;         // Systolic array dimension (e.g., 16)
        Size element_size;          // Element size in bytes (e.g., 4 for float32)

        // Memory hierarchy sizes (technology-agnostic)
        Size l3_tile_capacity;      // L3 tile capacity in bytes
        Size l2_bank_capacity;      // L2 bank capacity in bytes
        Size l1_buffer_capacity;    // L1 buffer capacity in bytes

        // Number of components
        uint8_t num_l3_tiles;
        uint8_t num_l2_banks;
        uint8_t num_l1_buffers;

        // Double-buffering enabled
        bool double_buffer;

        // Tile caching (Phase 1)
        bool enable_tile_caching = true;  // Track tile reuse in L3
    };

    explicit OutputStationaryProgramBuilder(const Config& config);

    /**
     * @brief Build the complete output-stationary program
     *
     * Loop order for output-stationary:
     *   for ti in 0..M/Ti:           // Output row tiles
     *     for tj in 0..N/Tj:         // Output col tiles
     *       // C[ti,tj] accumulates in PEs
     *       for tk in 0..K/Tk:       // Reduction tiles
     *         Load A[ti,tk] to L3 (if not cached)
     *         Load B[tk,tj] to L3 (if not cached)
     *         Move A[ti,tk] L3→L2
     *         Move B[tk,tj] L3→L2
     *         Stream A rows to systolic array
     *         Stream B cols to systolic array
     *         // Compute happens reactively in PEs
     *       Drain C[ti,tj] from PEs
     *       Store C[ti,tj] to external memory
     */
    DMProgram build();

    /**
     * @brief Get tile cache statistics after build()
     * @return String summary of cache hits/misses
     */
    std::string get_cache_stats() const;

private:
    Config config_;
    uint32_t next_instruction_id_;
    Address current_l3_offset_[2];  // Per buffer slot
    Address current_l2_offset_[2];  // Per buffer slot

    // Tile iteration counts
    Size m_tiles_, n_tiles_, k_tiles_;

    // Tile cache tracking (Phase 1)
    // Using simple set-based tracking: {matrix, ti, tk} for A, {matrix, tk, tj} for B
    struct TileCacheState {
        std::set<uint64_t> resident_tiles;  // Encoded tile keys
        Size capacity_bytes;
        Size used_bytes = 0;
        size_t hits = 0;
        size_t misses = 0;
        Size bytes_saved = 0;

        uint64_t encode_key(MatrixID mat, uint16_t i, uint16_t j, uint16_t k) const {
            return (static_cast<uint64_t>(mat) << 48) |
                   (static_cast<uint64_t>(i) << 32) |
                   (static_cast<uint64_t>(j) << 16) |
                   static_cast<uint64_t>(k);
        }

        bool is_resident(MatrixID mat, uint16_t i, uint16_t j, uint16_t k) const {
            return resident_tiles.count(encode_key(mat, i, j, k)) > 0;
        }

        void mark_resident(MatrixID mat, uint16_t i, uint16_t j, uint16_t k, Size size) {
            resident_tiles.insert(encode_key(mat, i, j, k));
            used_bytes += size;
        }

        void reset() {
            resident_tiles.clear();
            used_bytes = 0;
            hits = 0;
            misses = 0;
            bytes_saved = 0;
        }
    };
    mutable TileCacheState tile_cache_;

    // Address calculation
    Address calculate_a_tile_addr(TileCoord tile) const;
    Address calculate_b_tile_addr(TileCoord tile) const;
    Address calculate_c_tile_addr(TileCoord tile) const;

    // Instruction generation (with cache-aware variants)
    bool try_emit_load_a_tile(DMProgram& prog, TileCoord tile, BufferSlot buf);
    bool try_emit_load_b_tile(DMProgram& prog, TileCoord tile, BufferSlot buf);
    void emit_load_a_tile(DMProgram& prog, TileCoord tile, BufferSlot buf);
    void emit_load_b_tile(DMProgram& prog, TileCoord tile, BufferSlot buf);
    void emit_move_a_l3_to_l2(DMProgram& prog, TileCoord tile, BufferSlot buf);
    void emit_move_b_l3_to_l2(DMProgram& prog, TileCoord tile, BufferSlot buf);
    void emit_stream_a_rows(DMProgram& prog, TileCoord tile, BufferSlot buf);
    void emit_stream_b_cols(DMProgram& prog, TileCoord tile, BufferSlot buf);
    void emit_drain_c(DMProgram& prog, TileCoord tile);
    void emit_store_c_tile(DMProgram& prog, TileCoord tile);
    void emit_barrier(DMProgram& prog);
};

} // namespace sw::kpu::isa
