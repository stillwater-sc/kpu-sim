//
// Stillwater KPU Assembly - MLP Forward Pass
// Domain Flow Architecture with Programmable Systolic Array
//
// Architecture: Distributed Data Flow Machine with Tagged Token Matching
//
// Key Components:
//   1. Programmable Systolic Array (Compute Fabric)
//      - 2D/3D torus mesh of Processing Elements (PEs)
//      - Each PE has CAM (Content Addressable Memory) for tag matching
//      - Position-independent domain flow programs
//      - Tagged token routing based on N-dimensional abstract space
//
//   2. Data Movement Hierarchy (Sequencer-based)
//      - DMA Engines: DRAM ↔ L3 (with sequencers)
//      - Block Movers: L3 ↔ L2 (with sequencers)
//      - Streamers: L2 → L1 → Compute Fabric (with sequencers)
//
//   3. Distributed Memory Hierarchy
//      - DRAM (external)
//      - L3 (distributed on-chip, multiple banks)
//      - L2 (scratchpad memory)
//      - L1 (stream buffers feeding compute fabric)
//
// Execution Model:
//   - Install domain flow program in all PEs (broadcast)
//   - Configure sequencers in data movement engines
//   - Start data movement → triggers dataflow computation
//   - Results automatically route based on tags
//

.target kpu_v1
.precision bf16
.architecture domain_flow

// ============================================================================
// Memory Address Spaces
// ============================================================================

.address_space dram,     0x0000_0000_0000_0000
.address_space l3_bank0, 0x1000_0000_0000_0000
.address_space l3_bank1, 0x1000_0001_0000_0000
.address_space l3_bank2, 0x1000_0002_0000_0000
.address_space l3_bank3, 0x1000_0003_0000_0000
.address_space l2_mem,   0x2000_0000_0000_0000
.address_space l1_north, 0x3000_0000_0000_0000
.address_space l1_west,  0x3000_0001_0000_0000

// ============================================================================
// Function: mlp_layer_forward_kpu
// Computes: output = ReLU(weights × input + bias)
// ============================================================================

.entry mlp_layer_forward_kpu

// Input parameters
.param input_dram_addr,    %r0      // DRAM address of input [1 x 512]
.param weights_dram_addr,  %r1      // DRAM address of weights [256 x 512]
.param bias_dram_addr,     %r2      // DRAM address of bias [256]
.param output_dram_addr,   %r3      // DRAM address for output [1 x 256]

// Problem dimensions
.const input_size,    512
.const output_size,   256
.const tile_size,     128
.const bf16_size,     2


// ============================================================================
// STAGE 1: INSTALL DOMAIN FLOW PROGRAM IN COMPUTE FABRIC
// (This MUST be first - array needs to be ready before data arrives)
// ============================================================================

STAGE1_INSTALL_KERNEL:
    // The domain flow program is a small program (100-200 bytes) that runs
    // in each PE. It describes:
    //   - What tagged tokens to match
    //   - What operation to perform when operands arrive
    //   - What tag to assign to the result
    //   - How to route the result token
    //
    // The program is position-independent - same code in all PEs
    // Tags represent abstract N-dimensional computational coordinates

    // ========================================================================
    // Define the MLP domain flow kernel
    // ========================================================================

    .kernel mlp_matmul_kernel:
        // Kernel size: ~180 bytes
        // Implements: C[i,j] = Σ(A[i,k] × B[k,j]) with ReLU and bias

        // ------------------------
        // Instruction 1: Match input and weight tokens, compute MAC
        // ------------------------
        .inst mac_instruction:
            // Match Pattern (i,j,k) in PE's CAM
            .match  tag_pattern = {
                        reid: "a",        // recurrence id for input tensor
                        route: (1, 0, 0), // A tensor element propagation
                    },
                    {
                        reid: "b",        // recurrence id for weight tensor
                        route: (0, 1, 0), // B tensor element propagation
                    },
                    {
                        reid: "c",        // recurrence id for input tensor
                        route: (0, 0, 1), // C tensor element propagation
                    }
            // When both tokens with matching k_idx arrive at same PE:
            .operation  mac_bf16                        // Multiply-accumulate
            .consume    both                            // Consume both input tokens
            // No output token yet - accumulating

        // ------------------------
        // Instruction 2: Detect completion and emit partial sum
        // ------------------------
        .inst emit_partial:
            .condition  k == k_max                  // All k iterations done
            .operation  move
            .source     c
            .emit_token reid: "c",
                        value: c,
                        tag: (0,j)    // Route back to top array edge handling this output
            .reset      c                           // Clear for next use

        // ------------------------
        // Instruction 3: Add bias
        // ------------------------
        .inst add_bias:
            .match  tag_pattern = {
                        reid: "c",
                        dims: (row_idx, col_idx),
                    },
                    {
                        reid: "bias_element",
                        dims: (col_idx),
                    }
            .operation  add_bf16
            .emit_token signature: "C_biased",
                        dims: (row_idx, col_idx),
                        value: c + bias

        // ------------------------
        // Instruction 4: Apply ReLU
        // ------------------------
        .inst apply_relu:
            .match  tag_pattern = {
                        reid: "C_biased",
                        dims: (row_idx, col_idx),
                    }
            .operation  max_bf16(value, 0.0)            // max(x, 0)
            .emit_token reid: "C_output",
                        dims: (row_idx, col_idx),
                        value: max(C_biased, 0),
                        tag: OUTPUT_TAG                 // Route to L1

        // ------------------------
        // End of kernel definition
        // ------------------------
        .kernel_end

    // ========================================================================
    // Load kernel to edge of compute fabric
    // ========================================================================

    // The kernel binary is in program memory or L2
    // Load it to the broadcast interface

    kernel.load     src=program_mem:mlp_matmul_kernel,  // Kernel code
                    dst=fabric_edge_buffer,             // Edge buffer for broadcast
                    size=180                            // Kernel size in bytes

    // ========================================================================
    // Broadcast kernel to all PEs via program network overlay
    // ========================================================================

    // The fabric has a network overlay to emulate systolic array Nearest Neighbot communication
    // Broadcast sends the same program to all PEs simultaneously

    fabric.broadcast source=fabric_edge_buffer,
                     target=all_pes,                    // All PEs in mesh
                     mode=program_install,              // Install in PE program memory
                     overlay=configuration_network      // Use config network, not data network

    // Wait for broadcast to complete
    fabric.sync     broadcast_complete

    // ========================================================================
    // Initialize PE state
    // ========================================================================

    // Each PE needs to initialize its local state
    // - Clear CAM entries
    // - Reset accumulators
    // - Prepare for tagged token matching

    fabric.init     target=all_pes,
                    clear_cam=true,                     // Clear content addressable memory
                    clear_accumulators=true,            // Clear partial_sum accumulators
                    reset_counters=true                 // Reset k_idx counters

    fabric.sync     init_complete

    // Compute fabric is now READY to receive data tokens
    // As soon as tagged tokens arrive, computation will fire


// ============================================================================
// STAGE 2: CONFIGURE DMA ENGINE SEQUENCERS
// (Program the DMA engines to execute the block schedule)
// ============================================================================

STAGE2_CONFIGURE_DMA_SEQUENCERS:
    // DMA engines have sequencers that execute a schedule
    // Each sequencer is a small state machine that orchestrates transfers

    // ========================================================================
    // DMA Sequencer 0: Input Loading Schedule
    // ========================================================================

    dma.sequencer.define seq_input_load:
        // Step 1: Transfer input from DRAM to L3 bank 0
        .step   transfer:
                    src=dram:%r0,
                    dst=l3_bank0:0x0000,
                    size=input_size*bf16_size,
                    mode=burst,
                    priority=high

        // Step 2: Wait for completion
        .step   wait:
                    condition=transfer_complete

        // Step 3: Signal ready
        .step   signal:
                    flag=input_in_l3

        .seq_end

    // Install sequencer in DMA engine 0
    dma.install_sequencer   engine=dma0,
                            sequencer=seq_input_load,
                            auto_start=false            // Don't start yet

    // ========================================================================
    // DMA Sequencer 1: Weight Loading Schedule
    // ========================================================================

    dma.sequencer.define seq_weight_load:
        // Distributed loading across L3 banks
        // Loop over 4 quarters of weight matrix

        .step   loop_start:
                    counter=%quarter,
                    range=0..3

        .step   compute_addresses:
                    src_addr = dram:%r1 + %quarter*(output_size*input_size/4)*bf16_size,
                    dst_bank = l3_bank1 + (%quarter % 4),    // Rotate through banks
                    dst_offset = (%quarter / 4) * 0x8000     // Offset within bank

        .step   transfer:
                    src=%src_addr,
                    dst=%dst_bank:%dst_offset,
                    size=(output_size*input_size/4)*bf16_size,
                    mode=burst

        .step   wait:
                    condition=transfer_complete

        .step   loop_end:
                    increment=%quarter,
                    jump_if_not_done=loop_start

        .step   signal:
                    flag=weights_in_l3

        .seq_end

    dma.install_sequencer   engine=dma1,
                            sequencer=seq_weight_load,
                            auto_start=false

    // ========================================================================
    // DMA Sequencer 2: Bias Loading Schedule
    // ========================================================================

    dma.sequencer.define seq_bias_load:
        .step   transfer:
                    src=dram:%r2,
                    dst=l3_bank0:0x0400,
                    size=output_size*bf16_size,
                    mode=sequential

        .step   wait:
                    condition=transfer_complete

        .step   signal:
                    flag=bias_in_l3

        .seq_end

    dma.install_sequencer   engine=dma0,                // Reuse dma0 after input
                            sequencer=seq_bias_load,
                            trigger=input_in_l3,        // Start after input loaded
                            auto_start=false


// ============================================================================
// STAGE 3: CONFIGURE BLOCK MOVER SEQUENCERS
// (Program block movers to move data L3 → L2)
// ============================================================================

STAGE3_CONFIGURE_BLOCKMOVER_SEQUENCERS:

    // ========================================================================
    // Block Mover Sequencer 0: Input to L2
    // ========================================================================

    blkmov.sequencer.define seq_input_to_l2:
        .step   wait_for_data:
                    condition=input_in_l3               // Wait for DMA

        .step   transfer:
                    src=l3_bank0:0x0000,
                    dst=l2_mem:0x0000,                  // L2 input area
                    size=input_size*bf16_size,
                    mode=burst

        .step   wait:
                    condition=transfer_complete

        .step   signal:
                    flag=input_in_l2

        .seq_end

    blkmov.install_sequencer    engine=blkmov0,
                                sequencer=seq_input_to_l2,
                                auto_start=false

    // ========================================================================
    // Block Mover Sequencer 1: Tiled Weight Movement
    // ========================================================================

    blkmov.sequencer.define seq_weights_tiled:
        // This sequencer moves weight tiles on demand
        // Nested loop: output tiles × contraction tiles

        .step   outer_loop_start:
                    counter=%out_tile,
                    range=0..1                          // 2 output tiles (256/128)

        .step   inner_loop_start:
                    counter=%contract_tile,
                    range=0..3                          // 4 contraction tiles (512/128)

        .step   compute_tile_address:
                    tile_idx = %out_tile * 4 + %contract_tile,
                    bank_idx = tile_idx % 4,            // Which L3 bank
                    bank_offset = (tile_idx / 4) * (tile_size*tile_size*bf16_size)

        .step   transfer_tile:
                    src=l3_bank1 + %bank_idx : %bank_offset,
                    dst=l2_mem:0x0400,                  // L2 weight tile area
                    size=tile_size*tile_size*bf16_size,
                    mode=burst

        .step   wait:
                    condition=transfer_complete

        .step   signal:
                    flag=tile_ready,
                    data=(out_tile=%out_tile, contract_tile=%contract_tile)

        .step   inner_loop_end:
                    increment=%contract_tile,
                    jump_if_not_done=inner_loop_start

        .step   outer_loop_end:
                    increment=%out_tile,
                    jump_if_not_done=outer_loop_start

        .step   signal:
                    flag=all_tiles_done

        .seq_end

    blkmov.install_sequencer    engine=blkmov1,
                                sequencer=seq_weights_tiled,
                                trigger=weights_in_l3,  // Start when weights in L3
                                auto_start=false

    // ========================================================================
    // Block Mover Sequencer 2: Bias to L2
    // ========================================================================

    blkmov.sequencer.define seq_bias_to_l2:
        .step   wait_for_data:
                    condition=bias_in_l3

        .step   transfer:
                    src=l3_bank0:0x0400,
                    dst=l2_mem:0x8600,                  // L2 bias area
                    size=output_size*bf16_size,
                    mode=sequential

        .step   signal:
                    flag=bias_in_l2

        .seq_end

    blkmov.install_sequencer    engine=blkmov0,         // Reuse blkmov0
                                sequencer=seq_bias_to_l2,
                                trigger=input_in_l2,    // After input moved
                                auto_start=false


// ============================================================================
// STAGE 4: CONFIGURE STREAMER SEQUENCERS
// (Program streamers to feed L2 → L1 → Compute Fabric with tagged tokens)
// ============================================================================

STAGE4_CONFIGURE_STREAMER_SEQUENCERS:

    // Streamers are the interface between memory and the dataflow fabric
    // They attach TAGS to data as it streams, creating tagged tokens

    // ========================================================================
    // Streamer Sequencer 0: Input Token Stream
    // ========================================================================

    streamer.sequencer.define seq_input_stream:
        // Stream input elements with appropriate tags
        // Input is broadcast to all columns (different k_idx for each tile)

        .step   wait_for_data:
                    condition=input_in_l2

        .step   outer_loop:                             // For each tile
                    counter=%tile,
                    range=0..3                          // 4 contraction tiles

        .step   configure_stream:
                    src=l2_mem:0x0000 + (%tile * tile_size * bf16_size),
                    dst=l1_north,                       // North input to fabric
                    count=tile_size,                    // 128 elements
                    element_size=bf16_size

        .step   stream_with_tags:
                    // For each element i in [0..127]:
                    for_each_element %i:
                        value = src[%i],
                        tag_signature = "A_element",
                        tag_dims = (row_idx=0, k_idx=%tile*tile_size + %i),
                        tag_hash = hash(0, %tile*tile_size + %i),
                        emit_token(signature, dims, value, tag_hash)

        .step   wait_stream_complete

        .step   loop_end:
                    increment=%tile,
                    jump_if_not_done=outer_loop

        .seq_end

    streamer.install_sequencer  engine=stream0,
                                sequencer=seq_input_stream,
                                auto_start=false

    // ========================================================================
    // Streamer Sequencer 1: Weight Token Stream
    // ========================================================================

    streamer.sequencer.define seq_weight_stream:
        // Stream weight tiles with tags
        // Synchronized with block mover tile delivery

        .step   wait_for_tile:
                    condition=tile_ready,               // Block mover signals
                    capture_data=%tile_info             // (out_tile, contract_tile)

        .step   configure_stream:
                    src=l2_mem:0x0400,                  // Weight tile in L2
                    dst=l1_west,                        // West input to fabric
                    count=tile_size*tile_size,          // 128×128 elements
                    element_size=bf16_size

        .step   stream_2d_with_tags:
                    // For each weight w[i,j] in tile:
                    for_each_row %i in [0..127]:
                        for_each_col %j in [0..127]:
                            value = src[%i * tile_size + %j],
                            row_abs = %tile_info.out_tile * tile_size + %i,
                            col_abs = %tile_info.contract_tile * tile_size + %j,
                            tag_signature = "W_element",
                            tag_dims = (k_idx=col_abs, col_idx=row_abs),
                            tag_hash = hash(col_abs, row_abs),
                            emit_token(signature, dims, value, tag_hash)

        .step   signal_done:
                    flag=tile_streamed

        .step   check_more_tiles:
                    jump_if_condition=tile_ready, wait_for_tile

        .seq_end

    streamer.install_sequencer  engine=stream1,
                                sequencer=seq_weight_stream,
                                auto_start=false

    // ========================================================================
    // Streamer Sequencer 2: Bias Token Stream
    // ========================================================================

    streamer.sequencer.define seq_bias_stream:
        .step   wait_for_data:
                    condition=bias_in_l2

        .step   configure_stream:
                    src=l2_mem:0x8600,
                    dst=l1_west,                        // Inject into fabric
                    count=output_size,
                    element_size=bf16_size

        .step   stream_with_tags:
                    for_each_element %i in [0..255]:
                        value = src[%i],
                        tag_signature = "bias_element",
                        tag_dims = (col_idx=%i),
                        tag_hash = hash_bias(%i),
                        emit_token(signature, dims, value, tag_hash)

        .seq_end

    streamer.install_sequencer  engine=stream2,         // Use another streamer
                                sequencer=seq_bias_stream,
                                trigger=bias_in_l2,
                                auto_start=false


// ============================================================================
// STAGE 5: CONFIGURE OUTPUT DRAIN SEQUENCER
// (Collect results from compute fabric)
// ============================================================================

STAGE5_CONFIGURE_OUTPUT_DRAIN:

    // Output tokens with signature "C_output" need to be drained from fabric
    // They emerge from the dataflow computation and need to be collected

    fabric.drain.sequencer.define seq_output_collect:
        .step   configure_drain:
                    token_signature="C_output",         // Match output tokens
                    dst=l2_mem:0x8400,                  // L2 output buffer
                    count=output_size,                  // Expect 256 results
                    timeout=10000                       // Max cycles to wait

        .step   collect_tokens:
                    // For each output token that arrives:
                    for_each_arriving_token:
                        extract_value,
                        extract_coords(row_idx, col_idx),
                        dst[col_idx] = value            // Write to L2

        .step   wait_all_collected:
                    condition=count_reached

        .step   signal:
                    flag=output_in_l2

        .seq_end

    fabric.install_drain_sequencer  sequencer=seq_output_collect,
                                    auto_start=false


// ============================================================================
// STAGE 6: CONFIGURE WRITEBACK SEQUENCERS
// (Move results L2 → L3 → DRAM)
// ============================================================================

STAGE6_CONFIGURE_WRITEBACK_SEQUENCERS:

    // ========================================================================
    // Block Mover: L2 → L3
    // ========================================================================

    blkmov.sequencer.define seq_output_to_l3:
        .step   wait_for_data:
                    condition=output_in_l2

        .step   transfer:
                    src=l2_mem:0x8400,
                    dst=l3_bank0:0x0800,
                    size=output_size*bf16_size,
                    mode=burst

        .step   signal:
                    flag=output_in_l3

        .seq_end

    blkmov.install_sequencer    engine=blkmov0,
                                sequencer=seq_output_to_l3,
                                auto_start=false

    // ========================================================================
    // DMA: L3 → DRAM
    // ========================================================================

    dma.sequencer.define seq_output_to_dram:
        .step   wait_for_data:
                    condition=output_in_l3

        .step   transfer:
                    src=l3_bank0:0x0800,
                    dst=dram:%r3,
                    size=output_size*bf16_size,
                    mode=sequential

        .step   wait:
                    condition=transfer_complete

        .step   signal:
                    flag=kernel_complete

        .seq_end

    dma.install_sequencer   engine=dma0,
                            sequencer=seq_output_to_dram,
                            auto_start=false


// ============================================================================
// STAGE 7: START EXECUTION
// (Trigger the dataflow by starting sequencers)
// ============================================================================

STAGE7_EXECUTE:
    // Now that everything is configured:
    //   - Domain flow kernel installed in compute fabric
    //   - All sequencers programmed
    //   - Fabric is ready, waiting for tagged tokens
    //
    // Starting the data movement will trigger the dataflow computation

    // ========================================================================
    // Start all sequencers in orchestrated order
    // ========================================================================

    // Start DMA engines (DRAM → L3)
    dma.start               engine=dma0         // Input
    dma.start               engine=dma1         // Weights

    // Block movers and streamers start automatically via triggers
    // (configured with trigger= parameters earlier)

    // Start output drain (fabric → L2)
    fabric.start_drain      sequencer=seq_output_collect

    // ========================================================================
    // Wait for completion
    // ========================================================================

    // Wait for final signal that everything is done
    wait_for_signal         flag=kernel_complete

    // ========================================================================
    // Cleanup and exit
    // ========================================================================

    // Optionally: unload kernel from fabric for next use
    // fabric.unload         target=all_pes

    exit


// ============================================================================
// KPU DOMAIN FLOW INSTRUCTION SET
// ============================================================================
//
// FABRIC PROGRAMMING:
//   kernel.load           src, dst, size
//   fabric.broadcast      source, target, mode, overlay
//   fabric.sync           condition
//   fabric.init           target, clear_cam, clear_accumulators, reset_counters
//   fabric.unload         target
//
// KERNEL DEFINITION:
//   .kernel name:
//   .inst instruction_name:
//     .match tag_pattern = {...}
//     .operation op_type
//     .emit_token signature, dims, value, tag
//     .accumulate target
//     .condition expression
//   .kernel_end
//
// SEQUENCER CONFIGURATION:
//   dma.sequencer.define name: ... .seq_end
//   blkmov.sequencer.define name: ... .seq_end
//   streamer.sequencer.define name: ... .seq_end
//   fabric.drain.sequencer.define name: ... .seq_end
//
// SEQUENCER INSTALLATION:
//   dma.install_sequencer engine, sequencer, trigger, auto_start
//   blkmov.install_sequencer engine, sequencer, trigger, auto_start
//   streamer.install_sequencer engine, sequencer, auto_start
//   fabric.install_drain_sequencer sequencer, auto_start
//
// SEQUENCER CONTROL:
//   dma.start engine
//   blkmov.start engine
//   streamer.start engine
//   fabric.start_drain sequencer
//
// TAGGED TOKEN OPERATIONS:
//   emit_token signature, dims, value, tag_hash
//   tag_hash = hash(dimensions...)
//
// SYNCHRONIZATION:
//   wait_for_signal flag
//   signal flag, data
//
// ============================================================================
