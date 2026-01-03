# Checkboard Tile Domain Flow design


  Question 1: DMA Timing Model

  For a single 4KB tile transfer from memory to L3:
  - How many cycles from DMA request to transfer start?
  - How many cycles for the actual transfer (depends on bandwidth)?
  - What is the DMA bandwidth in bytes/cycle?

  Question 2: Compute Timing

  For a 32×32 × 32×32 tile multiply-accumulate operation:
  - How many cycles does this take?
  - Does compute overlap with data loading, or is it strictly sequential?

  Question 3: L3 Buffer Model

  When tiles arrive at L3[0,0]:
  - Is there a buffer/queue at L3 to hold multiple tiles?
  - If so, what is the capacity (number of tiles)?
  - Does L2 pull tiles immediately, or do they wait?

  Question 4: Retire Path

  For output tile C{0,0}:
  - Which mesh edge does it exit through (SOUTH, EAST, or dedicated path)?
  - Does it use a separate DMA engine for store, or share with load DMAs?

  ---
  Also, please review the vocabulary document - is the terminology correct? Any adjustments needed?

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Q#1 DMA Timing Model: 
  The egress throughput needs to match the NoC's ingress. For the whole checkerboard to work without contention, we need to move all
  the tiles in lock step. The current NoC design is a 512bit bus that can move one 64Byte vector of data per clock. Hence a 32x32 submatrix tile would take
  64 cycles. The L3 will need to have two concurrent write ports to be able to deal with writing two 64Bytes vectors per clock. In stable state, we will
  also have the partial C tile writes, so that would imply that we need at least three write ports. As the tiles in the L3 get consumed, we need at least
  three concurrent read ports, one for A tiles, one for B tiles, and one for C tiles. If we ALSO do L3 -> L3 block moves concurrently, we would need two
  more concurrent read ports and two more concurrent write ports. The ingress throughput of the DMA engine is governed by the DRAM bank latency and
  contention. To minimize the contention, the A and B and C tensors should be placed in memory in such a way that the schedule of touching concurrent tiles
  is such that no two tiles reside in the same bank. We should create a utility to check bank contention given a schedule and a memory layout. As there is
  quite a bit of concurrency, which may trigger contention, the DMA model cannot be clock counting based, it has to accurately model the cycle by cycle
  resource occupation and contention with arbitration. 

Q#2: Compute Timing Model:
  The compute timing is governed by a pipelined block matmul with a schedule vector tau of [1 1
  1] and a projection direction S of [0 0 1]. So we are projecting the 3D SURE embedding of a 3D Cartesian lattice onto the [i, j] plane, which yields a
  triangular fill and drain, and the subsequent partial sums can be perfectly pipelined in the fill and drain, yielding a 100% utilization and efficiency
  when the block matrix is pushed through. This projection direction S yields an output stationary schedule on the systolic array. The BlockMovers need to
  gather the right row and column segments out of the L3, and write them into the right structure in the banks of the L2, so that the Streamers can pick up
  a straight vector fetch and push them in the L1 stream buffers. Any hickup will cause the compute tile to be incorrect, so again, we need to model this
  as a cycle-accurate resource occupation and contention mechanism. 

Q#3: L3 Buffer Model:
  The L3 is a software managed on-chip memory, so there isn't a buffer. The flits need
  to be read and written directly to the L3 memory banks. Again, needs to be modeled cycle-accurately. An L3 tile is typically in the 1-16MB range
  depending on the checkboard and compute tile configurations, and should be highly concurrent with many independent SRAM banks to support the read and
  write concurrency required for satisfying the compute tile and L3 to L3 tile movements. 

Q#4: Result Tile Retirement Path
  The activity wavefront of an output stationary schedule is
  from top-left to bottom-right. This implies that the retire path needs to move in the opposite direction. Easiest would be to use the NORTH path meaning
  that the result tiles flow out via the NORTH links in the NoC. The Streamers need to receive the result C tile in the L1 stream buffers and write it into
  the L2 banks, where the BlockMovers can pick it up and write it to the L3. To continue with the push architecture, when a result tile shows up in the L2
  banks, it depends if it is going to be reused or not. If it is, then the BlockMover needs to move the partial tile into the L3 for later reuse. If it is
  complete, then the BlockMover needs to send this tile to the adjacent L3 and trigger that L3 tile's BlockMover to propagate the tile to the NORTH edge,
  and when it gets there, the edge BlockMover needs to be able to write that tile to a buffer in the appropriate DMA engine so that it can be retired by a
  DMA request to be written to DRAM. This design, all the data movements are push based and we need to manage this with flow control credits so that
  writting is always to a pre-allocated resource and thus there is no backpressure.

