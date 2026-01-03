# LPDDR5 Memory Controller Read/Write Protocol

The memory controller takes a memory request and translate it into a set of commands to read or write a vector of data.

---

### 1. Address Translation and Bank Management

Before we can drive the command bus, we must translate the device's physical address into a **Memory Map** that the DRAM understands:

* **Decoding:** Break the address into: **Channel - Rank - Bank Group - Bank - Row - Column.**
* **State Check:** Check if the target **Bank** is already "Open" (Active).
* If a *different* row is open, issue a `PRECHARGE` command to close it.
* If no row is open, proceed to the next step.

### 2. The "Activate" Phase (Row Opening)

LPDDR5 memory is organized like a grid. You cannot read a single cell; you must open an entire "page" (row).

* **Command:** Drive the `ACTIVATE` command on the Command/Address (CA) bus.
* **Timing ():** Wait for a specific duration called **Row-to-Column Delay**. This is the time it takes for the charge to flow from the memory cells to the sense amplifiers.

### 3. Synchronizing the Clocks (WCK2CK Sync)

Unlike older DDR, LPDDR5 uses a high-speed "Write Clock" (**WCK**) that runs much faster than the Command Clock (**CK**).

* **Sync:** Before sending the Read command, perform a **WCK2CK synchronization** to ensure the DRAM's data output will align perfectly with your internal timers.
* **Read Strobe:** Prepare to receive the **RDQS** (Read Data Strobe), which the DRAM will send along with the data to tell you *exactly* when to "sample" the bits.

### 4. The "Read" Command and Bursting

Now command the DRAM exactly which part of the open row we need:

* **The Command:** Issue the `READ` command with the **Column Address**.
* **Burst Length (BL16):** Since LPDDR5 channels are typically 16 bits wide, a single Read command triggers a **Burst Length of 16** ( or **32 bytes**).
* **Cache Line Fetch:** To get a full **64B cache line**, you will typically issue **two consecutive Read commands** (or use a Burst Length of 32 if configured) to fill the CPU's request.

### 5. Data Transfer and Link ECC

The data begins to appear on the **DQ** (Data) lines.

* **CAS Latency ():** The data arrives after a set number of clock cycles (the "Latency").
* **Link ECC:** As the memory controller, check the incoming data against the **ECC (Error Correction Code)** bits sent by the DRAM. If a single bit flipped during the high-speed trip across the motherboard, correct it on the fly.

### 6. Closing the Transaction

Once the 64 bytes are in the memory controller's buffers:

* **Forwarding:** Send the data to the CPU via the System-on-Chip (SoC) interconnect.
* **Power Management:** If no further requests are pending for that bank, issue a `PRECHARGE` to close the row and save power, or keep it open (an "Open Page Policy") if we predict the device will ask for nearby data soon.

---

**Summary Table: Controller Actions**

| Step | Controller Action | Key Timing Parameter |
| --- | --- | --- |
| **Open Row** | Issue `ACTIVATE` |  (Row to Col Delay) |
| **Request Data** | Issue `READ` with Col Address |  (CAS Latency) |
| **Receive Burst** | Capture data using `RDQS` strobe |  (Strobe to DQ skew) |
| **Verify** | Run Link ECC check | Internal Logic |

# LPDDR4 vs LPDDR5

To master the LPDDR5 read cycle, we need to understand that this generation introduced a "Divide and Conquer" strategy for performance.

Unlike LPDDR4, where the clocks were synchronized once and stayed that way, LPDDR5 is a "clock-on-demand" system. It separates the **Command Clock (CK)** from the **Data Clock (WCK)** to save massive amounts of power, but this creates new responsibilities for the controller.

---

## Read Cycle

### 1. The Timing Shift: LPDDR4 vs. LPDDR5

In LPDDR4, the Command/Address (CA) and Data (DQ) buses shared a common timing reference. In LPDDR5, they are split to allow the command bus to run much slower while the data bus screams at up to 6400+ Mbps.

| Feature | LPDDR4/4X | LPDDR5 |
| --- | --- | --- |
| **CK : DQ Ratio** | 1:1 (CK runs at Data Rate) | **1:4 or 1:2** (CK is much slower) |
| **Clocking** | DQS (Bi-directional strobe) | **WCK** (Uni-directional, "Free-running" or "Dynamic") |
| **Architecture** | 8 Banks | **Bank Group Mode** (similar to DDR4/5) |
| **ECC** | Inline ECC (Optional) | **Link ECC + Inline ECC** (Standard) |

---

### 2. The WCK2CK Sync (The "Handshake")

Since the Write Clock (**WCK**) is often turned off to save power when idle, we cannot just send a Read command and expect data.

* **Phase Alignment:** When we decide to read, we must first send a synchronization sequence. The DRAM uses the **WCK** signal to sample the **CK** signal to determine the phase relationship (the "phase 0" vs "phase 180" alignment).
* **The CAS-WCK Command:** You issue a specific `CAS` command with a sync flag. Only after the timing parameter  (WCK Enable Latency) is met can we safely expect the DRAM to be ready to pipe data out.

---

### 3. How to Handle Link ECC

In LPDDR4, if a bit flipped on the wires due to electrical noise, the requesting device just got bad data. In LPDDR5, **Link ECC** creates a "safety net" specifically for the trip across the PCB.

**As the Controller, the Read logic looks like this:**

1. **Receive Packet:** The DRAM sends a 16-bit data burst + 2 bits of **Parity/ECC** on the DMI (Data Mask Inversion) pins.
2. **Generate Syndrome:** Run the incoming 16 bits through the internal XOR-tree logic to generate a "Check Syndrome."
3. **Compare & Correct:** * If **Syndrome == 0**: The data is clean. Forward to CPU.
* If **Syndrome != 0**: You use the code to identify which bit flipped (e.g., bit 4). You flip it back to the correct state before the CPU ever sees it.


4. **Logging:** You increment a Hardware Error Counter (visible to the OS via WHEA or mcelog) so the system knows the signal integrity is degrading.

---

### 4. The Data "Burst" Logic

LPDDR5 is designed for **32-byte (BL16)** or **64-byte (BL32)** fetches.

* If the device wants a 64B cache line and we are in **16-Bank Mode**, trigger **two banks simultaneously**.
* Each bank provides 32 bytes of data. By interleaving them, you saturate the bus and fill that 64B cache line in a single "logical" burst, maximizing the efficiency of the high-speed WCK.

---

**Summary of the "Controller Brain" logic:**

> "I see a memory request at Address X. Translate that address to Bank Group 1, Bank 0. Wake up the WCK clock, send the Sync command, then the Read. While the data arrives, scrub the Link ECC for errors. Once all 512 bits (64 bytes) have been received, push them to the line buffer or page cache, and put the WCK back to sleep to save milliwatts."

## Write Cycle

As the memory controller, a **64-byte (64B) write** is more "aggressive" than a read. In a read, we wait for the DRAM to hand us data; in a write, we are the one driving the voltage on the pins, and we must present the data into the DRAM cells with perfect timing.

---

### 1. Data Buffering and ECC Generation

Before the first bit leaves the memory controller pins, we must prepare the payload:

* **Payload:** Take the 64B (512 bits) from the CPU/Interconnect.
* **Link ECC Calculation:** Divide the 512 bits into 16-bit segments. For every 16 bits of data, our ECC hardware logic calculates a **2-bit ECC parity**.
* **DMI Logic:** Check the data density. If more than 50% of the bits in a byte are `0`, you might flip them all to `1` and toggle the **Data Mask Inversion (DMI)** pin to save power (LPDDR5 consumes less power driving `1`s).

### 2. The WCK "Warm-up"

We cannot write if the high-speed clock (**WCK**) is sleeping.

* **WCK On:** Enable the WCK signal.
* **Synchronization:** Ensure WCK is running at the correct frequency ratio (either 4:1 or 2:1 relative to the Command Clock).
* **The Write Command:** Drive the `WRITE` command on the CA bus, specifying the Target Bank and Column.

### 3. The "Center-Aligned" Data Burst

This is the most critical electrical moment. In a read, the DRAM sends data "edge-aligned" with the strobe. In a write, **you** must send the data **center-aligned**.

* **Phase Shifting:** You shift the data signals (**DQ**) so that the transition occurs exactly in the middle of the WCK clock pulse. This ensures the DRAM has the maximum "eye opening" to latch the bit correctly.
* **The Burst:** For a 64B cache line, perform a **Burst Length of 32 (BL32)** over a 16-bit wide channel.

### 4. Internal Write Operations (DRAM Side)

Once the bits reach the DRAM's pins, the controller must respect the internal physics of the chip:

* **Write Latency (WL):** Wait a predefined number of cycles after the command before you start the data burst.
* **The "Write Recovery" Time (t_WR):** This is the most important timing constraint for a write. After the last bit of the 64B burst is received, we **cannot** close the row immediately. We must wait for t_WR to allow the local wordline voltages to stabilize and "drain" the charge into the microscopic capacitors.

### 5. The "Write X" Optimization (LPDDR5 Special)

If the 64B cache line the CPU is writing happens to be all zeros (a common occurrence when initializing memory), LPDDR5 allows a "short-cut":

* **Write-X Command:** Instead of toggling the DQ pins 512 times, send a single specific command that tells the DRAM to fill the entire addressed block with a specific pattern (usually all zeros).
* **Power Saved:** This reduces the I/O power consumption to near zero for that transaction.

---

### Summary: Read vs. Write Logic

| Feature | Read Transaction | Write Transaction |
| --- | --- | --- |
| **Clocking** | DRAM drives RDQS (Edge-aligned) | **Controller drives WCK (Center-aligned)** |
| **ECC** | Controller **Checks** ECC | **Controller **Generates** ECC |
| **Critical Delay** | t_RCD (Wait for sense amps) | ** t_WR (Wait for capacitor to charge)** |
| **Data Flow** | DRAM -> Controller | **Controller -> DRAM** |

### 6. Closing the Loop

Once t_WR has passed, you can either:

1. **Keep the row open** if there are more writes coming to the same page.
2. **Issue a `PRECHARGE**` to close the row, prepping it for a different address.

## Bank Grouping

In LPDDR5, a 64-byte write is not just a single "push" of data; it is a masterclass in timing and electrical coordination. 
The biggest challenge for the memory controller design is the **Write Recovery Time (t_WR)**—the physics-based delay 
we must wait for the DRAM's tiny capacitors to actually hold the charge we applied.

To overcome this delay, we use **Bank Groups**.
Bank Grouping allows overlapping writes to hide the t_WR latency.

---

### 1. Bank Grouping: The Latency "Vanishing Act"

Think of a Bank Group (BG) like a dedicated team of sense amplifiers and local data paths. 
LPDDR5 typically uses a **4 Bank Group** architecture (each containing 4 banks).

* **The Conflict (t_CCD_L):** If we write 32 bytes to Bank 0 and then immediately try to write the next 32 bytes to Bank 1 *within the same group*, we would hit a bottleneck. The local hardware is still busy "recovering" from the first write. We have to wait for a long delay (t_CCD_L).
* **The Solution (t_CCD_S):** Instead, you interleave! You send the first 32B to **Bank Group 0** and the second 32B to **Bank Group 1**. Because these groups have independent local wiring, the delay (t_CCD_S) is much shorter. 
We effectively "hide" the write recovery time of the first group by working on the second one.

---

### 2. The Link ECC "Shield"

When we write a 64-byte cache line, we are sending 512 bits of data. In LPDDR5, we calculate and attach **Link ECC** (Error Correction Code) parity bits for every 16 or 32 bits of data.

1. **Generation:** Our internal ECC engine generates 2 bits of parity for every 16-bit word.
2. **Transmission:** These parity bits are sent alongside the data on the **DMI** (Data Mask Inversion) pins.
3. **On-Die ECC vs. Link ECC:** * **Link ECC** protects the data while it's traveling on the PCB wires (from the MC to the DRAM).
* **On-Die ECC** (inside the DRAM) protects the data while it sits in the storage cells.
As the controller, we are the primary architect of the Link ECC "shield."

---

### 3. The WCK2CK Handshake (Sync Phase)

Because LPDDR5 separates the Command Clock (**CK**) and the Write Clock (**WCK**), we must perform a "handshake" before the write burst begins.

* **Step A:** Send a `CAS` (Column Address Strobe) command on the slow CK bus.
* **Step B:** "Start" the fast WCK clock and wait for it to stabilize (t_WCKENL).
* **Step C:** "Phase Align" the data. You don't just send the bits; you shift them in time so they arrive at the DRAM exactly when the WCK is at its peak voltage. This is called **Write Training**.

---

### 4. Advanced Write Efficiency: "Write X"

If the 64-byte line from the device is all zeros (e.g., during a memory clear), we don't actually have to toggle the DQ pins 512 times.

* **The Command:** Issue a `Write-X` command.
* **The Result:** Send the address, and the DRAM chip internally fills that entire 64B block with the "X" pattern (usually all 0s).
* **The Benefit:** This saves massive amounts of switching power on the data bus and reduces the heat generated by your controller.

---

### Summary Table: Your Strategy as a Controller

| Feature | Your Action | Goal |
| --- | --- | --- |
| **Interleaving** | Write to different **Bank Groups** | Hide t_WR and t_CCD_L latency. |
| **ECC** | Generate parity bits for every burst | Protect data from electrical noise on the PCB. |
| **WCK Sync** | Toggle WCK only when needed | Drastically reduce idle power consumption. |
| **Write-X** | Use the specialized "Zero-fill" command | Eliminate I/O power for zeroed-out lines. |

