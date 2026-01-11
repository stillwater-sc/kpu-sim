# L3 Analysis Tools - Updated with Small Distributed L3 Support

## Changes Made

Added support for small distributed L3 tiles (1MB and 2MB) to both focused and comprehensive analysis tools.

### Motivation
For distributed L3 architectures where L3 is split across multiple tiles, each tile may have a smaller L3 capacity (e.g., 1-2MB per tile). Understanding overfetch behavior at these smaller sizes is critical for:
- Distributed L3 tile sizing decisions
- Per-tile L3 capacity optimization
- Understanding the lower bound on effective L3 size

### Updated L3 Size Ranges

**Focused Analysis** (`l3_focused_analysis`)
- Old: 3 sizes (16MB, 64MB, 256MB)
- New: 5 sizes (1MB, 2MB, 16MB, 64MB, 256MB)
- Total configs: 108 → 180 (12 workloads × 3 strategies × 5 sizes)
- Runtime: ~5 min → ~8 min (estimated)

**Comprehensive Analysis** (`l3_comprehensive_analysis`)
- Old: 5 sizes (4MB, 16MB, 64MB, 256MB, 1GB)
- New: 7 sizes (1MB, 2MB, 4MB, 16MB, 64MB, 256MB, 1GB)
- Total configs: 405 → 567 (27 workloads × 3 strategies × 7 sizes)
- Runtime: ~2-3 hrs → ~3-4 hrs (estimated)

## Running the Analyses

### Quick Focused Analysis (~8 minutes)
```bash
./examples/compiler/l3_focused_analysis
```
Output: `l3_focused_analysis.csv`

### Comprehensive Overnight Run (~3-4 hours)
```bash
./run_comprehensive_overnight.sh
```
Output: `l3_comprehensive_analysis.csv` + `l3_comprehensive_analysis.log`

Or run in background with nohup:
```bash
nohup ./run_comprehensive_overnight.sh &
```

## Expected Insights for Small L3 (1-2MB)

Based on the workload characteristics:

### Small Workloads (BERT, Small Square)
- **1-2MB L3**: Should work reasonably well (tensors fit)
- **Expected overfetch**: 1.5-3× with good strategy

### Medium Workloads (GPT-2, Medium Square)
- **1MB L3**: High overfetch (10-20×)
- **2MB L3**: Moderate overfetch (5-10×)
- **Strategy choice critical**

### Large Workloads (LLaMA, 32k×7k)
- **1-2MB L3**: Very high overfetch (50-200×)
- **Need WS strategy**: Will help but still significant overfetch
- **4MB+ L3**: Starts to become manageable

### Strategy Impact at Small L3
With 1-2MB L3, strategy selection becomes **even more critical**:
- **WS (Weight-Stationary)**: Best for Wide/Deep (large B)
- **IS (Input-Stationary)**: Best for Tall (large A)
- **OS (Output-Stationary)**: Best for small tensors only

**Key insight**: At small L3 sizes, wrong strategy can mean 10-50× more DRAM traffic!

## Files Modified

1. `examples/compiler/l3_focused_analysis.cpp` - Added 1MB, 2MB
2. `examples/compiler/l3_comprehensive_analysis.cpp` - Added 1MB, 2MB + strategy-aware fix
3. Both now properly use strategy-aware L2 scheduling

## Quick Check After Running

```bash
# Check focused results for 1MB L3
grep "L3_Size_MB.*,1," l3_focused_analysis.csv | awk -F',' '{print $1, $7, $9}'

# Check comprehensive results for 2MB L3
grep "L3_Size_MB.*,2," l3_comprehensive_analysis.csv | awk -F',' '{print $1, $7, $9}'

# Find worst case overfetch at 1MB
awk -F',' '$8==1 {print $1, $7, $9}' l3_focused_analysis.csv | sort -t',' -k3 -rn | head -5
```

## Recommendation

Run **focused analysis first** to get quick insights on 1-2MB L3 behavior. If results look promising or you need comprehensive data for a paper, run the overnight comprehensive analysis.

The focused analysis will tell you:
- Is 1-2MB L3 viable for your target workloads?
- Which strategy is best for each workload at small L3?
- Where is the knee of the curve for distributed L3 sizing?
