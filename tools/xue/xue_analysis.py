#!/usr/bin/env python3
"""
XUE Operational Analysis Tool

Analyzes XUE event data to predict performance using the roofline model
and I/O complexity theory.

Usage:
    python xue_analysis.py events.json                    # Analyze events
    python xue_analysis.py events.json --validate sim.json # Validate vs simulation
    python xue_analysis.py --predict -M 1024 -N 1024 -K 1024  # Predict matmul

Version: 0.3.3
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class HardwareModel:
    """Hardware model for operational analysis."""
    peak_gflops: float = 1024.0
    clock_ghz: float = 1.0
    dram_bandwidth_gbs: float = 64.0
    l3_bandwidth_gbs: float = 128.0
    l2_bandwidth_gbs: float = 256.0
    l1_bandwidth_gbs: float = 512.0

    @property
    def ridge_point_dram(self) -> float:
        return self.peak_gflops / self.dram_bandwidth_gbs

    @property
    def ridge_point_l3(self) -> float:
        return self.peak_gflops / self.l3_bandwidth_gbs

    @property
    def ridge_point_l2(self) -> float:
        return self.peak_gflops / self.l2_bandwidth_gbs

    def predict_gflops(self, arithmetic_intensity: float) -> float:
        """Roofline model prediction."""
        mem_limited = arithmetic_intensity * self.dram_bandwidth_gbs
        return min(self.peak_gflops, mem_limited)

    def bottleneck(self, arithmetic_intensity: float) -> str:
        if arithmetic_intensity >= self.ridge_point_dram:
            return "compute"
        if arithmetic_intensity >= self.ridge_point_l3:
            return "dram"
        if arithmetic_intensity >= self.ridge_point_l2:
            return "l3"
        return "l2"


@dataclass
class OperationalResult:
    """Results of operational analysis."""
    total_flops: int = 0
    dram_bytes: int = 0
    l3_bytes: int = 0
    l2_bytes: int = 0
    l1_bytes: int = 0
    arithmetic_intensity: float = 0.0
    predicted_gflops: float = 0.0
    predicted_cycles: float = 0.0
    predicted_runtime_us: float = 0.0
    predicted_bottleneck: str = ""
    roofline_efficiency: float = 0.0
    event_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation against simulation."""
    prediction: OperationalResult
    actual_gflops: float = 0.0
    actual_cycles: float = 0.0
    gflops_error_percent: float = 0.0
    cycles_error_percent: float = 0.0
    within_10_percent: bool = False


def analyze_events(events: Dict[str, Any], hw: HardwareModel) -> OperationalResult:
    """Analyze XUE event data."""
    result = OperationalResult()

    # Extract summary if present
    if "summary" in events:
        result.total_flops = events["summary"].get("total_flops", 0)
        result.dram_bytes = events["summary"].get("dram_bytes", 0)
        result.arithmetic_intensity = events["summary"].get("arithmetic_intensity", 0.0)

    # Extract category data
    if "categories" in events:
        categories = events["categories"]

        # COMPUTE
        if "COMPUTE" in categories:
            compute = categories["COMPUTE"]
            result.total_flops = compute.get("total_flops", result.total_flops)
            result.event_counts["matmul"] = sum(
                v for k, v in compute.get("events", {}).items()
                if k.startswith("MATMUL")
            )
            result.event_counts["elementwise"] = sum(
                v for k, v in compute.get("events", {}).items()
                if k.startswith("ELEM")
            )
            result.event_counts["reduction"] = sum(
                v for k, v in compute.get("events", {}).items()
                if k.startswith("REDUCE")
            )

        # MEMORY
        if "MEMORY" in categories:
            memory = categories["MEMORY"]
            evts = memory.get("events", {})
            result.dram_bytes = memory.get("total_bytes", 0)  # Approximation

        # DATA_MOVEMENT
        if "DATA_MOVEMENT" in categories:
            dm = categories["DATA_MOVEMENT"]
            result.event_counts["data_movement"] = dm.get("total_events", 0)

        # SYNCHRONIZATION
        if "SYNCHRONIZATION" in categories:
            sync = categories["SYNCHRONIZATION"]
            result.event_counts["sync"] = sync.get("total_events", 0)

    # Calculate arithmetic intensity
    if result.dram_bytes > 0:
        result.arithmetic_intensity = result.total_flops / result.dram_bytes

    # Roofline prediction
    result.predicted_gflops = hw.predict_gflops(result.arithmetic_intensity)
    result.predicted_bottleneck = hw.bottleneck(result.arithmetic_intensity)

    # Predict timing
    if result.predicted_gflops > 0:
        seconds = result.total_flops / (result.predicted_gflops * 1e9)
        result.predicted_runtime_us = seconds * 1e6
        result.predicted_cycles = seconds * hw.clock_ghz * 1e9

    return result


def validate_prediction(events: Dict[str, Any], simulation: Dict[str, Any],
                       hw: HardwareModel) -> ValidationResult:
    """Validate operational analysis against simulation results."""
    prediction = analyze_events(events, hw)

    result = ValidationResult(prediction=prediction)

    # Extract simulation results
    if "compute" in simulation:
        result.actual_gflops = simulation["compute"].get("gflops", 0.0)
    if "cycles" in simulation:
        result.actual_cycles = simulation["cycles"]

    # Calculate errors
    if result.actual_gflops > 0:
        result.gflops_error_percent = 100.0 * abs(
            prediction.predicted_gflops - result.actual_gflops
        ) / result.actual_gflops

    if result.actual_cycles > 0:
        result.cycles_error_percent = 100.0 * abs(
            prediction.predicted_cycles - result.actual_cycles
        ) / result.actual_cycles

    result.within_10_percent = (
        result.gflops_error_percent <= 10.0 and
        result.cycles_error_percent <= 10.0
    )

    # Roofline efficiency
    if prediction.predicted_gflops > 0:
        prediction.roofline_efficiency = result.actual_gflops / prediction.predicted_gflops

    return result


def predict_matmul(M: int, N: int, K: int, hw: HardwareModel,
                   tile_size: int = 64) -> OperationalResult:
    """Predict performance for a matrix multiply workload."""
    result = OperationalResult()

    # FLOPs = 2*M*N*K
    result.total_flops = 2 * M * N * K

    # Memory traffic estimate (with tiling)
    # Minimum I/O per Hong-Kung: O(MNK / sqrt(M_fast))
    # Assuming L3 as fast memory (256KB = 64K floats)
    L3_elements = 64 * 1024
    sqrt_L3 = math.sqrt(L3_elements)

    # Estimated I/O (elements)
    io_elements = (M * N * K) / sqrt_L3

    # Add matrix sizes for initial load
    io_elements += M * K + K * N + M * N

    # Convert to bytes (float32)
    result.dram_bytes = int(io_elements * 4)

    # Arithmetic intensity
    result.arithmetic_intensity = result.total_flops / result.dram_bytes

    # Roofline prediction
    result.predicted_gflops = hw.predict_gflops(result.arithmetic_intensity)
    result.predicted_bottleneck = hw.bottleneck(result.arithmetic_intensity)

    # Timing
    if result.predicted_gflops > 0:
        seconds = result.total_flops / (result.predicted_gflops * 1e9)
        result.predicted_runtime_us = seconds * 1e6
        result.predicted_cycles = seconds * hw.clock_ghz * 1e9

    # Event count estimates
    systolic_size = 16
    tiles_m = (M + systolic_size - 1) // systolic_size
    tiles_n = (N + systolic_size - 1) // systolic_size
    tiles_k = (K + systolic_size - 1) // systolic_size

    result.event_counts = {
        "matmul_tiles": tiles_m * tiles_n * tiles_k,
        "accumulates": tiles_m * tiles_n * (tiles_k - 1),
        "writebacks": tiles_m * tiles_n,
    }

    return result


def format_result(result: OperationalResult, hw: HardwareModel) -> str:
    """Format result for display."""
    lines = [
        "=== XUE Operational Analysis (v0.3.3) ===",
        "",
        "Hardware Model:",
        f"  Peak Compute:        {hw.peak_gflops} GFLOPS",
        f"  DRAM Bandwidth:      {hw.dram_bandwidth_gbs} GB/s",
        f"  Ridge Point (DRAM):  {hw.ridge_point_dram:.1f} FLOP/byte",
        "",
        "Workload Characteristics:",
        f"  Total FLOPs:         {result.total_flops:,}",
        f"  DRAM Traffic:        {result.dram_bytes:,} bytes",
        f"  Arithmetic Intensity: {result.arithmetic_intensity:.2f} FLOP/byte",
        "",
        "Performance Prediction:",
        f"  Predicted GFLOPS:    {result.predicted_gflops:.1f}",
        f"  Predicted Cycles:    {result.predicted_cycles:,.0f}",
        f"  Predicted Runtime:   {result.predicted_runtime_us:.2f} us",
        f"  Bottleneck:          {result.predicted_bottleneck}",
    ]

    if result.event_counts:
        lines.extend([
            "",
            "Event Breakdown:",
        ])
        for name, count in result.event_counts.items():
            lines.append(f"  {name}: {count:,}")

    return "\n".join(lines)


def format_validation(result: ValidationResult, hw: HardwareModel) -> str:
    """Format validation result for display."""
    lines = format_result(result.prediction, hw).split("\n")
    lines.extend([
        "",
        "Validation Against Simulation:",
        f"  Actual GFLOPS:       {result.actual_gflops:.1f}",
        f"  Actual Cycles:       {result.actual_cycles:,.0f}",
        f"  GFLOPS Error:        {result.gflops_error_percent:.1f}%",
        f"  Cycles Error:        {result.cycles_error_percent:.1f}%",
        f"  Roofline Efficiency: {result.prediction.roofline_efficiency * 100:.1f}%",
        f"  Within 10% Target:   {'YES' if result.within_10_percent else 'NO'}",
    ])
    return "\n".join(lines)


def to_json(result: OperationalResult, hw: HardwareModel) -> Dict[str, Any]:
    """Convert result to JSON-serializable dict."""
    return {
        "version": "0.3.3",
        "hardware": {
            "peak_gflops": hw.peak_gflops,
            "dram_bandwidth_gbs": hw.dram_bandwidth_gbs,
            "ridge_point_dram": hw.ridge_point_dram,
            "ridge_point_l3": hw.ridge_point_l3,
            "ridge_point_l2": hw.ridge_point_l2,
        },
        "workload": {
            "total_flops": result.total_flops,
            "dram_bytes": result.dram_bytes,
            "arithmetic_intensity": result.arithmetic_intensity,
        },
        "prediction": {
            "gflops": result.predicted_gflops,
            "cycles": result.predicted_cycles,
            "runtime_us": result.predicted_runtime_us,
            "bottleneck": result.predicted_bottleneck,
        },
        "events": result.event_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="XUE Operational Analysis Tool (v0.3.3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s events.json                        Analyze event data
  %(prog)s events.json --validate sim.json    Validate against simulation
  %(prog)s --predict -M 1024 -N 1024 -K 1024  Predict matmul performance
  %(prog)s events.json -f json                Output as JSON
        """
    )

    parser.add_argument("events_file", nargs="?", help="XUE events JSON file")
    parser.add_argument("--validate", "-v", metavar="SIM",
                       help="Simulation results JSON for validation")
    parser.add_argument("--predict", "-p", action="store_true",
                       help="Predict matmul performance (use -M, -N, -K)")
    parser.add_argument("-M", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("-N", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("-K", type=int, default=1024, help="Matrix K dimension")
    parser.add_argument("--format", "-f", choices=["text", "json"],
                       default="text", help="Output format")
    parser.add_argument("--peak-gflops", type=float, default=1024.0,
                       help="Peak compute (GFLOPS)")
    parser.add_argument("--dram-bw", type=float, default=64.0,
                       help="DRAM bandwidth (GB/s)")

    args = parser.parse_args()

    # Create hardware model
    hw = HardwareModel(
        peak_gflops=args.peak_gflops,
        dram_bandwidth_gbs=args.dram_bw,
    )

    if args.predict:
        # Predict matmul performance
        result = predict_matmul(args.M, args.N, args.K, hw)
        if args.format == "json":
            print(json.dumps(to_json(result, hw), indent=2))
        else:
            print(format_result(result, hw))
    elif args.events_file:
        # Load events
        with open(args.events_file) as f:
            events = json.load(f)

        if args.validate:
            # Load simulation results
            with open(args.validate) as f:
                simulation = json.load(f)
            result = validate_prediction(events, simulation, hw)
            if args.format == "json":
                output = to_json(result.prediction, hw)
                output["validation"] = {
                    "actual_gflops": result.actual_gflops,
                    "actual_cycles": result.actual_cycles,
                    "gflops_error_percent": result.gflops_error_percent,
                    "cycles_error_percent": result.cycles_error_percent,
                    "within_10_percent": result.within_10_percent,
                }
                print(json.dumps(output, indent=2))
            else:
                print(format_validation(result, hw))
        else:
            # Analyze events only
            result = analyze_events(events, hw)
            if args.format == "json":
                print(json.dumps(to_json(result, hw), indent=2))
            else:
                print(format_result(result, hw))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
