#!/usr/bin/env python3
"""
KPU Roofline Analysis and Visualization Tool

Part of v0.3.2 - Roofline Analysis for Benchmarking & Observability.

This tool generates roofline plots from benchmark results, showing:
- Peak compute ceiling
- Memory bandwidth ceilings (external, L3, L2)
- Achieved performance points
- Efficiency analysis

Usage:
    # Generate roofline from benchmark JSON
    python roofline_plot.py benchmark_results.json -o roofline.png

    # Run benchmarks and plot directly
    python roofline_plot.py --run-benchmark -o roofline.png

    # Interactive mode with custom hardware spec
    python roofline_plot.py results.json --peak-gflops 2048 --ext-bw 128
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class HardwareSpec:
    """Hardware specification for roofline model."""
    peak_gflops: float = 1024.0      # Peak compute (GFLOPS)
    external_bw: float = 64.0        # External memory bandwidth (GB/s)
    l3_bw: float = 128.0             # L3 bandwidth (GB/s)
    l2_bw: float = 256.0             # L2 bandwidth (GB/s)

    @property
    def ridge_external(self) -> float:
        """Ridge point for external memory."""
        return self.peak_gflops / self.external_bw

    @property
    def ridge_l3(self) -> float:
        """Ridge point for L3."""
        return self.peak_gflops / self.l3_bw

    @property
    def ridge_l2(self) -> float:
        """Ridge point for L2."""
        return self.peak_gflops / self.l2_bw

    def roofline_gflops(self, ai: float, level: str = "external") -> float:
        """Calculate roofline-predicted GFLOPS at given arithmetic intensity."""
        bw = {"external": self.external_bw, "l3": self.l3_bw, "l2": self.l2_bw}[level]
        return min(ai * bw, self.peak_gflops)


@dataclass
class BenchmarkPoint:
    """A single benchmark data point for roofline plotting."""
    name: str
    config: str
    arithmetic_intensity: float
    gflops: float
    efficiency: float
    bottleneck: str = ""

    @classmethod
    def from_json(cls, data: dict) -> "BenchmarkPoint":
        """Create from benchmark JSON result."""
        return cls(
            name=data.get("name", "unknown"),
            config=data.get("config", ""),
            arithmetic_intensity=data.get("memory", {}).get("arithmetic_intensity",
                                  data.get("arithmetic_intensity", 0)),
            gflops=data.get("compute", {}).get("gflops", data.get("gflops", 0)),
            efficiency=data.get("compute", {}).get("efficiency", data.get("efficiency", 0)),
            bottleneck=data.get("memory", {}).get("bottleneck", data.get("bottleneck", ""))
        )


@dataclass
class RooflineAnalysis:
    """Complete roofline analysis results."""
    hw: HardwareSpec
    points: list = field(default_factory=list)

    def add_point(self, point: BenchmarkPoint):
        self.points.append(point)

    def classify_points(self) -> dict:
        """Classify points by their bottleneck region."""
        regions = {"compute-bound": [], "memory-bound": [], "l3-bound": []}
        for p in self.points:
            if p.arithmetic_intensity >= self.hw.ridge_external:
                regions["compute-bound"].append(p)
            elif p.arithmetic_intensity >= self.hw.ridge_l3:
                regions["l3-bound"].append(p)
            else:
                regions["memory-bound"].append(p)
        return regions

    def summary(self) -> str:
        """Generate text summary of roofline analysis."""
        lines = [
            "=" * 70,
            "ROOFLINE ANALYSIS SUMMARY",
            "=" * 70,
            "",
            "Hardware Specification:",
            f"  Peak Compute:        {self.hw.peak_gflops:.1f} GFLOPS",
            f"  External Bandwidth:  {self.hw.external_bw:.1f} GB/s",
            f"  L3 Bandwidth:        {self.hw.l3_bw:.1f} GB/s",
            f"  L2 Bandwidth:        {self.hw.l2_bw:.1f} GB/s",
            "",
            "Ridge Points (FLOP/byte):",
            f"  External Memory:     {self.hw.ridge_external:.2f}",
            f"  L3 Cache:            {self.hw.ridge_l3:.2f}",
            f"  L2 Cache:            {self.hw.ridge_l2:.2f}",
            "",
            "-" * 70,
            f"{'Config':<25} {'AI':>8} {'GFLOPS':>10} {'Eff%':>8} {'Region':<15}",
            "-" * 70,
        ]

        for p in sorted(self.points, key=lambda x: x.arithmetic_intensity):
            region = "compute" if p.arithmetic_intensity >= self.hw.ridge_external else "memory"
            lines.append(
                f"{p.config:<25} {p.arithmetic_intensity:>8.2f} {p.gflops:>10.2f} "
                f"{p.efficiency*100:>7.1f}% {region:<15}"
            )

        lines.extend([
            "-" * 70,
            "",
            "Analysis:",
        ])

        regions = self.classify_points()
        lines.append(f"  Compute-bound points: {len(regions['compute-bound'])}")
        lines.append(f"  Memory-bound points:  {len(regions['memory-bound'])}")

        if regions["compute-bound"]:
            avg_eff = sum(p.efficiency for p in regions["compute-bound"]) / len(regions["compute-bound"])
            lines.append(f"  Avg compute-bound efficiency: {avg_eff*100:.1f}%")

        if regions["memory-bound"]:
            avg_eff = sum(p.efficiency for p in regions["memory-bound"]) / len(regions["memory-bound"])
            lines.append(f"  Avg memory-bound efficiency: {avg_eff*100:.1f}%")

        lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export roofline analysis as JSON."""
        data = {
            "version": "0.3.2",
            "hardware": {
                "peak_gflops": self.hw.peak_gflops,
                "external_bandwidth_gbs": self.hw.external_bw,
                "l3_bandwidth_gbs": self.hw.l3_bw,
                "l2_bandwidth_gbs": self.hw.l2_bw,
                "ridge_point_external": self.hw.ridge_external,
                "ridge_point_l3": self.hw.ridge_l3,
                "ridge_point_l2": self.hw.ridge_l2,
            },
            "points": [
                {
                    "name": p.name,
                    "config": p.config,
                    "arithmetic_intensity": p.arithmetic_intensity,
                    "gflops": p.gflops,
                    "efficiency": p.efficiency,
                    "predicted_gflops": self.hw.roofline_gflops(p.arithmetic_intensity),
                    "bottleneck": "compute" if p.arithmetic_intensity >= self.hw.ridge_external else "memory"
                }
                for p in self.points
            ]
        }
        return json.dumps(data, indent=2)


def plot_roofline(analysis: RooflineAnalysis, output_path: Optional[str] = None,
                  title: str = "KPU Roofline Model", show_levels: bool = True):
    """Generate roofline plot using matplotlib."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available. Install with: pip install matplotlib numpy")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(12, 8))

    hw = analysis.hw

    # X-axis range (arithmetic intensity)
    ai_min, ai_max = 0.5, 2048
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)

    # Plot roofline ceilings
    # External memory roofline
    ext_roof = np.minimum(ai_range * hw.external_bw, hw.peak_gflops)
    ax.plot(ai_range, ext_roof, 'b-', linewidth=2.5, label=f'External Memory ({hw.external_bw} GB/s)')

    if show_levels:
        # L3 roofline
        l3_roof = np.minimum(ai_range * hw.l3_bw, hw.peak_gflops)
        ax.plot(ai_range, l3_roof, 'g--', linewidth=1.5, alpha=0.7, label=f'L3 ({hw.l3_bw} GB/s)')

        # L2 roofline
        l2_roof = np.minimum(ai_range * hw.l2_bw, hw.peak_gflops)
        ax.plot(ai_range, l2_roof, 'r:', linewidth=1.5, alpha=0.7, label=f'L2 ({hw.l2_bw} GB/s)')

    # Plot peak compute line
    ax.axhline(y=hw.peak_gflops, color='b', linestyle='-', linewidth=2.5, alpha=0.3)
    ax.text(ai_max * 0.7, hw.peak_gflops * 1.05, f'Peak: {hw.peak_gflops} GFLOPS',
            fontsize=10, color='blue')

    # Plot ridge point
    ax.axvline(x=hw.ridge_external, color='gray', linestyle='--', alpha=0.5)
    ax.text(hw.ridge_external * 1.1, hw.peak_gflops * 0.1,
            f'Ridge: {hw.ridge_external:.1f}', fontsize=9, color='gray', rotation=90)

    # Plot benchmark points
    if analysis.points:
        ais = [p.arithmetic_intensity for p in analysis.points]
        gflops = [p.gflops for p in analysis.points]
        effs = [p.efficiency for p in analysis.points]

        # Color by efficiency
        scatter = ax.scatter(ais, gflops, c=effs, cmap='RdYlGn',
                            s=100, edgecolors='black', linewidths=0.5,
                            vmin=0, vmax=1, zorder=5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Efficiency')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

        # Label points
        for p in analysis.points:
            ax.annotate(p.config, (p.arithmetic_intensity, p.gflops),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.8)

    # Formatting
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(1, hw.peak_gflops * 2)

    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    # Add region labels
    ax.fill_between([ai_min, hw.ridge_external], [1, 1], [hw.peak_gflops*2, hw.peak_gflops*2],
                    alpha=0.1, color='red', label='Memory-bound')
    ax.fill_between([hw.ridge_external, ai_max], [1, 1], [hw.peak_gflops*2, hw.peak_gflops*2],
                    alpha=0.1, color='green', label='Compute-bound')

    ax.text(ai_min * 2, hw.peak_gflops * 0.5, 'Memory-Bound\nRegion',
            fontsize=10, alpha=0.5, ha='left')
    ax.text(ai_max * 0.3, hw.peak_gflops * 0.5, 'Compute-Bound\nRegion',
            fontsize=10, alpha=0.5, ha='right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved roofline plot to {output_path}")
    else:
        plt.show()

    plt.close()


def load_benchmark_results(path: str) -> list:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        data = json.load(f)

    points = []

    # Handle suite format
    if "results" in data:
        for r in data["results"]:
            points.append(BenchmarkPoint.from_json(r))
    # Handle single result format
    elif "name" in data:
        points.append(BenchmarkPoint.from_json(data))
    # Handle array format
    elif isinstance(data, list):
        for r in data:
            points.append(BenchmarkPoint.from_json(r))

    return points


def run_benchmark_cli() -> list:
    """Run the benchmark CLI and parse results."""
    benchmark_cli = Path(__file__).parent.parent.parent / "build" / "tools" / "benchmark" / "kpu-benchmark"

    if not benchmark_cli.exists():
        print(f"Error: Benchmark CLI not found at {benchmark_cli}")
        print("Build the project first: cmake --build build --target benchmark")
        sys.exit(1)

    result = subprocess.run(
        [str(benchmark_cli), "sweep", "--quick", "-f", "json"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        sys.exit(1)

    data = json.loads(result.stdout)
    return [BenchmarkPoint.from_json(r) for r in data.get("results", [])]


def main():
    parser = argparse.ArgumentParser(
        description="KPU Roofline Analysis and Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("input", nargs="?", help="Input JSON file with benchmark results")
    parser.add_argument("-o", "--output", help="Output image file (PNG, PDF, SVG)")
    parser.add_argument("--run-benchmark", action="store_true",
                       help="Run benchmarks instead of reading from file")
    parser.add_argument("--json", action="store_true",
                       help="Output roofline analysis as JSON")
    parser.add_argument("--summary", action="store_true",
                       help="Print text summary only (no plot)")

    # Hardware spec options
    parser.add_argument("--peak-gflops", type=float, default=1024.0,
                       help="Peak compute in GFLOPS (default: 1024)")
    parser.add_argument("--ext-bw", type=float, default=64.0,
                       help="External memory bandwidth in GB/s (default: 64)")
    parser.add_argument("--l3-bw", type=float, default=128.0,
                       help="L3 bandwidth in GB/s (default: 128)")
    parser.add_argument("--l2-bw", type=float, default=256.0,
                       help="L2 bandwidth in GB/s (default: 256)")

    # Plot options
    parser.add_argument("--title", default="KPU Roofline Model",
                       help="Plot title")
    parser.add_argument("--no-levels", action="store_true",
                       help="Don't show L3/L2 roofline levels")

    args = parser.parse_args()

    # Create hardware spec
    hw = HardwareSpec(
        peak_gflops=args.peak_gflops,
        external_bw=args.ext_bw,
        l3_bw=args.l3_bw,
        l2_bw=args.l2_bw
    )

    # Get benchmark points
    if args.run_benchmark:
        points = run_benchmark_cli()
    elif args.input:
        points = load_benchmark_results(args.input)
    else:
        parser.print_help()
        print("\nError: Provide input file or use --run-benchmark")
        sys.exit(1)

    # Create analysis
    analysis = RooflineAnalysis(hw=hw, points=points)

    # Output
    if args.json:
        print(analysis.to_json())
    elif args.summary:
        print(analysis.summary())
    else:
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, printing summary instead:")
            print(analysis.summary())
        else:
            plot_roofline(analysis, args.output, args.title, not args.no_levels)
            if not args.output:
                print(analysis.summary())


if __name__ == "__main__":
    main()
