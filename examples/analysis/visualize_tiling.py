#!/usr/bin/env python3
"""
Visualize tiling configurations for KPU matmul scheduling.
Generates Pareto frontier and efficiency analysis plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data(csv_path: str) -> pd.DataFrame:
    """Load tiling configuration CSV."""
    df = pd.read_csv(csv_path)
    return df

def plot_efficiency_vs_memory(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: Memory per L3 vs Parallel Efficiency, colored by fits_in_l3."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Split by fits_in_l3
    fits = df[df['fits_in_l3'] == 1]
    no_fit = df[df['fits_in_l3'] == 0]

    # Plot non-fitting configurations (gray, smaller)
    ax.scatter(no_fit['total_per_l3_kb'], no_fit['parallel_efficiency'] * 100,
               c='lightgray', s=30, alpha=0.5, label='Does not fit in L3')

    # Plot fitting configurations (blue)
    scatter = ax.scatter(fits['total_per_l3_kb'], fits['parallel_efficiency'] * 100,
                        c='steelblue', s=60, alpha=0.7, label='Fits in L3')

    # Highlight Pareto optimal points
    pareto = df[df['pareto_optimal'] == 1]
    ax.scatter(pareto['total_per_l3_kb'], pareto['parallel_efficiency'] * 100,
               c='red', s=200, marker='*', edgecolors='darkred', linewidths=1.5,
               label='Pareto optimal', zorder=5)

    # Add labels for Pareto points
    for _, row in pareto.iterrows():
        label = f"{int(row['m_tiles'])}×{int(row['n_tiles'])}×{int(row['k_tiles'])}"
        ax.annotate(label,
                   (row['total_per_l3_kb'], row['parallel_efficiency'] * 100),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

    # Add L3 capacity line
    ax.axvline(x=512, color='red', linestyle='--', linewidth=2, alpha=0.7, label='L3 capacity (512KB)')

    ax.set_xlabel('Memory per L3 Tile (KB)', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title('Tiling Configuration: Efficiency vs Memory Footprint', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(df['total_per_l3_kb']) * 1.05)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_vs_memory.png', dpi=150)
    plt.savefig(output_dir / 'efficiency_vs_memory.svg')
    print(f"Saved: efficiency_vs_memory.png/svg")
    plt.close()

def plot_cycles_vs_efficiency(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: Total cycles vs Efficiency."""
    fig, ax = plt.subplots(figsize=(12, 8))

    fits = df[df['fits_in_l3'] == 1]
    no_fit = df[df['fits_in_l3'] == 0]

    # Plot non-fitting (gray)
    ax.scatter(no_fit['parallel_total_cycles'] / 1000, no_fit['parallel_efficiency'] * 100,
               c='lightgray', s=30, alpha=0.5, label='Does not fit in L3')

    # Plot fitting (colored by speedup)
    scatter = ax.scatter(fits['parallel_total_cycles'] / 1000, fits['parallel_efficiency'] * 100,
                        c=fits['speedup'], cmap='viridis', s=60, alpha=0.8,
                        label='Fits in L3')

    # Colorbar for speedup
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speedup (parallel/serial)', fontsize=11)

    # Highlight Pareto optimal
    pareto = df[df['pareto_optimal'] == 1]
    ax.scatter(pareto['parallel_total_cycles'] / 1000, pareto['parallel_efficiency'] * 100,
               c='red', s=200, marker='*', edgecolors='darkred', linewidths=1.5,
               label='Pareto optimal', zorder=5)

    for _, row in pareto.iterrows():
        label = f"{int(row['m_tiles'])}×{int(row['n_tiles'])}×{int(row['k_tiles'])}"
        ax.annotate(label,
                   (row['parallel_total_cycles'] / 1000, row['parallel_efficiency'] * 100),
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

    ax.set_xlabel('Parallel Execution Cycles (thousands)', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title('Tiling Configuration: Efficiency vs Execution Time', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'cycles_vs_efficiency.png', dpi=150)
    plt.savefig(output_dir / 'cycles_vs_efficiency.svg')
    print(f"Saved: cycles_vs_efficiency.png/svg")
    plt.close()

def plot_heatmap_mn(df: pd.DataFrame, output_dir: Path):
    """Heatmap: M_tiles × N_tiles, color by best parallel efficiency."""
    # Get best efficiency for each M×N combination (across all K values)
    fits = df[df['fits_in_l3'] == 1]

    if fits.empty:
        print("No configurations fit in L3, skipping heatmap")
        return

    # Aggregate: best efficiency for each M×N
    best_eff = fits.groupby(['m_tiles', 'n_tiles'])['parallel_efficiency'].max().reset_index()

    # Create pivot table
    m_vals = sorted(best_eff['m_tiles'].unique())
    n_vals = sorted(best_eff['n_tiles'].unique())

    heatmap_data = np.full((len(m_vals), len(n_vals)), np.nan)
    for _, row in best_eff.iterrows():
        i = m_vals.index(int(row['m_tiles']))
        j = n_vals.index(int(row['n_tiles']))
        heatmap_data[i, j] = row['parallel_efficiency'] * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Best Parallel Efficiency (%)', fontsize=11)

    # Set ticks
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels(n_vals)
    ax.set_yticks(range(len(m_vals)))
    ax.set_yticklabels(m_vals)

    ax.set_xlabel('N_tiles', fontsize=12)
    ax.set_ylabel('M_tiles', fontsize=12)
    ax.set_title('Best Efficiency by M×N Tiling (across all K values, fits in L3)', fontsize=14)

    # Add text annotations
    for i in range(len(m_vals)):
        for j in range(len(n_vals)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                       fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_mn_efficiency.png', dpi=150)
    plt.savefig(output_dir / 'heatmap_mn_efficiency.svg')
    print(f"Saved: heatmap_mn_efficiency.png/svg")
    plt.close()

def plot_k_sweep(df: pd.DataFrame, output_dir: Path):
    """Line plot: K_tiles vs efficiency for selected M×N configurations."""
    fits = df[df['fits_in_l3'] == 1]

    if fits.empty:
        print("No configurations fit in L3, skipping K sweep")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Select a few interesting M×N combinations
    mn_combos = fits.groupby(['m_tiles', 'n_tiles']).size().reset_index(name='count')
    mn_combos = mn_combos[mn_combos['count'] >= 3]  # Need at least 3 K values

    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(mn_combos))))

    for idx, (_, row) in enumerate(mn_combos.head(8).iterrows()):
        m, n = int(row['m_tiles']), int(row['n_tiles'])
        subset = fits[(fits['m_tiles'] == m) & (fits['n_tiles'] == n)].sort_values('k_tiles')

        ax.plot(subset['k_tiles'], subset['parallel_efficiency'] * 100,
                marker='o', linewidth=2, markersize=6,
                color=colors[idx], label=f'M={m}, N={n}')

    ax.set_xlabel('K_tiles (K-dimension subdivision)', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title('Effect of K-Tiling on Efficiency (for configurations fitting in L3)', fontsize=14)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'k_sweep_efficiency.png', dpi=150)
    plt.savefig(output_dir / 'k_sweep_efficiency.svg')
    print(f"Saved: k_sweep_efficiency.png/svg")
    plt.close()

def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path):
    """Dedicated Pareto frontier plot with detailed annotations."""
    fig, ax = plt.subplots(figsize=(14, 10))

    fits = df[df['fits_in_l3'] == 1].copy()

    # Plot all fitting configurations
    scatter = ax.scatter(fits['total_per_l3_kb'], fits['speedup'],
                        c=fits['parallel_efficiency'] * 100, cmap='viridis',
                        s=80, alpha=0.7, edgecolors='gray', linewidths=0.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Parallel Efficiency (%)', fontsize=11)

    # Highlight Pareto optimal
    pareto = df[df['pareto_optimal'] == 1]
    ax.scatter(pareto['total_per_l3_kb'], pareto['speedup'],
               c='red', s=300, marker='*', edgecolors='darkred', linewidths=2,
               label='Pareto optimal', zorder=5)

    # Connect Pareto points with line
    pareto_sorted = pareto.sort_values('total_per_l3_kb')
    ax.plot(pareto_sorted['total_per_l3_kb'], pareto_sorted['speedup'],
            'r--', linewidth=2, alpha=0.7)

    # Detailed annotations for Pareto points
    for _, row in pareto.iterrows():
        label = (f"{int(row['m_tiles'])}×{int(row['n_tiles'])}×{int(row['k_tiles'])}\n"
                f"Eff: {row['parallel_efficiency']*100:.0f}%\n"
                f"Cycles: {int(row['parallel_total_cycles']):,}")
        ax.annotate(label,
                   (row['total_per_l3_kb'], row['speedup']),
                   xytext=(15, 0), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

    # Add reference lines
    ax.axhline(y=16, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
               label='16× speedup (full tile parallelism)')

    ax.set_xlabel('Memory per L3 Tile (KB)', fontsize=12)
    ax.set_ylabel('Speedup (Parallel / Serial)', fontsize=12)
    ax.set_title('Pareto Frontier: Speedup vs Memory Footprint\n(Configurations fitting in 512KB L3)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png', dpi=150)
    plt.savefig(output_dir / 'pareto_frontier.svg')
    print(f"Saved: pareto_frontier.png/svg")
    plt.close()

def plot_speedup_vs_concurrency(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: Peak concurrency vs Speedup."""
    fig, ax = plt.subplots(figsize=(12, 8))

    fits = df[df['fits_in_l3'] == 1]
    no_fit = df[df['fits_in_l3'] == 0]

    ax.scatter(no_fit['parallel_peak_concurrency'], no_fit['speedup'],
               c='lightgray', s=30, alpha=0.5, label='Does not fit in L3')

    scatter = ax.scatter(fits['parallel_peak_concurrency'], fits['speedup'],
                        c=fits['parallel_efficiency'] * 100, cmap='viridis',
                        s=60, alpha=0.8, edgecolors='gray', linewidths=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Parallel Efficiency (%)', fontsize=11)

    # Highlight Pareto optimal
    pareto = df[df['pareto_optimal'] == 1]
    ax.scatter(pareto['parallel_peak_concurrency'], pareto['speedup'],
               c='red', s=200, marker='*', edgecolors='darkred', linewidths=1.5,
               label='Pareto optimal', zorder=5)

    # Reference: ideal linear speedup
    max_conc = df['parallel_peak_concurrency'].max()
    ax.plot([1, max_conc], [1, max_conc], 'g--', linewidth=2, alpha=0.5, label='Ideal linear speedup')

    ax.set_xlabel('Peak Concurrency (active compute tiles)', fontsize=12)
    ax.set_ylabel('Speedup (Parallel / Serial)', fontsize=12)
    ax.set_title('Speedup vs Peak Concurrency', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_vs_concurrency.png', dpi=150)
    plt.savefig(output_dir / 'speedup_vs_concurrency.svg')
    print(f"Saved: speedup_vs_concurrency.png/svg")
    plt.close()

def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("TILING CONFIGURATION ANALYSIS SUMMARY")
    print("="*70)

    fits = df[df['fits_in_l3'] == 1]
    pareto = df[df['pareto_optimal'] == 1]

    print(f"\nTotal configurations: {len(df)}")
    print(f"Fit in L3 (512KB):   {len(fits)}")
    print(f"Pareto optimal:      {len(pareto)}")

    if not fits.empty:
        print(f"\n--- Configurations Fitting in L3 ---")
        print(f"Efficiency range:  {fits['parallel_efficiency'].min()*100:.1f}% - {fits['parallel_efficiency'].max()*100:.1f}%")
        print(f"Speedup range:     {fits['speedup'].min():.2f}× - {fits['speedup'].max():.2f}×")
        print(f"Memory range:      {fits['total_per_l3_kb'].min():.0f}KB - {fits['total_per_l3_kb'].max():.0f}KB")

    print(f"\n--- Pareto Optimal Configurations ---")
    for _, row in pareto.iterrows():
        print(f"  {int(row['m_tiles'])}×{int(row['n_tiles'])}×{int(row['k_tiles'])}: "
              f"{row['total_per_l3_kb']:.0f}KB, "
              f"{row['parallel_efficiency']*100:.0f}% eff, "
              f"{row['speedup']:.1f}× speedup, "
              f"{int(row['parallel_total_cycles']):,} cycles")

    print("\n" + "="*70)

def main():
    import sys

    # Find CSV file
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    csv_candidates = [
        Path('tiling_configurations.csv'),
        repo_root / 'tiling_configurations.csv',
        script_dir / 'tiling_configurations.csv',
    ]

    csv_path = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        print("Error: tiling_configurations.csv not found")
        print("Run example_matmul_schedule_analysis first to generate the CSV")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = load_data(csv_path)

    # Output directory
    output_dir = script_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate all plots
    print("\nGenerating visualizations...")
    plot_efficiency_vs_memory(df, output_dir)
    plot_cycles_vs_efficiency(df, output_dir)
    plot_heatmap_mn(df, output_dir)
    plot_k_sweep(df, output_dir)
    plot_pareto_frontier(df, output_dir)
    plot_speedup_vs_concurrency(df, output_dir)

    # Print summary
    print_summary(df)

    print(f"\nAll plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
