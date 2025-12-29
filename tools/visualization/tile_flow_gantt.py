#!/usr/bin/env python3
"""
Tile Flow Gantt Chart Visualization

Generates a Gantt chart showing tile movements across L3 tiles over time.
This helps visualize the "hop-and-forward" data flow pattern in the KPU simulator.

Usage:
    python3 tile_flow_gantt.py tile_flow_trace.csv [output.png]
    python3 tile_flow_gantt.py tile_flow_trace.csv --interactive
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install with: pip3 install pandas matplotlib numpy")
    sys.exit(1)


# Color scheme for different event types
COLORS = {
    'L3_SEND_START': '#4CAF50',      # Green - L3 to L3 transfer
    'L3_RECEIVE': '#8BC34A',          # Light green - receive
    'L3_TO_L2_START': '#2196F3',      # Blue - push to L2
    'L3_TO_L2_COMPLETE': '#64B5F6',   # Light blue
    'BARRIER_START': '#FF9800',       # Orange - barrier
    'BARRIER_COMPLETE': '#FFB74D',    # Light orange
    'DMA_LOAD_START': '#9C27B0',      # Purple - DMA
    'DMA_STORE_START': '#E91E63',     # Pink - DMA store
}

# Tensor type names
TENSOR_NAMES = {0: 'A', 1: 'B', 2: 'C', 3: 'Bias'}


def load_trace(filename: str) -> pd.DataFrame:
    """Load trace CSV file into a DataFrame."""
    df = pd.read_csv(filename)

    # Add derived columns
    df['tensor_name'] = df['tensor'].map(lambda x: TENSOR_NAMES.get(x, '?'))
    df['tile_label'] = df.apply(
        lambda r: f"{r['tensor_name']}[{r['m_tile']},{r['n_tile']},{r['k_tile']}]",
        axis=1
    )

    return df


def compute_event_spans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert point events into spans with start/end times for Gantt chart.

    For now, we assign a fixed duration to each event type.
    In the future, this could be computed from paired start/complete events.
    """
    # Estimate duration based on bytes and assumed bandwidth
    # 64 bytes/cycle for L3 transfers
    BANDWIDTH = 64

    spans = []
    for _, row in df.iterrows():
        start = row['cycle']
        event_type = row['event_type']

        # Calculate duration based on event type
        if event_type in ['L3_SEND_START', 'L3_TO_L2_START']:
            duration = max(1, row['bytes'] // BANDWIDTH)
        elif event_type == 'BARRIER_START':
            duration = 100  # Visual indicator
        else:
            duration = 10  # Default small duration

        spans.append({
            'l3_id': row['l3_id'],
            'start': start,
            'end': start + duration,
            'duration': duration,
            'event_type': event_type,
            'tile_label': row['tile_label'],
            'tensor': row['tensor'],
            'src_l3': row['src_l3'],
            'dst_l3': row['dst_l3'],
            'bytes': row['bytes'],
        })

    return pd.DataFrame(spans)


def plot_gantt(df: pd.DataFrame, output_file: str = None,
               title: str = "Tile Flow Timeline", show: bool = True):
    """
    Create a Gantt chart visualization of tile movements.

    Parameters:
        df: DataFrame with event data
        output_file: Path to save the plot (optional)
        title: Chart title
        show: Whether to display the plot interactively
    """
    spans = compute_event_spans(df)

    # Get unique L3 tiles and sort them
    l3_ids = sorted(spans['l3_id'].unique())
    num_l3 = len(l3_ids)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(6, num_l3 * 0.5)))

    # Y-axis mapping: L3 ID to Y position
    y_positions = {l3_id: i for i, l3_id in enumerate(l3_ids)}

    # Plot each event as a horizontal bar
    for _, span in spans.iterrows():
        y = y_positions[span['l3_id']]
        color = COLORS.get(span['event_type'], '#888888')

        # Draw the bar
        rect = Rectangle(
            (span['start'], y - 0.35),
            span['duration'],
            0.7,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)

        # Add tile label for longer bars
        if span['duration'] > 50:
            ax.text(
                span['start'] + span['duration'] / 2,
                y,
                span['tile_label'],
                ha='center', va='center',
                fontsize=6,
                color='white'
            )

    # Configure axes
    ax.set_xlim(spans['start'].min() - 10, spans['end'].max() + 10)
    ax.set_ylim(-0.5, num_l3 - 0.5)

    ax.set_yticks(range(num_l3))
    ax.set_yticklabels([f'L3[{l3_id}]' for l3_id in l3_ids])

    ax.set_xlabel('Cycle')
    ax.set_ylabel('L3 Tile')
    ax.set_title(title)

    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.5, color='black', linewidth=0.5)
    for i in range(num_l3):
        ax.axhline(y=i + 0.5, color='gray', linewidth=0.5, alpha=0.3)

    # Create legend
    legend_patches = [
        mpatches.Patch(color=COLORS['L3_SEND_START'], label='L3→L3 Transfer'),
        mpatches.Patch(color=COLORS['L3_TO_L2_START'], label='L3→L2 Push'),
        mpatches.Patch(color=COLORS['BARRIER_START'], label='Barrier'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()

    # Save and/or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_file}")

    if show:
        plt.show()


def plot_l3_mesh(df: pd.DataFrame, output_file: str = None, show: bool = True):
    """
    Create a spatial view of the L3 mesh showing transfer patterns.
    """
    # Count transfers between L3 tiles
    l3_transfers = df[df['event_type'] == 'L3_SEND_START'].copy()

    # Assume 4x4 mesh
    rows, cols = 4, 4

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw L3 tiles as squares
    tile_size = 1.0
    for l3_id in range(rows * cols):
        row = l3_id // cols
        col = l3_id % cols

        # Count events for this tile
        tile_events = len(df[df['l3_id'] == l3_id])

        # Color intensity based on activity
        intensity = min(1.0, tile_events / 50)
        color = plt.cm.Blues(0.2 + intensity * 0.6)

        rect = Rectangle(
            (col * tile_size + 0.1, (rows - 1 - row) * tile_size + 0.1),
            tile_size - 0.2,
            tile_size - 0.2,
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)

        # Label
        ax.text(
            col * tile_size + tile_size / 2,
            (rows - 1 - row) * tile_size + tile_size / 2,
            f'L3[{l3_id}]\n{tile_events} events',
            ha='center', va='center',
            fontsize=9
        )

    # Draw transfer arrows
    transfer_counts = l3_transfers.groupby(['src_l3', 'dst_l3']).size()
    max_count = transfer_counts.max() if len(transfer_counts) > 0 else 1

    for (src, dst), count in transfer_counts.items():
        if src == dst:
            continue

        src_row, src_col = src // cols, src % cols
        dst_row, dst_col = dst // cols, dst % cols

        # Arrow from center of src to center of dst
        x1 = src_col * tile_size + tile_size / 2
        y1 = (rows - 1 - src_row) * tile_size + tile_size / 2
        x2 = dst_col * tile_size + tile_size / 2
        y2 = (rows - 1 - dst_row) * tile_size + tile_size / 2

        # Line width based on count
        width = 0.5 + 2.5 * (count / max_count)

        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                color='green',
                lw=width,
                alpha=0.7
            )
        )

    ax.set_xlim(-0.2, cols * tile_size + 0.2)
    ax.set_ylim(-0.2, rows * tile_size + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('L3 Mesh Transfer Pattern')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_file}")

    if show:
        plt.show()


def print_summary(df: pd.DataFrame):
    """Print a text summary of the trace data."""
    print("\n" + "="*60)
    print("TILE FLOW TRACE SUMMARY")
    print("="*60)

    print(f"\nTime Range: cycle {df['cycle'].min()} to {df['cycle'].max()}")
    print(f"Total Events: {len(df)}")

    print("\nEvents by Type:")
    for event_type, count in df['event_type'].value_counts().items():
        print(f"  {event_type}: {count}")

    print("\nEvents by L3 Tile:")
    for l3_id in sorted(df['l3_id'].unique()):
        count = len(df[df['l3_id'] == l3_id])
        print(f"  L3[{l3_id:2d}]: {count:4d} events")

    # Transfer analysis
    l3_transfers = df[df['event_type'] == 'L3_SEND_START']
    if len(l3_transfers) > 0:
        print("\nL3-to-L3 Transfer Patterns:")
        patterns = l3_transfers.groupby(['src_l3', 'dst_l3']).size()
        for (src, dst), count in patterns.items():
            print(f"  L3[{src}] -> L3[{dst}]: {count} transfers")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize tile flow trace data as a Gantt chart'
    )
    parser.add_argument('trace_file', help='Path to trace CSV file')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output image file (PNG, PDF, etc.)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Show interactive plot')
    parser.add_argument('--mesh', '-m', action='store_true',
                        help='Also show mesh transfer pattern')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='Print text summary of trace')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (only save)')

    args = parser.parse_args()

    # Check input file exists
    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"Error: Trace file not found: {args.trace_file}")
        sys.exit(1)

    # Load trace data
    print(f"Loading trace from: {args.trace_file}")
    df = load_trace(args.trace_file)
    print(f"Loaded {len(df)} events")

    # Print summary if requested
    if args.summary:
        print_summary(df)

    # Determine output file
    output_file = args.output
    if output_file is None and not args.interactive:
        output_file = trace_path.stem + '_gantt.png'

    # Generate Gantt chart
    show = args.interactive or (not args.no_show and output_file is None)
    plot_gantt(df, output_file=output_file, show=show)

    # Generate mesh view if requested
    if args.mesh:
        mesh_output = trace_path.stem + '_mesh.png' if output_file else None
        plot_l3_mesh(df, output_file=mesh_output, show=show)


if __name__ == '__main__':
    main()
