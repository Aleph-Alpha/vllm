import argparse 
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# ========== PLOT CONFIGURATION PARAMETERS ==========
# Borrowed and adapted from plotting_utils.py

# Annotation circle parameters
CIRCLE_RADIUS = 16                    # Radius of the main annotation circle (pixels)
OUTER_CIRCLE_MULTIPLIER = 1.5         # Multiplier for outer circle radius (1.5 = 50% larger)
NUM_MAIN_POSITIONS = 4                # Number of annotation positions in main circle (2 series = HAT, Llama)
START_ANGLE_OFFSET = -np.pi / 4       # Starting angle for first annotation

# Annotation positioning
VERTICAL_LIFT_ABOVE_MAX = 10          # Pixels to lift annotation center above highest data point
ANNOTATION_FONT_SIZE = 9              # Font size for annotation text

# Plot margins
ANNOTATION_BUFFER = 10                # Extra pixels around annotation bounding box

# Data point styling
MARKER_SIZE = 8                       # Size of data point markers
LINE_WIDTH = 1.0                      # Width of connecting lines between points

# Arrow/connection styling
ARROW_ALPHA = 0.3                     # Transparency of arrows connecting annotations to data points
ARROW_LINE_WIDTH = 0.5                # Width of connection arrows

# Plot dimensions
PLOT_WIDTH = 12                       # Width of each subplot
PLOT_HEIGHT_PER_SUBPLOT = 7           # Height per subplot
DPI = 500                             # Resolution for saved plots

# ===================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a graph of speed vs batch size"
    )
    parser.add_argument(
        "--hat-json",
        type=str,
        help="Name of the HAT JSON file",
        default="",
    )
    parser.add_argument(
        "--llama-json",
        type=str,
        help="Name of the Llama JSON file",
        default="",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix to add to the output file name",
        default="",
    )
    return parser.parse_args()

def get_json_files(hat_json_file: str, llama_json_file: str) -> tuple[dict, dict]:
    with open(hat_json_file, "r") as f:
        hat_json = json.load(f)
        
    with open(llama_json_file, "r") as f:
        llama_json = json.load(f)
    
    hat_json = hat_json[next(iter(hat_json.keys()))]
    llama_json = llama_json[next(iter(llama_json.keys()))]
    
    return hat_json, llama_json

def create_comparison_plot(ax, title, y_label, batch_sizes, hat_data, hat_errors, llama_data, llama_errors, x_label='Batch Size (Log2 Scale)'):
    """
    Create a comparison plot between HAT and Llama with error bars and annotations.
    Adapted from plotting_utils.py create_benchmark_plot function.
    """
    # Colors and markers for HAT and Llama
    colors = ['#1f77b4', '#ff7f0e']  # Blue for HAT, Orange for Llama
    markers = ['o', 's']  # Circle for HAT, Square for Llama
    labels = ['HAT', 'Llama']
    
    # Plot data with error bars
    ax.errorbar(batch_sizes, hat_data, yerr=hat_errors, marker=markers[0], linestyle='-', 
                color=colors[0], label=labels[0], markersize=MARKER_SIZE, linewidth=LINE_WIDTH, capsize=5)
    ax.errorbar(batch_sizes, llama_data, yerr=llama_errors, marker=markers[1], linestyle='-', 
                color=colors[1], label=labels[1], markersize=MARKER_SIZE, linewidth=LINE_WIDTH, capsize=5)
    
    # Create annotation positions around circles
    offset_positions = []
    for i in range(NUM_MAIN_POSITIONS):
        angle = 2 * np.pi * i / NUM_MAIN_POSITIONS + START_ANGLE_OFFSET
        x_offset = CIRCLE_RADIUS * np.sin(angle)
        y_offset = CIRCLE_RADIUS * np.cos(angle)
        offset_positions.append((x_offset, y_offset))
    
    # Add annotations for each batch size
    all_data = [hat_data, llama_data]
    
    for batch_size in batch_sizes:
        batch_idx = batch_sizes.index(batch_size)
        
        # Find the maximum value at this batch size for annotation positioning
        values_at_batch = [hat_data[batch_idx], llama_data[batch_idx]]
        max_value_at_batch = max(values_at_batch)
        
        # Add annotations for both HAT and Llama
        for series_idx, (data_series, color, label) in enumerate(zip(all_data, colors, labels)):
            value = data_series[batch_idx]
            
            # Get offset position for this series
            x_offset, y_offset = offset_positions[series_idx % len(offset_positions)]
            
            # Determine horizontal alignment based on x_offset
            if x_offset > 0:
                ha = 'left'
            elif x_offset < 0:
                ha = 'right'
            else:
                ha = 'center'
            
            # Create the annotation
            ax.annotate(f'{value:.0f}',
                        (batch_size, max_value_at_batch),
                        xytext=(x_offset, y_offset + VERTICAL_LIFT_ABOVE_MAX),
                        xycoords='data',
                        textcoords='offset points',
                        ha=ha,
                        color=color,
                        fontweight='bold',
                        fontsize=ANNOTATION_FONT_SIZE,
                        arrowprops=dict(arrowstyle='-', color=color, alpha=ARROW_ALPHA, 
                                      lw=ARROW_LINE_WIDTH, connectionstyle="arc3,rad=0.1"))
    
    # Set up axes and formatting
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_xscale('log', base=2)  # Use log2 scale
    
    # Set x-axis ticks to show actual batch size values instead of powers
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])

    ax.set_ylabel(y_label, fontsize=14)
    ax.set_yscale('log')
    
    # Set up y-axis ticks
    all_y_values = hat_data + llama_data
    if all_y_values:
        min_y = min(all_y_values)
        max_y = max(all_y_values)
        
        # Create log-spaced ticks that cover the data range
        log_min = int(np.floor(np.log10(min_y)))
        log_max = int(np.ceil(np.log10(max_y)))
        
        # Generate ticks at powers of 10 and their multiples
        yticks = []
        for power in range(log_min, log_max + 1):
            base = 10 ** power
            yticks.extend([base, 2*base, 5*base])
        
        # Filter ticks to reasonable range around our data
        yticks = [tick for tick in yticks if min_y/2 <= tick <= max_y*2]
        yticks = sorted(list(set(yticks)))
        
        if yticks:
            ax.set_yticks(yticks)
    
    # Format y-axis labels
    from matplotlib.ticker import StrMethodFormatter, NullFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    # Add grid and legend
    ax.grid(True, which="major", ls="--", c='0.7', alpha=0.5)
    ax.legend(loc='best')

def create_hat_vs_llama_plots(batch_sizes, hat_bytes_per_second, hat_bytes_per_second_std_error,
                             llama_bytes_per_second, llama_bytes_per_second_std_error,
                             hat_bytes_per_second_per_request, hat_bytes_per_second_per_request_std_error,
                             llama_bytes_per_second_per_request, llama_bytes_per_second_per_request_std_error,
                             filename="hat_vs_llama_comparison.png",
                             suffix=""):
    """
    Create two vertically arranged plots comparing HAT vs Llama performance.
    """
    fig, axs = plt.subplots(2, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT_PER_SUBPLOT * 2))
    
    # Top plot: Bytes per Second vs Batch Size
    title_top_plot = 'HAT vs Llama: Bytes per Second vs. Batch Size'
    if suffix:
        title_top_plot += f" | {suffix}"
    create_comparison_plot(
        ax=axs[0],
        title=title_top_plot,
        y_label='Bytes/s (Log Scale)',
        batch_sizes=batch_sizes,
        hat_data=hat_bytes_per_second,
        hat_errors=hat_bytes_per_second_std_error,
        llama_data=llama_bytes_per_second,
        llama_errors=llama_bytes_per_second_std_error
    )
    
    # Bottom plot: Bytes per Second per Request vs Batch Size  
    title_bottom_plot = 'HAT vs Llama: Bytes/s per Request vs. Batch Size'
    if suffix:
        title_bottom_plot += f" | {suffix}"
    create_comparison_plot(
        ax=axs[1],
        title=title_bottom_plot,
        y_label='Bytes/s per Request (Log Scale)',
        batch_sizes=batch_sizes,
        hat_data=hat_bytes_per_second_per_request,
        hat_errors=hat_bytes_per_second_per_request_std_error,
        llama_data=llama_bytes_per_second_per_request,
        llama_errors=llama_bytes_per_second_per_request_std_error
    )
    
    plt.tight_layout()
    try:
        os.makedirs("comparison", exist_ok=True)
        plt.savefig(f"comparison/{filename}", dpi=DPI)
        print(f"\nComparison plots saved to {filename}")
        plt.show()
    except Exception as e:
        print(f"\nError saving plots: {e}")

def main() -> None:
    args = parse_args()
    hat_file_path = args.hat_json
    llama_file_path = args.llama_json

    hat_json, llama_json = get_json_files(hat_file_path, llama_file_path)
    
    # Create lists to store the data
    batch_sizes = []
    
    hat_bytes_per_second = []
    hat_bytes_per_second_std_error = []
    llama_bytes_per_second = []
    llama_bytes_per_second_std_error = []
    
    hat_bytes_per_second_per_request = []
    hat_bytes_per_second_per_request_std_error = []
    llama_bytes_per_second_per_request = []
    llama_bytes_per_second_per_request_std_error = []
    
    for (batch_size_hat, data_hat), (batch_size_llama, data_llama) in zip(hat_json.items(), llama_json.items()):
        if batch_size_hat != batch_size_llama:
            raise ValueError(f"Batch size mismatch: {batch_size_hat} != {batch_size_llama}")
        
        # batch_size_hat = "batch_size_x" Want to just get the x value
        batch_size = int(batch_size_hat.split("_")[-1])
        batch_sizes.append(batch_size)
        
        hat_bytes_per_second.append(data_hat["bytes_per_second"])
        hat_bytes_per_second_std_error.append(data_hat["bytes_per_second_std_error"])
        llama_bytes_per_second.append(data_llama["bytes_per_second"])
        llama_bytes_per_second_std_error.append(data_llama["bytes_per_second_std_error"])
        
        hat_bytes_per_second_per_request.append(data_hat["bytes_per_second_per_request"])
        hat_bytes_per_second_per_request_std_error.append(data_hat["bytes_per_second_per_request_std_error"])
        llama_bytes_per_second_per_request.append(data_llama["bytes_per_second_per_request"])
        llama_bytes_per_second_per_request_std_error.append(data_llama["bytes_per_second_per_request_std_error"])

    # Create the comparison plots
    create_hat_vs_llama_plots(
        batch_sizes=batch_sizes,
        hat_bytes_per_second=hat_bytes_per_second,
        hat_bytes_per_second_std_error=hat_bytes_per_second_std_error,
        llama_bytes_per_second=llama_bytes_per_second,
        llama_bytes_per_second_std_error=llama_bytes_per_second_std_error,
        hat_bytes_per_second_per_request=hat_bytes_per_second_per_request,
        hat_bytes_per_second_per_request_std_error=hat_bytes_per_second_per_request_std_error,
        llama_bytes_per_second_per_request=llama_bytes_per_second_per_request,
        llama_bytes_per_second_per_request_std_error=llama_bytes_per_second_per_request_std_error,
        suffix=args.suffix
    )

if __name__ == "__main__":
    main()