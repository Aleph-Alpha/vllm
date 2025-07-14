from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, StrMethodFormatter, FuncFormatter
import numpy as np
import json
import os
import glob
import argparse
from enum import Enum
import re
import math


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        type=str,
        help="language to analyse",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size to analyse",
        required=True,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="suffix to add to the output files",
        default="",
    )
    return parser.parse_args()

def get_batch_size_from_filename(filename: str) -> int:
    """Extracts the batch size from the filename."""
    match = re.search(r'_maxbs(\d+)\D*\.json$', filename)
    if match:
        return int(match.group(1))
    return 0


def get_json_files_for_language_and_batch_size(language: str, batch_size: int) -> tuple[dict, dict]:
    # We now read jsons from directory and see if we have enough information
    # Find all json files that start with "benchmark_results_""
    hat_json_files_all = glob.glob(f"benchmark_results_hat_{language}_maxbs*.json")
    llama_json_files_all = glob.glob(f"benchmark_results_llama_{language}_maxbs*.json")
    
    hat_json_files = [f for f in hat_json_files_all if get_batch_size_from_filename(f) >= batch_size]
    llama_json_files = [f for f in llama_json_files_all if get_batch_size_from_filename(f) >= batch_size]
    
    assert len(hat_json_files) >= 1, f"We do not have any json for HAT for language {language} and batch size >= {batch_size}"
    assert len(llama_json_files) >= 1, f"We do not have any json for llama for language {language} and batch size >= {batch_size}"
    
    # We now select a file from the ones available
    hat_json_file = max(hat_json_files, key=get_batch_size_from_filename)
    llama_json_file = max(llama_json_files, key=get_batch_size_from_filename)
    
    print(f"Using HAT json file: {hat_json_file}")
    print(f"Using llama json file: {llama_json_file}")

    with open(hat_json_file, "r") as f:
        hat_json = json.load(f)
        
    with open(llama_json_file, "r") as f:
        llama_json = json.load(f)
    
    return hat_json, llama_json

if __name__ == "__main__":
    args = parse_args()
    language = args.language
    batch_size = args.batch_size
    print(f"Analyzing language: {language} with batch size: {batch_size}")
    
    hat_json, llama_json = get_json_files_for_language_and_batch_size(language, batch_size)
        
    # Get info for waterfall chart
    hat_bytes_per_s = hat_json[language][f"batch_size_{batch_size}"]["bytes_per_second"]
    llama_bytes_per_s = llama_json[language][f"batch_size_{batch_size}"]["bytes_per_second"]
    
    hat_bytes_per_s_se = hat_json[language][f"batch_size_{batch_size}"]["bytes_per_second_std_error"]
    llama_bytes_per_s_se = llama_json[language][f"batch_size_{batch_size}"]["bytes_per_second_std_error"]
    
    hat_compression = hat_json[language][f"batch_size_{batch_size}"]["compression_bytes_per_token"]
    llama_compression = llama_json[language][f"batch_size_{batch_size}"]["compression_bytes_per_token"]
    compression_gain = hat_compression / llama_compression - 1
    
    llama_bytes_per_s_gain_CR_hat_matched = llama_bytes_per_s * compression_gain
    # We assume the relative standard error is the same
    llama_bytes_per_s_gain_CR_hat_matched_se = (llama_bytes_per_s_se / llama_bytes_per_s) * llama_bytes_per_s_gain_CR_hat_matched if llama_bytes_per_s > 0 else 0
    
    # Get info for histograms
    hat_len_bytes_input_per_seq = hat_json[language][f"batch_size_{batch_size}"]["len_bytes_input_per_seq"]
    llama_len_bytes_input_per_seq = llama_json[language][f"batch_size_{batch_size}"]["len_bytes_input_per_seq"]
    assert len(hat_len_bytes_input_per_seq) == len(llama_len_bytes_input_per_seq), "Input lengths are not the same"
    len_bytes_input_per_seq = hat_len_bytes_input_per_seq
    len_bytes_input_per_seq = [item for sublist in len_bytes_input_per_seq for item in sublist]
    
    hat_len_bytes_generation_per_seq = hat_json[language][f"batch_size_{batch_size}"]["len_bytes_generation_per_seq"]
    hat_len_bytes_generation_per_seq = [item for sublist in hat_len_bytes_generation_per_seq for item in sublist]
    
    llama_len_bytes_generation_per_seq = llama_json[language][f"batch_size_{batch_size}"]["len_bytes_generation_per_seq"]
    llama_len_bytes_generation_per_seq = [item for sublist in llama_len_bytes_generation_per_seq for item in sublist]
    
    runs_averaged_over = hat_json[language][f"batch_size_{batch_size}"]["runs_averaged_over"]
    
    categories = ['Llama', 'Llama gain at HAT\nmatched compression', 'HAT']

    # Values for the waterfall part.
    waterfall_values = [llama_bytes_per_s, llama_bytes_per_s_gain_CR_hat_matched]  # Initial value and one increase

    # Value for the separate regular bar.
    independent_value = hat_bytes_per_s

    # --- Chart Creation ---
    fig, ax = plt.subplots()

    # --- Waterfall portion ---

    # 1. Plot the initial bar of the waterfall.
    ax.bar(
        categories[0],
        waterfall_values[0],
        yerr=llama_bytes_per_s_se,
        capsize=5 if llama_bytes_per_s_se > 0 else 0,
        color='teal'
    )
    ax.text(-0.4, waterfall_values[0], f' {waterfall_values[0]:.0f}', ha='left', va='bottom')

    # 2. Plot the 'Growth' bar.
    # The bottom of this bar is the top of the initial bar.
    ax.bar(
        categories[1],
        waterfall_values[1],
        bottom=waterfall_values[0], # Start from the top of the 'Initial' bar
        yerr=llama_bytes_per_s_gain_CR_hat_matched_se,
        capsize=5 if llama_bytes_per_s_gain_CR_hat_matched_se > 0 else 0,
        color='lightseagreen'
    )
    ax.text(0.6, waterfall_values[0] + waterfall_values[1], f' {waterfall_values[1]:.0f}', ha='left', va='bottom')

    # --- Regular bar portion ---
    # 3. Plot the independent regular bar.
    ax.bar(
        categories[2],
        independent_value,
        yerr=hat_bytes_per_s_se,
        capsize=5 if hat_bytes_per_s_se > 0 else 0,
        color='darkgoldenrod'
    )
    ax.text(1.6, independent_value, f' {independent_value:.0f}', ha='left', va='bottom')


    # --- Formatting ---
    ax.set_title(f'Inference Performance | {language.capitalize()} Dataset - Batch Size {batch_size}')
    ax.set_ylabel('Performance [bytes/s]')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout() # Adjust layout to make room for labels
    os.makedirs('waterfall', exist_ok=True)
    plt.savefig(f'waterfall/waterfall_{language}_dataset_bs_{batch_size}_{args.suffix}.png', dpi=300, bbox_inches="tight")
    
    def plot_hist_with_fallback(ax, data, x_max, normalize=False, **kwargs):
        """
        Plots a histogram with a dynamic x-axis range (0 to x_max) and bin width (20).
        """
        if not data:
            return  # Don't plot if there's no data

        bins = np.arange(0, x_max + 1, 20)
        ax.hist(data, bins=bins, density=normalize, **kwargs)
        ax.set_xlim(0, x_max)
    
    # --- Histogram Creation ---

    should_normalize = runs_averaged_over > 1

    # Determine the dynamic range for the x-axis based on the max value across all data.
    all_lengths = len_bytes_input_per_seq + hat_len_bytes_generation_per_seq + llama_len_bytes_generation_per_seq
    if not all_lengths:
        overall_max_val = 0
    else:
        overall_max_val = max(all_lengths)

    # Round up to the next 100 for a clean axis limit, with a minimum of 200.
    x_axis_max = max(200, math.ceil((overall_max_val+1) / 100) * 100)

    # A figure with 3 subplots, arranged horizontally
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True) # 1 row, 3 columns
    
    title = f'Distribution of Sequence Lengths | {language.capitalize()} Dataset - Batch Size {batch_size}'
    if should_normalize:
        title = 'Normalized ' + title
    fig.suptitle(title, fontsize=16)

    # Histogram for Input Lengths
    plot_hist_with_fallback(axes[0], len_bytes_input_per_seq, x_axis_max, normalize=should_normalize, color='skyblue', edgecolor='black')
    axes[0].set_title('Input Lengths (Bytes)')
    axes[0].set_xlabel('Length (bytes)')
    
    ylabel = 'Frequency'
    if should_normalize:
        ylabel = 'Probability Density'
    axes[0].set_ylabel(ylabel)

    # Histogram for HAT Generation Lengths
    plot_hist_with_fallback(axes[1], hat_len_bytes_generation_per_seq, x_axis_max, normalize=should_normalize, color='darkgoldenrod', edgecolor='black')
    axes[1].set_title('HAT Generation Lengths (Bytes)')
    axes[1].set_xlabel('Length (bytes)')

    # Histogram for Llama Generation Lengths
    plot_hist_with_fallback(axes[2], llama_len_bytes_generation_per_seq, x_axis_max, normalize=should_normalize, color='teal', edgecolor='black')
    axes[2].set_title('Llama Generation Lengths (Bytes)')
    axes[2].set_xlabel('Length (bytes)')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust layout to make room for suptitle
    os.makedirs('histogram', exist_ok=True)
    plt.savefig(f'histogram/histograms_{language}_dataset_bs_{batch_size}_{args.suffix}.png', dpi=300, bbox_inches="tight")