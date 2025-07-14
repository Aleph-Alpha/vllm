import argparse
import json
import os
import re
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create markdown tables from benchmark results."
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))

    hat_file_path = args.hat_json
    llama_file_path = args.llama_json

    if not hat_file_path:
        print(
            f"Error: Could not find hat benchmark file for language '{args.language}'."
        )
        return
    if not llama_file_path:
        print(
            f"Error: Could not find llama benchmark file for language '{args.language}'."
        )
        return

    with open(hat_file_path, "r") as f:
        hat_data = json.load(f)

    with open(llama_file_path, "r") as f:
        llama_data = json.load(f)

    batch_key = f"batch_size_{args.batch_size}"

    try:
        hat_results = hat_data[args.language][batch_key]
    except KeyError:
        print(
            f"Error: No data for language '{args.language}' and batch size {args.batch_size} in {hat_file_path}"
        )
        return

    try:
        llama_results = llama_data[args.language][batch_key]
    except KeyError:
        print(
            f"Error: No data for language '{args.language}' and batch size {args.batch_size} in {llama_file_path}"
        )
        return

    ignored_keys = {
        "status",
        "len_bytes_input_per_seq",
        "len_bytes_generation_per_seq",
    }

    keys = [
        key for key in hat_results.keys() if key not in ignored_keys
    ]
    # Move time_spent_s to the front
    if "time_spent_s" in keys:
        keys.remove("time_spent_s")
        keys.insert(0, "time_spent_s")

    headers = ["Model"]
    for key in keys:
        header = key.replace("_", " ").title()
        if key == "time_spent_s":
            header = "Time Spent in sec"
        headers.append(header)

    hat_values = [hat_results[key] for key in keys]
    llama_values = [llama_results[key] for key in keys]

    def format_value(value: float | str) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    hat_row = ["HAT"] + [format_value(v) for v in hat_values]
    llama_row = ["LLaMA"] + [format_value(v) for v in llama_values]

    # Print Markdown table
    print(f"### Results for {args.language.title()} with batch size {args.batch_size}")
    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['---'] * len(headers))}|")
    print(f"| {' | '.join(hat_row)} |")
    print(f"| {' | '.join(llama_row)} |")


if __name__ == "__main__":
    main()