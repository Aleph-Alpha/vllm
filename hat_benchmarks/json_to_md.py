import json
import sys

def create_markdown_table(data):
    """
    Creates a markdown table from the benchmark data.
    """
    # Assuming the first key is the language, e.g., 'english'
    lang_key = list(data.keys())[0]
    benchmark_data = data[lang_key]

    # Sort batch sizes numerically
    sorted_batch_keys = sorted(
        benchmark_data.keys(),
        key=lambda x: int(x.split('_')[-1])
    )

    # Define fields to ignore
    ignored_fields = {
        "status",
        "len_bytes_input_per_seq",
        "len_bytes_generation_per_seq"
    }

    # Determine headers from the first entry, excluding ignored fields
    if not sorted_batch_keys:
        return "No data to display."

    first_item = benchmark_data[sorted_batch_keys[0]]
    
    # Get the original headers, which are the keys from the JSON
    original_headers = [
        key for key in first_item.keys() if key not in ignored_fields
    ]

    # Manually order and format headers
    header_mapping = {
        "time_spent_s": "Time Spent in sec"
    }

    # Start with Batch Size and Time Spent
    display_headers = ["Batch Size", "Time Spent in sec"]
    data_keys_ordered = ["time_spent_s"]

    # Add the rest of the headers
    for h in original_headers:
        if h not in data_keys_ordered:
            display_headers.append(h.replace('_', ' ').title())
            data_keys_ordered.append(h)

    # Start creating the markdown table
    markdown = "| " + " | ".join(display_headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(display_headers)) + " |\n"

    # Populate the table with data
    for key in sorted_batch_keys:
        batch_size = key.split('_')[-1]
        row_data = benchmark_data[key]
        row = [batch_size] + [
            f"{row_data.get(h, ''):.2f}" if isinstance(row_data.get(h), float) else str(row_data.get(h, ''))
            for h in data_keys_ordered
        ]
        markdown += "| " + " | ".join(row) + " |\n"

    return markdown

def main():
    """
    Main function to read a JSON file and print a markdown table.
    """
    if len(sys.argv) != 2:
        print("Usage: python json_to_md.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        sys.exit(1)

    markdown_table = create_markdown_table(data)
    print(markdown_table)

if __name__ == "__main__":
    main() 