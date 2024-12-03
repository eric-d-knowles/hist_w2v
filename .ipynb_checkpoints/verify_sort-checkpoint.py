import argparse
from tqdm import tqdm


def is_file_sorted(input_file):
    """Check if a file is sorted lexicographically by lines."""
    previous_line = None

    with open(input_file, 'r', encoding='utf-8') as infile, tqdm(
        unit='line', desc="Lines", dynamic_ncols=True
    ) as pbar:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()

            if previous_line is not None and line < previous_line:
                print(f"Line {line_number}: File is not sorted. '{line}' appears after '{previous_line}'.")
                return False

            # Update previous line for the next comparison
            previous_line = line
            pbar.update(1)

    return True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify whether a file is sorted lexicographically by lines."
    )
    parser.add_argument(
        '--input_file',
        required=True,
        help="Path to the file to be checked."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    is_sorted = is_file_sorted(args.input_file)
    if is_sorted:
        print("\nThe file is sorted.")
    else:
        print("\nThe file is not sorted.")
    print("\nProcessing complete.")
