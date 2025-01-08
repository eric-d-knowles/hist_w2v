import argparse

def print_lines(file_path, start, end):
    """
    Print lines from `start` to `end` (inclusive) of the given plain text file.

    Args:
    - file_path (str): Path to the plain text file.
    - start (int): Line number to start printing from (1-based index).
    - end (int): Line number to stop printing at (1-based index).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number < start:
                    continue
                if current_line_number > end:
                    break
                print(line.strip())  # Print without trailing newline characters
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Print lines n to m of a plain text file.")
    parser.add_argument("--file", "-f", required=True, help="Path to the plain text file.")
    parser.add_argument("--start", "-s", type=int, required=True, help="Starting line number (1-based index).")
    parser.add_argument("--end", "-e", type=int, required=True, help="Ending line number (1-based index).")

    args = parser.parse_args()

    # Validate the range
    if args.start < 1 or args.end < args.start:
        print("Error: Invalid range. Ensure start >= 1 and end >= start.")
    else:
        print_lines(args.file, args.start, args.end)

if __name__ == "__main__":
    main()
