import os
import argparse
import orjson

from ngram_tools.helpers.file_handler import FileHandler


def print_jsonl_lines(file_path, start_line=1, end_line=5, parse_json=False):
    """
    Print or parse lines from a JSONL file using FileHandler.

    Args:
        file_path (str): Path to the JSONL file (can be .lz4-compressed).
        start_line (int): The line number to begin printing (1-based).
        end_line (int): The line number (inclusive) to stop printing.
        parse_json (bool): If True, parse each line as JSON using orjson. Otherwise,
            print each line as raw text.

    Raises:
        orjson.JSONDecodeError: If parse_json is True but a line cannot be parsed.
        Exception: For any I/O or FileHandler-related errors, a message is printed.
    """
    try:
        in_handler = FileHandler(file_path, is_output=False)
        with in_handler.open() as fin:
            for i, line in enumerate(fin, start=1):
                if i < start_line:
                    continue
                if i > end_line:
                    break

                if parse_json:
                    # Attempt to parse JSON
                    try:
                        parsed_line = in_handler.deserialize(line)
                        print(f"Line {i}: {parsed_line}")
                    except orjson.JSONDecodeError:
                        print(f"Line {i}: Error parsing JSON: {line.strip()}")
                else:
                    # Print raw line
                    print(f"Line {i}: {line.strip()}")
    except Exception as exc:
        print(f"Error reading the file '{file_path}': {exc}")


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inspect a range of lines from a JSONL file."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to the JSONL file (can be .lz4-compressed).",
        required=True
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="The starting line number (1-based). Default is 1."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=5,
        help="The ending line number (inclusive). Default is 5."
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse lines as JSON using orjson. Default is False (prints raw lines)."
    )
    return parser.parse_args()


def main():
    """
    Main entry point for command-line usage. Parses CLI arguments and calls
    print_jsonl_line to inspect lines from a JSONL file.
    """
    args = parse_arguments()

    print_jsonl_lines(
        file_path=args.file_path,
        start_line=args.start,
        end_line=args.end,
        parse_json=args.parse
    )


if __name__ == "__main__":
    main()
