import argparse
import orjson
from file_handler import FileHandler


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inspect a range of lines from a JSONL file."
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the JSONL file."
    )
    parser.add_argument(
        "--start", type=int, default=1, help=(
            "Starting line number (1-based, default is 1)."
        )
    )
    parser.add_argument(
        "--end", type=int, default=5, help=(
            "Ending line number (inclusive, default is 5)."
        )
    )
    parser.add_argument(
        "--parse", action="store_true", help=(
            "Parse lines as JSON (default is False)."
        )
    )
    return parser.parse_args()


def print_jsonl_line(file_path, start_line=1, end_line=5, parse_json=False):
    """
    Prints lines from a JSONL file (compressed or uncompressed) using
    FileHandler.

    Args:
        file_path (str): Path to the JSONL file.
        start_line (int): The starting line number (default is 1).
        end_line (int): The ending line number (inclusive, default is 5).
        parse_json (bool): Whether to parse lines as JSON (default is False).
    """
    try:
        fh = FileHandler(file_path, is_output=False)
        with fh.open() as fin:
            for i, line in enumerate(fin, start=1):
                if i < start_line:
                    continue
                if i > end_line:
                    break

                if parse_json:
                    # Attempt to parse JSON
                    try:
                        parsed_line = fh.deserialize(line)
                        print(f"Line {i}: {parsed_line}")
                    except orjson.JSONDecodeError:
                        print(f"Line {i}: Error parsing JSON: {line.strip()}")
                else:
                    # Print raw line
                    print(f"Line {i}: {line.strip()}\n")
    except Exception as e:
        print(f"Error reading the file '{file_path}': {e}")


if __name__ == "__main__":
    args = parse_args()
    print_jsonl_line(
        file_path=args.file_path,
        start_line=args.start,
        end_line=args.end,
        parse_json=args.parse
    )