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