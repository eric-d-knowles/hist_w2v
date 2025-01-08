import argparse
import lz4.frame
import orjson
from tqdm import tqdm


class FileHandler:
    def __init__(self, path, is_output=False, compress=False):
        self.path = path
        self.compress = compress if is_output else path.endswith('.lz4')
        self.binary_mode = self.compress
        self.mode = (
            'wb' if is_output and self.compress else
            'w' if is_output else
            'rb' if self.compress else
            'r'
        )
        self.encoding = None if self.binary_mode else 'utf-8'
        self.open_fn = lz4.frame.open if self.compress else open

    def open(self):
        if self.compress:
            return self.open_fn(self.path, self.mode)
        return self.open_fn(self.path, self.mode, encoding=self.encoding)

    def serialize(self, entry):
        serialized = orjson.dumps(entry)
        return serialized + b'\n' if self.binary_mode else serialized.decode(
            'utf-8'
        ) + '\n'

    def deserialize(self, line):
        return orjson.loads(line)


def is_file_sorted(file_handler, field, sort_order):
    """
    Check if a file is sorted based on the specified field and order.

    Args:
        file_handler (FileHandler): An instance of the FileHandler class.
        field (str): The JSON field to verify sorting on ('ngram' or 'freq_tot').
        sort_order (str): The order to check ('ascending' or 'descending').

    Returns:
        bool: True if the file is sorted, False otherwise.
    """
    previous_value = None

    with file_handler.open() as infile, tqdm(
        unit='line', desc="Lines", dynamic_ncols=True
    ) as pbar:
        for line_number, line in enumerate(infile, start=1):
            # Deserialize the line using FileHandler
            entry = file_handler.deserialize(line.strip())
            current_value = entry.get(field)

            if current_value is None:
                print(f"Line {line_number}: Missing '{field}' field.")
                return False

            # Compare values based on sort order
            if previous_value is not None:
                if sort_order == 'ascending' and current_value < previous_value:
                    print(f"Line {line_number}: File is not sorted. "
                          f"'{current_value}' appears after '{previous_value}'.")
                    return False
                elif sort_order == 'descending' and current_value > previous_value:
                    print(f"Line {line_number}: File is not sorted. "
                          f"'{current_value}' appears before '{previous_value}'.")
                    return False

            # Update previous value for the next comparison
            previous_value = current_value
            pbar.update(1)

    return True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify whether a JSONL file is sorted based on a specified field."
    )
    parser.add_argument(
        '--input_file',
        required=True,
        help="Path to the file to be checked."
    )
    parser.add_argument(
        '--field',
        required=True,
        choices=['ngram', 'freq_tot'],
        help="The field to verify sorting on ('ngram' or 'freq_tot')."
    )
    parser.add_argument(
        '--sort_order',
        required=True,
        choices=['ascending', 'descending'],
        help="The sort order to check ('ascending' or 'descending')."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create a FileHandler instance for the input file
    file_handler = FileHandler(path=args.input_file, is_output=False)

    # Check if the file is sorted
    is_sorted = is_file_sorted(file_handler, args.field, args.sort_order)
    if is_sorted:
        print("\nThe file is sorted.")
    else:
        print("\nThe file is not sorted.")
    print("\nProcessing complete.")