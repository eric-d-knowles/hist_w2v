import argparse
import os
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import orjson
from tqdm import tqdm

from ngram_embeddings.helpers.file_handler import FileHandler


def parse_args():
    """
    Parse command-line arguments for lowercasing ngrams.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Lowercase ngrams.")

    parser.add_argument(
        '--ngram_size',
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help='Ngram size to process (required).'
    )
    parser.add_argument(
        "--proj_dir",
        type=str,
        required=True,
        help='Path to the project base directory.'
    )
    parser.add_argument(
        '--file_range',
        type=int,
        nargs=2,
        help='Range of file indices to process [default=all].'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing files? (default=False).'
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        default=False,
        help='Compress saved files? (default=False).'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count(),
        help='Number of processors to use (default=all).'
    )
    parser.add_argument(
        '--delete_input',
        action='store_true',
        default=False,
        help='Delete input files (default=False).'
    )

    return parser.parse_args()


def construct_output_path(input_file, output_dir, compress):
    """
    Construct the output file path based on the input file path and compression.

    Args:
        input_file (str): Path to the input file.
        output_dir (str): Directory where the output will be saved.
        compress (bool): Whether the file should be compressed (lz4).

    Returns:
        str: The constructed output file path.
    """
    input_path = Path(input_file)
    # If the file has a ".lz4" suffix, remove it before constructing the base name.
    base_name = (
        input_path.stem if input_path.suffix == '.lz4' else input_path.name
    )
    return str(Path(output_dir) / (base_name + ('.lz4' if compress else '')))


def set_info(proj_dir, ngram_size, file_range, compress):
    """
    Validate and set up file paths and metadata needed for lowercasing ngrams.

    Args:
        proj_dir (str): Project base directory path.
        ngram_size (int): The size of the ngrams (1-5).
        file_range (tuple or list of int, optional): Range of files to process.
        compress (bool): Whether output files should be compressed (lz4).

    Returns:
        tuple: Contains:
            - input_dir (str): The input directory path.
            - output_dir (str): The output directory path.
            - file_range (tuple): Updated file range.
            - num_files_available (int): Count of all available input files.
            - input_paths_use (list[str]): The specific input file paths to use.
            - file_range (tuple): Same file range passed in (for printing).
            - num_files_to_use (int): Number of files that will be used.
            - first_file (str): The first file in the chosen range.
            - last_file (str): The last file in the chosen range.
            - output_paths (list[str]): Corresponding output file paths.
    """
    input_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/2convert')
    output_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/3lowercase')

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(
            f"Input directory {input_dir} does not exist or isn't a directory."
        )

    input_paths_elig = sorted(
        [entry.path for entry in os.scandir(input_dir) if entry.is_file()]
    )
    num_files_available = len(input_paths_elig)

    if not file_range:
        file_range = (0, len(input_paths_elig) - 1)

    input_paths_use = input_paths_elig[file_range[0] : file_range[1] + 1]
    num_files_to_use = len(input_paths_use)

    first_file = input_paths_use[0]
    last_file = input_paths_use[-1]

    output_paths = sorted([
        construct_output_path(file, output_dir, compress)
        for file in input_paths_use
    ])

    return (
        input_dir, output_dir, file_range, num_files_available,
        input_paths_use, file_range, num_files_to_use, first_file,
        last_file, output_paths
    )


def print_info(input_dir, output_dir, file_range, num_files_available,
               num_files_to_use, first_file, last_file, ngram_size, workers,
               compress, overwrite, start_time, delete_input):
    """
    Print a summary of the lowercasing process settings for the user.

    Args:
        input_dir (str): The input directory path.
        output_dir (str): The output directory path.
        file_range (tuple): Range of file indices to process.
        num_files_available (int): Count of all available input files.
        num_files_to_use (int): Number of input files to process.
        first_file (str): First file in the selected range.
        last_file (str): Last file in the selected range.
        ngram_size (int): The size of the ngrams (1-5).
        workers (int): Number of worker processes to spawn.
        compress (bool): Whether output files should be compressed (lz4).
        overwrite (bool): Whether to overwrite existing files.
        start_time (datetime): The time the process started.
        delete_input (bool): Whether input files should be deleted.
    """
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mLowercasing Info\033[0m')
    print(f'Input directory:           {input_dir}')
    print(f'Output directory:          {output_dir}')
    print(f'File index range:          {file_range[0]} to {file_range[1]}')
    print(f'Files available:           {num_files_available}')
    print(f'Files to use:              {num_files_to_use}')
    print(f'First file to get:         {first_file}')
    print(f'Last file to get:          {last_file}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Number of workers:         {workers}')
    print(f'Compress output files:     {compress}')
    print(f'Overwrite existing files:  {overwrite}')
    print(f'Delete input directory:    {delete_input}\n')


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


def create_progress_bar(total, description, unit=''):
    """
    Create a tqdm progress bar with a fixed description length and custom format.

    Args:
        total (int): Total number of items for the progress bar.
        description (str): Label or short message describing the progress bar.
        unit (str, optional): Unit of measurement (e.g., 'files').

    Returns:
        tqdm.std.tqdm: Configured tqdm progress bar object.
    """
    return tqdm(
        file=sys.stdout,
        total=total,
        desc=description.ljust(FIXED_DESC_LENGTH),
        leave=True,
        dynamic_ncols=False,
        ncols=100,
        unit=unit,
        colour='green',
        bar_format=BAR_FORMAT
    )


def process_a_line(line):
    """
    Process a single line of JSON by converting the 'ngram' tokens to lowercase.

    Args:
        line (bytes or str): A line of JSON (possibly bytes).

    Returns:
        dict: The updated JSON object after lowercasing.
    """
    if isinstance(line, bytes):
        line = line.decode('utf-8')

    try:
        data = orjson.loads(line)
    except orjson.JSONDecodeError as exc:
        print(f"Failed to parse JSON line: {line[:100]}... Error: {exc}")
        raise

    data['ngram'] = {
        key: value.lower() for key, value in data['ngram'].items()
    }

    return data


def process_a_file(args):
    """
    Process a single input file, apply lowercasing to each line,
    and save the results to the output file.

    Args:
        args (tuple): Contains:
            in_handler (FileHandler): File handler for reading the input.
            out_handler (FileHandler): File handler for writing the output.
            overwrite (bool): Whether to overwrite existing output files.
    """
    in_handler, out_handler, overwrite = args

    if not overwrite and os.path.exists(out_handler.path):
        return

    try:
        with in_handler.open() as infile, out_handler.open() as outfile:
            for line_no, line in enumerate(infile, start=1):
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                json_dict = process_a_line(line)
                jsonl_line = out_handler.serialize(json_dict)
                outfile.write(jsonl_line)
    except Exception as exc:
        print(f"Error processing {in_handler.path}: {exc}")


def process_a_directory(output_dir, input_paths, output_paths, workers,
                        overwrite, compress):
    """
    Process all eligible input files in a directory with multiple workers.

    Args:
        output_dir (str): Path to the output directory.
        input_paths (list[str]): List of input file paths.
        output_paths (list[str]): List of output file paths.
        workers (int): Number of worker processes to use.
        overwrite (bool): Whether to overwrite existing output files.
        compress (bool): Whether output files should be compressed (lz4).
    """
    os.makedirs(output_dir, exist_ok=True)

    handlers = []
    for input_path, output_path in zip(input_paths, output_paths):
        input_ext = Path(input_path).suffix
        input_size = os.path.getsize(input_path)

        # Skip empty JSONL or minimal-size lz4 files.
        if ((input_ext == '.jsonl' and input_size == 0) or
                (input_ext == '.lz4' and input_size == 11)):
            continue

        in_handler = FileHandler(input_path)
        out_handler = FileHandler(output_path, is_output=True, compress=compress)
        handlers.append((in_handler, out_handler))

    args = [(in_handler, out_handler, overwrite) for in_handler, out_handler in handlers]

    with create_progress_bar(len(handlers), "Lowercasing", 'files') as pbar:
        with Pool(processes=workers) as pool:
            for _ in pool.imap_unordered(process_a_file, args):
                pbar.update()


def clear_a_directory(directory_path):
    """
    Remove all files (and empty subdirectories) from the specified directory.

    Args:
        directory_path (str): Path to the directory to clear.
    """
    for entry in os.scandir(directory_path):
        if entry.is_file():
            os.remove(entry.path)
        elif entry.is_dir():
            os.rmdir(entry.path)


def lowercase_ngrams(ngram_size, proj_dir, file_range=None, overwrite=False,
                     compress=False, workers=os.cpu_count(), delete_input=False):
    """
    Main function to drive the lowercasing process. Reads the specified range
    of files from the project directory, applies lowercasing, and writes results.

    Args:
        ngram_size (int): The size of ngrams (1-5).
        proj_dir (str): The base directory path for the project.
        file_range (tuple[int], optional): Range of files to process.
        overwrite (bool, optional): Overwrite existing files.
        compress (bool, optional): Compress output files (lz4).
        workers (int, optional): Number of parallel processes to use.
        delete_input (bool, optional): Delete the input directory after processing.
    """
    start_time = datetime.now()

    (
        input_dir,
        output_dir,
        file_range,
        num_files_available,
        input_paths,
        file_range,
        num_files_to_use,
        first_file,
        last_file,
        output_paths
    ) = set_info(proj_dir, ngram_size, file_range, compress)

    print_info(
        input_dir, output_dir, file_range, num_files_available,
        num_files_to_use, first_file, last_file, ngram_size, workers,
        compress, overwrite, start_time, delete_input
    )

    process_a_directory(
        output_dir, input_paths, output_paths, workers, overwrite, compress
    )

    if delete_input:
        clear_a_directory(input_dir)

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    args = parse_args()
    lowercase_ngrams(
        ngram_size=args.ngram_size,
        proj_dir=args.proj_dir,
        file_range=args.file_range,
        overwrite=args.overwrite,
        compress=args.compress,
        workers=args.workers,
        delete_input=args.delete_input
    )
