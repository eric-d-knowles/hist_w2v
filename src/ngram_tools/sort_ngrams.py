import argparse
import heapq
import os
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from multiprocessing import Pool, Manager
from pathlib import Path

from tqdm import tqdm

from ngram_embeddings.helpers.file_handler import FileHandler


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)

ITER_FILE_REGEX = re.compile(r'^merged_iter_(\d+)_chunk_\d+(\.jsonl(\.lz4)?)?$')


def parse_args():
    """
    Parse command-line arguments for sorting and merging ngram files.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Sort and merge files in parallel.")

    parser.add_argument(
        '--ngram_size',
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help='Ngrams size.'
    )
    parser.add_argument(
        '--proj_dir',
        type=str,
        required=True,
        help='Path to the project base directory.'
    )
    parser.add_argument(
        '--file_range',
        type=int,
        nargs=2,
        help='Range of file indices to process (default=all).'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing files?'
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        default=False,
        help='Compress saved files (default=False).'
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
        help='Delete sorted files after merging?'
    )
    parser.add_argument(
        '--sort_key',
        type=str,
        choices=['freq_tot', 'ngram'],
        default='freq_tot',
        help="Key to sort on (freq_tot or ngram)."
    )
    parser.add_argument(
        '--sort_order',
        type=str,
        choices=['ascending', 'descending'],
        default='descending',
        help="Sort order (ascending or descending)."
    )
    parser.add_argument(
        '--start_iteration',
        type=int,
        default=1,
        help="Iteration to start merging from (default=1)."
    )
    parser.add_argument(
        '--end_iteration',
        type=int,
        default=None,
        help="Iteration to stop merging (default=None for all)."
    )

    return parser.parse_args()


def construct_output_path(input_file, output_dir, compress):
    """
    Construct the path for the output file, optionally appending .lz4 if compressed.

    Args:
        input_file (str): Path to the input file.
        output_dir (str): Directory where the output file will be saved.
        compress (bool): Whether the file should be compressed (lz4).

    Returns:
        str: The constructed output path.
    """
    input_path = Path(input_file)
    # If the file has '.lz4', remove it before building the base name
    base_name = input_path.stem if input_path.suffix == '.lz4' else input_path.name
    return str(Path(output_dir) / (base_name + ('.lz4' if compress else '')))


def set_info(proj_dir, ngram_size, file_range, compress):
    """
    Gather and prepare information about directories, input files, and final paths.

    Args:
        proj_dir (str): The base project directory.
        ngram_size (int): Size of the ngrams (1-5).
        file_range (tuple[int], optional): Range of file indices to process.
        compress (bool): Whether to compress output files.

    Returns:
        tuple: Contains:
            - input_dir (str): Path to the input directory.
            - sorted_dir (str): Path to a temporary directory for sorted files.
            - tmp_dir (str): Path to a temporary directory for merges.
            - merged_path (str): Path to the final merged output file.
            - num_files_available (int): Number of files available in input_dir.
            - first_file (str): Path to the first file in the specified range.
            - last_file (str): Path to the last file in the specified range.
            - num_files_to_use (int): Number of files to actually be used.
            - file_range (tuple[int]): The file range used.
            - input_paths_use (list[str]): List of input file paths to be processed.
            - sorted_paths (list[str]): Corresponding sorted output file paths.
    """
    input_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/5filter')
    sorted_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/temp')
    tmp_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/tmp')
    merged_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/6corpus')

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

    input_paths_use = input_paths_elig[file_range[0]: file_range[1] + 1]
    num_files_to_use = len(input_paths_use)

    first_file = input_paths_use[0]
    last_file = input_paths_use[-1]

    sorted_paths = sorted(
        construct_output_path(file, sorted_dir, compress)
        for file in input_paths_use
    )

    merged_path = os.path.join(
        merged_dir,
        f"{ngram_size}gram-merged.jsonl" + ('.lz4' if compress else '')
    )

    return (
        input_dir,
        sorted_dir,
        tmp_dir,
        merged_path,
        num_files_available,
        first_file,
        last_file,
        num_files_to_use,
        file_range,
        input_paths_use,
        sorted_paths
    )


def print_info(
    start_time,
    input_dir,
    sorted_dir,
    tmp_dir,
    merged_path,
    num_files_available,
    first_file,
    last_file,
    num_files_to_use,
    ngram_size,
    workers,
    compress,
    overwrite,
    sort_key,
    sort_order,
    start_iteration,
    end_iteration,
    delete_input
):
    """
    Print a configuration summary of the sorting/merging process.

    Args:
        start_time (datetime): Time when the process began.
        input_dir (str): Path to the input directory.
        sorted_dir (str): Path to the temporary sorted directory.
        tmp_dir (str): Path to the temporary merging directory.
        merged_path (str): Path where the final merged file will be saved.
        num_files_available (int): Number of files originally available.
        first_file (str): Path to the first file in the range.
        last_file (str): Path to the last file in the range.
        num_files_to_use (int): Number of files to actually process.
        ngram_size (int): Size of the ngrams (1-5).
        workers (int): Number of worker processes.
        compress (bool): Whether output files are compressed.
        overwrite (bool): Whether to overwrite existing files.
        sort_key (str): The key to sort on (freq_tot or ngram).
        sort_order (str): Ascending or descending.
        start_iteration (int): Iteration to begin merging from.
        end_iteration (int or None): Iteration at which to stop merging.
        delete_input (bool): Whether to delete sorted files after merging.
    """
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mSort Info\033[0m')
    print(f'Input directory:           {input_dir}')
    print(f'Sorted directory:          {sorted_dir}')
    print(f'Temp directory:            {tmp_dir}')
    print(f'Merged file:               {merged_path}')
    print(f'Files available:           {num_files_available}')
    print(f'First file to get:         {first_file}')
    print(f'Last file to get:          {last_file}')
    print(f'Files to use:              {num_files_to_use}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Number of workers:         {workers}')
    print(f'Compress output files:     {compress}')
    print(f'Overwrite existing files:  {overwrite}')
    print(f'Sort key:                  {sort_key}')
    print(f'Sort order:                {sort_order}')
    print(f'Heap-merge start iter:     {start_iteration}')
    print(f'Heap-merge end iter:       {end_iteration}')
    print(f'Deleted sorted files:      {delete_input}\n')


def create_progress_bar(total, description, unit=''):
    """
    Create and return a tqdm progress bar with the specified parameters.

    Args:
        total (int): The total number of items for the bar.
        description (str): A short label for the bar.
        unit (str, optional): Unit of measurement (e.g., 'files').

    Returns:
        tqdm.std.tqdm: A configured tqdm progress bar.
    """
    padded_desc = description.ljust(FIXED_DESC_LENGTH)
    return tqdm(
        file=sys.stdout,
        total=total,
        desc=padded_desc,
        leave=True,
        dynamic_ncols=False,
        ncols=100,
        unit=unit,
        colour='green',
        bar_format=BAR_FORMAT
    )


def process_a_file(args):
    """
    Sort a single input file by the specified sort key, then write the sorted
    lines to the output file.

    Args:
        args (tuple):
            input_handler (FileHandler): Handler to read from input file.
            output_handler (FileHandler): Handler to write to output file.
            overwrite (bool): Whether to overwrite existing output file.
            compress (bool): Whether the output is compressed.
            sort_key (str): Sorting key ('freq_tot' or 'ngram').
            sort_order (str): 'ascending' or 'descending'.

    Returns:
        int: The number of lines processed in this file.
    """
    input_handler, output_handler, overwrite, compress, sort_key, sort_order = args

    if not overwrite and os.path.exists(output_handler.path):
        # If not overwriting, just count lines in the existing input file.
        with input_handler.open() as infile:
            line_count = sum(1 for _ in infile)
        return line_count

    with input_handler.open() as infile, output_handler.open() as outfile:
        lines = []

        for line in infile:
            entry = input_handler.deserialize(line)
            if sort_key == 'ngram':
                # Convert ngram (dict of tokens) into a single string for sorting
                tokens = list(entry['ngram'].values())
                entry['ngram'] = " ".join(tokens)
            lines.append(entry)

        reverse = (sort_order == 'descending')

        if sort_key == 'freq_tot':
            lines.sort(key=lambda x: x['freq_tot'], reverse=reverse)
        elif sort_key == 'ngram':
            lines.sort(key=lambda x: x['ngram'], reverse=reverse)

        for line_data in lines:
            outfile.write(output_handler.serialize(line_data))

    return len(lines)


def process_a_directory(
    input_paths,
    output_dir,
    output_paths,
    overwrite,
    compress,
    workers,
    sort_key,
    sort_order
):
    """
    Sort multiple input files in parallel, writing results to a specified
    directory.

    Args:
        input_paths (list[str]): List of input file paths.
        output_dir (str): Directory for sorted output files.
        output_paths (list[str]): Corresponding output file paths.
        overwrite (bool): Whether to overwrite existing files.
        compress (bool): Whether the output files should be compressed (lz4).
        workers (int): Number of parallel processes.
        sort_key (str): The key to sort on (freq_tot or ngram).
        sort_order (str): 'ascending' or 'descending'.

    Returns:
        int: The total number of lines processed across all files in the
        directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_lines_dir = 0
    handlers = [
        (
            FileHandler(input_path),
            FileHandler(output_path, is_output=True, compress=compress)
        )
        for input_path, output_path in zip(input_paths, output_paths)
    ]

    args = [
        (inp, out, overwrite, compress, sort_key, sort_order)
        for inp, out in handlers
    ]

    with create_progress_bar(len(handlers), "Sorting", 'files') as pbar:
        with Manager() as manager:
            progress = manager.Value('i', 0)

            def update_progress(_):
                progress.value += 1
                pbar.update()

            with Pool(processes=workers) as pool:
                for total_lines_file in pool.imap_unordered(process_a_file, args):
                    total_lines_dir += total_lines_file
                    update_progress(None)

    return total_lines_dir


def iterative_merge(
    sorted_dir,
    tmp_dir,
    workers,
    sort_key,
    sort_order,
    compress,
    merged_path,
    total_lines_dir,
    start_iteration=1,
    end_iteration=None,
    overwrite=False
):
    """
    Iteratively merge sorted files in parallel chunks using heapq.merge. Each
    iteration merges chunks of files until a single final file remains or until
    end_iteration is reached.

    Args:
        sorted_dir (str): Directory where initial sorted files are stored.
        tmp_dir (str): Temporary directory used to store intermediate merges.
        workers (int): Number of parallel processes for merging.
        sort_key (str): 'freq_tot' or 'ngram' (heapq merge key).
        sort_order (str): 'ascending' or 'descending'.
        compress (bool): Whether final output is compressed (lz4).
        merged_path (str): Path to the final merged output file.
        total_lines_dir (int): Total number of lines for progress tracking.
        start_iteration (int, optional): Iteration to start merging at (default=1).
        end_iteration (int or None, optional): Iteration at which to stop merging.
    """
    complete = False

    workers = os.cpu_count()  # Force to max CPU count inside this function

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)

    # Iteration 0: the individually sorted files
    iteration_0_files = [
        os.path.join(sorted_dir, f)
        for f in os.listdir(sorted_dir)
        if os.path.isfile(os.path.join(sorted_dir, f))
    ]
    random.shuffle(iteration_0_files)

    # Decide which iteration's files to start with
    if start_iteration == 1:
        current_iter_files = iteration_0_files
    else:
        current_iter_files = find_iteration_files(tmp_dir, start_iteration - 1)
        if not current_iter_files:
            raise FileNotFoundError(
                f"No files found for iteration {start_iteration - 1}. "
                "Cannot resume from iteration {start_iteration}."
            )

    iteration = start_iteration

    while True:
        num_files = len(current_iter_files)

        # If there's 0 or 1 file, we've reached the final condition
        if num_files <= 1:
            if num_files == 1 and current_iter_files[0] != merged_path:
                shutil.move(current_iter_files[0], merged_path)
            print(f"Merging complete. Final file: {merged_path}")
            break

        # If 2 files remain, do a final merge
        if num_files == 2:
            if end_iteration is not None and iteration > end_iteration:
                break
            print(f"\nIteration {iteration}: final merge of 2 files.")

            heap_merge_chunk(
                current_iter_files,
                merged_path,
                sort_key,
                sort_order,
                compress,
                total_lines_dir,
                True,  # show progress bar
                iteration
            )
            complete = True

            print(f"\nMerging complete. Final merged file:\n{merged_path}")
            break

        # If we've exceeded end_iteration, stop
        if end_iteration is not None and iteration > end_iteration:
            break

        # Split current files into chunks for parallel merging
        base_chunk_size = num_files // workers
        remainder = num_files % workers

        while base_chunk_size <= 1 and workers > 2:
            workers = max(2, workers // 2)
            base_chunk_size = num_files // workers
            remainder = num_files % workers

        file_chunks = []
        start_idx = 0
        for i in range(workers):
            end_idx = start_idx + base_chunk_size + (1 if i < remainder else 0)
            if start_idx < num_files:
                file_chunks.append(current_iter_files[start_idx:end_idx])
            start_idx = end_idx

        # Create file paths for iteration's chunk merges
        chunk_output_paths = []
        for idx, chunk in enumerate(file_chunks, start=1):
            ext = ".jsonl.lz4" if compress else ".jsonl"
            out_name = f"merged_iter_{iteration}_chunk_{idx}{ext}"
            out_path = os.path.join(tmp_dir, out_name)
            chunk_output_paths.append(out_path)

        print(f"\nIteration {iteration}: merging {num_files} files into "
              f"{len(file_chunks)} chunks using {workers} workers.")
        c_sizes = [len(ch) for ch in file_chunks]
        for size, count in sorted(Counter(c_sizes).items()):
            print(f"  {count} chunk(s) with {size} file(s)")

        # Perform parallel merges on each chunk
        with Pool(processes=workers) as pool:
            pool.starmap(
                heap_merge_chunk,
                [
                    (
                        file_chunks[i],
                        chunk_output_paths[i],
                        sort_key,
                        sort_order,
                        compress,
                        total_lines_dir,
                        False,  # no progress bar for chunk merges
                        iteration
                    )
                    for i in range(len(file_chunks))
                ]
            )

        # Remove iteration i-1 files (if i-1 >= start_iteration) to save space
        if iteration >= start_iteration:
            remove_iteration_files(tmp_dir, iteration - 1)

        current_iter_files = chunk_output_paths
        iteration += 1

        return complete


def remove_iteration_files(tmp_dir, iteration):
    """
    Remove files for the specified iteration from the tmp_dir.

    Args:
        tmp_dir (str): Path to the directory containing iteration files.
        iteration (int): The iteration to remove (merged_iter_{iteration}).
    """
    if iteration < 1:
        return
    pattern = re.compile(rf"^merged_iter_{iteration}_chunk_\d+(\.jsonl(\.lz4)?)?$")
    for filename in os.listdir(tmp_dir):
        if pattern.match(filename):
            os.remove(os.path.join(tmp_dir, filename))


def find_iteration_files(tmp_dir, iteration):
    """
    Find all chunk files from a specified iteration in tmp_dir.

    Args:
        tmp_dir (str): Directory to search for iteration files.
        iteration (int): The iteration to look for.

    Returns:
        list[str]: A list of file paths matching the iteration pattern.
    """
    pattern = re.compile(rf"^merged_iter_{iteration}_chunk_\d+(\.jsonl(\.lz4)?)?$")
    results = []
    for filename in os.listdir(tmp_dir):
        if pattern.match(filename):
            results.append(os.path.join(tmp_dir, filename))
    return results


def heap_merge_chunk(
    chunk_files,
    output_path,
    sort_key,
    sort_order,
    compress,
    total_lines_dir,
    use_progress_bar,
    iteration
):
    """
    Merge a list of chunk files with heapq.merge, writing the result to output_path.

    Args:
        chunk_files (list[str]): Paths of files to merge.
        output_path (str): Path for the merged file.
        sort_key (str): 'freq_tot' or 'ngram'.
        sort_order (str): 'ascending' or 'descending'.
        compress (bool): Whether output file is compressed.
        total_lines_dir (int): Total lines (for progress bar).
        use_progress_bar (bool): Whether to show a progress bar.
        iteration (int): The current iteration number.
    """
    reverse = (sort_order == "descending")

    def merge_key_func(item):
        return item[sort_key]

    output_handler = FileHandler(output_path, is_output=True, compress=compress)

    file_iters = [
        map(FileHandler(file).deserialize, FileHandler(file).open())
        for file in chunk_files
    ]

    with output_handler.open() as outfile:
        if use_progress_bar and total_lines_dir > 0:
            with create_progress_bar(total_lines_dir, "Merging", "lines") as pbar:
                for item in heapq.merge(
                    *file_iters, key=merge_key_func, reverse=reverse
                ):
                    outfile.write(output_handler.serialize(item))
                    pbar.update(1)
        else:
            for item in heapq.merge(*file_iters, key=merge_key_func, reverse=reverse):
                outfile.write(output_handler.serialize(item))


def clear_directory(directory_path):
    """
    Remove all files and empty subdirectories from the specified directory.

    Args:
        directory_path (str): Path to the directory to clear.
    """
    for entry in os.scandir(directory_path):
        if entry.is_file():
            os.remove(entry.path)
        elif entry.is_dir():
            os.rmdir(entry.path)


def sort_ngrams(
    ngram_size,
    proj_dir,
    file_range=None,
    compress=False,
    overwrite=False,
    workers=os.cpu_count(),
    sort_key='freq_tot',
    sort_order='descending',
    delete_input=False,
    start_iteration=1,
    end_iteration=None
):
    """
    Main function to sort ngrams by a specified key, then optionally iteratively
    merge them into a single output file.

    Args:
        ngram_size (int): Size of the ngrams (1-5).
        proj_dir (str): Base path of the project directory.
        file_range (tuple[int], optional): Range of file indices to process.
        compress (bool, optional): Whether output files should be .lz4 compressed.
        overwrite (bool, optional): Overwrite existing files if True.
        workers (int, optional): Number of parallel processes for sorting/merging.
        sort_key (str, optional): 'freq_tot' or 'ngram' to sort by.
        sort_order (str, optional): 'ascending' or 'descending'.
        delete_input (bool, optional): If True, delete sorted intermediate files.
        start_iteration (int, optional): Iteration to begin merging (default=1).
        end_iteration (int or None, optional): Iteration to stop merging (default=None).
    """
    start_time = datetime.now()

    (
        input_dir,
        sorted_dir,
        tmp_dir,
        merged_path,
        num_files_available,
        first_file,
        last_file,
        num_files_to_use,
        file_range,
        input_paths_use,
        sorted_paths
    ) = set_info(proj_dir, ngram_size, file_range, compress)

    print_info(
        start_time,
        input_dir,
        sorted_dir,
        tmp_dir,
        merged_path,
        num_files_available,
        first_file,
        last_file,
        num_files_to_use,
        ngram_size,
        workers,
        compress,
        overwrite,
        sort_key,
        sort_order,
        start_iteration,
        end_iteration,
        delete_input
    )

    # Step 1: Sort each input file individually
    total_lines_dir = process_a_directory(
        input_paths_use, sorted_dir, sorted_paths,
        overwrite, compress, workers, sort_key, sort_order
    )

    # Step 2: Iteratively merge the sorted files
    complete = iterative_merge(
        sorted_dir,
        tmp_dir,
        workers,
        sort_key,
        sort_order,
        compress,
        merged_path,
        total_lines_dir,
        start_iteration,
        end_iteration
    )

    # If delete_input=True, clear the `5filter` directory and remove
    # `sorted_dir` and `tmp_dir`.
    # Only do this if the final merge is complete!
    if delete_input and complete:
        clear_directory(input_dir)
        shutil.rmtree(sorted_dir)
        shutil.rmtree(tmp_dir)

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    args = parse_args()
    sort_ngrams(
        ngram_size=args.ngram_size,
        proj_dir=args.proj_dir,
        file_range=args.file_range,
        overwrite=args.overwrite,
        compress=args.compress,
        workers=args.workers,
        sort_key=args.sort_key,
        sort_order=args.sort_order,
        delete_input=args.delete_input,
        start_iteration=args.start_iteration,
        end_iteration=args.end_iteration
    )