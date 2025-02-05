import argparse
import heapq
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from multiprocessing import Pool, Manager
from pathlib import Path

from tqdm.notebook import tqdm

from ngram_tools.helpers.file_handler import FileHandler


ITER_FILE_REGEX = re.compile(r'^merged_iter_(\d+)_chunk_\d+(\.jsonl(\.lz4)?)?$')


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


def process_a_file(args):
    """
    Sort a single input file by the specified sort key, then write the sorted
    lines to the output file.
    """
    (
        input_handler,
        output_handler,
        overwrite,
        compress,
        sort_key,
        sort_order
    ) = args

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

    with tqdm(total=len(handlers), desc="Sorting", unit='files') as pbar:
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
    """
    complete = False

    workers = os.cpu_count()  # Force to max CPU count inside this function

    os.makedirs(tmp_dir, exist_ok=True)  # Make temp dir for initial sort
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)  # Make corpus dir

    iteration_0_files = get_presorted_file_list(sorted_dir)

    current_iter_files = get_current_iter_files(
        start_iteration, iteration_0_files, tmp_dir
    )

    iteration = start_iteration
    while True:
        num_files = len(current_iter_files)  # How many files in the iteration?

        # If there's 1 file in the iteration, we're done; move to merged_dir
        if num_files == 1:
            shutil.move(current_iter_files[0], merged_path)
            print(f"Merging complete. Final file: {merged_path}")
            complete = True
            break

        # If 2 files in the iteration, do a final merge
        if num_files == 2:
            # Break if we've exceeded end_iteration
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

        # Adjust number of workers to ensure each worker gets at least 2 files
        max_workers = max(1, num_files // 2)
        active_workers = min(workers, max_workers)

        # Partition files among workers
        file_chunks = partition_files_among_workers(
            current_iter_files, active_workers
        )

        # Create file paths for iteration's chunk merges
        chunk_output_paths = []
        for idx, chunk in enumerate(file_chunks, start=1):
            ext = ".jsonl.lz4" if compress else ".jsonl"
            out_name = f"merged_iter_{iteration}_chunk_{idx}{ext}"
            out_path = os.path.join(tmp_dir, out_name)
            chunk_output_paths.append(out_path)

        print(f"\nIteration {iteration}: merging {num_files} files into "
              f"{len(file_chunks)} chunks using {active_workers} workers.")
        c_sizes = [len(ch) for ch in file_chunks]
        for size, count in sorted(Counter(c_sizes).items()):
            print(f"  {count} chunk(s) with {size} file(s)")

        # Perform parallel merges on each chunk
        with Pool(processes=active_workers) as pool:
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


def partition_files_among_workers(current_iter_files, workers):
    """
    Partition files into worker groups to balance total size across workers.
    """
    # Get file sizes
    file_sizes = [(os.path.getsize(file), file) for file in current_iter_files]

    # Sort files by size in descending order
    file_sizes.sort(reverse=True, key=lambda x: x[0])

    # Create a min-heap to track worker loads
    # Make tuples: (total_size, worker_id, files)
    worker_loads = [(0, i, []) for i in range(workers)]

    # Assign files to workers
    for size, file in file_sizes:
        # Pop the worker with the smallest current workload
        total_size, worker_id, worker_files = heapq.heappop(worker_loads)

        # Assign the file to this worker
        worker_files.append(file)
        total_size += size

        # Push the updated worker back into the heap
        heapq.heappush(worker_loads, (total_size, worker_id, worker_files))

    # Extract the file assignments from the heap
    worker_assignments = [
        worker_files for _, _, worker_files in sorted(
            worker_loads, key=lambda x: x[1]
        )
    ]

    return worker_assignments


def get_presorted_file_list(sorted_dir):
    iteration_0_files = [
        os.path.join(sorted_dir, f)
        for f in os.listdir(sorted_dir)
        if os.path.isfile(os.path.join(sorted_dir, f))
    ]

    return iteration_0_files


def get_current_iter_files(start_iteration, iteration_0_files, tmp_dir):
    """
    Get list of files for the current heapsort iteration.
    """
    if start_iteration == 1:
        current_iter_files = iteration_0_files
    else:
        current_iter_files = find_iteration_files(tmp_dir, start_iteration - 1)
        if not current_iter_files:
            raise FileNotFoundError(
                f"No files found for iteration {start_iteration - 1}. "
                "Cannot resume from iteration {start_iteration}."
            )

    return current_iter_files


def remove_iteration_files(tmp_dir, iteration):
    """
    Remove files for the specified iteration from the tmp_dir.
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
            with tqdm(total=total_lines_dir, desc="Merging", unit="lines") as pbar:
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