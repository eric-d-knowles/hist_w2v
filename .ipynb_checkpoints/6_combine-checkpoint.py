import os
import argparse
import lz4.frame
import heapq
from datetime import datetime
from tqdm.auto import tqdm
import tempfile
import multiprocessing
import orjson
import shutil
from pathlib import Path


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} \
    [{elapsed}<{remaining}, {rate_fmt}]"
)


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:6.2f}% {n_fmt:<15}/{total_fmt:<15} \
    [{elapsed}<{remaining}, {rate_fmt}]"
)


def parse_args():
    """
    Define and parse command-line arguments for the ngram downloader.

    Arguments:
        --ngram_size (int, required): Size of ngrams to get.
        --proj_dir (str, required): Local project directory.
        --file_range (int, default=all): File index range to get.
        --overwrite (bool, default=False): Overwrite existing files.
        --compress (bool, default=False): Compress the output files.
        --workers (int, default=all): Number of processes.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Combine, sort, and consolidate ngram files into a single "
                    "consolidated sorted file."
    )

    parser.add_argument('--proj_dir',
                        type=str,
                        required=True,
                        help='Base project directory (required).')

    parser.add_argument('--ngram_size',
                        type=int,
                        required=True,
                        help='Size of the ngrams to combine (required).')

    parser.add_argument('--file_range',
                        type=int,
                        nargs=2,
                        help='Range of file indices to get (default = all).')

    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing output file (default=False).')

    parser.add_argument('--compress',
                        action='store_true',
                        default=False,
                        help='Compress the output file (default=False).')

    parser.add_argument('--workers',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of processors to use (default=all).')

    return parser.parse_args()


def print_info(input_dir, output_dir, file_range, files_available,
               files_to_use, file_first, file_last, ngram_size, workers,
               overwrite, start_time):
    """
    Prints summary of configuration for downloading ngram files.
    """
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print(f'Input directory:           {input_dir}')
    print(f'Output directory:          {output_dir}')
    print(f'File index range:          {file_range[0]} to {file_range[1]}')
    print(f'Files available:           {files_available}')
    print(f'Files to use:              {files_to_use}')
    print(f'First file to get:         {file_first}')
    print(f'Last file to get:          {file_last}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Overwrite existing files:  {overwrite}')
    print(f'Workers:                   {workers}\n')


def create_progress_bar(total, description, unit=''):
    """
    Creates a tqdm.auto progress bar.

    Args:
        total (int or None): Total iterations. Use None for indeterminate.
        description (str): Description for the progress bar.
        unit (str): Unit of progress (e.g., 'files', 'lines').

    Returns:
        tqdm.tqdm: Configured tqdm.auto progress bar instance.
    """

    padded_desc = description.ljust(FIXED_DESC_LENGTH)
    return tqdm(
        total=total,
        desc=padded_desc,
        leave=True,
        dynamic_ncols=False,
        ncols=100,
        unit=unit,
        colour='green',
        bar_format=BAR_FORMAT
    )


def read_lines_from_file(file_path):
    """
    Read all lines from a file and return as a list of strings.

    Arguments:
        file_path (str): File path to read.

    Returns:
        lines (list[str]): A list of lines.
    """

    lines = []
    if file_path.endswith('.lz4'):
        with lz4.frame.open(file_path, 'rt', encoding='utf-8') as infile:
            for line in infile:
                lines.append(line)
    else:
        with open(file_path, 'r', encoding='utf-8') as infile:
            lines.extend(infile)
    return lines


def sort_file_in_memory(file_path, output_temp_path):
    """
    Sort the contents of a file in memory and write output_temp_path.

    Arguments:
        file_path (str): File path to read.
        output_temp_path (str): Path to the temporary compressed output file.

    Returns:
        int: Number of lines sorted.
    """

    lines = read_lines_from_file(file_path)
    lines.sort()
    line_count = len(lines)

    if output_temp_path.endswith('.lz4'):
        with lz4.frame.open(output_temp_path, 'wb') as out:
            for line in lines:
                out.write(line.encode('utf-8') + b'\n')
    else:
        with open(output_temp_path, 'w', encoding='utf-8') as out:
            for line in lines:
                out.write(line)
    return line_count


def heap_merge_files(
    file_paths, output_path, total_lines, compress=False,
    update_interval=100000
):
    """
    Heap-merge the sorted compressed files into a single sorted output file.

    Arguments:
        file_paths (list[str]): Compressed sorted file paths to read.
        output_path (str): Path to the final merged output file.
        total_lines (int): Total number of lines to process.
        compress (bool): Whether to compress the final output.
        update_interval (int): Lines after which to update the progress bar.

    Returns:
        int: The total number of lines written to the merged file.
    """

    # Open all sorted files appropriately
    file_handles = []
    for fp in file_paths:
        if fp.endswith('.lz4'):
            fh = lz4.frame.open(fp, 'rt', encoding='utf-8')
        else:
            fh = open(fp, 'r', encoding='utf-8')
        file_handles.append(fh)

    # Logging total_lines for debugging
    print(f"DEBUG: Total lines to merge: {total_lines}")

    lines_written = 0
    # Initialize the merging progress bar with units 'lines'
    with create_progress_bar(
        total=total_lines, description="Merging", unit='lines'
    ) as pbar_merge:

        # Open the output file (compressed or not)
        if compress:
            outfile = lz4.frame.open(output_path, 'wb')
        else:
            outfile = open(output_path, 'w', encoding='utf-8')

        with outfile:
            for i, line in enumerate(heapq.merge(*file_handles), 1):
                if compress:
                    outfile.write(line.encode('utf-8') + b'\n')
                else:
                    outfile.write(line)
                lines_written += 1

                # Update the progress bar every 'update_interval' lines
                if i % update_interval == 0:
                    pbar_merge.update(update_interval)

                # Debugging: Check if lines_written exceeds total_lines
                if lines_written > total_lines:
                    print(
                        f"WARNING: lines_written ({lines_written}) exceeds "
                        f"total_lines ({total_lines})"
                    )
                    break  # Prevent exceeding the progress bar total

        # Update the progress bar for any remaining lines
        remaining = lines_written % update_interval
        if remaining:
            pbar_merge.update(remaining)

    # Close all file handles
    for fh in file_handles:
        fh.close()

    # Logging lines_written for debugging
    print(f"DEBUG: Lines written during merging: {lines_written}")

    return lines_written


def sort_and_write_file(args):
    """
    Worker function for multiprocessing.

    Arguments:
        args (tuple):
            file_path (str): Input file to read.
            tmpdir (str): Temporary directory for sorted files.
            idx (int): File index number.
            compress (bool): Whether to compress the sorted output.

    Returns:
        tuple: (path to compressed sorted file, lines in the input file)
    """

    file_path, tmpdir, idx, compress = args
    sorted_file_path = create_sorted_file_path(
        file_path, tmpdir, new_suffix='.jsonl', compress=compress
    )
    line_count = sort_file_in_memory(file_path, sorted_file_path)
    # Logging for debugging
    print(f"DEBUG: Sorted file {sorted_file_path} with {line_count} lines.")
    return sorted_file_path, line_count


def count_total(input_path):
    total_lines = 0
    if input_path.endswith('.lz4'):
        with lz4.frame.open(input_path, 'rt', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
    return total_lines


def consolidate_duplicates(
    input_path, output_path, compress=False,
    update_interval=100000
):
    """
    Consolidate consecutive duplicate ngrams.

    Arguments:
        input_path (str): Source file path (uncompressed).
        output_path (str): Save location (can be compressed).
        compress (bool): Whether to compress the consolidated output.
        update_interval (int): Lines after which to update progress bar.

    Returns:
        int: The number of lines after consolidation.
    """

    count_total = True

    if count_total:
        count_total(input_path)
    else:
        total_lines = None

    line_count_out = 0

    with create_progress_bar(
        total=total_lines, description="Consolidating", unit='lines'
    ) as pbar_consolidate:

        if compress:
            outfile = lz4.frame.open(output_path, 'wb')
        else:
            outfile = open(output_path, 'w', encoding='utf-8')

        with (
            lz4.frame.open(
                input_path, 'rt', encoding='utf-8'
            ) if input_path.endswith('.lz4')
            else open(input_path, 'r', encoding='utf-8')
        ) as infile, outfile:

            current_entry = None

            for i, line in enumerate(infile, 1):
                entry = orjson.loads(line)
                if current_entry is None:
                    current_entry = entry
                else:
                    if entry['ngram'] == current_entry['ngram']:
                        # Merge totals
                        current_entry['freq_tot'] += entry['freq_tot']
                        current_entry['docs_tot'] += entry['docs_tot']

                        # Merge freq dictionary
                        for year, freq_val in entry['freq'].items():
                            current_entry['freq'][year] = (
                                current_entry['freq'].get(year, 0) + freq_val
                            )

                        # Merge docs dictionary
                        for year, doc_val in entry['docs'].items():
                            current_entry['docs'][year] = (
                                current_entry['docs'].get(year, 0) + doc_val
                            )
                    else:
                        # Write current_entry using orjson
                        if compress:
                            outfile.write(
                                orjson.dumps(current_entry) + b'\n'
                            )
                        else:
                            outfile.write(
                                orjson.dumps(
                                    current_entry
                                ).decode('utf-8') + '\n'
                            )
                        line_count_out += 1
                        current_entry = entry
                # Update the progress bar every 'update_interval' lines
                if i % update_interval == 0:
                    pbar_consolidate.update(update_interval)

                # Debugging: Check if i exceeds total_lines if known
                if count_total and i > total_lines:
                    print(f"WARNING: Processed lines ({i}) exceed total_lines "
                          f"({total_lines})")
                    break  # Prevent exceeding the progress bar total

    # Write out the last entry
    if current_entry is not None:
        if compress:
            outfile.write(orjson.dumps(current_entry) + b'\n')
        else:
            outfile.write(orjson.dumps(current_entry).decode('utf-8') + '\n')
        line_count_out += 1

    # Close outfile if not using 'with' (already handled)
    if not compress:
        outfile.close()

    # Logging lines_written for debugging
    print(f"DEBUG: Lines written during consolidation: {line_count_out}")

    return line_count_out


def create_sorted_file_path(
    file_path, tmpdir, new_suffix='.jsonl', compress=True
):
    """
    Removes up to two extensions from original file name, appends new suffixes.

    Args:
        file_path (str or Path): Original file path.
        tmpdir (str or Path): Temporary directory for sorted files.
        new_suffix (str): New suffix to append (default: '.jsonl').
        compress (bool): Whether to append '.lz4' for compression.

    Returns:
        str: Path to the sorted (and possibly compressed) file.
    """

    p = Path(file_path).with_suffix('').with_suffix('')
    new_filename = p.name + new_suffix + ('.lz4' if compress else '')
    return str(Path(tmpdir) / new_filename)


def clear_tmp_directory(tmpdir_parent):
    """
    Clears out the temporary directory to prevent file buildup.

    Args:
        tmpdir_parent (str): Path to the temporary directory.
    """

    if os.path.exists(tmpdir_parent):
        # Remove all contents inside tmpdir_parent
        shutil.rmtree(tmpdir_parent)
    # Recreate the tmpdir_parent
    os.makedirs(tmpdir_parent, exist_ok=True)


def main():
    args = parse_args()

    ngram_size = args.ngram_size
    proj_dir = args.proj_dir
    file_range = args.file_range
    overwrite = args.overwrite
    workers = args.workers
    compress = args.compress  # Capture the compress flag

    start_time = datetime.now()

    input_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/5filter')
    output_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/6corpus')

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(
            f"Input directory {input_dir} does not exist or isn't a directory."
        )

    path_list = [
        entry.path for entry in os.scandir(input_dir) if entry.is_file()
    ]
    path_list.sort()

    num_files_available = len(path_list)

    # If file_range not specified, use all
    if file_range:
        start_idx, end_idx = file_range
        if (
            start_idx < 0 or
            end_idx >= num_files_available or
            start_idx > end_idx
        ):
            raise ValueError("Invalid file_range provided.")
        num_files_to_use = end_idx - start_idx + 1
    else:
        num_files_to_use = len(path_list)
        file_range = (0, len(path_list) - 1)

    first_file = path_list[file_range[0]]
    last_file = path_list[file_range[1]]

    input_paths = path_list[file_range[0]:file_range[1] + 1]

    output_filename = f'{ngram_size}-{str(file_range[0]).zfill(5)}-to-' \
                      f'{str(file_range[1]).zfill(5)}.jsonl'
    if compress:
        output_filename += '.lz4'
    output_path = os.path.join(
        output_dir,
        output_filename
    )

    print_info(input_dir, output_dir, file_range, num_files_available,
               num_files_to_use, first_file, last_file, ngram_size,
               workers, overwrite, start_time)

    if not overwrite and os.path.exists(output_path):
        print(f"Output file {os.path.basename(output_path)} already exists. "
              f"Use --overwrite to replace it.")
        print(f'\nEnd Time:                  {datetime.now()}')
        return

    os.makedirs(output_dir, exist_ok=True)

    # Define temporary directory for sorted files
    tmpdir_parent = os.path.join(proj_dir, f'{ngram_size}gram_files/tmp')

    # Clear out the tmp directory to prevent file buildup
    clear_tmp_directory(tmpdir_parent)

    with tempfile.TemporaryDirectory(dir=tmpdir_parent) as tmpdir:
        # Prepare arguments for multiprocessing
        args_list = [
            (fp, tmpdir, i, compress) for i, fp in enumerate(input_paths)
        ]

        # Initialize and display the sorting progress bar without position
        pbar_sort = create_progress_bar(
            total=len(args_list),
            description="Sorting",
            unit='files'
        )

        # Sort files in parallel
        sorted_file_paths = []
        total_lines = 0
        with multiprocessing.Pool(processes=workers) as pool:
            for sorted_fp, line_count in pool.imap(
                sort_and_write_file, args_list, chunksize=1
            ):
                sorted_file_paths.append(sorted_fp)
                total_lines += line_count
                pbar_sort.update(1)

        pbar_sort.close()  # Close the sorting progress bar

        # Logging total_lines for debugging
        print(f"DEBUG: Total lines to merge: {total_lines}")

        # Initialize the merging progress bar without position
        merged_line_count = heap_merge_files(
            sorted_file_paths,
            output_path,
            total_lines,
            compress=compress,
            update_interval=100000  # Adjust as needed
        )

    # At this point, tmpdir is deleted automatically

    # Consolidate duplicates with a new progress bar without position
    consolidated_output_path = output_path.replace(
        '.jsonl', '-consolidated.jsonl'
    )
    if compress:
        consolidated_output_path += '.lz4'
    consolidated_line_count = consolidate_duplicates(
        input_path=output_path,
        output_path=consolidated_output_path,
        compress=compress,  # Use the compress flag
        update_interval=100000  # Adjust as needed
    )

    # Replace the original merged file with the consolidated file
    shutil.move(consolidated_output_path, output_path)

    # Print line counts before and after consolidation
    print(f"\nTotal lines before combination: {merged_line_count}")
    print(f"Total lines after combination:  {consolidated_line_count}\n")

    end_time = datetime.now()
    print(
        f"Combination complete. Final output file: "
        f"{os.path.basename(output_path)}"
    )

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    main()