import os
import argparse
import lz4.frame
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager
import orjson
import heapq
import shutil
import random
import re
from datetime import datetime
from collections import Counter


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} \
    [{elapsed}<{remaining}, {rate_fmt}]"
)

ITER_FILE_REGEX = re.compile(
    r'^merged_iter_(\d+)_chunk_\d+(\.jsonl(\.lz4)?)?$'
)


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


def parse_args():

    parser = argparse.ArgumentParser(
        description="Sort and merge files in parallel."
    )

    parser.add_argument('--ngram_size', type=int,
                        choices=[1, 2, 3, 4, 5],
                        required=True,
                        help='Ngrams size.')
    parser.add_argument('--proj_dir',
                        type=str,
                        required=True,
                        help='Path to the project base directory.')
    parser.add_argument('--file_range',
                        type=int,
                        nargs=2,
                        help='Range of file indices to process.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing files?')
    parser.add_argument('--compress',
                        action='store_true',
                        default=False,
                        help='Compress saved files?')
    parser.add_argument('--workers',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of processors to use.')
    parser.add_argument('--delete_sorted',
                        action='store_true',
                        default=False,
                        help='Delete sorted files after merging?')
    parser.add_argument('--sort_key',
                        type=str,
                        choices=['freq_tot', 'ngram'],
                        default='freq_tot',
                        help="Key to sort on.")
    parser.add_argument('--sort_order',
                        type=str,
                        choices=['ascending', 'descending'],
                        default='descending',
                        help="Sort order.")
    parser.add_argument('--start_iteration',
                        type=int,
                        default=1,
                        help="Iteration to start merging from (default=1).")
    parser.add_argument('--end_iteration',
                        type=int,
                        default=None,
                        help="Iteration to stop with (default=None for all).")

    return parser.parse_args()


def construct_output_path(input_file, output_dir, compress):

    input_path = Path(input_file)
    base_name = (
        input_path.stem if input_path.suffix == '.lz4' else input_path.name
    )
    return str(Path(output_dir) / (base_name + ('.lz4' if compress else '')))


def set_info(proj_dir, ngram_size, file_range, compress):

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

    input_paths_use = input_paths_elig[file_range[0]:file_range[1] + 1]
    num_files_to_use = len(input_paths_use)

    first_file = input_paths_use[0]
    last_file = input_paths_use[-1]

    sorted_paths = sorted([
        construct_output_path(
            file, sorted_dir, compress
        ) for file in input_paths_use
    ])

    merged_path = os.path.join(
        merged_dir, f"{ngram_size}gram-merged.jsonl" + (
            '.lz4' if compress else ''
        )
    )

    return (input_dir, sorted_dir, tmp_dir, merged_path, num_files_available,
            first_file, last_file, num_files_to_use, file_range,
            input_paths_use, sorted_paths)


def print_info(start_time, input_dir, sorted_dir, tmp_dir, merged_path,
               num_files_available, num_files_to_use, first_file, last_file,
               ngram_size, workers, compress, overwrite, sort_key, sort_order,
               start_iteration, end_iteration):

    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mSort Info\033[0m')
    print(f'Input directory:           {input_dir}')
    print(f'Sorted directory:          {sorted_dir}')
    print(f'Temp directory:            {tmp_dir}')
    print(f'Merged file:               {merged_path}')
    print(f'Files available:           {num_files_available}')
    print(f'Files to use:              {num_files_to_use}')
    print(f'First file to get:         {first_file}')
    print(f'Last file to get:          {last_file}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Number of workers:         {workers}')
    print(f'Compress output files:     {compress}')
    print(f'Overwrite existing files:  {overwrite}')
    print(f'Sort key:                  {sort_key}')
    print(f'Sort order:                {sort_order}')
    print(f'Heap-merge start iter:     {start_iteration}')
    print(f'Heap-merge end iter:       {end_iteration}\n')


def create_progress_bar(total, description, unit=''):

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


def process_a_file(args):

    (
        input_handler, output_handler, overwrite, compress,
        sort_key, sort_order
    ) = args

    if not overwrite and os.path.exists(output_handler.path):
        with input_handler.open() as infile:
            line_count = sum(1 for _ in infile)
        return line_count

    with input_handler.open() as infile, output_handler.open() as outfile:
        lines = []

        for line in infile:
            entry = input_handler.deserialize(line)
            if sort_key == 'ngram':
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


def process_a_directory(input_paths, output_dir, output_paths, overwrite,
                        compress, workers, sort_key, sort_order):

    os.makedirs(output_dir, exist_ok=True)

    total_lines_dir = 0
    handlers = [
        (FileHandler(input_path), FileHandler(output_path, is_output=True,
                                              compress=compress))
        for input_path, output_path in zip(input_paths, output_paths)
    ]

    args = [
        (
            input_handler, output_handler, overwrite, compress, sort_key,
            sort_order
        )
        for input_handler, output_handler in handlers
    ]

    with create_progress_bar(len(handlers), "Sorting", 'files') as pbar:
        with Manager() as manager:
            progress = manager.Value('i', 0)

            def update_progress(_):
                progress.value += 1
                pbar.update()

            with Pool(processes=workers) as pool:
                for total_lines_file in pool.imap_unordered(process_a_file,
                                                            args):
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
    end_iteration=None
):

    workers = os.cpu_count()

    # Create tmp_dir if it doesn't exist
    os.makedirs(tmp_dir, exist_ok=True)

    # Iteration 0: the individually sorted files
    iteration_0_files = [
        os.path.join(sorted_dir, f)
        for f in os.listdir(sorted_dir)
        if os.path.isfile(os.path.join(sorted_dir, f))
    ]
    random.shuffle(iteration_0_files)

    # If we're resuming from iteration 1, that means "use iteration 0 files."
    # Otherwise, if start_iteration=3, for example, we need iteration_2 files.
    if start_iteration == 1:
        current_iter_files = iteration_0_files
    else:
        current_iter_files = find_iteration_files(tmp_dir, start_iteration - 1)
        if not current_iter_files:
            raise FileNotFoundError(
                f"No files found for iteration {start_iteration - 1}. "
                f"Cannot resume from iteration {start_iteration}."
            )

    iteration = start_iteration
    while True:
        num_files = len(current_iter_files)

        if num_files <= 1:
            # Final result or nothing to merge
            if num_files == 1 and current_iter_files[0] != merged_path:
                shutil.move(current_iter_files[0], merged_path)
            print(f"Merging complete. Final file: {merged_path}")
            break

        if num_files == 2:
            # Final merge
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
            print(f"\nMerging complete. Final merged file:\n{merged_path}")
            break

        # Stop if we've reached the user's requested end_iteration
        if end_iteration is not None and iteration > end_iteration:
            break

        # Split the current_iter_files into chunks
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

        # Create iteration i chunk filenames
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

        # Perform parallel merges
        with Pool(processes=workers) as pool:
            pool.starmap(
                heap_merge_chunk,
                [
                    (file_chunks[i], chunk_output_paths[i],
                     sort_key, sort_order,
                     compress, total_lines_dir,
                     False,  # no progress bar for chunk merges
                     iteration)
                    for i in range(len(file_chunks))
                ]
            )

        # Now that iteration i is done, we can safely remove iteration i-1
        # (if i-1 >= start_iteration) to save space
        if iteration >= start_iteration:
            remove_iteration_files(tmp_dir, iteration - 1)

        # Prepare for the next iteration
        current_iter_files = chunk_output_paths
        iteration += 1

    # End of while loop


def remove_iteration_files(tmp_dir, iteration):
    """Get merged_iter_{iteration}_chunk_*.jsonl(.lz4) in tmp_dir."""
    if iteration < 1:
        return  # No iteration < 1 to remove
    pattern = re.compile(
        rf"^merged_iter_{iteration}_chunk_\d+(\.jsonl(\.lz4)?)?$"
    )
    for filename in os.listdir(tmp_dir):
        if pattern.match(filename):
            os.remove(os.path.join(tmp_dir, filename))


def find_iteration_files(tmp_dir, iteration):
    """Get iteration files (merged_iter_{iteration}_chunk_*) from tmp_dir."""
    pattern = re.compile(
        rf"^merged_iter_{iteration}_chunk_\d+(\.jsonl(\.lz4)?)?$"
    )
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
    """Merges chunk_files via heapq.merge into output_path."""
    reverse = (sort_order == "descending")

    def merge_key_func(item):
        return item[sort_key]

    output_handler = FileHandler(
        output_path, is_output=True, compress=compress
    )
    file_iters = [
        map(FileHandler(file).deserialize, FileHandler(file).open())
        for file in chunk_files
    ]

    with output_handler.open() as outfile:
        if use_progress_bar and total_lines_dir > 0:
            with create_progress_bar(
                total_lines_dir, "Merging", "lines"
            ) as pbar:
                for item in heapq.merge(
                    *file_iters, key=merge_key_func, reverse=reverse
                ):
                    outfile.write(output_handler.serialize(item))
                    pbar.update(1)
        else:
            for item in heapq.merge(
                *file_iters, key=merge_key_func, reverse=reverse
            ):
                outfile.write(output_handler.serialize(item))


def main():
    args = parse_args()

    ngram_size = args.ngram_size
    proj_dir = args.proj_dir
    file_range = args.file_range
    overwrite = args.overwrite
    compress = args.compress
    workers = args.workers
    sort_key = args.sort_key
    sort_order = args.sort_order
    delete_sorted = args.delete_sorted
    start_iteration = args.start_iteration
    end_iteration = args.end_iteration

    start_time = datetime.now()

    (input_dir, sorted_dir, tmp_dir, merged_path, num_files_available,
     first_file, last_file, num_files_to_use, file_range, input_paths_use,
     sorted_paths) = set_info(proj_dir, ngram_size, file_range, compress)

    print_info(start_time, input_dir, sorted_dir, tmp_dir, merged_path,
               num_files_available, num_files_to_use, first_file, last_file,
               ngram_size, workers, compress, overwrite, sort_key, sort_order,
               start_iteration, end_iteration)

    # Step 1: Initial sorting of input files
    total_lines_dir = process_a_directory(input_paths_use, sorted_dir,
                                          sorted_paths, overwrite, compress,
                                          workers, sort_key, sort_order)

    # Step 2: Iteratively merge sorted files, now with iteration range
    iterative_merge(sorted_dir, tmp_dir, workers, sort_key, sort_order,
                    compress, merged_path, total_lines_dir, start_iteration,
                    end_iteration)

    if delete_sorted:
        shutil.rmtree(sorted_dir)

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    main()