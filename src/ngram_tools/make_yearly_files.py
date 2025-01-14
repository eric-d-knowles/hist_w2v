import argparse
import os
import sys
import shutil
import uuid
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm

from ngram_tools.helpers.file_handler import FileHandler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a JSONL corpus file into yearly files."
    )
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
        '--compress',
        action='store_true',
        help='Compress the output files.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count(),
        help='Number of worker processes to use. Default is all available.'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=1000,
        help='Number of lines to process per chunk. Default is 1000.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing yearly files.'
    )
    return parser.parse_args()


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


def create_progress_bar(total, description, unit=''):
    """
    Creates a progress bar using tqdm.

    Args:
        total (int): The total number of items to process.
        description (str): The description for the progress bar.
        unit (str): The unit for the progress bar.

    Returns:
        tqdm: The progress bar instance.
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


def get_corpus_path(corpus_dir):
    """
    Look for exactly one file containing '-corpus.' in its name within
    `corpus_dir`. Returns the full path to that file if found, otherwise
    prints an error and exits.
    """
    corpus_files = [
        f for f in os.listdir(corpus_dir)
        if '-corpus.' in f and os.path.isfile(os.path.join(corpus_dir, f))
    ]
    if len(corpus_files) == 0:
        print("Error: No file with '-corpus.' found in the directory:")
        print(f"  {corpus_dir}")
        sys.exit(1)
    elif len(corpus_files) > 1:
        print("Error: Multiple files with '-corpus.' were found. "
              "The script doesn't know which one to use:")
        for file_name in corpus_files:
            print(f"  {file_name}")
        sys.exit(1)
    else:
        return os.path.join(corpus_dir, corpus_files[0])


def set_info(ngram_size, proj_dir, compress):
    """
    Example helper that finds the corpus path and decides on the output directory.
    Adjust if needed for your local naming scheme.
    """
    corpus_dir = os.path.join(proj_dir, f'{ngram_size}gram_files', '6corpus')
    corpus_path = get_corpus_path(corpus_dir)
    yearly_dir = os.path.join(corpus_dir, 'yearly_files')

    return corpus_path, yearly_dir


def print_info(
    start_time,
    corpus_path,
    yearly_dir,
    compress,
    workers,
    overwrite
):
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mProcessing Info\033[0m')
    print(f'Corpus file:               {corpus_path}')
    print(f'Yearly file directory:     {yearly_dir}')
    print(f'Compress output files:     {compress}')
    print(f'Number of workers:         {workers}')
    print(f'Overwrite existing files:  {overwrite}\n')


def process_ngram_line(line):
    """
    Processes a single n-gram line and returns
    {year: [ {ngram, frequency, documents}, ... ] }.
    """
    data = FileHandler.deserialize(None, line)
    ngram = data['ngram']
    freq = data['freq']
    doc = data['doc']

    yearly_data = {}
    for year, count in freq.items():
        yearly_data.setdefault(year, []).append({
            'ngram': ngram,
            'freq': count,
            'doc': doc[year]
        })
    return yearly_data


def chunk_generator(infile, chunk_size, pbar, temp_dir, compress):
    """
    Yields one chunk at a time (plus metadata), so we never store all chunks in memory.
    """
    chunk_buffer = []
    chunk_id = 0
    for line in infile:
        pbar.update(1)     # Update progress bar for each line read
        chunk_buffer.append(line)

        if len(chunk_buffer) >= chunk_size:
            yield (chunk_buffer, chunk_id, temp_dir, compress)
            chunk_buffer = []
            chunk_id += 1

    # Yield any leftover lines as a final chunk
    if chunk_buffer:
        yield (chunk_buffer, chunk_id, temp_dir, compress)


def process_chunk_temp_wrapper(args):
    """
    Unpack the tuple and call process_chunk_temp(chunk, chunk_id, temp_dir, compress).
    """
    chunk, chunk_id, temp_dir, compress = args
    return process_chunk_temp(chunk, chunk_id, temp_dir, compress)


def process_chunk_temp(chunk, chunk_id, temp_dir, compress):
    """
    Process a chunk of lines in memory and write results to worker-specific,
    chunk-specific temp files.  E.g. "temp_dir/1987_chunk-7.jsonl"
    """
    combined_year_data = {}

    for line in chunk:
        yearly_data = process_ngram_line(line)
        for year, entries in yearly_data.items():
            combined_year_data.setdefault(year, []).extend(entries)

    for year, entries in combined_year_data.items():
        temp_file = os.path.join(
            temp_dir,
            f'{year}_chunk-{chunk_id}.jsonl' + ('.lz4' if compress else '')
        )
        output_handler = FileHandler(
            temp_file, is_output=True, compress=compress
        )
        with output_handler.open() as f:
            for entry in entries:
                serialized_line = output_handler.serialize(entry)
                f.write(serialized_line)


def merge_one_year(args):
    """
    Merges all chunk files for a single 'year' into one final file at 'final_path'.
    Returns the 'year' or some status so we can update progress in the main process.
    """
    year, file_list, final_path, compress, overwrite = args

    # If user doesn't want to overwrite existing files, skip
    if not overwrite and os.path.exists(final_path):
        return year  # or return None, etc.

    # Merge all chunk files for this year
    output_handler = FileHandler(final_path, is_output=True, compress=compress)
    with output_handler.open() as out_f:
        for tmp_file in file_list:
            input_handler = FileHandler(tmp_file)
            with input_handler.open() as in_f:
                shutil.copyfileobj(in_f, out_f)

    return year


def merge_temp_files(temp_dir, final_dir, compress, overwrite):
    """
    Merge all per-chunk temp files for each year into a single final
    file, e.g. "1987.jsonl" (or "1987.jsonl.lz4" if compressed).
    Now parallelized per year.
    """
    print("\nMerging temporary files into yearly files:")

    # Decide on chunk-file extension
    chunk_ext = ".jsonl.lz4" if compress else ".jsonl"

    # Gather all chunk files
    all_temp_files = [f for f in os.listdir(temp_dir) if f.endswith(chunk_ext)]

    # Group them by the year prefix
    year_to_tempfiles = defaultdict(list)
    for filename in all_temp_files:
        year = filename.split("_chunk-")[0]
        year_to_tempfiles[year].append(os.path.join(temp_dir, filename))

    os.makedirs(final_dir, exist_ok=True)
    num_unique_years = len(year_to_tempfiles)

    # Build tasks = list of (year, file_list, final_path, compress, overwrite)
    tasks = []
    for year, file_list in year_to_tempfiles.items():
        final_name = f'{year}.jsonl' + ('.lz4' if compress else '')
        final_path = os.path.join(final_dir, final_name)
        tasks.append((year, file_list, final_path, compress, overwrite))

    # Use a Pool of workers to merge different years in parallel
    with create_progress_bar(
        description="Years merged",
        unit=" years",
        total=num_unique_years
    ) as pbar, Pool() as pool:
        # imap_unordered yields results as soon as each worker finishes
        for _ in pool.imap_unordered(merge_one_year, tasks):
            # Each time we get a result (a 'year' merged), update the bar
            pbar.update(1)


def process_corpus_file(
    corpus_path,
    yearly_dir,
    compress=False,
    workers=os.cpu_count(),
    chunk_size=1000,
    overwrite=False
):
    """
    Splits a corpus file into yearly n-gram files using a temp-file approach,
    but uses a generator so we don't keep all chunks in memory.
    """
    temp_dir = os.path.join(yearly_dir, f"temp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    input_handler = FileHandler(corpus_path)

    # -- Open the input file and a progress bar --
    with input_handler.open() as infile, tqdm(
        desc="Reading & chunking",
        unit=" lines",
        dynamic_ncols=True
    ) as pbar, Pool(processes=workers) as pool:

        # 1) Create the generator so it yields chunks on-the-fly
        gen = chunk_generator(
            infile=infile,
            chunk_size=chunk_size,
            pbar=pbar,
            temp_dir=temp_dir,
            compress=compress
        )

        # 2) Map those yielded chunks to your worker function in parallel
        #    Using imap_unordered to keep concurrency and not block.
        for _ in pool.imap_unordered(process_chunk_temp_wrapper, gen):
            # We don't need the return value, so we just discard with `_`.
            # This loop consumes the generator, ensuring lines get read
            # and tasks get dispatched to workers.
            pass

        # 3) Once the generator is exhausted and workers finish, clean up
        pool.close()
        pool.join()

    # -- Now merge the temp chunk files by year --
    merge_temp_files(temp_dir, yearly_dir, compress, overwrite)
    # Optionally remove the temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def make_yearly_files(
    ngram_size,
    proj_dir,
    overwrite=False,
    compress=False,
    workers=os.cpu_count(),
    chunk_size=1000
):
    start_time = datetime.now()

    corpus_path, yearly_dir = set_info(ngram_size, proj_dir, compress)

    print_info(
        start_time,
        corpus_path,
        yearly_dir,
        compress,
        workers,
        overwrite
    )

    process_corpus_file(
        corpus_path=corpus_path,
        yearly_dir=yearly_dir,
        compress=compress,
        overwrite=overwrite,
        workers=workers,
        chunk_size=chunk_size
    )

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    args = parse_args()
    make_yearly_files(
        ngram_size=args.ngram_size,
        proj_dir=args.proj_dir,
        overwrite=args.overwrite,
        compress=args.compress,
        workers=args.workers,
        chunk_size=args.chunk_size
    )