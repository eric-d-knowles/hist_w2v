import argparse
import os
import sys
import shutil
import uuid
import gc
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


def merge_temp_files(temp_dir, final_dir, compress, overwrite):
    """
    Merge all per-chunk temp files for each year into a single final
    file, e.g. "1987.jsonl" (or "1987.jsonl.lz4" if compressed).
    """
    print("\nMerging temporary files into yearly files:")

    if compress:
        chunk_ext = ".jsonl.lz4"
    else:
        chunk_ext = ".jsonl"

    all_temp_files = [
        f for f in os.listdir(temp_dir)
        if f.endswith(chunk_ext)
    ]

    from collections import defaultdict
    year_to_tempfiles = defaultdict(list)
    for filename in all_temp_files:
        year = filename.split("_chunk-")[0]
        year_to_tempfiles[year].append(os.path.join(temp_dir, filename))
    num_unique_years = len(year_to_tempfiles)

    os.makedirs(final_dir, exist_ok=True)

    with create_progress_bar(
        description="Years merged",
        unit=" years",
        total=len(year_to_tempfiles)
    ) as pbar:
        for year, file_list in year_to_tempfiles.items():
            final_name = f'{year}.jsonl' + ('.lz4' if compress else '')
            final_path = os.path.join(final_dir, final_name)

            if not overwrite and os.path.exists(final_path):
                continue

            output_handler = FileHandler(
                final_path,
                is_output=True,
                compress=compress
            )
            with output_handler.open() as out_f:
                for tmp_file in file_list:
                    input_handler = FileHandler(tmp_file)
                    with input_handler.open() as in_f:
                        shutil.copyfileobj(in_f, out_f)
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
    Splits a corpus file into yearly n-gram files using a temp-file approach.
    """
    temp_dir = os.path.join(yearly_dir, f"temp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    input_handler = FileHandler(corpus_path)
    chunk_buffer = []
    chunk_id = 0

    with input_handler.open() as infile, tqdm(
        desc="Reading and chunking",
        unit=" lines",
        dynamic_ncols=True
    ) as pbar, Pool(processes=workers) as pool:
        for line in infile:
            pbar.update(1)
            chunk_buffer.append(line)
            if len(chunk_buffer) >= chunk_size:
                pool.apply(
                    process_chunk_temp,
                    args=(chunk_buffer, chunk_id, temp_dir, compress)
                )
                chunk_buffer = []
                chunk_id += 1
                gc.collect()

        if chunk_buffer:
            pool.apply(
                process_chunk_temp,
                args=(chunk_buffer, chunk_id, temp_dir)
            )
            chunk_buffer = []
            chunk_id += 1
            gc.collect()

        pool.close()
        pool.join()

    merge_temp_files(temp_dir, yearly_dir, compress, overwrite)

    # Optionally, remove the temp_dir
    # shutil.rmtree(temp_dir, ignore_errors=True)


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