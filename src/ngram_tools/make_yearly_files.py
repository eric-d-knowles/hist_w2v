import argparse
import os
import sys

from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime

from ngram_embeddings.helpers.file_handler import FileHandler


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


def print_info(
    start_time,
    corpus_path,
    output_dir,
    compress,
    workers,
    overwrite
):
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mProcessing Info\033[0m')
    print(f'Corpus file:               {corpus_path}')
    print(f'Output directory:          {output_dir}')
    print(f'Compress output files:     {compress}')
    print(f'Number of workers:         {workers}')
    print(f'Overwrite existing files:  {overwrite}\n')


def create_progress_bar(total, description, unit=''):
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


def process_ngram_line(line):
    """
    Processes a single n-gram line and organizes it by year.

    Args:
        line (str): A JSONL string containing an n-gram and its metadata.

    Returns:
        dict: A dictionary where years are keys, and n-grams with metadata
              are values.
    """
    data = FileHandler.deserialize(None, line)
    ngram = data['ngram']
    freq = data['freq']
    doc = data['doc']

    yearly_data = {}
    for year, count in freq.items():
        if year not in yearly_data:
            yearly_data[year] = []
        yearly_data[year].append({
            'ngram': ngram,
            'frequency': count,
            'documents': doc[year]
        })
    return yearly_data


def write_yearly_data(yearly_data, output_dir, compress, overwrite):
    """
    Writes yearly n-gram data to output files.

    Args:
        yearly_data (dict): Dictionary with years as keys and n-grams as
           values.
        output_dir (str): Directory to store yearly files.
        compress (bool): Whether to compress the output files.
        overwrite (bool): Whether to overwrite existing files.
    """
    for year, entries in yearly_data.items():
        year_file = os.path.join(output_dir, f"{year}.jsonl")

        if not overwrite and os.path.exists(year_file):
            continue

        handler = FileHandler(year_file, is_output=True, compress=compress)
        with handler.open() as file:
            for entry in entries:
                file.write(handler.serialize(entry))


def process_chunk(chunk, output_dir, compress, overwrite):
    """
    Processes a chunk of n-grams and organizes them into yearly files.

    Args:
        chunk (list): List of JSONL lines to process.
        output_dir (str): Directory to store yearly files.
        compress (bool): Whether to compress the output files.
        overwrite (bool): Whether to overwrite existing files.
    """
    yearly_data_combined = {}
    for line in chunk:
        yearly_data = process_ngram_line(line)
        for year, entries in yearly_data.items():
            if year not in yearly_data_combined:
                yearly_data_combined[year] = []
            yearly_data_combined[year].extend(entries)

    write_yearly_data(yearly_data_combined, output_dir, compress, overwrite)


def process_corpus_file(
    corpus_path,
    output_dir,
    compress=False,
    workers=os.cpu_count(),
    chunk_size=1000,
    overwrite=False
):
    """
    Splits a corpus file into yearly n-gram files.

    Args:
        corpus_path (str): Path to the corpus file.
        output_dir (str): Directory to store yearly files.
        compress (bool): Whether to compress the output files.
        workers (int): Number of workers for multiprocessing.
        chunk_size (int): Number of lines to process per chunk.
        overwrite (bool): Whether to overwrite existing files.
    """
    os.makedirs(output_dir, exist_ok=True)
    handler = FileHandler(corpus_path)

    with handler.open() as file:
        total_lines = sum(1 for _ in file)
        file.seek(0)

        progress_bar = create_progress_bar(
            total_lines,
            "Processing",
            unit="line"
        )

        chunk = []
        with Pool(workers) as pool:
            for line in file:
                chunk.append(line)
                progress_bar.update(1)

                if len(chunk) >= chunk_size:
                    pool.apply_async(
                        process_chunk,
                        args=(chunk,
                              output_dir,
                              compress,
                              overwrite)
                    )
                    chunk = []

            if chunk:
                pool.apply_async(
                    process_chunk,
                    args=(
                        chunk,
                        output_dir,
                        compress,
                        overwrite)
                )

            pool.close()
            pool.join()

        progress_bar.close()


def main():
    parser = argparse.ArgumentParser(
        description="Split a JSONL corpus file into yearly files."
    )
    parser.add_argument(
        "--corpus_path",
        required=True,
        help="Path to the corpus file."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory."
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the output files."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use. Default is all available."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of lines to process per chunk. Default is 1000."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing yearly files."
    )

    args = parser.parse_args()

    start_time = datetime.now()
    formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print_info(
        formatted_start_time,
        args.corpus_path,
        args.output_dir,
        args.compress,
        args.workers,
        args.overwrite
    )

    process_corpus_file(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        compress=args.compress,
        workers=args.workers,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite
    )

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    main()