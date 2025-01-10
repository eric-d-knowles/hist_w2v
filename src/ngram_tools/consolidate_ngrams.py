import os
import sys
import argparse
import shutil
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from ngram_tools.helpers.file_handler import FileHandler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort, merge, and consolidate files in parallel."
    )

    parser.add_argument('--ngram_size',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        required=True,
                        help='Ngrams size.')
    parser.add_argument("--proj_dir",
                        type=str,
                        required=True,
                        help='Path to the project base directory.')
    parser.add_argument('--compress',
                        action='store_true',
                        default=False,
                        help='Compress saved files?')
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing files?')
    parser.add_argument('--lines_per_chunk',
                        type=int,
                        default=10_000,
                        help='Number of lines per chunk.')
    parser.add_argument('--workers',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of worker processes to use.')

    return parser.parse_args()


def get_merged_path(merged_dir):
    """
    Look for exactly one file containing '-merged' in its name within
    `merged_dir`. Returns the full path to that file if found, otherwise
    prints an error and exits.
    """
    merged_files = [
        f for f in os.listdir(merged_dir)
        if '-merged' in f and os.path.isfile(os.path.join(merged_dir, f))
    ]

    if len(merged_files) == 0:
        print("Error: No file with '-merged' found in the directory:")
        print(f"  {merged_dir}")
        sys.exit(1)
    elif len(merged_files) > 1:
        print("Error: Multiple files with '-merged' were found. "
              "The script doesn't know which one to use:")
        for file_name in merged_files:
            print(f"  {file_name}")
        sys.exit(1)
    else:
        return os.path.join(merged_dir, merged_files[0])


def set_info(proj_dir, ngram_size, compress):
    merged_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/6corpus')
    merged_path = get_merged_path(merged_dir)
    consolidated_path = os.path.join(
        merged_dir, f"{ngram_size}gram-corpus.jsonl" + (
            '.lz4' if compress else ''
        )
    )
    return merged_path, consolidated_path


def create_temp_dir(ngram_size, proj_dir):
    temp_dir_path = (
        os.path.join(proj_dir, f"{ngram_size}gram_files", "temp_chunks")
    )
    os.makedirs(temp_dir_path, exist_ok=True)
    return temp_dir_path


def print_info(
    start_time,
    merged_path,
    consolidated_path,
    temp_dir_path,
    ngram_size,
    compress,
    overwrite
):
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mConsolidation Info\033[0m')
    print(f'Merged file:               {merged_path}')
    print(f'Corpus file:               {consolidated_path}')
    print(f'Temporary file:            {temp_dir_path}')
    print(f'Temporary directory:       {temp_dir_path}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Compress output files:     {compress}')
    print(f'Overwrite existing files:  {overwrite}\n')


def create_and_process_chunks(
    input_path, lines_per_chunk, temp_dir, compress, workers
):
    """
    Splits the file into chunks, saves them, and processes them in parallel.
    Returns a list of consolidated chunk paths.
    """
    input_handler = FileHandler(input_path)
    consolidated_chunk_paths = []

    with input_handler.open() as f, tqdm(  # Open merged file
        desc="Creating and Processing Chunks",
        unit=" chunks",
        dynamic_ncols=True
    ) as pbar:
        chunk = []  # Initialize an empty chunk
        last_ngram = None

        with Pool(processes=workers) as pool:
            results = []

            for i, line in enumerate(f):
                # Get a line and extract the ngram
                entry = input_handler.deserialize(line)
                current_ngram = entry['ngram']

                # Check if it's time to consolidate and write a chunk
                if (
                    len(chunk) >= lines_per_chunk and
                    current_ngram != last_ngram
                ):
                    chunk_path = os.path.join(
                        temp_dir, f'chunk_{i}.jsonl' + (
                            '.lz4' if compress else ''
                        )
                    )

                    # Make a name for the consolidated chunk
                    consolidated_path = (
                        chunk_path.replace(".jsonl", "_consolidated.jsonl")
                    )

                    # Give chunk to an available worker
                    results.append(
                        pool.apply_async(
                            consolidate_chunk, args=(
                                chunk, consolidated_path, compress
                            )
                        )
                    )

                    chunk = []
                    pbar.update(1)

                chunk.append(line)
                last_ngram = current_ngram

            # Handle the last chunk
            if chunk:
                chunk_path = os.path.join(
                    temp_dir, 'chunk_final.jsonl' + (
                        '.lz4' if compress else ''
                    )
                )
                consolidated_path = (
                    chunk_path.replace(".jsonl", "_consolidated.jsonl")
                )
                results.append(pool.apply_async(
                    consolidate_chunk, args=(
                        chunk, consolidated_path, compress
                    )
                ))
                pbar.update(1)

            # Collect results
            consolidated_chunk_paths = [res.get() for res in results]

    return consolidated_chunk_paths


def consolidate_chunk(lines, output_path, compress):
    """
    Consolidates duplicates in a single chunk and saves it to a new file.
    """
    output_handler = FileHandler(
        output_path, is_output=True, compress=compress
    )
    consolidated = {}

    for line in lines:
        entry = output_handler.deserialize(line)
        ngram = entry['ngram']
        if ngram in consolidated:
            consolidated[ngram]['freq_tot'] += entry['freq_tot']
            consolidated[ngram]['doc_tot'] += entry['doc_tot']
            for year, freq in entry['freq'].items():
                consolidated[ngram]['freq'][year] = (
                    consolidated[ngram]['freq'].get(year, 0) + freq
                )
            for year, doc in entry['doc'].items():
                consolidated[ngram]['doc'][year] = (
                    consolidated[ngram]['doc'].get(year, 0) + doc
                )
        else:
            consolidated[ngram] = entry

    with output_handler.open() as outfile:
        for ngram, data in consolidated.items():
            outfile.write(output_handler.serialize(data))

    return output_path


def merge_chunks(chunk_files, final_output_path, compress):
    """
    Merges all consolidated chunks into the final output file with dynamic
    progress tracking.
    """
    output_handler = FileHandler(
        final_output_path, is_output=True, compress=compress
    )

    with output_handler.open() as outfile, tqdm(
        desc="Merging Chunks", unit=" chunks", dynamic_ncols=True
    ) as pbar:
        for chunk_path in sorted(chunk_files):
            handler = FileHandler(chunk_path)
            with handler.open() as infile:
                for line in infile:
                    outfile.write(line)
                pbar.update(1)


def consolidate_duplicate_ngrams(
    ngram_size,
    proj_dir,
    compress=False,
    overwrite=False,
    lines_per_chunk=10_000,
    workers=os.cpu_count()
):
    """
    Main function that orchestrates chunk creation, consolidation, and merging.
    """
    start_time = datetime.now()

    merged_path, consolidated_path = set_info(proj_dir, ngram_size, compress)
    temp_dir_path = create_temp_dir(ngram_size, proj_dir)

    print_info(
        start_time,
        merged_path,
        consolidated_path,
        temp_dir_path,
        ngram_size,
        compress,
        overwrite
    )

    if not os.path.exists(merged_path):
        print(f"Input file {merged_path} does not exist.")
        sys.exit(1)

    # Phase 1: Create and Process Chunks
    consolidated_chunk_paths = create_and_process_chunks(
        merged_path, lines_per_chunk, temp_dir_path, compress, workers
    )

    # Phase 2: Merge Consolidated Chunks
    merge_chunks(consolidated_chunk_paths, consolidated_path, compress)

    # Delete temporary files and directory
    shutil.rmtree(temp_dir_path)

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    args = parse_args()
    consolidate_duplicate_ngrams(
        ngram_size=args.ngram_size,
        proj_dir=args.proj_dir,
        compress=args.compress,
        overwrite=args.overwrite,
        lines_per_chunk=args.lines_per_chunk,
        workers=args.workers
    )