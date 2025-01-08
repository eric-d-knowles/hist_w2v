import os
import sys
import argparse
from tqdm import tqdm
from datetime import datetime

from ngram_embeddings.helpers.file_handler import FileHandler


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} \
    [{elapsed}<{remaining}, {rate_fmt}]"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort and merge files in parallel."
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

    return parser.parse_args()


def get_merged_path(merged_dir):
    """
    Look for exactly one file containing '-merge' in its name within `merged_dir`.
    Returns the full path to that file if found, otherwise prints an error and exits.
    """
    # Collect all files containing '-merge' in the name
    merged_files = [
        f for f in os.listdir(merged_dir)
        if '-merge' in f and os.path.isfile(os.path.join(merged_dir, f))
    ]

    if len(merged_files) == 0:
        print("Error: No file with '-merge' was found in the directory:")
        print(f"  {merged_dir}")
        sys.exit(1)
    elif len(merged_files) > 1:
        print("Error: Multiple files with '-merge' were found. "
              "The script doesn't know which one to use:")
        for file_name in merged_files:
            print(f"  {file_name}")
        sys.exit(1)
    else:
        # Exactly one matching file
        return os.path.join(merged_dir, merged_files[0])


def set_info(proj_dir, ngram_size, compress):
    merged_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/6corpus')

    merged_path = get_merged_path(merged_dir)

    consolidated_path = os.path.join(
        merged_dir, f"{ngram_size}gram-corpus.jsonl" + (
            '.lz4' if compress else ''
        )
    )

    return (merged_path, consolidated_path)


def print_info(start_time, merged_path, consolidated_path, ngram_size, compress,
               overwrite):
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mConsolidation Info\033[0m')
    print(f'Merged file:               {merged_path}')
    print(f'Corpus file:               {consolidated_path}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Compress output files:     {compress}')
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


def count_lines(merged_path):
    input_handler = FileHandler(merged_path)

    with input_handler.open() as infile:
        line_count = sum(1 for _ in infile)
    return line_count


def consolidate_duplicates(input_path, output_path, total_lines, compress,
                           overwrite):
    input_handler = FileHandler(input_path, is_output=False)
    output_handler = FileHandler(
        output_path, is_output=True, compress=compress
    )

    if not overwrite and os.path.exists(output_handler.path):
        print(f'File {output_path} already exists, '
              'and overwrite is not specified')
        return

    line_count_out = 0
    BATCH_SIZE = 10_000  # Adjust as needed

    with input_handler.open() as infile, output_handler.open() as outfile, \
         create_progress_bar(
             total=total_lines, description="Consolidating", unit='lines'
         ) as pbar:
        current_entry = None

        count = 0  # Track how many lines we've processed so far
        for line in infile:
            entry = input_handler.deserialize(line)

            if current_entry is None:
                current_entry = entry
            elif entry['ngram'] == current_entry['ngram']:
                # Merge duplicate entries
                current_entry['freq_tot'] += entry['freq_tot']
                current_entry['doc_tot'] += entry['doc_tot']
                for year, freq_val in entry['freq'].items():
                    current_entry['freq'][year] = (
                        current_entry['freq'].get(year, 0) + freq_val
                    )
                for year, doc_val in entry['doc'].items():
                    current_entry['doc'][year] = (
                        current_entry['doc'].get(year, 0) + doc_val
                    )
            else:
                # Write the previous consolidated entry
                outfile.write(output_handler.serialize(current_entry))
                line_count_out += 1
                current_entry = entry

            count += 1
            if count % BATCH_SIZE == 0:
                pbar.update(BATCH_SIZE)

        # Write the last accumulated entry
        if current_entry is not None:
            outfile.write(output_handler.serialize(current_entry))
            line_count_out += 1

        # Update the progress bar for the leftover lines
        pbar.update(count % BATCH_SIZE)

    return line_count_out


def consolidate_duplicate_ngrams(
    ngram_size,
    proj_dir,
    compress=False,
    overwrite=False
):
    start_time = datetime.now()

    (merged_path, consolidated_path) = set_info(proj_dir, ngram_size, compress)

    print_info(
        start_time,
        merged_path,
        consolidated_path,
        ngram_size,
        compress,
        overwrite,
    )

    line_count_in = count_lines(merged_path)

    line_count_out = consolidate_duplicates(
        merged_path, consolidated_path, line_count_in, compress, overwrite
    )

    print(f'\nLines before consolidation:  {line_count_in}')
    print(f'Lines after consolidation:   {line_count_out}')

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    args = parse_args()
    consolidate_duplicate_ngrams(
        ngram_size=args.ngram_size,
        proj_dir=args.proj_dir,
        overwrite=args.overwrite,
        compress=args.compress
    )