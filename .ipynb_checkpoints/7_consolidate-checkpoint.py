import os
import argparse
import lz4.frame
from tqdm import tqdm
import orjson
from datetime import datetime


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} \
    [{elapsed}<{remaining}, {rate_fmt}]"
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

    parser.add_argument('--ngram_size',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        required=True,
                        help='Ngrams size.')
    parser.add_argument("--proj_dir",
                        type=str,
                        required=True,
                        help='Path to the project base directory.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing files?')
    parser.add_argument('--compress',
                        action='store_true',
                        default=False,
                        help='Compress saved files?')

    return parser.parse_args()


def set_info(proj_dir, ngram_size, compress):
    # Construct directory paths
    merged_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/6corpus')

    merged_path = os.path.join(
        merged_dir, f"{ngram_size}gram-merged.jsonl" + (
            '.lz4' if compress else ''
        )
    )

    consolidated_path = os.path.join(
        merged_dir, f"{ngram_size}gram-consolidated.jsonl" + (
            '.lz4' if compress else ''
        )
    )

    return (merged_path, consolidated_path)


def print_info(merged_path, consolidated_path, ngram_size, compress,
               overwrite, start_time):
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mConsolidation Info\033[0m')
    print(f'Merged file:               {merged_path}')
    print(f'Consolidated directory:    {consolidated_path}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Compress output files:     {compress}')
    print(f'Overwrite existing files:  {overwrite}\n')


def create_progress_bar(total, description, unit=''):
    return tqdm(
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


def consolidate_duplicates(input_path, output_path, total_lines, compress):
    input_handler = FileHandler(input_path)
    output_handler = FileHandler(
        output_path, is_output=True, compress=compress
    )

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


def main():
    args = parse_args()

    ngram_size = args.ngram_size
    proj_dir = args.proj_dir
    overwrite = args.overwrite
    compress = args.compress

    start_time = datetime.now()

    (merged_path, consolidated_path) = set_info(
         proj_dir, ngram_size, compress
    )

    print_info(merged_path, consolidated_path, ngram_size, compress,
               overwrite, start_time)

    line_count_in = count_lines(merged_path)

    line_count_out = consolidate_duplicates(
        merged_path, consolidated_path, line_count_in, compress
    )

    print(f'\nLines before consolidation:  {line_count_in}')
    print(f'Lines after consolidation:   {line_count_out}')

    print(f'\033[31m\nEnd Time:                  {datetime.now()}\n\033[0m')


if __name__ == "__main__":
    main()