import re
import os
import orjson
from pathlib import Path
import lz4.frame
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import nltk
from nltk.corpus import stopwords
from datetime import datetime


nltk.download('stopwords', quiet=True)

NUMERALS_REGEX = re.compile(r'\d')
NONALPHA_REGEX = re.compile(r'[^a-zA-Z]')

global_vocab_set = frozenset()
global_stopword_set = frozenset()
filters_global = {}
min_tokens_global = 2

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


def initializer(vocab_set, stopword_set, filters, min_tokens):
    global global_vocab_set
    global global_stopword_set
    global filters_global
    global min_tokens_global

    global_vocab_set = vocab_set
    global_stopword_set = stopword_set
    filters_global = filters
    min_tokens_global = min_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Filter ngrams.")

    parser.add_argument('--ngram_size',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        required=True,
                        help='Ngrams size to get (required).')
    parser.add_argument("--proj_dir",
                        type=str,
                        required=True,
                        help='Path to the project base directory.')
    parser.add_argument('--file_range',
                        type=int,
                        nargs=2,
                        help='Range of file indices to get [default = all].')
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing files? (default=False).')
    parser.add_argument('--compress',
                        action='store_true',
                        default=False,
                        help='Compress saved files? (default=False).')
    parser.add_argument('--workers',
                        type=int,
                        default=cpu_count(),
                        help='Number of processors to use (default=all]).')
    parser.add_argument('--stopwords',
                        action='store_true',
                        default=False,
                        help='Filter out stopwords.')
    parser.add_argument('--min_token_length',
                        type=int,
                        default=0,
                        help='Minimum token length to retain (default=0).')
    parser.add_argument('--numerals',
                        action='store_true',
                        default=False,
                        help='Filter out tokens containing numerals.')
    parser.add_argument('--nonalpha',
                        action='store_true',
                        default=False,
                        help='Filter out tokens with non-alpha characters.')
    parser.add_argument('--min_tokens',
                        type=int,
                        default=2,
                        help='Shortest filtered ngrams to retain.')
    parser.add_argument('--vocab_file',
                        type=str,
                        help='Relative path to a vocabulary file.')
    parser.add_argument('--delete_input',
                        action='store_true',
                        default=False,
                        help='Delete input files (default=True).')

    return parser.parse_args()


def construct_output_path(input_file, output_dir, compress):
    input_path = Path(input_file)
    base_name = (
        input_path.stem if input_path.suffix == '.lz4' else input_path.name
    )
    return str(Path(output_dir) / (base_name + ('.lz4' if compress else '')))


def set_info(proj_dir, ngram_size, file_range, compress):
    input_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/4lemmatize')
    output_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/5filter')

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

    output_paths = sorted([
        construct_output_path(
            file, output_dir, compress
        ) for file in input_paths_use
    ])

    return (input_dir, output_dir, num_files_available, num_files_to_use,
            first_file, last_file, num_files_to_use, file_range,
            input_paths_use, output_paths)


def print_info(input_dir, output_dir, file_range, num_files_available,
               num_files_to_use, first_file, last_file, ngram_size, workers,
               compress, overwrite, filters, min_tokens, start_time,
               vocab_file, delete_input):
    print(f'\033[31mStart Time:                   {start_time}\n\033[0m')
    print('\033[4mFiltering Info\033[0m')
    print(f'Input directory:              {input_dir}')
    print(f'Output directory:             {output_dir}')
    print(f'File index range:             {file_range[0]} to {file_range[1]}')
    print(f'Files available:              {num_files_available}')
    print(f'Files to use:                 {num_files_to_use}')
    print(f'First file to get:            {first_file}')
    print(f'Last file to get:             {last_file}')
    print(f'Ngram size:                   {ngram_size}')
    print(f'Number of workers:            {workers}')
    print(f'Compress output files:        {compress}')
    print(f'Overwrite existing files:     {overwrite}')
    print(f'Delete input directory:       {delete_input}\n')
    print('\033[4mFiltering Options\033[0m')
    print(f'Drop stopwords:               {filters["stopwords"]}')
    print(f'Drop tokens under:            {filters["min_token_length"]} chars')
    print(f'Drop tokens with numerals:    {filters["numerals"]}')
    print(f'Drop non-alphabetic:          {filters["nonalpha"]}')
    print(f'Drop ngrams under:            {min_tokens} token(s)')
    if vocab_file:
        print(f'Vocab file:                   {vocab_file}')
    print()


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


def passes_filters(token):
    token_lower = token.lower()

    # Vocabulary Check
    if filters_global.get('vocab_file') and token_lower not in global_vocab_set:
        return False, 'dropped_vocab'

    # Stopwords Check
    if filters_global.get('stopwords') and token_lower in global_stopword_set:
        return False, 'dropped_stop'

    # Numerals Check
    if filters_global.get('numerals') and NUMERALS_REGEX.search(token):
        return False, 'dropped_numeral'

    # Non-alpha Check
    if filters_global.get('nonalpha') and NONALPHA_REGEX.search(token):
        return False, 'dropped_nonalpha'

    # Minimum Token Length Check
    if (
        filters_global.get('min_token_length', 0) > 0 and
        len(token) < filters_global['min_token_length']
    ):
        return False, 'dropped_short'

    return True, None


def process_a_line(ngram_dict):
    local_counts = {
        'dropped_stop': 0,
        'dropped_short': 0,
        'dropped_numeral': 0,
        'dropped_nonalpha': 0,
        'dropped_vocab': 0,
        'dropped_ngrams': 0
    }

    filtered_ngram = {}
    for token, word in ngram_dict.get('ngram', {}).items():

        passes, count_key = passes_filters(word)
        if passes:
            filtered_ngram[token] = word
        else:
            if count_key:
                local_counts[count_key] += 1

    if len(filtered_ngram) < min_tokens_global:
        local_counts['dropped_ngrams'] += 1
        return None, local_counts

    ngram_dict['ngram'] = filtered_ngram

    return ngram_dict, local_counts


def process_a_file(args):
    input_handler, output_handler, overwrite = args

    file_counts = {
        'dropped_stop': 0,
        'dropped_short': 0,
        'dropped_numeral': 0,
        'dropped_nonalpha': 0,
        'dropped_vocab': 0,
        'dropped_ngrams': 0
    }

    if not overwrite and os.path.exists(output_handler.path):
        return

    try:
        with input_handler.open() as infile, output_handler.open() as outfile:
            for line_no, line in enumerate(infile, start=1):
                if isinstance(line, bytes):
                    line = line.decode('utf-8')

                data = input_handler.deserialize(line)

                filtered_data, line_counts = process_a_line(data)

                for key in file_counts:
                    if key in line_counts:
                        file_counts[key] += line_counts[key]

                if filtered_data is not None:
                    outfile.write(output_handler.serialize(filtered_data))

    except Exception as e:
        print(f"Error processing {input_handler.path}: {e}")

    output_ext = Path(output_handler.path).suffix
    output_size = os.path.getsize(output_handler.path)

    if (
        (output_ext == '.lz4' and output_size == 11) or
        (output_ext == '.jsonl' and output_size == 0)
    ):
        os.remove(output_handler.path)

    return file_counts


def process_a_directory(output_dir, input_paths, output_paths, overwrite,
                        compress, workers, filters, min_tokens, vocab_set,
                        stopword_set):
    os.makedirs(output_dir, exist_ok=True)

    handlers = []
    for input_path, output_path in zip(input_paths, output_paths):

        input_ext = Path(input_path).suffix
        input_size = os.path.getsize(input_path)

        if (input_ext == '.jsonl' and input_size == 0) or \
           (input_ext == '.lz4' and input_size == 11):
            continue

        input_handler = FileHandler(input_path)
        output_handler = FileHandler(
            output_path, is_output=True, compress=compress
        )
        handlers.append((input_handler, output_handler))

    args = [
        (input_handler, output_handler, overwrite)
        for input_handler, output_handler in handlers
    ]

    print('\033[4mFiltering Progress\033[0m')

    with Pool(
        processes=workers,
        initializer=initializer,
        initargs=(vocab_set, stopword_set, filters, min_tokens)
    ) as pool:
        args = [
            (input_handler, output_handler, overwrite)
            for input_handler, output_handler in handlers
        ]

        with create_progress_bar(len(handlers), "Filtering", 'files') as pbar:
            results = []
            for result in pool.imap_unordered(process_a_file, args):
                results.append(result)
                pbar.update()

    agg_counters = {
        'dropped_stop': 0,
        'dropped_short': 0,
        'dropped_numeral': 0,
        'dropped_nonalpha': 0,
        'dropped_vocab': 0,
        'dropped_ngrams': 0
    }

    if any(item is not None for item in results):
        for result in results:
            for key in agg_counters:
                agg_counters[key] += result.get(key, 0)

    return agg_counters


def clear_directory(directory_path):
    for entry in os.scandir(directory_path):
        if entry.is_file():
            os.remove(entry.path)
        elif entry.is_dir():
            os.rmdir(entry.path)


def main():
    args = parse_args()

    ngram_size = args.ngram_size
    file_range = args.file_range
    proj_dir = args.proj_dir
    overwrite = args.overwrite
    compress = args.compress
    workers = args.workers
    min_tokens = args.min_tokens
    delete_input = args.delete_input

    start_time = datetime.now()

    filters = {
        'stopwords': args.stopwords,
        'min_token_length': args.min_token_length,
        'numerals': args.numerals,
        'nonalpha': args.nonalpha
    }

    stopword_set = frozenset(
        stopwords.words('english')
    ) if filters['stopwords'] else frozenset()

    # Add vocab_file to filters if provided
    vocab_file = None
    vocab_set = frozenset()
    if args.vocab_file:
        vocab_file = os.path.join(proj_dir, args.vocab_file)
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"Vocab file not found at {vocab_file}")

        filters['vocab_file'] = True
        with open(vocab_file, 'r', encoding='utf-8') as vf:
            vocab_list = {line.strip().lower() for line in vf if line.strip()}
        vocab_set = frozenset(vocab_list)
    else:
        filters['vocab_file'] = False

    (
        input_dir, output_dir, num_files_available, num_files_to_use,
        first_file, last_file, num_files_to_use, file_range, input_paths,
        output_paths
    ) = set_info(
         proj_dir, ngram_size, file_range, compress
    )

    # Print configuration info
    print_info(input_dir, output_dir, file_range, num_files_available,
               num_files_to_use, first_file, last_file, ngram_size, workers,
               compress, overwrite, filters, min_tokens, start_time,
               vocab_file, delete_input)

    # Process the directory with multiprocessing
    agg_counters = process_a_directory(
        output_dir, input_paths, output_paths, overwrite, compress, workers,
        filters, min_tokens, vocab_set, stopword_set
    )

    end_time = datetime.now()

    # Print line counts before and after consolidation
    print('\n\033[4mFiltering Results (Dropped)\033[0m')
    print(f'Stopword tokens:              {agg_counters["dropped_stop"]} ')
    print(f'Short-word tokens:            {agg_counters["dropped_short"]} ')
    print(f'Tokens with numerals:         {agg_counters["dropped_numeral"]} ')
    print(f'Tokens with non-alpha chars:  {agg_counters["dropped_nonalpha"]}')
    print(f'Out-of-vocab tokens:          {agg_counters["dropped_vocab"]}')
    print(f'Entire ngrams:                {agg_counters["dropped_ngrams"]} ')

    if delete_input:
        clear_directory(input_dir)

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == '__main__':
    main()