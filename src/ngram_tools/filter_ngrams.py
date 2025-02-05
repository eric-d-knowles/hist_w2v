import argparse
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from tqdm.notebook import tqdm

from ngram_tools.helpers.file_handler import FileHandler


nltk.download('stopwords', quiet=True)

NUMERALS_REGEX = re.compile(r'\d')
NONALPHA_REGEX = re.compile(r'[^a-zA-Z]')

# Global variables for multiprocessing initialization
global_vocab_set = frozenset()
global_stopword_set = frozenset()
filters_global = {}
min_tokens_global = 2


def initializer(vocab_set, stopword_set, filters, min_tokens):
    """
    Initialize global variables in each worker process.

    Args:
        vocab_set (frozenset): A set of valid vocabulary tokens (if applicable).
        stopword_set (frozenset): A set of stopwords to filter out (if applicable).
        filters (dict): A dictionary of filter flags and parameters.
        min_tokens (int): The minimum number of tokens for an ngram to be retained.
    """
    global global_vocab_set
    global global_stopword_set
    global filters_global
    global min_tokens_global

    global_vocab_set = vocab_set
    global_stopword_set = stopword_set
    filters_global = filters
    min_tokens_global = min_tokens


def construct_output_path(input_file, output_dir, compress):
    """
    Construct the output file path based on the input file and compression flag.

    Args:
        input_file (str): Path to the input file.
        output_dir (str): Directory where the output will be written.
        compress (bool): Whether output files should be compressed (lz4).

    Returns:
        str: The constructed output file path.
    """
    input_path = Path(input_file)
    # If the file has a ".lz4" suffix, remove it before constructing the base name.
    base_name = input_path.stem if input_path.suffix == '.lz4' else input_path.name
    return str(Path(output_dir) / (base_name + ('.lz4' if compress else '')))


def set_info(proj_dir, ngram_size, file_range, compress):
    """
    Prepare necessary information about input and output directories and file paths.

    Args:
        proj_dir (str): Base directory of the project.
        ngram_size (int): Size of the ngrams (1–5).
        file_range (tuple[int], optional): Range of file indices to process.
        compress (bool): Whether output files should be compressed (lz4).

    Returns:
        tuple:
            - input_dir (str): Path to the input directory.
            - output_dir (str): Path to the output directory.
            - num_files_available (int): Count of available input files.
            - num_files_to_use (int): Count of files to be processed.
            - first_file (str): Path to the first file in the range.
            - last_file (str): Path to the last file in the range.
            - num_files_to_use (int): Same as above (for convenience).
            - file_range (tuple[int]): Possibly adjusted file range.
            - input_paths_use (list[str]): Paths of input files to process.
            - output_paths (list[str]): Corresponding output file paths.
    """
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

    input_paths_use = input_paths_elig[file_range[0]: file_range[1] + 1]
    num_files_to_use = len(input_paths_use)

    first_file = input_paths_use[0]
    last_file = input_paths_use[-1]

    output_paths = sorted(
        construct_output_path(file, output_dir, compress)
        for file in input_paths_use
    )

    return (
        input_dir,
        output_dir,
        num_files_available,
        num_files_to_use,
        first_file,
        last_file,
        num_files_to_use,
        file_range,
        input_paths_use,
        output_paths
    )


def print_info(
    input_dir,
    output_dir,
    file_range,
    num_files_available,
    num_files_to_use,
    first_file,
    last_file,
    ngram_size,
    workers,
    compress,
    overwrite,
    filters,
    min_tokens,
    start_time,
    vocab_file,
    delete_input
):
    """
    Print a summary of the filtering operation and its parameters.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        file_range (tuple[int]): Range of file indices to process.
        num_files_available (int): Total number of available files in input_dir.
        num_files_to_use (int): Number of files that will be processed.
        first_file (str): Path to the first file in the range.
        last_file (str): Path to the last file in the range.
        ngram_size (int): Size of the ngrams (1–5).
        workers (int): Number of worker processes to use.
        compress (bool): Whether output files should be compressed.
        overwrite (bool): Whether existing files may be overwritten.
        filters (dict): Dictionary of filtering flags/parameters.
        min_tokens (int): Minimum number of tokens per ngram to retain.
        start_time (datetime): The time the process started.
        vocab_file (str or None): Path to a vocabulary file, if any.
        delete_input (bool): Whether input files will be deleted after processing.
    """
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


def passes_filters(token):
    """
    Check whether a token passes the global filtering criteria.

    Args:
        token (str): The token to evaluate.

    Returns:
        tuple:
            - (bool) True if the token passes all filters, otherwise False.
            - (str or None) The reason key if it fails, else None.
    """
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
        filters_global.get('min_token_length', 0) > 0
        and len(token) < filters_global['min_token_length']
    ):
        return False, 'dropped_short'

    return True, None


def process_a_line(ngram_dict):
    """
    Process a single JSON-decoded line (representing an ngram), filtering tokens
    based on global criteria.

    Args:
        ngram_dict (dict): The JSON-decoded dictionary with key 'ngram'.

    Returns:
        tuple:
            - (dict or None) The updated dictionary if retained, else None.
            - (dict) A dictionary counting dropped tokens or ngrams.
    """
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
        passes_check, count_key = passes_filters(word)
        if passes_check:
            filtered_ngram[token] = word
        else:
            if count_key:
                local_counts[count_key] += 1

    # If the ngram is too short after filtering, drop it entirely.
    if len(filtered_ngram) < min_tokens_global:
        local_counts['dropped_ngrams'] += 1
        return None, local_counts

    ngram_dict['ngram'] = filtered_ngram
    return ngram_dict, local_counts


def process_a_file(args):
    """
    Process a single file: read each JSON line, apply filtering, and write output.

    Args:
        args (tuple):
            - input_handler (FileHandler): Handler for reading the input file.
            - output_handler (FileHandler): Handler for writing the output file.
            - overwrite (bool): Whether to overwrite existing output files.

    Returns:
        dict: A dictionary of counts for dropped tokens/ngrams in this file.
    """
    input_handler, output_handler, overwrite = args

    file_counts = {
        'dropped_stop': 0,
        'dropped_short': 0,
        'dropped_numeral': 0,
        'dropped_nonalpha': 0,
        'dropped_vocab': 0,
        'dropped_ngrams': 0
    }

    # If output file exists and overwrite is False, skip processing.
    if not overwrite and os.path.exists(output_handler.path):
        return

    try:
        with input_handler.open() as infile, output_handler.open() as outfile:
            for line_no, line in enumerate(infile, start=1):
                if isinstance(line, bytes):
                    line = line.decode('utf-8')

                data = input_handler.deserialize(line)
                filtered_data, line_counts = process_a_line(data)

                # Aggregate line-level counts into file-level counts.
                for key, value in line_counts.items():
                    file_counts[key] += value

                if filtered_data is not None:
                    outfile.write(output_handler.serialize(filtered_data))
    except Exception as exc:
        print(f"Error processing {input_handler.path}: {exc}")

    # Remove empty or minimal-size output files.
    output_ext = Path(output_handler.path).suffix
    if os.path.exists(output_handler.path):
        output_size = os.path.getsize(output_handler.path)
        if (
            (output_ext == '.lz4' and output_size == 11)
            or (output_ext == '.jsonl' and output_size == 0)
        ):
            os.remove(output_handler.path)

    return file_counts


def process_a_directory(
    output_dir,
    input_paths,
    output_paths,
    overwrite,
    compress,
    workers,
    filters,
    min_tokens,
    vocab_set,
    stopword_set
):
    """
    Process multiple files in a directory using a pool of worker processes.

    Args:
        output_dir (str): Path to the output directory.
        input_paths (list[str]): Paths to the input files.
        output_paths (list[str]): Paths to the output files.
        overwrite (bool): Whether to overwrite existing output files.
        compress (bool): Whether output files should be compressed (lz4).
        workers (int): Number of worker processes.
        filters (dict): Filter parameters (stopwords, numerals, etc.).
        min_tokens (int): Minimum number of tokens per ngram to retain.
        vocab_set (frozenset): Set of allowed vocabulary tokens.
        stopword_set (frozenset): Set of stopword tokens to remove.

    Returns:
        dict: Aggregate counts of dropped tokens/ngrams across all files.
    """
    os.makedirs(output_dir, exist_ok=True)

    handlers = []
    for input_path, output_path in zip(input_paths, output_paths):
        input_ext = Path(input_path).suffix
        input_size = os.path.getsize(input_path)

        # Skip empty files.
        if (
            (input_ext == '.jsonl' and input_size == 0)
            or (input_ext == '.lz4' and input_size == 11)
        ):
            continue

        in_handler = FileHandler(input_path)
        out_handler = FileHandler(output_path, is_output=True, compress=compress)
        handlers.append((in_handler, out_handler))

    args = [
        (in_handler, out_handler, overwrite)
        for in_handler, out_handler in handlers
    ]

    with tqdm(total=len(handlers), desc='Filtering', unit='files') as pbar:
        with Pool(
            processes=workers,
            initializer=initializer,
            initargs=(vocab_set, stopword_set, filters, min_tokens)
        ) as pool:
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

    # Consolidate counts from all worker results.
    if any(item is not None for item in results):
        for file_result in results:
            if file_result is not None:
                for key in agg_counters:
                    agg_counters[key] += file_result.get(key, 0)

    return agg_counters


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


def filter_ngrams(
    ngram_size,
    proj_dir,
    file_range=None,
    overwrite=False,
    compress=False,
    workers=os.cpu_count(),
    numerals=True,
    nonalpha=True,
    stops=True,
    min_token_length=3,
    vocab_file=None,
    min_tokens=2,
    delete_input=False
):
    """
    Main function to filter ngrams based on various flags:
    stopwords, numerals, non-alpha chars, short tokens, or an external vocabulary.

    Args:
        ngram_size (int): Size of the ngrams (1–5).
        proj_dir (str): Base directory path for the project.
        file_range (tuple[int], optional): Range of file indices to process.
        overwrite (bool, optional): Whether to overwrite existing files.
        compress (bool, optional): Whether to compress output files (lz4).
        workers (int, optional): Number of parallel worker processes.
        numerals (bool, optional): Drop tokens containing digits if True.
        nonalpha (bool, optional): Drop tokens with non-alphabetic chars if True.
        stops (bool, optional): Drop stopwords if True.
        min_token_length (int, optional): Minimum token length to retain.
        vocab_file (str, optional): Relative path to a vocabulary file.
        min_tokens (int, optional): Minimum tokens in an ngram to retain.
        delete_input (bool, optional): Remove input directory after processing.
    """
    start_time = datetime.now()

    filters = {
        'stopwords': stops,
        'min_token_length': min_token_length,
        'numerals': numerals,
        'nonalpha': nonalpha
    }

    # Build a stopword set if requested.
    stopword_set = (
        frozenset(stopwords.words('english'))
        if filters['stopwords']
        else frozenset()
    )

    # Build a vocabulary set if the user provides a vocab_file
    vocab_set = frozenset()
    if vocab_file:
        vocab_path = os.path.join(proj_dir, '1gram_files/6corpus/', vocab_file)
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"Vocab file not found at {vocab_path}")

        filters['vocab_file'] = True
        with open(vocab_path, 'r', encoding='utf-8') as vf:
            vocab_list = {line.strip().lower() for line in vf if line.strip()}
        vocab_set = frozenset(vocab_list)
    else:
        filters['vocab_file'] = False

    # Gather info about the input/output directories and file paths.
    (
        input_dir,
        output_dir,
        num_files_available,
        num_files_to_use,
        first_file,
        last_file,
        num_files_to_use,
        file_range,
        input_paths,
        output_paths
    ) = set_info(proj_dir, ngram_size, file_range, compress)

    # Print configuration info
    print_info(
        input_dir,
        output_dir,
        file_range,
        num_files_available,
        num_files_to_use,
        first_file,
        last_file,
        ngram_size,
        workers,
        compress,
        overwrite,
        filters,
        min_tokens,
        start_time,
        vocab_file,
        delete_input
    )

    # Process files in the directory
    agg_counters = process_a_directory(
        output_dir,
        input_paths,
        output_paths,
        overwrite,
        compress,
        workers,
        filters,
        min_tokens,
        vocab_set,
        stopword_set
    )

    end_time = datetime.now()

    # Print summary of dropped tokens/ngrams
    print('\n\033[4mFiltering Results (Dropped)\033[0m')
    print(f'Stopword tokens:              {agg_counters["dropped_stop"]} ')
    print(f'Short-word tokens:            {agg_counters["dropped_short"]} ')
    print(f'Tokens with numerals:         {agg_counters["dropped_numeral"]} ')
    print(f'Tokens with non-alpha chars:  {agg_counters["dropped_nonalpha"]}')
    print(f'Out-of-vocab tokens:          {agg_counters["dropped_vocab"]}')
    print(f'Entire ngrams:                {agg_counters["dropped_ngrams"]} ')

    # Optionally remove the entire input directory.
    if delete_input:
        clear_directory(input_dir)

    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')
    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')