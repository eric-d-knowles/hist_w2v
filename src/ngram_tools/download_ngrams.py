import argparse
import gzip
import logging
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool

import requests
from tqdm import tqdm

from ngram_tools.helpers.file_handler import FileHandler


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Download ngrams.")

    parser.add_argument(
        '--ngram_size', type=int, choices=[1, 2, 3, 4, 5],
        required=True, help='Ngrams size to get (required).'
    )
    parser.add_argument(
        '--ngram_type', type=str, choices=['tagged', 'untagged'],
        required=True, help='Ngram type to retain (required).'
    )
    parser.add_argument(
        '--proj_dir', type=str, required=True,
        help="Local project directory (required)."
    )
    parser.add_argument(
        '--file_range', type=int, nargs=2,
        help='File index range to get (default=all).'
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='Overwrite existing files? (default=False).'
    )
    parser.add_argument(
        '--compress', action='store_true', default=False,
        help='Compress saved files? (default=False).'
    )
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count(),
        help='Number of processors to use (default=all]).'
    )

    return parser.parse_args()


def set_location_info(ngram_size, proj_dir):
    """
    Sets the repository URL and file patterns based on ngram size and project
    dir.

    Args:
        ngram_size (int): The ngram size (1, 2, 3, 4, or 5).
        proj_dir (str): The local project directory.

    Returns:
        tuple: ngram_repo_url, file_pattern, output_dir
    """
    ngram_repo_url = f'https://storage.googleapis.com/books/ngrams/' \
                     f'books/20200217/eng/eng-{ngram_size}-ngrams_exports.html'
    file_pattern = rf'{ngram_size}-\d{{5}}-of-\d{{5}}\.gz'
    output_dir = os.path.join(f'{proj_dir}',
                              f'{ngram_size}gram_files/1download')
    return ngram_repo_url, file_pattern, output_dir


def fetch_file_urls(ngram_repo_url, file_pattern):
    """
    Fetches file URLs from the repository using the provided URL and regex.

    Args:
        ngram_repo_url (str): The URL of the ngram repository.
        file_pattern (str): The regex pattern to match file names.

    Returns:
        list: A list of URLs matching the file pattern.

    Raises:
        RuntimeError: If fetching or regex fails.
    """
    try:
        logging.info(f"Fetching file URLs from {ngram_repo_url}...")
        response = requests.get(ngram_repo_url, timeout=30)
        response.raise_for_status()
        file_urls = [
            requests.compat.urljoin(ngram_repo_url, filename)
            for filename in re.findall(file_pattern, response.text)
        ]
        logging.info(f"Found {len(file_urls)} matching files.")
        return file_urls
    except requests.RequestException as req_err:
        logging.error(f"Request failed: {req_err}")
        raise RuntimeError("Failed to fetch file URLs.") from req_err
    except re.error as regex_err:
        logging.error(f"Regex error: {regex_err}")
        raise RuntimeError("Invalid file pattern.") from regex_err


def define_regex(ngram_type):
    """
    Defines the regular expression for matching ngram tokens.

    Args:
        ngram_type (str): The ngram type ('tagged' or 'untagged').

    Returns:
        re.Pattern: The regex pattern for matching the specified ngram type.
    """
    valid_tags = r'NOUN|PROPN|VERB|ADJ|ADV|PRON|DET|ADP|NUM|CONJ|X|\.'
    if ngram_type == 'tagged':
        return re.compile(rf'^(\S+_(?:{valid_tags})\s?)+$')
    elif ngram_type == 'untagged':
        return re.compile(rf'^(?!.*_(?:{valid_tags})\s?)(\S+\s?)*$')


def print_info(ngram_repo_url, output_dir, file_range, file_urls_available,
               file_urls_to_use, ngram_size, ngram_type, workers, compress,
               overwrite, start_time):
    """
    Prints the information about the download process.

    Args:
        ngram_repo_url (str): The URL of the ngram repository.
        output_dir (str): The output directory where files will be saved.
        file_range (tuple): The range of file indexes to download.
        file_urls_available (list): List of all available file URLs.
        file_urls_to_use (list): List of file URLs to be downloaded.
        ngram_size (int): The ngram size being downloaded.
        ngram_type (str): The ngram type ('tagged' or 'untagged').
        workers (int): The number of worker processes.
        compress (bool): Whether to compress the output files.
        overwrite (bool): Whether to overwrite existing files.
        start_time (datetime): The start time of the process.
    """
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mDownload Info\033[0m')
    print(f'Ngram repository:          {ngram_repo_url}')
    print(f'Output directory:          {output_dir}')
    print(f'File index range:          {file_range[0]} to {file_range[1]}')
    print(f'File URLs available:       {len(file_urls_available)}')
    print(f'File URLs to use:          {len(file_urls_to_use)}')
    print(f'First file to get:         {file_urls_to_use[0]}')
    print(f'Last file to get:          {file_urls_to_use[-1]}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Ngram type:                {ngram_type}')
    print(f'Number of workers:         {workers}')
    print(f'Compress saved files:      {compress}')
    print(f'Overwrite existing files:  {overwrite}\n')


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


def process_a_file(args):
    """
    Downloads and processes a single file from the URL.

    Args:
        args (tuple): The arguments for processing a file, including the
                      URL, output directory, regex, overwrite flag, and
                      compression flag.
    """
    url, output_dir, find_regex, overwrite, compress = args

    file_name = os.path.splitext(os.path.basename(url))[0] + '.txt'
    output_file_path = os.path.join(output_dir, file_name)
    if compress:
        output_file_path += '.lz4'

    out_handler = FileHandler(output_file_path, is_output=True,
                              compress=compress)

    if not overwrite and os.path.exists(output_file_path):
        return

    try:
        response = requests.get(url, stream=True, timeout=(10, 60))
        response.raise_for_status()
        with out_handler.open() as outfile, \
             gzip.GzipFile(fileobj=response.raw, mode='rb') as infile:
            for line in infile:
                line = line.decode('utf-8')
                if find_regex.match(line.split('\t', 1)[0]):
                    outfile.write(line.encode('utf-8') if compress else line)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"Error processing {url}: {e}")


def process_files_in_parallel(file_urls_to_use, output_dir, workers,
                              find_regex, overwrite, compress):
    """
    Processes multiple files in parallel.

    Args:
        file_urls_to_use (list): List of file URLs to download and process.
        output_dir (str): The output directory where files will be saved.
        workers (int): The number of worker processes to use.
        find_regex (re.Pattern): The regex pattern to match ngram tokens.
        overwrite (bool): Whether to overwrite existing files.
        compress (bool): Whether to compress the output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    args = [
        (url, output_dir, find_regex, overwrite, compress)
        for url in file_urls_to_use
    ]
    with create_progress_bar(
        len(file_urls_to_use), "Downloading", 'files'
    ) as pbar:
        with Pool(processes=workers) as pool:
            for _ in pool.imap_unordered(process_a_file, args):
                pbar.update()


def download_ngram_files(ngram_size, ngram_type, proj_dir, file_range=None,
                         overwrite=False, compress=False,
                         workers=os.cpu_count()):
    """
    The main function to download and process ngram files.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """

    ngram_repo_url, file_pattern, output_dir = set_location_info(ngram_size,
                                                                 proj_dir)
    start_time = datetime.now()
    file_urls_available = fetch_file_urls(ngram_repo_url, file_pattern)

    if not file_range:
        file_range = (0, len(file_urls_available) - 1)
    file_urls_to_use = file_urls_available[file_range[0]:file_range[1] + 1]

    print_info(ngram_repo_url, output_dir, file_range, file_urls_available,
               file_urls_to_use, ngram_size, ngram_type, workers, compress,
               overwrite, start_time)

    find_regex = define_regex(ngram_type)
    process_files_in_parallel(file_urls_to_use, output_dir, workers,
                              find_regex, overwrite, compress)

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')
    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    args = parse_args()
    download_ngram_files(
        ngram_size=args.ngram_size,
        ngram_type=args.ngram_type,
        proj_dir=args.proj_dir,
        file_range=args.file_range,
        overwrite=args.overwrite,
        compress=args.compress,
        workers=args.workers
    )