import os
import re
import gzip
import orjson
import lz4.frame
import argparse
import requests
import logging
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime


class FileHandler:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.compress = path.endswith('.lz4')
        self.encoding = None if self.compress else 'utf-8'
        self.open_fn = lz4.frame.open if self.compress else open

    def open(self):
        return self.open_fn(self.path, self.mode, encoding=self.encoding)

    def serialize(self, entry):
        serialized = orjson.dumps(entry)
        return serialized + b'\n' if self.compress else serialized.decode(
            'utf-8') + '\n'

    def deserialize(self, line):
        return orjson.loads(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Download ngrams.")

    parser.add_argument('--ngram_size',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        required=True,
                        help='Ngrams size to get (required).')
    parser.add_argument('--ngram_type',
                        type=str,
                        choices=['tagged', 'untagged'],
                        required=True,
                        help='Ngram type to retain (required).')
    parser.add_argument('--proj_dir',
                        type=str,
                        required=True,
                        help="Local project directory (required).")
    parser.add_argument('--file_range',
                        type=int,
                        nargs=2,
                        help='File index range to get (default=all).')
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
                        default=os.cpu_count(),
                        help='Number of processors to use (default=all]).')
    return parser.parse_args()


def set_location_info(ngram_size, proj_dir):
    ngram_repo_url = (
        'https://storage.googleapis.com/books/ngrams/books/20200217/eng/'
        f'eng-{ngram_size}-ngrams_exports.html'
    )
    file_pattern = rf'{ngram_size}-\d{{5}}-of-\d{{5}}\.gz'
    output_dir = os.path.join(
        f'{proj_dir}', f'{ngram_size}gram_files/1download'
    )

    return ngram_repo_url, file_pattern, output_dir


def fetch_file_urls(ngram_repo_url, file_pattern):
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


def print_info(ngram_repo_url, output_dir, file_range, file_urls_available,
               file_urls_to_use, ngram_size, ngram_type, workers, compress,
               overwrite, start_time):
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
    print(f'Overwrite existing files:  {overwrite}')
    print('')


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} \
    [{elapsed}<{remaining}, {rate_fmt}]"
)


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


def define_regex(ngram_type):
    valid_tags = r'NOUN|PROPN|VERB|ADJ|ADV|PRON|DET|ADP|NUM|CONJ|X|\.'
    if ngram_type == 'tagged':
        return re.compile(rf'^(\S+_(?:{valid_tags})\s?)+$')
    elif ngram_type == 'untagged':
        return re.compile(rf'^(?!.*_(?:{valid_tags})\s?)(\S+\s?)*$')


def process_a_file(args):
    url, output_dir, find_regex, overwrite, compress = args

    file_name = os.path.splitext(os.path.basename(url))[0] + '.txt'
    output_file_path = os.path.join(output_dir, file_name)
    if compress:
        output_file_path += '.lz4'

    output_handler = FileHandler(output_file_path, 'wb' if compress else 'w')

    if not overwrite and os.path.exists(output_file_path):
        return

    try:
        response = requests.get(url, stream=True, timeout=(10, 60))
        response.raise_for_status()
        with output_handler.open() as outfile, \
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


def main():
    args = parse_args()

    ngram_size = args.ngram_size
    ngram_type = args.ngram_type
    file_range = args.file_range
    workers = args.workers
    proj_dir = args.proj_dir
    overwrite = args.overwrite
    compress = args.compress

    ngram_repo_url, file_pattern, output_dir = set_location_info(
        ngram_size, proj_dir
    )

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
    main()