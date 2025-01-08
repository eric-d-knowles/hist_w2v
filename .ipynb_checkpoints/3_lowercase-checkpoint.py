import os
import orjson
import lz4.frame
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime


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
    parser = argparse.ArgumentParser(description="Lowercase ngrams.")

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
                        default=os.cpu_count(),
                        help='Number of processors to use (default=all]).')
    parser.add_argument('--delete_input',
                        action='store_true',
                        default=False,
                        help='Delete input files (default=False).')

    return parser.parse_args()


def construct_output_path(input_file, output_dir, compress):
    input_path = Path(input_file)
    base_name = (
        input_path.stem if input_path.suffix == '.lz4' else input_path.name
    )
    return str(Path(output_dir) / (base_name + ('.lz4' if compress else '')))


def set_info(proj_dir, ngram_size, file_range, compress):
    input_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/2convert')
    output_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/3lowercase')

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(
            f"Input directory {input_dir} does not exist or isn't a directory."
        )

    path_list = sorted([
        entry.path for entry in os.scandir(input_dir) if entry.is_file()
    ])
    num_files_available = len(path_list)

    if file_range:
        num_files_to_use = file_range[1] - file_range[0] + 1
    else:
        num_files_to_use = len(path_list)
        file_range = (0, len(path_list) - 1)

    first_file = path_list[file_range[0]]
    last_file = path_list[file_range[1]]

    input_paths = path_list[file_range[0]:file_range[1] + 1]
    output_paths = [
        construct_output_path(
            file, output_dir, compress
        ) for file in input_paths
    ]

    return (input_dir, output_dir, num_files_available, num_files_to_use,
            first_file, last_file, num_files_to_use, file_range, input_paths,
            output_paths)


def print_info(input_dir, output_dir, file_range, num_files_available,
               num_files_to_use, first_file, last_file, ngram_size, workers,
               compress, overwrite, start_time, delete_input):
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mLowercasing Info\033[0m')
    print(f'Input directory:           {input_dir}')
    print(f'Output directory:          {output_dir}')
    print(f'File index range:          {file_range[0]} to {file_range[1]}')
    print(f'Files available:           {num_files_available}')
    print(f'Files to use:              {num_files_to_use}')
    print(f'First file to get:         {first_file}')
    print(f'Last file to get:          {last_file}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Number of workers:         {workers}')
    print(f'Compress output files:     {compress}')
    print(f'Overwrite existing files:  {overwrite}')
    print(f'Delete input directory:    {delete_input}\n')


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


def process_a_line(line):
    if isinstance(line, bytes):
        line = line.decode('utf-8')

    try:
        data = orjson.loads(line)

    except orjson.JSONDecodeError as e:
        print(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
        raise

    data['ngram'] = {
        key: value.lower() for key, value in data['ngram'].items()
    }

    return data


def process_a_file(args):
    input_handler, output_handler, overwrite = args

    if not overwrite and os.path.exists(output_handler.path):
        return

    try:
        with input_handler.open() as infile, output_handler.open() as outfile:
            for line_no, line in enumerate(infile, start=1):
                if isinstance(line, bytes):
                    line = line.decode('utf-8')

                json_obj = process_a_line(line)

                if json_obj:
                    outfile.write(output_handler.serialize(json_obj))
                else:
                    print(
                        f"Skipped line {line_no} in file {input_handler.path}")

    except Exception as e:
        print(f"Error processing {input_handler.path}: {e}")


def process_a_directory(output_dir, input_paths, output_paths, workers,
                        overwrite, compress):
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

    print('\033[4mLowercasing Progress\033[0m')

    with create_progress_bar(len(handlers), "Lowercasing", 'files') as pbar:
        with Pool(processes=workers) as pool:
            for _ in pool.imap_unordered(process_a_file, args):
                pbar.update()


def clear_a_directory(directory_path):
    for entry in os.scandir(directory_path):
        if entry.is_file():
            os.remove(entry.path)
        elif entry.is_dir():
            os.rmdir(entry.path)


def main():
    args = parse_args()

    ngram_size = args.ngram_size
    proj_dir = args.proj_dir
    file_range = args.file_range
    overwrite = args.overwrite
    compress = args.compress
    workers = args.workers
    delete_input = args.delete_input

    start_time = datetime.now()

    (
        input_dir, output_dir, num_files_available, num_files_to_use,
        first_file, last_file, num_files_to_use, file_range, input_paths,
        output_paths
    ) = set_info(
         proj_dir, ngram_size, file_range, compress
    )

    print_info(input_dir, output_dir, file_range, num_files_available,
               num_files_to_use, first_file, last_file, ngram_size, workers,
               compress, overwrite, start_time, delete_input)

    process_a_directory(output_dir, input_paths, output_paths, workers,
                        overwrite, compress)

    if delete_input:
        clear_a_directory(input_dir)

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    main()