import argparse
import os
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import orjson
from tqdm.notebook import tqdm

from ngram_tools.helpers.file_handler import FileHandler


def construct_output_path(input_file, output_dir, compress):

    input_path = Path(input_file)
    base_name = (
        input_path.stem if input_path.suffix == '.lz4' else input_path.name
    )
    base_name = base_name.replace('.txt', '.jsonl')
    
    return str(Path(output_dir) / (base_name + ('.lz4' if compress else '')))


def set_info(proj_dir, ngram_size, file_range, compress):
    """
    Set input/output directory paths and gather file information.

    Args:
        proj_dir (str): The project directory path where ngram files are
        stored.
        ngram_size (int): The size of ngrams (1 to 5).
        file_range (tuple, optional): The range of file indices to process.
        compress (bool): Whether to compress the output files.

    Returns:
        tuple: A tuple containing directory paths, file counts, and paths for
        input and output files.
    """
    input_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/1download')
    output_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/2convert')

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

    return (input_dir, output_dir, file_range, num_files_available,
            input_paths_use, file_range, num_files_to_use, first_file,
            last_file, output_paths)


def print_info(input_dir, output_dir, file_range, num_files_available,
               num_files_to_use, first_file, last_file, ngram_size, ngram_type,
               workers, compress, overwrite, start_time, delete_input):
    """
    Prints information about the conversion process, such as input and output
    directories, file range, ngram details, and processing parameters.

    Args:
        input_dir (str): The input directory path.
        output_dir (str): The output directory path.
        file_range (tuple): The range of files to process.
        num_files_available (int): Total number of available files in the input
            directory.
        num_files_to_use (int): Number of files to process based on the file
            range.
        first_file (str): Path to the first file to process.
        last_file (str): Path to the last file to process.
        ngram_size (int): The size of the ngrams.
        ngram_type (str): The type of the ngrams (e.g., tagged or untagged).
        workers (int): The number of worker processes to use.
        compress (bool): Whether to compress the output files.
        overwrite (bool): Whether to overwrite existing files.
        start_time (datetime): The start time of the conversion process.
        delete_input (bool): Whether to delete input files after processing.
    """
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mConversion Info\033[0m')
    print(f'Input directory:           {input_dir}')
    print(f'Output directory:          {output_dir}')
    print(f'File index range:          {file_range[0]} to {file_range[1]}')
    print(f'Files available:           {num_files_available}')
    print(f'Files to use:              {num_files_to_use}')
    print(f'First file to get:         {first_file}')
    print(f'Last file to get:          {last_file}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Ngram type:                {ngram_type}')
    print(f'Number of workers:         {workers}')
    print(f'Compress output files:     {compress}')
    print(f'Overwrite existing files:  {overwrite}')
    print(f'Delete input directory:    {delete_input}\n')


def process_a_line(line, ngram_type):
    """
    Processes a single line of text to extract ngram tokens, frequencies, and
    document counts.

    Args:
        line (str): A single line from the ngram file.
        ngram_type (str): The ngram type ('tagged' or 'untagged').

    Returns:
        bytes: A serialized JSON object containing the processed ngram data.
        None: If there was an error processing the line.
    """
    try:
        tokens_part, *year_data_parts = line.strip().split('\t')

        tokens = tokens_part.split()
        pos = {}
        ngram = {}

        for i in range(len(tokens)):
            token_key = f"token{i+1}"
            token = tokens[i]

            if ngram_type == "tagged":
                base_token, pos_tag = token.rsplit("_", 1)
                ngram[token_key] = base_token
                pos[token_key] = pos_tag
            else:
                ngram[token_key] = token

        freq_tot = 0
        doc_tot = 0
        freq = {}
        doc = {}

        for year_data in year_data_parts:
            year, freq_val, doc_val = year_data.split(',')
            freq_val = int(freq_val)
            doc_val = int(doc_val)

            freq[year] = freq_val
            doc[year] = doc_val
            freq_tot += freq_val
            doc_tot += doc_val

        json_obj = {
            "ngram": ngram,
            "freq_tot": freq_tot,
            "doc_tot": doc_tot,
            "freq": freq,
            "doc": doc,
        }

        if ngram_type == "tagged":
            json_obj["pos"] = pos

        return json_obj

    except Exception as e:
        print(f"Error processing line: {line.strip()[:50]}... Error: {e}")
        return None


def process_a_file(args):
    """
    Processes a single file by reading, converting each line, and writing the
    output.

    Args:
        args (tuple): A tuple containing input path, output path, overwrite
            flag, compress flag, and ngram type.

    Returns:
        None: If an error occurs during file processing, the function returns
            early.
    """
    input_path, output_path, overwrite, compress, ngram_type = args

    input_ext = Path(input_path).suffix
    input_size = os.path.getsize(input_path)
    if (
        (input_ext == '.txt' and input_size == 0) or
        (input_ext == '.lz4' and input_size == 11)
    ):
        return

    in_handler = FileHandler(input_path, is_output=False)
    out_handler = FileHandler(output_path, is_output=True, compress=compress)

    if not overwrite and os.path.exists(output_path):
        return

    try:
        with in_handler.open() as infile, out_handler.open() as outfile:
            for line_no, line in enumerate(infile, start=1):
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                    
                json_dict = process_a_line(line, ngram_type)
                
                jsonl_line = out_handler.serialize(json_dict)
                outfile.write(jsonl_line)

    except Exception as e:
        print(f"Error converting {input_path}: {e}")


def process_a_directory(output_dir, input_paths, output_paths, workers,
                        overwrite, compress, ngram_type):
    """
    Processes all files in a directory using multiprocessing.

    Args:
        output_dir (str): The directory to store the converted files.
        input_paths (list): A list of input file paths.
        output_paths (list): A list of output file paths.
        workers (int): Number of workers for multiprocessing.
        overwrite (bool): Whether to overwrite existing files.
        compress (bool): Whether to compress the output files.
        ngram_type (str): The type of ngrams (tagged or untagged).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        (input_path, output_path, overwrite, compress, ngram_type)
        for input_path, output_path in zip(input_paths, output_paths)
    ]

    with tqdm(total=len(input_paths), desc="Converting", unit='files') as pbar:
        with Pool(processes=workers) as pool:
            for _ in pool.imap_unordered(process_a_file, args):
                pbar.update()


def clear_a_directory(directory_path):
    """
    Clears all files and subdirectories in a specified directory.

    Args:
        directory_path (str): The path of the directory to be cleared.

    Returns:
        None
    """
    for entry in os.scandir(directory_path):
        if entry.is_file():
            os.remove(entry.path)
        elif entry.is_dir():
            os.rmdir(entry.path)


def convert_to_jsonl_files(ngram_size, ngram_type, proj_dir, file_range=None,
                           overwrite=False, compress=False,
                           workers=os.cpu_count(), delete_input=False):
    """
    Converts ngram files to JSONL format.

    Args:
        ngram_size (int): The ngram size (1, 2, 3, 4, 5).
        ngram_type (str): The ngram type ('tagged' or 'untagged').
        proj_dir (str): The project directory where input and output files
            reside.
        file_range (tuple, optional): Range of files to process.
        overwrite (bool): Whether to overwrite existing files.
        compress (bool): Whether to compress the output files.
        workers (int): The number of worker processes to use.
        delete_input (bool): Whether to delete input files after conversion.

    Returns:
        None
    """
    start_time = datetime.now()

    (
        input_dir, output_dir, file_range, num_files_available,
        input_paths, file_range, num_files_to_use, first_file,
        last_file, output_paths
    ) = set_info(
        proj_dir, ngram_size, file_range, compress
    )

    print_info(input_dir, output_dir, file_range, num_files_available,
               num_files_to_use, first_file, last_file, ngram_size, ngram_type,
               workers, compress, overwrite, start_time, delete_input)

    process_a_directory(output_dir, input_paths, output_paths, workers,
                        overwrite, compress, ngram_type)

    if delete_input:
        clear_a_directory(input_dir)

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')