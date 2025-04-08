import os
import lz4.frame
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.notebook import tqdm


def extract_ngrams_from_file(input_path, output_path, overwrite=False):
    """
    Extract 5-gram strings from a single .jsonl or .jsonl.lz4 file and write them to a .txt file.

    Args:
        input_path (str): Path to the input JSONL or JSONL.LZ4 file.
        output_path (str): Path to the output plain-text file.
    """
    if input_path.endswith(".lz4"):
        infile_opener = lambda path: lz4.frame.open(path, 'rt')
    else:
        infile_opener = lambda path: open(path, 'r')

    if not overwrite and os.path.exists(output_path):
        return

    with infile_opener(input_path) as infile, open(output_path, 'w') as outfile:
        for line in infile:
            try:
                record = json.loads(line)
                outfile.write(record["ngram"] + "\n")
            except (json.JSONDecodeError, KeyError):
                continue  # skip malformed lines


def _process_file(filename, *, input_dir, overwrite=False):
    year = filename.split(".")[0]
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(input_dir, f"{year}.txt")
    extract_ngrams_from_file(input_path, output_path, overwrite=overwrite)


def extract_all_ngrams(proj_dir, workers=cpu_count(), overwrite=False):
    """
    Iterate over all .jsonl and .jsonl.lz4 files in the project directory and extract n-grams in parallel.

    Args:
        proj_dir (str): Top-level project directory.
        workers (int, optional): Number of parallel worker processes. Defaults to number of CPUs.
    """
    input_dir = os.path.join(proj_dir, "5gram_files", "6corpus", "yearly_files", "data")
    filenames = [f for f in os.listdir(input_dir) if f.endswith(".jsonl") or f.endswith(".jsonl.lz4")]

    workers = workers or cpu_count()
    with Pool(processes=workers) as pool:
        process_func = partial(_process_file, input_dir=input_dir, overwrite=overwrite)
        list(tqdm(pool.imap(process_func, filenames), total=len(filenames), desc="Extracting ngrams"))
