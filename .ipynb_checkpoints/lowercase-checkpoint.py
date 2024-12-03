from multiprocessing import Pool, cpu_count, Manager
import os
import glob
import argparse
import orjson
from tqdm import tqdm


def lowercase_ngram_tokens(ngram):
    """Lowercase the tokens in the ngram, leaving POS tags intact."""
    ngram_tokens = ngram.split()
    
    for i, token in enumerate(ngram_tokens):
        word, pos_tag = token.rsplit("_", 1)
        ngram_tokens[i] = word.lower() + "_" + pos_tag

    return ' '.join(ngram_tokens)


def process_file(file_path, output_dir):
    """Process a single file: lowercase the tokens in ngrams and save to the output directory."""
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    temp_output = []

    try:
        with open(file_path, 'rb') as infile:
            for line in infile:
                try:
                    data = orjson.loads(line)
                    ngram = data.get('ngram')
                    if ngram:
                        data['ngram'] = lowercase_ngram_tokens(ngram)
                        temp_output.append(orjson.dumps(data) + b'\n')
                except orjson.JSONDecodeError:
                    print(f"Skipping invalid JSON in file {file_path}")

        # Write to output only if there's valid data
        if temp_output:
            with open(output_file, 'wb') as outfile:
                outfile.writelines(temp_output)
        else:
            print(f"Skipping empty output for file {file_path}")
    except IOError:
        print(f"Error reading or writing file {file_path}")


def process_files_in_parallel(input_files, output_dir, processes=None):
    """Process the files in parallel to lowercase the ngram tokens."""
    if processes is None:
        processes = cpu_count()

    print("Lowercasing.\n")
    with Manager() as manager:
        progress_bar = manager.list()  # Shared list for progress updates
        tqdm_bar = tqdm(total=len(input_files), desc="Files", unit="file")

        def update_progress(*args):
            tqdm_bar.update()

        with Pool(processes) as pool:
            for file_path in input_files:
                pool.apply_async(process_file, args=(file_path, output_dir), callback=update_progress)
            pool.close()
            pool.join()
        
        tqdm_bar.close()


def lowercase_ngram_tokens_in_directory(input_dir, output_dir, processes=None):
    """Lowercase the ngram tokens in all files in a directory using multiprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out empty files during input file list preparation
    input_files = sorted(
        file for file in glob.glob(os.path.join(input_dir, "*.jsonl")) if os.path.getsize(file) > 0
    )
    
    if not input_files:
        print(f"No files found to process in {input_dir}. Please check the input directory.")
        return

    process_files_in_parallel(input_files, output_dir, processes)

    print("\nProcessing complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Lowercase ngram tokens in all files in a directory.")
    parser.add_argument('--input_dir', required=True, help='Path to the directory containing input JSONL files')
    parser.add_argument('--output_dir', required=True, help='Path to the directory where lowercased files will be saved')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: cpu_count())')
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")  # Use "fork" for faster process spawning

    args = parse_args()
    lowercase_ngram_tokens_in_directory(args.input_dir, args.output_dir, processes=args.processes)
