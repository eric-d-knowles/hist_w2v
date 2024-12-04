import orjson
import os
from multiprocessing import Pool
import argparse
from tqdm import tqdm

def process_ngram_line(line):
    """
    Processes a single line of n-gram data to extract yearly data.

    Args:
        line (str): A line of JSONL input representing an n-gram and its frequency data.

    Returns:
        dict: A dictionary where keys are years, and values are lists of n-gram entries.
    """
    data = orjson.loads(line)  # Deserialize the JSON line
    ngram = data["ngram"]
    frequency = data["frequency"]

    yearly_data = {}
    for year, count in frequency.items():
        if year not in yearly_data:
            yearly_data[year] = []
        yearly_data[year].append({"ngram": ngram, "frequency": count})
    return yearly_data

def write_yearly_data_for_chunk(yearly_data, output_dir):
    """
    Writes the n-gram data for a single chunk to the appropriate yearly files.

    Args:
        yearly_data (dict): Dictionary with years as keys and n-gram data as values.
        output_dir (str): Directory to write the output files.
    """
    for year, entries in yearly_data.items():
        year_file = os.path.join(output_dir, f"{year}.jsonl")
        with open(year_file, "ab") as f:  # Append mode to avoid overwrites
            for entry in entries:
                f.write(orjson.dumps(entry) + b"\n")  # Serialize with orjson

def process_chunk(chunk, output_dir):
    """
    Processes a chunk of lines to extract yearly n-gram data and writes it to disk.

    Args:
        chunk (list): A chunk of lines from the input file.
        output_dir (str): Directory to write the output files.

    Returns:
        None
    """
    yearly_data_combined = {}
    for line in chunk:
        yearly_data = process_ngram_line(line)
        for year, entries in yearly_data.items():
            if year not in yearly_data_combined:
                yearly_data_combined[year] = []
            yearly_data_combined[year].extend(entries)

    # Immediately write the processed data to disk
    write_yearly_data_for_chunk(yearly_data_combined, output_dir)

def process_file_line_by_line(input_file, output_dir, num_processes, chunk_size=1000):
    """
    Processes an input file line by line and writes yearly n-gram data immediately.
    This method is memory-efficient since it doesn't load the entire file into memory.

    Args:
        input_file (str): Path to the input file.
        output_dir (str): Path to the directory for output files.
        num_processes (int): Number of processes to use for parallelization.
        chunk_size (int): The number of lines to send to each worker process in a single batch.
    """
    with open(input_file, "r") as f:
        # Initialize progress bar with no total count
        progress_bar = tqdm(desc="Reading lines", unit="line", dynamic_ncols=True)

        # Initialize chunk buffer
        chunk = []
        with Pool(num_processes) as pool:
            for line in f:
                chunk.append(line)
                progress_bar.update(1)  # Increment progress bar for each line read

                if len(chunk) >= chunk_size:
                    # Process the current chunk and immediately write it to disk
                    pool.apply_async(process_chunk, (chunk, output_dir))
                    chunk = []  # Reset chunk buffer to free memory

            # Ensure any remaining lines in the last chunk are processed
            if chunk:
                pool.apply_async(process_chunk, (chunk, output_dir))

            # Wait for all the processes to complete
            pool.close()
            pool.join()

        # Close the progress bar
        progress_bar.close()

def main():
    parser = argparse.ArgumentParser(description="Process n-gram JSONL files and extract yearly data.")
    parser.add_argument("--input_file", help="Path to the input JSONL file containing n-gram data.")
    parser.add_argument("--output_dir", help="Path to the directory where output files will be saved.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of file chunks.")
    parser.add_argument("--processes", type=int, default=4, help="Number of processes to use for parallelization. Default is the number of CPUs.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)  # Ensure output directory exists

    print(f"Making yearly files.\n")
    process_file_line_by_line(args.input_file, args.output_dir, args.processes, args.chunk_size)
    print(f"\nProcessing complete.")

if __name__ == "__main__":
    main()
