import os
import argparse
import pandas as pd
import orjson
from multiprocessing import Pool, Manager
from tqdm import tqdm  # Import tqdm for progress bar functionality

def convert_file(input_path, output_path, progress_queue):
    """
    Convert a single JSONL file to Parquet format and save it to the output path.
    """
    # Read the JSONL file into a DataFrame
    with open(input_path, "r") as f:
        data = [orjson.loads(line) for line in f]

    # Convert to DataFrame and save as Parquet
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)

    # Update progress
    progress_queue.put(1)

def convert_to_parquet(input_dir, output_dir, processes):
    """
    Convert all JSONL files in the input directory to Parquet format and save them in the output directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of files to process
    files_to_process = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.parquet")
            files_to_process.append((input_path, output_path))

    # Use multiprocessing Manager to share progress data
    with Manager() as manager:
        progress_queue = manager.Queue()

        # Use tqdm to create a progress bar for the number of files
        with Pool(processes=processes) as pool:
            # Apply starmap with progress updates
            for _ in tqdm(pool.starmap(convert_file, [(input_path, output_path, progress_queue) for input_path, output_path in files_to_process]),
                          total=len(files_to_process), desc="Files"):
                pass  # The progress bar will update as the queue receives updates

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert JSONL files to Parquet format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSONL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save Parquet files.")
    parser.add_argument("--processes", type=int, default=os.cpu_count(), help="Number of processes to use for parallel processing.")

    args = parser.parse_args()

    # Run the conversion
    print("Converting files.\n")
    convert_to_parquet(args.input_dir, args.output_dir, args.processes)
    print("\nProcessing complete.")
