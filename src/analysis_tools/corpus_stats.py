import os
import json
import lz4.frame
import glob
import multiprocessing as mp
import pandas as pd
from tqdm.notebook import tqdm

def process_ngram_file(file_path):
    """
    Processes a single LZ4-compressed JSONL file containing n-gram data.

    Parameters:
        file_path (str): Path to the LZ4-compressed JSONL file.

    Returns:
        tuple: (year, stats_dict) where stats_dict contains:
            - unique_ngrams: Number of unique n-grams.
            - total_freq: Sum of 'freq' across all n-grams.
            - total_docs: Sum of 'doc' across all n-grams.
    """
    year = os.path.basename(file_path).split('.')[0]
    unique_ngrams = 0
    total_freq = 0
    total_docs = 0

    try:
        with lz4.frame.open(file_path, 'rt') as f:
            for line in f:
                data = json.loads(line.strip())
                unique_ngrams += 1
                total_freq += data.get("freq", 0)
                total_docs += data.get("doc", 0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return year, None

    return year, {
        'unique_ngrams': unique_ngrams,
        'total_freq': total_freq,
        'total_docs': total_docs
    }

def get_corpus_stats(data_dir=None, out_dir=None, filename="corpus_stats.csv", overwrite=False, num_workers=None):
    """
    Processes all LZ4-compressed JSONL n-gram files in a directory and saves the statistics as a CSV file.

    Parameters:
        data_dir (str): Directory containing LZ4-compressed JSONL files.
        out_dir (str): Directory where the CSV file should be saved.
        filename (str, optional): Name of the output CSV file. Default is 'corpus_stats.csv'.
        overwrite (bool, optional): If True, overwrite an existing file. Otherwise, exit if file exists.
        num_workers (int, optional): Number of parallel workers.

    Returns:
        pandas.DataFrame: DataFrame containing the corpus statistics.
    """
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.jsonl.lz4")))

    if not file_paths:
        print("No files found in the specified directory.")
        return None

    output_path = os.path.join(out_dir, filename)
    
    if os.path.exists(output_path) and not overwrite:
        print(f"File {output_path} already exists. Use `overwrite=True` to replace it.")
        return None

    num_workers = num_workers or min(mp.cpu_count(), len(file_paths))
    results = {}

    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(file_paths), desc="Processing corpus files", unit="file") as pbar:
            for year, stats in pool.imap_unordered(process_ngram_file, file_paths):
                if stats:
                    results[year] = stats
                pbar.update(1)

    # Convert results to a DataFrame
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Year"
    df.sort_index(inplace=True)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save DataFrame to CSV
    df.to_csv(output_path)
    print(f"Statistics saved to {output_path}")

    return df  # Return the DataFrame for further use if needed