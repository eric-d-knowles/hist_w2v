import os
import argparse
import json
from tqdm import tqdm
import multiprocessing

def process_file(input_file, output_file, include_frequencies):
    """Processes the JSONL file and saves it as plain text, with or without frequency values."""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # Load the JSON data from the current line
            data = json.loads(line.strip())
            ngram = data['ngram']
            
            if include_frequencies:
                total_frequency = data['total_frequency']
                # Writing the ngram along with its frequency
                outfile.write(f"{ngram} {total_frequency}\n")
            else:
                # Just writing the ngram (stripping the frequency)
                outfile.write(f"{ngram}\n")

def process_directory(input_dir, output_dir, include_frequencies, processes):
    """Process all JSONL files in the input directory and save them as plain-text files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    file_paths = [os.path.join(input_dir, f) for f in files]

    # Initialize the progress bar for the files
    progress = tqdm(total=len(file_paths), desc="Processing files", unit="file")

    # Use multiprocessing to process multiple files in parallel
    with multiprocessing.Pool(processes) as pool:
        # Start processing files asynchronously
        for file_path in file_paths:
            output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.jsonl', '.txt'))
            pool.apply_async(process_file, args=(file_path, output_file, include_frequencies), callback=lambda _: progress.update(1))

        # Close the progress bar once done
        pool.close()
        pool.join()
        progress.close()

def main():
    """Main function to handle arguments and call the processing functions."""
    parser = argparse.ArgumentParser(description="Convert JSONL files to plain-text format.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing the input JSONL files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output plain-text files.")
    parser.add_argument('--processes', type=int, required=True, help="Number of processes to use.")
    parser.add_argument('--include_frequencies', action='store_true', help="Include frequency values in the output text files.")
    args = parser.parse_args()

    # Start processing the files
    process_directory(args.input_dir, args.output_dir, args.include_frequencies, args.processes)

if __name__ == "__main__":
    main()
