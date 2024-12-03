import os
import argparse
import shutil
import heapq
import tempfile
from tqdm import tqdm
from multiprocessing import Pool


def sort_a_file(input_output_pair):
    """Sort a single JSONL file by the 'ngram' key as text."""
    input_file, output_file = input_output_pair

    # Read lines as plain text
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Sort lines lexicographically
    lines.sort()  # Simple text-based sorting

    # Write the sorted lines back to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(lines)


def sort_individual_files(input_dir, temp_dir, processes):
    """Sort individual JSONL files and handle empty files."""
    print("Sorting individual files:\n")
    
    # Prepare the input-output pairs for sorting
    input_output_pairs = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_dir, filename)

            # Check if the file is empty
            if os.stat(file_path).st_size == 0:
                continue  # Skip empty files
            
            output_path = os.path.join(temp_dir, f"sorted_{filename}")
            input_output_pairs.append((file_path, output_path))
    
    if not input_output_pairs:
        print("No files to process. Exiting.")
        return []

    # Process the files in parallel using multiprocessing
    with Pool(processes=processes) as pool:
        list(tqdm(
            pool.imap_unordered(sort_a_file, input_output_pairs),
            total=len(input_output_pairs),
            desc="Files",
            unit="file")
        )

    # Collect the sorted files from the temp directory
    sorted_files = [
        os.path.join(temp_dir, filename)
        for filename in os.listdir(temp_dir) if filename.startswith("sorted")
    ]

    return sorted_files


def merge_sorted_files(sorted_files, temp_dir):
    """Merge sorted JSONL files into a single sorted output file."""
    print("\nMerge-sorting files:\n")

    heap = []
    open_files = []
    total_lines = 0  # Total lines to process

    # First, calculate the total number of lines in all sorted files for progress
    for file_path in sorted_files:
        with open(file_path, 'r') as file:
            total_lines += sum(1 for _ in file)

    # Open all sorted files and prepare the heap
    try:
        for file_index, file_path in enumerate(sorted_files):
            file = open(file_path, 'r')
            open_files.append(file)  # Keep track of open file handles
            first_line = file.readline()
            if first_line:
                # Directly push the entire line to the heap
                heapq.heappush(heap, (first_line, file_index))

        # Create a temporary output file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', dir=temp_dir) as output_file:
            output_file_path = output_file.name

            # Initialize progress bar
            with tqdm(total=total_lines, desc="Lines", unit="line") as pbar:
                while heap:
                    # Pop the smallest item from the heap (sorted by the line itself)
                    line, file_index = heapq.heappop(heap)
                    output_file.write(line)

                    # Update progress bar
                    pbar.update(1)

                    # Read the next line from the file that provided this entry
                    next_line = open_files[file_index].readline()
                    if next_line:
                        # Push the next line to the heap
                        heapq.heappush(heap, (next_line, file_index))

        return output_file_path

    finally:
        # Close all open files
        for file in open_files:
            file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort and concatenate multiple JSONL files by the 'ngram' key.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input JSONL files.")
    parser.add_argument("--temp_dir", type=str, required=True, help="Directory to store intermediate files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the final sorted output file.")
    parser.add_argument("--processes", type=int, default=1, help="Number of processes to use for sorting files.")
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    if os.path.dirname(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Sort individual files
    sorted_files = sort_individual_files(args.input_dir, args.temp_dir, args.processes)

    # Merge sorted files
    merged_file = merge_sorted_files(sorted_files, args.temp_dir)

    # Move the merged file to the final output location
    shutil.move(merged_file, args.output_file)

    print("\nProcessing complete.")
    
    # Cleanup temporary directory
    shutil.rmtree(args.temp_dir, ignore_errors=True)
