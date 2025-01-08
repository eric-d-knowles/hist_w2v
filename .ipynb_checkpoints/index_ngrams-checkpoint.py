import os
import argparse
import orjson
from tqdm import tqdm

def sort_and_index_ngrams(input_file, output_file):
    """Sorts ngrams by total frequency in descending order and assigns sequential indices starting from 0."""

    with open(input_file, 'r') as infile:
        ngrams = [orjson.loads(line.strip()) for line in tqdm(infile, desc="Loading", unit="lines")]
    
    ngrams.sort(key=lambda x: x['total_frequency'], reverse=True)

    for index, ngram in enumerate(ngrams):
        ngram['index'] = index

    with open(output_file, 'w') as outfile:
        for ngram in tqdm(ngrams, desc="Writing", unit="ngrams"):
            outfile.write(orjson.dumps(ngram).decode() + '\n')

def main():
    parser = argparse.ArgumentParser(description="Sort ngrams by total frequency and reindex them starting from 0.")
    parser.add_argument('--input_file', type=str, help="Path to the input JSONL file containing ngrams.")
    parser.add_argument('--output_file', type=str, help="Path to the output JSONL file for sorted and indexed ngrams.")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite the output file if it exists.")
    args = parser.parse_args()

    # Check if output file exists and handle overwrite flag
    if os.path.exists(args.output_file):
        if args.overwrite:
            print(f"Overwriting existing file:\n{args.output_file}\n")
        else:
            print(f"Skipping existing file:\n{args.output_file}")
            exit()
            
    print("Indexing ngrams.\n")
    sort_and_index_ngrams(args.input_file, args.output_file)

    print("\nProcessing complete.")

if __name__ == '__main__':
    main()
