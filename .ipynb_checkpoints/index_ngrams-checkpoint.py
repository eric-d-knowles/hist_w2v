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
    
    args = parser.parse_args()

    print("Indexing ngrams.\n")
    sort_and_index_ngrams(args.input_file, args.output_file)

    print("\nProcessing complete.")

if __name__ == '__main__':
    main()
