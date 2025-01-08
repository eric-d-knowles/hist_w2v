import os
import orjson
import argparse
from tqdm import tqdm
from collections import defaultdict

def consolidate_ngrams(input_file, output_file, strip_tags):
    """Consolidates n-grams from the input file by summing frequencies of consecutive duplicates."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        previous_ngram = None
        previous_data = None

        for line in tqdm(infile, desc="Lines", unit="lines"):
            # Parse the current line
            data = orjson.loads(line.strip())

            ngram = data['ngram']
            # If we're stripping POS tags, strip the current ngram
            if strip_tags:
                ngram = ' '.join([word.split('_')[0] for word in ngram.split()])  # Strip POS tags
                
            total_frequency = data['total_frequency']
            year_data = data['frequency']

            if ngram == previous_ngram:
                # Merge total frequencies
                previous_data['total_frequency'] += total_frequency

                # Merge yearly frequencies
                for year, freq in year_data.items():
                    previous_data['frequency'][year] += freq
            else:
                # Write the consolidated data for the previous n-gram
                if previous_data:
                    outfile.write(orjson.dumps(previous_data).decode('utf-8') + '\n')

                # The current ngram now becomes the previous ngram
                previous_ngram = ngram
                previous_data = {
                    'ngram': ngram,
                    'total_frequency': total_frequency,
                    'frequency': defaultdict(int, year_data)
                }

        # Write the last ngram to the output file
        if previous_data:
            outfile.write(orjson.dumps(previous_data).decode('utf-8') + '\n')


def main():
    """Main function to parse arguments and run the consolidation process."""
    parser = argparse.ArgumentParser(description="Consolidate consecutive ngrams by summing frequencies.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--strip_tags", action="store_true", help="If set, part-of-speech tags will be discarded during consolidation.")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite the output file if it exists.")
    args = parser.parse_args()

    # Check if output file exists and handle overwrite flag
    if os.path.exists(args.output_file):
        if args.overwrite:
            print(f"Overwriting existing file:\n{args.output_file}\n")
        else:
            print(f"Skipping existing file:\n{args.output_file}")
            exit()

    print("Consolidating ngrams.\n")
    consolidate_ngrams(args.input_file, args.output_file, args.strip_tags)
    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
