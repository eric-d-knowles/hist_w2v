import orjson
import argparse
from tqdm import tqdm
from collections import defaultdict

def consolidate_ngrams(input_file, output_file):
    """Consolidates n-grams from the input file by summing frequencies of consecutive duplicates."""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        previous_ngram = None
        previous_data = None
        progress = tqdm(infile, desc="Lines", unit="lines")

        for line in progress:
            # Parse the current line
            data = orjson.loads(line.strip())
            ngram = data['ngram']
            total_frequency = data['total_frequency']
            year_data = data['frequency']

            if ngram == previous_ngram:
                # Merge total frequencies
                previous_data['total_frequency'] += total_frequency

                # Merge yearly frequencies using defaultdict
                for year, freq in year_data.items():
                    previous_data['frequency'][year] += freq
            else:
                # Write the consolidated data for the previous n-gram
                if previous_data:
                    outfile.write(orjson.dumps(previous_data).decode('utf-8') + '\n')

                # Update to the current n-gram
                previous_ngram = ngram
                previous_data = {
                    'ngram': ngram,
                    'total_frequency': total_frequency,
                    'frequency': defaultdict(int, year_data)  # Use defaultdict for efficient merging
                }

        # Write the last n-gram to the output file
        if previous_data:
            outfile.write(orjson.dumps(previous_data).decode('utf-8') + '\n')


def main():
    """Main function to parse arguments and run the consolidation process."""
    parser = argparse.ArgumentParser(description="Consolidate consecutive ngrams by summing frequencies.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    print("Consolidating ngrams.\n")
    consolidate_ngrams(args.input_file, args.output_file)
    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
