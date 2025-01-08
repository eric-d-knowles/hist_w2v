import argparse
import pandas as pd
import os
from tqdm import tqdm

def build_weighted_corpus(input_file, output_file):
    """
    Reads a Parquet file containing n-grams and their frequencies,
    and builds a weighted corpus for Word2Vec training, saving as a plain text file.
    N-grams are repeated according to their frequency, without storing the frequency in the output.
    """
    try:
        print("Loading data from Parquet file.\n")
        # Load the DataFrame from the input Parquet file
        df = pd.read_parquet(input_file)
        
        if 'ngram' not in df.columns or 'frequency' not in df.columns:
            raise ValueError("Input file must contain 'ngram' and 'frequency' columns.")
        
        # Initialize the list to hold repeated n-grams
        repeated_corpus = []
        
        print("Building corpus with repeated n-grams.\n")
        # Iterate through rows of the DataFrame with tqdm progress bar
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Lines"):
            ngram = row.ngram
            frequency = row.frequency
            
            # Split n-gram string into words and repeat it by frequency
            ngram_tokens = ngram.split()
            repeated_corpus.extend([ngram_tokens] * frequency)  # Repeat n-gram by frequency

        print(f"\nTotal n-grams in corpus: {len(repeated_corpus):,}\n")

        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save the repeated corpus (n-grams repeated by frequency) as a plain text file
        print(f"Saving repeated weighted corpus to {output_file}.")
        with open(output_file, 'w') as f:
            for ngram in repeated_corpus:
                # Join n-gram words with spaces and write each n-gram on a new line
                f.write(f"{' '.join(ngram)}\n")  # No frequency, just the n-gram

        print("\nWeighted corpus successfully saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare a weighted corpus for Word2Vec training from a Parquet file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input Parquet file (must contain 'ngram' and 'frequency' columns).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file where the weighted corpus will be saved (plain text format).")
    args = parser.parse_args()
    
    build_weighted_corpus(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
