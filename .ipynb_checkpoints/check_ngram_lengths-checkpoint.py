import argparse
from tqdm import tqdm

def check_ngram_lengths(corpus, min_ngram_length=1, max_ngram_length=5):
    """
    Checks if any n-gram in the corpus is shorter than min_ngram_length or longer than max_ngram_length.
    Prints invalid n-grams that fall outside the valid range.
    """
    invalid_ngram_count = 0
    
    print("Checking ngram lengths.\n")
    for idx, ngram in tqdm(enumerate(corpus), total=len(corpus), desc="Lines"):
        ngram_tokens = ngram.split()  # Assuming n-gram is space-separated tokens
        
        # Check for invalid n-grams (too short or too long)
        if len(ngram_tokens) < min_ngram_length:
            invalid_ngram_count += 1
            print(f"Invalid n-gram (too short) found at index {idx}: {ngram} (Length: {len(ngram_tokens)})")
        elif len(ngram_tokens) > max_ngram_length:
            invalid_ngram_count += 1
            print(f"Invalid n-gram (too long) found at index {idx}: {ngram} (Length: {len(ngram_tokens)})")
    
    if invalid_ngram_count == 0:
        print(f"All n-grams are valid (length between {min_ngram_length} and {max_ngram_length}).")
    else:
        print(f"Total {invalid_ngram_count} invalid n-grams found (too short or too long).")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Check the lengths of n-grams in a corpus file.")
    parser.add_argument("--corpus_path", help="Path to the corpus file.")
    parser.add_argument("--min_ngram_length", type=int, default=1, help="Minimum allowed n-gram length.")
    parser.add_argument("--max_ngram_length", type=int, default=5, help="Maximum allowed n-gram length.")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Load the corpus from the file
    try:
        with open(args.corpus_path, "r") as f:
            corpus = [line.strip() for line in f]
        
        # Check n-gram lengths
        check_ngram_lengths(corpus, min_ngram_length=args.min_ngram_length, max_ngram_length=args.max_ngram_length)
    
    except FileNotFoundError:
        print(f"Error: File '{args.corpus_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
