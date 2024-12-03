import argparse
import os
import nltk
from nltk.corpus import stopwords
import orjson
from tqdm import tqdm
from multiprocessing import Pool


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def filter_stopwords(input_file, output_file, removal_method, min_tokens):
    """Filters stopwords from a JSONL file and writes the filtered results to an output file."""
    
    excluded_stopwords = set()

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = orjson.loads(line.strip())
            
            ngram = data['ngram']
            frequency = data['total_frequency']
            year_data = data['frequency']

            ngram_tokens = ngram.split()
            filtered_tokens = []
            
            for token in ngram_tokens:
                base_word = token.rsplit('_')[0].lower()
                
                if base_word in stop_words:
                    excluded_stopwords.add(base_word)
                    
                    if removal_method == "ngram":
                        break
                    elif removal_method == "token":
                        continue
                else:
                    filtered_tokens.append(token)

            if len(filtered_tokens) >= min_tokens:
                output_line = {
                    "ngram": ' '.join(filtered_tokens), 
                    "total_frequency": frequency, 
                    "frequency": year_data
                }
                outfile.write(orjson.dumps(output_line).decode('utf-8') + '\n')

    return excluded_stopwords


def process_file(args):
    """Processes a single file with the given arguments."""
    input_file, output_file, removal_method, min_tokens = args
    return filter_stopwords(input_file, output_file, removal_method, min_tokens)


def main():
    parser = argparse.ArgumentParser(description="Filter stopwords from JSONL files in a directory.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing input JSONL files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the directory for saving filtered JSONL files.")
    parser.add_argument('--processes', type=int, default=4, help="Number of processes for multiprocessing (default: 4).")
    parser.add_argument('--removal_method', type=str, default="ngram", choices=["ngram", "token"], help="Filtering method: remove 'ngram' or remove 'token'.")
    parser.add_argument('--min_tokens', type=int, default=1, help="Minimum tokens in preserved ngrams.")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare input-output file pairs, excluding empty files
    files = [
        (os.path.join(args.input_dir, f), os.path.join(args.output_dir, f), args.removal_method, args.min_tokens)
        for f in os.listdir(args.input_dir)
        if f.endswith('.jsonl') and os.path.getsize(os.path.join(args.input_dir, f)) > 0
    ]

    # Display the number of valid files to process
    if not files:
        print("No valid JSONL files to process in the input directory.")
        return

    # Process files with multiprocessing and a progress bar
    excluded_stopwords = set()
    print("Processing files:\n")
    with Pool(processes=args.processes) as pool:
        with tqdm(total=len(files), desc="Files", unit="file") as pbar:
            for result in pool.imap(process_file, files):
                excluded_stopwords.update(result)
                pbar.update(1)

    # Print excluded stopwords summary
    print("\nUnique stopwords excluded across all files:\n")
    print(", ".join(sorted(excluded_stopwords)))


if __name__ == '__main__':
    main()
