import argparse
import logging
import requests
import gzip
import os
import orjson
import re
from tqdm import tqdm 
from functools import partial
from multiprocessing import Pool, cpu_count
from nltk import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


google_to_wordnet_pos = {
    "NOUN": "n",
    "PROPN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r"
}


valid_pos_tags = re.compile(r'_[A-Z.]+$')
numeral = re.compile(r'\d')
nonalpha = re.compile(r'[^a-zA-Z_]|_(?!(NOUN|PROPN|VERB|ADJ|ADV|PRON|DET|ADP|NUM|CONJ|X|\.)$)')


logging.basicConfig(
    filename="download_and_filter_ngrams.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S"
)
    

def parse_args():
    """Define argument parser for command-line execution"""
    parser = argparse.ArgumentParser(description="Download and process ngrams.")
    parser.add_argument('--ngram_size', type=int, required=True, help='Size of the ngrams (e.g., 1, 2, 3, 4, 5)')
    parser.add_argument('--processes', type=int, default=cpu_count(), help='Number of processes to use')
    parser.add_argument('--file_range', type=int, nargs=2, default=[0, 23], help='Range of files to process')
    parser.add_argument('--vocab_file', type=str, default=None, help="Path to the frozenset vocabulary file (comma-delimited list of quoted strings).")
    parser.add_argument('--overwrite', action='store_true', default=None, help='Overwrite existing output JSONLs')
    parser.add_argument('--save_empty', action='store_true', default=False, help='Save empty files.')
    parser.add_argument('--test_file', type=str, default=None, help="Path to an test file.")
    return parser.parse_args()


def load_vocab(vocab_file):
    """Load the frozenset from a comma-delimited list of quoted strings."""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        tokens = content.split(',')
        vocab_set = frozenset(token.strip().strip('"') for token in tokens)
        
    print("Loaded vocabulary file.\n")
    return vocab_set


def check_ngram_tokens(token, vocab_set):
    """Checks if each token in the ngram is present in the frozenset vocabulary."""    
    base_word, pos_tag = token.split('_')
    base_word_lc = base_word.lower()
    base_word_lemma = lemmatizer.lemmatize(base_word_lc, pos=google_to_wordnet_pos.get(pos_tag, "n"))
    test_token = base_word_lemma + "_" + pos_tag
    
    if test_token not in vocab_set:
        return False
    return True
    

def fetch_file_urls(ngram_repo_url, file_pattern):
    """Fetch file URLs from the Google Ngram repository."""
    try:
        logging.info(f"Fetching file URLs from: {ngram_repo_url}")
        response = requests.get(ngram_repo_url, timeout=30)
        response.raise_for_status()
        html_content = response.text
        file_urls = [
            requests.compat.urljoin(ngram_repo_url, filename)
            for filename in re.findall(file_pattern, html_content)
        ]
        logging.info(f"Found {len(file_urls)} files matching pattern {file_pattern}.")
        return file_urls
    except Exception as e:
        logging.error(f"Error fetching file URLs: {e}")
        raise RuntimeError(f"Error fetching file URLs: {e}")


def filter_ngram_lines(file_url=None, file_pattern=None, vocab_set=None, test_file=None):
    """Stream and pre-filter ngrams from Google Ngrams."""
    rejection_counters = {
        'invalid_pos_tag': 0,
        'numeral': 0,
        'nonalpha': 0,
        'not_in_vocab': 0,
        'other': 0
    }
    try:
        if test_file:
            logging.info(f"Opening test file: {test_file}")
            file = open(test_file, 'r')
        else:
            logging.info(f"Downloading file: {file_url}")
            response = requests.get(file_url, stream=True, timeout=30)
            response.raise_for_status()
            file = gzip.GzipFile(fileobj=response.raw)
            
        with file as f:
            for line in f:
                
                if test_file:
                    decoded_line = line.strip()
                else:
                    decoded_line = line.strip().decode('utf-8')
                
                ngram, *year_data = decoded_line.split('\t')
                tokens = ngram.split()

                bad_token = False
                for token in tokens:
                    if not valid_pos_tags.search(token):
                        rejection_counters['invalid_pos_tag'] += 1
                        bad_token = True
                        break
                    if numeral.search(token):
                        rejection_counters['numeral'] += 1
                        bad_token = True
                        break
                    if nonalpha.search(token):
                        rejection_counters['nonalpha'] += 1
                        bad_token = True
                        break
                    if vocab_set:
                        if not check_ngram_tokens(token, vocab_set):
                            rejection_counters['not_in_vocab'] += 1
                            bad_token = True
                            continue
                            
                if bad_token:
                    continue

                ngram = " ".join(tokens)
                ngram_dict = {}
                
                for entry in year_data:
                    year, freq, _ = entry.split(',')
                    ngram_dict[year] = int(freq)
    
                yield ngram, ngram_dict

    except Exception as e:
        rejection_counters['other'] += 1
        logging.error(f"Error filtering file {file_url}: {e}")

    logging.info(f"Rejection counts for file {file_url}: {rejection_counters}")

    try:
        rejection_json = orjson.dumps(rejection_counters).decode('utf-8')
        logging.info(f"Rejection summary: {rejection_json}")
    except Exception as e:
        logging.error(f"Error serializing rejection counters: {e}")


def process_single_file(file_url=None, output_dir=None, file_pattern=None, vocab_set=None, overwrite=None, save_empty=None, test_file=None):
    """Inline logic for filtering and saving n-gram data."""
    if test_file:
        filename = 'output.jsonl'
        output_path = os.path.join(os.path.dirname(test_file), filename)
    else:
        filename = re.search(r"\d{5}-of-\d{5}", file_url).group(0)    
        output_path = os.path.join(output_dir, f"{filename}.jsonl")    

    if os.path.exists(output_path):
        if overwrite:
            logging.info(f"Overwriting {output_path}.")
        else:
            logging.info(f"Skipping {output_path}, as it already exists.")
            return

    # Flag to check if the file has any content
    has_content = False

    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for ngram, year_data in filter_ngram_lines(file_url=file_url, file_pattern=file_pattern, vocab_set=vocab_set, test_file=test_file):
                total_frequency = sum(year_data.values())
                
                output_line = {"ngram": ngram, "total_frequency": total_frequency, "frequency": year_data}
                output_file.write(orjson.dumps(output_line).decode('utf-8') + '\n')
                
                # If we write at least one line, set has_content to True
                has_content = True

        # Only log the success message if the file actually contains data
        if has_content or save_empty:
            logging.info(f"Finished processing file: {filename}")
        else:
            logging.info(f"File {filename} is empty. No data written.")
            os.remove(output_path)  # Delete the empty file

    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")



def process_all_files(ngram_repo_url, file_pattern, output_dir, vocab_set, processes, file_range, overwrite, save_empty):
    """Process all ngram files using multiprocessing, with inline filtering and saving logic."""
    file_urls = fetch_file_urls(ngram_repo_url, file_pattern)
    if file_range:
        file_urls = [
            url for url in file_urls
            if int(re.search(r"(\d{5})-of-(\d{5})", url).group(1)) in range(file_range[0], file_range[1] + 1)
        ]
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting multiprocessing for file processing.")

    print("Downloading and filtering.\n")

    with Pool(processes=processes) as pool:
        # Wrapping the map function to add progress bar
        for _ in tqdm(pool.imap_unordered(partial(process_single_file, output_dir=output_dir, file_pattern=file_pattern, vocab_set=vocab_set, overwrite=overwrite, save_empty=save_empty), file_urls), total=len(file_urls), desc="Files"):
            pass

    print("\nProcessing complete.")


if __name__ == "__main__":
    args = parse_args()
    
    if args.vocab_file and args.ngram_size == 1:
        raise ValueError(
            "ERROR: I can't filter unigrams using a vocabulary file. "
            "Please increase --ngram_size or take out --vocab_file."
        )

    ngram_repo_url = f"https://storage.googleapis.com/books/ngrams/books/20200217/eng/eng-{args.ngram_size}-ngrams_exports.html"
    file_pattern = rf"{args.ngram_size}-\d{{5}}-of-\d{{5}}\.gz"
    output_dir = f"/vast/edk202/NLP_corpora/Google_Books/20200217/eng/{args.ngram_size}gram_files/orig"

    if args.vocab_file:
        vocab_set = load_vocab(args.vocab_file)
    else:
        vocab_set = None

    if args.test_file:
        process_single_file(vocab_set=vocab_set, test_file=args.test_file, overwrite=args.overwrite, save_empty=args.save_empty)
    else:
        process_all_files(
            ngram_repo_url, 
            file_pattern,
            output_dir,
            vocab_set,
            args.processes,
            args.file_range,
            args.overwrite,
            args.save_empty
        )
