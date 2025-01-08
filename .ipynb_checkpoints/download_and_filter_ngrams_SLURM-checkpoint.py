import argparse
import logging
import requests
import gzip
import os
import orjson
import re
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
    parser.add_argument('--file_url', type=str, required=True, help='URL of the file to process')
    parser.add_argument('--vocab_file', type=str, default=None, help="Path to the frozenset vocabulary file.")
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output JSONL')
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


def filter_ngram_lines(file_url, vocab_set):
    """Stream and pre-filter ngrams from Google Ngrams."""
    rejection_counters = {
        'invalid_pos_tag': 0,
        'numeral': 0,
        'nonalpha': 0,
        'not_in_vocab': 0,
        'other': 0
    }
    try:
        logging.info(f"Downloading file: {file_url}")
        response = requests.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        file = gzip.GzipFile(fileobj=response.raw)
        
        with file as f:
            for line in f:
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
                            break

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


def process_single_file(file_url, output_dir, vocab_set, overwrite):
    """Process a single ngram file."""
    filename = re.search(r"\d{5}-of-\d{5}", file_url).group(0)
    output_path = os.path.join(output_dir, f"{filename}.jsonl")

    if os.path.exists(output_path) and not overwrite:
        logging.info(f"Skipping {output_path}, as it already exists.")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for ngram, year_data in filter_ngram_lines(file_url, vocab_set):
                total_frequency = sum(year_data.values())
                
                output_line = {"ngram": ngram, "total_frequency": total_frequency, "frequency": year_data}
                output_file.write(orjson.dumps(output_line).decode('utf-8') + '\n')

        logging.info(f"Finished processing file: {filename}")
        
    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")


if __name__ == "__main__":
    args = parse_args()

    if args.vocab_file:
        vocab_set = load_vocab(args.vocab_file)
    else:
        vocab_set = None

    output_dir = f"/vast/edk202/NLP_corpora/Google_Books/20200217/eng/{args.ngram_size}gram_files/original"

    process_single_file(
        file_url=args.file_url,
        output_dir=output_dir,
        vocab_set=vocab_set,
        overwrite=args.overwrite,
    )
