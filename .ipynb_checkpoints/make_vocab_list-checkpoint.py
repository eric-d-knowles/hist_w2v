import argparse
import orjson
import logging
import os
from tqdm import tqdm

def load_vocab(input_file, n_vocab=None):
    vocab = []
    try:
        print("Loading unigram file.\n")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Lines", unit="line"):
                token_data = orjson.loads(line.strip())  # Use orjson to load the line
                token = token_data['ngram']
                index = token_data.get('index')  # Use 'index' to sort later
                vocab.append((token, index))
        
        vocab.sort(key=lambda x: x[1])

        if n_vocab:
            vocab = vocab[:n_vocab]
        
        print(f"\nVocabulary contains {len(vocab)} words.\n")
    except Exception as e:
        logging.error(f"Error loading unigram file {input_file}: {e}")
        raise RuntimeError(f"Error loading unigram file {input_file}: {e}")
    
    # Return the token-index pairs for JSONL saving, and also a frozenset for membership testing
    return vocab, frozenset(token for token, _ in vocab)

def save_vocab_for_lookup(vocab, output_file, plain_text=False):
    """ Save the token-to-index mapping to a file in either JSONL or plain text format. """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if plain_text:
            # Save as plain text (token <TAB> index per line)
            with open(output_file, 'w', encoding='utf-8') as f:
                for token, index in vocab:
                    f.write(f"{token}\t{index}\n")
            print(f"Vocabulary saved as plain text to {output_file}\n")
        else:
            # Save as JSONL
            with open(output_file, 'wb') as f:
                for token, index in vocab:
                    f.write(orjson.dumps({'token': token, 'index': index}) + b'\n')
            print(f"Vocabulary saved as JSONL to {output_file}\n")
    except Exception as e:
        logging.error(f"Error saving vocabulary to {output_file}: {e}")
        raise RuntimeError(f"Error saving vocabulary to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Build a ranked vocabulary list from a JSONL unigram file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSONL file containing unigrams.")
    parser.add_argument("--n_vocab", type=int, default=None, help="Limit the vocabulary size to the top n_vocab items. Default is None (no limit).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the token-to-index file for lookup.")
    parser.add_argument("--membership_file", type=str, required=True, help="Path to save the frozenset for membership testing.")
    parser.add_argument("--plain_text", action="store_true", help="Save the vocabulary as plain text instead of JSONL.")
    args = parser.parse_args()

    # Load the vocabulary (returns list of (token, index) tuples and a frozenset for membership testing)
    vocab, vocab_set = load_vocab(args.input_file, n_vocab=args.n_vocab)

    # Save the token-to-index mapping in the requested format (plain text or JSONL)
    save_vocab_for_lookup(vocab, args.output_file, plain_text=args.plain_text)

    # Save the frozenset for membership testing
    try:
        # Ensure the membership directory exists
        os.makedirs(os.path.dirname(args.membership_file), exist_ok=True)
        
        with open(args.membership_file, 'wb') as f:
            f.write(orjson.dumps(list(vocab_set)) + b'\n')  # Save the frozenset as a list for future loading
        print(f"Frozenset saved to {args.membership_file}\n")
    except Exception as e:
        logging.error(f"Error saving frozenset to {args.membership_file}: {e}")
        raise RuntimeError(f"Error saving frozenset to {args.membership_file}: {e}")

if __name__ == "__main__":
    main()
