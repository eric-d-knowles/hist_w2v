import argparse
import os
from collections import Counter
import torch

def build_vocab(corpus_file, output_dir):
    """
    Builds a vocabulary from a weighted corpus in plain text format and saves the mapping.
    This version is designed to work with PyTorch's Dataset and DataLoader objects.
    """
    print("Loading weighted corpus from plain text file.\n")
    
    # Initialize a counter to hold the word frequencies
    word_counter = Counter()

    # Read the plain text file
    with open(corpus_file, 'r') as f:
        for line in f:
            # Tokenize each line (ngram) and update word counter
            ngram = line.strip().split()
            word_counter.update(ngram)

    # Add a special token for unknown words
    word_counter['<UNK>'] = 0  # `<UNK>` token for unknown words

    # Build vocabulary using the Counter object
    vocab = {word: idx for idx, (word, _) in enumerate(word_counter.items())}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the word-to-index mapping
    word_to_index_path = os.path.join(output_dir, 'word_to_index.txt')
    
    print(f"Vocabulary size: {len(vocab):,}\n")
    print(f"Saving word-to-index mapping to {word_to_index_path}\n")
    
    with open(word_to_index_path, 'w') as f:
        for word, idx in vocab.items():
            f.write(f"{word} {idx}\n")
    
    print("Vocabulary and mapping successfully saved.")

    return vocab

def main():
    parser = argparse.ArgumentParser(description="Create vocabulary from weighted corpus plain text.")
    parser.add_argument("--corpus_file", type=str, required=True, help="Path to the weighted corpus plain text file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the vocabulary files.")
    args = parser.parse_args()
    
    # Generate the vocabulary and save it
    vocab = build_vocab(args.corpus_file, args.output_dir)
    
    # Optionally, create a PyTorch-friendly tensor version of the vocab
    torch_vocab = {word: torch.tensor(idx) for word, idx in vocab.items()}
    return torch_vocab

if __name__ == "__main__":
    main()
