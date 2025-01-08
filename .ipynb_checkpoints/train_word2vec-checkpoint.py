import argparse
import logging
from gensim.models import Word2Vec
import multiprocessing

# Set up logging to log to a file
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    filename='/scratch/edk202/hist_w2v/training_log.txt',
    filemode='w'
)

def load_corpus(corpus_file):
    """
    Function to load and tokenize the corpus.
    Each line in the corpus file should be a space-separated list of word_POS tokens.
    """
    sentences = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip().split()  # Tokenize the line (space-separated tokens)
            sentences.append(sentence)
    return sentences

def train_word2vec_model(sentences, vector_size, window, sg, negative, min_count, sample, workers, epochs, alpha, min_alpha):
    """
    Train a Word2Vec model using the given sentences with multi-threading.
    """
    model = Word2Vec(
        sentences,          # The tokenized corpus
        vector_size=300,    # Dimensionality of word vectors
        window=5,           # Context window size
        sg=1,               # Use Skip-gram (1) or CBOW (0)
        negative=10,        # Negative sampling (number of samples per update)
        min_count=5,        # Minimum word frequency
        sample=1e-5,        # Subsampling of frequent words
        workers=8,          # Number of CPU cores to use
        epochs=5,           # Number of training epochs
        alpha=0.025,        # Initial learning rate
        min_alpha=0.0001,   # Minimum learning rate
    )
    return model

def save_model(model, model_file, vector_file):
    """
    Save the trained Word2Vec model and its word vectors.
    """
    # Save the Word2Vec model
    model.save(model_file)
    print(f"Word2Vec model saved to {model_file}")
    
    # Save the word vectors to a text file (plain text format)
    model.wv.save_word2vec_format(vector_file, binary=False)
    print(f"Word vectors saved to {vector_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a Word2Vec model using a space-delimited corpus file.")
    parser.add_argument('--corpus_file', type=str, required=True, help="Path to the tokenized corpus.")
    parser.add_argument('--model_file', type=str, required=True, help="Path to save the trained Word2Vec model.")
    parser.add_argument('--vector_file', type=str, required=True, help="Path to save the word vectors.")
    parser.add_argument('--vector_size', type=int, default=300, help="Dimensionality of word vectors.")
    parser.add_argument('--window', type=int, default=5, help="Context window size.")
    parser.add_argument('--sg', type=int, choices=[0, 1], default=1, help="Use skip-gram (1) or CBOW (0).")
    parser.add_argument('--negative', type=int, default=10, help="Negative sampling (number of samples per update).")
    parser.add_argument('--min_count', type=int, default=1, help="Minimum word frequency.")
    parser.add_argument('--sample', type=float, default=1e-5, help="Subsampling of frequent words.")
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--alpha', type=float, default=0.025, help="Initial learning rate.")
    parser.add_argument('--min_alpha', type=float, default=0.0001, help="Minimum learning rate.")

    args = parser.parse_args()

    # Load the corpus from the file
    sentences = load_corpus(args.corpus_file)
    print(f"Loaded {len(sentences)} sentences from the corpus.")

    # Train the Word2Vec model with multi-threading
    model = train_word2vec_model(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        sg=args.sg,
        negative=args.negative,
        min_count=args.min_count,
        sample=args.sample,
        workers=args.workers,
        epochs=args.epochs,
        alpha=args.alpha,
        min_alpha=args.min_alpha
    )
    
    # Save the trained model and word vectors
    save_model(model, args.model_file, args.vector_file)

if __name__ == "__main__":
    main()
