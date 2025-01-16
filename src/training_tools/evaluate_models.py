import argparse
import logging
import os
import re
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath
import pandas as pd


def evaluate_word2vec(model_path, similarity_dataset=None, analogy_dataset=None):
    """
    Run intrinsic evaluations on a Word2Vec model.

    Args:
        model_path (str): Path to the Word2Vec model file.
        similarity_dataset (str): Path to similarity data (or Gensim default).
        analogy_dataset (str): Path to analogy data (or Gensim default).

    Returns:
        dict: Evaluation results.
    """
    logging.info(f"Loading KeyedVectors from: {model_path}")
    try:
        model = KeyedVectors.load(model_path)
    except Exception as e:
        logging.error(f"Failed to load KeyedVectors from {model_path}: {e}")
        return None

    # Word similarity evaluation
    if similarity_dataset is None:
        similarity_dataset = datapath('wordsim353.tsv')
    sim_results = model.evaluate_word_pairs(similarity_dataset)
    similarity_score = sim_results[0][0]  # Spearman correlation

    # Word analogy evaluation
    if analogy_dataset is None:
        analogy_dataset = datapath('questions-words.txt')
    analogy_results = model.evaluate_word_analogies(analogy_dataset)
    analogy_score = analogy_results[0]  # Overall accuracy

    return {
        "similarity_score": similarity_score,
        "analogy_score": analogy_score
    }


def extract_model_metadata(file_name):
    """
    Extract metadata from the model filename using regex.

    Args:
        file_name (str): Filename of the model.

    Returns:
        tuple: Extracted metadata or None if no match is found.
    """
    pattern = re.compile(r"w2v_(\d+)_(\w+)_([\d]+)_([\d]+)_([\d]+)\.kv")
    match = pattern.match(file_name)
    if match:
        return match.groups()
    return None


def evaluate_models(
    model_dir,
    similarity_dataset=None,
    analogy_dataset=None,
    results_file="evaluation_results.csv"
):
    """
    Evaluate all Word2Vec models in a directory based on their filenames.

    Args:
        model_dir (str): Directory containing Word2Vec models.
        similarity_dataset (str): Path to similarity data (or Gensim default).
        analogy_dataset (str): Path to analogy data (or Gensim default).
        results_file (str): Path to save the evaluation results as a CSV file.
    """
    if not os.path.exists(model_dir):
        logging.error(f"Specified data directory does not exist: {model_dir}")
        return

    results = []
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.kv'):
            metadata = extract_model_metadata(file_name)
            if metadata:
                year, weight_by, vector_size, window, min_count = metadata
                model_path = os.path.join(model_dir, file_name)
                logging.info(f"Evaluating model: {file_name}")

                evaluation = evaluate_word2vec(
                    model_path,
                    similarity_dataset=similarity_dataset,
                    analogy_dataset=analogy_dataset
                )

                if evaluation:
                    results.append({
                        "model": file_name,
                        "year": int(year),
                        "weight_by": weight_by,
                        "vector_size": int(vector_size),
                        "window": int(window),
                        "min_count": int(min_count),
                        "similarity_score": evaluation["similarity_score"],
                        "analogy_score": evaluation["analogy_score"]
                    })

    # Save results to a CSV file
    if results:
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        logging.info(f"Evaluation results saved to: {results_file}")
    else:
        logging.warning(
            "No KeyedVector were evaluated or no valid results were found."
        )


def parse_args():
    """
    Parse command-line arguments for model evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Word2Vec models on intrinsic tasks."
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing Word2Vec models.'
    )
    parser.add_argument(
        '--similarity_dataset',
        type=str,
        default=None,
        help='Path to similarity data (leave empty for Gensim default).'
    )
    parser.add_argument(
        '--analogy_dataset',
        type=str,
        default=None,
        help='Path to analogy data (leave empty for Gensim default).'
    )
    parser.add_argument(
        '--results_file',
        type=str,
        default="evaluation_results.csv",
        help='Path to save the evaluation results as a CSV file.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()

    evaluate_models(
        model_dir=args.model_dir,
        similarity_dataset=args.similarity_dataset,
        analogy_dataset=args.analogy_dataset,
        results_file=args.results_file
    )