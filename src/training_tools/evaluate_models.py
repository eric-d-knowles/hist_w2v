import argparse
import os
import logging
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


def evaluate_word2vec(
    model_path,
    similarity_dataset=None,
    analogy_dataset=None
):
    """
    Run intrinsic evaluations on a Word2Vec model.

    Args:
        model_path (str): Path to the Word2Vec model file.
        similarity_dataset (str): Path to similarity data (or Gensim default).
        analogy_dataset (str): Path to analogy data (or Gensim default).

    Returns:
        dict: Evaluation results.
    """
    model = KeyedVectors.load(model_path)

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


def evaluate_models_by_years(
    data_dir,
    years,
    similarity_dataset=None,
    analogy_dataset=None
):
    """
    Evaluate Word2Vec models for a range of years.

    Args:
        data_dir (str): Directory containing Word2Vec models.
        years (tuple): Start and end year (inclusive).
        similarity_dataset (str): Path to similarity data (or Gensim default).
        analogy_dataset (str): Path to analogy data (or Gensim default).

    Returns:
        dict: Evaluation results for all years.
    """
    start_year, end_year = years
    evaluation_results = {}

    for year in range(start_year, end_year + 1):
        model_path = f"{data_dir}/word2vec_{year}.model"
        if os.path.exists(model_path):
            logging.info(f"Evaluating model for year {year}...")
            results = evaluate_word2vec(
                model_path, similarity_dataset, analogy_dataset
            )
            evaluation_results[year] = results
            logging.info(
                f"Year {year} - Similarity: "
                f"{results['similarity_score']:.4f}, "
                f"Analogy: {results['analogy_score']:.4f}"
            )
        else:
            logging.warning(f"Model for year {year} not found. Skipping...")

    return evaluation_results


def parse_args():
    """
    Parse command-line arguments for model evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Word2Vec models on yearly intrinsic tasks."
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing Word2Vec models.'
    )
    parser.add_argument(
        '--start_year',
        type=int,
        required=True,
        help='Start year for evaluation.'
    )
    parser.add_argument(
        '--end_year',
        type=int,
        required=True,
        help='End year for evaluation.'
    )
    parser.add_argument(
        '--similarity_dataset',
        type=str,
        required=False,
        default=None,
        help='Path to similarity data (leave empty for Gensim default).'
    )
    parser.add_argument(
        '--analogy_dataset',
        type=str,
        required=False,
        default=None,
        help='Path to an analogy dataset (leave empty for Gensim default).'
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    evaluate_models_by_years(
        data_dir=args.data_dir,
        years=(args.start_year, args.end_year),
        similarity_dataset=args.similarity_dataset,
        analogy_dataset=args.analogy_dataset
    )