import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import log

import pandas as pd
from gensim.models import Word2Vec
from tqdm.notebook import tqdm

from ngram_tools.helpers.file_handler import FileHandler


class SentencesIterable:
    """
    An iterable wrapper for sentences generated from JSONL files.
    Allows multiple iterations over the data.
    """

    def __init__(self, file_path, weight_by="freq", log_base=10, year=None):
        self.file_path = file_path
        self.weight_by = weight_by
        self.log_base = log_base
        self.year = year  # Optional year for labeling

    def __iter__(self):
        file_handler = FileHandler(self.file_path)
        desc = (f"Processing Year {self.year}" if self.year
                else f"Processing {self.file_path}")
        with file_handler.open() as file:
            for line in tqdm(file, desc=desc, leave=True):
                try:
                    data = file_handler.deserialize(line)
                    ngram_tokens = data['ngram'].split()
                    freq = data['freq']
                    doc = data['doc']

                    # Weighting logic
                    if self.weight_by == "freq":
                        yield from repeat(ngram_tokens, freq)
                    elif self.weight_by == "doc_freq":
                        weight = int(freq * log(doc, self.log_base))
                        yield from repeat(ngram_tokens, weight)
                    else:  # No weighting
                        yield ngram_tokens
                except Exception as e:
                    logging.error(f"Error processing line: {line}. Error: {e}")


def load_results_file(results_path):
    """
    Load or initialize a results DataFrame.
    """
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    else:
        columns = [
            "model_spec", "year", "weight_by", "vector_size",
            "window", "min_count", "similarity_score", "analogy_score"
        ]
        return pd.DataFrame(columns=columns)


def append_results(
    df,
    model_spec,
    year,
    weight_by,
    vector_size,
    window,
    min_count,
    similarity_score,
    analogy_score
):
    """
    Append a new row of results to the DataFrame.
    """
    new_row = {
        "model_spec": model_spec,
        "year": year,
        "weight_by": weight_by,
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "similarity_score": similarity_score,
        "analogy_score": analogy_score
    }
    return df.append(new_row, ignore_index=True)


def save_results_file(df, results_path):
    """
    Save the results DataFrame to a CSV file.
    """
    df.to_csv(results_path, index=False)


def train_word2vec(
    file_path,
    weight_by="freq",
    log_base=10,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    **kwargs
):
    """
    Train a Word2Vec model on the given sentences.
    """
    sentences = SentencesIterable(
        file_path,
        weight_by=weight_by,
        log_base=log_base
    )
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        **kwargs
    )
    return model


def train_model_for_year(
    year,
    data_dir,
    weight_by,
    log_base,
    vector_size,
    window,
    min_count,
    workers,
    results_path
):
    """
    Train a Word2Vec model for a single year and record results.
    """
    file_path = f"{data_dir}/{year}.jsonl.lz4"

    if not os.path.exists(file_path):
        logging.warning(f"File for year {year} not found. Skipping...")
        return

    # Ensure log directory exists within data_dir
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging for the year
    year_logger = logging.getLogger(f"Year_{year}")
    year_logger.setLevel(logging.INFO)
    log_file_path = f"{log_dir}/year_{year}_process.log"
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    year_logger.addHandler(handler)

    # Direct Gensim logging to the same file
    gensim_logger = logging.getLogger("gensim")
    gensim_logger.setLevel(logging.INFO)
    gensim_logger.addHandler(handler)

    try:
        year_logger.info(f"Processing year {year}...")

        # Train model
        model = train_word2vec(
            file_path=file_path,
            weight_by=weight_by,
            log_base=log_base,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )

        # Construct model filename with parameters
        model_filename = (
            f"w2v_{year}_{weight_by}_{vector_size}_{window}_{min_count}.model")
        model_save_path = os.path.join(data_dir, model_filename)
        model.save(model_save_path)
        year_logger.info(f"Model for year {year} saved to {model_save_path}.")

        # Simulated evaluation results (replace with actual evaluation calls)
        similarity_score = 0.5  # Placeholder
        analogy_score = 0.4  # Placeholder

        # Load or initialize results DataFrame
        results_df = load_results_file(results_path)

        # Append results
        results_df = append_results(
            results_df,
            model_spec=(
                f"{year}_{weight_by}_{vector_size}_{window}_{min_count}"
            ),
            year=year,
            weight_by=weight_by,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            similarity_score=similarity_score,
            analogy_score=analogy_score
        )

        # Save results
        save_results_file(results_df, results_path)
    except Exception as e:
        year_logger.error(f"Error training model for year {year}: {e}")
    finally:
        # Remove handlers to prevent duplicate logging
        year_logger.removeHandler(handler)
        gensim_logger.removeHandler(handler)
        handler.close()


def train_models_by_years(
    data_dir,
    years,
    weight_by="freq",
    log_base=10,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
):
    """
    Train Word2Vec models for a range of years using multiprocessing.

    Args:
        data_dir (str): Directory containing JSONL files.
        years (tuple): Start and end year (inclusive).
        weight_by (str): Weighting for n-grams ('none', 'freq', 'doc_freq').
        log_base (float): Base for logarithm in document weighting.
        vector_size (int): Size of word vectors.
        window (int): Maximum distance between current and predicted word.
        min_count (int): Minimum frequency of words to include.
        workers (int): Number of worker threads for training.
    """
    results_path = os.path.join(data_dir, "training_results.csv")

    start_year, end_year = years
    year_range = list(range(start_year, end_year + 1))

    with tqdm(
        total=len(year_range), desc="Training Models", leave=True
    ) as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    train_model_for_year,
                    year,
                    data_dir,
                    weight_by,
                    log_base,
                    vector_size,
                    window,
                    min_count,
                    workers,
                    results_path
                )
                for year in year_range
            ]
            for future in futures:
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Error occurred while processing: {e}")


def parse_args():
    """
    Parse command-line arguments for model training.
    """
    parser = argparse.ArgumentParser(
        description="Train Word2Vec models on yearly n-gram data."
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing JSONL files.'
    )
    parser.add_argument(
        '--start_year',
        type=int,
        required=True,
        help='Start year for training.'
    )
    parser.add_argument(
        '--end_year',
        type=int,
        required=True,
        help='End year for training.'
    )
    parser.add_argument(
        '--weight_by',
        type=str,
        default='freq',
        choices=['none', 'freq', 'doc_freq'],
        help='Weighting strategy for n-grams.'
    )
    parser.add_argument(
        '--log_base',
        type=float,
        default=10,
        help='Logarithm base for document weighting.'
    )
    parser.add_argument(
        '--vector_size',
        type=int,
        default=100,
        help='Size of word vectors.'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=5,
        help='Maximum distance between current and predicted word.'
    )
    parser.add_argument(
        '--min_count',
        type=int,
        default=1,
        help='Minimum frequency of words to include.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of worker threads for training.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    train_models_by_years(
        data_dir=args.data_dir,
        years=(args.start_year, args.end_year),
        weight_by=args.weight_by,
        log_base=args.log_base,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers
    )