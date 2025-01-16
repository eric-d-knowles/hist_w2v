import argparse
import logging
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from itertools import product, repeat
from math import log, floor

from gensim.models import Word2Vec
from tqdm.notebook import tqdm

from ngram_tools.helpers.file_handler import FileHandler


def ensure_iterable(param):
    """
    Ensure the input parameter is iterable (e.g., a tuple).
    """
    return param if isinstance(param, (tuple, list)) else (param,)


def set_info(ngram_size, proj_dir):
    """
    Set up project paths for data, models, and logs.

    Args:
        ngram_size (int): The size of ngrams.
        proj_dir (str): Base project directory.

    Returns:
        tuple: Start time, data directory, model directory, log directory.
    """
    start_time = datetime.now()
    data_dir = os.path.join(
        proj_dir, f"{ngram_size}gram_files/6corpus/yearly_files/data"
    )
    model_dir = os.path.join(
        proj_dir, f"{ngram_size}gram_files/6corpus/yearly_files/models"
    )
    log_dir = os.path.join(
        proj_dir, f"{ngram_size}gram_files/6corpus/yearly_files/logs"
    )
    return start_time, data_dir, model_dir, log_dir


def print_info(start_time, data_dir, model_dir, log_dir, ngram_size, workers):
    """
    Print project setup information.

    Args:
        start_time (datetime): Start time of the process.
        data_dir (str): Data directory path.
        model_dir (str): Model directory path.
        log_dir (str): Log directory path.
        ngram_size (int): The size of ngrams.
        workers (int): Number of workers for multiprocessing.
    """
    print(f"\033[31mStart Time:                {start_time}\n\033[0m")
    print("\033[4mTraining Info\033[0m")
    print(f"Data directory:            {data_dir}")
    print(f"Model directory:           {model_dir}")
    print(f"Log directory:             {log_dir}")
    print(f"Ngram size:                {ngram_size}")
    print(f"Number of workers:         {workers}\n")


def calculate_weight(freq, base=10):
    """
    Calculate the weight of an n-gram using logarithmic scaling.

    Args:
        freq (int): Raw frequency of the n-gram.
        base (float): Logarithm base for scaling.

    Returns:
        int: Scaled weight.
    """
    return max(1, floor(log(freq + 1, base)))  # Minimum weight is 1


class SentencesIterable:
    """
    An iterable wrapper for sentences generated from JSONL files.
    Allows multiple iterations over the data.
    """

    def __init__(self, file_path, weight_by="freq", log_base=10, year=None):
        self.file_path = file_path
        self.weight_by = weight_by
        self.log_base = log_base
        self.year = year

    def __iter__(self):
        file_handler = FileHandler(self.file_path)
        desc = (
            f"Processing Year {self.year}"
            if self.year else f"Processing {self.file_path}"
        )
        with file_handler.open() as file:
            for line in tqdm(file, desc=desc, leave=True):
                data = file_handler.deserialize(line)
                ngram_tokens = data["ngram"].split()
                freq = data["freq"]
                doc = data["doc"]

                if self.weight_by == "freq":
                    weight = calculate_weight(freq, base=10)
                    yield from repeat(ngram_tokens, weight)
                elif self.weight_by == "doc_freq":
                    weight = calculate_weight(doc, base=10)
                    yield from repeat(ngram_tokens, weight)
                else:
                    yield ngram_tokens


def configure_logging(log_dir, filename):
    """
    Configure and return a logger for a child process, adding Gensim's logs.

    Args:
        log_dir (str): Directory to store log files.
        filename (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, filename)

    # Create a logger named after the filename
    logger_name = os.path.splitext(filename)[0]
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    if not logger.hasHandlers():
        # Create and configure a file handler
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)

        # Attach the handler to Gensim's logger to include its output
        gensim_logger = logging.getLogger("gensim")
        gensim_logger.setLevel(logging.INFO)
        if (
            not any(
                isinstance(h, logging.FileHandler)
                and h.baseFilename == log_file_path for h in gensim_logger.handlers
            )
        ):
            gensim_logger.addHandler(file_handler)

    return logger


def train_word2vec(
    file_path,
    weight_by,
    vector_size,
    window,
    min_count,
    workers,
    **kwargs
):
    """
    Train a Word2Vec model on the given sentences.

    Args:
        file_path (str): Path to the JSONL file containing ngrams.
        weight_by (str): Weighting strategy ("freq", "doc_freq", or "none").
        vector_size (int): Size of word vectors.
        window (int): Context window size.
        min_count (int): Minimum frequency of words to include.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    sentences = SentencesIterable(
        file_path, weight_by=weight_by, log_base=10
    )
    return Word2Vec(sentences, vector_size=vector_size, window=window,
                    min_count=min_count, workers=workers, **kwargs)


def train_model(year, data_dir, model_dir, log_dir, weight_by,
                vector_size, window, min_count, workers):
    """
    Train a Word2Vec model for a specific year.

    Args:
        year (int): Year to process.
        data_dir (str): Directory containing JSONL files.
        model_dir (str): Directory to save trained models.
        log_dir (str): Directory to save logs.
        weight_by (str): Weighting strategy.
        vector_size (int): Size of word vectors.
        window (int): Context window size.
        min_count (int): Minimum frequency of words.
        workers (int): Number of worker threads.
    """
    logger = configure_logging(
        log_dir,
        filename=(
            f"w2v_{year}_{weight_by}_{vector_size}_{window}_{min_count}.log"
        )
    )

    file_path = os.path.join(data_dir, f"{year}.jsonl.lz4")

    if not os.path.exists(file_path):
        logger.warning(f"File for year {year} not found. Skipping...")
        return

    os.makedirs(model_dir, exist_ok=True)

    try:
        logger.info(
            f"Processing year {year} with parameters: "
            f"vector_size={vector_size}, window={window}, "
            f"min_count={min_count}..."
        )

        model = train_word2vec(
            file_path=file_path,
            weight_by=weight_by,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )

        model_filename = (
            f"w2v_{year}_{weight_by}_{vector_size}_{window}_{min_count}.kv"
        )
        model_save_path = os.path.join(model_dir, model_filename)
        model.wv.save(model_save_path)

        logger.info(f"Model for year {year} saved to {model_save_path}.")
    except Exception as e:
        logger.error(f"Error training model for year {year}: {e}")


def train_models(
    ngram_size,
    proj_dir,
    years,
    weight_by=("freq",),
    vector_size=(100,),
    window=(5,),
    min_count=(1,),
    workers=os.cpu_count()
):
    """
    Train Word2Vec models for multiple years.

    Args:
        ngram_size (int): Size of ngrams.
        proj_dir (str): Base project directory.
        years (tuple): Range of years to process.
        weight_by (tuple): Weighting strategies.
        vector_size (tuple): Sizes of word vectors.
        window (tuple): Context window sizes.
        min_count (tuple): Minimum frequencies of words.
        workers (int): Number of worker threads.
    """
    start_time, data_dir, model_dir, log_dir = set_info(ngram_size, proj_dir)

    print_info(start_time, data_dir, model_dir, log_dir, ngram_size, workers)

    weight_by = ensure_iterable(weight_by)
    vector_size = ensure_iterable(vector_size)
    window = ensure_iterable(window)
    min_count = ensure_iterable(min_count)

    param_combinations = list(
        product(weight_by, vector_size, window, min_count)
    )
    years = range(years[0], years[1] + 1)

    tasks = [
        (year, data_dir, model_dir, log_dir, params[0], params[1],
         params[2], params[3], workers)
        for year in years for params in param_combinations
    ]

    with tqdm(total=len(tasks), desc="Training Models", leave=True) as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(train_model, *task) for task in tasks]
            for future in futures:
                future.result()
                pbar.update(1)


def parse_args():
    """
    Parse command-line arguments for model training.
    """
    parser = argparse.ArgumentParser(description="Train Word2Vec models.")
    parser.add_argument(
        "--ngram_size",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Ngram size."
    )
    parser.add_argument(
        "--proj_dir",
        type=str,
        required=True,
        help="Project directory."
    )
    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        help="Start year."
    )
    parser.add_argument(
        "--end_year",
        type=int,
        required=True,
        help="End year."
    )
    parser.add_argument(
        "--weight_by",
        type=str,
        choices=['none', 'freq', 'doc_freq'],
        default='none',
        help='Ngram weighting strategy.'
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=100,
        help="Number of vector dimensions."
    )
    parser.add_argument(
        "--window",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=3,
        help="Width of context window."
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum ngram frequency."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(
        ngram_size=args.ngram_size,
        proj_dir=args.proj_dir,
        years=(args.start_year, args.end_year),
        weight_by=args.weight_by,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers
    )