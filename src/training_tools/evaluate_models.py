import argparse
import logging
import os
import re
import sys
from datetime import datetime
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from tqdm.notebook import tqdm
import pandas as pd
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Word2Vec models on intrinsic tasks."
    )
    parser.add_argument(
        "--ngram_size",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Ngram size."
    )
    parser.add_argument(
        '--proj_dir',
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
        '--eval_dir',
        type=str,
        default="evaluation_results.csv",
        help='Path to directory to store evaluation results.'
    )
    parser.add_argument(
        '--save_mode',
        type=str,
        choices=['overwrite', 'append'],
        default="append",
        help='Path to directory to store evaluation results.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count(),
        help='Number of parallel workers to evaluate the models.'
    )
    return parser.parse_args()


def set_info(
    ngram_size,
    proj_dir,
    eval_dir,
    similarity_dataset,
    analogy_dataset
):
    """
    Set up project paths and other info.
    """
    start_time = datetime.now()

    # Construct path to models
    model_dir = os.path.join(
        proj_dir, f"{ngram_size}gram_files/6corpus/yearly_files/models"
    )
    if not os.path.exists(model_dir):
        logging.error(f"Specified model directory does not exist: {model_dir}")
        return

    # Construct evaluation output file path
    if os.path.exists(eval_dir):
        eval_file = os.path.join(eval_dir, "evaluation_results.csv")
    else:
        logging.error(f"Specified evaluation directory does not exist: {eval_dir}")
        return

    # Construct logging path (and create if necessary)
    log_dir = os.path.join(
        proj_dir, f"{ngram_size}gram_files/6corpus/yearly_files/logs/evaluation"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Get similarity and analogy datasets
    if similarity_dataset is None:
        similarity_dataset = datapath('wordsim353.tsv')
    if analogy_dataset is None:
        analogy_dataset = datapath('questions-words.txt')

    return (
        start_time,
        model_dir,
        eval_file,
        log_dir,
        similarity_dataset,
        analogy_dataset
    )


def print_info(
    start_time,
    model_dir,
    eval_file,
    log_dir,
    ngram_size,
    similarity_dataset,
    analogy_dataset,
    save_mode=None,
):
    """
    Print project setup information to the console.
    """
    print(f"\033[31mStart Time:            {start_time}\n\033[0m")
    print("\033[4mEvaluation Info\033[0m")
    print(f"Ngram size:            {ngram_size}")
    print(f"Model directory:       {model_dir}")
    print(f"Evaluation file path:  {eval_file}")
    print(f"Log directory:         {log_dir}")
    print(f"Save mode:             {save_mode}")
    print(f"Similarity dataset:    {similarity_dataset}")
    print(f"Analogy dataset:       {analogy_dataset}\n")


def evaluate_a_model(model_path, similarity_dataset, analogy_dataset, model_logger):
    """
    Run intrinsic evaluations on a Word2Vec model, using a model-specific logger.
    """
    model_logger.info(f"Loading KeyedVectors from: {model_path}")
    try:
        model = KeyedVectors.load(model_path)
    except Exception as e:
        model_logger.error(f"Failed to load KeyedVectors from {model_path}: {e}")
        return None

    # Word similarity evaluation
    sim_results = model.evaluate_word_pairs(similarity_dataset)
    similarity_score = sim_results[0][0]  # Spearman correlation
    model_logger.info(f"Similarity Score (Spearman): {similarity_score}")

    # Word analogy evaluation
    analogy_results = model.evaluate_word_analogies(analogy_dataset)
    analogy_score = analogy_results[0]  # Overall accuracy
    model_logger.info(f"Analogy Score: {analogy_score}")

    return {
        "similarity_score": similarity_score,
        "analogy_score": analogy_score
    }


def extract_model_metadata(file_name):
    """
    Extract metadata from the model filename using regex.
    """
    pattern = re.compile(
        r"w2v_y(\d+)_wb(\w+)_vs(\d+)_w(\d+)_mc(\d+)_sg(\d+)_e(\d+)\.kv"
    )
    match = pattern.match(file_name)
    if match:
        return match.groups()
    return None


def evaluate_one_file(params):
    """
    Helper to evaluate a single model file with its own log file. 
    params is a tuple of (file_name, model_dir, similarity_dataset, analogy_dataset, log_dir).
    """
    (file_name, model_dir, similarity_dataset, analogy_dataset, log_dir) = params

    metadata = extract_model_metadata(file_name)
    if not metadata:
        # The filename doesn't match the pattern, skip
        return None

    (year, weight_by, vector_size, window,
     min_count, approach, epochs) = metadata

    model_path = os.path.join(model_dir, file_name)

    # Create a unique log file for this model
    model_log_file = os.path.join(
        log_dir,
        f"{os.path.splitext(file_name)[0]}.log"
    )

    # Set up a local logger for this specific model
    model_logger = logging.getLogger(f"logger_{file_name}")
    model_logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplication (especially if code is re-run)
    while model_logger.handlers:
        model_logger.handlers.pop()

    file_handler = logging.FileHandler(model_log_file, mode='a')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    model_logger.addHandler(file_handler)

    # Start logging
    model_logger.info("--------------------------------------------------")
    model_logger.info(f"Beginning evaluation for model: {file_name}")

    try:
        evaluation = evaluate_a_model(
            model_path,
            similarity_dataset=similarity_dataset,
            analogy_dataset=analogy_dataset,
            model_logger=model_logger
        )
        if not evaluation:
            model_logger.info("Evaluation returned None.")
            return None

        result_dict = {
            "model": file_name,
            "year": int(year),
            "weight_by": weight_by,
            "vector_size": int(vector_size),
            "window": int(window),
            "min_count": int(min_count),
            "approach": approach,
            "epochs": int(epochs),
            "similarity_score": evaluation["similarity_score"],
            "analogy_score": evaluation["analogy_score"]
        }
        model_logger.info(f"Evaluation completed for {file_name}")
        return result_dict

    except Exception as e:
        model_logger.error(f"Error evaluating {file_name}: {e}")
        return None

    finally:
        # Close the file handler so logs are flushed
        model_logger.removeHandler(file_handler)
        file_handler.close()


def evaluate_models_in_directory(
    model_dir,
    eval_file,
    log_dir,
    save_mode,
    similarity_dataset,
    analogy_dataset,
    workers=os.cpu_count()
):
    """
    Evaluate all Word2Vec models in a directory using multiprocessing.Pool,
    and create a dedicated log file for each model.
    """
    # Identify which files haven't been evaluated yet
    if os.path.isfile(eval_file):
        existing = pd.read_csv(eval_file)['model'].values
    else:
        existing = []

    files_to_evaluate = [
        f for f in os.listdir(model_dir)
        if f.endswith('.kv') and f not in existing
    ]
    if not files_to_evaluate:
        logging.info("No new files to evaluate.")
        return

    # Build list of parameter tuples for parallel processing
    param_list = [
        (f, model_dir, similarity_dataset, analogy_dataset, log_dir)
        for f in files_to_evaluate
    ]

    results = []
    with Pool(processes=workers) as pool:
        # imap_unordered yields results as they come in, not in order
        for result in tqdm(
            pool.imap_unordered(evaluate_one_file, param_list),
            total=len(param_list),
            desc="Evaluating",
            leave=True
        ):
            if result is not None:
                results.append(result)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        if save_mode == 'overwrite':
            df.to_csv(eval_file, mode='w', index=False)
        else:
            file_exists = os.path.isfile(eval_file)
            df.to_csv(eval_file, mode='a', index=False, header=not file_exists)
        logging.info(f"Evaluation results saved to: {eval_file}")
    else:
        logging.warning("No valid results were returned after evaluation.")


def evaluate_models(
    ngram_size,
    proj_dir,
    eval_dir,
    save_mode,
    similarity_dataset=None,
    analogy_dataset=None,
    workers=os.cpu_count()
):
    info = set_info(
        ngram_size,
        proj_dir,
        eval_dir,
        similarity_dataset,
        analogy_dataset
    )
    if not info:
        return  # set_info encountered a problem and already logged an error

    (
        start_time,
        model_dir,
        eval_file,
        log_dir,
        similarity_dataset,
        analogy_dataset
    ) = info

    print_info(
        start_time,
        model_dir,
        eval_file,
        log_dir,
        ngram_size,
        similarity_dataset,
        analogy_dataset,
        save_mode
    )

    evaluate_models_in_directory(
        model_dir=model_dir,
        eval_file=eval_file,
        log_dir=log_dir,
        save_mode=save_mode,
        similarity_dataset=similarity_dataset,
        analogy_dataset=analogy_dataset,
        workers=workers
    )


if __name__ == "__main__":
    args = parse_args()

    evaluate_models(
        ngram_size=args.ngram_size,
        proj_dir=args.proj_dir,
        eval_dir=args.eval_dir,
        save_mode=args.save_mode,
        similarity_dataset=args.similarity_dataset,
        analogy_dataset=args.analogy_dataset,
        workers=args.workers
    )