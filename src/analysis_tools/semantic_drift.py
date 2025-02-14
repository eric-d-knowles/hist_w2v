import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from common.w2v_model import W2VModel
from multiprocessing import Pool, cpu_count


def compute_similarity(args):
    """Helper function for multiprocessing: Computes mean cosine similarity and its standard deviation."""
    year, model_path, anchor_model, word = args
    try:
        model = W2VModel(model_path)

        # Ensure all vocabulary words are strings
        model_vocab = {str(word) for word in model.vocab}
        anchor_vocab = {str(word) for word in anchor_model.vocab}

        if word:
            # Ensure word is a string and exists in both models
            word = str(word)  
            if word in model_vocab and word in anchor_vocab:
                similarity = np.dot(model.model[word], anchor_model.model[word])  # ✅ Correct
                std_dev = 0  # A single word doesn't have a spread
                shared_vocab_size = 1  # Only 1 word being compared
                return (year, similarity, std_dev, shared_vocab_size)
            else:
                raise ValueError(f"Word '{word}' not found in one or both models.")
        else:
            # Compute similarity for all shared words (after ensuring valid string keys)
            common_words = model_vocab.intersection(anchor_vocab)
            if not common_words:
                raise ValueError("No shared vocabulary.")

            similarities = [
                np.dot(model.model[word], anchor_model.model[word]) for word in common_words
            ]

            mean_similarity = np.mean(similarities)
            std_dev = np.std(similarities, ddof=1)  # Spread of similarity scores
            shared_vocab_size = len(common_words)  # Number of shared words

            return (year, mean_similarity, std_dev, shared_vocab_size)

    except Exception as e:
        return (year, None, str(e), 0)  # Return None and error message


def track_semantic_drift(
    start_year, end_year, model_dir, anchor_year=None, word=None,
    plot=True, smooth=False, sigma=2, confidence=0.95, error_type="CI",
    num_workers=None
):
    """
    Compute semantic drift over time using multiprocessing.

    Args:
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        anchor_year (int or None): The reference year for comparison. Defaults to the last available year.
        word (str or None): If provided, tracks drift for a specific word instead of entire vocabulary.
        plot (bool or int): If `True`, plots without chunking. If an integer `N`, averages every `N` years for plotting.
        smooth (bool): Whether to overlay a smoothing line over the graph.
        sigma (float): Standard deviation for Gaussian smoothing.
        confidence (float): Confidence level for error bands.
        error_type (str): Either "CI" for confidence intervals or "SE" for standard errors.
        num_workers (int or None): Number of parallel workers (default: max CPU cores).

    Returns:
        dict: A dictionary mapping years to (mean cosine similarity, error measure).
    """
    drift_scores = {}
    missing_years = []
    error_years = {}

    # Detect available models
    model_paths = {}
    for year in range(start_year, end_year + 1):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = sorted(glob.glob(model_pattern))
        if model_files:
            model_paths[year] = model_files[-1]  # Pick the most recent file
        else:
            missing_years.append(year)

    if not model_paths:
        print("❌ No valid models found in the specified range. Exiting.")
        return {}

    # Set the anchor model (default: last available year)
    if anchor_year is None:
        anchor_year = max(model_paths.keys())

    if anchor_year not in model_paths:
        raise ValueError(f"Anchor year {anchor_year} not found in available models.")

    print(f"Using {anchor_year} as anchor model...")
    anchor_model = W2VModel(model_paths[anchor_year])
    drift_scores[anchor_year] = (1.0, 0.0)  # Reference year has perfect similarity

    # Prepare multiprocessing arguments
    args = [(year, path, anchor_model, word) for year, path in model_paths.items() if year != anchor_year]

    # Use multiprocessing to compute similarities in parallel
    num_workers = num_workers or min(cpu_count(), len(args))  # Limit workers to available tasks
    with Pool(num_workers) as pool:
        results = pool.map(compute_similarity, args)

    # Process results
    for year, similarity, std_dev, shared_vocab_size in results:
        if similarity is not None:
            if error_type == "CI":
                error_measure = stats.norm.ppf(1 - (1 - confidence) / 2) * std_dev
            elif error_type == "SE":
                error_measure = std_dev / np.sqrt(shared_vocab_size) if shared_vocab_size > 1 else 0
            else:
                raise ValueError("Invalid error_type. Choose 'CI' or 'SE'.")

            drift_scores[year] = (similarity, error_measure)
        else:
            error_years[year] = std_dev  # std_dev contains error message

    # Print missing years and errors
    if missing_years:
        print(f"⚠️ No models found for these years: {missing_years}")
    if error_years:
        print("❌ Errors occurred in the following years:")
        for year, err in error_years.items():
            print(f"  {year}: {err}")

    # Convert to NumPy arrays for plotting
    if not drift_scores:
        print("❌ No valid drift scores computed. Exiting.")
        return {}

    years = np.array(sorted(drift_scores.keys()))
    similarities = np.array([drift_scores[year][0] for year in years])
    error_measures = np.array([drift_scores[year][1] for year in years])

    # Apply Smoothing
    smoothed_values = gaussian_filter1d(similarities, sigma=sigma) if smooth else None

    # ✅ Plot Results
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(years, similarities, marker='o', linestyle='-', label=f"Semantic Drift of '{word}'" if word else "Mean Cosine Similarity", color='blue')

        if smooth and smoothed_values is not None:
            plt.plot(years, smoothed_values, linestyle='--', color='red', label=f'Smoothed Trend')

        if error_measures is not None:
            plt.fill_between(years, similarities - error_measures, similarities + error_measures, color='blue', alpha=0.2, label=f"{int(confidence * 100)}% {error_type}")

        plt.axhline(y=1.0, color="gray", linestyle="--", label=f"Anchor Year: {anchor_year}")

        plt.xlabel("Year")
        plt.ylabel("Cosine Similarity to Anchor Year")
        plt.title(f"Semantic Drift Over Time {'for ' + word if word else ''}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return drift_scores
