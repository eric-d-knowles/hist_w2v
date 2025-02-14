import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from common.w2v_model import W2VModel
from multiprocessing import Pool, cpu_count


def compute_yearly_mean_similarity(args):
    """Helper function for multiprocessing: Computes mean cosine similarity of a word to all others."""
    year, model_path, word, excluded_words = args
    try:
        model = W2VModel(model_path)

        # Ensure word exists in vocabulary
        if word not in model.vocab:
            raise ValueError(f"Word '{word}' not found in the model for year {year}.")

        # Filter out excluded words
        excluded_words = set(excluded_words)  # Ensure it's a set for fast lookup

        # Compute mean similarity excluding the unwanted words
        mean_similarity = model.mean_cosine_similarity_to_all(word, excluded_words)
        return (year, mean_similarity, 0)  # Standard deviation isn't applicable here

    except Exception as e:
        return (year, None, str(e))  # Return None and error message


def track_word_semantic_drift(
    word, start_year, end_year, model_dir,
    excluded_words=None, plot=True, smooth=False, sigma=2,
    confidence=0.95, error_type="CI", num_workers=None
):
    """
    Compute and track the yearly mean cosine similarity of a word to all other words.

    Args:
        word (str): The target word to track across years.
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        excluded_words (list or set): Words to exclude from similarity calculations.
        plot (bool or int): If `True`, plots without chunking. If an integer `N`, averages every `N` years.
        smooth (bool): Whether to apply smoothing.
        sigma (float): Standard deviation for Gaussian smoothing.
        confidence (float): Confidence level for error bands.
        error_type (str): Either "CI" (confidence intervals) or "SE" (standard error).
        num_workers (int or None): Number of parallel workers (default: max CPU cores).

    Returns:
        dict: A dictionary mapping years to (mean cosine similarity, error measure).
    """
    drift_scores = {}
    missing_years = []
    error_years = {}
    
    # Convert excluded_words to a set for quick lookup
    excluded_words = set(excluded_words) if excluded_words else set()

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

    print(f"Tracking semantic drift for word: '{word}' (Excluding: {len(excluded_words)} words)")

    # Prepare multiprocessing arguments
    args = [(year, path, word, excluded_words) for year, path in model_paths.items()]

    # Use multiprocessing to compute similarities in parallel
    num_workers = num_workers or min(cpu_count(), len(args))
    with Pool(num_workers) as pool:
        results = pool.map(compute_yearly_mean_similarity, args)

    # Process results
    for year, mean_similarity, std_dev in results:
        if mean_similarity is not None:
            drift_scores[year] = (mean_similarity, std_dev)
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

    # Apply Smoothing
    smoothed_values = gaussian_filter1d(similarities, sigma=sigma) if smooth else None

    # Handle Chunking
    if isinstance(plot, int) and plot > 1:
        chunk_size = plot
        chunked_years = []
        chunked_similarities = []

        for i in range(0, len(years), chunk_size):
            chunk = years[i:i + chunk_size]
            chunk_values = similarities[i:i + chunk_size]

            if len(chunk) > 0:
                chunked_years.append(np.mean(chunk))
                chunked_similarities.append(np.mean(chunk_values))

        years = np.array(chunked_years)
        similarities = np.array(chunked_similarities)

        if smooth:
            smoothed_values = gaussian_filter1d(similarities, sigma=sigma)

    # ✅ Plot Results
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(years, similarities, marker='o', linestyle='-', label=f"Semantic Drift of '{word}'", color='blue')

        if smooth and smoothed_values is not None:
            plt.plot(years, smoothed_values, linestyle='--', color='red', label='Smoothed Trend')

        plt.xlabel("Year")
        plt.ylabel("Mean Cosine Similarity")
        plt.title(f"Semantic Drift of '{word}' Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    return drift_scores