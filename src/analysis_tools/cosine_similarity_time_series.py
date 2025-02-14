import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from gensim.models import KeyedVectors

def cosine_similarity_over_years(word1, word2, start_year, end_year, model_dir, plot=True, smooth=False, sigma=2):
    """
    Compute the cosine similarity between two words across a range of yearly models.

    Args:
        word1 (str): The first word.
        word2 (str): The second word.
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        plot (bool or int): If `True`, plots yearly data. If an integer `N`, averages every `N` years for plotting.
        smooth (bool): Whether to overlay a smoothing line on the graph.
        sigma (float): Standard deviation for Gaussian smoothing (higher = smoother curve).

    Returns:
        dict: A dictionary mapping years to cosine similarity scores.
    """
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' does not exist. Please check the path.")
        return {}

    similarities = {}
    missing_models = []
    
    for year in range(start_year, end_year + 1):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            missing_models.append(year)
            continue  # Skip missing models
        
        model_path = model_files[0]  # Pick the first matching model
        
        try:
            yearly_model = KeyedVectors.load(model_path, mmap="r")
            
            if word1 in yearly_model.key_to_index and word2 in yearly_model.key_to_index:
                sim = yearly_model.similarity(word1, word2)
                similarities[year] = sim
        except Exception as e:
            print(f"Skipping {year} due to error: {e}")
            continue
    
    if not similarities:
        print("❌ No valid similarity scores computed. Exiting.")
        return {}

    # ✅ Create full range of years and fill missing values with NaN
    all_years = np.arange(start_year, end_year + 1)
    similarity_values = np.array([similarities.get(year, np.nan) for year in all_years])

    # ✅ Interpolate missing values
    mask = ~np.isnan(similarity_values)
    if mask.any():
        similarity_values = np.interp(all_years, all_years[mask], similarity_values[mask])

    # ✅ Set `chunk_size` Based on `plot`
    chunk_size = plot if isinstance(plot, int) else 1  # If `plot=N`, set `chunk_size=N`
    
    # ✅ Apply Chunking (Averaging Consecutive Years)
    if chunk_size > 1:
        chunked_years = []
        chunked_similarities = []

        for i in range(0, len(all_years), chunk_size):
            chunk = all_years[i:i + chunk_size]
            chunk_values = similarity_values[i:i + chunk_size]

            if np.isnan(chunk_values).all():
                continue  # Skip chunks with only missing values

            chunk_mean = np.nanmean(chunk_values)  # Ignore NaNs in averaging
            chunk_year = np.mean(chunk)  # Center of the chunk

            chunked_years.append(chunk_year)
            chunked_similarities.append(chunk_mean)

        # ✅ Replace full-year data with chunked data
        all_years, similarity_values = np.array(chunked_years), np.array(chunked_similarities)

    # ✅ Apply Smoothing (After Chunking)
    smoothed_values = gaussian_filter1d(similarity_values, sigma=sigma) if smooth else None

    # ✅ Plot the Results
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(all_years, similarity_values, marker='o', linestyle='-', label=f"Similarity ({word1}, {word2})", color='blue')

        # ✅ Overlay Smoothing Line (If Enabled)
        if smooth and smoothed_values is not None:
            plt.plot(all_years, smoothed_values, linestyle='--', color='red', label=f'Smoothed (σ={sigma})')

        plt.xlabel("Year")
        plt.ylabel("Cosine Similarity")
        plt.title(f"Cosine Similarity of '{word1}' and '{word2}' Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    return similarities