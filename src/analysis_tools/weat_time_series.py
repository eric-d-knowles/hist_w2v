import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from common.w2v_model import W2VModel

def compute_weat_over_years(target1, target2, attribute1, attribute2, start_year, end_year, model_dir, num_permutations=10000, plot=True, smooth=False, sigma=2, confidence=0.95, return_std=True):
    """
    Compute the WEAT effect size, p-value, and error bands over a range of yearly models.

    Args:
        target1 (list of str): First set of category words.
        target2 (list of str): Second set of category words.
        attribute1 (list of str): First set of target words.
        attribute2 (list of str): Second set of target words.
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        num_permutations (int): Number of permutations for significance testing (0 to disable).
        plot (bool or int): If `True`, plots without chunking. If an integer `N`, averages every `N` years for plotting.
        smooth (bool): Whether to overlay a smoothing line over the graph.
        sigma (float): Standard deviation for Gaussian smoothing.
        confidence (float): Confidence level for error bands.
        return_std (bool): Whether to return the standard deviation for error bands.

    Returns:
        dict: A dictionary mapping years to (WEAT effect size, p-value, std_dev).
    """
    weat_scores = {}
    missing_years = []
    error_years = {}

    for year in range(start_year, end_year + 1):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = sorted(glob.glob(model_pattern))

        if not model_files:
            missing_years.append(year)
            continue
        
        model_path = model_files[-1]  # Pick the most recent model

        try:
            yearly_model = W2VModel(model_path)
            weat_result = yearly_model.compute_weat(
                target1, target2, attribute1, attribute2, num_permutations, return_std=return_std
            )
            if return_std:
                effect_size, p_value, std_dev = weat_result
            else:
                effect_size, p_value = weat_result
                std_dev = None
            
            weat_scores[year] = (effect_size, p_value, std_dev)

        except Exception as e:
            error_years[year] = str(e)
            continue
    
    if missing_years:
        print(f"⚠️ No models found for these years: {missing_years}")

    if error_years:
        print("❌ Errors occurred in the following years:")
        for year, err in error_years.items():
            print(f"  {year}: {err}")

    # Convert results to NumPy arrays for plotting
    if not weat_scores:
        print("❌ No valid WEAT scores computed. Exiting.")
        return {}

    years = np.array(sorted(weat_scores.keys()))
    effect_sizes = np.array([weat_scores[year][0] for year in years])

    # ✅ Handle standard deviations for confidence intervals
    if return_std:
        std_devs = np.array([
            weat_scores[year][2] if weat_scores[year][2] not in [None, 0] else np.nan 
            for year in years
        ])
        if np.all(np.isnan(std_devs)):
            print("⚠️ Warning: All standard deviations are NaN. Confidence intervals cannot be plotted.")
            ci_range = np.zeros_like(effect_sizes)  # No error bars
        else:
            ci_range = stats.norm.ppf(1 - (1 - confidence) / 2) * np.nan_to_num(std_devs, nan=np.nanmean(std_devs))
    else:
        ci_range = None  # No confidence intervals if return_std=False

    # ✅ Handle Chunking for Plotting
    chunk_size = plot if isinstance(plot, int) else 1  # Use `plot=N` as chunk size

    if chunk_size > 1:
        chunked_years = []
        chunked_effects = []
        chunked_stds = [] if return_std else None

        for i in range(0, len(years), chunk_size):
            chunk = years[i:i + chunk_size]
            chunk_mean = np.nanmean(effect_sizes[i:i + chunk_size])  # Ignore NaNs
            chunk_year = np.mean(chunk)  # Center of the chunk

            chunked_years.append(chunk_year)
            chunked_effects.append(chunk_mean)

            if return_std:
                chunk_stds = std_devs[i:i + chunk_size]
                chunked_stds.append(np.nanmean(chunk_stds))  # Use mean of std deviations
        
        years, effect_sizes = np.array(chunked_years), np.array(chunked_effects)
        if return_std:
            std_devs = np.array(chunked_stds)
            ci_range = stats.norm.ppf(1 - (1 - confidence) / 2) * std_devs

    # ✅ Apply Smoothing After Chunking
    smoothed_values = gaussian_filter1d(effect_sizes, sigma=sigma) if smooth else None

    # ✅ Plot Results
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(years, effect_sizes, marker='o', linestyle='-', label='WEAT Effect Size', color='blue')

        if smooth and smoothed_values is not None:
            plt.plot(years, smoothed_values, linestyle='--', color='red', label=f'Smoothed Trend')

        if return_std and ci_range is not None:
            plt.fill_between(years, effect_sizes - ci_range, effect_sizes + ci_range, color='blue', alpha=0.2, label=f"{int(confidence * 100)}% CI")

        plt.xlabel("Year")
        plt.ylabel("WEAT Effect Size (Cohen's d)")
        plt.title("WEAT Effect Size Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    return weat_scores