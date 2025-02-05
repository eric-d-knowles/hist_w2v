import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from training_tools.w2v_model import W2VModel

def compute_weat_over_years(category1, category2, target1, target2, start_year, end_year, model_dir, plot=False, smooth=False, sigma=2):
    """
    Compute the WEAT effect size over a range of yearly models.
    
    Args:
        category1 (list of str): First set of category words.
        category2 (list of str): Second set of category words.
        target1 (list of str): First set of target words.
        target2 (list of str): Second set of target words.
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        plot (bool): Whether to plot the WEAT effect size trend over time.
        smooth (bool): Whether to overlay a smoothing line over the graph.
        sigma (float): Standard deviation for Gaussian smoothing (higher values = smoother curve).
    
    Returns:
        dict: A dictionary mapping years to WEAT effect size (Cohen's d).
    """
    weat_scores = {}
    
    for year in range(start_year, end_year + 1):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            continue  # Skip missing models
        
        model_path = model_files[0]
        
        try:
            yearly_model = W2VModel(model_path)
            weat_scores[year] = yearly_model.compute_weat(category1, category2, target1, target2)
        except Exception as e:
            print(f"Skipping {year} due to error: {e}")
            continue
    
    if plot and weat_scores:
        years = np.array(list(weat_scores.keys()))
        scores = np.array(list(weat_scores.values()))
        
        plt.figure(figsize=(10, 5))
        plt.plot(years, scores, marker='o', linestyle='-', label='WEAT Effect Size')
        
        if smooth and len(years) > 1:
            smoothed_scores = gaussian_filter1d(scores, sigma=sigma)
            plt.plot(years, smoothed_scores, linestyle='--', color='red', label='Smoothed Trend')
        
        plt.xlabel("Year")
        plt.ylabel("WEAT Effect Size (Cohen's d)")
        plt.title("WEAT Effect Size Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return weat_scores