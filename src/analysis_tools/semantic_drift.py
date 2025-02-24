import os
from tqdm.notebook import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from common.w2v_model import W2VModel
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
import pandas as pd

def compute_similarity_to_previous_year(args):
    year, prev_model_path, model_path, word = args
    
    prev_model = W2VModel(prev_model_path)
    model = W2VModel(model_path)

    similarity_mean, similarity_sd, common_words = model.compare_models_cosim(prev_model, word)

    return year, similarity_mean, similarity_sd, common_words

def track_yearly_drift(
    start_year, end_year, model_dir, word=None, plot=True, smooth=False, sigma=2, 
    confidence=0.95, error_type="CI", num_workers=None, df=None, regress_on=None
):
    drift_data = {}
    missing_years = []
    error_years = {}

    model_paths = {}
    for year in range(start_year, end_year + 1):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = sorted(glob.glob(model_pattern))
        if model_files:
            model_paths[year] = model_files[-1]
        else:
            missing_years.append(year)

    if not model_paths or len(model_paths) < 2:
        print("❌ Not enough valid models found for year-over-year analysis. Exiting.")
        return {}

    years_available = sorted(model_paths.keys())
    args = [(years_available[i], model_paths[years_available[i-1]], model_paths[years_available[i]], word)
            for i in range(1, len(years_available))]
    
    num_workers = num_workers or min(cpu_count(), len(args))
    with Pool(num_workers) as pool:
        results = pool.map(compute_similarity_to_previous_year, args)

    for result in results:
        year, similarity_mean, similarity_sd, shared_vocab_size = result
        
        if similarity_mean:
            change_mean = 1 - similarity_mean
    
            if error_type == "CI":
                error_measure = stats.norm.ppf(1 - (1 - confidence) / 2) * (similarity_sd / np.sqrt(shared_vocab_size))
            elif error_type == "SE":
                error_measure = change_mean / np.sqrt(shared_vocab_size) if shared_vocab_size > 1 else 0
            else:
                raise ValueError("Invalid error_type. Choose 'CI' or 'SE'.")
            
            drift_data[year] = (change_mean, error_measure, shared_vocab_size)

    if missing_years:
        print(f"⚠️ No models found for these years: {missing_years}")
        
    if error_years:
        print("❌ Errors occurred in the following years:")
        for year, err in error_years.items():
            print(f"  {year}: {err}")

    if not drift_data:
        print("❌ No valid drift scores computed. Exiting.")
        return {}

    df_drift = pd.DataFrame.from_dict(drift_data, orient="index", columns=["Drift", "Error", "Shared"])
    df_drift.index.name = "Year"

    adjusted = False
    if df is not None and regress_on is not None:
        if regress_on not in df.columns:
            print(f"⚠️ Regressor '{regress_on}' not found in the provided DataFrame. Proceeding without adjustment.")
        else:
            df_drift = df_drift.merge(df[[regress_on]], left_index=True, right_index=True, how="left").dropna()
            X = df_drift[[regress_on]].values.reshape(-1, 1)
            y = df_drift["Drift"].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            df_drift["Drift_Adjusted"] = y - y_pred  # Residuals
            adjusted = True

    # Function to plot drift data
    def plot_drift(ax, years, drift, errors, label, title):
        ax.scatter(years, drift, color='blue', alpha=0.2, label=label)
        ax.errorbar(years, drift, yerr=errors, fmt='o', color='blue', alpha=0.3, label="Error bars")

        if smooth:
            smoothed = gaussian_filter1d(drift, sigma=sigma)
            ax.plot(years, smoothed, linestyle='--', color='red', linewidth=2, label=f"Smoothed (σ={sigma})")
            derivative = savgol_filter(smoothed, window_length=11, polyorder=3, deriv=1, delta=np.mean(np.diff(years)))
            
            ax2 = ax.twinx()
            ax2.plot(years, derivative, linestyle='-', color='green', linewidth=1, label="First Derivative")
            ax2.set_ylabel("Rate of Change")
            ax2.set_ylim(-0.005, 0.003)

            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")

        ax.set_xlabel("Year")
        ax.set_ylabel("Change Magnitude")
        ax.set_title(title)
        ax.grid(True)

    # Plot Unadjusted Scores
    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        plot_drift(ax1, df_drift.index, df_drift["Drift"], df_drift["Error"], "Unadjusted Drift", 
                   f"Year-over-Year Semantic Change {'for ' + word if word else ''}")
        plt.tight_layout()
        plt.show()

    # Plot Adjusted Scores if Regression was Performed
    if adjusted:
        fig, ax2 = plt.subplots(figsize=(10, 5))
        plot_drift(ax2, df_drift.index, df_drift["Drift_Adjusted"], df_drift["Error"], "Adjusted Drift (Residuals)", 
                   f"Adjusted Year-over-Year Semantic Change {'for ' + word if word else ''}")
        plt.tight_layout()
        plt.show()

    return df_drift