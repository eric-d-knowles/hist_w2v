import os
import glob
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

def cosine_similarity_over_years(word1, word2, start_year, end_year, model_dir, plot=False):
    """
    Compute the cosine similarity between two words across a range of yearly models.
    
    Args:
        word1 (str): The first word.
        word2 (str): The second word.
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        plot (bool): Whether to plot the similarity trend over time.
    
    Returns:
        dict: A dictionary mapping years to cosine similarity scores.
    """
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' does not exist. Please check the path.")
        return {}
    
    similarities = {}
    found_any = False
    missing_models = []
    missing_word1 = []
    missing_word2 = []
    missing_both = []
    
    for year in range(start_year, end_year + 1):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            missing_models.append(year)
            continue  # Skip missing models
        
        model_path = model_files[0]
        found_any = True
        
        try:
            yearly_model = KeyedVectors.load(model_path, mmap="r")
            
            has_word1 = word1 in yearly_model.key_to_index
            has_word2 = word2 in yearly_model.key_to_index
            
            if has_word1 and has_word2:
                sim = yearly_model.similarity(word1, word2)
                similarities[year] = sim
            elif not has_word1 and not has_word2:
                missing_both.append(year)
            elif not has_word1:
                missing_word1.append(year)
            elif not has_word2:
                missing_word2.append(year)
        except Exception as e:
            print(f"Skipping {year} due to error: {e}")
            continue
    
    if missing_models:
        print(f"Models missing for years: {', '.join(map(str, missing_models))}")
    if missing_word1:
        print(f"'{word1}' missing in models for years: {', '.join(map(str, missing_word1))}")
    if missing_word2:
        print(f"'{word2}' missing in models for years: {', '.join(map(str, missing_word2))}")
    if missing_both:
        print(f"Both words missing in models for years: {', '.join(map(str, missing_both))}")
    if not found_any:
        print("No models found in the specified range.")
    
    if plot and similarities:
        plt.figure(figsize=(10, 5))
        plt.plot(list(similarities.keys()), list(similarities.values()), marker='o', linestyle='-')
        plt.xlabel("Year")
        plt.ylabel("Cosine Similarity")
        plt.title(f"Cosine Similarity of '{word1}' and '{word2}' Over Time")
        plt.grid(True)
        plt.show()
    
    return similarities