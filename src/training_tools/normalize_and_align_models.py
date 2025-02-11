import argparse
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

from tqdm.notebook import tqdm
from training_tools.w2v_model import W2VModel


def get_model_paths(model_dir):
    """
    Retrieve paths of all Word2Vec model files in the specified directory.
    Extracts the year robustly using regex.
    """
    model_paths = []
    pattern = re.compile(r'w2v_y(\d{4})')
    
    for f in Path(model_dir).glob("w2v_y*.kv"):
        match = pattern.search(f.name)
        if match:
            year = int(match.group(1))
            model_paths.append((year, str(f)))
        else:
            print(f"Skipping file with unexpected format: {f.name}")
    
    return sorted(model_paths)


def process_model(args):
    """
    Normalize and align a given model to the anchor model.
    """
    year, model_path, anchor_model, dir_suffix = args
    model = W2VModel(model_path)
    
    # Ensure vectors are writeable before normalization
    model.model.vectors = model.model.vectors.copy()
    model = model.normalize()
    
    if year != anchor_model[0]:
        model.filter_vocab(anchor_model[1].filtered_vocab)
        model.align_to(anchor_model[1])
    
    output_path = model_path.replace(f"models_{dir_suffix}",
                                     f"models_{dir_suffix}/norm_and_align")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def normalize_and_align_vectors(ngram_size, proj_dir, dir_suffix, anchor_year,
                                workers=os.cpu_count()):
    """
    Normalize and align Word2Vec models in the given project directory.
    """
    start_time = datetime.now()
    model_dir = os.path.join(
        proj_dir, f'{ngram_size}gram_files/6corpus/yearly_files/models_{dir_suffix}'
    )
    
    model_paths = get_model_paths(model_dir)
    if not model_paths:
        raise FileNotFoundError("No .kv models found in the specified directory.")
    
    # Load the anchor model
    anchor_model_path = next((p for y, p in model_paths if y == anchor_year), None)
    if not anchor_model_path:
        raise ValueError(f"Anchor model for year {anchor_year} not found.")
    
    anchor_model = W2VModel(anchor_model_path)
    anchor_model.model.vectors = anchor_model.model.vectors.copy()
    anchor_model = anchor_model.normalize()
    
    # Ensure anchor model has filtered_vocab before multiprocessing
    anchor_model.filter_vocab(anchor_model.extract_vocab())
    
    # Save the anchor model in the output directory
    output_anchor_path = anchor_model_path.replace(f"models_{dir_suffix}", f"models_{dir_suffix}/norm_and_align")
    Path(output_anchor_path).parent.mkdir(parents=True, exist_ok=True)
    anchor_model.save(output_anchor_path)  # Save normalized anchor model
    
    print(f"Saved normalized anchor model to {output_anchor_path}")

    # Prepare non-anchor models for multiprocessing
    tasks = [(y, p, (anchor_year, anchor_model), dir_suffix) for y, p in model_paths if y != anchor_year]
    
    with Pool(processes=workers) as pool:
        with tqdm(
            total=len(tasks), desc="Processing models", unit="file", file=sys.stdout
        ) as pbar:
            for _ in pool.imap_unordered(process_model, tasks):
                pbar.update()
    
    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")
    print(f"Processed {len(model_paths)} models. Aligned to anchor year {anchor_year}.")