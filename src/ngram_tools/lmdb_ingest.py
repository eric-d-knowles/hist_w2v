"""
LMDB Ngram Ingestion Module.

This module ingests filtered ngram files into an LMDB database for efficient
storage and retrieval. It supports both plain JSONL and LZ4-compressed JSONL
files, and provides progress bars for file processing.
"""

import os
from glob import glob

import lmdb
import lz4.frame
import orjson
from tqdm import tqdm


def get_filtered_files(filtered_dir):
    """
    Find all .jsonl and .jsonl.lz4 files in the filtered directory.

    Args:
        filtered_dir (str): Directory containing filtered ngram files.

    Returns:
        list: List of file paths matching the pattern.
    """
    return glob(os.path.join(filtered_dir, '*.jsonl*'))


def ngram_key(entry):
    """
    Generate a normalized string key for each ngram entry.

    Assumes ngram is a dictionary with token1, token2, ..., token5 fields.

    Args:
        entry (dict): Dictionary containing an 'ngram' field.

    Returns:
        str: Normalized string key for the ngram.
    """
    ngram = entry.get('ngram')
    tokens = [
        str(ngram.get(f'token{i+1}', '')).lower()
        for i in range(5)
    ]
    return ' '.join(tokens)


def ingest_to_lmdb(filtered_dir, lmdb_path, map_size=600 * 1024 ** 3):
    """
    Ingest filtered ngram files into LMDB database.

    Args:
        filtered_dir (str): Directory containing filtered ngram files.
        lmdb_path (str): Path to LMDB database directory.
        map_size (int): Maximum size of LMDB database in bytes.
                       Defaults to 600 GB.
    """
    # Ensure LMDB directory exists
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    # Get list of filtered ngram files (.jsonl or .jsonl.lz4)
    files = [
        os.path.join(filtered_dir, f)
        for f in os.listdir(filtered_dir)
        if f.endswith('.jsonl') or f.endswith('.jsonl.lz4')
    ]

    # Open LMDB environment
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    try:
        with env.begin(write=True) as txn:
            # Iterate over each file with a progress bar
            for file_path in tqdm(files, desc='Processing files', unit='file'):
                _process_file(file_path, txn)
    finally:
        # Ensure all data is written to disk
        env.sync()
        env.close()


def _process_file(file_path, txn):
    """
    Process a single ngram file and insert entries into LMDB transaction.

    Args:
        file_path (str): Path to the ngram file to process.
        txn: LMDB write transaction.
    """
    # Select appropriate open function for compressed or plain files
    if file_path.endswith('.lz4'):
        open_func = lambda f: lz4.frame.open(f, mode='rt')
    else:
        open_func = lambda f: open(f, 'r')
    
    # Process each line in the file
    with open_func(file_path) as f:
        for line in f:
            # Parse JSON line
            try:
                obj = orjson.loads(line)
            except Exception:
                continue  # Skip malformed lines
            
            # Use ngram field as key
            key = ngram_key(obj).encode('utf-8')
            value = orjson.dumps(obj)
            
            # Insert into LMDB
            txn.put(key, value)


def _process_file(file_path, txn):
    """
    Process a single ngram file and insert entries into LMDB transaction.

    Args:
        file_path (str): Path to the ngram file to process.
        txn: LMDB write transaction.
    """
    # Select appropriate open function for compressed or plain files
    if file_path.endswith('.lz4'):
        open_func = lambda f: lz4.frame.open(f, mode='rt')
    else:
        open_func = lambda f: open(f, 'r')
    
    # Process each line in the file
    with open_func(file_path) as f:
        for line in f:
            # Parse JSON line
            try:
                obj = orjson.loads(line)
            except Exception:
                continue  # Skip malformed lines
            
            # Use ngram field as key
            key = ngram_key(obj).encode('utf-8')
            value = orjson.dumps(obj)
            
            # Insert into LMDB
            txn.put(key, value)
