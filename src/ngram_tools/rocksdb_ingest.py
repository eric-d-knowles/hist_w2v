"""
RocksDB ingestion module for efficient ngram storage and processing.

This module can replace the entire preprocessing pipeline by ingesting
downloaded ngrams directly into RocksDB with streaming transformations.
Supports both direct ingestion from downloads and filtered file ingestion.
"""

import argparse
import gzip
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import nltk
import orjson
import rocksdb
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm

from ngram_tools.helpers.file_handler import FileHandler

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


def ngram_key(ngram_text: str) -> bytes:
    """Generate RocksDB key from ngram text."""
    return ngram_text.encode('utf-8')


def serialize_ngram_value(data: dict) -> bytes:
    """Serialize ngram data for RocksDB storage."""
    return orjson.dumps({
        'freq_tot': data['freq_tot'],
        'doc_tot': data['doc_tot'],
        'freq': data['freq'],
        'doc': data['doc']
    })


def deserialize_ngram_value(value_bytes: bytes) -> dict:
    """Deserialize ngram data from RocksDB storage."""
    return orjson.loads(value_bytes)


# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

# POS tag mapping from Google to WordNet
google_to_wordnet = {
    "NOUN": "n", "PROPN": "n",
    "VERB": "v",
    "ADJ": "a", 
    "ADV": "r"
}

# Regex patterns for filtering
NUMERALS_REGEX = re.compile(r'\d')
NONALPHA_REGEX = re.compile(r'[^a-zA-Z]')


def process_raw_ngram_line(
    line: str, 
    ngram_type: str,
    apply_lowercase: bool = True,
    apply_lemmatize: bool = True,
    apply_filters: dict = None,
    vocab_set: set = None,
    min_tokens: int = 1
) -> dict:
    """
    Process a raw ngram line with all transformations applied in-stream.
    
    This replaces convert_to_jsonl + lowercase + lemmatize + filter steps
    by applying all transformations in a single pass.
    
    Args:
        line: Raw ngram line from downloaded file
        ngram_type: 'tagged' or 'untagged'
        apply_lowercase: Whether to lowercase tokens
        apply_lemmatize: Whether to lemmatize tokens (requires tagged)
        apply_filters: Dict of filter options
        vocab_set: Set of allowed vocabulary
        min_tokens: Minimum tokens required after filtering
        
    Returns:
        Processed ngram dict or None if filtered out
    """
    if apply_filters is None:
        apply_filters = {}
    
    try:
        # Parse the raw line (same as convert_to_jsonl)
        tokens_part, *year_data_parts = line.strip().split('\t')
        tokens = tokens_part.split()
        
        # Process tokens with transformations
        processed_tokens = {}
        valid_token_count = 0
        
        for i, token in enumerate(tokens):
            token_key = f"token{i+1}"
            
            # Extract base token and POS tag if tagged
            if ngram_type == "tagged" and "_" in token:
                base_token, pos_tag = token.rsplit("_", 1)
            else:
                base_token = token
                pos_tag = None
            
            # Apply lowercase
            if apply_lowercase:
                base_token = base_token.lower()
            
            # Apply lemmatization if requested and POS tag available
            if apply_lemmatize and pos_tag and pos_tag in google_to_wordnet:
                try:
                    base_token = lemmatizer.lemmatize(
                        base_token, google_to_wordnet[pos_tag]
                    )
                except:
                    pass  # Keep original if lemmatization fails
            
            # Apply filters
            keep_token = True
            
            if apply_filters.get('numerals', False) and NUMERALS_REGEX.search(base_token):
                keep_token = False
            elif apply_filters.get('nonalpha', False) and NONALPHA_REGEX.search(base_token):
                keep_token = False
            elif apply_filters.get('stops', False) and base_token in english_stopwords:
                keep_token = False
            elif apply_filters.get('min_token_length', 0) > 0:
                if len(base_token) < apply_filters['min_token_length']:
                    keep_token = False
            elif vocab_set and base_token not in vocab_set:
                keep_token = False
            
            if keep_token:
                processed_tokens[token_key] = base_token
                valid_token_count += 1
            elif apply_filters.get('replace_unk', False):
                processed_tokens[token_key] = 'UNK'
                # Don't count UNK toward valid token count
        
        # Check minimum token requirement
        if valid_token_count < min_tokens:
            return None
        
        # Parse frequency and document data
        freq_tot = 0
        doc_tot = 0
        freq = {}
        doc = {}
        
        for year_data in year_data_parts:
            year, freq_val, doc_val = year_data.split(',')
            freq_val = int(freq_val)
            doc_val = int(doc_val)
            
            freq[year] = freq_val
            doc[year] = doc_val
            freq_tot += freq_val
            doc_tot += doc_val
        
        # Create final ngram string for key
        ngram_str = ' '.join(processed_tokens.values())
        
        return {
            'ngram': ngram_str,
            'freq_tot': freq_tot,
            'doc_tot': doc_tot,
            'freq': freq,
            'doc': doc
        }
        
    except Exception as e:
        print(f"Error processing line: {line[:100]}... Error: {e}")
        return None


def load_vocabulary_set(vocab_file_path: str) -> set:
    """Load vocabulary from file into a set for fast lookup."""
    if not vocab_file_path or not os.path.exists(vocab_file_path):
        return set()
    
    vocab_set = set()
    with open(vocab_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab_set.add(line.strip().lower())
    
    return vocab_set


def merge_ngram_data(existing_data: dict, new_data: dict) -> dict:
    """Merge frequency and document data for duplicate ngrams."""
    merged = {
        'freq_tot': existing_data['freq_tot'] + new_data['freq_tot'],
        'doc_tot': existing_data['doc_tot'] + new_data['doc_tot'],
        'freq': existing_data['freq'].copy(),
        'doc': existing_data['doc'].copy()
    }
    
    # Merge yearly frequency data
    for year, freq in new_data['freq'].items():
        merged['freq'][year] = merged['freq'].get(year, 0) + freq
    
    # Merge yearly document data
    for year, doc in new_data['doc'].items():
        merged['doc'][year] = merged['doc'].get(year, 0) + doc
    
    return merged


def setup_rocksdb(db_path: str, optimize_for_writes: bool = True) -> rocksdb.DB:
    """
    Set up RocksDB with optimized options for ngram ingestion.
    
    Args:
        db_path: Path to the RocksDB database
        optimize_for_writes: Whether to optimize for write performance
        
    Returns:
        RocksDB database instance
    """
    opts = rocksdb.Options()
    
    if optimize_for_writes:
        # Optimize for bulk writes
        opts.create_if_missing = True
        opts.write_buffer_size = 256 * 1024 * 1024  # 256MB
        opts.max_write_buffer_number = 4
        opts.target_file_size_base = 128 * 1024 * 1024  # 128MB
        opts.max_background_jobs = 8
        opts.level0_file_num_compaction_trigger = 8
        opts.level0_slowdown_writes_trigger = 20
        opts.level0_stop_writes_trigger = 36
        opts.max_bytes_for_level_base = 1024 * 1024 * 1024  # 1GB
        opts.compression = rocksdb.CompressionType.lz4_compression
        
        # Disable WAL for faster writes during bulk loading
        opts.use_fsync = False
        
    else:
        opts.create_if_missing = True
        
    return rocksdb.DB(db_path, opts)


def process_file_to_rocksdb(args):
    """
    Process a single filtered ngram file and ingest into RocksDB.
    
    Args:
        args: Tuple of (file_path, db_path, batch_size)
        
    Returns:
        Number of ngrams processed
    """
    file_path, db_path, batch_size = args
    
    # Open database connection for this worker
    db = setup_rocksdb(db_path)
    
    input_handler = FileHandler(file_path)
    batch = rocksdb.WriteBatch()
    batch_count = 0
    total_processed = 0
    
    try:
        with input_handler.open() as f:
            for line in f:
                try:
                    entry = input_handler.deserialize(line)
                    ngram = entry['ngram']
                    key = ngram_key(ngram)
                    
                    # Check if ngram already exists
                    existing_value = db.get(key)
                    if existing_value:
                        # Merge with existing data
                        existing_data = deserialize_ngram_value(existing_value)
                        merged_data = merge_ngram_data(existing_data, entry)
                        new_value = serialize_ngram_value(merged_data)
                    else:
                        # New ngram
                        new_value = serialize_ngram_value(entry)
                    
                    batch.put(key, new_value)
                    batch_count += 1
                    total_processed += 1
                    
                    # Write batch when it reaches batch_size
                    if batch_count >= batch_size:
                        db.write(batch)
                        batch = rocksdb.WriteBatch()
                        batch_count = 0
                        
                except Exception as e:
                    print(f"Error processing line in {file_path}: {e}")
                    continue
        
        # Write any remaining items in the batch
        if batch_count > 0:
            db.write(batch)
            
    finally:
        db.close()
    
    return total_processed


def ingest_downloaded_files_to_rocksdb(
    ngram_size: int,
    proj_dir: str,
    ngram_type: str = 'tagged',
    db_name: str = "ngrams.db",
    workers: int = None,
    batch_size: int = 1000,
    overwrite: bool = False,
    file_range: tuple = None,
    # Processing options
    apply_lowercase: bool = True,
    apply_lemmatize: bool = True,
    apply_filters: dict = None,
    vocab_file: str = None,
    min_tokens: int = 2
) -> str:
    """
    Ingest downloaded ngram files directly into RocksDB with all processing.
    
    This replaces the entire pipeline:
    download → convert → lowercase → lemmatize → filter → sort → consolidate
    
    With a single step:
    download → RocksDB (with streaming transformations)
    
    Args:
        ngram_size: Size of ngrams (1-5)
        proj_dir: Project directory containing downloaded files
        ngram_type: 'tagged' or 'untagged'
        db_name: Name of the RocksDB database
        workers: Number of parallel workers
        batch_size: Number of operations per batch
        overwrite: Whether to overwrite existing database
        file_range: Tuple of (start_idx, end_idx) for file range
        apply_lowercase: Whether to lowercase tokens
        apply_lemmatize: Whether to lemmatize tokens
        apply_filters: Dict with filter options (numerals, nonalpha, stops, min_token_length)
        vocab_file: Path to vocabulary file for filtering
        min_tokens: Minimum tokens required after filtering
        
    Returns:
        Path to the created database
    """
    if workers is None:
        workers = min(os.cpu_count(), 8)
    
    if apply_filters is None:
        apply_filters = {
            'numerals': True,
            'nonalpha': True, 
            'stops': True,
            'min_token_length': 3,
            'replace_unk': False
        }
    
    # Set up paths
    download_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/1download')
    corpus_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/6corpus')
    os.makedirs(corpus_dir, exist_ok=True)
    
    db_path = os.path.join(corpus_dir, db_name)
    
    # Remove existing database if overwrite is True
    if overwrite and os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
    
    # Get downloaded files
    if not os.path.exists(download_dir):
        raise FileNotFoundError(f"Download directory not found: {download_dir}")
    
    downloaded_files = sorted([
        os.path.join(download_dir, f) 
        for f in os.listdir(download_dir) 
        if f.endswith('.gz') or f.endswith('.txt') or f.endswith('.lz4')
    ])
    
    if not downloaded_files:
        raise FileNotFoundError(f"No downloaded files found in {download_dir}")
    
    # Apply file range if specified
    if file_range:
        start_idx, end_idx = file_range
        downloaded_files = downloaded_files[start_idx:end_idx + 1]
    
    # Load vocabulary if specified
    vocab_set = None
    if vocab_file:
        vocab_path = os.path.join(proj_dir, vocab_file)
        vocab_set = load_vocabulary_set(vocab_path)
        print(f"Loaded vocabulary with {len(vocab_set)} terms")
    
    print(f"Processing {len(downloaded_files)} downloaded files into RocksDB")
    print(f"Database path: {db_path}")
    print(f"Workers: {workers}")
    print(f"Transformations: lowercase={apply_lowercase}, lemmatize={apply_lemmatize}")
    print(f"Filters: {apply_filters}")
    
    start_time = datetime.now()
    
    # Prepare arguments for parallel processing
    args_list = [
        (file_path, db_path, batch_size, ngram_type, apply_lowercase, 
         apply_lemmatize, apply_filters, vocab_set, min_tokens)
        for file_path in downloaded_files
    ]
    
    # Process files in parallel
    total_ngrams = 0
    with Pool(processes=workers) as pool:
        with tqdm(total=len(downloaded_files), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_downloaded_file_to_rocksdb, args_list):
                total_ngrams += result
                pbar.update(1)
    
    # Compact database after bulk loading
    print("Compacting database...")
    db = setup_rocksdb(db_path, optimize_for_writes=False)
    db.compact_range()
    db.close()
    
    end_time = datetime.now()
    runtime = end_time - start_time
    
    print(f"\nIngestion complete!")
    print(f"Total ngrams processed: {total_ngrams:,}")
    print(f"Runtime: {runtime}")
    print(f"Database path: {db_path}")
    
    return db_path


def process_downloaded_file_to_rocksdb(args):
    """
    Process a single downloaded ngram file with full transformation pipeline.
    
    Args:
        args: Tuple of (file_path, db_path, batch_size, ngram_type, 
                       apply_lowercase, apply_lemmatize, apply_filters, 
                       vocab_set, min_tokens)
        
    Returns:
        Number of ngrams processed
    """
    (file_path, db_path, batch_size, ngram_type, apply_lowercase, 
     apply_lemmatize, apply_filters, vocab_set, min_tokens) = args
    
    # Open database connection for this worker
    db = setup_rocksdb(db_path)
    
    batch = rocksdb.WriteBatch()
    batch_count = 0
    total_processed = 0
    
    try:
        # Handle different file formats
        if file_path.endswith('.gz'):
            file_opener = gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            input_handler = FileHandler(file_path)
            file_opener = input_handler.open()
        
        with file_opener as f:
            for line in f:
                try:
                    # Apply full transformation pipeline to raw line
                    processed_data = process_raw_ngram_line(
                        line, ngram_type, apply_lowercase, apply_lemmatize,
                        apply_filters, vocab_set, min_tokens
                    )
                    
                    if processed_data is None:
                        continue  # Filtered out
                    
                    ngram = processed_data['ngram']
                    key = ngram_key(ngram)
                    
                    # Check if ngram already exists
                    existing_value = db.get(key)
                    if existing_value:
                        # Merge with existing data
                        existing_data = deserialize_ngram_value(existing_value)
                        merged_data = merge_ngram_data(existing_data, processed_data)
                        new_value = serialize_ngram_value(merged_data)
                    else:
                        # New ngram
                        new_value = serialize_ngram_value(processed_data)
                    
                    batch.put(key, new_value)
                    batch_count += 1
                    total_processed += 1
                    
                    # Write batch when it reaches batch_size
                    if batch_count >= batch_size:
                        db.write(batch)
                        batch = rocksdb.WriteBatch()
                        batch_count = 0
                        
                except Exception as e:
                    print(f"Error processing line in {file_path}: {e}")
                    continue
        
        # Write any remaining items in the batch
        if batch_count > 0:
            db.write(batch)
            
    finally:
        db.close()
    
    return total_processed
def ingest_filtered_files_to_rocksdb(
    ngram_size: int,
    proj_dir: str,
    db_name: str = "ngrams.db",
    workers: int = None,
    batch_size: int = 1000,
    overwrite: bool = False,
    file_range: tuple = None
) -> str:
    """
    Ingest pre-filtered ngram files into RocksDB.
    
    Use this if you've already run the traditional pipeline and want to 
    convert the filtered files to RocksDB format.
    
    Args:
        ngram_size: Size of ngrams (1-5)
        proj_dir: Project directory containing ngram files
        db_name: Name of the RocksDB database
        workers: Number of parallel workers
        batch_size: Number of operations per batch
        overwrite: Whether to overwrite existing database
        file_range: Tuple of (start_idx, end_idx) for file range
        
    Returns:
        Path to the created database
    """
    if workers is None:
        workers = min(os.cpu_count(), 8)  # Limit to prevent too many DB connections
    
    # Set up paths
    filter_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/5filter')
    corpus_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/6corpus')
    os.makedirs(corpus_dir, exist_ok=True)
    
    db_path = os.path.join(corpus_dir, db_name)
    
    # Remove existing database if overwrite is True
    if overwrite and os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
    
    # Get filtered files
    if not os.path.exists(filter_dir):
        raise FileNotFoundError(f"Filter directory not found: {filter_dir}")
    
    filtered_files = sorted([
        os.path.join(filter_dir, f) 
        for f in os.listdir(filter_dir) 
        if f.endswith('.jsonl') or f.endswith('.jsonl.lz4')
    ])
    
    if not filtered_files:
        raise FileNotFoundError(f"No filtered files found in {filter_dir}")
    
    # Apply file range if specified
    if file_range:
        start_idx, end_idx = file_range
        filtered_files = filtered_files[start_idx:end_idx + 1]
    
    print(f"Processing {len(filtered_files)} filtered files into RocksDB")
    print(f"Database path: {db_path}")
    print(f"Workers: {workers}")
    print(f"Batch size: {batch_size}")
    
    start_time = datetime.now()
    
    # Prepare arguments for parallel processing
    args_list = [(file_path, db_path, batch_size) for file_path in filtered_files]
    
    # Process files in parallel
    total_ngrams = 0
    with Pool(processes=workers) as pool:
        with tqdm(total=len(filtered_files), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_file_to_rocksdb, args_list):
                total_ngrams += result
                pbar.update(1)
    
    # Compact database after bulk loading
    print("Compacting database...")
    db = setup_rocksdb(db_path, optimize_for_writes=False)
    db.compact_range()
    db.close()
    
    end_time = datetime.now()
    runtime = end_time - start_time
    
    print(f"\nIngestion complete!")
    print(f"Total ngrams processed: {total_ngrams:,}")
    print(f"Runtime: {runtime}")
    print(f"Database path: {db_path}")
    
    return db_path


def export_rocksdb_to_yearly_files(
    db_path: str,
    yearly_dir: str,
    compress: bool = False,
    workers: int = None,
    chunk_size: int = 1000
):
    """
    Export RocksDB data to yearly files for word2vec training.
    
    This replaces the make_yearly_files functionality by reading from RocksDB
    instead of a consolidated JSONL file.
    
    Args:
        db_path: Path to the RocksDB database
        yearly_dir: Directory to create yearly files
        compress: Whether to compress yearly files
        workers: Number of parallel workers for file writing
        chunk_size: Number of entries to process per chunk
    """
    from collections import defaultdict
    import uuid
    import shutil
    from multiprocessing import Pool
    
    if workers is None:
        workers = os.cpu_count()
    
    os.makedirs(yearly_dir, exist_ok=True)
    
    print(f"Exporting RocksDB data to yearly files: {yearly_dir}")
    print(f"Workers: {workers}, Chunk size: {chunk_size}")
    
    start_time = datetime.now()
    
    # Create temporary directory for chunk processing
    temp_dir = os.path.join(yearly_dir, f"temp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Phase 1: Read RocksDB and create temporary yearly chunks
        db = setup_rocksdb(db_path, optimize_for_writes=False)
        
        # Iterate through all entries in RocksDB
        year_chunks = defaultdict(list)
        chunk_counter = 0
        total_entries = 0
        
        print("Phase 1: Reading RocksDB and creating yearly chunks...")
        
        # Create iterator for RocksDB
        iterator = db.iterkeys()
        iterator.seek_to_first()
        
        with Pool(processes=workers) as pool:
            chunk_tasks = []
            
            while iterator.valid():
                # Get key and value
                key = iterator.key()
                value_bytes = db.get(key)
                
                if value_bytes:
                    try:
                        # Deserialize the ngram data
                        ngram_data = deserialize_ngram_value(value_bytes)
                        ngram_text = key.decode('utf-8')
                        
                        # Process each year in the ngram data
                        for year, freq in ngram_data['freq'].items():
                            year_entry = {
                                'ngram': ngram_text,
                                'freq': freq,
                                'doc': ngram_data['doc'].get(year, 0)
                            }
                            year_chunks[year].append(year_entry)
                        
                        total_entries += 1
                        
                        # Process chunks when they reach chunk_size
                        if total_entries % chunk_size == 0:
                            # Submit chunk processing tasks
                            for year, entries in year_chunks.items():
                                if len(entries) >= chunk_size:
                                    chunk_file = os.path.join(
                                        temp_dir, 
                                        f"{year}_chunk_{chunk_counter}.jsonl" + ('.lz4' if compress else '')
                                    )
                                    chunk_tasks.append(
                                        pool.apply_async(
                                            write_yearly_chunk, 
                                            (entries, chunk_file, compress)
                                        )
                                    )
                                    year_chunks[year] = []
                                    chunk_counter += 1
                            
                            if total_entries % (chunk_size * 10) == 0:
                                print(f"Processed {total_entries:,} entries...")
                        
                    except Exception as e:
                        print(f"Error processing entry: {e}")
                
                iterator.next()
            
            # Process remaining chunks
            for year, entries in year_chunks.items():
                if entries:
                    chunk_file = os.path.join(
                        temp_dir, 
                        f"{year}_chunk_{chunk_counter}.jsonl" + ('.lz4' if compress else '')
                    )
                    chunk_tasks.append(
                        pool.apply_async(
                            write_yearly_chunk, 
                            (entries, chunk_file, compress)
                        )
                    )
                    chunk_counter += 1
            
            # Wait for all chunk tasks to complete
            for task in chunk_tasks:
                task.get()
        
        db.close()
        print(f"Phase 1 complete: {total_entries:,} entries processed into {chunk_counter} chunks")
        
        # Phase 2: Merge chunks by year into final yearly files
        print("Phase 2: Merging chunks into final yearly files...")
        merge_yearly_chunks(temp_dir, yearly_dir, compress, workers)
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    end_time = datetime.now()
    runtime = end_time - start_time
    
    print(f"\n✅ Yearly export complete!")
    print(f"Runtime: {runtime}")
    print(f"Yearly files created in: {yearly_dir}")


def write_yearly_chunk(entries, chunk_file, compress):
    """Write a chunk of yearly entries to a temporary file."""
    output_handler = FileHandler(chunk_file, is_output=True, compress=compress)
    
    with output_handler.open() as f:
        for entry in entries:
            line = output_handler.serialize(entry)
            f.write(line)
    
    return chunk_file


def merge_yearly_chunks(temp_dir, final_dir, compress, workers):
    """Merge temporary chunk files by year into final yearly files."""
    from collections import defaultdict
    from multiprocessing import Pool
    
    # Group chunk files by year
    chunk_files = [f for f in os.listdir(temp_dir) if f.endswith('.jsonl') or f.endswith('.lz4')]
    year_chunks = defaultdict(list)
    
    for chunk_file in chunk_files:
        year = chunk_file.split('_chunk_')[0]
        year_chunks[year].append(os.path.join(temp_dir, chunk_file))
    
    print(f"Merging chunks for {len(year_chunks)} years...")
    
    # Create merge tasks
    merge_tasks = []
    for year, chunk_list in year_chunks.items():
        final_file = os.path.join(
            final_dir, 
            f"{year}.jsonl" + ('.lz4' if compress else '')
        )
        merge_tasks.append((year, chunk_list, final_file, compress))
    
    # Execute merges in parallel
    with Pool(processes=workers) as pool:
        with tqdm(total=len(merge_tasks), desc="Merging years") as pbar:
            for result in pool.imap_unordered(merge_year_chunks, merge_tasks):
                pbar.update(1)


def merge_year_chunks(args):
    """Merge all chunk files for a single year."""
    year, chunk_files, final_file, compress = args
    
    output_handler = FileHandler(final_file, is_output=True, compress=compress)
    
    with output_handler.open() as out_f:
        for chunk_file in chunk_files:
            input_handler = FileHandler(chunk_file)
            with input_handler.open() as in_f:
                shutil.copyfileobj(in_f, out_f)
    
    return year


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ngrams into RocksDB")
    parser.add_argument("--mode", choices=["downloaded", "filtered"], default="downloaded",
                       help="Mode: 'downloaded' for direct ingestion with processing, 'filtered' for pre-processed files")
    parser.add_argument("--ngram_size", type=int, required=True, help="Ngram size (1-5)")
    parser.add_argument("--proj_dir", type=str, required=True, help="Project directory")
    parser.add_argument("--ngram_type", type=str, default="tagged", choices=["tagged", "untagged"],
                       help="Ngram type (for downloaded mode)")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing database")
    
    # Processing options for downloaded mode
    parser.add_argument("--no-lowercase", action="store_true", help="Don't apply lowercasing")
    parser.add_argument("--no-lemmatize", action="store_true", help="Don't apply lemmatization")
    parser.add_argument("--vocab-file", type=str, help="Vocabulary file for filtering")
    parser.add_argument("--min-tokens", type=int, default=2, help="Minimum tokens after filtering")
    
    args = parser.parse_args()
    
    if args.mode == "downloaded":
        # Process downloaded files with full transformation pipeline
        apply_filters = {
            'numerals': True,
            'nonalpha': True,
            'stops': True,
            'min_token_length': 3,
            'replace_unk': False
        }
        
        db_path = ingest_downloaded_files_to_rocksdb(
            ngram_size=args.ngram_size,
            proj_dir=args.proj_dir,
            ngram_type=args.ngram_type,
            workers=args.workers,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            apply_lowercase=not args.no_lowercase,
            apply_lemmatize=not args.no_lemmatize,
            apply_filters=apply_filters,
            vocab_file=args.vocab_file,
            min_tokens=args.min_tokens
        )
    else:
        # Process pre-filtered files
        db_path = ingest_filtered_files_to_rocksdb(
            ngram_size=args.ngram_size,
            proj_dir=args.proj_dir,
            workers=args.workers,
            batch_size=args.batch_size,
            overwrite=args.overwrite
        )
    
    print(f"Database created at: {db_path}")
