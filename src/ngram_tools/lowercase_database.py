#!/usr/bin/env python3
"""
Lowercase ngram database keys and aggregate frequency data for collisions.

This module processes a RocksDB database containing ngram data, converts all keys
to lowercase, and properly aggregates frequency data when multiple case variants
of the same ngram exist.
"""

import signal
import sys
import time
import orjson
import rocksdict
from typing import Dict, Any, Optional, Union
from tqdm import tqdm


def aggregate_frequency_data(existing_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate frequency data from two ngram entries.
    
    Combines frequency lists and sums frequencies/document_counts for matching years.
    
    Args:
        existing_data: The data already in the target database
        new_data: The new data to aggregate
        
    Returns:
        Combined data with aggregated frequencies
    """
    # Start with existing data
    result = existing_data.copy()
    
    # Ensure the ngram field is lowercase
    if 'ngram' in new_data:
        result['ngram'] = new_data['ngram'].lower()
    
    # Handle frequencies list aggregation
    if 'frequencies' in new_data and 'frequencies' in existing_data:
        # Create a dictionary to aggregate by year
        year_aggregates = {}
        
        # Process existing frequencies
        for freq_entry in existing_data['frequencies']:
            year = freq_entry['year']
            year_aggregates[year] = {
                'year': year,
                'frequency': freq_entry.get('frequency', 0),
                'document_count': freq_entry.get('document_count', 0)
            }
        
        # Add new frequencies (summing if year already exists)
        for freq_entry in new_data['frequencies']:
            year = freq_entry['year']
            if year in year_aggregates:
                # Sum frequencies and document counts for existing year
                year_aggregates[year]['frequency'] += freq_entry.get('frequency', 0)
                year_aggregates[year]['document_count'] += freq_entry.get('document_count', 0)
            else:
                # Add new year
                year_aggregates[year] = {
                    'year': year,
                    'frequency': freq_entry.get('frequency', 0),
                    'document_count': freq_entry.get('document_count', 0)
                }
        
        # Convert back to list, sorted by year
        result['frequencies'] = sorted(year_aggregates.values(), key=lambda x: x['year'])
    
    elif 'frequencies' in new_data:
        # Only new_data has frequencies
        result['frequencies'] = new_data['frequencies']
    
    # Handle any other fields
    for key, value in new_data.items():
        if key not in ['ngram', 'frequencies']:
            result[key] = value
    
    return result


def lowercase_ngram_database(
    source_db_path: str,
    target_db_path: str,
    progress_interval: int = 1_000_000,
    max_items: Optional[int] = None
) -> Dict[str, Union[int, float]]:
    """
    Convert all ngram keys to lowercase and aggregate frequency data for collisions.
    
    Args:
        source_db_path: Path to the source RocksDB database (read-only)
        target_db_path: Path to the target RocksDB database (will be created/overwritten)
        progress_interval: How often to print progress updates
        max_items: Maximum number of items to process (None for all)
        
    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()
    source_db = None
    target_db = None
    
    # Set up interrupt handler for clean database closure
    def signal_handler(signum, frame):
        print(f"\nInterrupt received (signal {signum})")
        print("Cleaning up databases...")
        if source_db:
            try:
                source_db.close()
                print("Source database closed.")
            except Exception as e:
                print(f"Error closing source database: {e}")
        if target_db:
            try:
                target_db.close()
                print("Target database closed.")
            except Exception as e:
                print(f"Error closing target database: {e}")
        print("Cleanup complete. Exiting...")
        sys.exit(0)
    
    # Register signal handlers for common interrupt signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    # Display configuration header in your ingestion style
    print('\033[4mNgram Database Lowercasing Configuration\033[0m')
    print(f'Source database:        {source_db_path}')
    print(f'Target database:        {target_db_path}')
    print(f'Progress interval:      {progress_interval:,} items')
    print(f'Max items:              {max_items:,} items' if max_items else 'Max items:              No limit')
    print()
    
    # Open databases
    print(f"Opening source database: {source_db_path}")
    source_db = rocksdict.Rdict(source_db_path, readonly=True)
    
    print(f"Opening target database: {target_db_path}")
    target_db = rocksdict.Rdict(target_db_path)
    
    # Statistics
    total_processed = 0
    collision_count = 0
    unique_keys = 0
    next_progress = progress_interval
    
    print(f"\nStarting lowercase processing...")
    print("="*80)
    
    try:
        # Use tqdm for progress tracking like your ingestion script
        with tqdm(
            desc="Processing Entries", 
            unit='entries',
            colour='blue',
            dynamic_ncols=True
        ) as pbar:
            
            for original_key in source_db:
                # Check if we've hit the limit
                if max_items is not None and total_processed >= max_items:
                    break
                
                # Convert to lowercase
                lowercase_key = original_key.lower()
                
                # Progress reporting (like your ingestion script style)
                if total_processed >= next_progress:
                    elapsed = time.time() - start_time
                    items_per_second = total_processed / elapsed if elapsed > 0 else 0
                    print(f"[PROGRESS] Processed {total_processed:,} items | "
                          f"Collisions: {collision_count:,} | "
                          f"Rate: {items_per_second:.0f} items/sec", flush=True)
                    next_progress += progress_interval
                
                # Check for collision
                if lowercase_key in target_db:
                    # Collision detected - aggregate the data
                    collision_count += 1
                    existing_data = orjson.loads(target_db[lowercase_key])
                    new_data = orjson.loads(source_db[original_key])
                    
                    # Aggregate frequency data
                    merged_data = aggregate_frequency_data(existing_data, new_data)
                    
                    # Store merged result
                    target_db[lowercase_key] = orjson.dumps(merged_data)
                    
                else:
                    # No collision - direct copy with lowercase key
                    unique_keys += 1
                    data = orjson.loads(source_db[original_key])
                    
                    # Ensure ngram field is lowercase
                    if 'ngram' in data:
                        data['ngram'] = data['ngram'].lower()
                    
                    target_db[lowercase_key] = orjson.dumps(data)
                
                total_processed += 1
                pbar.update(1)
    
    except KeyboardInterrupt:
        print(f"\nProcess interrupted by user")
        raise
    finally:
        # Always close databases, even on interrupt
        if source_db:
            try:
                source_db.close()
            except Exception as e:
                print(f"Error closing source database: {e}")
        if target_db:
            try:
                target_db.close()
            except Exception as e:
                print(f"Error closing target database: {e}")
    
    # Final statistics in your ingestion script style
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print('\033[32mProcessing completed successfully!\033[0m')
    print(f'Items processed:        {total_processed:,}')
    print(f'Unique lowercase keys:  {unique_keys:,}')
    print(f'Collisions handled:     {collision_count:,}')
    print(f'\033[31mTotal Runtime: {elapsed_time:.2f} seconds\033[0m')
    
    if elapsed_time > 0:
        items_per_second = total_processed / elapsed_time
        print(f'\033[34mAverage rate: {items_per_second:.0f} items/second\033[0m')
    
    # Return statistics
    return {
        'total_processed': total_processed,
        'unique_keys': unique_keys,
        'collisions': collision_count,
        'elapsed_time': elapsed_time
    }
    
    print(f"Opening target database: {target_db_path}")
    target_db = rocksdict.Rdict(target_db_path)
    
    # Statistics
    total_processed = 0
    collision_count = 0
    unique_keys = 0
    next_progress = progress_interval
    
    print(f"\nStarting lowercase processing...")
    print("="*80)
    
    try:
        # Use tqdm for progress tracking like your ingestion script
        with tqdm(
            desc="Processing Entries", 
            unit='entries',
            colour='blue',
            dynamic_ncols=True
        ) as pbar:
            
            for original_key in source_db:
                # Check if we've hit the limit
                if max_items is not None and total_processed >= max_items:
                    break
                
                # Convert to lowercase
                lowercase_key = original_key.lower()
                
                # Progress reporting
                if total_processed >= next_progress:
                    elapsed = time.time() - start_time
                    items_per_second = total_processed / elapsed if elapsed > 0 else 0
                    print(f"[PROGRESS] Processed {total_processed:,} items | "
                          f"Collisions: {collision_count:,} | "
                          f"Rate: {items_per_second:.0f} items/sec", flush=True)
                    next_progress += progress_interval
                
                # Check for collision
                if lowercase_key in target_db:
                    # Collision detected - aggregate the data
                    collision_count += 1
                    existing_data = orjson.loads(target_db[lowercase_key])
                    new_data = orjson.loads(source_db[original_key])
                    
                    # Aggregate frequency data
                    merged_data = aggregate_frequency_data(existing_data, new_data)
                    
                    # Store merged result
                    target_db[lowercase_key] = orjson.dumps(merged_data)
                    
                else:
                    # No collision - direct copy with lowercase key
                    unique_keys += 1
                    data = orjson.loads(source_db[original_key])
                    
                    # Ensure ngram field is lowercase
                    if 'ngram' in data:
                        data['ngram'] = data['ngram'].lower()
                    
                    target_db[lowercase_key] = orjson.dumps(data)
                
                total_processed += 1
                pbar.update(1)
    
    finally:
        # Close databases
        source_db.close()
        target_db.close()
    
    # Final statistics in your ingestion script style
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print('\033[32mProcessing completed successfully!\033[0m')
    print(f'Items processed:        {total_processed:,}')
    print(f'Unique lowercase keys:  {unique_keys:,}')
    print(f'Collisions handled:     {collision_count:,}')
    print(f'\033[31mTotal Runtime: {elapsed_time:.2f} seconds\033[0m')
    
    if elapsed_time > 0:
        items_per_second = total_processed / elapsed_time
        print(f'\033[34mAverage rate: {items_per_second:.0f} items/second\033[0m')
    
    # Return statistics
    return {
        'total_processed': total_processed,
        'unique_keys': unique_keys,
        'collisions': collision_count,
        'elapsed_time': elapsed_time
    }
