#!/usr/bin/env python3
"""
Simple script to count the total number of items in a RocksDB database.
"""

import sys
import time
import rocksdict


def count_db_items(db_path, progress_interval=10_000_000):
    """
    Count the total number of items in a RocksDB database.
    
    Args:
        db_path: Path to the RocksDB database
        progress_interval: How often to print progress updates
    
    Returns:
        int: Total number of items in the database
    """
    print(f"Database Path: {db_path}")
    print("=" * 60)
    
    # Open database
    try:
        db = rocksdict.Rdict(db_path)
    except Exception as e:
        print(f"Error opening database: {e}")
        return 0
    
    # Start timing
    start_time = time.perf_counter()
    count = 0
    
    try:
        # Count items
        for key in db.keys():
            count += 1
            
            # Print progress updates
            if count % progress_interval == 0:
                elapsed = time.perf_counter() - start_time
                rate = count / elapsed if elapsed > 0 else 0
                print(f"Progress: {count:,} items | {rate:,.0f} items/sec | {elapsed:.1f}s elapsed")
        
        # Final results
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print("=" * 60)
        print(f"FINAL COUNT: {count:,} items")
        print(f"Total Time:  {total_time:.2f} seconds")
        if total_time > 0:
            print(f"Average Rate: {count / total_time:,.0f} items/second")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print(f"\nInterrupted! Counted {count:,} items so far.")
    
    finally:
        db.close()
    
    return count


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python count_db_items.py <database_path>")
        print("Example: python count_db_items.py /path/to/your/database.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    count = count_db_items(db_path)
    
    return count


if __name__ == "__main__":
    main()
