import cython
import signal
import sys
import time as py_time
import rocksdict
from libc.stdlib cimport rand, srand
from libc.time cimport time
from random import randint

# Cython-specific optimizations
@cython.boundscheck(False)  # Disable bounds-checking for faster performance
@cython.wraparound(False)   # Disable negative indexing checks
def reservoir_sampling(
    db_path,
    sample_size,
    progress_interval=10000000,
    max_items=None,
    key_type="check",
    return_keys: bool = False,
):
    """
    Perform reservoir sampling on a RocksDB database with progress updates.
    
    Args:
        db_path: Path to the RocksDB database
        sample_size: Number of items to sample
        progress_interval: How often to print progress updates (default: every 10,000,000 items)
        max_items: Maximum number of database entries to traverse (default: None for unlimited)
        key_type: Key handling strategy - "check" (auto-detect), "string" (assume strings), "bytes" (assume bytes)
    
    Returns:
        List of sampled values (bytes) or list of (key, value) tuples if
        return_keys is True. Keys are returned as str.
    """
    # Set up interrupt handler for clean database closure
    cdef db = None
    
    def signal_handler(signum, frame):
        print(f"\nInterrupt received (signal {signum})")
        print("Cleaning up database...")
        if db is not None:
            try:
                db.close()
                print("Database closed.")
            except Exception as e:
                print(f"Error closing database: {e}")
        print("Cleanup complete. Exiting...")
        sys.exit(0)
    
    # Register signal handlers for common interrupt signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    # Open database
    db = rocksdict.Rdict(db_path, readonly=True)
    
    try:
        # Initialize the reservoir
    cdef list reservoir = []
        cdef int idx
        cdef int i = 0
        cdef long long total_processed = 0
        cdef long long skipped_metadata = 0
        cdef long long next_progress = progress_interval  # Pre-compute next progress checkpoint

        # Seed the random number generator
        srand(int(time(NULL)))
        
        # Start timing
        start_time = py_time.time()
        
        print("="*60)
        print(f"RESERVOIR SAMPLING CONFIGURATION")
        print("-"*60)
    print(f"Target sample size:     {sample_size:,} items")
    print(f"Key handling strategy:  {key_type}")
    print(f"Progress interval:      {progress_interval:,} items")
    if max_items is not None:
        print(f"Database limit:         {max_items:,} entries")
    else:
        print(f"Database limit:         No limit (full traversal)")
    print("="*60)

    # Collect items for the reservoir while skipping metadata
    for key in db.keys():
        # Check if we've hit the maximum items limit
        if max_items is not None and (total_processed + skipped_metadata) >= max_items:
            print(f"[INFO] Reached traversal limit of {max_items:,} entries")
            break
            
        # Handle key type based on user preference for performance
        if key_type == "string":
            # Assume keys are strings - no type checking
            key_str = key
        elif key_type == "bytes":
            # Assume keys are bytes - always decode
            key_str = key.decode('utf-8')
        else:
            # Auto-detect key type (default behavior with type checking)
            if isinstance(key, bytes):
                key_str = key.decode('utf-8')
            else:
                key_str = str(key)
        
        # Skip metadata keys
        if key_str.startswith('__'):
            skipped_metadata += 1
            continue
            
        total_processed += 1
        
        # Optimized progress updates - use checkpoint instead of expensive modulo
        if total_processed >= next_progress:
            print(f"[PROGRESS] Processed {total_processed:,} items", flush=True)
            next_progress += progress_interval  # Pre-compute next checkpoint
        
        # Fill initial reservoir
        if len(reservoir) < sample_size:
            if return_keys:
                reservoir.append((key_str, db[key]))
            else:
                reservoir.append(db[key])
        else:
            # Reservoir sampling algorithm - C-level random for speed
            idx = rand() % total_processed
            if idx < sample_size:
                if return_keys:
                    reservoir[idx] = (key_str, db[key])
                else:
                    reservoir[idx] = db[key]

    # Calculate timing
    end_time = py_time.time()
    elapsed_time = end_time - start_time

    print("="*60)
    print("RESERVOIR SAMPLING RESULTS")
    print("-"*60)
    print(f"Items processed:        {total_processed:,}")
    print(f"Metadata entries:       {skipped_metadata:,}")
    print(f"Final sample size:      {len(reservoir):,}")
    print(f"Execution time:         {elapsed_time:.4f} seconds")
    print("-"*60)
    
    # Calculate performance metrics
    if elapsed_time > 0:
        items_per_second = total_processed / elapsed_time
        microseconds_per_item = (elapsed_time * 1_000_000) / total_processed if total_processed > 0 else 0
        print("PERFORMANCE METRICS")
        print("-"*60)
        print(f"Processing rate:        {items_per_second:,.0f} items/second")
        print(f"Time per item:          {microseconds_per_item:.2f} microseconds")
    print("="*60)
    
    finally:
        # Close the database connection
        try:
            db.close()
        except Exception as e:
            print(f"Warning: Error closing database: {e}", file=sys.stderr)
    
    return reservoir