import signal
import sys
import time
import rocksdict


def count_db_items(db_path, long progress_interval=10_000_000):
    """
    Count the total number of items in a RocksDB database using Cython optimization.
    
    Args:
        db_path: Path to the RocksDB database
        progress_interval: How often to print progress updates
    
    Returns:
        int: Total number of items in the database
    """
    # Cython variable declarations must be at the top
    cdef db = None
    cdef double start_time, elapsed, rate, end_time, total_time
    cdef long count = 0
    cdef long next_progress = progress_interval  # Pre-compute next progress checkpoint

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
        print("Database Item Counter")
        print("=" * 60)

        # Start timing
        start_time = time.perf_counter()

        try:
            # Optimized iteration with pre-computed checkpoints
            iterator = db.keys()

            # Count items with optimized loop
            for key in iterator:
                count += 1

                # Optimized progress check - only check when we hit the checkpoint
                if count >= next_progress:
                    elapsed = time.perf_counter() - start_time
                    print(f"Progress: {count:,} items | {elapsed:.1f}s elapsed", flush=True)
                    next_progress += progress_interval  # Pre-compute next checkpoint

            # Final results
            end_time = time.perf_counter()
            total_time = end_time - start_time

            print("=" * 60)
            print(f"FINAL COUNT: {count:,} items")
            print(f"Total Time:  {total_time:.2f} seconds")
            if total_time > 0:
                microseconds_per_item = (total_time * 1_000_000) / count
                print(f"Average Rate: {count / total_time:,.0f} items/second")
                print(f"Time per item: {microseconds_per_item:.3f} microseconds/item")
            print("=" * 60)

        except KeyboardInterrupt:
            print(f"\nInterrupted! Counted {count:,} items so far.")

    finally:
        # Close the database connection
        try:
            db.close()
        except Exception as e:
            print(f"Warning: Error closing database: {e}", file=sys.stderr)

    return count
