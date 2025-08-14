"""
rocksdb_post_ingest.py
Optimized utilities for post-ingestion RocksDB operations: compaction, read-tuning, and fast queries.
"""
import rocksdict
import logging

# Example: Read-optimized options (tune as needed)
READ_OPT_OPTIONS = {
    "max_open_files": 1000,
    "max_background_jobs": 2,
    "block_cache_size": 512 * 1024 * 1024,  # 512MB
    "table_cache_numshardbits": 6,
    "level0_file_num_compaction_trigger": 4,
    "compression": "snappy"
}

# Example: Write-optimized options (tune as needed)
WRITE_OPT_OPTIONS = {
    "max_open_files": 1000,
    "max_background_jobs": 8,
    "write_buffer_size": 256 * 1024 * 1024,  # 256MB
    "level0_file_num_compaction_trigger": 8,
    "compression": "none",
    "allow_concurrent_memtable_write": True,
    "enable_write_thread_adaptive_yield": True
}

def open_db_read_optimized(db_path, options=None):
    """Open RocksDB with read-optimized settings."""
    opts = rocksdict.Options()
    if options is None:
        options = READ_OPT_OPTIONS
    for k, v in options.items():
        if hasattr(opts, k):
            setattr(opts, k, v)
    db = rocksdict.Rdict(db_path, opts)
    return db

def open_db_write_optimized(db_path, options=None):
    """Open RocksDB with write-optimized settings."""
    opts = rocksdict.Options()
    if options is None:
        options = WRITE_OPT_OPTIONS
    for k, v in options.items():
        if hasattr(opts, k):
            setattr(opts, k, v)
    db = rocksdict.Rdict(db_path, opts)
    return db

def run_manual_compaction(db):
    """Trigger manual compaction on the entire DB."""
    try:
        db.compact_range()
        logging.info("Manual compaction completed.")
    except Exception as e:
        logging.error(f"Compaction failed: {e}")

