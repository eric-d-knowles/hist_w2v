"""
rocksdb_post_ingest.py
Optimized utilities for post-ingestion RocksDB operations: compaction, read-tuning, and fast queries.
"""
import rocksdict   
from rocksdict import Options, Rdict
import logging


def open_db_read_optimized(db_path, options=None):
    """Open RocksDB with read-optimized settings."""
    options = rocksdict.Options()
    options.create_if_missing(True)
    options.set_max_background_jobs(8)
    options.set_write_buffer_size(512 * 1024 * 1024)
    options.set_level_zero_file_num_compaction_trigger(4)
    db = rocksdict.Rdict(db_path, options)
    return db

def open_db_write_optimized(db_path, options=None):
    """Open RocksDB with write-optimized settings."""
    options = rocksdict.Options()
    options.create_if_missing(True)
    options.set_max_background_jobs(8)
    options.set_write_buffer_size(256 * 1024 * 1024)
    options.set_level_zero_file_num_compaction_trigger(8)
    db = rocksdict.Rdict(db_path, options)
    return db


def run_manual_compaction(db_path, optimization="read"):
    """Trigger manual compaction on the entire DB with specified optimization ('read' or 'write')."""
    if optimization == "read":
        db = open_db_read_optimized(db_path)
    elif optimization == "write":
        db = open_db_write_optimized(db_path)
    else:
        raise ValueError("optimization must be 'read' or 'write'")
    try:
        db.compact_range(None, None)
        logging.info(f"Manual compaction completed with {optimization}-optimized settings.")
    except Exception as e:
        logging.error(f"Compaction failed: {e}")
    finally:
        db.close()
