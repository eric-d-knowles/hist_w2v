"""Download Google ngram files and batch-ingest them into RocksDB.

Pipeline:
    1. Discover file URLs
    2. Optional shuffle for load balancing
    3. Parallel download & parsing (processes or threads)
    4. Sequential batched RocksDB writes

Value payloads store only frequency lists; the RocksDB key is the ngram.

Example:
    from ngram_tools.download_and_ingest_to_rocksdb import (
        download_and_ingest_to_rocksdb,
    )

    download_and_ingest_to_rocksdb(
        ngram_size=2,
        repo_release_id="20200217",
        repo_corpus_id="eng-us-all",
        db_path="/data/ngrams.db",
        file_range=(0, 9),
        workers=8,
        random_seed=42,
        ngram_type="tagged",
        overwrite=False,
        use_threads=True,
    )
"""

# Standard library imports
import gzip
import logging
import os
import shutil
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import orjson
import requests
import rocksdict
from rocksdict import Options, WriteBatch
from urllib.parse import urljoin
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
import random
from typing import TypedDict

# Define NgramRecord as a TypedDict for type hints
class NgramRecord(TypedDict, total=False):
    ngram: str
    year: int
    match_count: int
    volume_count: int
    frequencies: list

def cleanup_database_safely(db_path: str, max_retries: int = 5) -> bool:
    """
    Safely remove a RocksDB database directory, handling NFS temp files.
    Args:
        db_path: Path to the database directory.
        max_retries: Number of attempts to remove the directory.
    Returns:
        True if successful, False otherwise.
    """
    import glob
    if not os.path.exists(db_path):
        return True
    for attempt in range(max_retries):
        try:
            nfs_files = glob.glob(os.path.join(db_path, '.nfs*'))
            for nfs_file in nfs_files:
                try:
                    os.remove(nfs_file)
                    logging.info(f"Removed NFS temp file: {os.path.basename(nfs_file)}")
                except OSError:
                    pass  # Some may still be in use
            shutil.rmtree(db_path)
            logging.info(f"Successfully removed database (attempt {attempt + 1})")
            return True
        except OSError as e:
            if attempt < max_retries - 1:
                logging.warning(f"Database cleanup attempt {attempt + 1} failed: {e}")
                logging.info(f"Waiting 2 seconds before retry...")
                time.sleep(2)
            else:
                logging.error(f"Failed to remove database after {max_retries} attempts: {e}")
                return False
    return False


def define_regex(ngram_type: str) -> re.Pattern:
    """
    Defines the regular expression for matching ngram tokens.
    
    Filters ngrams based on whether they contain POS tags or not, allowing
    for corpus size reduction and memory pressure relief during ingestion.

    Args:
        ngram_type: The ngram type to filter for:
            - 'tagged': Only include ngrams with POS tags (e.g., "word_NOUN")
            - 'untagged': Only include ngrams without POS tags (plain words)
            - 'all': Include all ngrams (no filtering)

    Returns:
        Compiled regex pattern for matching the specified ngram type.
        
    Example:
        For tagged: matches "cat_NOUN dog_VERB" but not "cat dog"
        For untagged: matches "cat dog" but not "cat_NOUN dog_VERB"
        For all: matches everything (no filtering applied)
    """
    valid_tags = r'NOUN|PROPN|VERB|ADJ|ADV|PRON|DET|ADP|NUM|CONJ|X|\.'
    
    if ngram_type == 'tagged':
        return re.compile(rf'^(\S+_(?:{valid_tags})\s?)+$')
    if ngram_type == 'untagged':
        return re.compile(rf'^(?!.*_(?:{valid_tags})\s?)(\S+\s?)*$')
    if ngram_type == 'all':
        return re.compile(r'.*')
    raise ValueError(
        f"Invalid ngram_type: {ngram_type}. Must be 'tagged', 'untagged', or 'all'."
    )


def setup_logging(db_path: str) -> None:
    """
    Configure logging to write to a file in the same directory as the database.
    
    Args:
        db_path: Path to the database (used to determine log file location).
    """
    # Get the directory containing the database
    log_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "."
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ngram_download_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
        ],
        force=True  # Override any existing configuration
    )
    
    # Log the setup
    logging.info(f"Logging to: {log_path}")


# Configuration constants
DEFAULT_WRITE_BATCH_SIZE = 10000000       # Default: write every 10M entries
REQUEST_TIMEOUT = (15, 120)      # Connection and read timeouts for downloads


def write_batch_to_db(db: rocksdict.Rdict, pending_data: Dict[str, bytes]) -> int:
    """Write accumulated ngrams to RocksDB, returning count written."""
    if not pending_data:
        return 0
    
    entries_count = len(pending_data)
    logging.info(f"Writing batch: {entries_count:,} entries")
    
    try:
        # Use WriteBatch for optimized bulk operations
        wb = WriteBatch()
        for ngram_key, serialized_data in pending_data.items():
            wb.put(ngram_key, serialized_data)
        # Execute the entire batch in one atomic operation
        db.write(wb)
        del wb
        logging.info(f"Batch complete")
        return entries_count
    except Exception as e:
        logging.error(f"Error writing batch: {e}")
        raise


def get_processed_files(db: rocksdict.Rdict) -> set:
    """Get the set of files that have already been processed.
    
    Retrieves the list of processed files from a metadata key in the database.
    This allows resuming processing from where it left off in previous sessions.
    
    Args:
        db: RocksDB database instance
        
    Returns:
        Set of filenames that have been successfully processed
    """
    try:
        metadata_key = "__processed_files__"
        if metadata_key in db:
            raw_data = db[metadata_key]
            file_list = orjson.loads(raw_data)
            return set(file_list)
        return set()
    except Exception as e:
        logging.warning(f"Could not retrieve processed files list: {e}")
        return set()


def mark_file_as_processed(db: rocksdict.Rdict, filename: str) -> None:
    """Mark a file as successfully processed in the database metadata.
    
    Updates the metadata tracking which files have been processed to enable
    resuming across sessions without reprocessing completed files.
    
    Args:
        db: RocksDB database instance
        filename: Name of the file that was successfully processed
    """
    try:
        metadata_key = "__processed_files__"
        
        # Get current list
        processed_files = get_processed_files(db)
        
        # Add new file
        processed_files.add(filename)
        
        # Save updated list
        db[metadata_key] = orjson.dumps(list(processed_files))
        
        logging.info(f"Marked processed: {filename}")
    except Exception as e:
        logging.error(f"Could not mark file as processed {filename}: {e}")


def setup_rocksdb(db_path: str) -> rocksdict.Rdict:
    """Set up and return a RocksDB instance optimized for high-performance.
    
    Creates the database directory if needed and uses rocksdict's simple
    constructor which doesn't support complex options but is reliable.
    
    Args:
        db_path: Absolute path to the RocksDB database directory.
        
    Returns:
        RocksDB instance ready for high-throughput operations.
        
    Raises:
        Exception: If database cannot be opened or created.
    """
    # Ensure parent directory exists
    parent_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "."
    os.makedirs(parent_dir, exist_ok=True)
    
    # Advanced options for compaction and performance tuning (using supported setter methods)
    options = rocksdict.Options()
    options.create_if_missing(True)
    options.set_max_background_jobs(8)
    options.set_write_buffer_size(128 * 1024 * 1024)
    options.set_level_zero_file_num_compaction_trigger(12)
    # You can further tune these if needed:
    # options.set_target_file_size_base(128 * 1024 * 1024)
    # options.set_max_bytes_for_level_base(1024 * 1024 * 1024)
    # options.set_max_bytes_for_level_multiplier(4.0)

    try:
        db = rocksdict.Rdict(db_path, options)
        logging.info(f"RocksDB opened with tuned options: {db_path}")
        return db
    except Exception as e:
        # Handle lock conflicts from previous processes
        if "lock" in str(e).lower():
            logging.warning(f"Database lock issue detected: {e}")
            time.sleep(0.2)  # Brief wait
            try:
                return rocksdict.Rdict(db_path, options)
            except Exception as retry_e:
                logging.error(
                    f"Failed to open database after retry: {retry_e}"
                )
                raise
        else:
            logging.error(f"Failed to open RocksDB: {e}")
            raise


def set_location_info(
    ngram_size: int, 
    repo_release_id: str,
    repo_corpus_id: str
) -> Tuple[str, str]:
    """Construct repository URL and file patterns based on ngram parameters.
    
    Builds the Google Books ngram repository URL and creates a regex pattern
    to match the compressed ngram files for the specified size and corpus.
    
    Args:
        ngram_size: The ngram size (1=unigrams, 2=bigrams, etc.). Must be 1-5.
        repo_release_id: Google's release identifier (e.g., "20200217").
        repo_corpus_id: Corpus identifier (e.g., "eng-us-all").

    Returns:
        A tuple containing:
            - ngram_repo_url: Complete URL to the ngram repository page
            - file_pattern: Regex pattern to match downloadable .gz files
            
    Example:
        url, pattern = set_location_info(2, "20200217", "eng-us-all")
    # Returns URL for bigrams and pattern like r'2-\\d{5}-of-\\d{5}\\.gz'
    """
    # Construct the Google Books ngram repository URL
    base_url = "https://storage.googleapis.com/books/ngrams"
    ngram_repo_url = (
        f"{base_url}/books/{repo_release_id}/{repo_corpus_id}/"
        f"{repo_corpus_id}-{ngram_size}-ngrams_exports.html"
    )
    
    # Create regex pattern to match numbered .gz files 
    # (e.g., "2-00001-of-00024.gz")
    file_pattern = rf'{ngram_size}-\d{{5}}-of-\d{{5}}\.gz'
    
    return ngram_repo_url, file_pattern


def fetch_file_urls(
    ngram_repo_url: str,
    file_pattern: str,
    *,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> List[str]:
    """Fetch downloadable file URLs with simple retry/backoff.

    Args:
        ngram_repo_url: Repository HTML page.
        file_pattern: Regex to extract file names.
        max_retries: Attempts before failing.
        backoff: Seconds base for exponential backoff.
    """
    attempt = 0
    while True:
        try:
            logging.info(
                f"Fetching URLs from {ngram_repo_url} (attempt {attempt + 1})"
            )
            response = requests.get(ngram_repo_url, timeout=30)
            response.raise_for_status()
            file_urls = [
                urljoin(ngram_repo_url, filename)
                for filename in re.findall(file_pattern, response.text)
            ]
            logging.info(f"Found {len(file_urls)} files")
            return file_urls
        except (requests.RequestException, re.error) as e:
            attempt += 1
            if attempt >= max_retries:
                logging.error(
                    f"Failed to fetch file URLs after {attempt} attempts: {e}"
                )
                raise RuntimeError("Failed to fetch file URLs") from e
            sleep_for = backoff ** attempt
            logging.warning(
                f"Fetch failed ({e}); retrying in {sleep_for:.1f}s..."
            )
            time.sleep(sleep_for)


def parse_ngram_line(
    line: str, filter_regex: Optional[re.Pattern] = None
) -> Tuple[Optional[str], Optional[NgramRecord]]:
    """Parse a raw ngram line into (ngram_key, record) or (None, None).

    Input format: "ngram\tYEAR,FREQ,DOC\tYEAR,FREQ,DOC...".
    The returned record omits the ngram string itself (held as the DB key).
    Filtering occurs before parsing frequency tuples.
    """
    line = line.strip()
    if not line:
        return None, None
        
    try:
        # Split into ngram text and frequency data on FIRST tab only
        parts = line.split('\t', 1)  # Split on first tab only
        if len(parts) < 2:
            return None, None
            
        ngram_text = parts[0]
        
        # Apply filtering if regex pattern is provided
        if filter_regex is not None and not filter_regex.match(ngram_text):
            return None, None
        
        frequency_data = parts[1]  # This now contains all the yearly tuples
        
        # Create structured data with ngram as string
        ngram_data: NgramRecord = {
            'frequencies': []
        }
        
        # Parse frequency entries: "year,frequency,document_count" separated by tabs
        for entry in frequency_data.split('\t'):
            try:
                year_str, freq_str, doc_count_str = entry.split(',')
                ngram_data['frequencies'].append({  # type: ignore[arg-type]
                    'year': int(year_str),
                    'frequency': int(freq_str),
                    'document_count': int(doc_count_str)
                })
            except (ValueError, IndexError):
                # Skip malformed frequency entries
                continue
        
        return ngram_text, ngram_data
        
    except Exception as e:
        # Log parsing errors but continue processing
        logging.warning(f"Failed to parse line: {line[:100]}... Error: {e}")
        return None, None


def _stream_download_with_retries(
    url: str, *, max_retries: int = 3, backoff: float = 2.0
):
    """Return streaming HTTP response with retry/backoff."""
    attempt = 0
    while True:
        try:
            resp = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep_for = backoff ** attempt
            logging.warning(
                "Download retry %s/%s for %s after error %s (sleep %.1fs)",
                attempt,
                max_retries,
                url.split('/')[-1],
                e,
                sleep_for,
            )
            time.sleep(sleep_for)


def process_and_ingest_file(
    url: str, worker_id: int, filter_pattern: Optional[str]
) -> Tuple[str, Dict[str, bytes]]:
    """Download & parse a single gzip ngram file; return status & data."""
    # Label worker process for htop
    try:
        import setproctitle
        setproctitle.setproctitle("PROC_WORKER")
    except ImportError:
        pass
    filename = url.split('/')[-1]
    parsed_data: Dict[str, bytes] = {}
    pid = os.getpid()
    try:
        logging.info(f"Worker {worker_id}: Processing {filename}")
        response = _stream_download_with_retries(url)
        content_length = response.headers.get('content-length')
        if content_length:
            logging.info(f"Worker {worker_id}: File size: {int(content_length):,} bytes")
        filter_regex = re.compile(filter_pattern) if filter_pattern else None
        lines_processed = 0
        with gzip.GzipFile(fileobj=response.raw, mode='rb') as infile:
            for line in infile:
                lines_processed += 1
                # Periodic manual gc removed (unnecessary for streaming loop)
                try:
                    ngram_key, ngram_data = parse_ngram_line(line.decode('utf-8'), filter_regex)
                    if ngram_key and ngram_data:
                        # Convert ngram_data['frequencies'] (list of dicts) to packed binary
                        import struct
                        freq_tuples = [
                            (
                                freq['year'],
                                freq['frequency'],
                                freq['document_count'],
                            )
                            for freq in ngram_data.get('frequencies', [])
                        ]
                        packed = struct.pack(
                            f'<{len(freq_tuples)*3}I',
                            *(
                                x
                                for tup in freq_tuples
                                for x in tup
                            ),
                        )
                        parsed_data[ngram_key] = packed
                except UnicodeDecodeError as e:
                    logging.warning(
                        "Worker %s (PID %s): Unicode decode error in %s line %s: %s",
                        worker_id,
                        pid,
                        filename,
                        lines_processed,
                        e,
                    )
                except Exception as e:  # noqa: BLE001
                    logging.warning(
                        "Worker %s (PID %s): Error processing line %s from %s: %s",
                        worker_id,
                        pid,
                        lines_processed,
                        filename,
                        e,
                    )
        success_msg = (
            f"SUCCESS: {filename} - {lines_processed:,} lines, "
            f"{len(parsed_data):,} entries"
        )
        logging.info(f"Worker {worker_id}: {success_msg}")
        return success_msg, parsed_data
    except requests.Timeout:
        msg = f"TIMEOUT: {filename}"
        logging.error(f"Worker {worker_id}: Timeout - {filename}")
        return msg, {}
    except requests.RequestException:
        msg = f"NETWORK_ERROR: {filename}"
        logging.error(f"Worker {worker_id}: Network error - {filename}")
        return msg, {}
    except Exception as e:  # noqa: BLE001
        msg = f"ERROR: {filename} - {e}"
        logging.error(f"Worker {worker_id}: Error - {filename}: {e}")
        return msg, {}


def print_info(
    ngram_repo_url: str, 
    db_path: str, 
    file_range: Tuple[int, int],
    file_urls_available: List[str], 
    file_urls_to_use: List[str],
    ngram_size: int, 
    workers: int, 
    executor_name: str, 
    start_time: datetime,
    ngram_type: str = 'all',
    overwrite: bool = True,
    files_to_skip: int = 0,
    write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE
    ) -> None:
    """Display comprehensive information about download and ingestion process.
    
    Provides a formatted summary of all parameters and configuration before
    starting the actual download/ingestion process. Useful for verification
    and logging purposes.
    
    Args:
        ngram_repo_url: Source repository URL.
        db_path: Target RocksDB database path.
        file_range: Tuple of (start_index, end_index) for files to process.
        file_urls_available: Complete list of available download URLs.
        file_urls_to_use: Subset of URLs that will actually be processed.
        ngram_size: Size of ngrams being processed.
        workers: Number of parallel worker processes/threads.
        executor_name: Type of executor being used ("threads" or "processes").
        start_time: Timestamp when the process began.
        ngram_type: Type of ngram filtering ('all', 'tagged', 'untagged').
        overwrite: Whether overwriting existing database or resuming.
        files_to_skip: Number of files being skipped (already processed).
    """
    print(f'\033[31mStart Time: {start_time}\n\033[0m')
    print('\033[4mDownload & Ingestion Configuration\033[0m')
    print(f'Ngram repository:           {ngram_repo_url}')
    print(f'RocksDB database path:      {db_path}')  # noqa: T201 (intentional user output)
    print(f'File index range:           {file_range[0]} to {file_range[1]}')
    print(f'Total files available:      {len(file_urls_available)}')
    print(f'Files to process:           {len(file_urls_to_use)}')
    first_url = file_urls_to_use[0] if file_urls_to_use else "None"
    print(f'First file URL:             {first_url}')
    last_url = file_urls_to_use[-1] if file_urls_to_use else "None"
    print(f'Last file URL:              {last_url}')
    print(f'Ngram size:                 {ngram_size}')
    print(f'Ngram filtering:            {ngram_type}')
    print(f'Overwrite mode:             {overwrite}')
    if files_to_skip > 0:
        print(f'Files to skip (processed):  {files_to_skip}')
    print(f'Worker processes/threads:   {workers} ({executor_name})')
    print(f'Write batch size:           {write_batch_size:,}')
    print()


def process_files_simple(
    worker_args: List[Tuple[str, int, Optional[str]]],
    executor_class,
    workers: int,
    db: rocksdict.Rdict,
    write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
    ) -> Tuple[List[str], List[str], int, int]:
    """Process files concurrently; return (success, failure, entries, batches)."""
    successful_files: List[str] = []
    failed_files: List[str] = []
    total_entries_written = 0
    write_batches = 0
    pending_data: Dict[str, bytes] = {}

    with tqdm(total=len(worker_args), desc="Processing Files", unit='files', colour='blue') as pbar:
        with executor_class(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_and_ingest_file, url, worker_id, pattern
                ): (url, worker_id, pattern)
                for (url, worker_id, pattern) in worker_args
            }
            for future in as_completed(futures):
                url, worker_id, _pattern = futures[future]
                filename = url.split('/')[-1]
                try:
                    result_msg, parsed_data = future.result()
                    if result_msg.startswith("SUCCESS"):
                        successful_files.append(result_msg)
                        mark_file_as_processed(db, filename)
                        pending_data.update(parsed_data)
                        if len(pending_data) >= write_batch_size:
                            entries_written = write_batch_to_db(db, pending_data)
                            total_entries_written += entries_written
                            write_batches += 1
                            pending_data.clear()
                        logging.info(f"Processed: {filename}")
                    else:
                        failed_files.append(result_msg)
                    pbar.update(1)
                except Exception as exc:  # noqa: BLE001
                    msg = f"ERROR: {filename} - {exc}"
                    failed_files.append(msg)
                    logging.error(msg)
                    pbar.update(1)

    # Final flush
    if pending_data:
        entries_written = write_batch_to_db(db, pending_data)
        total_entries_written += entries_written
        write_batches += 1
        pending_data.clear()

    return successful_files, failed_files, total_entries_written, write_batches


def download_and_ingest_to_rocksdb(
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str,
    db_path: str,
    file_range: Optional[Tuple[int, int]] = None,
    workers: Optional[int] = None,
    use_threads: bool = False,
    ngram_type: str = 'all',
    overwrite: bool = True,
    random_seed: Optional[int] = None,
    write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
) -> None:
    """Main function to download Google ngram files and ingest into RocksDB.
    """
    # Label main process for htop
    try:
        import setproctitle
        setproctitle.setproctitle("PROC_MAIN")
    except ImportError:
        pass
    """Main function to download Google ngram files and ingest into RocksDB.
    
    Orchestrates the complete process of discovering, downloading, and
    ingesting Google Books ngram data. Uses parallel processing to handle
    multiple files simultaneously for optimal performance.
    
    Process overview:
        1. Construct repository URL and discover available files
        2. Select subset of files based on file_range parameter
        3. Create/configure RocksDB database with optimized settings
        4. Launch parallel workers to download and ingest files
        5. Report completion statistics
    
    Args:
        ngram_size: The ngram size to download (1-5). 1=unigrams, 
            2=bigrams, etc.
        repo_release_id: Google's release identifier (e.g., "20200217").
        repo_corpus_id: Corpus identifier (e.g., "eng-us-all", "eng-fiction").
        db_path: Absolute path where RocksDB database will be created/updated.
        file_range: Optional tuple (start_idx, end_idx) to process subset
            of files. If None, processes all available files.
        workers: Number of parallel worker processes/threads. If None,
            uses CPU count.
        use_threads: If True, uses ThreadPoolExecutor (better for I/O-bound
            tasks). If False, uses ProcessPoolExecutor (better for CPU-bound
            tasks). Default False for backward compatibility.
        ngram_type: Type of ngrams to include for corpus size reduction:
            - 'all': Include all ngrams (no filtering, default)
            - 'tagged': Only include POS-tagged ngrams (e.g., "word_NOUN")
            - 'untagged': Only include untagged ngrams (plain words)
        overwrite: Whether to overwrite existing database and reprocess all files.
            - True: Remove existing database and start fresh (default)
            - False: Resume processing, skip files that have already been processed
        random_seed: Optional seed for randomizing file processing order.
            If None, files are processed in original order.
        
    Raises:
        RuntimeError: If file discovery fails or no files are found.
        OSError: If database directory cannot be created.
        ValueError: If ngram_type is invalid.
        
    Example:
        # Download bigrams from files 0-9 using 4 workers with threads
        download_and_ingest_to_rocksdb(
            ngram_size=2,
            repo_release_id="20200217", 
            repo_corpus_id="eng-us-all",
            db_path="/data/ngrams.db",
            file_range=(0, 9),
            workers=4,
            use_threads=True,
            ngram_type='tagged',  # Only POS-tagged ngrams
            overwrite=False,     # Resume from previous session
            random_seed=42       # Randomize file order for load balancing
        )
    """
    # Initialize timing and logging
    start_time = datetime.now()
    # Value payload now handled in process_and_ingest_file
    # Cap at 40 workers for stability
    if workers is None:
        cpu_count = os.cpu_count() or 4  # Default to 4 if None
        workers = min(40, cpu_count * 2)

    # Setup file-based logging
    setup_logging(db_path)

    # Start RocksDB log monitor (background thread) after logging setup
    try:
        from utils.rocksdb_log_monitor import start_rocksdb_log_monitor
        db_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "."
        log_path = os.path.join(db_dir, os.path.basename(db_path), "LOG")
        if not os.path.exists(log_path):
            # Try LOG in db_dir if not found in db_path
            log_path = os.path.join(db_dir, "LOG")
        start_rocksdb_log_monitor(log_path, poll_interval=10, logger=logging.getLogger())
    except Exception as e:
        logging.warning(f"Could not start RocksDB log monitor: {e}")
    
    # Handle database initialization based on overwrite setting
    if overwrite:
        # Reset database - remove existing database safely with NFS handling
        if os.path.exists(db_path):
            logging.info("Removing existing database for fresh start...")
            if not cleanup_database_safely(db_path):
                raise RuntimeError(f"Failed to remove existing database at {db_path}. "
                                 "Try manually removing the directory or check for open file handles.")
            logging.info("Successfully removed existing database")
    else:
        # Resume mode - check if database exists
        if os.path.exists(db_path):
            logging.info("Resume mode: Using existing database")
        else:
            logging.info("Resume mode: No existing database found, creating new one")
    
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir:  # Only create if path has a directory component
        os.makedirs(db_dir, exist_ok=True)
    
    # Discover available files from Google's repository
    ngram_repo_url, file_pattern = set_location_info(
        ngram_size, repo_release_id, repo_corpus_id
    )
    file_urls_available = fetch_file_urls(ngram_repo_url, file_pattern)
    
    if not file_urls_available:
        raise RuntimeError("No ngram files found in repository")
    
    # Determine which files to process
    if file_range is None:
        file_range = (0, len(file_urls_available) - 1)
    
    # Validate file range
    start_idx, end_idx = file_range
    if (start_idx < 0 or end_idx >= len(file_urls_available) or 
            start_idx > end_idx):
        raise ValueError(
            f"Invalid file range {file_range}. Available files: 0-"
            f"{len(file_urls_available) - 1}"
        )
    
    file_urls_to_use = file_urls_available[start_idx:end_idx + 1]
    
    # Handle resume mode: filter out already processed files
    files_to_skip = 0
    if not overwrite:
        # Open database to check for processed files
        temp_db = setup_rocksdb(db_path)
        processed_files = get_processed_files(temp_db)
        temp_db.close()
        
        if processed_files:
            # Filter out URLs for files that have already been processed
            original_count = len(file_urls_to_use)
            file_urls_to_use = [
                url for url in file_urls_to_use 
                if url.split('/')[-1] not in processed_files
            ]
            files_to_skip = original_count - len(file_urls_to_use)
            
            if files_to_skip > 0:
                logging.info(f"Resume mode: Skipping {files_to_skip} processed files")
            
            if not file_urls_to_use:
                print("🎉 All files in the specified range have already been processed!")
                return
    
    # Optional randomization
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(file_urls_to_use)
        logging.info(f"Randomized file order with seed {random_seed}")
    
    # Choose executor type based on workload characteristics
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    executor_name = "threads" if use_threads else "processes"
    
    # Display configuration information
    print_info(
        ngram_repo_url, db_path, file_range, file_urls_available,
        file_urls_to_use, ngram_size, workers, executor_name, start_time,
        ngram_type=ngram_type, overwrite=overwrite, files_to_skip=files_to_skip, write_batch_size=write_batch_size
    )
    
    # Create filter regex for ngram type filtering (store pattern string for workers)
    filter_regex = define_regex(ngram_type)
    filter_pattern = filter_regex.pattern if filter_regex else None
    logging.info(f"Ngram filtering: {ngram_type}")
        
    # Process files with periodic database writes to manage memory
    # Use modulo to assign logical worker IDs that match actual worker count
    worker_args: List[Tuple[str, int, Optional[str]]] = [
        (url, i % workers, filter_pattern) for i, url in enumerate(file_urls_to_use)
    ]

    # Open database connection once for the entire process
    db = setup_rocksdb(db_path)

    successful_files, failed_files, total_entries_written, write_batches = process_files_simple(
        worker_args=worker_args,
        executor_class=executor_class,
        workers=workers,
        db=db,
        write_batch_size=write_batch_size,
    )
    
    # Close database connection
    db.close()
    
    # Calculate and display completion statistics
    end_time = datetime.now()
    total_runtime = end_time - start_time
    
    # Calculate time per file
    successful_count = len(successful_files)
    actual_failed_count = len(failed_files)
    
    if successful_count > 0:
        time_per_file = total_runtime / successful_count
        if time_per_file.total_seconds() > 0:
            files_per_hour = 3600 / time_per_file.total_seconds()
        else:
            files_per_hour = 0
    else:
        time_per_file = total_runtime
        files_per_hour = 0
    
    print(f'\033[32m\nProcessing completed!\033[0m')
    print(f'Fully processed files: {successful_count}')
    if actual_failed_count > 0:
        print(f'\033[31mFailed files: {actual_failed_count}\033[0m')
    print(f'Total entries written: {total_entries_written:,}')
    print(f'Write batches flushed: {write_batches}')
    print(f'\033[31m\nEnd Time: {end_time}\033[0m')
    print(f'\033[31mTotal Runtime: {total_runtime}\033[0m')
    print(f'\033[34m\nTime per file: {time_per_file}\033[0m')
    print(f'\033[34mFiles per hour: {files_per_hour:.1f}\033[0m')
