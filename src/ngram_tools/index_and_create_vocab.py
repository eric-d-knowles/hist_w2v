import os
import sys
import argparse
import heapq
from datetime import datetime
from tqdm import tqdm
import tempfile
import multiprocessing
import orjson
import shutil


from ngram_tools.helpers.file_handler import FileHandler


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} \
    [{elapsed}<{remaining}, {rate_fmt}]"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort corpus file by freq_tot in descending order, "
                    "index the ngrams, and optionally create a vocab file."
    )

    parser.add_argument('--proj_dir',
                        type=str,
                        required=True,
                        help='Base project directory (required).')
    parser.add_argument('--ngram_size',
                        type=int,
                        required=True,
                        help='Size of the ngrams (required).')
    parser.add_argument('--compress',
                        action='store_true',
                        default=False,
                        help='Compress saved files?')
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing files (default=False).')
    parser.add_argument('--workers',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of processors to use (default=all).')
    parser.add_argument('--vocab_n',
                        type=int,
                        help='Produce vocab_list_match and vocab_list_lookup '
                             'with the top N ngrams.')

    return parser.parse_args()


def get_corpus_path(corpus_dir):
    """
    Look for exactly one file containing '-corpus' in its name within
    `corpus_dir`. Returns the full path to that file if found, otherwise
    prints an error and exits.
    """
    # Collect all files containing '--corpus' in the name
    corpus_files = [
        f for f in os.listdir(corpus_dir)
        if '-corpus' in f and os.path.isfile(
            os.path.join(corpus_dir, f)
        )
    ]

    if len(corpus_files) == 0:
        print("Error: No file with '-corpus' found in the directory:")
        print(f"  {corpus_dir}")
        sys.exit(1)
    elif len(corpus_files) > 1:
        print("Error: Multiple files with '-corpus' were found. "
              "The script doesn't know which one to use:")
        for file_name in corpus_files:
            print(f"  {file_name}")
        sys.exit(1)
    else:
        # Exactly one matching file
        return os.path.join(corpus_dir, corpus_files[0])


def set_info(proj_dir, ngram_size, compress):
    corpus_dir = os.path.join(
        proj_dir,
        f'{ngram_size}gram_files/6corpus'
    )
    corpus_path = get_corpus_path(corpus_dir)

    indexed_path = os.path.join(
        corpus_dir, f"{ngram_size}gram-indexed.jsonl" + (
            '.lz4' if compress else ''
        )
    )

    match_path = os.path.join(
        corpus_dir, f"{ngram_size}1ram-corpus-vocab_list_match.txt"
    )

    lookup_path = os.path.join(
        corpus_dir, f"{ngram_size}gram-corpus-vocab_list_lookup.jsonl"
    )


    return (corpus_path, indexed_path, match_path, lookup_path)


def print_info(start_time, corpus_path, indexed_path, ngram_size,
               workers, compress, overwrite, vocab_n, match_path,
               lookup_path):
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mIndexing Info\033[0m')
    print(f'Corpus file:               {corpus_path}')
    print(f'Indexed file:              {indexed_path}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Number of workers:         {workers}')
    print(f'Compress output files:     {compress}')
    print(f'Overwrite existing files:  {overwrite}')
    if vocab_n is not None and vocab_n > 0:
        print(f'Vocab size (top N):        {vocab_n}')
        print(f'Match File:                {match_path}')
        print(f'Lookup File:               {lookup_path}')


def create_progress_bar(total, description, unit=''):
    padded_desc = description.ljust(FIXED_DESC_LENGTH)
    return tqdm(
        file=sys.stdout,
        total=total,
        desc=padded_desc,
        leave=True,
        dynamic_ncols=False,
        ncols=100,
        unit=unit,
        colour='green',
        bar_format=BAR_FORMAT
    )


def chunk_sort(args):
    """
    Worker function to sort a chunk of data by freq_tot descending.
    Each element in chunk_lines is JSON string (possibly bytes if compressed).
    We'll parse with orjson or via FileHandler's deserialize method.
    """
    chunk_lines, chunk_idx, tmpdir = args

    # Convert each line to a Python object, storing (freq_tot, data)
    entries = []
    for line in chunk_lines:
        if isinstance(line, bytes):
            # If chunk_lines came from a compressed source, line is bytes
            data = orjson.loads(line)
        else:
            # Otherwise, line is str
            data = orjson.loads(line)
        freq_tot = data['freq_tot']
        entries.append((freq_tot, data))

    # Sort by freq_tot descending
    entries.sort(key=lambda x: x[0], reverse=True)

    # Write sorted lines using FileHandler (uncompressed)
    sorted_chunk_path = os.path.join(tmpdir, f'sorted_chunk_{chunk_idx}.jsonl')
    out_handler = FileHandler(sorted_chunk_path, is_output=True,
                              compress=False)
    with out_handler.open() as out_f:
        for _, obj in entries:
            out_f.write(out_handler.serialize(obj))

    return sorted_chunk_path


def external_sort_descending_by_freq(
    corpus_path,
    workers,
    chunk_size=100000,
    compress=False
):
    """
    External sort the input file by freq_tot in descending order.
    """
    tmpdir = tempfile.mkdtemp()

    # Step 1: Count total lines
    total_lines = 0
    in_count_handler = FileHandler(corpus_path, is_output=False)
    with in_count_handler.open() as count_f:
        for _ in count_f:
            total_lines += 1

    # Step 2: Read in chunks and spawn parallel sorts
    chunk_paths = []
    current_chunk = []
    chunk_idx = 0

    in_handler = FileHandler(corpus_path, is_output=False)
    out_handler = FileHandler(corpus_path, is_output=True, compress=True)

    with in_handler.open() as infile, create_progress_bar(
        total_lines, 'Chunking', 'lines'
    ) as pbar:
        for line in infile:
            current_chunk.append(line)
            pbar.update(1)
            if len(current_chunk) >= chunk_size:
                chunk_paths.append((current_chunk, chunk_idx, tmpdir))
                current_chunk = []
                chunk_idx += 1
        # Handle leftover lines
        if current_chunk:
            chunk_paths.append((current_chunk, chunk_idx, tmpdir))

    # Step 3: Sort chunks in parallel
    with multiprocessing.Pool(processes=workers) as pool:
        with create_progress_bar(
            len(chunk_paths), 'Sorting', 'chunks'
        ) as pbar:
            sorted_chunk_paths = []
            for chunk_result in pool.imap(chunk_sort, chunk_paths):
                sorted_chunk_paths.append(chunk_result)
                pbar.update(1)

    # Step 4: Merge chunks with heapq.merge
    def file_generator(fp):
        gen_handler = FileHandler(fp, is_output=False)
        with gen_handler.open() as f:
            for line in f:
                obj = gen_handler.deserialize(line)
                yield obj

    generators = [file_generator(fp) for fp in sorted_chunk_paths]

    merged_sorted_path = corpus_path.replace('.jsonl', '-desc.jsonl')

    # Overwrite if necessary
    if os.path.exists(merged_sorted_path):
        os.remove(merged_sorted_path)

    out_handler = FileHandler(
        merged_sorted_path, is_output=True, compress=compress)
    with out_handler.open() as outfile, \
         create_progress_bar(total_lines, "Merging", "lines") as pbar:

        # Merge in descending order by freq_tot
        for obj in heapq.merge(
            *generators, key=lambda x: x['freq_tot'], reverse=True
        ):
            outfile.write(out_handler.serialize(obj))
            pbar.update(1)

    # Clean up temp
    shutil.rmtree(tmpdir)

    return merged_sorted_path


def index_ngrams(
    sorted_file,
    indexed_path,
    compress=False
):
    """
    Index the ngrams in the order they appear in the reverse-sorted file.
    We read line by line, parse to dict, add an 'idx', and write to a new file.
    """
    indexed_path = sorted_file.replace('-desc.jsonl', '-indexed.jsonl')

    # Count lines
    total_lines = 0
    count_handler = FileHandler(sorted_file, is_output=False)
    with count_handler.open() as count_f:
        for _ in count_f:
            total_lines += 1

    line_count = 0
    in_handler = FileHandler(sorted_file, is_output=False)
    out_handler = FileHandler(indexed_path, is_output=True, compress=compress)

    with in_handler.open() as infile, \
         out_handler.open() as outfile, \
         create_progress_bar(
             total_lines,
             "Indexing",
             "lines"
         ) as pbar:
        idx = 1
        for line in infile:
            obj = in_handler.deserialize(line)
            obj['idx'] = idx
            idx += 1
            line_count += 1
            pbar.update(1)

            outfile.write(out_handler.serialize(obj))

    return indexed_path, line_count


def create_vocab_files(
    indexed_path,
    vocab_n,
    match_path,
    lookup_path
):
    """
    Create two output files for the top `vocab_n` ngrams:
      - vocab_list_match.txt  (ngram as text)
      - vocab_list_lookup.jsonl (ngram + freq_tot + idx)
    """
    top_ngrams = []

    # Read the indexed file
    in_handler = FileHandler(indexed_path, is_output=False)
    with in_handler.open() as f:
        for i, line in enumerate(f):
            if i >= vocab_n:
                break
            obj = in_handler.deserialize(line)
            top_ngrams.append(obj)

    # Write match file
    out_match_handler = FileHandler(match_path, is_output=True,
                                    compress=False)
    with out_match_handler.open() as txtfile:
        for entry in top_ngrams:
            # 'ngram' is now just a string
            txtfile.write(entry['ngram'] + '\n')

    # Write lookup file
    out_lookup_handler = FileHandler(lookup_path, is_output=True,
                                     compress=False)
    with out_lookup_handler.open() as jsonfile:
        for entry in top_ngrams:
            out_entry = {
                "ngram": entry['ngram'],
                "freq_tot": entry['freq_tot'],
                "idx": entry['idx']
            }
            jsonfile.write(out_lookup_handler.serialize(out_entry))


def index_and_create_vocab_files(
    proj_dir,
    ngram_size,
    compress=False,
    overwrite=False,
    workers=os.cpu_count(),
    vocab_n=0
):
    start_time = datetime.now()

    (corpus_path, indexed_path, match_path, lookup_path) = set_info(
        proj_dir, ngram_size, compress
    )

    print_info(
        start_time,
        corpus_path,
        indexed_path,
        ngram_size,
        workers,
        compress,
        overwrite,
        vocab_n
    )

    # 1) Sort descending by freq_tot
    sorted_desc_file = external_sort_descending_by_freq(
        corpus_path,
        workers=workers,
        compress=compress
    )

    # 2) Index the ngrams
    indexed_file, count = index_ngrams(
        sorted_desc_file,
        indexed_path,
        compress
    )

    # 3) Optionally, create vocab files
    if vocab_n is not None and vocab_n > 0:
        create_vocab_files(indexed_file, vocab_n, match_path, lookup_path)

    # Remove the intermediate descending-sorted file
    if os.path.exists(sorted_desc_file):
        os.remove(sorted_desc_file)

    end_time = datetime.now()
    print(f"\nIndexed {count} lines.")
    print(f"Final indexed file: {os.path.basename(indexed_file)}")
    if vocab_n:
        print("Created vocab_list_match and vocab_list_lookup files for top "
              f"{vocab_n} ngrams.")

    end_time = datetime.now()
    print(f'\033[31m\nEnd Time:                  {end_time}\033[0m')

    total_runtime = end_time - start_time
    print(f'\033[31mTotal runtime:             {total_runtime}\n\033[0m')


if __name__ == "__main__":
    args = parse_args()
    index_and_create_vocab_files(
        proj_dir=args.proj_dir,
        ngram_size=args.ngram_size,
        compress=args.compress,
        overwrite=args.overwrite,
        workers=args.workers,
        vocab_n=args.vocab_n
    )