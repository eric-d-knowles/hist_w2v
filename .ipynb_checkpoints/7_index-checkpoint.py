import os
import argparse
import heapq
from datetime import datetime
from tqdm import tqdm
import tempfile
import multiprocessing
import orjson
import shutil
import lz4.frame


FIXED_DESC_LENGTH = 15
BAR_FORMAT = (
    "{desc:<15} |{bar:50}| {percentage:5.1f}% {n_fmt:<12}/{total_fmt:<12} \
    [{elapsed}<{remaining}, {rate_fmt}]"
)


class FileHandler:
    def __init__(self, path, is_output=False, compress=False):
        self.path = path
        self.compress = compress if is_output else path.endswith('.lz4')
        self.binary_mode = self.compress
        self.mode = (
            'wb' if is_output and self.compress else
            'w' if is_output else
            'rb' if self.compress else
            'r'
        )
        self.encoding = None if self.binary_mode else 'utf-8'
        self.open_fn = lz4.frame.open if self.compress else open

    def open(self):
        if self.compress:
            return self.open_fn(self.path, self.mode)
        return self.open_fn(self.path, self.mode, encoding=self.encoding)

    def serialize(self, entry):
        serialized = orjson.dumps(entry)
        return serialized + b'\n' if self.binary_mode else serialized.decode(
            'utf-8'
        ) + '\n'

    def deserialize(self, line):
        return orjson.loads(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort consolidated file by freq_tot in descending order, "
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
    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        help='Path to the consolidated file (required).')
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing files (default=False).')
    parser.add_argument('--workers',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of processors to use (default=all).')
    parser.add_argument('--vocab_file',
                        type=int,
                        help='Produce vocab_list_match and vocab_list_lookup '
                             'with the top N ngrams.')

    return parser.parse_args()


def print_info(proj_dir, output_dir, input_file, ngram_size, workers,
               overwrite, vocab_n, start_time):
    print(f'\033[31mStart Time:                {start_time}\n\033[0m')
    print('\033[4mIndexing Info\033[0m')
    print(f'Project directory:         {proj_dir}')
    print(f'Output directory:          {output_dir}')
    print(f'Input file:                {input_file}')
    print(f'Ngram size:                {ngram_size}')
    print(f'Overwrite existing files:  {overwrite}')
    print(f'Workers:                   {workers}')
    if vocab_n is not None:
        print(f'Vocab size (top N):        {vocab_n}')
    print('\n')


def create_progress_bar(total, description, unit=''):
    padded_desc = description.ljust(FIXED_DESC_LENGTH)
    return tqdm(
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


def external_sort_descending_by_freq(input_file, workers, chunk_size=100000):
    """
    External sort the input file by freq_tot in descending order.
    """
    tmpdir = tempfile.mkdtemp()

    # --- Step 1: Count total lines ---
    total_lines = 0
    in_count_handler = FileHandler(input_file, is_output=False)
    with in_count_handler.open() as count_f:
        for _ in count_f:
            total_lines += 1

    # --- Step 2: Read in chunks and spawn parallel sorts ---
    chunk_paths = []
    current_chunk = []
    chunk_idx = 0

    print('\033[4mIndexing Info\033[0m')

    with FileHandler(input_file, is_output=False).open() as infile, \
         create_progress_bar(total_lines, "Chunking") as pbar:
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

    # --- Step 3: Sort chunks in parallel ---
    with multiprocessing.Pool(processes=workers) as pool:
        with create_progress_bar(
            len(chunk_paths), "Sorting", 'chunks'
        ) as pbar:
            sorted_chunk_paths = []
            for chunk_result in pool.imap(chunk_sort, chunk_paths):
                sorted_chunk_paths.append(chunk_result)
                pbar.update(1)

    # --- Step 4: Merge chunks with heapq.merge ---
    # heapq.merge merges ascending; we invert freq_tot by storing -freq_tot
    def file_generator(fp):
        gen_handler = FileHandler(fp, is_output=False)
        with gen_handler.open() as f:
            for line in f:
                obj = gen_handler.deserialize(line)
                # Just yield the dict
                yield obj

    generators = [file_generator(fp) for fp in sorted_chunk_paths]

    merged_sorted_path = input_file.replace('.jsonl.lz4', '-desc.jsonl')

    # Overwrite if necessary.
    if os.path.exists(merged_sorted_path):
        os.remove(merged_sorted_path)

    out_handler = FileHandler(merged_sorted_path, is_output=True,
                              compress=False)
    with out_handler.open() as outfile, \
         create_progress_bar(total_lines, "Merging", "lines") as pbar:

        # Merge in descending order by freq_tot
        for obj in heapq.merge(*generators, key=lambda x: x['freq_tot'],
                               reverse=True):
            outfile.write(out_handler.serialize(obj))
            pbar.update(1)

    # Clean up temp
    shutil.rmtree(tmpdir)
    return merged_sorted_path


def index_ngrams(sorted_file, output_dir):
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
    out_handler = FileHandler(indexed_path, is_output=True, compress=False)

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


def create_vocab_files(indexed_path, vocab_n):
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
    vocab_match_path = indexed_path.replace('-indexed.jsonl',
                                            '-vocab_list_match.txt')
    out_match_handler = FileHandler(vocab_match_path, is_output=True,
                                    compress=False)
    with out_match_handler.open() as txtfile:
        for entry in top_ngrams:
            # 'ngram' is now just a string
            txtfile.write(entry['ngram'] + '\n')

    # Write lookup file
    vocab_lookup_path = indexed_path.replace('-indexed.jsonl',
                                             '-vocab_list_lookup.jsonl')
    out_lookup_handler = FileHandler(vocab_lookup_path, is_output=True,
                                     compress=False)
    with out_lookup_handler.open() as jsonfile:
        for entry in top_ngrams:
            out_entry = {
                "ngram": entry['ngram'],
                "freq_tot": entry['freq_tot'],
                "idx": entry['idx']
            }
            jsonfile.write(out_lookup_handler.serialize(out_entry))


def main():
    args = parse_args()

    proj_dir = args.proj_dir
    ngram_size = args.ngram_size
    input_file = args.input_file
    overwrite = args.overwrite
    workers = args.workers
    vocab_n = args.vocab_file

    start_time = datetime.now()

    output_dir = os.path.join(proj_dir, f'{ngram_size}gram_files', '6corpus')
    if not os.path.exists(output_dir):
        raise NotADirectoryError(
            f"Output directory {output_dir} does not exist."
        )

    print_info(
        proj_dir, output_dir, input_file, ngram_size, workers,
        overwrite, vocab_n, start_time
    )

    # 1) Sort descending by freq_tot
    sorted_desc_file = external_sort_descending_by_freq(input_file,
                                                        workers=workers)

    # 2) Index the ngrams
    indexed_file, count = index_ngrams(sorted_desc_file, output_dir)

    # 3) Optionally, create vocab files
    if vocab_n is not None and vocab_n > 0:
        create_vocab_files(indexed_file, vocab_n)

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
    main()