"""
optimize_buckets.py

Histogram and bucket optimization utilities for ngram processing.
"""


import os
from glob import glob
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import string
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from ngram_tools.helpers.file_handler import FileHandler

def optimize_buckets(
    input_dir,
    bucket_depth=5,
    num_buckets=20,
    max_depth=3,
    limit_per_file=None,
    show_plot=False,
    show_buckets=False,
    workers=None,
    save_path=None
):
    """
    Run the full pipeline: build histogram, partition buckets, and optionally plot.

    Args:
        input_dir (str): Directory containing input files to process.
        bucket_depth (int): Number of tokens to use for prefix/bucket key.
        num_buckets (int): Desired number of buckets.
        max_depth (int): Maximum depth for partitioning.
        limit_per_file (int, optional): Max number of lines to scan per file.
        show_plot (bool): Whether to show the plot. Defaults to False.
        show_buckets (bool): Whether to print aggregated buckets. Defaults to False.
        workers (int, optional): Number of parallel workers for histogram scan.

    Returns:
        dict: {'buckets': list, 'histogram': Counter}

    Example:
        result = optimize_buckets(
            input_dir="/path/to/ngram/files",
            bucket_depth=2,
            num_buckets=20,
            max_depth=5,
            show_plot=True,
            workers=8
        )
    """
    input_paths = sorted(
        glob(os.path.join(input_dir, '*.jsonl')) +
        glob(os.path.join(input_dir, '*.jsonl.lz4'))
    )
    hist = histogram_prefixes_across_files(
        input_paths,
        bucket_depth=bucket_depth,
        limit_per_file=limit_per_file,
        workers=workers
    )
    buckets = partition_prefix_tree(hist, num_buckets, max_depth)
    if save_path is not None:
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(buckets, f, ensure_ascii=False, indent=2)
    if show_plot:
        plot_partitioned_buckets(hist, max_depth, num_buckets)
    print(f"Total buckets created: {len(buckets)}")
    if show_buckets:
        print('Aggregated buckets (alphabetical order):')
        for prefix, size in sorted(buckets, key=lambda x: x[0]):
            print(f'Prefix: {prefix if prefix else "<root>"}, Total ngrams: {size:,}')
    return buckets


def build_prefix_tree(hist):
    """
    Build a prefix tree (trie) from a histogram of ngram prefixes.
    Args:
        hist (dict): Mapping from prefix to count.
    Returns:
        defaultdict: Nested dictionary representing the prefix tree.
    """
    tree = defaultdict(dict)
    for prefix, count in hist.items():
        node = tree
        for c in prefix:
            node = node.setdefault(c, {})
        node['#'] = count
    return tree

def sum_tree(node):
    """
    Recursively sum counts in a prefix tree node.
    Args:
        node (dict): Node in the prefix tree.
    Returns:
        int: Total count for the subtree.
    """
    if '#' in node:
        return node['#']
    return sum(sum_tree(child) for k, child in node.items() if k != '#')

def recursive_partition(node, prefix, target, max_depth, buckets):
    """
    Recursively partition the prefix tree into buckets.
    Args:
        node (dict): Current node in the prefix tree.
        prefix (str): Current prefix string.
        target (float): Target bucket size.
        max_depth (int): Maximum depth for partitioning.
        buckets (list): List to append bucket tuples (prefix, total).
    """
    total = sum_tree(node)
    if (
        len(prefix) >= max_depth or
        total <= target or
        all('#' in child for k, child in node.items() if k != '#')
    ):
        buckets.append((prefix, total))
        return
    for k, child in node.items():
        if k == '#':
            continue
        recursive_partition(child, prefix + k, target, max_depth, buckets)

def partition_prefix_tree(hist, num_buckets, max_depth):
    """
    Partition a histogram into buckets using a prefix tree.
    Args:
        hist (dict): Mapping from prefix to count.
        num_buckets (int): Desired number of buckets.
        max_depth (int): Maximum depth for partitioning.
    Returns:
        list: List of (prefix, total) tuples for each bucket.
    """
    tree = build_prefix_tree(hist)
    total = sum(hist.values())
    target = total / num_buckets
    buckets = []
    recursive_partition(tree, '', target, max_depth, buckets)
    for _ in range(5):
        if len(buckets) > num_buckets:
            target *= 1.1
        elif len(buckets) < num_buckets:
            target *= 0.9
        else:
            break
        buckets = []
        recursive_partition(tree, '', target, max_depth, buckets)
    return buckets

def plot_partitioned_buckets(hist, max_depth, num_buckets):
    """
    Plot bucket sizes and boundaries for a partitioned prefix tree.
    Args:
        hist (dict): Mapping from prefix to count.
        max_depth (int): Maximum depth for partitioning.
        num_buckets (int): Desired number of buckets.
    """
    buckets = partition_prefix_tree(hist, num_buckets, max_depth)
    sorted_items = sorted(hist.items())
    bucket_sizes = [s for _, s in sorted_items]
    plt.figure(figsize=(12, 5))
    plt.plot(bucket_sizes, alpha=0.5, label='Fine buckets')
    bucket_indices = []
    bucket_labels = []
    sorted_labels = [b for b, _ in sorted_items]
    for prefix, size in buckets:
        idxs = [i for i, label in enumerate(sorted_labels) if label.startswith(prefix)]
        if idxs:
            idx = idxs[0]
            bucket_indices.append(idx)
            bucket_labels.append(prefix)
    for idx in bucket_indices:
        plt.axvline(idx, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Bucket (alphabetical order)')
    plt.ylabel('Number of ngrams in bucket')
    plt.title(f'Prefix Tree Buckets (max_depth={max_depth}, num_buckets={num_buckets})')
    plt.tight_layout()
    plt.show()

def histogram_ngram_prefixes(input_path, bucket_depth=1, limit=None):
    """
    Scan an input file and build a histogram of ngram prefixes (bucket keys).

    Args:
        input_path (str): Path to the input file.
        bucket_depth (int): Number of tokens to use for prefix/bucket key.
        limit (int, optional): Max number of lines to scan (for sampling).

    Returns:
        Counter: Mapping from bucket key to count.
    """
    def bucket_func(entry):
        ngram = entry['ngram']
        chars = []
        if isinstance(ngram, dict):
            for i in range(1, bucket_depth + 1):
                token = ngram.get(f'token{i}', None)
                if token and isinstance(token, str):
                    c = token.strip().lower()[:1]
                    chars.append(c if c in string.ascii_lowercase else '_')
                else:
                    chars.append('_')
        else:
            c = str(ngram).strip().lower()[:1]
            chars.append(c if c in string.ascii_lowercase else '_')
        return ''.join(chars)

    file_handler = FileHandler(input_path)
    counter = Counter()
    with file_handler.open() as infile:
        for i, line in enumerate(infile):
            if limit is not None and i >= limit:
                break
            entry = file_handler.deserialize(line)
            bucket = bucket_func(entry)
            counter[bucket] += 1
    return counter

def _histogram_worker(args):
    path, bucket_depth, limit_per_file = args
    return histogram_ngram_prefixes(
        path,
        bucket_depth=bucket_depth,
        limit=limit_per_file
    )

def histogram_prefixes_across_files(input_paths, bucket_depth=1, limit_per_file=None, workers=None):
    """
    Build a combined histogram of ngram prefixes across multiple input files.

    Args:
        input_paths (list[str]): List of input file paths.
        bucket_depth (int): Number of tokens to use for prefix/bucket key.
        limit_per_file (int, optional): Max number of lines to scan per file (for sampling).
        workers (int, optional): Number of parallel workers for histogram scan.

    Returns:
        Counter: Mapping from bucket key to count (across all files).
    """
    counter = Counter()
    args_list = [
        (path, bucket_depth, limit_per_file)
        for path in input_paths
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_histogram_worker, args)
            for args in args_list
        ]
        with tqdm(total=len(futures), desc="Scanning", unit="file") as pbar:
            for future in as_completed(futures):
                file_hist = future.result()
                counter.update(file_hist)
                pbar.update(1)
    return counter
