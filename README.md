# hist_w2v

## Tools for downloading, processing, and training word2vec models on Google Ngrams
Python package to assist researchers in using Google Ngrams to examine semantic change over years, decades, and centuries. `hist_w2v` automates _downloading and pre-processing_ raw ngrams and _training_ `word2vec` models on a corpus.

### Installation
There are two ways to install `hist_w2v`:

1. Clone the GitHub repository (https://github.com/eric-d-knowles/hist_w2v) to your Python environmen t.
2. Install from PyPI.org by running `pip install hist_w2v` in your Python environment.

After installing `hist_w2v`, the best way to learn how to use it by working through the provided Jupyter Notebook workflows. Together, these notebooks provide a fully documented, end-to-end illustration of the package's functionality. 

### Package Contents
The library consists of the following modules and notebooks:

`src/ngram_tools`
1. `downoad_ngrams.py`: downloads the desired ngram types (e.g., 3-grams with part-of-speech [POS] tags, 5-grams without POS tags).
2. `convert_to_jsonl.py`: converts the raw-text ngrams from Google into a more flexible JSONL format.
3. `lowercase_ngrams.py`: makes the ngrams all lowercase.
4. `lemmatize_ngrams.py`: lemmatizes the ngrams (i.e., reduce them to their base grammatical forms).
5. `filter_ngrams.py`: screens out undesired tokens (e.g., stop words, numbers, words not in a vocabulary file) from the ngrams.
6. `sort_ngrams.py`: combines multiple ngrams files into a single sorted file.
7. `consolidate_ngrams.py`: consolidates duplicate ngrams resulting from the previous steps.
8. `index_and_create_vocabulary.py`: numerically indexes a list of unigrams and create a "vocabulary file" to screen multigrams.
9. `create_yearly_files.py`: splits the master corpus into yearly sub-corpora.
10. `helpers/file_handler.py`: helper script to simplify reading and writing files in the other modules.
11. `helpers/print_jsonl_lines.py`: helper script to view a snippet of ngrams in a JSONL file.
12. `helpers/verify_sort.py`: helper script to confirm whether an ngram file is properly sorted. 

`src/training_tools`
1. `train_ngrams.py`: train `word2vec` models on pre-processed multigram corpora.
2. `evaluate_models.py`: evaluate training quality on intrinsic benchmarks (i.e., similarity and analogy tests).
3. `plotting.py`: plot various types of model results.
4. `w2v_model.py`: a Python class (`W2VModel`) to aid in the evaluation, normalization, and alignment of yearly `word2vec` models 

`notebooks`
1. `workflow_unigrams.ipynb`: Jupyter Notebook showing how to download and preprocess _unigrams_.
2. `workflow_multigrams.ipynb`: Jupyter Notebook showing how to download and preprocess _multigrams_.
3. `workflow_training.ipynb`: Jupyter Notebook showing how to train, evaluate, and plots results from `word2vec` models.

Finally, the `training_results` folder is where a file containing evaluation metrics for a set of models is stored. 

### System Requirements
Efficiently downloading, processing, and training models on ngrams takes lots of processors and memory. Unless you have a very powerful PC, you should only try to run `hist_w2v` on a high-performance computing (HPC) cluster or similar platform. On my university's HPC, I typically request 14 cores and 128G of RAM. A priority for development is refactoring the code for individual systems.

### Citing hist_w2v
If you use `hist_w2v` in your research or other publications, I kindly ask you to cite it. Use the GitHub citation to create citation text.

### License

This project is released under the [MIT License](https://github.com/eric-d-knowles/hist_w2v/blob/main/LICENSE).
