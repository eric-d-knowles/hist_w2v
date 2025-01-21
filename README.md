# **hist_w2v**:

Tools for downloading, processing, and training word2vec models on Google Ngrams
## Version 0.1.0

I wanted to study the evolution of group stereotypes over time using Google Ngrams corpora, but wasn't satisfied with the existing tools I found online. So, I created a Python package to streamline the process of (1) downloading and pre-processing raw ngrams and (2) training and evaluating `word2vec` models on the ngrams. After installing, the best way to learn how to use these tools is to work through the provided Jupyter Notebook workflows.

## Package Contents
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

`notebooks`
1. `workflow_unigrams.ipynb`: Jupyter Notebook showing how to download and preprocess unigrams.
2. `workflow_multigrams.ipynb`: Jupyter Notebook showing how to download and preprocess multigrams.
3. `workflow_training.ipynb`: Jupyter Notebook showing how to train, evaluate, and plots results from `word2vec` models.

Finally, the `training_results` folder is where a file containing evaluation metrics for a set of models is stored. 

## System Requirements
Unless you have an extremely powerful personal computer, the code is probably only suitable to run on a high-performance computing (HPC) cluster; efficiently downloading, processing, and training models on ngrams in parallel takes lots of processors and memory. On my university's HPC, I typically request 14 cores and 128G of RAM. A priority for future development is to streamline the code for individual systems.

## Citing hist_w2v
If you use `hist_w2v` in your research or other publications, I kindly ask you to cite it. Used the GitHub citation to create citation text.

## License

This project is released under the [MIT License](https://github.com/eric-d-knowles/hist_w2v/blob/main/LICENSE).
