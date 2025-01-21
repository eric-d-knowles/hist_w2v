# **hist_w2v**: Tools for Training Word2Vec Models on Google Ngrams

I wanted to study the evolution of group stereotypes over time using Google Ngrams corpora, but wasn't satisfied with the existing tools I found online. So, I created a Python package to streamline the process of (1) downloading and pre-processing raw ngrams and (2) training and evluating `word2vec` models on the ngrams. The library consists of the following:

`src/ngram_tools`
1. `downoad_ngrams.py`: downloads the desired ngram types (e.g., 3-grams with part-of-speech [POS] tags, 5-grams with POS tags).
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
