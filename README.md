# hist_w2v: Tools for Training Word2Vec Models on Google Ngrams

I wanted to study the evolution of group stereotypes over time using Google Ngrams corpora, but wasn't satisfied with the existing tools I found online. So, I created a Python package to streamline the process of (1) downloading and pre-processing raw ngrams and (2) training and evluating `word2vec` models on the ngrams. These are the scripts in the library:

`src/ngram_tools`
1. `downoad_ngrams.py`: downloads the desired ngram types (e.g., 3-grams with part-of-speech [POS] tags, 5-grams with POS tags).
2. `convert_to_jsonl.py`: converts the raw-text ngrams from Google into a more flexible JSONL format.
3. `lowercase_ngrams.py`: make the ngrams all lowercase.
4. `lemmatize_ngrams.py`: lemmatize the ngrams (i.e., reduce them to their base grammatical forms).
5. `filter_ngrams.py`: screen out undesired tokens (e.g., stop words, numbers, words not in a vocabulary file) from the ngrams.
6. `sort_ngrams.py`: combine multiple ngrams files into a single sorted file.
7. `consolidate_ngrams.py`: consolidate duplicate ngrams resulting from the previous steps.
8. `index_and_create_vocabulary.py`: numerically index a list of unigrams and create a "vocabulary file" to screen multigrams.
9. `create_yearly_files.py`: collate the master corpus into yearly sub-corpora.
10. `helpers/file_handler.py`: helper script to simplify reading and writing files in the other modules.
11. `helpers/print_jsonl_lines.py`: helper script to view a snippet of ngrams in a JSONL file.
12. `helpers/verify_sort.py`: helper script to confirm whether an ngram file is properly sorted. 

`src/training_tools`
1. train_ngrams
