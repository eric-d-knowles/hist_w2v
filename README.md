# hist_w2v: Tools for Training Word2Vec Models on Google Ngrams

I began this project because I wanted to study the evolution of group stereotypes over decades and centuries using the Google Ngrams corpora. I'm nowhere near the first to think of this—yet I wasn't satisfied with the ready-made tools I found online. So, with help from ChatGPT, I created a library of Python scripts to perform the following steps:

1. ```download_and_filter_ngrams.py``` downloads raw unigram and multigram files from the [Google Ngram Exports](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html) website. 
3. Download and prefilter_ ngrams, dropping those that contain numerals or other nonalphabetic characters (e.g., punctuation) or that lack Google part-of-speech tags (e.g., _NOUN, _ADV). You can further filter your multigrams using a vocabulary file, either by dropping rare words or the ngrams containing them (see step 7).
4. _Transform_ ngrams by lowercasing and/or lemmatizing (i.e., de-inflecting) them.
5. _Postfilter_ ngrams by removing stopwords (e.g., "the") and/or short words (e.g., "hi"), either by dropping individual tokens or the ngrams containing them.
6. _Consolidate_ ngrams by removing duplicates created during postfiltering and concatenating them into a single file.
7. _Indexing_ ngrams by giving each a unique number.
8. Create a _vocabulary file_ of the _N_ most common unigrams (1-grams) in the corpus, which can be used to prefilter a set of multigrams.
9. Create _yearly files_ containing the ngrams and their number of occurrences in each year of the corpus.

I've included a _run_ script containing a sample workflow that combines all of these steps.

The raw ngram files, even after prefiltering, are large and numerous. Given the demands on storage space, RAM, and processors, **the code is intended for use on an HPC cluster**. I've incorporated parallel processing where possible and attempted to make the code relatively memory-efficient—although some scripts still use a great deal of RAM. In future updates, I'll try to optimize the code for smaller systems.
