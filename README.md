# hist_w2v: Tools for Training Word2Vec Models on Google Ngrams

I began this project because I wanted to study the evolution of group stereotypes over decades and centuries using the Google Ngrams corpora. I'm nowhere near the first to think of this—yet I wasn't satisfied with the ready-made tools I found online. So, with help from ChatGPT, I created a library of Python scripts to perform the following steps:

1. _Scrape_ the Google Ngrams [Google Ngram Exports](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html) website for raw unigram (1gram) or multigram (2-, 3-, 4-, or 5gram) files.
2. Download and prefilter_ ngrams, dropping those that contain numerals or other nonalphabetic characters (e.g., punctuation) or that lack Google part-of-speech tags (e.g., _NOUN, _ADV). You can further filter your multigrams using a vocabulary file, either by dropping rare words or the ngrams containing them (see step 7).
3. _Transform_ ngrams by lowercasing and/or lemmatizing (i.e., de-inflecting) them.
4. _Postfilter_ ngrams by removing stopwords (e.g., "the") and/or short words (e.g., "hi"), either by dropping individual tokens or the ngrams containing them.
5. _Consolidate_ ngrams by removing duplicates created during postfiltering and concatenating them into a single file.
6. _Indexing_ ngrams by giving each a unique number.
7. Create a _vocabulary file_ of the _N_ most common unigrams (1-grams) in the corpus, which can be used to prefilter a set of multigrams.
8. Create _yearly files_ containing the ngrams and their number of occurrences in each year of the corpus.
