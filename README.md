# hist_w2v: Tools for Training Word2Vec Models on Google Ngrams

I began this project because I wanted to study the evolution of group stereotypes over decades and centuries using the Google Ngrams corpora. I'm nowhere near the first to think of this—yet I wasn't satisfied with the ready-made tools I found online. So, with help from ChatGPT, I created a library of Python scripts to perform the following steps:

1. ```download_and_filter_ngrams.py``` downloads raw unigram and multigram files from the [Google Ngram Exports](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html) website. The scripts drops ngrams that contain numerals or other nonalphabetic characters (e.g., punctuation) or lack Google part-of-speech (POS) tags (e.g., _NOUN, _ADV). You can further filter your multigrams using a vocabulary file, either by dropping rare words or the ngrams containing them (see step 9).
2. ```lowercase.py``` lowercases all letters in an ngram (but leaving the POS tags untouched).
3. ```lemmatize.py``` reduces words to their base (uninflected) forms.
4. ```remove_stopwords.py``` removes stopwords (e.g., "the" and "it"), either by dropping individual token or the ngrams that contain them.
5. ```remove_short_words.py``` removes words short that _N_ characters (e.g., "hi" or "no"), either by dropping individual tokens or the ngrams that contain them.
6. ```consolidate_ ngrams.py``` consolidates duplicates created during postfiltering by summing their frequencies and concatenates ngrams into a single file.
7. ```index_ngrams.py``` assigns all ngrams a unique number.
8. ```make_vocab_list.py``` generates a "vocabulary file" containing the _N_ most common unigrams (1-grams) in the corpus, which can be used to prefilter a set of multigrams (see step 1).
9. Create _yearly files_ containing the ngrams and their number of occurrences in each year of the corpus.

I've included a _run_ script containing a sample workflow that combines all of these steps.

The raw ngram files, even after prefiltering, are large and numerous. Given the demands on storage space, RAM, and processors, **the code is intended for use on an HPC cluster**. I've incorporated parallel processing where possible and attempted to make the code relatively memory-efficient—although some scripts still use a great deal of RAM. In future updates, I'll try to optimize the code for smaller systems.
