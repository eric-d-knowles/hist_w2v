# **hist_w2v**: Tools for Training Word2Vec Models on Google Ngrams

I started this project because I wanted to study the evolution of group stereotypes over time using Google Ngrams corpora. I'm not the first to think of this—however, I wasn't satisfied with the existing tools I found online. So, I created a Python package to streamline the process of (1) downloading and pre-processing raw ngrams and (2) training and evluating `word2vec` models on the ngrams. These are the specific scripts in the library:

`downoad_ngrams.py`: downloads the desired ngram types (e.g., 3-grams with part-of-speech [POS] tags, 5-grams with POS tags).
`lowercase_ngrams.py': make the ngrams all lowercase.
`lemmatize_ngrams.py': lemmatize the ngrams (i.e., reduce them to their base grammatical forms).
`filter_ngrams.py': screen out undesired tokens (e.g., stop words, numbers, words not in a vocabulary file) from the ngrams.
`index_and_create_vocabulary.py`: 
