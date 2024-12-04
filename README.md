# hist_w2v: Tools for Training Word2Vec Models on Google Ngrams

I began this project because I wanted to measure semantic change (especially the evolution of group stereotypes) over decades and centuries—but wasn't satisfied with the readymade tools I found online. So, with lots of help from ChatGPT (with whom I've developed an almost parasocial relationship), I created a library of Python scripts to perform the following steps:

1. Scrape the Google Ngrams [Google Ngram Exports](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html) website for the raw 1-, 2-, 3-, 4-, or 5gram files
2. Filter these 
