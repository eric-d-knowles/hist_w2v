# hist_w2v: Tools for Training Word2Vec Models on Google Ngrams

I began this project because I wanted to study the evolution of group stereotypes over decades and centuries using the Google Ngrams corpora. I'm nowhere near the first to think of this—yet I wasn't satisfied with the ready-made tools I found online. So, I created a library of Python scripts to perform the following steps:

The raw ngram files, even after prefiltering, are large and numerous. Given the demands on storage space, RAM, and processors, **the code is intended for use on an HPC cluster**. I've incorporated parallel processing where possible and attempted to make the code relatively memory-efficient—although some scripts still use a great deal of RAM. In future updates, I'll try to optimize the code for smaller systems.

