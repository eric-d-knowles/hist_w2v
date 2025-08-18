# cython: boundscheck=False, wraparound=False, cdivision=True
import re
import struct
cimport cython

tags = re.compile(
    r'_(?:NOUN|PROPN|VERB|ADJ|ADV|PRON|DET|ADP|NUM|CONJ|X|\.)$'
)

def process(
    str ngram,
    bint opt_lower=False,
    bint opt_alpha=False,
    bint opt_shorts=False,
    bint opt_stops=False,
    bint opt_lemmas=False,
    int min=3,
    set stop_set=None,
    object lemma_gen=None,
    dict tag_map=None
) -> str:
    tok_tups = split(tokenize(ngram))
    if opt_lower:
        tok_tups = lower(tok_tups)
    if opt_alpha:
        tok_tups = alpha(tok_tups)
    if opt_shorts:
        tok_tups = shorts(tok_tups, min)
    if opt_stops and stop_set is not None:
        tok_tups = stops(tok_tups, stop_set)
    if opt_lemmas and lemma_gen is not None:
        tok_tups = lemmas(tok_tups, lemma_gen, tag_map)
    return rejoin(tok_tups)

def tokenize(str ngram) -> list:
    cdef list toks = ngram.split()
    return toks

def split(list toks) -> list:
    cdef list tok_tups = []
    cdef str tok
    cdef str word
    cdef str tag
    for tok in toks:
        if tags.search(tok):
            word, tag = tok.rsplit('_', 1)
            tok_tups.append((word, tag))
        else:
            tok_tups.append((tok, None))
    return tok_tups

def lower(list tok_tups) -> list:
    cdef list result = [
        (tok.lower(), pos)
        for tok, pos in tok_tups
    ]
    return result

def alpha(list tok_tups) -> list:
    cdef list result = [
        (tok, pos) if tok.isalpha()
        else ("<UNK>", pos)
        for tok, pos in tok_tups
    ]
    return result

def shorts(list tok_tups, int min=3) -> list:
    cdef list result = [
        (tok, pos) if len(tok) >= min
        else ("<UNK>", pos)
        for tok, pos in tok_tups
    ]
    return result

def stops(list tok_tups, set stop_set) -> list:
    cdef list result = [
        (tok, pos) if tok not in stop_set
        else ("<UNK>", pos)
        for tok, pos in tok_tups
    ]
    return result
    
def lemmas(list tok_tups, object lemma_gen, dict tag_map=None) -> list:
    if tag_map is None:
        tag_map = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
    cdef list result = [
        (lemma_gen.lemmatize(tok, pos=tag_map.get(pos, 'n')), pos)
        if pos is not None
        else (tok, None)
        for tok, pos in tok_tups
    ]
    return result

def rejoin(list toks) -> str:
    return " ".join(t for t, _ in toks)
