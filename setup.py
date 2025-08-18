from setuptools import setup
from Cython.Build import cythonize
import os
import numpy

cython_sources = [
    os.path.join("src", "reservoir_sampler.pyx"),
    os.path.join("src", "count_db_items.pyx"),
    os.path.join("src", "ngram_filters.pyx"),
]

setup(
    name="hist_w2v",
    ext_modules=cythonize(cython_sources),
    packages=["hist_w2v"],
)